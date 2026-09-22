"""
fno_pass/train.py — baseline JAX trainer (Adam + MSE + StepLR).  Trains a given forward
(reference.forward = plain jnp.fft, or ours.forward = reim) on any dataset, then reports test
rel-L2, train wall-clock, and inference us/sample.

`compare(root, dataset)` trains BOTH (same seed / same init weights) and prints them side by side —
so you see the reference and the fast one on the exact same footing.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from tpu.fno_pass import data, ours, reference
    from tpu.fno_pass.layers import adam_init, adam_step, init_fno, mse, n_params
except ImportError:
    import data
    import ours
    import reference
    from layers import adam_init, adam_step, init_fno, mse, n_params


def infer_us_per_sample(fwd, P, x, m, reps=30, cap=128):
    x = jnp.asarray(x[:cap])                               # cap so 128² doesn't OOM
    fn = jax.jit(lambda z: fwd(P, z, m))
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] / x.shape[0] * 1e6


def train(fwd, xtr, ytr, xte, yte, m=16, channels=32, n_layers=4, steps=100, batch=16,
          lr=1e-3, wd=1e-4, step_size=100, gamma=0.5, seed=0):
    xtr, ytr = np.asarray(xtr, np.float32), np.asarray(ytr, np.float32)   # host-resident
    xte, yte = np.asarray(xte, np.float32), np.asarray(yte, np.float32)
    ntr = xtr.shape[0]
    P = init_fno(jax.random.PRNGKey(seed), 1, 1, channels, m, n_layers)
    opt = adam_init(P)

    @jax.jit
    def step(P, opt, x, y, lr_ep):
        loss, g = jax.value_and_grad(lambda P: mse(fwd(P, x, m), y))(P)
        P, opt = adam_step(P, g, opt, lr=lr_ep, wd=wd)
        return P, opt, loss

    @jax.jit
    def ev(P, x, y):
        p = fwd(P, x, m)
        return jnp.sum((p - y) ** 2, (-1, -2, -3)), jnp.sum(y ** 2, (-1, -2, -3))

    def test_rel_l2(P):                                    # chunked to avoid OOM at 128²
        num = den = 0.0
        for i in range(0, xte.shape[0], 64):
            n, d = ev(P, jnp.asarray(xte[i:i + 64]), jnp.asarray(yte[i:i + 64]))
            num += float(jnp.sum(jnp.sqrt(n))); den += float(jnp.sum(jnp.sqrt(d)))
        return num / den

    key = jax.random.PRNGKey(seed); nb = max(1, ntr // batch)
    t0 = time.perf_counter()
    for ep in range(steps):
        lr_ep = lr * (gamma ** (ep // step_size))
        key, ks = jax.random.split(key); perm = np.asarray(jax.random.permutation(ks, ntr))
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            P, opt, _ = step(P, opt, jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]), lr_ep)
    return P, test_rel_l2(P), time.perf_counter() - t0


def compare(root, dataset="darcy", steps=100, m=16, channels=32, batch=16):
    load = {"darcy":   lambda: data.load_darcy(root, 32),
            "ns":      lambda: data.load_ns(root, 128),
            "burgers": lambda: data.load_burgers(root)}[dataset]
    xtr, ytr, xte, yte = load()
    print(f"dataset={dataset}  xtr{xtr.shape}  m={m} channels={channels} steps={steps} batch={batch}", flush=True)
    for name, mod in [("reference (jnp.fft)", reference), ("ours (reim)", ours)]:
        P, rl, secs = train(mod.forward, xtr, ytr, xte, yte, m=m, channels=channels, steps=steps, batch=batch)
        us = infer_us_per_sample(mod.forward, P, xte, m)
        print(f"  {name:22}: relL2={rl:.4e}  train={secs:6.1f}s  infer={us:7.2f} us/sample  "
              f"params={n_params(P):,}", flush=True)
    print("COMPARE_DONE", flush=True)


def compare_seeds(root, dataset="burgers", seeds=5, steps=200, m=6, channels=24, batch=16):
    """Train BOTH variants over `seeds` different init seeds; report rel-L2 mean ± std.  If ours and
    reference overlap, a single-seed gap (e.g. Burgers) was just optimizer variance, not a flaw."""
    load = {"darcy":   lambda: data.load_darcy(root, 32),
            "ns":      lambda: data.load_ns(root, 128),
            "burgers": lambda: data.load_burgers(root)}[dataset]
    xtr, ytr, xte, yte = load()
    print(f"SEED-SWEEP dataset={dataset}  xtr{xtr.shape}  m={m} channels={channels} steps={steps} "
          f"batch={batch} seeds={seeds}", flush=True)
    for name, mod in [("reference (jnp.fft)", reference), ("ours (reim)", ours)]:
        rls = []
        for s in range(seeds):
            _, rl, _ = train(mod.forward, xtr, ytr, xte, yte, m=m, channels=channels,
                             steps=steps, batch=batch, seed=s)
            rls.append(rl)
        arr = np.asarray(rls)
        print(f"  {name:22}: relL2 mean={arr.mean():.4e}  std={arr.std():.4e}  "
              f"runs={[f'{r:.4f}' for r in rls]}", flush=True)
    print("SEED_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    import os
    compare(os.path.expanduser("~/data"), "darcy", steps=100)
