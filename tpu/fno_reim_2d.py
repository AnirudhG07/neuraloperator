"""
fno_reim_2d.py — a 2D Fourier Neural Operator in pure JAX, real/imag all the way (NO complex64).
Trained on the neuralop Darcy-flow dataset.

The whole model is ONE readable function, `fno_forward` — read it first; everything else is a
named helper it calls.  The flow of one Darcy sample (B, 1, H, W):

        x ──► add (x,y) grid ──► LIFT (1x1 MLP, ch→width)
              │
              ▼   ┌─────────────── Fourier block (× n_layers) ───────────────┐
              x ──┤  spectral: FFT2 ─► keep low modes ─► complex filter ─► iFFT2  ┐
                  │  skip:     1x1 conv (local, per-pixel)                         ├─► + ─► GELU ─► x
                  └──────────────────────────────────────────────────────────────┘
              │
              ▼
        PROJECT (1x1 MLP, width→1) ──► u (B, 1, H, W)

"spectral" = global mixing via Fourier; "skip" = local mixing.  They run in PARALLEL, are summed,
then activated.  The FFT is done as real/imag matmuls (cos/sin), so no complex64 ever appears.

Run on TPU:  PYTHONPATH=~/nop ~/nopenv/bin/python -c "import tpu.fno_reim_2d as M; M.train_darcy()"
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

try:
    from tpu.fft_core import ACC, F
except ImportError:
    from fft_core import ACC, F

gelu = jax.nn.gelu


# ═══════════════════════════════════════════════════════════════════════════
#  THE MODEL — one forward function.  Read this first.
# ═══════════════════════════════════════════════════════════════════════════
def fno_forward(P, x):
    """P = the whole weight tree, x = (B, in_ch, H, W) real field  ->  (B, out_ch, H, W)."""
    x = add_grid(x)                                   # append (x,y) coordinate channels
    x = channel_mlp(P["lift"], x)                     # LIFT: in_ch -> width

    for blk in P["blocks"]:                           # each Fourier block:
        fr, fi = fft2_lowmodes(x, m=modes_of(blk))    #   spectral: FFT2 -> low modes (re, im)
        gr, gi = spectral_filter(blk["filter"], fr, fi)  #            learned complex channel mix
        spectral = ifft2_real(gr, gi, x.shape[-2], x.shape[-1])  #     iFFT2 -> REAL field
        skip = conv1x1(blk["skip"], x)                #   skip: local 1x1 conv (parallel path)
        x = gelu(spectral + skip)                     #   JOIN the two paths, then activate

    return channel_mlp(P["proj"], x)                  # PROJECT: width -> out_ch


# ═══════════════════════════════════════════════════════════════════════════
#  The spectral path, as three named steps (FFT -> filter -> iFFT), all real/imag.
# ═══════════════════════════════════════════════════════════════════════════
def _dft(m, L, sign):
    """(cos, sin) rows of a length-L DFT kept to m modes.  sign=-1 forward, +1 inverse."""
    k = jnp.arange(m)[:, None]
    n = jnp.arange(L)[None, :]
    a = sign * 2.0 * jnp.pi * k * n / L
    return jnp.cos(a).astype(F), jnp.sin(a).astype(F)      # each (m, L)


def fft2_lowmodes(x, m):
    """Forward 2D DFT of a REAL field, keeping the m×m low-frequency block.  Real in -> (re, im).
    x (B, C, H, W) -> (fr, fi) each (B, C, m, m)."""
    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    ChR, ChI = _dft(m, H, -1)                             # DFT along H (real input -> 2 matmuls)
    Ar = jnp.einsum('kh,bchw->bckw', ChR, x, preferred_element_type=ACC)
    Ai = jnp.einsum('kh,bchw->bckw', ChI, x, preferred_element_type=ACC)
    CwR, CwI = _dft(m, W, -1)                             # DFT along W (complex now)
    e = lambda M, Z: jnp.einsum('jw,bckw->bckj', M, Z, preferred_element_type=ACC)
    return e(CwR, Ar) - e(CwI, Ai), e(CwR, Ai) + e(CwI, Ar)


def spectral_filter(w, fr, fi):
    """Per-mode complex channel mix: (Wr+iWi)·(fr+ifi).  w = {w_re, w_im} (Cin, Cout, m, m).
    (B, Cin, m, m) -> (B, Cout, m, m)."""
    mix = lambda M, Z: jnp.einsum('iokl,bikl->bokl', M, Z, preferred_element_type=ACC)
    return (mix(w["w_re"], fr) - mix(w["w_im"], fi),      # real part
            mix(w["w_re"], fi) + mix(w["w_im"], fr))      # imag part


def ifft2_real(gr, gi, H, W):
    """Inverse 2D DFT back to a REAL field: u = Re(iFFT2), computed directly in real/imag.
    (gr, gi) each (B, C, m, m) -> u (B, C, H, W) real."""
    m = gr.shape[-1]
    IwR, IwI = _dft(m, W, +1)                             # (m, W); use as (W, m) below
    ew = lambda M, Z: jnp.einsum('jw,bckj->bckw', M, Z, preferred_element_type=ACC)
    Pr = ew(IwR, gr) - ew(IwI, gi)                        # inverse along W
    Pi = ew(IwR, gi) + ew(IwI, gr)
    IhR, IhI = _dft(m, H, +1)                             # (m, H)
    eh = lambda M, Z: jnp.einsum('kh,bckw->bchw', M, Z, preferred_element_type=ACC)
    u = eh(IhR, Pr) - eh(IhI, Pi)                         # inverse along H, keep REAL part
    return (u / (H * W)).astype(F)


# ═══════════════════════════════════════════════════════════════════════════
#  The plain (non-spectral) helpers.
# ═══════════════════════════════════════════════════════════════════════════
def add_grid(x):
    """Append normalized (x, y) coordinate channels so the net knows absolute position."""
    B, _, H, W = x.shape
    gx = jnp.broadcast_to(jnp.linspace(0, 1, H)[:, None], (H, W))
    gy = jnp.broadcast_to(jnp.linspace(0, 1, W)[None, :], (H, W))
    grid = jnp.broadcast_to(jnp.stack([gx, gy])[None], (B, 2, H, W))
    return jnp.concatenate([x, grid], axis=1)


def conv1x1(w, x):
    """1×1 conv = same linear map at every pixel.  w = {W, b}.  (B, Cin, H, W) -> (B, Cout, H, W)."""
    return jnp.einsum('oi,bihw->bohw', w["W"], x) + w["b"][None, :, None, None]


def channel_mlp(w, x):
    """2-layer pointwise MLP (conv1x1 -> GELU -> conv1x1); used for lift & project."""
    return conv1x1(w["l2"], gelu(conv1x1(w["l1"], x)))


def modes_of(blk):
    return blk["filter"]["w_re"].shape[-1]


# ═══════════════════════════════════════════════════════════════════════════
#  Weight initialisation — mirrors fno_forward's tree exactly.
# ═══════════════════════════════════════════════════════════════════════════
def _conv1x1(key, cin, cout):
    std = (2.0 / (cin + cout)) ** 0.5
    return {"W": std * jax.random.normal(key, (cout, cin)), "b": jnp.zeros((cout,))}


def _mlp(key, cin, hid, cout):
    k1, k2 = jax.random.split(key)
    return {"l1": _conv1x1(k1, cin, hid), "l2": _conv1x1(k2, hid, cout)}


def _filter(key, cin, cout, m):
    std = (2.0 / (cin + cout)) ** 0.5
    kr, ki = jax.random.split(key)
    return {"w_re": std * jax.random.normal(kr, (cin, cout, m, m)),
            "w_im": std * jax.random.normal(ki, (cin, cout, m, m))}


def init_fno(key, in_ch=1, out_ch=1, width=24, m=16, n_layers=4,
             lift_ratio=2, proj_ratio=2):
    """Build the whole weight tree that fno_forward walks.  Defaults = neuralop FNO_Small2d:
    hidden_channels=24, n_modes=16, n_layers=4, lifting/projection channel ratios."""
    ks = jax.random.split(key, n_layers + 2)
    return {
        "lift": _mlp(ks[0], in_ch + 2, lift_ratio * width, width),   # +2 for the (x,y) grid channels
        "blocks": [{"filter": _filter(jax.random.fold_in(ks[i + 1], 0), width, width, m),
                    "skip":   _conv1x1(jax.random.fold_in(ks[i + 1], 1), width, width)}
                   for i in range(n_layers)],
        "proj": _mlp(ks[-1], width, proj_ratio * width, out_ch),
    }


def load_burgers(root=None):
    """neuralop Burgers .pt -> 2D time-space task (like the repo's FNO_Small2d config):
    input = initial condition u0 broadcast over the 17 time rows -> (B,1,T,X); target = the
    trajectory y -> (B,1,T,X).  Standardized by train stats.  T=17, X=16."""
    import torch
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "neuralop", "data", "datasets", "data")
    tr = torch.load(os.path.join(root, "burgers_train_16.pt"), weights_only=False)
    te = torch.load(os.path.join(root, "burgers_test_16.pt"), weights_only=False)
    def build(d):
        x0 = np.asarray(d["x"], np.float32)                 # (B, X)  initial condition
        y = np.asarray(d["y"], np.float32)                  # (B, T, X)  trajectory
        T = y.shape[1]
        x = np.broadcast_to(x0[:, None, :], (x0.shape[0], T, x0.shape[1]))  # repeat over time
        return x[:, None], y[:, None]                       # (B,1,T,X)
    xtr, ytr = build(tr)
    xte, yte = build(te)
    xm, xs, ym, ys = xtr.mean(), xtr.std() + 1e-8, ytr.mean(), ytr.std() + 1e-8
    return (xtr - xm) / xs, (ytr - ym) / ys, (xte - xm) / xs, (yte - ym) / ys


def save_burgers_npz(path):
    xtr, ytr, xte, yte = load_burgers()
    np.savez(path, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Darcy data + training loop.
# ═══════════════════════════════════════════════════════════════════════════
def load_darcy(res=16, root=None):
    """neuralop Darcy .pt -> (x_train, y_train, x_test, y_test), (B,1,H,W), standardized."""
    import torch
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "neuralop", "data", "datasets", "data")
    tr = torch.load(os.path.join(root, f"darcy_train_{res}.pt"), weights_only=False)
    te = torch.load(os.path.join(root, f"darcy_test_{res}.pt"), weights_only=False)
    xtr, ytr = np.asarray(tr["x"], np.float32), np.asarray(tr["y"], np.float32)
    xte, yte = np.asarray(te["x"], np.float32), np.asarray(te["y"], np.float32)
    xm, xs, ym, ys = xtr.mean(), xtr.std() + 1e-8, ytr.mean(), ytr.std() + 1e-8
    nrm = lambda a, mu, sd: ((a - mu) / sd)[:, None, :, :]
    return nrm(xtr, xm, xs), nrm(ytr, ym, ys), nrm(xte, xm, xs), nrm(yte, ym, ys)


def save_darcy_npz(res, path):
    """Convert the torch .pt to a torch-free .npz (so a TPU VM without torch can read it)."""
    xtr, ytr, xte, yte = load_darcy(res)
    np.savez(path, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return path


def mse(p, y):
    return jnp.mean((p - y) ** 2)


def rel_l2(p, y):
    n = jnp.sqrt(jnp.sum((p - y) ** 2, axis=(-1, -2, -3)))
    d = jnp.sqrt(jnp.sum(y ** 2, axis=(-1, -2, -3)))
    return jnp.mean(n / d)


def _adam_init(P):
    z = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
    return {"m": z(P), "v": z(P), "t": 0}


def _adam_step(P, g, st, lr=1e-3, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, st["v"], g)
    mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    # AdamW: decoupled weight decay (p -= lr*wd*p) alongside the Adam step
    P = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / (jnp.sqrt(b) + eps) + wd * p), P, mh, vh)
    return P, {"m": m, "v": v, "t": t}


def inference_time(P, x, reps=50, trace_dir=None, tag="darcy"):
    """Time a jitted forward pass (inference only, no grad) over the whole set x (B,1,H,W).
    Warmup/compile is excluded.  If trace_dir is set, also capture an xprof device trace of the
    timed passes (loadable in TensorBoard).  Returns a dict of timings."""
    import time
    fn = jax.jit(lambda z: fno_forward(P, z))
    jax.block_until_ready(fn(x))                       # compile + warmup (NOT timed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append(time.perf_counter() - t0)
    if trace_dir:                                      # capture a profile of the same passes
        with jax.profiler.trace(trace_dir):
            for _ in range(reps):
                jax.block_until_ready(fn(x))
    ts = sorted(ts)
    med, n = ts[len(ts) // 2], x.shape[0]
    print(f"INFER {tag}: {n} samples  {med*1e3:.3f} ms/full-pass  {med/n*1e6:.2f} us/sample  "
          f"{n/med:,.0f} samples/s  (min {ts[0]*1e3:.3f} ms)")
    return {"tag": tag, "n": n, "ms_per_pass": med * 1e3, "us_per_sample": med / n * 1e6,
            "samples_per_s": n / med}


def train(name, xtr, ytr, xte, yte, width=24, m=16, n_layers=4, steps=300, batch=8, lr=5e-3,
          weight_decay=1e-4, step_size=60, gamma=0.5, seed=0, out=None):
    """Train the FNO with the neuralop-config recipe: AdamW (weight_decay) + StepLR
    (lr *= gamma every step_size epochs).  Architecture defaults = FNO_Small2d.
    Returns (params, (x_test, y_test), loss_rows)."""
    xtr, ytr, xte, yte = map(jnp.asarray, (xtr, ytr, xte, yte))
    ntr = xtr.shape[0]
    key = jax.random.PRNGKey(seed)
    P = init_fno(key, in_ch=1, out_ch=1, width=width, m=m, n_layers=n_layers)
    opt = _adam_init(P)

    @jax.jit
    def step(P, opt, x, y, lr_ep):
        loss, g = jax.value_and_grad(lambda P: mse(fno_forward(P, x), y))(P)
        P, opt = _adam_step(P, g, opt, lr=lr_ep, wd=weight_decay)
        return P, opt, loss

    print(f"FNO-2d {name} (reim, no c64)  width={width} modes={m} layers={n_layers}  "
          f"lr={lr} batch={batch} wd={weight_decay} StepLR({step_size},{gamma})  "
          f"| {ntr} train / {xte.shape[0]} test  shape {tuple(xtr.shape[1:])}")
    rows = [("epoch", "train_mse", "test_mse", "test_relL2")]
    nb = max(1, ntr // batch)
    for ep in range(steps):
        lr_ep = lr * (gamma ** (ep // step_size))       # StepLR
        key, ks = jax.random.split(key)
        perm = jax.random.permutation(ks, ntr)
        tot = 0.0
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            P, opt, loss = step(P, opt, xtr[idx], ytr[idx], lr_ep)
            tot += float(loss)
        if ep % 25 == 0 or ep == steps - 1:
            pred = fno_forward(P, xte)
            tr, te, rl = tot / nb, float(mse(pred, yte)), float(rel_l2(pred, yte))
            rows.append((ep, tr, te, rl))
            print(f"  epoch {ep:4d}  train_mse {tr:.4e}  test_mse {te:.4e}  test_relL2 {rl:.4e}")
    if out:
        with open(out, "w") as fh:
            for r in rows:
                fh.write(",".join(str(v) for v in r) + "\n")
        print(f"wrote loss log -> {out}")
    return P, (xte, yte), rows


def train_darcy(res=32, steps=300, npz=None, out="tpu/fno_darcy_train_log.csv"):
    """Darcy, repo config (FNO_Small2d): lr 5e-3, batch 8, StepLR(60,0.5), wd 1e-4."""
    if npz:
        d = np.load(npz); data = (d["xtr"], d["ytr"], d["xte"], d["yte"])
    else:
        data = load_darcy(res)
    return train("darcy", *data, width=24, m=16, n_layers=4, steps=steps, batch=8, lr=5e-3,
                 weight_decay=1e-4, step_size=60, gamma=0.5, out=out)


def train_burgers(steps=300, npz=None, out="tpu/fno_burgers_train_log.csv"):
    """Burgers (2D time-space), repo config: lr 1e-4, batch 16, StepLR(60,0.5), wd 1e-4."""
    if npz:
        d = np.load(npz); data = (d["xtr"], d["ytr"], d["xte"], d["yte"])
    else:
        data = load_burgers()
    return train("burgers", *data, width=24, m=16, n_layers=4, steps=steps, batch=16, lr=1e-4,
                 weight_decay=1e-4, step_size=60, gamma=0.5, out=out)


if __name__ == "__main__":
    train_darcy()
