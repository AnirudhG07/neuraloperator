"""
run_infer_hybrid.py — run ON the v5e.  Loads the trained tensors and measures FNO inference with the
CURRENT spectral conv (skinny partial axis-0) vs the HYBRID (full-MXU axis-0 + select ±band).
Same modes -> identical predictions (checked); the only difference is speed.
"""
import os, time
import jax, jax.numpy as jnp, numpy as np

import tpu.fft_core  # noqa: F401
from tpu.experimental import fno_proper as PR
from tpu.experimental.fno_proper import (spectral_2corner, spectral_2corner_hybrid,
                                         spectral_2corner_herm)

print("jax", jax.__version__, "|", jax.devices(), flush=True)
MDL = os.path.expanduser("~/nop/tpu/results/models")
# (name, grid, m, file, width)
CFG = [("NS", (128, 128), 16, "ns_proper.npz", 32),
       ("DARCY", (32, 32), 12, "darcy_proper.npz", 32),
       ("BURGERS", (17, 16), 6, "burgers_proper.npz", 24)]


def bench(P, x, m, spec, reps=30):
    fn = jax.jit(lambda z: PR.forward_fno(P, z, m, spec_fn=spec))
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2], fn(x)


VARIANTS = [("current", spectral_2corner), ("full-hybrid", spectral_2corner_hybrid),
            ("hermitian", spectral_2corner_herm)]
for name, (H, W), m, f, width in CFG:
    P = PR.load_params(os.path.join(MDL, f), width, m, 4)
    x = jnp.asarray(np.random.randn(128, 1, H, W).astype(np.float32)); B = x.shape[0]
    t_cur, y_cur = bench(P, x, m, spectral_2corner)
    print(f"\n{name} {H}x{W} (batch {B}):", flush=True)
    for vname, spec in VARIANTS:
        t, y = bench(P, x, m, spec)
        rel = float(jnp.linalg.norm(y - y_cur) / jnp.linalg.norm(y_cur))
        print(f"    {vname:12s}: {t*1e6/B:8.2f} us/sample   speedup {t_cur/t:.3f}x   match {rel:.1e}",
              flush=True)
print("\nINFER_DONE", flush=True)
