"""
Run ON the v5e: load the trained tensors and time FNO inference with the main spectral conv
(skinny partial axis-0) against the HYBRID variant (full-MXU axis-0, then select the band).
Same modes, so predictions are identical (checked) — the only difference is speed.
"""
import os, time
import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F  # noqa: F401  (import sets global highest matmul precision)
from ...fno import model as PR
from ...fno.spectral import spectral_conv as spectral_2corner
from ..fno.spectral_variants import spectral_2corner_herm, spectral_2corner_hybrid

def _forward_with(P, x, m, spec):
    """model.forward, but with the spectral conv swapped for the variant under test."""
    from ...fno.layers import add_grid, channel_mlp, conv1x1, gelu
    x = channel_mlp(P["lift"], add_grid(x))
    for block in P["blocks"]:
        x = gelu(spec(block["filter"], x, m) + conv1x1(block["skip"], x))
    return channel_mlp(P["proj"], x)


print("jax", jax.__version__, "|", jax.devices(), flush=True)
MDL = os.environ.get("NOP_MODELS",
                     os.path.join(os.path.dirname(__file__), "..", "..", "results", "models"))
# (name, grid, m, file, width)
CFG = [("NS", (128, 128), 16, "ns_proper.npz", 32),
       ("DARCY", (32, 32), 12, "darcy_proper.npz", 32),
       ("BURGERS", (17, 16), 6, "burgers_proper.npz", 24)]


def bench(P, x, m, spec, reps=30):
    fn = jax.jit(lambda z: _forward_with(P, z, m, spec))
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
