"""
bench_fused_tpu.py — run ON the v5e (via upload -> ~/nop/tpu, PYTHONPATH=~/nop).

Compares the FNO forward at every fusion level, SAME math (same weights), NS size:
  baseline : plain-JAX FNO (fno_proper.forward_proper)   — XLA fuses what it can, inter-layer HBM
  A        : per-block Pallas kernel, JAX loops layers    — inter-layer HBM round-trip
  B        : all blocks in ONE kernel, field VMEM-resident — no inter-layer HBM
  C        : lift + blocks + project in one kernel         — only add_grid in JAX

Robust: every (level, dtype, batch) cell is independent (try/except), so an OOM in one (e.g. B/C
f32, whose stacked 4-layer weights can exceed the 128 MB VMEM) does NOT abort the others.  Reports
inference LATENCY (batch 1) and THROUGHPUT (batch 32), f32 + bf16, us/sample + overall ms + relL2.
"""
import time
import jax, jax.numpy as jnp

import tpu.fft_core  # noqa: F401  (global highest matmul precision)
from tpu.experimental import fno_proper as PR
from tpu.experimental.pallas_partial2d import forward_A, forward_B, forward_C

print("jax", jax.__version__, "| devices:", jax.devices(), flush=True)
WIDTH, M, LAYERS, H, W = 32, 16, 4, 128, 128
P = PR.init_proper(jax.random.PRNGKey(1), 1, 1, WIDTH, M, LAYERS)


def bench(fn, x, reps=30, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


levels = {"baseline": lambda P, z, mdt: PR.forward_proper(P, z, M, 0),
          "A":        lambda P, z, mdt: forward_A(P, z, M, 0, mdt=mdt),
          "B":        lambda P, z, mdt: forward_B(P, z, M, 0, mdt=mdt),
          "C":        lambda P, z, mdt: forward_C(P, z, M, 0, mdt=mdt)}

for B in (1, 32):
    x = jax.random.normal(jax.random.PRNGKey(0), (B, 1, H, W))
    kind = "LATENCY (single instance, batch 1)" if B == 1 else "THROUGHPUT (batched, batch 32)"
    print(f"\n=== INFERENCE {kind} @ {H}x{W}, width {WIDTH}, m {M}, {LAYERS} layers ===", flush=True)
    ref = jax.jit(lambda z: PR.forward_proper(P, z, M, 0))(x)
    for name, mk in levels.items():
        for mdt, tag in [(jnp.float32, "f32"), (jnp.bfloat16, "bf16")]:
            if name == "baseline" and tag == "bf16":
                continue                                              # baseline is f32 only
            label = f"{name} {tag}"
            try:
                fn = jax.jit(lambda z, mk=mk, mdt=mdt: mk(P, z, mdt))
                med = bench(fn, x)
                rel = float(jnp.linalg.norm(fn(x) - ref) / jnp.linalg.norm(ref))
                print(f"  {label:14s}: {med*1e6/B:8.2f} us/sample   {med*1e3:7.2f} ms overall/{B}"
                      f"   relL2 {rel:.1e}", flush=True)
            except Exception as e:
                msg = str(e).replace("\n", " ")[:90]
                print(f"  {label:14s}: FAILED  {type(e).__name__}: {msg}", flush=True)
print("BENCH_DONE", flush=True)
