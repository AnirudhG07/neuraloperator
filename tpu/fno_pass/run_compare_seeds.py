"""Seed-sweep: reference (jnp.fft) vs ours (reim) over several init seeds -> rel-L2 mean ± std.
Answers whether the single-seed Burgers gap is just optimizer variance.  Runs ON the v5e."""
import os
from tpu.fno_pass import train

D = os.path.expanduser("~/data")
train.compare_seeds(D, "burgers", seeds=5, steps=200, m=6,  channels=24, batch=16)   # the case in question (full config)
train.compare_seeds(D, "darcy",   seeds=5, steps=50,  m=12, channels=32, batch=8)    # lighter (variance, not peak acc)
train.compare_seeds(D, "ns",      seeds=3, steps=15,  m=16, channels=32, batch=32)   # expensive -> 3 seeds, short
print("ALL_SEED_SWEEPS_DONE", flush=True)
