"""Run the reference (jnp.fft) vs ours (reim) comparison ON the v5e.  Datasets in ~/data."""
import os
from tpu.fno_pass import train

D = os.path.expanduser("~/data")
train.compare(D, "darcy", steps=100, m=12, channels=32, batch=8)
train.compare(D, "ns", steps=30, m=16, channels=32, batch=32)
train.compare(D, "burgers", steps=200, m=6, channels=24, batch=16)
print("ALL_COMPARE_DONE", flush=True)
