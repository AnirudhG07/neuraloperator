"""
Trains the complex64 jnp.fft baseline and the real/imag version from IDENTICAL init weights and
prints them side by side — accuracy, train wall-clock and inference us/sample on the same footing.
`compare_seeds` repeats that over several seeds, which is how the single-seed Burgers gap was
shown to be optimizer variance rather than a flaw in the fast path.

On the v5e:  PYTHONPATH=~/nop ~/nopenv/bin/python -c \
                "import tpu.experimental.bench.compare_ref as c; c.main()"
"""
import os

import numpy as np

from ...fno import data, model, train
from ..fno import reference

DATASETS = {"darcy":   lambda root: data.load_darcy(root, 32),
            "ns":      lambda root: data.load_ns(root, 128),
            "burgers": lambda root: data.load_burgers(root)}

VARIANTS = [("reference (jnp.fft)", reference.forward), ("ours (reim)", model.forward)]


def compare(root, dataset="darcy", steps=100, n_modes=16, channels=32, batch=16):
    xtr, ytr, xte, yte = DATASETS[dataset](root)
    print(f"dataset={dataset}  xtr{xtr.shape}  n_modes={n_modes} channels={channels} "
          f"steps={steps} batch={batch}", flush=True)
    for name, fwd in VARIANTS:
        P, rel, seconds = train.train(xtr, ytr, xte, yte, n_modes=n_modes, channels=channels,
                                      steps=steps, batch=batch, fwd=fwd, verbose=False)
        us = train.infer_us_per_sample(P, xte, n_modes, fwd=fwd)
        print(f"  {name:22}: rel_l2={rel:.4e}  train={seconds:6.1f}s  infer={us:7.2f} us/sample  "
              f"params={model.n_params(P):,}", flush=True)
    print("COMPARE_DONE", flush=True)


def compare_seeds(root, dataset="burgers", seeds=5, steps=200, n_modes=6, channels=24, batch=16):
    xtr, ytr, xte, yte = DATASETS[dataset](root)
    print(f"SEED-SWEEP dataset={dataset}  xtr{xtr.shape}  n_modes={n_modes} channels={channels} "
          f"steps={steps} batch={batch} seeds={seeds}", flush=True)
    for name, fwd in VARIANTS:
        results = [train.train(xtr, ytr, xte, yte, n_modes=n_modes, channels=channels, steps=steps,
                               batch=batch, seed=s, fwd=fwd, verbose=False)[1]
                   for s in range(seeds)]
        arr = np.asarray(results)
        print(f"  {name:22}: rel_l2 mean={arr.mean():.4e}  std={arr.std():.4e}  "
              f"runs={[f'{r:.4f}' for r in results]}", flush=True)
    print("SEED_SWEEP_DONE", flush=True)


def main(root="~/data"):
    root = os.path.expanduser(root)
    compare(root, "darcy", steps=100, n_modes=12, channels=32, batch=8)
    compare(root, "ns", steps=30, n_modes=16, channels=32, batch=32)
    compare(root, "burgers", steps=200, n_modes=6, channels=24, batch=16)
    print("ALL_COMPARE_DONE", flush=True)


def main_seeds(root="~/data"):
    root = os.path.expanduser(root)
    compare_seeds(root, "burgers", seeds=5, steps=200, n_modes=6, channels=24, batch=16)
    compare_seeds(root, "darcy", seeds=5, steps=50, n_modes=12, channels=32, batch=8)
    compare_seeds(root, "ns", seeds=3, steps=15, n_modes=16, channels=32, batch=32)
    print("ALL_SEED_SWEEPS_DONE", flush=True)
