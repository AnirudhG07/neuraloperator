"""
run_train_timed.py — retrain all 3 datasets at PAPER configs (FULL data), capture EXACT training
wall-clock, save the trained tensors, and print their shapes.  Runs ON the v5e.
Datasets pulled to ~/data (raw neuralop .pt); torch is used only to load them.
"""
import os
import numpy as np
import jax
import tpu.fno_reim_2d as M
import tpu.experimental.fno_proper as PR

DATA = os.path.expanduser("~/data")
print("jax", jax.__version__, "|", jax.devices(), flush=True)


def report(name, P, relL2, tsecs, xte, m):
    us, sps = PR.infer_us_per_sample(P, xte, m)
    print(f"RESULT {name}: relL2={relL2:.4e}  TRAIN={tsecs:.1f}s  infer={us:.2f}us/sample  "
          f"({sps:,.0f}/s)  params={PR.n_params(P):,}", flush=True)


def run(name, loader, cfg, save):
    try:
        xtr, ytr, xte, yte = loader()
        print(f"{name} data shapes: xtr{xtr.shape} ytr{ytr.shape} xte{xte.shape} yte{yte.shape}",
              flush=True)
        P, (xh, yh), rl, tsec = PR.train_fno(name, xtr, ytr, xte, yte, save=save, **cfg)
        report(name, P, rl, tsec, xh, cfg["m"])
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)


run("NS", lambda: M.load_ns(DATA, res=128, n_train=10000, n_test=1000),
    dict(width=32, m=16, steps=60, batch=32, lr=1e-3), "/tmp/ns_proper.npz")
run("DARCY", lambda: M.load_darcy(res=32, root=DATA),
    dict(width=32, m=12, steps=300, batch=8, lr=1e-3), "/tmp/darcy_proper.npz")
run("BURGERS", lambda: M.load_burgers(root=DATA),
    dict(width=24, m=6, steps=200, batch=16, lr=1e-4), "/tmp/burgers_proper.npz")

print("##### SAVED TENSOR SHAPES #####", flush=True)
for f in ("ns", "darcy", "burgers"):
    p = f"/tmp/{f}_proper.npz"
    if os.path.exists(p):
        d = np.load(p)
        print(f"{f}: {len(d.files)} arrays, {sum(d[k].size for k in d.files):,} params; "
              f"spectral_W arr_0 {d['arr_0'].shape}, lift arr_16 {d['arr_16'].shape}", flush=True)
print("TRAIN_ALL_DONE", flush=True)
