"""
bench_jax.py  —  one configurable harness to compare FFT engines and cases.

All the comparison logic that used to live in pallas_fft.py / rfft_variants.py is here.
An "engine" turns a real (N, K) input into a spectrum; a "case" is just a named list of
engines to line up.  For each (N, engine) it reports rel_err (vs numpy), HBM bytes, flops,
and whether the Pallas kernels lower to TPU.  Bytes/flops come from XLA cost_analysis for
the JAX engines; the Pallas kernels report ANALYTICAL VMEM bytes (interpret over-counts)
and interpret flops.  The x-columns are ratios vs the first engine in the list (the
baseline), so ordering picks the baseline.

Configure it two ways:
    compare(case="rfft")                      # a predefined case (see CASES)
    compare(engines=["jnp.fft", "pallas_full"], Ns=(1024, 16384), K=256, baseline="jnp.fft")

Run:  python -m tpu.profiling.bench_jax        (or edit __main__ at the bottom)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "tpu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from tpu import fft_jax
from tpu.fft_iter import fft_iter
from tpu.pallas_fft import pallas_full, pallas_half, pallas_trickB, pallas_hbm_bytes
from tpu.rfft_variants import rfft_trickA, rfft_trickB

C = jnp.complex64
K_TILE = 128  # Pallas block width (must be a multiple of 128)


# ─── engine registry ────────────────────────────────────────────────────────
# A JAX engine is a fn(real (N,K)) -> spectrum; bytes/flops via cost_analysis.
# A Pallas engine is a kernel(xr, K_TILE, interpret); bytes are analytical (VMEM).
def _jax(fn):
    return {"pallas": False, "fn": fn}


def _pal(kernel, half):
    return {"pallas": True, "kernel": kernel, "half": half}


ENGINES = {
    "jnp.fft":     _jax(lambda xr: jnp.fft.fft(xr.astype(C), axis=0)),  # full
    "jnp.rfft":    _jax(lambda xr: jnp.fft.rfft(xr, axis=0)),  # half, native
    "fft_recur":   _jax(lambda xr: fft_jax.small_dft_baseline(xr.astype(C), xr.shape[0])),
    "fft_iter":    _jax(lambda xr: fft_iter(xr.astype(C))),  # full
    "trickA":      _jax(lambda xr: rfft_trickA(xr, fft_iter)),  # half, on fft_iter
    "trickB":      _jax(lambda xr: rfft_trickB(xr, fft_iter)),  # half, on fft_iter
    "pallas_full":   _pal(pallas_full, half=False),  # full, VMEM
    "pallas_half":   _pal(pallas_half, half=True),  # half via output truncation, VMEM
    "pallas_trickB": _pal(pallas_trickB, half=True),  # half via self-made reversal (O(N^2))
}

# Named line-ups. The first entry is the baseline (x = 1.00).
CASES = {
    "full":             ["jnp.fft", "fft_recur", "fft_iter", "pallas_full"],
    "rfft":             ["jnp.rfft", "trickA", "trickB", "pallas_half"],
    "pallas_vs_native": ["jnp.rfft", "pallas_half"],
    "pallas_rfft":      ["pallas_half", "pallas_trickB"],  # truncation vs self-made reversal
}


# ─── shared helpers ─────────────────────────────────────────────────────────
def _to_cplx(out):
    if isinstance(out, tuple):
        return np.asarray(out[0]) + 1j * np.asarray(out[1])
    return np.asarray(out)


def _relerr(out, ref):
    g = _to_cplx(out)
    return float(np.max(np.abs(g - ref)) / np.max(np.abs(ref)))


def _measure(fn, x):
    """(bytes, flops, output) from one compiled run + XLA cost_analysis."""
    jfn = jax.jit(fn)
    out = jfn(x)
    ca = jfn.lower(x).compile().cost_analysis() or {}
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    return float(ca.get("bytes accessed", 0) or 0), float(ca.get("flops", 0) or 0), out


def _lowers(fn, x):
    try:
        pl.lower_as_mlir(fn, x, platforms=["tpu"])
        return "yes"
    except Exception:
        return "NO"


def _h(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def _eval(spec, xr):
    """Run one engine on real xr -> (bytes, flops, output, lowers)."""
    xj = jnp.asarray(xr)
    if spec["pallas"]:
        kern = spec["kernel"]
        out = kern(xj, K_TILE, True)  # interpret run (correctness)
        _, fl, _ = _measure(lambda x: kern(x, K_TILE, True), xj)  # interpret flops
        by = pallas_hbm_bytes(*xr.shape, half=spec["half"])  # analytical VMEM bytes
        low = _lowers(lambda x: kern(x, K_TILE, False), xj)  # TPU lowering check
        return by, fl, out, low
    by, fl, out = _measure(spec["fn"], xj)
    return by, fl, out, "-"


# ─── the one comparison entry point ─────────────────────────────────────────
def compare(engines=None, case=None, Ns=(256, 1024, 16384), K=256, baseline=None):
    """Compare `engines` (or the named `case`) over `Ns` at batch `K`.
    x-columns are ratios vs `baseline` (defaults to the first engine listed)."""
    names = engines or (CASES[case] if case else list(ENGINES))
    baseline = baseline or names[0]
    title = case or ", ".join(names)
    print("=" * 82)
    print(f"[{title}]   K={K}   (x = vs {baseline})   "
          f"pallas bytes = analytical VMEM (confirm on v5e)")
    print("=" * 82)
    print(f"  {'N':>8} {'engine':>12} {'rel_err':>9} {'bytes':>10} {'x':>6} "
          f"{'flops':>10} {'x':>6} {'lowers':>6}")
    for N in Ns:
        xr = np.random.randn(N, K).astype(np.float32)
        ref = np.fft.fft(xr, axis=0)  # full reference; slice per output rows
        res = {n: _eval(ENGINES[n], xr) for n in names}
        b0, f0 = res[baseline][0], res[baseline][1]
        for n in names:
            by, fl, out, low = res[n]
            rows = (out[0] if isinstance(out, tuple) else out).shape[0]
            rel = _relerr(out, ref[:rows])
            print(f"  {N:>8,} {n:>12} {rel:>9.1e} {_h(by):>10} {by/b0:>5.2f}x "
                  f"{_h(fl):>10} {fl/max(f0,1):>5.2f}x {low:>6}")
        print()


if __name__ == "__main__":
    # edit these calls to configure what to compare
    compare(case="full")
    compare(case="rfft")
    compare(case="pallas_vs_native", Ns=(1024, 16384, 65536))
