"""
bench_jax.py  —  one configurable harness to compare FFT engines and cases.

All the comparison logic that used to live in pallas_fft.py / rfft_variants.py is here.
An "engine" turns a real (N, K) input into a spectrum; a "case" is a named list of engines.
`compare()` reports rel_err (vs numpy), bytes, flops, and TIME per (N, engine).

What is MEASURED vs ESTIMATED (important):
  * time  -> MEASURED (median blocked wall-clock).  On TPU this is the real verdict.
  * flops -> ESTIMATED (XLA cost_analysis static model; never counted at runtime).
  * bytes -> ESTIMATED (cost_analysis for JAX engines; an ANALYTICAL formula for Pallas,
             since a Pallas custom-call is opaque to cost_analysis).
For the REAL measured mem / MXU% / VPU% / HBM bytes, use `profile()` -> an xprof trace.

Configure:
    compare(case="rfft")                      # a predefined case (see CASES)
    compare(engines=["jnp.fft", "pallas_full"], Ns=(1024, 16384), K=256, baseline="jnp.fft")
    profile(case="rfft", N=1024, K=256)       # real device profile (open in TensorBoard)

Run:  python -m tpu.profiling.bench_jax        (or edit __main__ at the bottom)
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "tpu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np

from tpu import fft_jax
from tpu.fft_iter import fft_iter
from tpu.pallas_fft import (
    BF,
    F,
    jax_rfft,
    pallas_full,
    pallas_half,
    pallas_hbm_bytes,
    pallas_trickB,
)
from tpu.rfft_variants import rfft_trickA, rfft_trickB

C = jnp.complex64
K_TILE = 128  # Pallas block width (must be a multiple of 128)
BACKEND = jax.devices()[0].platform
# Pallas runs the REAL kernel on TPU (interpret=False) and the CPU emulation elsewhere.
# So on CPU the pallas `time` column is interpret emulation (NOT representative) — trust it
# only on a real v5e run.  JAX-engine times are real wall-clock on whatever backend this is.
_ITP = (BACKEND != "tpu")
TIME_REPS = 15  # median over this many blocked runs

ACCUM  = jnp.complex64
CDTYPE = jnp.complex64

# ─── engine registry ────────────────────────────────────────────────────────
# A JAX engine is a fn(real (N,K)) -> spectrum; bytes/flops via cost_analysis.
# A Pallas engine is a kernel(xr, K_TILE, interpret, dt); bytes are analytical (VMEM).
def _jax(fn):
    return {"pallas": False, "fn": fn}


def _pal(kernel, half, dt=F):
    return {"pallas": True, "kernel": kernel, "half": half, "dt": dt}


ENGINES = {
    "jnp.fft":     _jax(lambda xr: jnp.fft.fft(xr.astype(C), axis=0)),  # full
    "jnp.rfft":    _jax(lambda xr: jnp.fft.rfft(xr, axis=0)),  # half, native
    "fft_recur":   _jax(lambda xr: fft_jax.small_dft_baseline(xr.astype(C), xr.shape[0])),
    "fft_iter":    _jax(lambda xr: fft_iter(xr.astype(C))),  # full
    "trickA":      _jax(lambda xr: rfft_trickA(xr, fft_iter)),  # half, on fft_iter
    "trickB":      _jax(lambda xr: rfft_trickB(xr, fft_iter)),  # half, on fft_iter
    "jax_half":      _jax(lambda xr: jax_rfft(xr, True, F)),  # pure-JAX real/imag, f32
    "jax_half_bf16": _jax(lambda xr: jax_rfft(xr.astype(BF), True, BF)),  # pure-JAX, bf16
    "pallas_full":     _pal(pallas_full, half=False),  # full, VMEM
    "pallas_half":     _pal(pallas_half, half=True),  # half via output truncation, VMEM
    "pallas_half_bf16": _pal(pallas_half, half=True, dt=BF),  # bf16 'complex32' — half bytes
    "pallas_trickB":   _pal(pallas_trickB, half=True),  # half via self-made reversal
}

# Named line-ups. The first entry is the baseline (x = 1.00).
CASES = {
    "full":             ["jnp.fft", "fft_recur", "fft_iter", "pallas_full"],
    "rfft":             ["jnp.rfft", "trickA", "trickB", "pallas_half"],
    "pallas_vs_native": ["jnp.rfft", "pallas_half"],
    "pallas_rfft":      ["pallas_half", "pallas_trickB"],  # truncation vs self-made reversal
    "bf16":             ["pallas_half", "pallas_half_bf16", "jax_half", "jax_half_bf16"],
}


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


def _h(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def _ft(t):
    """Format a wall-clock time (seconds)."""
    if t is None:
        return "-"
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    if t < 1:
        return f"{t*1e3:.2f}ms"
    return f"{t:.2f}s"


def _time(fn, x, reps=TIME_REPS):
    """Median blocked wall-clock of `fn(x)` (jitted), after one warmup/compile."""
    jfn = jax.jit(fn)
    jax.block_until_ready(jfn(x))  # warmup + compile (not timed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(x))
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def _flops(fn, x):
    """flops from a compile + cost_analysis (no run). Returns 0 if unavailable."""
    try:
        ca = jax.jit(fn).lower(x).compile().cost_analysis() or {}
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        return float(ca.get("flops", 0) or 0)
    except Exception:
        return 0.0


def _eval(spec, xr):
    """Run one engine on real xr -> (bytes, flops, output, time)."""
    xj = jnp.asarray(xr)
    if spec["pallas"]:
        kern, half, dt = spec["kernel"], spec["half"], spec["dt"]
        xd = xj.astype(dt)  # input in the kernel's dtype (bf16 halves the read)
        run = lambda x: kern(x, K_TILE, _ITP, dt)  # real kernel on TPU, emulation on CPU
        out = run(xd)  # correctness (and this is what fails/ERRs if VMEM OOMs)
        t = _time(run, xd)
        # A Pallas kernel is an OPAQUE custom-call, so cost_analysis can't see inside it
        # (bytes/flops come back 0 on TPU). So bytes = analytical VMEM bound (profiler
        # confirms the real HBM); flops = from an interpret compile (runs the real matmuls).
        by = pallas_hbm_bytes(*xr.shape, half=half, dt=dt)
        fl = _flops(lambda x: kern(x, K_TILE, True, dt), xd)
        return by, fl, out, t
    by, fl, out = _measure(spec["fn"], xj)
    return by, fl, out, _time(spec["fn"], xj)


# ─── the one comparison entry point ─────────────────────────────────────────
def compare(engines=None, case=None, Ns=(256, 1024, 16384), K=256, baseline=None):
    """Compare `engines` (or the named `case`) over `Ns` at batch `K`.
    x-columns are ratios vs `baseline` (defaults to the first engine listed)."""
    names = engines or (CASES[case] if case else list(ENGINES))
    baseline = baseline or names[0]
    title = case or ", ".join(names)
    tnote = "real" if BACKEND == "tpu" else f"{BACKEND}/pallas-interp"
    print("=" * 88)
    print(f"[{title}]   K={K}   backend={BACKEND}   (x = vs {baseline})   "
          f"pallas bytes = analytical VMEM (profiler = real HBM); time = {tnote}")
    print("=" * 88)
    print(f"  {'N':>8} {'engine':>13} {'rel_err':>9} {'bytes':>9} {'x':>6} "
          f"{'flops':>9} {'x':>6} {'time':>9} {'x':>6}")
    for N in Ns:
        xr = np.random.randn(N, K).astype(np.float32)
        ref = np.fft.fft(xr, axis=0)  # full reference; slice per output rows
        res = {}
        for n in names:
            try:
                res[n] = _eval(ENGINES[n], xr)
            except Exception as e:  # e.g. VMEM OOM at large N — don't kill the whole run
                res[n] = ("ERR", repr(e)[:50])
        base = res[baseline]
        b0, f0, t0 = (base[0], base[1], base[3]) if base[0] != "ERR" else (1, 1, 1)
        for n in names:
            r = res[n]
            if r[0] == "ERR":
                print(f"  {N:>8,} {n:>13}   ERR: {r[1]}")
                continue
            by, fl, out, t = r
            rows = (out[0] if isinstance(out, tuple) else out).shape[0]
            rel = _relerr(out, ref[:rows])
            print(f"  {N:>8,} {n:>13} {rel:>9.1e} {_h(by):>9} {by/b0:>5.2f}x "
                  f"{_h(fl):>9} {fl/max(f0,1):>5.2f}x {_ft(t):>9} {t/t0:>5.2f}x")
        print()


# ─── real profile capture (xprof) — the only way to get MEASURED mem/MXU/VPU ─────
def profile(engines=None, case=None, N=1024, K=256, logdir="/tmp/tpu_trace", reps=20):
    """Capture a REAL TPU profile of each engine (unlike the estimated bytes/flops above).
    Compile/warm up OUTSIDE the trace, then run each engine `reps` times under a named_scope
    so it's identifiable.  What the trace gives you (all MEASURED on the device):
        memory_viewer -> real HBM bytes         op_profile   -> MXU / VPU utilization
        trace_viewer  -> per-op timeline        op_stats     -> per-op time + achieved BW
    View (Colab):  %load_ext tensorboard ;  %tensorboard --logdir <logdir>   (PROFILE tab)."""
    if BACKEND != "tpu":
        print(f"!! backend={BACKEND}: a profile here is not v5e — run on a Colab TPU runtime.")
    names = engines or (CASES[case] if case else list(ENGINES))
    xj = jnp.asarray(np.random.randn(N, K).astype(np.float32))

    runs = {}  # name -> (jitted fn, input) ; compiled + warmed up before tracing
    for n in names:
        spec = ENGINES[n]
        if spec["pallas"]:
            kern, dt = spec["kernel"], spec["dt"]
            fn = jax.jit(lambda x, kern=kern, dt=dt: kern(x, K_TILE, _ITP, dt))
            x = xj.astype(dt)
        else:
            fn = jax.jit(spec["fn"])
            x = xj
        try:
            jax.block_until_ready(fn(x))  # warmup/compile OUTSIDE the trace
            runs[n] = (fn, x)
        except Exception as e:
            print(f"  skip {n}: {repr(e)[:60]}")

    with jax.profiler.trace(logdir):
        for n, (fn, x) in runs.items():
            with jax.named_scope(n):
                for _ in range(reps):
                    jax.block_until_ready(fn(x))

    print(f"\nprofile written to {logdir}  (N={N}, K={K}, {len(runs)} engines x {reps} runs)")
    print(f"view:  %load_ext tensorboard ;  %tensorboard --logdir {logdir}   -> PROFILE tab")
    print("  memory_viewer = real HBM bytes | op_profile = MXU/VPU% | trace_viewer = timeline")


if __name__ == "__main__":
    # edit these calls to configure what to compare
    compare(case="full")
    compare(case="rfft")
    compare(case="pallas_vs_native", Ns=(1024, 16384, 65536))
    compare(case="pallas_rfft", Ns=(1024, 16384, 65536))
    compare(case="bf16", Ns=(1024, 16384, 65536))

    # real measured profile (open the trace in TensorBoard's PROFILE tab):
    profile(case="rfft", N=1024, K=256)

