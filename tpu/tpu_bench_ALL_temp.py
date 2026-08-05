"""
tpu_bench_ALL_temp.py  —  TEMP throwaway benchmark (delete after today).

Self-contained: NO repo imports, upload straight to a Colab **TPU v5e** runtime and run
    !python tpu_bench_ALL_temp.py
It benchmarks FIVE engines on the same task (real (N,K) -> spectrum), reporting for each:
    rel_err        vs numpy.fft (correctness)
    time           median blocked wall-clock (the real metric)
    flops, TFLOP/s XLA cost_analysis flops / measured time   (+ % of v5e 197 TFLOP/s peak)
    bytes,  GB/s   XLA cost_analysis 'bytes accessed' / time (+ % of v5e 819 GB/s peak)
    AI             flops/byte (roofline x-axis; v5e ridge = 240)

Engines:
    jnp.fft       XLA native FFT (butterfly O(N logN))          — the bar to beat
    fft_recur     our RECURSIVE radix-B matmul FFT (baseline)
    fft_iter      our ITERATIVE radix-B matmul FFT (optimized JAX)
    pallas        our VMEM-resident radix-B FFT (optimized Pallas, full complex)
    pallas_rfft   our VMEM-resident real-input Hermitian rfft (optimized Pallas, half out)

NOTE on Pallas bytes: on a real TPU, cost_analysis reports the ACTUAL kernel HBM traffic
(unlike CPU interpret mode, which over-counts). So on v5e these numbers are trustworthy.
Large N may OOM the VMEM-resident Pallas kernels — those rows print ERR and are skipped.
"""
import time as _time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
    HAVE_PALLAS = True
except Exception:
    HAVE_PALLAS = False

B = 128
C = jnp.complex64
F = jnp.float32
BACKEND = jax.devices()[0].platform

# v5e roofline
PEAK_TFLOPS = 197.0
PEAK_GBS = 819.0
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GBS * 1e9)     # ~240 flop/byte

# sweep
NS = [256, 1024, 4096, 16384, 65536, 262144]
K = 256
K_TILE = 128


# ─────────────────────────── engines ───────────────────────────
def dft_matrix(m):
    k = jnp.arange(m)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m).astype(C)


# 1) recursive baseline (complex twiddle)
def small_dft_recur(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=C)
    N1 = m // B
    Kb = vecs.shape[1]
    Xb = vecs.reshape(B, N1, Kb)
    Yb = jnp.einsum('br,rck->bck', dft_matrix(B), Xb, preferred_element_type=C)
    r = jnp.arange(B)[:, None]
    c = jnp.arange(N1)[None, :]
    Zb = Yb * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(C)[:, :, None]
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * Kb)
    return small_dft_recur(Zc, N1).reshape(N1, B, Kb).reshape(m, Kb)


# 2) iterative optimized JAX (complex)
def fft_iter(x):
    N, Kb = x.shape
    cur, m, batch, stages = x, N, Kb, []
    while m > B and m % B == 0:
        N1 = m // B
        cur = cur.reshape(B, N1, batch)
        cur = jnp.einsum('br,rck->bck', dft_matrix(B), cur, preferred_element_type=C)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        cur = cur * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(C)[:, :, None]
        cur = cur.transpose(1, 0, 2).reshape(N1, B * batch)
        stages.append((B, N1))
        m, batch = N1, B * batch
    cur = dft_matrix(m) @ cur
    out, length = cur, m
    for (Bi, _n) in reversed(stages):
        out = out.reshape(length, Bi, -1).reshape(length * Bi, -1)
        length = length * Bi
    return out


# real/imag primitives for Pallas (no complex64 in Mosaic)
def _CB_reim(m):
    k = jnp.arange(m)
    ang = -2 * jnp.pi * jnp.outer(k, k) / m
    return jnp.cos(ang).astype(F), jnp.sin(ang).astype(F)


def _fft_reim(xr, xi, N):
    cr, ci, m, batch, stages = xr, xi, N, xr.shape[1], []
    while m > B and m % B == 0:
        N1 = m // B
        cr = cr.reshape(B, N1, batch)
        ci = ci.reshape(B, N1, batch)
        Cr, Ci = _CB_reim(B)
        yr = jnp.einsum('br,rck->bck', Cr, cr) - jnp.einsum('br,rck->bck', Ci, ci)
        yi = jnp.einsum('br,rck->bck', Cr, ci) + jnp.einsum('br,rck->bck', Ci, cr)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        ang = (-2 * jnp.pi * (r * c) / m).astype(F)
        wr = jnp.cos(ang)[:, :, None]
        wi = jnp.sin(ang)[:, :, None]
        cr = (yr * wr - yi * wi).transpose(1, 0, 2).reshape(N1, B * batch)
        ci = (yr * wi + yi * wr).transpose(1, 0, 2).reshape(N1, B * batch)
        stages.append((B, N1))
        m, batch = N1, B * batch
    Cr, Ci = _CB_reim(m)
    nr, ni = Cr @ cr - Ci @ ci, Cr @ ci + Ci @ cr
    lr, li, length = nr, ni, m
    for (Bi, _n) in reversed(stages):
        lr = lr.reshape(length, Bi, -1).reshape(length * Bi, -1)
        li = li.reshape(length, Bi, -1).reshape(length * Bi, -1)
        length = length * Bi
    return lr, li


def _fft_reim_real(xr, N):
    cr, ci, m, batch, stages, first = xr, None, N, xr.shape[1], [], True
    while m > B and m % B == 0:
        N1 = m // B
        Cr, Ci = _CB_reim(B)
        if first:
            x3 = cr.reshape(B, N1, batch)
            yr = jnp.einsum('br,rck->bck', Cr, x3)
            yi = jnp.einsum('br,rck->bck', Ci, x3)
            first = False
        else:
            cr3 = cr.reshape(B, N1, batch)
            ci3 = ci.reshape(B, N1, batch)
            yr = jnp.einsum('br,rck->bck', Cr, cr3) - jnp.einsum('br,rck->bck', Ci, ci3)
            yi = jnp.einsum('br,rck->bck', Cr, ci3) + jnp.einsum('br,rck->bck', Ci, cr3)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        ang = (-2 * jnp.pi * (r * c) / m).astype(F)
        wr = jnp.cos(ang)[:, :, None]
        wi = jnp.sin(ang)[:, :, None]
        cr = (yr * wr - yi * wi).transpose(1, 0, 2).reshape(N1, B * batch)
        ci = (yr * wi + yi * wr).transpose(1, 0, 2).reshape(N1, B * batch)
        stages.append((B, N1))
        m, batch = N1, B * batch
    Cr, Ci = _CB_reim(m)
    if first:
        nr, ni = Cr @ cr, Ci @ cr
    else:
        nr, ni = Cr @ cr - Ci @ ci, Cr @ ci + Ci @ cr
    lr, li, length = nr, ni, m
    for (Bi, _n) in reversed(stages):
        lr = lr.reshape(length, Bi, -1).reshape(length * Bi, -1)
        li = li.reshape(length, Bi, -1).reshape(length * Bi, -1)
        length = length * Bi
    return lr[:N // 2], li[:N // 2]


def _pallas_full_kernel(xr_ref, yr_ref, yi_ref, *, N):
    yr, yi = _fft_reim(xr_ref[...], jnp.zeros_like(xr_ref[...]), N)
    yr_ref[...] = yr
    yi_ref[...] = yi


def _pallas_rfft_kernel(xr_ref, yr_ref, yi_ref, *, N):
    yr, yi = _fft_reim_real(xr_ref[...], N)
    yr_ref[...] = yr
    yi_ref[...] = yi


def pallas_fft(xr):
    N, Kk = xr.shape
    return pl.pallas_call(
        partial(_pallas_full_kernel, N=N), grid=(Kk // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((N, Kk), F)] * 2, interpret=False)(xr)


def pallas_rfft(xr):
    N, Kk = xr.shape
    h = N // 2
    return pl.pallas_call(
        partial(_pallas_rfft_kernel, N=N), grid=(Kk // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((h, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((h, Kk), F)] * 2, interpret=False)(xr)


# ─────────────────────────── harness ───────────────────────────
def _to_complex(out):
    if isinstance(out, (tuple, list)):
        return np.asarray(out[0]) + 1j * np.asarray(out[1])
    return np.asarray(out)


def _measure(fn, x, reps=50):
    jfn = jax.jit(fn)
    out = jfn(x)
    jax.block_until_ready(out)
    try:
        ca = jfn.lower(x).compile().cost_analysis() or {}
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        flops = float(ca.get("flops", 0) or 0)
        byts = float(ca.get("bytes accessed", 0) or 0)
    except Exception:
        flops = byts = 0.0
    ts = []
    for _ in range(reps):
        t0 = _time.perf_counter()
        jax.block_until_ready(jfn(x))
        ts.append(_time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], flops, byts, out


def _h(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def main():
    print(f"backend = {BACKEND.upper()}   K={K}   v5e peak {PEAK_TFLOPS:.0f} TFLOP/s / "
          f"{PEAK_GBS:.0f} GB/s   ridge {RIDGE:.0f} flop/byte")
    if BACKEND != "tpu":
        print("!! NOT a TPU — numbers are not v5e; run on a Colab TPU runtime.")
    if not HAVE_PALLAS:
        print("!! pallas unavailable — pallas rows skipped.")

    engines = [
        ("jnp.fft",   lambda x: jnp.fft.fft(x.astype(C), axis=0)),
        ("fft_recur", lambda x: small_dft_recur(x.astype(C), x.shape[0])),
        ("fft_iter",  lambda x: fft_iter(x.astype(C))),
    ]
    if HAVE_PALLAS:
        engines += [("pallas", pallas_fft), ("pallas_rfft", pallas_rfft)]

    hdr = (f"\n{'N':>8} {'engine':>12} {'rel_err':>9} {'time':>10} "
           f"{'TFLOP/s':>9} {'%pk':>5} {'GB/s':>8} {'%pk':>5} {'AI':>6} {'bound':>5}")
    for N in NS:
        print(hdr)
        print("  " + "-" * 92)
        xr = np.random.randn(N, K).astype(np.float32)
        ref = np.fft.fft(xr, axis=0)
        xj = jnp.asarray(xr)
        for name, fn in engines:
            try:
                t, flops, byts, out = _measure(fn, xj)
                got = _to_complex(out)
                rows = got.shape[0]
                rel = np.max(np.abs(got - ref[:rows])) / np.max(np.abs(ref[:rows]))
                tfs = flops / t / 1e12 if t else 0
                gbs = byts / t / 1e9 if t else 0
                ai = flops / byts if byts else 0
                bound = "mem" if (ai and ai < RIDGE) else "cmp"
                tstr = f"{t*1e6:7.1f}us" if t < 1e-3 else f"{t*1e3:7.2f}ms"
                print(f"{N:>8} {name:>12} {rel:>9.1e} {tstr:>10} "
                      f"{tfs:>9.1f} {100*tfs/PEAK_TFLOPS:>4.0f}% {gbs:>8.0f} "
                      f"{100*gbs/PEAK_GBS:>4.0f}% {ai:>6.1f} {bound:>5}")
            except Exception as e:
                print(f"{N:>8} {name:>12} {'ERR':>9}  {repr(e)[:52]}")
    print("\ntime = median blocked wall-clock (the real metric). bytes/flops = XLA "
          "cost_analysis (real on TPU).\nrfft rel_err is vs the first N//2 bins. "
          "pallas ERR at large N = VMEM OOM (expected; needs cache-blocking).")


if __name__ == "__main__":
    main()
