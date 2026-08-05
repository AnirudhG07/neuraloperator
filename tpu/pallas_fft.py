"""
pallas_fft.py  —  a VMEM-resident radix-B REAL-input FFT as a single Pallas TPU kernel.

Both public kernels take REAL input (N, K); they differ only in output length:
    pallas_full  -> all N bins        pallas_half  -> first N//2 bins (Hermitian rfft)
The whole transform runs in ONE kernel with the working set resident in VMEM, so the only
HBM traffic is: read the input once, write the output once (intermediates never leave VMEM).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

B = 128
F = jnp.float32
DT = 4  # bytes per f32 component

# K-grid is embarrassingly parallel (each K-tile is an independent transform); telling
# Mosaic so lets it pipeline the tiles — overlap tile k+1's HBM read with tile k's compute.
_PARALLEL = pltpu.CompilerParams(dimension_semantics=("parallel",))

_EIN = lambda A, x: jnp.einsum('br,rck->bck', A, x)  # radix-B matmul over axis 0 (3-D)
_MAT = lambda A, x: A @ x  # leaf matmul (2-D)


def _CB_reim(m):
    """radix-m DFT matrix as (Cr=cos, Ci=sin, Cs=Cr+Ci) f32 — built in VMEM, no HBM.
    Cs is precomputed for the Karatsuba complex matmul."""
    k = jnp.arange(m)
    ang = -2 * jnp.pi * jnp.outer(k, k) / m
    Cr, Ci = jnp.cos(ang).astype(F), jnp.sin(ang).astype(F)
    return Cr, Ci, Cr + Ci


def _karatsuba(mm, Cr, Ci, Cs, xr, xi):
    """Complex matmul (Cr+iCi)@(xr+ixi) in 3 real matmuls (Gauss), not 4.
    `mm(A,x)` is the matmul primitive (_EIN for stages, _MAT for the leaf)."""
    m1, m2, m3 = mm(Cr, xr), mm(Ci, xi), mm(Cs, xr + xi)
    return m1 - m2, m3 - m1 - m2  # real = m1-m2,  imag = m3-m1-m2


def _fft_reim_real(xr, N, half=False):
    """Radix-B FFT of REAL columns xr (N, batch), carried as real/imag f32.
    Peels one radix-B stage per iteration (length /B, batch *B), then a leaf DFT, then a
    reshape-only digit-reversal rebuild.  Flop savings: xi=0 first stage (2 matmuls) and
    Karatsuba elsewhere (3 matmuls).  half=True -> first N//2 bins (Hermitian rfft)."""
    cr, ci, m, batch, stages, first = xr, None, N, xr.shape[1], [], True
    while m > B and m % B == 0:
        N1 = m // B
        Cr, Ci, Cs = _CB_reim(B)
        if first:  # xi == 0: 2 matmuls
            x3 = cr.reshape(B, N1, batch)
            yr, yi, first = _EIN(Cr, x3), _EIN(Ci, x3), False
        else:  # Karatsuba: 3 matmuls
            yr, yi = _karatsuba(_EIN, Cr, Ci, Cs,
                cr.reshape(B, N1, batch), ci.reshape(B, N1, batch))
        c = jnp.arange(N1)[None, :]
        ang = (-2 * jnp.pi * (jnp.arange(B)[:, None] * c) / m).astype(F)  # twiddle W_m^(b*c)
        wr, wi = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
        cr = (yr * wr - yi * wi).transpose(1, 0, 2).reshape(N1, B * batch)  # twiddle + fold
        ci = (yr * wi + yi * wr).transpose(1, 0, 2).reshape(N1, B * batch)
        stages.append((B, N1))
        m, batch = N1, B * batch

    Cr, Ci, Cs = _CB_reim(m)  # leaf: 2 matmuls (real) or 3 (Karatsuba)
    nr, ni = (_MAT(Cr, cr), _MAT(Ci, cr)) if first else _karatsuba(_MAT, Cr, Ci, Cs, cr, ci)
    lr, li, length = nr, ni, m
    for (Bi, _n) in reversed(stages):  # digit-reversal (reshape only)
        lr = lr.reshape(length, Bi, -1).reshape(length * Bi, -1)
        li = li.reshape(length, Bi, -1).reshape(length * Bi, -1)
        length *= Bi
    return (lr[:N // 2], li[:N // 2]) if half else (lr, li)


def _kernel(xr_ref, yr_ref, yi_ref, *, N, half):
    """Kernel body: whole FFT of one K_TILE-column block, entirely in VMEM."""
    yr_ref[...], yi_ref[...] = _fft_reim_real(xr_ref[...], N, half)


def _pallas(xr, K_TILE, interpret, half):
    """Shared pallas_call: grid over K; output rows = N//2 (half) or N (full)."""
    N, K = xr.shape
    rows = N // 2 if half else N
    return pl.pallas_call(
        partial(_kernel, N=N, half=half),
        grid=(K // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((rows, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((rows, K), F)] * 2,
        compiler_params=_PARALLEL,
        interpret=interpret,
    )(xr)


def pallas_full(xr, K_TILE=128, interpret=True):
    """Real (N, K) -> FULL spectrum (yr, yi), each (N, K)  — all N bins."""
    return _pallas(xr, K_TILE, interpret, half=False)


def pallas_half(xr, K_TILE=128, interpret=True):
    """Real (N, K) -> HALF spectrum (yr, yi), each (N//2, K)  — Hermitian rfft."""
    return _pallas(xr, K_TILE, interpret, half=True)


def pallas_hbm_bytes(N, K, half=False):
    """Analytical HBM traffic (VMEM-resident): read input once + write output once.
    Intermediates never touch HBM by design.  half writes only N//2 rows."""
    rows = N // 2 if half else N
    return N * K * DT + 2 * rows * K * DT  # read xr + write (yr, yi)


# ─── a real Trick-B rfft with a SELF-MADE row reversal ──────────────────────
# Mosaic has no rev/gather, but a row reversal is O(N) data movement, done as: reverse
# within each 128-block (one fixed 128x128 anti-identity matmul, batched) + reverse the
# block ORDER (a small nb x nb matmul), nb = M/128.  The Hermitian mirror conj(Z[(h-k)%h])
# is then roll(reverse(Z), 1) — and pltpu.roll lowers.  This enables the genuine Trick B
# (even/odd -> HALF-length FFT -> mirror -> twist), which halves the FFT compute.
# VERDICT (see bench): lowers, correct, and the O(N) mirror is cheap enough that Trick B
# BEATS pallas_half at large N (~0.76x flops at N=16384); roughly ties at small N.
# (The nb x nb block-order reverse is O(nb^2); fine while nb<=128 i.e. N<=~16384 — the
# VMEM-resident regime.  Larger N would need to recurse the block reversal too.)
def _rev(x):
    """Reverse the rows of x (M, K), M a multiple of 128, in O(M*128) via block matmuls."""
    M, K = x.shape
    nb = M // 128
    c = jnp.arange(128)
    Jb = (c[None, :] == (127 - c[:, None])).astype(F)  # 128x128 anti-identity
    x3 = jnp.einsum('cd,ndk->nck', Jb, x.reshape(nb, 128, K))  # reverse within each block
    if nb > 1:
        n = jnp.arange(nb)
        Jn = (n[None, :] == (nb - 1 - n[:, None])).astype(F)  # nb x nb anti-identity
        x3 = jnp.einsum('nm,mck->nck', Jn, x3)  # reverse block order
    return x3.reshape(M, K)
def _fft_c(cr, ci, N):
    """General complex radix-B FFT (Karatsuba on every stage), full N-bin output."""
    m, batch, stages = N, cr.shape[1], []
    while m > B and m % B == 0:
        N1 = m // B
        Cr, Ci, Cs = _CB_reim(B)
        yr, yi = _karatsuba(_EIN, Cr, Ci, Cs, cr.reshape(B, N1, batch), ci.reshape(B, N1, batch))
        c = jnp.arange(N1)[None, :]
        ang = (-2 * jnp.pi * (jnp.arange(B)[:, None] * c) / m).astype(F)
        wr, wi = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
        cr = (yr * wr - yi * wi).transpose(1, 0, 2).reshape(N1, B * batch)
        ci = (yr * wi + yi * wr).transpose(1, 0, 2).reshape(N1, B * batch)
        stages.append((B, N1))
        m, batch = N1, B * batch
    Cr, Ci, Cs = _CB_reim(m)
    lr, li = _karatsuba(_MAT, Cr, Ci, Cs, cr, ci)
    length = m
    for (Bi, _n) in reversed(stages):
        lr = lr.reshape(length, Bi, -1).reshape(length * Bi, -1)
        li = li.reshape(length, Bi, -1).reshape(length * Bi, -1)
        length *= Bi
    return lr, li


def _trickB(xr, N):
    """Even/odd -> half-length FFT -> self-made mirror -> twist.  Returns first N//2 bins."""
    h, batch = N // 2, xr.shape[1]
    x2 = xr.reshape(h, 2, batch)
    Zr, Zi = _fft_c(x2[:, 0, :], x2[:, 1, :], h)  # HALF-length FFT of (even + i*odd)
    Mr = pltpu.roll(_rev(Zr), 1, axis=0)  # mirror M[k]=conj(Z[(h-k)%h]) = roll(rev(Z),1)
    Mi = -pltpu.roll(_rev(Zi), 1, axis=0)  # conj -> negate imag
    Er, Ei = 0.5 * (Zr + Mr), 0.5 * (Zi + Mi)  # even sub-DFT E = 0.5(Z+M)
    Dr, Di = Zr - Mr, Zi - Mi
    Or_, Oi = 0.5 * Di, -0.5 * Dr  # odd sub-DFT O = -0.5j(Z-M)
    k = jnp.arange(h)
    twr = jnp.cos(-2 * jnp.pi * k / N).astype(F)[:, None]
    twi = jnp.sin(-2 * jnp.pi * k / N).astype(F)[:, None]
    return Er + (twr * Or_ - twi * Oi), Ei + (twr * Oi + twi * Or_)  # X[k]=E[k]+W_N^k O[k]


def _trickB_kernel(xr_ref, yr_ref, yi_ref, *, N):
    yr_ref[...], yi_ref[...] = _trickB(xr_ref[...], N)


def pallas_trickB(xr, K_TILE=128, interpret=True):
    """Real (N, K) -> HALF spectrum via Trick B + self-made reversal.  See note above:
    correct and lowers, but the O(N^2) mirror makes it slower than pallas_half."""
    N, K = xr.shape
    h = N // 2
    return pl.pallas_call(
        partial(_trickB_kernel, N=N),
        grid=(K // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((h, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((h, K), F)] * 2,
        compiler_params=_PARALLEL,
        interpret=interpret,
    )(xr)

