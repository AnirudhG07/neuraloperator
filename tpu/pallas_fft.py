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
BF = jnp.bfloat16
ACC = jnp.float32  # matmuls always accumulate in f32, whatever the operand dtype
_BYTES = {F: 4, BF: 2}  # bytes per real component (f32 -> complex64, bf16 -> "complex32")

# K-grid is embarrassingly parallel (each K-tile is an independent transform); telling
# Mosaic so lets it pipeline the tiles — overlap tile k+1's HBM read with tile k's compute.
_PARALLEL = pltpu.CompilerParams(dimension_semantics=("parallel",))

# radix-B matmul (3-D) / leaf matmul (2-D). operand dtype = the caller's; accumulate f32.
_EIN = lambda A, x: jnp.einsum('br,rck->bck', A, x, preferred_element_type=ACC)
_MAT = lambda A, x: jnp.matmul(A, x, preferred_element_type=ACC)


def _CB_reim(m, dt=F):
    """radix-m DFT matrix as (Cr=cos, Ci=sin, Cs=Cr+Ci) in dtype `dt` — built in VMEM.
    Cs is precomputed for the Karatsuba complex matmul."""
    k = jnp.arange(m)
    ang = -2 * jnp.pi * jnp.outer(k, k) / m
    cr, ci = jnp.cos(ang), jnp.sin(ang)
    return cr.astype(dt), ci.astype(dt), (cr + ci).astype(dt)


def _karatsuba(mm, Cr, Ci, Cs, xr, xi):
    """Complex matmul (Cr+iCi)@(xr+ixi) in 3 real matmuls (Gauss), not 4.
    `mm(A,x)` is the matmul primitive (_EIN for stages, _MAT for the leaf)."""
    m1, m2, m3 = mm(Cr, xr), mm(Ci, xi), mm(Cs, xr + xi)
    return m1 - m2, m3 - m1 - m2  # real = m1-m2,  imag = m3-m1-m2


def _fft_reim_real(xr, N, half=False, dt=F):
    """Radix-B FFT of REAL columns xr (N, batch), carried as real/imag in dtype `dt`.
    Peels one radix-B stage per iteration (length /B, batch *B), then a leaf DFT, then a
    reshape-only digit-reversal rebuild."""
    cr, ci, m, batch, stages, first = xr.astype(dt), None, N, xr.shape[1], [], True

    while m > B and m % B == 0:
        N1 = m // B
        Cr, Ci, Cs = _CB_reim(B, dt)

        if first:  # xi == 0: 2 matmuls
            x3 = cr.reshape(B, N1, batch)
            yr, yi, first = _EIN(Cr, x3), _EIN(Ci, x3), False
        else:  # Karatsuba: 3 matmuls
            yr, yi = _karatsuba(_EIN, Cr, Ci, Cs,
                cr.reshape(B, N1, batch), ci.reshape(B, N1, batch))

        c = jnp.arange(N1)[None, :]
        ang = (-2 * jnp.pi * (jnp.arange(B)[:, None] * c) / m).astype(F)  # twiddle W_m^(b*c)

        wr, wi = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]  # f32 twiddle on f32 yr,yi
        cr = (yr * wr - yi * wi).transpose(1, 0, 2).reshape(N1, B * batch).astype(dt)  # store dt
        ci = (yr * wi + yi * wr).transpose(1, 0, 2).reshape(N1, B * batch).astype(dt)

        stages.append((B, N1))
        m, batch = N1, B * batch

    Cr, Ci, Cs = _CB_reim(m, dt)  # leaf: 2 matmuls (real) or 3 (Karatsuba)
    nr, ni = (_MAT(Cr, cr), _MAT(Ci, cr)) if first else _karatsuba(_MAT, Cr, Ci, Cs, cr, ci)
    lr, li, length = nr, ni, m

    for (Bi, _n) in reversed(stages):  # digit-reversal (reshape only)
        lr = lr.reshape(length, Bi, -1).reshape(length * Bi, -1)
        li = li.reshape(length, Bi, -1).reshape(length * Bi, -1)
        length *= Bi

    lr, li = (lr[:N // 2], li[:N // 2]) if half else (lr, li)
    return lr.astype(dt), li.astype(dt)  # output in dt ("complex32" if bf16)


def _kernel(xr_ref, yr_ref, yi_ref, *, N, half, dt):
    """Kernel body: whole FFT of one K_TILE-column block, entirely in VMEM."""
    yr_ref[...], yi_ref[...] = _fft_reim_real(xr_ref[...], N, half, dt)


def _pallas(xr, K_TILE, interpret, half, dt):
    """Shared pallas_call: grid over K; output rows = N//2 (half) or N (full), dtype dt."""
    N, K = xr.shape
    rows = N // 2 if half else N
    return pl.pallas_call(
        partial(_kernel, N=N, half=half, dt=dt),
        grid=(K // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((rows, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((rows, K), dt)] * 2,
        compiler_params=_PARALLEL,
        interpret=interpret,
    )(xr)


def pallas_full(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N, K) -> FULL spectrum (yr, yi), each (N, K)  — all N bins.
    dt=BF gives the bf16 'complex32' variant (half the bytes, ~5e-3 error)."""
    return _pallas(xr, K_TILE, interpret, half=False, dt=dt)


def pallas_half(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N, K) -> HALF spectrum (yr, yi), each (N//2, K)  — Hermitian rfft.
    dt=BF gives the bf16 'complex32' variant (half the bytes, ~5e-3 error)."""
    return _pallas(xr, K_TILE, interpret, half=True, dt=dt)


def pallas_hbm_bytes(N, K, half=False, dt=F):
    """Analytical HBM traffic (VMEM-resident): read input once + write output once.
    Intermediates never touch HBM by design.  half writes only N//2 rows.
    dt sets bytes/component: f32 -> 4 (complex64), bf16 -> 2 ('complex32')."""
    dtb = _BYTES[dt]
    rows = N // 2 if half else N
    return N * K * dtb + 2 * rows * K * dtb  # read xr + write (yr, yi)


def jax_rfft(xr, half=True, dt=F):
    """Pure-JAX (no Pallas) real FFT via the SAME real/imag path as the kernel — same `dt`
    option (F -> complex64, BF -> bf16 'complex32').  Unlike the kernel this is plain XLA,
    so stages round-trip HBM; but it lets the bf16 win be used/measured in the JAX path.
    Real (N, K) -> (yr, yi).  half=True keeps the first N//2 bins."""
    return _fft_reim_real(xr, xr.shape[0], half=half, dt=dt)


# Row reversal
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


def pallas_trickB(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N, K) -> HALF spectrum via Trick B + self-made reversal.  See note above:
    correct and lowers, but the O(N^2) mirror makes it slower than pallas_half.
    (dt accepted for a uniform interface; the Trick-B path runs in f32.)"""
    xr = xr.astype(F)
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

