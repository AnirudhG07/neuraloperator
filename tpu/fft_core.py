"""
fft_core.py — shared radix-B "FFT pass" building blocks, used by BOTH 1D and 2D, JAX and
Pallas.  Pure jnp numerics (no pallas_call here):

  constants        B, F, BF, I8, I32, ACC, CDTYPE, _BYTES, _EIN, _MAT
  DFT matrices     dft_matrix (complex),  _CB_reim (real/imag + Karatsuba sum)
  complex matmul   _karatsuba (Gauss: 3 real matmuls, not 4)
  the FFT pass     _fft_reim_real (staged radix-B REAL rfft),  _fft_c (staged complex FFT)
  helper           _rev (O(N) row reversal via block anti-identity matmuls)

The entry points live in tpu/fft1d/{jax_fft,pallas_fft}.py and tpu/fft2d/{jax_fft,pallas_fft}.py
and all import their numerics from here — so the radix-B pass is written once.
"""
import jax.numpy as jnp

B = 128
F = jnp.float32
BF = jnp.bfloat16
I8 = jnp.int8
I32 = jnp.int32
ACC = jnp.float32          # matmuls accumulate in f32 whatever the operand dtype
CDTYPE = jnp.complex64     # for the complex iterative/recursive 1D FFTs (fft_iter/fft_recur)
_BYTES = {F: 4, BF: 2, I8: 1}  # bytes/real component (f32->complex64, bf16->"c32", int8->"c16")

# radix-B matmul (3-D) / leaf matmul (2-D). operand dtype = the caller's; accumulate f32.
_EIN = lambda A, x: jnp.einsum('br,rck->bck', A, x, preferred_element_type=ACC)
_MAT = lambda A, x: jnp.matmul(A, x, preferred_element_type=ACC)


def dft_matrix(m, dtype=CDTYPE):
    """m x m COMPLEX DFT matrix C[r,c] = W_m^(r*c)  (for the complex iterative/recursive FFT)."""
    k = jnp.arange(m)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m).astype(dtype)


def _CB_reim(m, dt=F):
    """radix-m DFT matrix as (Cr=cos, Ci=sin, Cs=Cr+Ci) in dtype `dt`.
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


def _fft_reim_real(xr, N, half=True, dt=F):
    """Radix-B FFT of REAL columns xr (N, batch), carried as real/imag in dtype `dt`.
    Peels one radix-B stage per iteration (length /B, batch *B), then a leaf DFT, then a
    reshape-only digit-reversal rebuild.  This is THE shared real pass (1D and 2D axis-0)."""
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


def _fft_c(cr, ci, N):
    """General COMPLEX radix-B FFT (Karatsuba on every stage), full N-bin output.
    Used by the Trick-B pallas kernel (half-length complex FFT of the even+i*odd packing)."""
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


def _rev(x):
    """Reverse the rows of x (M, K), M a multiple of 128, in O(M*128) via block matmuls.
    (Mosaic has no rev/flip/gather; this is the VMEM-lowerable row reversal.)"""
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
