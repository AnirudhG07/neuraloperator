"""
rfft_variants.py  —  two ways to exploit REAL input, on TWO engines, XLA-inspected.

Strategies (real x: X[N-k]=conj(X[k]); derivation in tpu/math/01_rfft.md):
  Trick A (signal pairing) : pack two real signals a,b -> z=a+i b, ONE full-length complex
                             FFT, unpack.  Halves the NUMBER of transforms.
  Trick B (even/odd pair)  : pack z=x_even+i x_odd, ONE HALF-length FFT, last-layer twist
                             X[k]=E[k]+W_N^k O[k].  Halves the transform LENGTH.

Engines:
  jnpfft   = jnp.fft.fft  — an OPAQUE XLA custom-call; pack/unpack CANNOT fuse into it.
  fft_iter = our radix-B iterative FFT (plain matmuls) — pack/unpack DO fuse in.
"""
import jax.numpy as jnp

try:
    from tpu.fft_iter import fft_iter
except ImportError:  # allow running from inside tpu/
    from fft_iter import fft_iter

C = jnp.complex64

ENGINES = {
    "jnpfft":   lambda z: jnp.fft.fft(z, axis=0),   # opaque custom-call (no fusion)
    "fft_iter": fft_iter,                            # our matmul FFT (fusable)
}


def _mirror(Z):
    """conj(Z[(M-k) mod M]) along axis 0 — the Hermitian partner of each bin."""
    return jnp.conj(jnp.roll(Z[::-1], 1, axis=0))


# ── the three variants, parameterized by engine ─────────────────────────────
def rfft_base(xr, eng):     
    """
    Args:
        xr: real (N, K) array
        eng: engine function (complex FFT along axis 0)
    
    xr real (N, K) -> (N//2+1, K)
    """
    N = xr.shape[0]
    return eng(xr.astype(C))[:N // 2 + 1]


def rfft_trickA(xr, eng):
    """
    Args:
        xr: real (N, K) array
        eng: engine function (complex FFT along axis 0)
    Return:
        (N//2+1, K) array   -- same as base; the concat(axis=1) restores full K,
                               the [:N//2+1] truncates rows (Hermitian). NOT (N, K/2).
    """

    N, K = xr.shape
    a, b = xr[:, :K // 2], xr[:, K // 2:] # pair signals: a,b are (N, K/2)
    Z = eng((a + 1j * b).astype(C)) # ONE FFT on K/2 packed cols -> (N, K/2)
    M = _mirror(Z) # M = conj(Z[(N-k) mod N]) along axis 0
    A = (0.5 * (Z + M))[:N // 2 + 1]  # rfft(a): (N//2+1, K/2)
    Bb = (-0.5j * (Z - M))[:N // 2 + 1] # rfft(b): (N//2+1, K/2)
    return jnp.concatenate([A, Bb], axis=1) # (N//2+1, K)


def rfft_trickB(xr, eng):
    """
    Args:
        xr: real (N, K) array
        eng: engine function (complex FFT along axis 0)
    Return:
        (N//2+1, K) array

    """
    N, K = xr.shape
    h = N // 2
    z = (xr[0::2] + 1j * xr[1::2]).astype(C)         # (h, K)
    Z = eng(z) # K HALF-length FFTs
    M = _mirror(Z) # M = conj(Z[(N-k) mod N]) along axis 0
    E = 0.5 * (Z + M)
    O = -0.5j * (Z - M)
    kf = jnp.arange(h + 1)
    kk = kf % h
    tw = jnp.exp(-2j * jnp.pi * kf / N).astype(C)[:, None]
    return E[kk] + tw * O[kk]     # (N//2+1, K)


_VARIANTS = {"base": rfft_base, "trickA": rfft_trickA, "trickB": rfft_trickB}


# ── full-spectrum (N, K) reconstruction from the half, vs recompute ──────────
def _reconstruct_full(half, N):
    """(N//2+1, K) Hermitian half -> full (N, K) by conjugate-mirroring the upper bins."""
    upper = jnp.conj(half[N // 2 - 1:0:-1])          # bins N/2+1..N-1 = conj(bins N/2-1..1)
    return jnp.concatenate([half, upper], axis=0)    # (N, K)


def full_recompute(xr, eng):
    """FULL (N,K): compute ALL N bins directly (plain complex FFT, no symmetry used)."""
    return eng(xr.astype(C))


def full_via_trickA(xr, eng):
    """FULL (N,K): Trick-A rfft (half work) + conjugate reconstruct."""
    return _reconstruct_full(rfft_trickA(xr, eng), xr.shape[0])


def full_via_trickB(xr, eng):
    """FULL (N,K): Trick-B rfft (half work) + conjugate reconstruct."""
    return _reconstruct_full(rfft_trickB(xr, eng), xr.shape[0])


_FULL = {"recompute": full_recompute,
         "viaTrickA": full_via_trickA,
         "viaTrickB": full_via_trickB}
