"""
fft2d/jax_fft.py — 2D pure-JAX real FFT.

  jax_rfft2(xr)             : FULL rfft2, reusing the shared 1D pass. (N1,N2,K)->(N1//2,N2,K).
  rfft2_partial(xr, ratio)  : FNO transform — keep only m=round(N*ratio) low modes per axis via
                              a PARTIAL DFT (skips the full FFT). (N1,N2,K)->(m1,m2,K).
"""
import jax.numpy as jnp

try:
    from tpu.fft_core import ACC, F, _CB_reim, _fft_reim_real, _karatsuba
except ImportError:
    from fft_core import ACC, F, _CB_reim, _fft_reim_real, _karatsuba


def _dft_axis1(Br, Bi, m, dt=F):
    """Complex DFT of length m along AXIS-1 of (R,m,K): a direct m x m matmul contracting the
    middle axis (no transpose). Reuses _CB_reim + _karatsuba."""
    Cr, Ci, Cs = _CB_reim(m, dt) # twiddle number multiple
    mm = lambda A, x: jnp.einsum('on,rnk->rok', A, x, preferred_element_type=ACC)
    return _karatsuba(mm, Cr, Ci, Cs, Br, Bi)


def jax_rfft2(xr, dt=F):
    """FULL 2D rfft. axis-0 (N1): staged real rfft (_fft_reim_real). axis-1 (N2): direct complex
    DFT contracting the middle axis. No transpose. (N1,N2,K)->(yr,yi), each (N1//2,N2,K)."""
    N1, N2, K = xr.shape
    Ar, Ai = _fft_reim_real(xr.reshape(N1, N2 * K), N1, half=True, dt=dt)
    R = Ar.shape[0] # N1//2
    Ar, Ai = Ar.reshape(R, N2, K), Ai.reshape(R, N2, K)
    return _dft_axis1(Ar, Ai, N2, dt=dt)


def rfft2_partial(xr, ratio=0.5, dt=F):
    """FNO 2D rfft: keep only m=round(N*ratio) LOW modes per axis via a partial DFT (no full FFT,
    no twiddle). axis-0 real -> 2 matmuls; axis-1 complex Karatsuba -> 3; contract-in-place, no
    transpose. Real (N1,N2,K) -> (yr,yi), each (m1,m2,K). (For the FNO's +/- band on axis-1,
    stack the top m2 rows of C2 too — one extra skinny matmul.)"""
    N1, N2, K = xr.shape
    m1, m2 = max(1, int(N1 * ratio)), max(1, int(N2 * ratio))
    # axis-0: partial real DFT (m1 x N1), real input -> 2 matmuls
    a1 = -2 * jnp.pi * jnp.outer(jnp.arange(m1), jnp.arange(N1)) / N1
    Cr1, Ci1 = jnp.cos(a1).astype(dt), jnp.sin(a1).astype(dt)
    x2 = xr.reshape(N1, N2 * K).astype(dt)
    Ar = jnp.matmul(Cr1, x2, preferred_element_type=ACC).reshape(m1, N2, K)
    Ai = jnp.matmul(Ci1, x2, preferred_element_type=ACC).reshape(m1, N2, K)
    # axis-1: partial complex DFT (m2 x N2), Karatsuba, contract the middle axis
    a2 = -2 * jnp.pi * jnp.outer(jnp.arange(m2), jnp.arange(N2)) / N2
    Cr2, Ci2 = jnp.cos(a2).astype(dt), jnp.sin(a2).astype(dt)
    mm = lambda A, x: jnp.einsum('on,rnk->rok', A, x, preferred_element_type=ACC)
    return _karatsuba(mm, Cr2, Ci2, (Cr2 + Ci2).astype(dt), Ar, Ai)  # (m1, m2, K)
