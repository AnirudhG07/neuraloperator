"""
fft2d/jax_fft.py — 2D pure-JAX real FFT.

  jax_rfft2(xr)             : FULL rfft2, reusing the shared 1D pass. (N1,N2,K)->(N1//2,N2,K).
  rfft2_partial(xr, ratio)  : FNO transform — keep only m=round(N*ratio) low modes per axis via
                              a PARTIAL DFT (skips the full FFT). (N1,N2,K)->(m1,m2,K).
"""
import jax.numpy as jnp

try:
    from tpu.fft_core import ACC, F, _CB_reim, _fft_c, _fft_reim_real, _karatsuba
    from tpu.fft1d.jax_fft import rfft_trickA_reim, rfft_trickB_reim
except ImportError:
    from fft_core import ACC, F, _CB_reim, _fft_c, _fft_reim_real, _karatsuba
    from fft1d.jax_fft import rfft_trickA_reim, rfft_trickB_reim


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


def jax_rfft2_fast(xr, dt=F):
    """FULL 2D rfft, but axis-1 uses a STAGED radix-B FFT (_fft_c, O(N2 log N2)) instead of the
    direct O(N2^2) DFT in jax_rfft2.  axis-0 staged real rfft as before; axis-1 needs the axis on
    the front, so transpose (R,N2,K)->(N2,R*K), stage, transpose back.  (N1,N2,K)->(yr,yi), each
    (N1//2, N2, K).  Should win at large N2 where the quadratic axis-1 dominates the transpose cost."""
    N1, N2, K = xr.shape
    Ar, Ai = _fft_reim_real(xr.reshape(N1, N2 * K), N1, half=True, dt=dt)   # axis-0 staged rfft
    R = Ar.shape[0]
    Ar = Ar.reshape(R, N2, K).transpose(1, 0, 2).reshape(N2, R * K)         # N2 -> front
    Ai = Ai.reshape(R, N2, K).transpose(1, 0, 2).reshape(N2, R * K)
    Yr, Yi = _fft_c(Ar.astype(dt), Ai.astype(dt), N2)                       # axis-1 staged FFT
    Yr = Yr.reshape(N2, R, K).transpose(1, 0, 2)                            # back to (R, N2, K)
    Yi = Yi.reshape(N2, R, K).transpose(1, 0, 2)
    return Yr, Yi


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


def rfft2_hybrid(xr, m1, m2, axis0="reim", rows="partial", dt=F):
    """HYBRID low-mode 2D rfft: FULL real FFT on axis-0 (all cols) then keep the low m1; then rows:
      rows='partial' -> PARTIAL direct DFT on axis-1 (skinny (m2 x N2), only the low m2) [the winner]
      rows='staged'  -> full STAGED FFT on axis-1 (transpose, _fft_c) then TRUNCATE to m2 (computes
                        all modes then slices — the 'truncated reim rows' variant to test)
    axis0='reim' (staged rfft) or 'trickA'.  (N1,N2,K) -> (yr,yi), each (m1,m2,K)."""
    N1, N2, K = xr.shape
    x2 = xr.reshape(N1, N2 * K)
    if axis0 == "trickA":
        Ar, Ai = rfft_trickA_reim(x2, m1 / N1, dt)               # full real rfft (trick A), m1 modes
    elif axis0 == "trickB":
        Ar, Ai = rfft_trickB_reim(x2, m1 / N1, dt)               # full real rfft (trick B), m1 modes
    else:
        Ar, Ai = _fft_reim_real(x2, N1, half=True, dt=dt)        # full staged real rfft
    Ar, Ai = Ar[:m1].reshape(m1, N2, K), Ai[:m1].reshape(m1, N2, K)   # keep low m1 col-modes
    if rows == "staged":                                          # staged FFT on axis-1 + truncate
        Ar2 = Ar.transpose(1, 0, 2).reshape(N2, m1 * K)          # N2 -> front (needs transpose)
        Ai2 = Ai.transpose(1, 0, 2).reshape(N2, m1 * K)
        Yr, Yi = _fft_c(Ar2.astype(dt), Ai2.astype(dt), N2)      # full complex FFT (all N2 modes)
        Yr = Yr.reshape(N2, m1, K).transpose(1, 0, 2)[:, :m2]    # back, TRUNCATE to low m2
        Yi = Yi.reshape(N2, m1, K).transpose(1, 0, 2)[:, :m2]
        return Yr, Yi                                            # (m1, m2, K)
    a2 = -2 * jnp.pi * jnp.outer(jnp.arange(m2), jnp.arange(N2)) / N2  # axis-1 PARTIAL DFT (m2 x N2)
    Cr2, Ci2 = jnp.cos(a2).astype(dt), jnp.sin(a2).astype(dt)
    mm = lambda A, x: jnp.einsum('on,rnk->rok', A, x, preferred_element_type=ACC)
    return _karatsuba(mm, Cr2, Ci2, (Cr2 + Ci2).astype(dt), Ar, Ai)  # (m1, m2, K)
