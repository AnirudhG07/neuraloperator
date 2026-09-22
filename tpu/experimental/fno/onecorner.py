"""
The 1-CORNER spectral conv: keeps only the [0,m) x [0,m) mode block, i.e. one low-frequency
corner instead of the +/- band.  That is HALF the modes of the standard FNO, which is what
caused the Navier-Stokes rel-L2 of 0.222 before it was fixed (results/REPORT.md §3).  Kept as
the record of that result; the correct 2-corner version is fno/spectral.py.
"""
import jax.numpy as jnp

from ...fft.core import ACC, F


def _dft(m, L, sign):
    """(cos, sin) rows of a length-L DFT kept to m modes.  sign=-1 forward, +1 inverse."""
    k = jnp.arange(m)[:, None]
    n = jnp.arange(L)[None, :]
    a = sign * 2.0 * jnp.pi * k * n / L
    return jnp.cos(a).astype(F), jnp.sin(a).astype(F)      # each (m, L)


def fft2_lowmodes(x, m):
    """Forward 2D DFT of a REAL field, keeping the m×m low-frequency block.  Real in -> (re, im).
    x (B, C, H, W) -> (fr, fi) each (B, C, m, m)."""
    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    ChR, ChI = _dft(m, H, -1)                             # DFT along H (real input -> 2 matmuls)
    Ar = jnp.einsum('kh,bchw->bckw', ChR, x, preferred_element_type=ACC)
    Ai = jnp.einsum('kh,bchw->bckw', ChI, x, preferred_element_type=ACC)
    CwR, CwI = _dft(m, W, -1)                             # DFT along W (complex now)
    e = lambda M, Z: jnp.einsum('jw,bckw->bckj', M, Z, preferred_element_type=ACC)
    return e(CwR, Ar) - e(CwI, Ai), e(CwR, Ai) + e(CwI, Ar)


def spectral_filter(w, fr, fi):
    """Per-mode complex channel mix: (Wr+iWi)·(fr+ifi).  w = {w_re, w_im} (Cin, Cout, m, m).
    (B, Cin, m, m) -> (B, Cout, m, m)."""
    mix = lambda M, Z: jnp.einsum('iokl,bikl->bokl', M, Z, preferred_element_type=ACC)
    return (mix(w["w_re"], fr) - mix(w["w_im"], fi),      # real part
            mix(w["w_re"], fi) + mix(w["w_im"], fr))      # imag part


def ifft2_real(gr, gi, H, W):
    """Inverse 2D DFT back to a REAL field: u = Re(iFFT2), computed directly in real/imag.
    (gr, gi) each (B, C, m, m) -> u (B, C, H, W) real."""
    m = gr.shape[-1]
    IwR, IwI = _dft(m, W, +1)                             # (m, W); use as (W, m) below
    ew = lambda M, Z: jnp.einsum('jw,bckj->bckw', M, Z, preferred_element_type=ACC)
    Pr = ew(IwR, gr) - ew(IwI, gi)                        # inverse along W
    Pi = ew(IwR, gi) + ew(IwI, gr)
    IhR, IhI = _dft(m, H, +1)                             # (m, H)
    eh = lambda M, Z: jnp.einsum('kh,bckw->bchw', M, Z, preferred_element_type=ACC)
    u = eh(IhR, Pr) - eh(IhI, Pi)                         # inverse along H, keep REAL part
    return (u / (H * W)).astype(F)
