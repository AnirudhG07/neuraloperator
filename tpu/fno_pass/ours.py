"""
fno_pass/ours.py — OUR FNO2d.  Same 2-corner spectral conv as reference.py, but done ENTIRELY in
real/imag: the FFT is cos/sin matmuls, and it computes ONLY the kept low modes as a direct partial
DFT (no full FFT, no complex64).  Same weight tree + modes as reference -> outputs should match.
"""
import jax.numpy as jnp

try:
    from tpu.fft_core import ACC, F
    from tpu.fno_pass.layers import add_grid, channel_mlp, conv1x1, gelu, init_fno
except ImportError:
    from fft_core import ACC, F
    from fno_pass.layers import add_grid, channel_mlp, conv1x1, gelu, init_fno

_E = lambda s, M, Z: jnp.einsum(s, M, Z, preferred_element_type=ACC)


def _rows(k_idx, L, sign):
    """(cos, sin) rows of a length-L DFT for the mode indices k_idx.  sign=-1 forward, +1 inverse."""
    k = jnp.asarray(k_idx, F)[:, None]
    n = jnp.arange(L, dtype=F)[None, :]
    a = sign * 2.0 * jnp.pi * k * n / L
    return jnp.cos(a), jnp.sin(a)                    # (len(k_idx), L)


def spectral_conv(w, x, m):
    """2-corner spectral conv, real/imag.  x (B,C,H,W) real -> (B,Cout,H,W) real.
    H = rows (x.shape[-2]), W = cols (x.shape[-1]).  einsum letters: n=batch c=chan h=row w=col
    a=row-mode band (2m) d=col modes (m) i/o=in/out channels."""
    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)

    # forward FFT2 — keep only the low modes
    h_band = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])   # ±m band on rows (2m indices)

    # FFT along rows (contract H); real input -> 2 matmuls -> (n, c, 2m, W)
    cos_h, sin_h = _rows(h_band, H, -1)
    xh_re = _E("ah,nchw->ncaw", cos_h, x)
    xh_im = _E("ah,nchw->ncaw", sin_h, x)

    # FFT along cols (contract W, keep low m); complex -> (n, c, 2m, m)
    cos_w, sin_w = _rows(jnp.arange(m), W, -1)
    fft_w = lambda cs, z: _E("dw,ncaw->ncad", cs, z)
    spec_re = fft_w(cos_w, xh_re) - fft_w(sin_w, xh_im)
    spec_im = fft_w(cos_w, xh_im) + fft_w(sin_w, xh_re)

    # learned complex channel mix — weights (Cin, Cout, 2m, m)
    mix = lambda wt, z: _E("ioad,niad->noad", wt, z)
    mixed_re = mix(w["w_re"], spec_re) - mix(w["w_im"], spec_im)
    mixed_im = mix(w["w_re"], spec_im) + mix(w["w_im"], spec_re)

    # iFFT along cols with Hermitian fold (DC once, others twice — the −k conjugate of the rfft'd axis)
    col_modes = jnp.arange(m)
    fold = jnp.where(col_modes == 0, 1.0, 2.0).astype(F)
    icos_w, isin_w = _rows(col_modes, W, +1)
    icos_w, isin_w = (icos_w * fold[:, None]).T, (isin_w * fold[:, None]).T   # (W, m)
    ifft_w = lambda cs, z: _E("wd,noad->noaw", cs, z)
    recon_re = ifft_w(icos_w, mixed_re) - ifft_w(isin_w, mixed_im)
    recon_im = ifft_w(icos_w, mixed_im) + ifft_w(isin_w, mixed_re)

    # iFFT along rows over the ±band, take the REAL part
    icos_h, isin_h = _rows(h_band, H, +1)            # (2m, H) -> used as (H, 2m)
    ifft_h = lambda cs, z: _E("ha,noaw->nohw", cs.T, z)
    field = ifft_h(icos_h, recon_re) - ifft_h(isin_h, recon_im)

    return (field / (H * W)).astype(F)               # normalize (1/HW)


def forward(P, x, m):
    """add (x,y) grid -> LIFT -> [spectral + skip -> GELU] * L -> PROJECT."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        x = gelu(spectral_conv(blk["filter"], x, m) + conv1x1(blk["skip"], x))
    return channel_mlp(P["proj"], x)


init = init_fno   # same weight tree as reference (directly comparable)
