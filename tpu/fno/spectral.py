import jax.numpy as jnp

from ..fft.core import ACC, F

# einsum letters: n=sample c/i=in-channel o=out-channel h=row w=col
#                 a=row mode (the 2m band)  d=col mode (the low m)
_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)


def _dft_rows(mode_idx, length, sign):
    """(cos, sin) rows of a length-`length` DFT restricted to the mode indices `mode_idx`.
    sign=-1 builds the forward transform, +1 the inverse.  Each is (len(mode_idx), length)."""
    k = jnp.asarray(mode_idx, F)[:, None]
    n = jnp.arange(length, dtype=F)[None, :]
    angle = sign * 2.0 * jnp.pi * k * n / length
    return jnp.cos(angle), jnp.sin(angle)


def spectral_conv(weight, x, n_modes):
    """One FNO spectral convolution, done entirely in real/imag — no complex64 anywhere.

    Four steps, each a matmul over one axis:
      1. transform the rows, keeping only the low +/- band  (real input, so 2 matmuls)
      2. transform the cols, keeping only the low modes     (complex now, so 4)
      3. mix channels independently at every kept mode      (the learned weight)
      4. transform both axes back and take the real part

    Steps 1-2 compute ONLY the modes that survive truncation, so no full FFT ever happens, and
    step 4 reconstructs the full field straight from them — the un-kept modes are implicitly zero,
    with no padding tensor.  `weight` is {w_re, w_im} of shape (in_ch, out_ch, 2*n_modes, n_modes).
    x (B, in_ch, H, W) real -> (B, out_ch, H, W) real."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    x = x.astype(F)

    # 1. rows -> the +/- band k in [0, n_modes) U [H - n_modes, H), i.e. both low-frequency corners
    row_band = jnp.concatenate([jnp.arange(n_modes), jnp.arange(n_rows - n_modes, n_rows)])
    row_cos, row_sin = _dft_rows(row_band, n_rows, -1)
    rows_re = _E("ah,nchw->ncaw", row_cos, x)
    rows_im = _E("ah,nchw->ncaw", row_sin, x)

    # 2. cols -> the low n_modes only (this axis is rfft'd, so negative cols are redundant)
    col_cos, col_sin = _dft_rows(jnp.arange(n_modes), n_cols, -1)
    over_cols = lambda mat, z: _E("dw,ncaw->ncad", mat, z)
    modes_re = over_cols(col_cos, rows_re) - over_cols(col_sin, rows_im)
    modes_im = over_cols(col_cos, rows_im) + over_cols(col_sin, rows_re)

    # 3. learned complex channel mix, per kept mode
    mix = lambda w, z: _E("ioad,niad->noad", w, z)
    mixed_re = mix(weight["w_re"], modes_re) - mix(weight["w_im"], modes_im)
    mixed_im = mix(weight["w_re"], modes_im) + mix(weight["w_im"], modes_re)

    # 4a. cols back, folding in the Hermitian partners the rfft'd axis left out
    #     (DC counts once, every other mode twice — that is its -k conjugate)
    col_modes = jnp.arange(n_modes)
    hermitian_fold = jnp.where(col_modes == 0, 1.0, 2.0).astype(F)
    inv_col_cos, inv_col_sin = _dft_rows(col_modes, n_cols, +1)
    inv_col_cos = (inv_col_cos * hermitian_fold[:, None]).T          # (n_cols, n_modes)
    inv_col_sin = (inv_col_sin * hermitian_fold[:, None]).T
    back_over_cols = lambda mat, z: _E("wd,noad->noaw", mat, z)
    cols_back_re = back_over_cols(inv_col_cos, mixed_re) - back_over_cols(inv_col_sin, mixed_im)
    cols_back_im = back_over_cols(inv_col_cos, mixed_im) + back_over_cols(inv_col_sin, mixed_re)

    # 4b. rows back over the same band; the field is real, so only the real part survives
    inv_row_cos, inv_row_sin = _dft_rows(row_band, n_rows, +1)
    back_over_rows = lambda mat, z: _E("ha,noaw->nohw", mat.T, z)
    field = back_over_rows(inv_row_cos, cols_back_re) - back_over_rows(inv_row_sin, cols_back_im)

    return (field / (n_rows * n_cols)).astype(F)
