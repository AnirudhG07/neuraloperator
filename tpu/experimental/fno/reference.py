"""
The plain-JAX BASELINE FNO2d.  The spectral conv is the textbook version:
  jnp.fft.rfft2  ->  keep the low ±m band (2 corners)  ->  complex channel mix (jnp.einsum / gemm)
  ->  jnp.fft.irfft2.
Uses complex64 throughout — this is the straightforward reference we check `ours` against.
"""
import jax.numpy as jnp

from ...fno.layers import add_grid, channel_mlp, conv1x1, gelu, init_fno


def spectral_conv(w, x, m):
    """2-corner spectral conv via jnp.fft (complex64).  x (B,C,H,W) real -> (B,Cout,H,W) real.
    Keeps k1 ∈ [0,m)∪[H-m,H) on rows (H), k2 ∈ [0,m) on cols (W) — the standard low-mode band."""
    B, C, H, W = x.shape
    x_ft = jnp.fft.rfft2(x, axes=(-2, -1))                      # (B, C, H, W//2+1) complex64
    weight = w["w_re"] + 1j * w["w_im"]                          # (Cin, Cout, 2m, m) complex
    out = jnp.zeros((B, w["w_re"].shape[1], H, W // 2 + 1), x_ft.dtype)

    mix = lambda Wm, Xm: jnp.einsum("ioab,niab->noab", Wm, Xm)   # complex gemm, contract Cin -> Cout
    top = mix(weight[:, :, :m], x_ft[:, :, :m, :m])              # +corner rows [0:m]
    bot = mix(weight[:, :, m:], x_ft[:, :, H - m:, :m])          # -corner rows [H-m:]
    out = out.at[:, :, :m, :m].set(top).at[:, :, H - m:, :m].set(bot)

    return jnp.fft.irfft2(out, s=(H, W), axes=(-2, -1))          # back to a real field


def forward(P, x, m):
    """add (x,y) grid -> LIFT -> [spectral + skip -> GELU] * L -> PROJECT."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        x = gelu(spectral_conv(blk["filter"], x, m) + conv1x1(blk["skip"], x))
    return channel_mlp(P["proj"], x)


init = init_fno   # same weight tree as ours (directly comparable)
