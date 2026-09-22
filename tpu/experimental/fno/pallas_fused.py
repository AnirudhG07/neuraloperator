"""
FUSED FNO block (fft + GEMM + ifft + skip + GELU) as ONE Pallas
TPU kernel.  The whole 2-corner spectral conv AND the skip conv1x1 AND the GELU stay VMEM-resident:
one HBM read (input field) + one HBM write (output field), weights read once.  TurboFNO on v5e.

Movement controlled by BlockSpec + grid (Mosaic does async DMA / double-buffering — no manual copy):
  grid = (B // B_TILE,)                             one step per batch-tile of B_TILE samples
  in   : field (B_TILE,Cin,H,W) -> VMEM  |  spectral W (Cin,Cout,2m,m), skip W/b, DFT matrices ->
         loaded once (constant index_map, resident)
  out  : field (B_TILE,Cout,H,W) -> HBM

Three optimizations vs the naive fused kernel:
  (1) bf16 matmul operands (mdt), f32 accumulate — halves field+weight HBM, pushes the resolution
      ceiling toward 256².  Storage dtype of x/weights = mdt.
  (2) skip conv1x1 + bias + GELU fused into the same kernel — kills another full-res HBM round-trip;
      the kernel now computes the WHOLE FNO block, not just the spectral conv.
  (3) Karatsuba/Gauss complex matmul (3 real matmuls, not 4) on the two complex DFT stages and the
      channel mix; DFT matrices HOISTED out of the kernel (passed as inputs) so zero runtime trig.

Validate OFFLINE (no TPU): interpret=True vs the plain-JAX block reference.  mdt=F -> ~1e-6 (exact);
mdt=bf16 -> ~1e-2 (expected, FNO-tolerant).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...fft.core import ACC, BF, F
from ...fno.layers import add_grid, channel_mlp, conv1x1
from ...fno.spectral import spectral_conv as spectral_2corner

gelu = jax.nn.gelu                                                    # default = tanh approx (Mosaic-safe)


def _params(vmem_mb=128):
    """dimension_semantics=('parallel',) marks the batch-tile grid as independent -> Mosaic emits the
    DOUBLE-BUFFERED pipeline (prefetch tile b+1 while computing b) automatically; that IS the double-
    buffer switch.  vmem_limit_bytes raises the 16 MB scoped ceiling (confirm vs physical VMEM)."""
    return pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=vmem_mb * 1024 * 1024)


def _rows(k_idx, L, sign):
    """DFT rows for explicit mode indices (built ONCE in the wrapper, passed in as constants)."""
    k = jnp.asarray(k_idx, F)[:, None]; n = jnp.arange(L, dtype=F)[None, :]
    a = sign * 2.0 * jnp.pi * k * n / L
    return jnp.cos(a), jnp.sin(a)


def _dft_mats(H, W, m):
    """All partial-DFT matrices for one block, precomputed (no in-kernel trig)."""
    k1 = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])       # ± band, 2m rows
    Cr0, Ci0 = _rows(k1, H, -1)                                       # fwd axis-0 (2m,H)
    Cr1, Ci1 = _rows(jnp.arange(m), W, -1)                            # fwd axis-1 (m,W)
    d = jnp.arange(m); alpha = jnp.where(d == 0, 1.0, 2.0).astype(F)  # Hermitian fold (DC once)
    IwR, IwI = _rows(d, W, +1)
    IwR, IwI = (IwR * alpha[:, None]).T, (IwI * alpha[:, None]).T     # inv axis-1 (W,m)
    IhR, IhI = _rows(k1, H, +1)
    IhR, IhI = IhR.T, IhI.T                                           # inv axis-0 (H,2m)
    return [Cr0, Ci0, Cr1, Ci1, IwR, IwI, IhR, IhI]


# ═══════════════════════════════════════════════════════════════════════════
#  kernel: one field tile -> full FNO block (spectral + skip + GELU)
# ═══════════════════════════════════════════════════════════════════════════
def _kernel(x_ref, wre_ref, wim_ref, sW_ref, sb_ref,
            Cr0_ref, Ci0_ref, Cr1_ref, Ci1_ref, IwR_ref, IwI_ref, IhR_ref, IhI_ref,
            out_ref, *, m, H, W, mdt, dt):
    x = x_ref[...]                                                    # (n,c,h,w)
    E = lambda ein, A, Z: jnp.einsum(ein, A.astype(mdt), Z.astype(mdt), preferred_element_type=ACC)

    def cmul(ein, Ar, Ai, Br, Bi):                                   # (Ar+iAi)(Br+iBi), 3 matmuls
        p, q = E(ein, Ar, Br), E(ein, Ai, Bi)
        s = E(ein, Ar + Ai, Br + Bi)
        return p - q, s - p - q                                      # (real, imag)

    # fwd DFT axis-0 (contract h) — real input -> 2 matmuls
    Ar = E("ah,nchw->ncaw", Cr0_ref[...], x); Ai = E("ah,nchw->ncaw", Ci0_ref[...], x)
    # fwd DFT axis-1 (contract w, low m) — complex -> Karatsuba
    Xr, Xi = cmul("dw,ncaw->ncad", Cr1_ref[...], Ci1_ref[...], Ar, Ai)
    # channel-mix GEMM (complex weight) -> Karatsuba
    Or_, Oi = cmul("ioad,niad->noad", wre_ref[...], wim_ref[...], Xr, Xi)
    # inv DFT axis-1 (d->w, Hermitian-folded matrices) — complex -> Karatsuba
    Pr, Pi = cmul("wd,noad->noaw", IwR_ref[...], IwI_ref[...], Or_, Oi)
    # inv DFT axis-0 (a->h) — REAL part only -> 2 matmuls
    u = E("ha,noaw->nohw", IhR_ref[...], Pr) - E("ha,noaw->nohw", IhI_ref[...], Pi)
    u = u / (H * W)
    # fused skip conv1x1 + bias, then residual add + GELU  (whole block on-chip)
    skip = E("oi,nihw->nohw", sW_ref[...], x) + sb_ref[...][None, :, None, None]
    out_ref[...] = gelu(u + skip).astype(dt)


def fused_block(x, w_re, w_im, skip_W, skip_b, m, B_TILE=1, vmem_mb=128, mdt=F, interpret=True, dt=F):
    """x (B,Cin,H,W) real -> full FNO block gelu(spectral(x)+skip(x)), (B,Cout,H,W).
    mdt = matmul-operand dtype (F exact / BF fast).  Weights stored in mdt (bf16 halves their HBM)."""
    B, Cin, H, W = x.shape; Cout = w_re.shape[1]
    mats = _dft_mats(H, W, m)                                         # 8 small DFT matrices (f32)
    _z = lambda a: pl.BlockSpec(a.shape, lambda b: tuple(0 for _ in a.shape))   # constant, resident
    in_specs = [pl.BlockSpec((B_TILE, Cin, H, W), lambda b: (b, 0, 0, 0)),
                _z(w_re), _z(w_im), _z(skip_W), _z(skip_b), *[_z(a) for a in mats]]
    args = (x.astype(mdt), w_re.astype(mdt), w_im.astype(mdt), skip_W.astype(mdt), skip_b.astype(F),
            *mats)
    return pl.pallas_call(
        partial(_kernel, m=m, H=H, W=W, mdt=mdt, dt=dt),
        grid=(B // B_TILE,),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((B_TILE, Cout, H, W), lambda b: (b, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, Cout, H, W), dt),
        compiler_params=_params(vmem_mb),
        interpret=interpret,
    )(*args)


# ── reference: the exact same FNO block in plain JAX (validate against this) ──
def block_ref(x, w, skip, m):
    return gelu(spectral_2corner(w, x, m) + conv1x1(skip, x))


# ═══════════════════════════════════════════════════════════════════════════
#  pure-array building blocks (shared by levels B and C — no refs, so callable in a loop)
# ═══════════════════════════════════════════════════════════════════════════
def _E(ein, A, Z, mdt):
    return jnp.einsum(ein, A.astype(mdt), Z.astype(mdt), preferred_element_type=ACC)


def _block_compute(x, wre, wim, sW, sb, mats, m, H, W, mdt):
    """One FNO block gelu(spectral(x)+skip(x)) on plain arrays (Karatsuba 3-mul complex products)."""
    Cr0, Ci0, Cr1, Ci1, IwR, IwI, IhR, IhI = mats

    def cmul(ein, Ar, Ai, Br, Bi):
        p, q = _E(ein, Ar, Br, mdt), _E(ein, Ai, Bi, mdt)
        s = _E(ein, Ar + Ai, Br + Bi, mdt)
        return p - q, s - p - q

    Ar = _E("ah,nchw->ncaw", Cr0, x, mdt); Ai = _E("ah,nchw->ncaw", Ci0, x, mdt)
    Xr, Xi = cmul("dw,ncaw->ncad", Cr1, Ci1, Ar, Ai)
    Or_, Oi = cmul("ioad,niad->noad", wre, wim, Xr, Xi)
    Pr, Pi = cmul("wd,noad->noaw", IwR, IwI, Or_, Oi)
    u = _E("ha,noaw->nohw", IhR, Pr, mdt) - _E("ha,noaw->nohw", IhI, Pi, mdt)
    skip = _E("oi,nihw->nohw", sW, x, mdt) + sb[None, :, None, None]
    return gelu(u / (H * W) + skip)


def _cmlp(x, l1W, l1b, l2W, l2b, mdt):
    """Pointwise 2-layer MLP (conv1x1 -> GELU -> conv1x1) — lift & project, on plain arrays."""
    h = gelu(_E("oi,nihw->nohw", l1W, x, mdt) + l1b[None, :, None, None])
    return _E("oi,nihw->nohw", l2W, h, mdt) + l2b[None, :, None, None]


# ═══════════════════════════════════════════════════════════════════════════
#  Level B: ALL blocks fused in one kernel — field stays in VMEM across layers
#  (1 HBM read + 1 write total; no inter-layer round-trip).  lift/project stay in JAX.
# ═══════════════════════════════════════════════════════════════════════════
def _kernel_B(x_ref, wre_ref, wim_ref, sW_ref, sb_ref,
              a0, b0, a1, b1, iwr, iwi, ihr, ihi, out_ref, *, L, m, H, W, mdt, dt):
    mats = (a0[...], b0[...], a1[...], b1[...], iwr[...], iwi[...], ihr[...], ihi[...])
    x = x_ref[...]
    for l in range(L):                                                # unrolled; field never leaves VMEM
        x = _block_compute(x, wre_ref[l], wim_ref[l], sW_ref[l], sb_ref[l], mats, m, H, W, mdt)
    out_ref[...] = x.astype(dt)


def fused_all_blocks(x, wre, wim, sW, sb, m, B_TILE=1, vmem_mb=128, mdt=F, interpret=True, dt=F):
    """x (B,width,H,W) lifted field, stacked weights wre/wim (L,width,width,2m,m), sW (L,width,width),
    sb (L,width) -> field after L blocks, all fused in one kernel."""
    B, width, H, W = x.shape; L = wre.shape[0]
    mats = _dft_mats(H, W, m)
    _z = lambda a: pl.BlockSpec(a.shape, lambda b: tuple(0 for _ in a.shape))
    in_specs = [pl.BlockSpec((B_TILE, width, H, W), lambda b: (b, 0, 0, 0)),
                _z(wre), _z(wim), _z(sW), _z(sb), *[_z(a) for a in mats]]
    args = (x.astype(mdt), wre.astype(mdt), wim.astype(mdt), sW.astype(mdt), sb.astype(F), *mats)
    return pl.pallas_call(
        partial(_kernel_B, L=L, m=m, H=H, W=W, mdt=mdt, dt=dt),
        grid=(B // B_TILE,), in_specs=in_specs,
        out_specs=pl.BlockSpec((B_TILE, width, H, W), lambda b: (b, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, width, H, W), dt),
        compiler_params=_params(vmem_mb), interpret=interpret,
    )(*args)


# ═══════════════════════════════════════════════════════════════════════════
#  Level C: lift + all blocks + project fused — nothing but add_grid stays in JAX
# ═══════════════════════════════════════════════════════════════════════════
def _kernel_C(xg_ref, l1W, l1b, l2W, l2b, wre_ref, wim_ref, sW_ref, sb_ref,
              p1W, p1b, p2W, p2b, a0, b0, a1, b1, iwr, iwi, ihr, ihi, out_ref, *, L, m, H, W, mdt, dt):
    mats = (a0[...], b0[...], a1[...], b1[...], iwr[...], iwi[...], ihr[...], ihi[...])
    x = _cmlp(xg_ref[...], l1W[...], l1b[...], l2W[...], l2b[...], mdt)      # lift
    for l in range(L):
        x = _block_compute(x, wre_ref[l], wim_ref[l], sW_ref[l], sb_ref[l], mats, m, H, W, mdt)
    out_ref[...] = _cmlp(x, p1W[...], p1b[...], p2W[...], p2b[...], mdt).astype(dt)   # project


def fused_full(xg, lift, wre, wim, sW, sb, proj, m, B_TILE=1, vmem_mb=128, mdt=F, interpret=True, dt=F,
               out_ch=1):
    """xg (B,in_ch+2,H,W) grid-augmented field -> (B,out_ch,H,W).  lift/proj = {'l1','l2'} MLP dicts."""
    B, Cin, H, W = xg.shape
    mats = _dft_mats(H, W, m)
    _z = lambda a: pl.BlockSpec(a.shape, lambda b: tuple(0 for _ in a.shape))
    L1W, L1b, L2W, L2b = lift["l1"]["W"], lift["l1"]["b"], lift["l2"]["W"], lift["l2"]["b"]
    P1W, P1b, P2W, P2b = proj["l1"]["W"], proj["l1"]["b"], proj["l2"]["W"], proj["l2"]["b"]
    extras = [L1W, L1b, L2W, L2b, wre, wim, sW, sb, P1W, P1b, P2W, P2b, *mats]
    in_specs = [pl.BlockSpec((B_TILE, Cin, H, W), lambda b: (b, 0, 0, 0)), *[_z(a) for a in extras]]
    cast = lambda a: a.astype(mdt)
    args = (cast(xg), cast(L1W), L1b.astype(F), cast(L2W), L2b.astype(F),
            cast(wre), cast(wim), cast(sW), sb.astype(F),
            cast(P1W), P1b.astype(F), cast(P2W), P2b.astype(F), *mats)
    return pl.pallas_call(
        partial(_kernel_C, L=wre.shape[0], m=m, H=H, W=W, mdt=mdt, dt=dt),
        grid=(B // B_TILE,), in_specs=in_specs,
        out_specs=pl.BlockSpec((B_TILE, out_ch, H, W), lambda b: (b, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, out_ch, H, W), dt),
        compiler_params=_params(vmem_mb), interpret=interpret,
    )(*args)


# ═══════════════════════════════════════════════════════════════════════════
#  Whole-FNO forwards at each fusion level (same math as fno.model.forward)
# ═══════════════════════════════════════════════════════════════════════════
def _stack_blocks(P):
    bl = P["blocks"]
    wre = jnp.stack([b["filter"]["w_re"] for b in bl]); wim = jnp.stack([b["filter"]["w_im"] for b in bl])
    sW = jnp.stack([b["skip"]["W"] for b in bl]); sb = jnp.stack([b["skip"]["b"] for b in bl])
    return wre, wim, sW, sb


def _pad_grid_lift(P, x, pad):
    if pad:
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, pad)))
    return add_grid(x)


def forward_A(P, x, m, pad, mdt=F, interpret=False):
    """Level A: per-block Pallas kernel, JAX loops the layers (inter-layer HBM round-trip)."""
    xg = _pad_grid_lift(P, x, pad); h = channel_mlp(P["lift"], xg)
    for b in P["blocks"]:
        f = b["filter"]
        h = fused_block(h, f["w_re"], f["w_im"], b["skip"]["W"], b["skip"]["b"], m, mdt=mdt,
                        interpret=interpret)
    out = channel_mlp(P["proj"], h)
    return out[..., :-pad, :-pad] if pad else out


def forward_B(P, x, m, pad, mdt=F, interpret=False):
    """Level B: all blocks in one kernel (no inter-layer HBM); lift/project in JAX."""
    xg = _pad_grid_lift(P, x, pad); h = channel_mlp(P["lift"], xg)
    wre, wim, sW, sb = _stack_blocks(P)
    h = fused_all_blocks(h, wre, wim, sW, sb, m, mdt=mdt, interpret=interpret)
    out = channel_mlp(P["proj"], h)
    return out[..., :-pad, :-pad] if pad else out


def forward_C(P, x, m, pad, mdt=F, interpret=False, out_ch=1):
    """Level C: lift + blocks + project all in one kernel (only add_grid stays in JAX)."""
    xg = _pad_grid_lift(P, x, pad)
    wre, wim, sW, sb = _stack_blocks(P)
    out = fused_full(xg, P["lift"], wre, wim, sW, sb, P["proj"], m, mdt=mdt, interpret=interpret,
                     out_ch=out_ch)
    return out[..., :-pad, :-pad] if pad else out
