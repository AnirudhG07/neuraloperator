"""
Triton kernels for the remaining FNO block ops (K4, K5).

5 compute phases per FNO layer:
  K1 - butterfly DIT FFT       (triton_spectral_conv, log₂N launches per dim)
  K2 - spectral contraction    (ButterflySpectralConv._contract → cuBLAS)
  K3 - butterfly DIF IFFT      (triton_spectral_conv, log₂N launches per dim)
  K4 - fused add + GELU        (_fused_add_gelu_kernel, 1 launch)
  K5 - GroupNorm               (_group_norm_fwd_kernel, 1 launch, Welford)

Public API:
  fused_add_gelu(a, b)                                → gelu(a + b)
  fused_group_norm(x, weight, bias, num_groups, eps)  → GroupNorm(x)
  fused_norm_add_act(x_fft, x_skip, norm, act)        → dispatches K4+K5
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

_GELU_BLOCK = 1024   # tile size for add+gelu kernel
_NORM_BLOCK = 256    # tile size for groupnorm reduction passes


if HAS_TRITON:
    # ── K4: fused elementwise add + GELU ──────────────────────────────────────
    @triton.jit
    def _fused_add_gelu_kernel(
        a_ptr, b_ptr, out_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """out = gelu_tanh(a + b)  elementwise.  Inputs/output: float32."""
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = a + b

        # GELU tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
        c0     = 0.7978845608028654
        c1     = 0.044715
        inner  = c0 * (x + c1 * x * x * x)
        # clamp to avoid exp overflow; tanh saturates at ±1 for |inner|>10
        inner  = tl.where(inner >  20.0,  20.0, inner)
        inner  = tl.where(inner < -20.0, -20.0, inner)
        exp2   = tl.exp(2.0 * inner)
        tanh_v = (exp2 - 1.0) / (exp2 + 1.0)
        result = 0.5 * x * (1.0 + tanh_v)

        tl.store(out_ptr + offs, result, mask=mask)

    # ── K5: GroupNorm forward (Welford 3-pass) ─────────────────────────────────
    @triton.jit
    def _group_norm_fwd_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        C, G: tl.constexpr, spatial, eps,
        BLOCK: tl.constexpr,
    ):
        """GroupNorm for contiguous (B, C, spatial) input reshaped as (B*G, CPG*spatial).

        One program per (batch, group) pair.
        w_ptr / b_ptr: shape (C,) — affine scale and shift per channel.
        """
        row     = tl.program_id(0)
        g       = row % G
        CPG     = C // G        # channels per group (compile-time if G is constexpr)
        row_len = CPG * spatial
        base    = row * row_len

        # ── Pass 1: mean ────────────────────────────────────────────────────────
        total = 0.0
        for off in range(0, row_len, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < row_len
            xv   = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
            total += tl.sum(tl.where(mask, xv, 0.0))
        mean = total / row_len

        # ── Pass 2: variance ────────────────────────────────────────────────────
        var = 0.0
        for off in range(0, row_len, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < row_len
            xv   = tl.load(x_ptr + base + cols, mask=mask, other=mean).to(tl.float32)
            diff = tl.where(mask, xv - mean, 0.0)
            var  += tl.sum(diff * diff)
        rstd = 1.0 / tl.sqrt(var / row_len + eps)

        # ── Pass 3: normalize + affine ──────────────────────────────────────────
        for off in range(0, row_len, BLOCK):
            cols  = off + tl.arange(0, BLOCK)
            mask  = cols < row_len
            xv    = tl.load(x_ptr + base + cols, mask=mask, other=mean).to(tl.float32)

            # global channel index: g * CPG + local_channel_index
            c_idx = g * CPG + cols // spatial
            wv    = tl.load(w_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
            bv    = tl.load(b_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

            y = (xv - mean) * rstd * wv + bv
            tl.store(out_ptr + base + cols, y, mask=mask)


# ── Python wrappers ────────────────────────────────────────────────────────────

def fused_add_gelu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """K4: elementwise gelu(a + b).  Triton on CUDA, PyTorch fallback elsewhere."""
    if not (HAS_TRITON and a.is_cuda and not os.getenv("NO_TRITON")):
        return F.gelu(a + b)

    a   = a.contiguous().float()
    b   = b.contiguous().float()
    out = torch.empty_like(a)
    N   = a.numel()
    grid = (triton.cdiv(N, _GELU_BLOCK),)
    _fused_add_gelu_kernel[grid](a, b, out, N, BLOCK=_GELU_BLOCK)
    return out


def fused_group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """K5: GroupNorm forward.  Triton on CUDA, PyTorch fallback elsewhere."""
    if not (HAS_TRITON and x.is_cuda and not os.getenv("NO_TRITON")):
        return F.group_norm(x, num_groups, weight, bias, eps)

    orig_shape = x.shape
    B, C       = orig_shape[0], orig_shape[1]
    spatial    = x.numel() // (B * C)
    G          = num_groups

    # Reshape to (B, G, CPG*spatial) = (B*G, CPG*spatial)
    x_flat  = x.contiguous().float().view(B * G, -1)
    out_flat = torch.empty_like(x_flat)
    w_flat   = weight.contiguous().float()
    b_flat   = bias.contiguous().float()

    _group_norm_fwd_kernel[(B * G,)](
        x_flat, w_flat, b_flat, out_flat,
        C=C, G=G, spatial=spatial, eps=eps,
        BLOCK=_NORM_BLOCK,
    )
    return out_flat.view(orig_shape).to(x.dtype)


def fused_norm_add_act(
    x_fft:      torch.Tensor,
    x_skip:     torch.Tensor | None,
    norm_module: nn.Module | None,
    activation,
) -> torch.Tensor:
    """Fuse norm(x_fft) + x_skip → activation.

    Triton path when:
      - Triton available + CUDA + NO_TRITON not set
      - norm is nn.GroupNorm or None
      - activation is F.gelu or None

    Falls back to PyTorch for all other combinations.
    """
    use_triton = (
        HAS_TRITON
        and x_fft.is_cuda
        and not os.getenv("NO_TRITON")
        and (norm_module is None or isinstance(norm_module, nn.GroupNorm))
        and (activation is None or activation is F.gelu)
    )

    if not use_triton:
        out = norm_module(x_fft) if norm_module is not None else x_fft
        out = out + x_skip if x_skip is not None else out
        return activation(out) if activation is not None else out

    # ── Triton path ────────────────────────────────────────────────────────────
    # K5: GroupNorm (if present)
    if norm_module is not None:
        x_normed = fused_group_norm(
            x_fft,
            norm_module.weight,
            norm_module.bias,
            norm_module.num_groups,
            norm_module.eps,
        )
    else:
        x_normed = x_fft

    # K4: fused add + GELU
    if x_skip is not None and activation is F.gelu:
        return fused_add_gelu(x_normed, x_skip)
    elif x_skip is not None:
        out = x_normed + x_skip
        return activation(out) if activation is not None else out
    else:
        out = x_normed
        return activation(out) if activation is not None else out
