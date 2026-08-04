"""Triton JIT butterfly FFT kernels.

Called by butterfly_fft_1d / butterfly_ifft_1d in butterfly_fft.py when
HAS_TRITON=True and the input tensor is on CUDA.  This file does NOT import
from butterfly_fft.py, so there is no circular dependency.

Kernel design (per stage, in-place):
  Grid: (M, N//2)  — M = all non-butterfly dims flattened, N//2 = butterfly pairs
  Thread (row, tid):
    group = tid // half,  k = tid % half
    top   = row*N + group*bfly_stride + k
    bot   = top + half
  Memory: split cfloat into contiguous float32 re/im buffers.
  Pairs within one stage are non-overlapping → no write-after-read hazard.
"""

import math
import torch

# ── Triton availability ────────────────────────────────────────────────────────
HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass


if HAS_TRITON:
    # ── K1: DIT butterfly stage ────────────────────────────────────────────────
    @triton.jit
    def _butterfly_dit_stage_kernel(
        x_re_ptr, x_im_ptr,
        N:           tl.constexpr,
        half:        tl.constexpr,
        bfly_stride: tl.constexpr,
    ):
        row   = tl.program_id(0)
        tid   = tl.program_id(1)
        group = tid // half
        k     = tid % half
        top   = row * N + group * bfly_stride + k
        bot   = top + half

        angle = -2.0 * 3.141592653589793 * k.to(tl.float32) / bfly_stride
        w_re  = tl.cos(angle);  w_im = tl.sin(angle)

        u_re = tl.load(x_re_ptr + top);  u_im = tl.load(x_im_ptr + top)
        v_re = tl.load(x_re_ptr + bot);  v_im = tl.load(x_im_ptr + bot)

        vw_re = v_re * w_re - v_im * w_im
        vw_im = v_re * w_im + v_im * w_re

        tl.store(x_re_ptr + top, u_re + vw_re)
        tl.store(x_im_ptr + top, u_im + vw_im)
        tl.store(x_re_ptr + bot, u_re - vw_re)
        tl.store(x_im_ptr + bot, u_im - vw_im)

    # ── K3: DIF butterfly stage ────────────────────────────────────────────────
    @triton.jit
    def _butterfly_dif_stage_kernel(
        x_re_ptr, x_im_ptr,
        N:           tl.constexpr,
        half:        tl.constexpr,
        bfly_stride: tl.constexpr,
    ):
        row   = tl.program_id(0)
        tid   = tl.program_id(1)
        group = tid // half
        k     = tid % half
        top   = row * N + group * bfly_stride + k
        bot   = top + half

        angle = 2.0 * 3.141592653589793 * k.to(tl.float32) / bfly_stride
        w_re  = tl.cos(angle);  w_im = tl.sin(angle)

        u_re = tl.load(x_re_ptr + top);  u_im = tl.load(x_im_ptr + top)
        v_re = tl.load(x_re_ptr + bot);  v_im = tl.load(x_im_ptr + bot)

        d_re = u_re - v_re;  d_im = u_im - v_im

        tl.store(x_re_ptr + top, u_re + v_re)
        tl.store(x_im_ptr + top, u_im + v_im)
        tl.store(x_re_ptr + bot, d_re * w_re - d_im * w_im)
        tl.store(x_im_ptr + bot, d_re * w_im + d_im * w_re)

    # ── Python wrappers ────────────────────────────────────────────────────────

    def _butterfly_fft_1d_triton(x: torch.Tensor, dim: int, fft_norm: str) -> torch.Tensor:
        """DIT butterfly FFT via Triton. log₂(N) kernel launches."""
        N     = x.shape[dim]
        log2N = N.bit_length() - 1
        x     = x.movedim(dim, -1).contiguous()
        if not x.is_complex():
            x = x.to(torch.cfloat)
        orig_shape = x.shape
        M    = x.numel() // N
        flat = x.reshape(M, N)
        x_re = flat.real.contiguous().clone()
        x_im = flat.imag.contiguous().clone()
        for s in range(log2N):
            bs = 1 << (s + 1);  h = bs >> 1
            _butterfly_dit_stage_kernel[(M, N // 2)](x_re, x_im, N=N, half=h, bfly_stride=bs)
        x = torch.complex(x_re, x_im).reshape(orig_shape)
        if fft_norm == "forward":
            x = x / N
        elif fft_norm == "ortho":
            x = x / math.sqrt(N)
        return x.movedim(-1, dim)

    def _butterfly_ifft_1d_triton(X: torch.Tensor, dim: int, fft_norm: str) -> torch.Tensor:
        """DIF butterfly IFFT via Triton. log₂(N) kernel launches."""
        N     = X.shape[dim]
        log2N = N.bit_length() - 1
        x     = X.movedim(dim, -1).contiguous()
        orig_shape = x.shape
        M    = x.numel() // N
        flat = x.reshape(M, N)
        x_re = flat.real.contiguous().clone()
        x_im = flat.imag.contiguous().clone()
        for s in range(log2N - 1, -1, -1):
            bs = 1 << (s + 1);  h = bs >> 1
            _butterfly_dif_stage_kernel[(M, N // 2)](x_re, x_im, N=N, half=h, bfly_stride=bs)
        x = torch.complex(x_re, x_im).reshape(orig_shape)
        if fft_norm == "backward":
            x = x / N
        elif fft_norm == "ortho":
            x = x / math.sqrt(N)
        return x.movedim(-1, dim)

else:
    # Stubs so imports don't fail when Triton is absent
    def _butterfly_fft_1d_triton(x, dim, fft_norm):
        raise RuntimeError("Triton is not installed — cannot use _butterfly_fft_1d_triton")

    def _butterfly_ifft_1d_triton(X, dim, fft_norm):
        raise RuntimeError("Triton is not installed — cannot use _butterfly_ifft_1d_triton")
