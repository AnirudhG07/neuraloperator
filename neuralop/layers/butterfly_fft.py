"""Butterfly FFT (pure-PyTorch DIT/DIF) + dispatchers + ButterflySpectralConv."""

import math
import os
import torch
from .spectral_convolution import SpectralConv


# ── Butterfly math ─────────────────────────────────────────────────────────────

def _is_pow2(n: int) -> bool:
    return n > 1 and (n & (n - 1)) == 0


def _twiddle(half: int, stride: int, device: torch.device, inverse: bool = False) -> torch.Tensor:
    sign  = 1.0 if inverse else -1.0
    angle = sign * 2.0 * math.pi / stride * torch.arange(half, dtype=torch.float32, device=device)
    return torch.polar(torch.ones(half, dtype=torch.float32, device=device), angle)


def _butterfly_fft_1d(x: torch.Tensor, dim: int, fft_norm: str) -> torch.Tensor:
    """DIT butterfly FFT. Input must be bit-reversed; returns DFT in natural order."""
    N     = x.shape[dim]
    log2N = N.bit_length() - 1
    x     = x.movedim(dim, -1).contiguous()
    if not x.is_complex():
        x = x.to(torch.cfloat)
    shape = x.shape
    for s in range(log2N):
        stride = 1 << (s + 1)
        half   = stride >> 1
        w      = _twiddle(half, stride, x.device, inverse=False)
        x      = x.view(*shape[:-1], N // stride, stride)
        u      = x[..., :half].clone()
        v      = x[..., half:] * w
        x      = torch.cat([u + v, u - v], dim=-1).view(shape)
    if fft_norm == "forward":
        x = x / N
    elif fft_norm == "ortho":
        x = x / math.sqrt(N)
    return x.movedim(-1, dim)


def _butterfly_ifft_1d(X: torch.Tensor, dim: int, fft_norm: str) -> torch.Tensor:
    """DIF butterfly IFFT. Input DFT natural order; output IDFT in bit-reversed order."""
    N     = X.shape[dim]
    log2N = N.bit_length() - 1
    x     = X.movedim(dim, -1).contiguous()
    shape = x.shape
    for s in range(log2N - 1, -1, -1):
        stride = 1 << (s + 1)
        half   = stride >> 1
        w_inv  = _twiddle(half, stride, x.device, inverse=True)
        x      = x.view(*shape[:-1], N // stride, stride)
        u      = x[..., :half].clone()
        v      = x[..., half:].clone()
        x      = torch.cat([u + v, (u - v) * w_inv], dim=-1).view(shape)
    if fft_norm == "backward":
        x = x / N
    elif fft_norm == "ortho":
        x = x / math.sqrt(N)
    return x.movedim(-1, dim)


def _hermitian_extend(X_half: torch.Tensor, dim: int, N_out: int) -> torch.Tensor:
    """Extend rfft half-spectrum (N//2+1) to full hermitian spectrum (N)."""
    n_extra  = N_out - X_half.shape[dim]
    negative = torch.flip(X_half.narrow(dim, 1, n_extra), dims=[dim]).conj()
    return torch.cat([X_half, negative], dim=dim)


def butterfly_fft_1d(x: torch.Tensor, dim: int, fft_norm: str) -> torch.Tensor:
    """Dispatch: Triton (CUDA) or pure-PyTorch fallback."""
    if x.is_cuda and not os.getenv("NO_TRITON"):
        try:
            from .triton_spectral_conv import HAS_TRITON as _ht, _butterfly_fft_1d_triton
            if _ht:
                return _butterfly_fft_1d_triton(x, dim, fft_norm)
        except ImportError:
            pass
    return _butterfly_fft_1d(x, dim, fft_norm)


def butterfly_ifft_1d(X: torch.Tensor, dim: int, fft_norm: str) -> torch.Tensor:
    """Dispatch: Triton (CUDA) or pure-PyTorch fallback."""
    if X.is_cuda and not os.getenv("NO_TRITON"):
        try:
            from .triton_spectral_conv import HAS_TRITON as _ht, _butterfly_ifft_1d_triton
            if _ht:
                return _butterfly_ifft_1d_triton(X, dim, fft_norm)
        except ImportError:
            pass
    return _butterfly_ifft_1d(X, dim, fft_norm)


# ── ButterflySpectralConv ──────────────────────────────────────────────────────

class ButterflySpectralConv(SpectralConv):
    """SpectralConv using DIT/DIF butterfly FFT (Triton on CUDA, PyTorch fallback).

    Expects input in bit-reversed spatial order (FNOStaged applies _bitrev_spatial
    before each layer when no_br=True).  Returns output in bit-reversed spatial
    order — FNOStaged's second bitrev restores natural order (P²=I).

    Falls back to super().forward() when:
      output_shape given | complex_data=True | precision != 'full' | any dim not pow-2
    """

    def forward(self, x: torch.Tensor, output_shape=None):
        # ── Fallback check ─────────────────────────────────────────────────────
        if output_shape is not None:
            return super().forward(x, output_shape=output_shape)

        spatial_dims = x.ndim - 2
        mode_sizes   = list(x.shape[2:])

        if (
            self.complex_data
            or self.fno_block_precision != "full"
            or not all(_is_pow2(s) for s in mode_sizes)
        ):
            return super().forward(x, output_shape=output_shape)

        # ── Butterfly FFT ──────────────────────────────────────────────────────
        fft_dims  = list(range(2, 2 + spatial_dims))
        batchsize = x.shape[0]

        out_fft = x.to(torch.cfloat)
        for d in fft_dims:
            out_fft = butterfly_fft_1d(out_fft, dim=d, fft_norm=self.fft_norm)

        # -- Truncate rfft half + fftshift non-last dims so DC is at centre ──
        # Discard conjugate half on last dim (equivalent to rfft)
        last_half = mode_sizes[-1] // 2 + 1
        out_fft   = out_fft.narrow(fft_dims[-1], 0, last_half)
        fft_size  = mode_sizes[:-1] + [last_half]

        # fftshift on non-last dims so DC is at centre
        # mode-selection convention exactly (symmetric around DC)
        if spatial_dims > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        # ── Mode selection + weight contraction ──────────────────────────────
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        slices_w += [
            slice(start // 2, -start // 2) if start else slice(None)
            for start in starts[:-1]
        ]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self.weight[tuple(slices_w)]

        weight_start = 1 if self.separable else 2
        slices_x = [slice(None), slice(None)]
        for i, (all_modes, kept_modes) in enumerate(
            zip(fft_size, list(weight.shape[weight_start:]))
        ):
            if i < spatial_dims - 1:  # fftshifted non-last dims
                center = all_modes // 2
                neg    = kept_modes // 2
                pos    = kept_modes // 2 + kept_modes % 2
                slices_x.append(slice(center - neg, center + pos))
            else:                     # rfft-half last dim (DC at index 0)
                slices_x.append(slice(None, kept_modes) if kept_modes < all_modes else slice(None))
        slices_x = tuple(slices_x)

        result_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size],
            device=x.device, dtype=torch.cfloat,
        )
        result_fft[slices_x] = self._contract(out_fft[slices_x], weight, separable=self.separable)

        # ifftshift to restore DC at index 0 before butterfly IFFT
        if spatial_dims > 1:
            result_fft = torch.fft.ifftshift(result_fft, dim=fft_dims[:-1])

        # ── Butterfly IFFT — mirrors SpectralConv's enforce_hermitian_symmetry ──
        # Non-last dims first (so hermitian extend on last dim is still valid)
        for d in fft_dims[:-1]:
            result_fft = butterfly_ifft_1d(result_fft, dim=d, fft_norm=self.fft_norm)

        # Enforce real-valued DC and Nyquist before hermitian extension
        result_fft.narrow(fft_dims[-1], 0, 1).imag.zero_()
        if mode_sizes[-1] % 2 == 0:
            result_fft.narrow(fft_dims[-1], result_fft.shape[fft_dims[-1]] - 1, 1).imag.zero_()

        # Extend rfft half → full spectrum, then IFFT last dim
        result_fft = _hermitian_extend(result_fft, dim=fft_dims[-1], N_out=mode_sizes[-1])
        result_fft = butterfly_ifft_1d(result_fft, dim=fft_dims[-1], fft_norm=self.fft_norm)

        out = result_fft.real
        if self.bias is not None:
            out = out + self.bias
        return out
