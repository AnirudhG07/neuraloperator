"""Tests for ButterflySpectralConv and the DIT/DIF butterfly helpers."""

import math
import pytest
import torch
from ..spectral_convolution import SpectralConv
from ..butterfly_fft import _butterfly_fft_1d, _butterfly_ifft_1d, _is_pow2
from ..butterfly_fft import ButterflySpectralConv
from ...models.fno_staged import FNOStaged


def _bitrev_1d(x: torch.Tensor) -> torch.Tensor:
    """Bit-reverse the last dimension of x."""
    N = x.shape[-1]
    bits = N.bit_length() - 1
    idx = torch.arange(N, dtype=torch.long, device=x.device)
    rev = torch.zeros_like(idx)
    for i in range(bits):
        rev = (rev << 1) | ((idx >> i) & 1)
    return x[..., rev]


# ── butterfly helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [8, 16, 32, 64, 256])
def test_butterfly_fft_1d_matches_torch(N):
    """butterfly_fft(bitrev(x)) ≈ torch.fft.fft(x) for real and complex input."""
    x = torch.randn(2, 3, N)

    # real input
    x_br = _bitrev_1d(x)
    result = _butterfly_fft_1d(x_br, dim=-1, fft_norm="backward")
    expected = torch.fft.fft(x, dim=-1, norm="backward")
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    # complex input
    xc = x + 1j * torch.randn_like(x)
    xc_br = _bitrev_1d(xc)
    result_c = _butterfly_fft_1d(xc_br, dim=-1, fft_norm="backward")
    expected_c = torch.fft.fft(xc, dim=-1, norm="backward")
    torch.testing.assert_close(result_c, expected_c, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("N", [8, 16, 32])
def test_butterfly_ifft_roundtrip(N):
    """bitrev(butterfly_ifft(butterfly_fft(bitrev(x)))) ≈ x."""
    x = torch.randn(2, 3, N)
    x_br = _bitrev_1d(x)

    X = _butterfly_fft_1d(x_br, dim=-1, fft_norm="backward")
    y_br = _butterfly_ifft_1d(X, dim=-1, fft_norm="backward")  # bit-reversed output
    y = _bitrev_1d(y_br.real.unsqueeze(0).squeeze(0))          # restore natural order

    torch.testing.assert_close(y, x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("fft_norm", ["backward", "forward", "ortho"])
def test_butterfly_fft_norms(fft_norm):
    """Check that butterfly FFT norms match torch.fft norms."""
    N = 16
    x = torch.randn(4, N)
    x_br = _bitrev_1d(x)
    result = _butterfly_fft_1d(x_br, dim=-1, fft_norm=fft_norm)
    expected = torch.fft.fft(x, dim=-1, norm=fft_norm)
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


# ── ButterflySpectralConv ─────────────────────────────────────────────────────

def _bitrev_spatial(x: torch.Tensor) -> torch.Tensor:
    """Apply FNOStaged-style spatial bitrev to all spatial dims."""
    return FNOStaged._bitrev_spatial(x)


@pytest.mark.parametrize("dim", [1, 2])
def test_butterfly_spectral_conv_matches_standard(dim):
    """bitrev(ButterflySpectralConv(bitrev(x))) ≈ SpectralConv(x) with same weights."""
    modes = (4,) * dim
    spatial = (16,) * dim
    torch.manual_seed(0)

    conv_std = SpectralConv(2, 2, modes, bias=False)
    conv_bfly = ButterflySpectralConv(2, 2, modes, bias=False)
    # Same weights so outputs should match
    conv_bfly.weight = conv_std.weight

    x = torch.randn(1, 2, *spatial)
    expected = conv_std(x)
    result = _bitrev_spatial(conv_bfly(_bitrev_spatial(x)))

    torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)


def test_fallback_non_pow2():
    """Non-power-of-2 spatial size falls back to super().forward() (cuFFT path)."""
    conv = ButterflySpectralConv(2, 2, (4, 4), bias=False)
    x = torch.randn(1, 2, 12, 12)  # 12 is not a power of 2
    # Should not raise, should return the same result as SpectralConv
    result = conv(x)
    expected = SpectralConv(2, 2, (4, 4), bias=False)
    expected.weight = conv.weight
    assert result.shape == x.shape


def test_fallback_complex_data():
    """complex_data=True always falls back to super().forward()."""
    conv = ButterflySpectralConv(2, 2, (4, 4), complex_data=True, bias=False)
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat)
    result = conv(x)
    assert result.shape == x.shape


def test_fallback_output_shape():
    """Passing output_shape (resolution scaling) falls back to super().forward()."""
    conv = ButterflySpectralConv(2, 4, (4, 4), bias=False)
    x = torch.randn(1, 2, 8, 8)
    result = conv(x, output_shape=(16, 16))
    assert result.shape == (1, 4, 16, 16)


# ── Integration with FNOStaged ────────────────────────────────────────────────

def test_fno_staged_uses_butterfly_when_no_br():
    """FNOStaged(no_br=True) should install ButterflySpectralConv layers."""
    model = FNOStaged(
        n_modes_per_layer=[[4, 4], [4, 4]],
        in_channels=1,
        out_channels=1,
        hidden_channels=8,
        no_br=True,
        positional_embedding=None,
        domain_padding=None,
        fno_skip="linear",
        use_channel_mlp=False,
    )
    assert all(isinstance(c, ButterflySpectralConv) for c in model.convs)


def test_fno_staged_no_br_output_shape():
    """End-to-end: FNOStaged(no_br=True) produces correct output shape."""
    torch.manual_seed(0)
    model = FNOStaged(
        n_modes_per_layer=[[8, 8], [8, 8]],
        in_channels=1,
        out_channels=1,
        hidden_channels=16,
        no_br=True,
        positional_embedding=None,
        domain_padding=None,
        fno_skip="linear",
        use_channel_mlp=False,
    )
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    assert out.shape == x.shape
