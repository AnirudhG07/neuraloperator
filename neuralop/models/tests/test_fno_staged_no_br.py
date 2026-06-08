"""Tests for FNOStaged no_br (bit-reversal) mode.

When no_br=True the forward wraps the Fourier layers with a spatial
bit-reversal permutation:

    bitrev(x) → [SpectralConv layers] → bitrev(output)

The two bit-reversals cancel (P²=I) so the network sees the data in a
permuted order but the final output is back in natural spatial order.
Spatial dims that are not a power-of-2 are left unchanged (no-op).
"""

import torch
import pytest
from neuralop.models.fno_staged import FNOStaged


def _make_model(no_br: bool, n_modes, hidden, n_layers) -> FNOStaged:
    return FNOStaged(
        n_modes_per_layer=[n_modes] * n_layers,
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden,
        no_br=no_br,
        positional_embedding=None,
        domain_padding=None,
        fno_skip="linear",
        use_channel_mlp=False,
    )


def test_bitrev_is_involution():
    """Applying _bitrev_spatial twice must return the original tensor (P²=I)."""
    x = torch.randn(2, 4, 16, 16)
    assert torch.equal(x, FNOStaged._bitrev_spatial(FNOStaged._bitrev_spatial(x)))


def test_bitrev_permutes_spatial_dims():
    """_bitrev_spatial must actually reorder spatial dimensions, not be a no-op."""
    x = torch.randn(1, 1, 8, 8)
    x_br = FNOStaged._bitrev_spatial(x)
    assert not torch.equal(x, x_br), "bitrev should permute a non-trivial tensor"
    assert x_br.shape == x.shape


@pytest.mark.parametrize("spatial", [(16, 16), (32, 32)])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_no_br_output_shape(spatial, n_layers):
    """no_br=True must produce an output of the same shape as no_br=False."""
    torch.manual_seed(0)
    model = _make_model(no_br=True, n_modes=[8, 8], hidden=8, n_layers=n_layers)
    x = torch.randn(2, 1, *spatial)
    out = model(x)
    assert out.shape == x.shape


def test_no_br_fallback_non_power_of_2():
    """For spatial dims that are not power-of-2, _bitrev_spatial is a no-op
    and the model still runs without error."""
    x = torch.randn(1, 1, 12, 12)
    # _bitrev_spatial should be a no-op for non-power-of-2
    assert torch.equal(x, FNOStaged._bitrev_spatial(x))

    model = _make_model(no_br=True, n_modes=[6, 6], hidden=8, n_layers=2)
    out = model(x)
    assert out.shape == x.shape
