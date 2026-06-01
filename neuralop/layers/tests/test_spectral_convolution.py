import pytest
import torch
from tltorch import FactorizedTensor
from ..spectral_convolution import SpectralConv


@pytest.mark.parametrize("factorization", ["Dense", "CP", "Tucker", "TT"])
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("complex_data", [False, True])
def test_SpectralConv(factorization, implementation, separable, dim, complex_data):
    """Test for SpectralConv of any order

    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    modes = (10, 8, 6, 6)
    incremental_modes = (6, 6, 4, 4)
    dtype = torch.cfloat if complex_data else torch.float32

    # Test for Conv1D to Conv4D
    conv = SpectralConv(
        3,
        3,
        modes[:dim],
        bias=False,
        implementation=implementation,
        factorization=factorization,
        complex_data=complex_data,
        separable=separable,
    )

    conv_dense = SpectralConv(
        3,
        3,
        modes[:dim],
        bias=False,
        implementation="reconstructed",
        factorization=None,
        complex_data=complex_data,
    )

    x = torch.randn(2, 3, *(12,) * dim, dtype=dtype)

    assert torch.is_complex(conv.weight)
    assert torch.is_complex(conv_dense.weight)

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        conv_dense.weight = FactorizedTensor.from_tensor(
            conv.weight.to_tensor(), rank=None, factorization="ComplexDense"
        )

    res_dense = conv_dense(x)
    res = conv(x)
    res_shape = res.shape

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        torch.testing.assert_close(res_dense, res)

    # Dynamically reduce the number of modes in Fourier space
    conv.n_modes = incremental_modes[:dim]
    res = conv(x)
    assert res_shape == res.shape

    # Downsample outputs
    block = SpectralConv(3, 4, modes[:dim], resolution_scaling_factor=0.5)

    x = torch.randn(2, 3, *(12,) * dim)
    res = block(x)
    assert list(res.shape[2:]) == [12 // 2] * dim

    # Upsample outputs
    block = SpectralConv(3, 4, modes[:dim], resolution_scaling_factor=2)

    x = torch.randn(2, 3, *(12,) * dim)
    res = block(x)
    assert res.shape[1] == 4  # Check out channels
    assert list(res.shape[2:]) == [12 * 2] * dim


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("separable", [False, True])
def test_SpectralConv_no_br(dim, separable):
    """Test bit-reversal-free (no_br) mode against standard path."""
    modes = (4, 4, 4)
    size = (8, 8, 8)
    
    # Standard path
    conv_std = SpectralConv(
        in_channels=3,
        out_channels=3,
        n_modes=modes[:dim],
        no_br=False,
        separable=separable,
        bias=False
    )
    
    # no_br path
    conv_no_br = SpectralConv(
        in_channels=3,
        out_channels=3,
        n_modes=modes[:dim],
        no_br=True,
        separable=separable,
        bias=False
    )
    
    # Copy weights from standard to no_br
    # Note: SpectralConv.__init__ applies bit-reversal to weights IF no_br is True.
    # To compare fairly, we take the standard weight, and MANUALLY apply 
    # the permutation so that conv_no_br has the 'same' semantic weight.
    
    with torch.no_grad():
        w = conv_std.weight.to_tensor()
        # The no_br mode expects the weight to be pre-permuted.
        # R̃[bitrev(k)] = R[k]
        mode_start = 1 if separable else 2
        w_perm = w.clone()
        for axis in range(mode_start, w.ndim):
            n = w.shape[axis]
            if n > 0 and (n & (n - 1)) == 0:
                perm = SpectralConv._bitrev_indices(n)
                # We want R̃[perm] = R, so R̃ = R[inv_perm]. 
                # For bit-reversal, perm is its own inverse.
                w_perm = torch.index_select(w_perm, axis, perm)
        
        if torch.is_tensor(conv_no_br.weight):
            conv_no_br.weight.data.copy_(w_perm)
        else:
            conv_no_br.weight.data.copy_(w_perm)

    x = torch.randn(2, 3, *size[:dim])
    
    res_std = conv_std(x)
    res_no_br = conv_no_br(x)
    
    # The output should be identical (within float precision)
    torch.testing.assert_close(res_std, res_no_br, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("spatial_size", [8, 9])  # Even and odd: Nyquist handling differs
@pytest.mark.parametrize("resolution_scaling_factor", [None, 0.5, 2])
@pytest.mark.parametrize("modes", [(4, 4, 4), (4, 5, 7)])
def test_SpectralConv2(enforce_hermitian_symmetry, dim, spatial_size, modes, resolution_scaling_factor):
    modes = modes[:dim]
    size = [spatial_size] * dim
    if resolution_scaling_factor is None:
        out_size = size
    else:
        out_size = [round(s * resolution_scaling_factor) for s in size]

    # Test with real-valued data
    conv = SpectralConv(
        3,
        4,
        modes,
        enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        complex_data=False,
        resolution_scaling_factor=resolution_scaling_factor,
    )
    x = torch.randn(2, 3, *size, dtype=torch.float32)
    res = conv(x)

    assert res.shape == (2, 4, *out_size)
    assert res.dtype == torch.float32
    assert not torch.is_complex(res)
