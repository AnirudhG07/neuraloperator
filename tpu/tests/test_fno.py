"""
Correctness net for the fast FNO: the real/imag path must agree with a textbook complex64
jnp.fft implementation.  Runs on CPU, so it guards refactors without needing a TPU.

    .venv/bin/python -m pytest tpu/tests -q
"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ..experimental.fno import reference
from ..fft import core, rfft1d, rfft2d
from ..fno import layers, model, spectral, train

SHAPES = [(32, 32, 8), (64, 64, 12), (17, 16, 6)]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


@pytest.mark.parametrize("n", [64, 256, 1024])
def test_rfft_matches_numpy(n):
    x = jnp.asarray(np.random.default_rng(0).standard_normal((n, 8)), jnp.float32)
    re, im = rfft1d.rfft_full(x)
    want = np.fft.fft(np.asarray(x), axis=0)
    assert _rel(re, want.real) < 1e-4
    assert _rel(im, want.imag) < 1e-4


@pytest.mark.parametrize("n,n_modes", [(256, 4), (256, 16), (1024, 8)])
def test_rfft_low_modes_matches_numpy(n, n_modes):
    x = jnp.asarray(np.random.default_rng(1).standard_normal((n, 8)), jnp.float32)
    re, im = rfft1d.rfft_low_modes(x, n_modes)
    want = np.fft.fft(np.asarray(x), axis=0)[:n_modes]
    assert _rel(re, want.real) < 1e-4
    assert _rel(im, want.imag) < 1e-4


@pytest.mark.parametrize("n_rows,n_cols", [(128, 32), (256, 64)])
def test_rfft2_matches_numpy(n_rows, n_cols):
    field = jnp.asarray(np.random.default_rng(2).standard_normal((n_rows, n_cols, 4)), jnp.float32)
    re, im = rfft2d.rfft2_full(field)
    want = np.fft.fft2(np.asarray(field), axes=(0, 1))[:n_rows // 2]
    assert _rel(re, want.real) < 1e-4
    assert _rel(im, want.imag) < 1e-4


def test_rfft2_staged_cols_matches_direct():
    field = jnp.asarray(np.random.default_rng(3).standard_normal((128, 32, 4)), jnp.float32)
    direct, staged = rfft2d.rfft2_full(field), rfft2d.rfft2_full_staged_cols(field)
    assert _rel(staged[0], direct[0]) < 1e-4
    assert _rel(staged[1], direct[1]) < 1e-4


def test_fft_staged_complex_matches_numpy():
    rng = np.random.default_rng(4)
    re = jnp.asarray(rng.standard_normal((256, 8)), jnp.float32)
    im = jnp.asarray(rng.standard_normal((256, 8)), jnp.float32)
    got_re, got_im = core.fft_staged_complex(re, im, 256)
    want = np.fft.fft(np.asarray(re) + 1j * np.asarray(im), axis=0)
    assert _rel(got_re, want.real) < 1e-4
    assert _rel(got_im, want.imag) < 1e-4


@pytest.mark.parametrize("n_rows,n_cols,n_modes", SHAPES)
def test_spectral_conv_matches_complex64_reference(n_rows, n_cols, n_modes):
    """The whole point of the real/imag rewrite: same answer as the complex64 version."""
    w = layers._init_spectral_weight(jax.random.PRNGKey(0), 4, 4, n_modes)
    x = jnp.asarray(np.random.default_rng(5).standard_normal((2, 4, n_rows, n_cols)), jnp.float32)
    assert _rel(spectral.spectral_conv(w, x, n_modes), reference.spectral_conv(w, x, n_modes)) < 2e-3


def test_forward_matches_reference():
    P = layers.init_fno(jax.random.PRNGKey(7), 1, 1, channels=8, n_modes=6, n_layers=3)
    x = jnp.asarray(np.random.default_rng(6).standard_normal((3, 1, 32, 32)), jnp.float32)
    assert _rel(model.forward(P, x, 6), reference.forward(P, x, 6)) < 2e-3


def test_save_load_round_trip():
    P = layers.init_fno(jax.random.PRNGKey(8), 1, 1, channels=8, n_modes=6, n_layers=2)
    x = jnp.asarray(np.random.default_rng(7).standard_normal((2, 1, 32, 32)), jnp.float32)
    path = os.path.join(tempfile.mkdtemp(), "params.npz")
    model.save_params(P, path)
    restored = model.load_params(path, channels=8, n_modes=6, n_layers=2)
    assert _rel(model.forward(restored, x, 6), model.forward(P, x, 6)) == 0.0


def test_training_reduces_error():
    rng = np.random.default_rng(9)
    x = np.asarray(rng.standard_normal((32, 1, 16, 16)), np.float32)
    y = np.asarray(np.tanh(2 * x), np.float32)
    _, rel, _ = train.train(x, y, x[:8], y[:8], n_modes=4, channels=8, n_layers=2,
                            steps=6, batch=8, log_every=5, verbose=False)
    assert rel < 1.0
