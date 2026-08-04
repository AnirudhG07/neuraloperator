"""
fno.py  —  a 1-D Fourier Neural Operator in pure JAX, built on tpu/fft_jax.py.

Same layering as the torch `neuralop` FNO, just minimal + functional:

    SpectralConv1d   ->  fft -> keep low modes -> complex channel-mix -> ifft   (the "R" operator)
    Linear1x1        ->  pointwise (1x1 conv) channel mixing                     (skip / lifting / projection)
    ChannelMLP       ->  Linear1x1 -> act -> Linear1x1                           (lifting & projection heads)
    FNOBlock         ->  act( SpectralConv1d(x) + Linear1x1_skip(x) )            (one Fourier layer)
    FNO              ->  lifting -> [FNOBlock]*L -> projection                   (the whole model)

There is no nn.Module: every layer is (init_fn, apply_fn).  Params are plain
nested dicts (a JAX pytree), so jax.grad / jax.jit / optax-free Adam all just work.

The spectral transform is OUR TPU FFT from fft_jax (fft/ifft), applied along the
spatial axis with vmap — NOT jnp.fft.  Data generation is allowed to use jnp.fft
(it only builds the ground-truth targets, it is not part of the model).

Run:  python tpu/fno.py
"""
import jax
import jax.numpy as jnp

from tpu.fft_jax_old import fft, ifft  # our TPU FFT / iFFT

# ── batched spectral transform: apply the 1-D fft/ifft along the last axis ──
def _batch_along_last(fn, x):
    """Apply a length-N 1-D transform `fn` to every length-N row of x (..., N)."""
    N = x.shape[-1]
    flat = x.reshape(-1, N)
    out = jax.vmap(fn)(flat)                 # each row -> fn(row)
    return out.reshape(*x.shape[:-1], out.shape[-1])

def batch_fft(x):
    return _batch_along_last(fft, x)

def batch_ifft(x):
    return _batch_along_last(ifft, x)

# ═══════════════════════════════════════════════════════════════════════════
#  Layer 1 — SpectralConv1d   (the Fourier "R" operator)
# ═══════════════════════════════════════════════════════════════════════════
def spectral_conv_init(key, in_ch, out_ch, n_modes):
    """Complex weight R of shape (in_ch, out_ch, n_modes), stored as real+imag."""
    std = (2.0 / (in_ch + out_ch)) ** 0.5
    kr, ki = jax.random.split(key)
    shape = (in_ch, out_ch, n_modes)
    return {
        "w_re": std * jax.random.normal(kr, shape),
        "w_im": std * jax.random.normal(ki, shape),
    }

def spectral_conv_apply(p, x):
    """
    x : (batch, in_ch, N) real  ->  (batch, out_ch, N) real.
    fft -> keep the lowest n_modes frequencies -> mix channels with complex R
    -> scatter back into a zero spectrum -> ifft -> real part.
    """
    m = p["w_re"].shape[-1]                           # n_modes (static, from weight shape)
    X = batch_fft(x)                                  # (batch, in_ch, N) complex
    Xlow = X[..., :m]                                 # keep lowest m frequencies
    R = p["w_re"] + 1j * p["w_im"]                    # (in_ch, out_ch, m)
    # channel mix per kept mode:  out[b,o,k] = sum_i Xlow[b,i,k] * R[i,o,k]
    out_low = jnp.einsum("bik,iok->bok", Xlow, R.astype(Xlow.dtype))
    # scatter into a full zero spectrum, then invert
    B, _, N = x.shape
    full = jnp.zeros((B, out_low.shape[1], N), dtype=out_low.dtype)
    full = full.at[..., :m].set(out_low)
    return batch_ifft(full).real                      # back to space, real field

# ═══════════════════════════════════════════════════════════════════════════
#  Layer 2 — Linear1x1   (pointwise / 1x1-conv channel mixing: skip, lift, project)
# ═══════════════════════════════════════════════════════════════════════════
def linear1x1_init(key, in_ch, out_ch):
    std = (2.0 / (in_ch + out_ch)) ** 0.5
    return {
        "W": std * jax.random.normal(key, (out_ch, in_ch)),
        "b": jnp.zeros((out_ch,)),
    }

def linear1x1_apply(p, x):
    """x : (batch, in_ch, N) -> (batch, out_ch, N).  Same weight at every point."""
    return jnp.einsum("oi,bin->bon", p["W"], x) + p["b"][None, :, None]

# ═══════════════════════════════════════════════════════════════════════════
#  Layer 3 — ChannelMLP   (2-layer pointwise MLP; used for lifting & projection)
# ═══════════════════════════════════════════════════════════════════════════
def channel_mlp_init(key, in_ch, hidden_ch, out_ch):
    k1, k2 = jax.random.split(key)
    return {"l1": linear1x1_init(k1, in_ch, hidden_ch),
            "l2": linear1x1_init(k2, hidden_ch, out_ch)}

def channel_mlp_apply(p, x, act):
    return linear1x1_apply(p["l2"], act(linear1x1_apply(p["l1"], x)))

# ═══════════════════════════════════════════════════════════════════════════
#  Layer 4 — FNOBlock   (one Fourier layer:  act(spectral(x) + skip(x)) )
# ═══════════════════════════════════════════════════════════════════════════
def fno_block_init(key, width, n_modes):
    ks, kl = jax.random.split(key)
    return {"spectral": spectral_conv_init(ks, width, width, n_modes),
            "skip":     linear1x1_init(kl, width, width)}

def fno_block_apply(p, x, act):
    return act(spectral_conv_apply(p["spectral"], x) + linear1x1_apply(p["skip"], x))

# ═══════════════════════════════════════════════════════════════════════════
#  Model — FNO   (lifting -> L Fourier blocks -> projection)
# ═══════════════════════════════════════════════════════════════════════════
def gelu(x):
    return jax.nn.gelu(x)

def fno_init(key, in_ch, out_ch, width=32, n_modes=16, n_layers=4,
             lift_hidden=None, proj_hidden=None, use_grid=True):
    """Returns (params, config).  in_ch is the DATA channels; +1 if use_grid."""
    lift_hidden = lift_hidden or width
    proj_hidden = proj_hidden or width
    lift_in = in_ch + (1 if use_grid else 0)          # positional grid channel
    keys = jax.random.split(key, n_layers + 2)
    params = {
        "lifting": channel_mlp_init(keys[0], lift_in, lift_hidden, width),
        "blocks": [fno_block_init(keys[i + 1], width, n_modes) for i in range(n_layers)],
        "projection": channel_mlp_init(keys[-1], width, proj_hidden, out_ch),
    }
    cfg = {"use_grid": use_grid, "n_layers": n_layers}
    return params, cfg

def fno_apply(params, x, cfg):
    """x : (batch, in_ch, N) -> (batch, out_ch, N)."""
    if cfg["use_grid"]:
        B, _, N = x.shape
        grid = jnp.linspace(0.0, 1.0, N, endpoint=False)
        grid = jnp.broadcast_to(grid, (B, 1, N))
        x = jnp.concatenate([x, grid], axis=1)         # append coordinate channel
    x = channel_mlp_apply(params["lifting"], x, gelu)
    for blk in params["blocks"]:
        x = fno_block_apply(blk, x, gelu)
    return channel_mlp_apply(params["projection"], x, gelu)

# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic operator-learning data:  a(x)  ->  u(x) = (I - c d^2/dx^2)^{-1} a
#  (a periodic Helmholtz smoothing; the true map is diagonal in Fourier, so a
#   well-specified FNO can fit it near-exactly.  Data gen may use jnp.fft.)
# ═══════════════════════════════════════════════════════════════════════════
def make_dataset(key, n_samples, N, k_max=12, c=1e-2):
    ks = jnp.arange(N // 2 + 1)                        # rfft frequencies
    # random smooth inputs: random low-freq spectrum with 1/k decay
    key, sub = jax.random.split(key)
    re = jax.random.normal(sub, (n_samples, N // 2 + 1))
    key, sub = jax.random.split(key)
    im = jax.random.normal(sub, (n_samples, N // 2 + 1))
    decay = jnp.where(ks <= k_max, 1.0 / (1.0 + ks), 0.0)[None, :]
    a_hat = (re + 1j * im) * decay
    a_hat = a_hat.at[:, 0].set(a_hat[:, 0].real)       # real DC
    a = jnp.fft.irfft(a_hat, n=N, axis=1)              # (n_samples, N) real inputs

    # true operator (diagonal in Fourier): u_hat = a_hat / (1 + c*(2*pi*k)^2)
    g = 1.0 / (1.0 + c * (2.0 * jnp.pi * ks) ** 2)
    u = jnp.fft.irfft(a_hat * g[None, :], n=N, axis=1)  # (n_samples, N) targets

    a = a[:, None, :]                                  # (n_samples, 1, N)
    u = u[:, None, :]
    return a, u

# ═══════════════════════════════════════════════════════════════════════════
#  Training  (hand-written Adam over the param pytree; no optax needed)
# ═══════════════════════════════════════════════════════════════════════════
def mse(pred, y):
    return jnp.mean((pred - y) ** 2)

def rel_l2(pred, y):
    num = jnp.sqrt(jnp.sum((pred - y) ** 2, axis=(-1, -2)))
    den = jnp.sqrt(jnp.sum(y ** 2, axis=(-1, -2)))
    return jnp.mean(num / den)

def adam_init(params):
    z = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
    return {"m": z(params), "v": z(params), "t": 0}

def adam_update(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    new = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                                 params, mhat, vhat)
    return new, {"m": m, "v": v, "t": t}

def train(seed=0, N=256, width=32, n_modes=16, n_layers=4,
          n_train=256, n_test=64, steps=300, batch=32, lr=1e-3):
    key = jax.random.PRNGKey(seed)
    key, kd = jax.random.split(key)
    a, u = make_dataset(kd, n_train + n_test, N)
    xa, ya = a[:n_train], u[:n_train]
    xt, yt = a[n_train:], u[n_train:]

    key, ki = jax.random.split(key)
    params, cfg = fno_init(ki, in_ch=1, out_ch=1, width=width,
                           n_modes=n_modes, n_layers=n_layers)
    opt = adam_init(params)

    def loss_fn(params, x, y):
        return mse(fno_apply(params, x, cfg), y)

    @jax.jit
    def step(params, opt, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params, opt = adam_update(params, grads, opt, lr=lr)
        return params, opt, loss

    print(f"FNO-1d (JAX/TPU-fft)  N={N}  width={width}  modes={n_modes}  "
          f"layers={n_layers}  |  {n_train} train / {n_test} test")
    n_batches = max(1, n_train // batch)
    for epoch in range(steps):
        key, ks = jax.random.split(key)
        perm = jax.random.permutation(ks, n_train)
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch:(b + 1) * batch]
            params, opt, loss = step(params, opt, xa[idx], ya[idx])
            ep_loss += float(loss)
        if epoch % 25 == 0 or epoch == steps - 1:
            pred = fno_apply(params, xt, cfg)
            print(f"  epoch {epoch:4d}  train_mse {ep_loss / n_batches:.3e}   "
                  f"test_mse {float(mse(pred, yt)):.3e}   test_relL2 {float(rel_l2(pred, yt)):.3e}")
    return params, cfg

if __name__ == "__main__":
    train()
