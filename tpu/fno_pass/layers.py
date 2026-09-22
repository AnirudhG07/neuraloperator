"""
fno_pass/layers.py — the non-spectral pieces, shared by reference.py and ours.py.

Weight tree (from init_fno), walked by both forwards:
  P = {"lift":  2-layer MLP (in_ch+2 -> 2*channels -> channels),
       "blocks": [ {"filter": {"w_re","w_im": (Cin,Cout,2m,m)}, "skip": {"W","b"}} ] * n_layers,
       "proj":  2-layer MLP (channels -> 2*channels -> out_ch)}
So the two spectral convs get IDENTICAL weights and modes -> directly comparable.
"""
import jax
import jax.numpy as jnp

try:
    from tpu.fft_core import F
except ImportError:
    from fft_core import F

gelu = jax.nn.gelu


# ── pointwise / local layers ────────────────────────────────────────────────
def add_grid(x):
    """Append normalized (x, y) coordinate channels.  (B,C,H,W) -> (B,C+2,H,W)."""
    B, _, H, W = x.shape
    gx = jnp.broadcast_to(jnp.linspace(0, 1, H)[:, None], (H, W))
    gy = jnp.broadcast_to(jnp.linspace(0, 1, W)[None, :], (H, W))
    grid = jnp.broadcast_to(jnp.stack([gx, gy])[None], (B, 2, H, W))
    return jnp.concatenate([x, grid], axis=1)


def conv1x1(w, x):
    """1x1 conv = same linear map at every pixel.  w = {W (Cout,Cin), b (Cout,)}."""
    return jnp.einsum("oi,bihw->bohw", w["W"], x) + w["b"][None, :, None, None]


def channel_mlp(w, x):
    """2-layer pointwise MLP (conv1x1 -> GELU -> conv1x1) — lift & project."""
    return conv1x1(w["l2"], gelu(conv1x1(w["l1"], x)))


# ── weight initialisation ───────────────────────────────────────────────────
def _conv1x1(key, cin, cout):
    std = (2.0 / (cin + cout)) ** 0.5
    return {"W": std * jax.random.normal(key, (cout, cin)), "b": jnp.zeros((cout,))}


def _mlp(key, cin, hid, cout):
    k1, k2 = jax.random.split(key)
    return {"l1": _conv1x1(k1, cin, hid), "l2": _conv1x1(k2, hid, cout)}


def _filter(key, cin, cout, m):
    """Complex spectral weight (Cin, Cout, 2m, m) as (w_re, w_im) — the ±m band on rows, low m on cols."""
    std = (2.0 / (cin + cout)) ** 0.5
    kr, ki = jax.random.split(key)
    return {"w_re": std * jax.random.normal(kr, (cin, cout, 2 * m, m)),
            "w_im": std * jax.random.normal(ki, (cin, cout, 2 * m, m))}


def init_fno(key, in_ch=1, out_ch=1, channels=32, m=16, n_layers=4):
    ks = jax.random.split(key, n_layers + 2)
    return {"lift": _mlp(ks[0], in_ch + 2, 2 * channels, channels),
            "blocks": [{"filter": _filter(jax.random.fold_in(ks[i + 1], 0), channels, channels, m),
                        "skip": _conv1x1(jax.random.fold_in(ks[i + 1], 1), channels, channels)}
                       for i in range(n_layers)],
            "proj": _mlp(ks[-1], channels, 2 * channels, out_ch)}


def n_params(P):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(P)))


# ── losses + Adam(W) ─────────────────────────────────────────────────────────
def mse(p, y):
    return jnp.mean((p - y) ** 2)


def rel_l2(p, y):
    n = jnp.sqrt(jnp.sum((p - y) ** 2, axis=(-1, -2, -3)))
    d = jnp.sqrt(jnp.sum(y ** 2, axis=(-1, -2, -3)))
    return jnp.mean(n / d)


def adam_init(P):
    z = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
    return {"m": z(P), "v": z(P), "t": 0}


def adam_step(P, g, st, lr=1e-3, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, st["v"], g)
    mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    P = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / (jnp.sqrt(b) + eps) + wd * p), P, mh, vh)
    return P, {"m": m, "v": v, "t": t}
