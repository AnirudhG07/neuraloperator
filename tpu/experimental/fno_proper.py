import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from tpu.fft_core import ACC, F
    from tpu.fno_reim_2d import (
        _adam_init,
        _adam_step,
        _conv1x1,
        _mlp,
        add_grid,
        channel_mlp,
        conv1x1,
        mse,
        rel_l2,  # noqa: F401
    )
except ImportError:
    from fft_core import ACC, F
    from fno_reim_2d import (
        _adam_init,
        _adam_step,
        _conv1x1,
        _mlp,
        add_grid,
        channel_mlp,
        conv1x1,
        mse,
    )

gelu = jax.nn.gelu
_E = lambda s, M, Z: jnp.einsum(s, M, Z, preferred_element_type=ACC)


# ── DFT rows for an explicit set of mode indices (so we can pick the ± band) ──
def _rows(k_idx, L, sign):
    k = jnp.asarray(k_idx, F)[:, None]
    n = jnp.arange(L, dtype=F)[None, :]
    a = sign * 2.0 * jnp.pi * k * n / L
    return jnp.cos(a), jnp.sin(a)                             # (len(k_idx), L)


#  2-corner spectral conv (real/imag): keep k1 ∈ [0,m) ∪ [H-m,H) on H(rows) ; k2 ∈ [0,m) on W(cols)
def spectral_2corner(w, x, m):
    """x (B,C,H,W) real -> (B,Cout,H,W) real.  FFT2 (keep the ±m low band on the H axis, low m on
    the W axis) -> learned complex channel mix -> iFFT2 (Hermitian fold on the rfft'd W axis) ->
    real field.  Everything is carried as real/imag (no complex64).
    H = x.shape[-2] = rows (height);  W = x.shape[-1] = cols (width)."""
    H, W = x.shape[-2], x.shape[-1]                  # H = rows (axis -2), W = cols (axis -1)
    x = x.astype(F)

    # forward FFT2 — keep only the low modes
    h_band = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])  # ±m band on H (rows), 2m indices

    # FFT along H (rows) (contract H); real input -> 2 matmuls -> (n, c, 2m, W)
    cos_h, sin_h = _rows(h_band, H, -1)
    xh_re = _E("ah,nchw->ncaw", cos_h, x)   # H-transformed, real part
    xh_im = _E("ah,nchw->ncaw", sin_h, x)   #  imag part

    # FFT along W (cols) (contract W, keep low m); complex now -> (n, c, 2m, m)
    cos_w, sin_w = _rows(jnp.arange(m), W, -1)
    fft_w = lambda cos_or_sin, z: _E("dw,ncaw->ncad", cos_or_sin, z)
    spec_re = fft_w(cos_w, xh_re) - fft_w(sin_w, xh_im)   # low-mode spectrum, real
    spec_im = fft_w(cos_w, xh_im) + fft_w(sin_w, xh_re)   #  imag

    # learned complex channel mix — weights (Cin, Cout, 2m, m)
    mix = lambda weight, z: _E("ioad,niad->noad", weight, z)   # contract Cin -> Cout, per mode
    mixed_re = mix(w["w_re"], spec_re) - mix(w["w_im"], spec_im)
    mixed_im = mix(w["w_re"], spec_im) + mix(w["w_im"], spec_re)

    # iFFT along W (cols), Hermitian fold (DC once, others twice — the −k conjugate of the rfft'd axis)
    w_modes = jnp.arange(m)
    herm_fold = jnp.where(w_modes == 0, 1.0, 2.0).astype(F)
    inv_cos_w, inv_sin_w = _rows(w_modes, W, +1)   # (m, W)
    inv_cos_w, inv_sin_w = (inv_cos_w * herm_fold[:, None]).T, (inv_sin_w * herm_fold[:, None]).T  # (W, m)
    # ZERO-PADDING (implicit): this (m -> W) inverse reconstructs W from only m kept modes,
    #   i.e. all un-kept W modes are treated as 0 — no explicit jnp.pad / zeros tensor.
    ifft_w = lambda cos_or_sin, z: _E("wd,noad->noaw", cos_or_sin, z)
    recon_w_re = ifft_w(inv_cos_w, mixed_re) - ifft_w(inv_sin_w, mixed_im)   # W-reconstructed, real
    recon_w_im = ifft_w(inv_cos_w, mixed_im) + ifft_w(inv_sin_w, mixed_re)   #  imag

    # iFFT along H (rows) over the ±band, take the REAL part
    # ZERO-PADDING (implicit): (2m -> H) inverse likewise treats all un-kept H modes as 0.
    inv_cos_h, inv_sin_h = _rows(h_band, H, +1)   # (2m, H) -> used as (H, 2m)
    ifft_h = lambda cos_or_sin, z: _E("ha,noaw->nohw", cos_or_sin.T, z)
    field = ifft_h(inv_cos_h, recon_w_re) - ifft_h(inv_sin_h, recon_w_im)

    return (field / (H * W)).astype(F)   # normalize (1/HW)


def spectral_2corner_hybrid(w, x, m):
    """Same 2-corner spectral conv as spectral_2corner, but axis-0 uses the FULL (H×H) DFT (a full
    128×128 MXU matmul at 100% utilization) and then SELECTS the ±m band — instead of the skinny
    (2m×H) partial matmul that wastes most of the MXU.  axis-0 is the big-batch axis (n·c·W columns),
    so full-MXU efficiency there wins (the measured 'reim cols + partial rows' hybrid).  Identical
    output to spectral_2corner (same modes).  axis-1 stays partial; inverse unchanged."""
    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    k1 = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])   # ± band, 2m indices

    # forward axis-0: FULL (H×H) DFT (full MXU), then select the ±band  <-- the only change
    Cr0f, Ci0f = _rows(jnp.arange(H), H, -1)  # (H, H) full DFT
    Arf = _E("ah,nchw->ncaw", Cr0f, x)  # (n,c,H,W) all modes
    Aif = _E("ah,nchw->ncaw", Ci0f, x)

    Ar = jnp.concatenate([Arf[:, :, :m], Arf[:, :, H - m:]], axis=2)   # ±band -> (n,c,2m,W)
    Ai = jnp.concatenate([Aif[:, :, :m], Aif[:, :, H - m:]], axis=2)

    # forward axis-1 (partial, low m) — identical to spectral_2corner
    Cr1, Ci1 = _rows(jnp.arange(m), W, -1)
    ew = lambda M, Z: _E("dw,ncaw->ncad", M, Z)
    Xr = ew(Cr1, Ar) - ew(Ci1, Ai); Xi = ew(Cr1, Ai) + ew(Ci1, Ar)
    mix = lambda M, Z: _E("ioad,niad->noad", M, Z)

    Or_ = mix(w["w_re"], Xr) - mix(w["w_im"], Xi); Oi = mix(w["w_re"], Xi) + mix(w["w_im"], Xr)
    d = jnp.arange(m); alpha = jnp.where(d == 0, 1.0, 2.0).astype(F)
    IwR, IwI = _rows(d, W, +1)
    IwR, IwI = (IwR * alpha[:, None]).T, (IwI * alpha[:, None]).T
    iw = lambda M, Z: _E("wd,noad->noaw", M, Z)

    Pr = iw(IwR, Or_) - iw(IwI, Oi); Pi = iw(IwR, Oi) + iw(IwI, Or_)
    IhR, IhI = _rows(k1, H, +1)
    ih = lambda M, Z: _E("ha,noaw->nohw", M.T, Z)
    u = ih(IhR, Pr) - ih(IhI, Pi)
    
    return (u / (H * W)).astype(F)


def spectral_2corner_herm(w, x, m):
    """Same 2-corner conv, but exploit HERMITIAN symmetry of the real axis-0 transform:
    X[H-k] = conj(X[k]).  So compute ONLY the low-positive modes X[0..m] (m+1 rows) with a
    ((m+1)×H) DFT, then get the NEGATIVE (top) corner FREE as the conjugate — halving the axis-0
    forward matmul rows (m+1 vs 2m).  Same modes/output as spectral_2corner."""

    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    k1 = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])

    # forward axis-0: ONLY the low-positive modes [0..m] (m+1 rows)  <-- half the current 2m rows
    Cr0, Ci0 = _rows(jnp.arange(m + 1), H, -1)                   # (m+1, H)
    Ap_r = _E("ah,nchw->ncaw", Cr0, x); Ap_i = _E("ah,nchw->ncaw", Ci0, x)   # (n,c,m+1,W)

    # ±band (k1 order): bottom = X[0..m); top = conj(X[m],X[m-1],...,X[1])  (free conjugate)
    Ar = jnp.concatenate([Ap_r[:, :, :m], Ap_r[:, :, m:0:-1]], axis=2)       # (n,c,2m,W)
    Ai = jnp.concatenate([Ap_i[:, :, :m], -Ap_i[:, :, m:0:-1]], axis=2)      # top imag negated

    # identical from here (axis-1 partial, mix, inverse)
    Cr1, Ci1 = _rows(jnp.arange(m), W, -1)
    ew = lambda M, Z: _E("dw,ncaw->ncad", M, Z)
    Xr = ew(Cr1, Ar) - ew(Ci1, Ai); Xi = ew(Cr1, Ai) + ew(Ci1, Ar)

    mix = lambda M, Z: _E("ioad,niad->noad", M, Z)
    Or_ = mix(w["w_re"], Xr) - mix(w["w_im"], Xi); Oi = mix(w["w_re"], Xi) + mix(w["w_im"], Xr)
    d = jnp.arange(m); alpha = jnp.where(d == 0, 1.0, 2.0).astype(F)
    IwR, IwI = _rows(d, W, +1)
    IwR, IwI = (IwR * alpha[:, None]).T, (IwI * alpha[:, None]).T
    iw = lambda M, Z: _E("wd,noad->noaw", M, Z)

    Pr = iw(IwR, Or_) - iw(IwI, Oi); Pi = iw(IwR, Oi) + iw(IwI, Or_)
    IhR, IhI = _rows(k1, H, +1)
    ih = lambda M, Z: _E("ha,noaw->nohw", M.T, Z)
    u = ih(IhR, Pr) - ih(IhI, Pi)
    return (u / (H * W)).astype(F)


def init_2corner(key, cin, cout, m):
    s = (2.0 / (cin + cout)) ** 0.5
    kr, ki = jax.random.split(key)
    return {"w_re": s * jax.random.normal(kr, (cin, cout, 2 * m, m)),
            "w_im": s * jax.random.normal(ki, (cin, cout, 2 * m, m))}


# FNO forward: 2-corner spectral conv (no domain padding — matches the original Li-2021 FNO)
def forward_fno(P, x, m, spec_fn=spectral_2corner):
    """add (x,y) grid -> LIFT -> [2corner spectral + skip -> GELU]*L -> PROJECT.
    spec_fn = spectral_2corner (default) or spectral_2corner_hybrid (full-MXU axis-0)."""
    x = add_grid(x)                                   # append (x,y) coordinate channels
    x = channel_mlp(P["lift"], x)                     # LIFT: in_ch -> width
    for blk in P["blocks"]:                           # each Fourier block:
        spec = spec_fn(blk["filter"], x, m)           #   spectral: FFT2 -> low modes -> mix -> iFFT2
        x = gelu(spec + conv1x1(blk["skip"], x))      #   + skip 1x1 conv, then GELU
    return channel_mlp(P["proj"], x)                  # PROJECT: width -> out_ch


def init_fno(key, in_ch=1, out_ch=1, width=32, m=16, n_layers=4):
    ks = jax.random.split(key, n_layers + 2)
    return {"lift": _mlp(ks[0], in_ch + 2, 2 * width, width),
            "blocks": [{"filter": init_2corner(jax.random.fold_in(ks[i + 1], 0), width, width, m),
                        "skip": _conv1x1(jax.random.fold_in(ks[i + 1], 1), width, width)}
                       for i in range(n_layers)],
            "proj": _mlp(ks[-1], width, 2 * width, out_ch)}


def n_params(P):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(P)))


def save_params(P, path):
    np.savez(path, *[np.asarray(a) for a in jax.tree_util.tree_leaves(P)])
    return path


def load_params(path, width, m, n_layers=4, in_ch=1, out_ch=1):
    d = np.load(path)
    leaves = [jnp.asarray(d[f"arr_{i}"]) for i in range(len(d.files))]
    _, td = jax.tree_util.tree_flatten(init_fno(jax.random.PRNGKey(0), in_ch, out_ch, width, m, n_layers))
    return jax.tree_util.tree_unflatten(td, leaves)


def infer_us_per_sample(P, x, m, reps=30, max_batch=128):
    import time
    x = jnp.asarray(x[:max_batch])                        # cap batch so 128² doesn't OOM
    fn = jax.jit(lambda z: forward_fno(P, z, m))
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts) // 2]
    return med / x.shape[0] * 1e6, x.shape[0] / med


def train_fno(name, xtr, ytr, xte, yte, width=32, m=16, steps=200, batch=16, lr=1e-3,
                 wd=1e-4, step_size=100, gamma=0.5, seed=0, out=None, save=None):
    """Proper FNO training with EARLY STOPPING (keep the best-test params) — so overfitting can't
    silently hurt the reported/saved model."""
    xtr, ytr = np.asarray(xtr, np.float32), np.asarray(ytr, np.float32)   # keep on HOST (numpy)
    xte, yte = np.asarray(xte, np.float32), np.asarray(yte, np.float32)
    ntr = xtr.shape[0]
    key = jax.random.PRNGKey(seed)
    P = init_fno(key, 1, 1, width, m, 4)
    opt = _adam_init(P)

    @jax.jit
    def step(P, opt, x, y, lr_ep):
        loss, g = jax.value_and_grad(lambda P: mse(forward_fno(P, x, m), y))(P)
        P, opt = _adam_step(P, g, opt, lr=lr_ep, wd=wd)
        return P, opt, loss

    @jax.jit
    def eval_batch(P, x, y):
        p = forward_fno(P, x, m)
        return jnp.sum((p - y) ** 2, (-1, -2, -3)), jnp.sum(y ** 2, (-1, -2, -3))   # per-sample

    def test_relL2(P):                                    # eval in chunks to avoid OOM
        num = den = 0.0
        for i in range(0, xte.shape[0], 64):
            n, d = eval_batch(P, jnp.asarray(xte[i:i + 64]), jnp.asarray(yte[i:i + 64]))
            num += float(jnp.sum(jnp.sqrt(n))); den += float(jnp.sum(jnp.sqrt(d)))
        return num / den                                  # mean rel L2 (num/den summed per-sample)

    print(f"FNO-PROPER {name}: width={width} m={m} 2-corner  | {ntr} train / {xte.shape[0]} test  "
          f"params={n_params(P):,}", flush=True)
    best_rl, best_P, rows, nb = 1e9, P, [("epoch", "train_mse", "test_relL2", "best")], max(1, ntr // batch)
    t_train0 = time.perf_counter()                             # wall-clock of the whole train loop
    for ep in range(steps):
        lr_ep = lr * (gamma ** (ep // step_size))
        key, ks = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(ks, ntr))
        tot = 0.0
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            P, opt, loss = step(P, opt, jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]), lr_ep)
            tot += float(loss)
        if ep % 10 == 0 or ep == steps - 1:
            rl = test_relL2(P)
            if rl < best_rl:
                best_rl, best_P = rl, jax.tree_util.tree_map(lambda a: a, P)   # keep best
            rows.append((ep, tot / nb, rl, best_rl))
            print(f"  epoch {ep:4d}  train_mse {tot/nb:.4e}  test_relL2 {rl:.4e}  best {best_rl:.4e}", flush=True)
    train_secs = time.perf_counter() - t_train0
    print(f"TRAIN_TIME {name}: {train_secs:.2f} s  ({steps} epochs x {nb} steps/epoch = "
          f"{steps*nb} steps, batch {batch})", flush=True)
    if out:
        open(out, "w").write("\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
    if save:
        save_params(best_P, save)
        print(f"saved BEST model (relL2 {best_rl:.4e}) -> {save}", flush=True)
    return best_P, (xte, yte), best_rl, train_secs


# ── reference: standard jnp.fft.rfft2 2-corner spectral conv (to verify ours) ──
def _spectral_2corner_ref(w, x, m):
    B, C, H, W = x.shape
    Xf = jnp.fft.rfft2(x, axes=(-2, -1))
    Wc = w["w_re"] + 1j * w["w_im"]
    out = jnp.zeros((B, w["w_re"].shape[1], H, W // 2 + 1), Xf.dtype)
    top = jnp.einsum("ioad,niad->noad", Wc[:, :, :m], Xf[..., :m, :m])
    bot = jnp.einsum("ioad,niad->noad", Wc[:, :, m:], Xf[..., H - m:, :m])
    out = out.at[..., :m, :m].set(top).at[..., H - m:, :m].set(bot)
    return jnp.fft.irfft2(out, s=(H, W), axes=(-2, -1))
