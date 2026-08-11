"""
fft1d/jax_fft.py — 1D pure-JAX (plain-XLA, no pallas_call) FFT entry points.

  fft_iter / fft_recur : complex radix-B FFT (iterative stage-loop / recursive twin)
  jax_rfft             : real/imag rfft via the shared staged pass (dt = F/BF precision)
  jax_rfft_int8        : minimal int8 'complex16' direct-DFT variant (dtype comparison)
  rfft_trickA / trickB : real-input packing tricks (parameterized by a complex-FFT engine)

Numerics come from tpu.fft_core; the matching VMEM-resident kernels are in fft1d/pallas_fft.py.
"""
import jax
import jax.numpy as jnp

try:
    from tpu.fft_core import (
        CDTYPE,
        I8,
        I32,
        B,
        F,
        _fft_reim_real,
        dft_matrix,
    )
except ImportError:  # allow running from inside tpu/
    from fft_core import (
        CDTYPE,
        I8,
        I32,
        B,
        F,
        _fft_reim_real,
        dft_matrix,
    )

C = CDTYPE  # local alias (complex64)


# ── complex radix-B FFT: iterative + recursive ───────────────────────────────
def fft_iter(x):
    """Iterative radix-B FFT of each column of x (N, K)  (or a 1-D length-N vector)."""
    one_d = (x.ndim == 1)
    if one_d:
        x = x[:, None]
    N, K = x.shape
    cur, m, batch = x.astype(CDTYPE), N, K
    stages = []  # (B, N1) per stage, for the rebuild

    while m > B and m % B == 0:  # forward sweep: one radix-B stage per iteration
        N1 = m // B
        cur = cur.reshape(B, N1, batch)
        cur = jnp.einsum('br,rck->bck', dft_matrix(B), cur, preferred_element_type=CDTYPE)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        cur = cur * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]  # twiddle
        cur = cur.transpose(1, 0, 2).reshape(N1, B * batch)   # rotate c to axis0; fold b in
        stages.append((B, N1))
        m, batch = N1, B * batch

    cur = dft_matrix(m) @ cur   # leaf: one direct DFT over the remaining axis 0

    out, length = cur, m
    for (Bi, _N1i) in reversed(stages):  # rebuild: undo the folds in reverse order
        out = out.reshape(length, Bi, -1).reshape(length * Bi, -1)  # merge b_i back (outer)
        length = length * Bi
    return out[:, 0] if one_d else out


def fft_recur(x):
    """Recursive radix-B FFT of each column of x (N, K) (or 1-D length-N).  Recursive twin of
    fft_iter: radix-B stage -> twiddle -> transpose -> recurse on the length-N1 rows; leaf is a
    direct m-DFT matmul (m <= B or m % B != 0).  (Was small_dft_baseline in the archived code.)"""
    one_d = (x.ndim == 1)
    if one_d:
        x = x[:, None]
    out = _recur(x.astype(CDTYPE), x.shape[0])
    return out[:, 0] if one_d else out


def _recur(vecs, m):
    """Recursive radix-B DFT of the columns of vecs (m, K).  Leaf = direct m-DFT matmul."""
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=CDTYPE)
    N1, K = m // B, vecs.shape[1]
    Xb = vecs.reshape(B, N1, K)
    Yb = jnp.einsum('br,rck->bck', dft_matrix(B), Xb, preferred_element_type=CDTYPE)  # radix-B
    r = jnp.arange(B)[:, None]
    c = jnp.arange(N1)[None, :]
    tw = jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]  # twiddle W_m^(b*c)
    Zc = (Yb * tw).transpose(1, 0, 2).reshape(N1, B * K)  # twiddle, then transpose + fold b in
    Wd = _recur(Zc, N1)  # recurse on the length-N1 row DFT
    return Wd.reshape(N1, B, K).reshape(m, K)  # merge b back (outer)


@jax.jit
def fft(x):
    return fft_iter(x)


# ── real/imag rfft via the shared staged pass ────────────────────────────────
def jax_rfft(xr, half=True, dt=F):
    """1D real FFT via the SAME real/imag path as the kernel — `dt` picks precision
    (F -> complex64, BF -> bf16 'complex32').  Real (N, K) -> (yr, yi); half keeps N//2 bins."""
    return _fft_reim_real(xr, xr.shape[0], half=half, dt=dt)


_MAT_I8 = lambda A, x: jnp.matmul(A, x, preferred_element_type=I32)  # int8 x int8 -> int32

def jax_rfft_int8(xr, half=True, scale=127.0):
    """Minimal 8-bit variant for the dtype-comparison case: a DIRECT (single-matmul) real DFT
    with symmetric-quantized int8 operands, int32 accumulate.  int8 CANNOT represent the DFT
    matrix by a plain .astype like bf16 (its entries live in [-1,1] and underflow to {-1,0,1}),
    so it needs a SCALE — that scale is the FFT-algorithm knob to tune (per-column input scale
    beats the per-tensor one below; a staged int8 radix would requant each stage).  This is a
    runnable starting point: fair on I/O bytes (1 byte/component) but O(N*rows) direct DFT, so
    keep N modest (OOMs like any direct DFT at large N).  Real (N, K) -> (yr, yi)."""
    N, K = xr.shape
    rows = N // 2 if half else N
    k, j = jnp.arange(rows)[:, None], jnp.arange(N)[None, :]
    ang = -2 * jnp.pi * k * j / N
    Cr = jnp.round(jnp.cos(ang) * scale).astype(I8)  # DFT matrix -> int8, scale 1/`scale`
    Ci = jnp.round(jnp.sin(ang) * scale).astype(I8)
    s = jnp.max(jnp.abs(xr)) / scale + 1e-9  # input symmetric scale (per-tensor; your knob)
    xq = jnp.round(xr / s).astype(I8)
    deq = s / scale  # dequant: undo both operand scales after the int32 matmul
    return _MAT_I8(Cr, xq).astype(F) * deq, _MAT_I8(Ci, xq).astype(F) * deq


# ── real-input packing tricks (parameterized by a complex-FFT engine) ─────────
ENGINES = {
    "jnpfft":   lambda z: jnp.fft.fft(z, axis=0),   # opaque custom-call (no fusion)
    "fft_iter": fft_iter,                            # our matmul FFT (fusable)
}


def _mirror(Z):
    """conj(Z[(M-k) mod M]) along axis 0 — the Hermitian partner of each bin."""
    return jnp.conj(jnp.roll(Z[::-1], 1, axis=0))


def rfft_base(xr, eng):
    """xr real (N, K) -> (N//2+1, K).  eng = complex FFT along axis 0."""
    N = xr.shape[0]
    return eng(xr.astype(C))[:N // 2 + 1]


def rfft_trickA(xr, eng, ratio=0.5):
    """Pair two real signals a,b -> z=a+ib, ONE length-N complex FFT, unpack the first
    m = round(N*ratio) Hermitian rows of each (m = n_modes for the FNO).  K must be even.
    NOTE: `eng` still computes the FULL N-bin FFT; `ratio` only trims the UNPACK.  Returns (m, K)."""
    N, K = xr.shape
    a, b = xr[:, :K // 2], xr[:, K // 2:]  # pair signals: a,b are (N, K/2)
    Z = eng((a + 1j * b).astype(C))  # ONE FFT on K/2 packed cols -> (N, K/2)
    _N = int(N * ratio)
    _N = _N if _N % 2 == 0 else _N + 1  # ensure even
    M_slice = jnp.conj(jnp.roll(Z[::-1], 1, axis=0)[:_N])  # Hermitian partner (reversed slice)
    A = 0.5 * (Z[:_N] + M_slice)     # rfft(a): (_N, K/2)
    Bb = -0.5j * (Z[:_N] - M_slice)  # rfft(b): (_N, K/2)
    return jnp.concatenate([A, Bb], axis=1)  # (_N, K)


def rfft_trickB(xr, eng, ratio=0.5):
    """Pack even/odd -> z=x_even+i*x_odd, ONE HALF-length FFT, last-layer twist
    X[k]=E[k]+W_N^k O[k].  Returns the first round(N*ratio) bins.  NOTE: `eng` still returns all
    N/2 bins; `ratio` only trims the twist/unpack output."""
    N, _K = xr.shape
    h = N * ratio
    h = int(h) if h % 2 == 0 else int(h) + 1  # ensure even
    z = (xr[0::2] + 1j * xr[1::2]).astype(C)  # (h, K)
    Z = eng(z)  # K HALF-length FFTs
    M = jnp.conj(jnp.concatenate([Z[:1], Z[:0:-1]], axis=0))  # mirror, gather-free
    E = 0.5 * (Z + M)
    O = -0.5j * (Z - M)
    rows = max(1, min(int(N * ratio), h + 1))  # needed Hermitian rows
    main_rows = min(rows, h)  # k in [0, h-1]: direct index, no modulo/gather
    kf = jnp.arange(main_rows)
    tw = jnp.exp(-2j * jnp.pi * kf / N).astype(C)[:, None]
    out_main = E[:main_rows] + tw * O[:main_rows]
    if rows == h + 1:  # k = h (Nyquist) maps to index 0 with tw = -1 (static branch)
        nyq = (E[0] - O[0])[None, :]
        return jnp.concatenate([out_main, nyq], axis=0)
    return out_main


_VARIANTS = {"base": rfft_base, "trickA": rfft_trickA, "trickB": rfft_trickB}


# ── full-spectrum (N, K) reconstruction from the half, vs recompute ──────────
def _reconstruct_full(half, N):
    """(N//2+1, K) Hermitian half -> full (N, K) by conjugate-mirroring the upper bins."""
    upper = jnp.conj(half[N // 2 - 1:0:-1])          # bins N/2+1..N-1 = conj(bins N/2-1..1)
    return jnp.concatenate([half, upper], axis=0)    # (N, K)


def full_recompute(xr, eng):
    """FULL (N,K): compute ALL N bins directly (plain complex FFT, no symmetry used)."""
    return eng(xr.astype(C))


def full_via_trickA(xr, eng):
    """FULL (N,K): Trick-A rfft (half work) + conjugate reconstruct."""
    return _reconstruct_full(rfft_trickA(xr, eng), xr.shape[0])


def full_via_trickB(xr, eng):
    """FULL (N,K): Trick-B rfft (half work) + conjugate reconstruct."""
    return _reconstruct_full(rfft_trickB(xr, eng), xr.shape[0])


_FULL = {"recompute": full_recompute,
         "viaTrickA": full_via_trickA,
         "viaTrickB": full_via_trickB}
