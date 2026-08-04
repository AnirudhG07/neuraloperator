"""
FFT variants for TPU A/B testing  —  radix-B Cooley-Tukey, split all the way down.

Four variants, to isolate two independent questions:

  Q_transpose : keep the explicit corner-turn, or contract the inner axis in place?
  Q_fuse      : leave the twiddle as a separate elementwise pass, or bake it into the
                inner DFT matrix?

      variant                | inner DFT        | twiddle
      -----------------------|------------------|---------------------------
      baseline               | transpose+recurse| separate elementwise pass
      no_transpose           | in-place einsum  | separate elementwise pass
      fused                  | transpose+recurse| baked into the matrix (Ctwid)
      fused_no_transpose     | in-place einsum  | baked into the matrix (Ctwid)

TWO MEANINGS OF "FUSE" — do not confuse them:
  1. XLA KERNEL FUSION (automatic).  XLA merges adjacent ops into ONE kernel so the
     intermediate never leaves VMEM — no HBM round-trip.  You don't write it; you ENABLE
     it by keeping ops simple/branch-free and adjacent under one jit.  This is the thing
     that actually saves the HBM traffic.  You verify it AFTER compiling (see `inspect`).
  2. ALGEBRAIC FOLDING (manual) — what the `fused` variants do: bake W_m into the DFT
     matrix.  This is NOT kernel fusion; it changes the math.  Here it BACKFIRES: the
     folded matrix Ctwid is (B, N1, N1) and b-dependent, i.e. ~N1x bigger than the (B,N1)
     twiddle vector and a *batched* matmul instead of a shared one -> MORE bytes.  The
     `inspect` table below shows this directly (bytes accessed goes UP).

HOW DO YOU KNOW FUSION HAPPENED?  `inspect()` compiles each variant and reports:
    - device-op counts from the optimized HLO (dot / fusion / transpose / copy):
      fewer transpose+copy = fewer materialized HBM round-trips.
    - XLA `cost_analysis` "bytes accessed": the HBM-traffic estimate. LOWER = better.
  These are PROXIES on CPU/GPU; the definitive read is on the TPU via
  `XLA_FLAGS=--xla_dump_to=/tmp/hlo` (inspect the *.after_optimizations.txt) or the
  profiler's op_profile / memory_viewer (actual kernels + real HBM bytes).

Pure JAX; jax.jit on the top-level entry point.
"""
import re
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# ─── config ────────────────────────────────────────────────────────────────
B = 128
CDTYPE = jnp.complex64
ACCUM  = jnp.complex64


# ─── shared building blocks ────────────────────────────────────────────────
def dft_matrix(m):
    """m x m DFT coefficient matrix, C[r,c] = W_m^(r*c)."""
    k = jnp.arange(m)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m).astype(CDTYPE)


def _radix_b_dft(vecs, m):
    """Peel one radix-B stage: reshape to (B,N1,K) and do the length-B column DFT.
    Returns (Yb, N1, K) with Yb of shape (B, N1, K)."""
    N1 = m // B
    K = vecs.shape[1]
    Xb = vecs.reshape(B, N1, K)
    Yb = jnp.einsum('br,rck->bck', dft_matrix(B), Xb, preferred_element_type=ACCUM)
    return Yb, N1, K


def _deep_split(Yb, m, N1, K, recurse):
    """Shared DEEP path (N1 > B): apply the twiddle as a separate elementwise pass,
    transpose to put the inner axis first, and recurse.  All four variants share this
    for the deep case — only the DIRECT case (N1 <= B) differs between them."""
    r = jnp.arange(B)[:, None]
    c = jnp.arange(N1)[None, :]
    ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)
    tw = jnp.exp(1j * ang).astype(CDTYPE)[:, :, None]        # W_m^(b*c)
    Zb = Yb * tw
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)     # corner-turn -> (N1, B*K)
    Wd = recurse(Zc, N1)
    return Wd.reshape(N1, B, K).reshape(m, K)


# ═══════════════════════════════════════════════════════════════════════════
#  variant 1 — BASELINE : separate twiddle, explicit transpose + recurse
# ═══════════════════════════════════════════════════════════════════════════
def small_dft_baseline(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)   # leaf
    Yb, N1, K = _radix_b_dft(vecs, m)
    return _deep_split(Yb, m, N1, K, small_dft_baseline)


# ═══════════════════════════════════════════════════════════════════════════
#  variant 2 — NO_TRANSPOSE : separate twiddle, inner DFT contracted IN PLACE
#  (still needs ONE transpose at the end to get output order d*B+b — see note)
# ═══════════════════════════════════════════════════════════════════════════
def small_dft_no_transpose(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)   # leaf
    Yb, N1, K = _radix_b_dft(vecs, m)
    if N1 <= B:                                              # direct inner DFT
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)
        Zb = Yb * jnp.exp(1j * ang).astype(CDTYPE)[:, :, None]
        # contract c (N1 axis) in place instead of transposing first:
        Wd = jnp.einsum('bck,dc->bdk', Zb, dft_matrix(N1), preferred_element_type=ACCUM)
        # NOTE: this transpose did NOT go away — it just moved here.  freq = d*B+b needs
        # d outer / b inner, but Wd is (b,d,k), so we still corner-turn once.
        return jnp.transpose(Wd, (1, 0, 2)).reshape(m, K)
    return _deep_split(Yb, m, N1, K, small_dft_no_transpose)  # deep: same as baseline


# ═══════════════════════════════════════════════════════════════════════════
#  variant 3 — FUSED : twiddle BAKED into the inner DFT matrix (algebraic fold)
#  W[b,d] = sum_c Ctwid_b[d,c] * Y[b,c],  Ctwid_b[d,c] = C_N1[d,c] * W_m^(b*c)
#  WARNING: Ctwid is (B,N1,N1) and b-dependent -> a big, batched operand.  This is
#  the "fold, don't fuse" trap; `inspect` shows bytes accessed goes UP, not down.
# ═══════════════════════════════════════════════════════════════════════════
def small_dft_fused(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)   # leaf
    Yb, N1, K = _radix_b_dft(vecs, m)
    if N1 <= B:
        Ctwid = _twiddled_inner_matrix(N1, m)               # (B, N1, N1)
        Wd = jnp.einsum('bdc,bck->bdk', Ctwid, Yb, preferred_element_type=ACCUM)
        return jnp.transpose(Wd, (1, 0, 2)).reshape(m, K)
    return _deep_split(Yb, m, N1, K, small_dft_fused)        # deep: fall back to separate


# ═══════════════════════════════════════════════════════════════════════════
#  variant 4 — FUSED_NO_TRANSPOSE : Q2 + Q3 together
# ═══════════════════════════════════════════════════════════════════════════
def small_dft_fused_no_transpose(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)   # leaf
    Yb, N1, K = _radix_b_dft(vecs, m)
    if N1 <= B:
        Ctwid = _twiddled_inner_matrix(N1, m)               # (B, N1, N1)
        Wd = jnp.einsum('bdc,bck->bdk', Ctwid, Yb, preferred_element_type=ACCUM)
        return jnp.transpose(Wd, (1, 0, 2)).reshape(m, K)
    return _deep_split(Yb, m, N1, K, small_dft_fused_no_transpose)


def _twiddled_inner_matrix(N1, m):
    """(B, N1, N1) inner-DFT matrix with the level twiddle W_m^(b*c) baked in per b."""
    d = jnp.arange(N1)[:, None]
    c = jnp.arange(N1)[None, :]
    C1 = jnp.exp(-2j * jnp.pi * (d * c) / N1)                # (N1, N1) inner DFT
    b = jnp.arange(B)[:, None, None]
    cc = jnp.arange(N1)[None, None, :]
    tw = jnp.exp(-2j * jnp.pi * (b * cc).astype(jnp.float32) / m)   # (B, 1, N1)
    return (C1[None] * tw).astype(CDTYPE)                    # (B, N1, N1)


# ─── top-level entry point ─────────────────────────────────────────────────
_VARIANTS = {
    'baseline':           small_dft_baseline,
    'no_transpose':       small_dft_no_transpose,
    'fused':              small_dft_fused,
    'fused_no_transpose': small_dft_fused_no_transpose,
}


@partial(jax.jit, static_argnums=(1,))
def fft_dispatch(x, variant):
    N = x.shape[0]
    v = x.astype(CDTYPE)[:, None]
    return _VARIANTS[variant](v, N)[:, 0]


# ─── correctness vs numpy ──────────────────────────────────────────────────
def correctness_check():
    print(f"radix-B FFT variants  B={B}   (correctness vs numpy.fft)")
    for variant in _VARIANTS:
        print(f"\n=== {variant} ===")
        for N in [4, 64, 128, 256, 1024, 16384, 16384 * 4]:
            x = np.arange(1, N + 1, dtype=np.float32)
            try:
                X = np.asarray(fft_dispatch(jnp.asarray(x), variant))
                ref = np.fft.fft(x)
                rel = float(np.max(np.abs(X - ref))) / float(np.max(np.abs(ref)))
                print(f"  N={N:<7} rel_err={rel:.2e}  {'OK ' if rel < 1e-4 else 'BAD'}")
            except Exception as e:
                print(f"  N={N:<7} ERROR: {str(e)[:60]}")


# ─── DID FUSION HAPPEN?  op-count + HBM bytes from the compiled HLO ─────────
def inspect(N=16384, variants=None):
    """Compile each variant and report what XLA actually built.

    dot/fusion/transpose/copy : device-op counts in the optimized HLO.  Every
        transpose/copy is a materialized HBM round-trip; fewer = more got fused away.
    bytes : XLA cost_analysis 'bytes accessed' — the HBM-traffic estimate. THE number
        to watch: real kernel fusion makes it drop; the `fused` fold makes it RISE.

    CPU/GPU HLO here is a proxy for the shape of the graph; the definitive TPU read is
    XLA_FLAGS=--xla_dump_to=... or the profiler op_profile/memory_viewer.
    """
    variants = variants or list(_VARIANTS)
    x = jnp.asarray(np.random.randn(N).astype(np.float32))
    dev = jax.devices()[0].platform
    print(f"\n{'='*94}\nFUSION INSPECTION   N={N:,}   device={dev.upper()}   "
          f"(bytes = XLA HBM estimate; lower = fewer round-trips)\n{'='*94}")
    print(f"{'variant':>20}{'dot':>5}{'fusion':>7}{'transpose':>10}{'copy':>6}"
          f"{'exp':>5}{'select':>7}{'bytes':>12}{'flops':>11}{'AI':>6}")
    toks = ['dot', 'fusion', 'transpose', 'copy', 'exponential', 'select']
    for v in variants:
        comp = jax.jit(lambda x, v=v: fft_dispatch(x, v)).lower(x).compile()
        hlo = comp.as_text()
        cnt = {t: len(re.findall(r'\b' + t + r'\(', hlo)) for t in toks}
        ca = comp.cost_analysis() or {}
        by = float(ca.get('bytes accessed', 0) or 0)
        fl = float(ca.get('flops', 0) or 0)
        print(f"{v:>20}{cnt['dot']:>5}{cnt['fusion']:>7}{cnt['transpose']:>10}"
              f"{cnt['copy']:>6}{cnt['exponential']:>5}{cnt['select']:>7}"
              f"{by:>12.2e}{fl:>11.2e}{fl/max(by,1):>6.1f}")


# ─── wall-clock benchmark (run on TPU) ─────────────────────────────────────
def benchmark(Ns=None, variants=None, reps=50):
    Ns = Ns or [1024, 16384, 65536, 262144, 1048576, 4194304]
    variants = variants or list(_VARIANTS)
    print(f"\n{'='*90}\nBENCHMARK   device={jax.devices()[0].platform.upper()}   "
          f"(median of {reps}, us)\n{'='*90}")
    print(f"{'N':>10}" + "".join(f"{v[:17]:>19}" for v in variants))
    for N in Ns:
        x = jnp.asarray(np.random.randn(N).astype(np.float32))
        row = f"{N:>10}"
        for v in variants:
            try:
                f = lambda x, v=v: fft_dispatch(x, v)
                f(x).block_until_ready()
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    f(x).block_until_ready()
                    ts.append(time.perf_counter() - t0)
                row += f"{np.median(ts)*1e6:>16.1f}us"
            except Exception as e:
                row += f"{'ERR':>19}"
        print(row)


if __name__ == "__main__":
    correctness_check()
    inspect()
    benchmark()
