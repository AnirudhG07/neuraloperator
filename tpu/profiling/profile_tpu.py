"""
profile_tpu.py  —  WHY does splitting beat direct/leaf on TPU wall-time?

Run on a Colab **TPU** runtime.  It profiles a CURATED set of ~20 (N, LEAF) configs
(not the full brute sweep) and, for each, reports the three numbers that explain the
"why" the math can't:

    HLO flops      — real work XLA will do        (from the compiled executable)
    bytes accessed — HBM traffic                   (compiled cost analysis)
    intensity      — flops / byte                  (roofline x-axis)
    temp MB        — peak scratch (the DFT matrix) (compiled memory analysis)
    time, TFLOP/s, GB/s  — measured on the TPU

Hypothesis to test:  a big DIRECT/LEAF DFT is MEMORY-BOUND — it must stream a huge
s×s DFT matrix (O(s²) bytes) from HBM at low arithmetic intensity, so the MXU
starves.  SPLIT keeps every matmul at 128×128 (tiny, reused across the batch), so
it runs MXU-bound.  If true, the leaf configs will show HIGH bytes / LOW intensity /
LOW achieved TFLOP/s, and the split configs the opposite — even at m=2.

────────────────────────────────────────────────────────────────────────────────
COLAB SETUP  (Runtime ▸ Change runtime type ▸ TPU):
    !pip install -q tensorboard-plugin-profile
    # upload this file, then:
    !python profile_tpu.py
    # visual trace (MXU %, op timeline, memory):
    %load_ext tensorboard
    %tensorboard --logdir ./tpu_profile
  In TensorBoard ▸ PROFILE: open `op_profile` (MXU utilisation, FLOP util per op),
  `trace_viewer` (timeline), and `memory_viewer` (who allocates the big buffers).
────────────────────────────────────────────────────────────────────────────────
"""
import time as _time

import jax
import jax.numpy as jnp

b = 128
CDTYPE = jnp.complex64
ACCUM  = jnp.complex64
BACKEND = jax.devices()[0].platform
LOGDIR = "./tpu_profile"

# Per-core bf16 TFLOP/s and HBM GB/s.  v5e ridge = 197e12/819e9 = 240 flops/byte.
# v2 ~ 46/300 ; v3 ~ 123/900 ; v4 ~ 275/1200 ; v5e ~ 197/819 (VMEM 128 MiB) ; v5p ~ 459/2765.
PEAK_TFLOPS = 197.0     # TPU v5e
PEAK_GBS    = 819.0     # TPU v5e

# ── parameterized FFT (mirrors fft_jax.small_dft, LEAF passed explicitly) ───
def dft_matrix(m):
    k = jnp.arange(m)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m).astype(CDTYPE)

def small_dft(vecs, m, LEAF):
    if m <= LEAF or m % b != 0:
        return jnp.matmul(dft_matrix(m), vecs.astype(CDTYPE), preferred_element_type=ACCUM)
    N1 = m // b
    K = vecs.shape[1]
    Xb = vecs.reshape(b, N1, K).astype(CDTYPE)
    Yb = jnp.einsum('br,rck->bck', dft_matrix(b), Xb, preferred_element_type=ACCUM)
    r = jnp.arange(b)[:, None]; c = jnp.arange(N1)[None, :]
    Zb = Yb * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, b * K)
    return small_dft(Zc, N1, LEAF).reshape(N1, b, K).reshape(m, K)

def make_fft(N, LEAF):
    @jax.jit
    def f(x):
        return small_dft(x[:, None], N, LEAF)[:, 0]
    return f

def levels(N, LEAF):
    return 1 if (N <= LEAF or N % b != 0) else 1 + levels(N // b, LEAF)

def leaf_size(N, LEAF):
    m = N
    while m > LEAF and m % b == 0:
        m //= b
    return m

def model_tiles(N, LEAF):
    """Padded-tile 'work' (the metric that anti-correlates with time)."""
    import math
    if N <= LEAF or N % b != 0:
        r = math.ceil(N / b); return r * r
    return (N // b) + b * model_tiles(N // b, LEAF)

# ── the curated ~20 configs: pairs of SPLIT (LEAF=b) vs LEAF/DIRECT ─────────
#   tag encodes N=b^p·m and what LEAF does.  DIRECT/leaf sizes kept < ~4k so the
#   DFT matrix fits HBM (full direct on huge N would OOM — mirrors run()'s "too big").
def _cfgs():
    B, B2, B3 = b, b * b, b * b * b
    return [
        # baseline atom
        (B,          B,      "b        atom  (1 tile)"),
        # 1-split: direct(mat-vec) vs split  — the clearest direct-is-slow case
        (B * 8,      B,      "b·8      SPLIT (leaf 8)"),
        (B * 8,      B * 8,  "b·8      DIRECT 1024²  (mat-vec)"),
        (B * 16,     B,      "b·16     SPLIT (leaf 16)"),
        (B * 16,     B * 16, "b·16     DIRECT 2048²  (mat-vec)"),
        # 2-split disagreement region (work says leaf, time says split)
        (B2 * 2,     B,      "b²·2     SPLIT (leaf 2)"),
        (B2 * 2,     B * 2,  "b²·2     leaf 256   (m=2, the tie case)"),
        (B2 * 4,     B,      "b²·4     SPLIT (leaf 4)"),
        (B2 * 4,     B * 4,  "b²·4     leaf 512"),
        (B2 * 8,     B,      "b²·8     SPLIT (leaf 8)"),
        (B2 * 8,     B * 8,  "b²·8     leaf 1024  (4.5× gap)"),
        (B2 * 12,    B,      "b²·12    SPLIT (leaf 12)"),
        (B2 * 12,    B * 12, "b²·12    leaf 1536  (m=12 boundary)"),
        # clean powers (work & time agree on split)
        (B2,         B,      "b²       clean split (leaf 128)"),
        (B3,         B,      "b³       clean split (leaf 128)"),
        # 3-split large N
        (B3 * 2,     B,      "b³·2     SPLIT (leaf 2)"),
        (B3 * 2,     B * 2,  "b³·2     leaf 256"),
        (B3 * 4,     B,      "b³·4     SPLIT (leaf 4)"),
        (B3 * 4,     B * 4,  "b³·4     leaf 512"),
        (B3 * 8,     B,      "b³·8     SPLIT (leaf 8)"),
        (B3 * 8,     B * 8,  "b³·8     leaf 1024"),
    ]

MAX_LEAF = 4096                       # skip configs whose DFT matrix is too big to hold

def _cost(comp):
    ca = comp.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    return ca or {}

def analyze(N, LEAF, reps=20):
    lsz = leaf_size(N, LEAF)
    if lsz > MAX_LEAF:
        return None                   # would OOM on the DFT matrix
    f = make_fft(N, LEAF)
    x = jnp.arange(1, N + 1, dtype=jnp.float32).astype(CDTYPE)
    comp = f.lower(x).compile()
    ca = _cost(comp)
    flops = float(ca.get("flops", 0.0) or 0.0)
    byts  = float(ca.get("bytes accessed", 0.0) or 0.0)
    try:
        ma = comp.memory_analysis()
        temp = float(getattr(ma, "temp_size_in_bytes", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        temp = 0.0
    f(x).block_until_ready()          # compile + warm
    ts = []
    for _ in range(reps):
        t0 = _time.perf_counter(); f(x).block_until_ready()
        ts.append(_time.perf_counter() - t0)
    ts.sort(); t = ts[len(ts) // 2]
    return {"N": N, "LEAF": LEAF, "levels": levels(N, LEAF), "leaf": lsz,
            "tiles": model_tiles(N, LEAF), "flops": flops, "bytes": byts, "temp": temp,
            "time": t, "tflops": flops / t / 1e12 if t else 0,
            "gbs": byts / t / 1e9 if t else 0,
            "ai": flops / byts if byts else 0}

def _h(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= u: return f"{x/u:.1f}{s}"
    return f"{x:.0f}"

def main():
    print(f"backend = {BACKEND.upper()}   (roofline peaks assumed: "
          f"{PEAK_TFLOPS:.0f} TFLOP/s, {PEAK_GBS:.0f} GB/s — EDIT for your TPU gen)")
    if BACKEND != "tpu":
        print("!! not a TPU — the memory-bound story is TPU-specific; run on a Colab TPU.")
    ridge = PEAK_TFLOPS * 1e12 / (PEAK_GBS * 1e9)
    print(f"   roofline ridge point ≈ {ridge:.0f} flops/byte  "
          f"(intensity below this = MEMORY-BOUND)\n")

    hdr = (f"{'config':<34}{'lvls':>4}{'leaf':>6}{'model':>7}{'HLOflops':>9}"
           f"{'bytes':>8}{'AI':>7}{'tempMB':>8}{'time':>10}{'TFLOP/s':>9}{'GB/s':>8}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for N, LEAF, tag in _cfgs():
        r = analyze(N, LEAF)
        if r is None:
            print(f"{tag:<34}{'':>4}{'':>6}  (skipped — DFT matrix > {MAX_LEAF}² too big)")
            continue
        rows.append((tag, r))
        tms = f"{r['time']*1e6:7.1f}us" if r['time'] < 1e-3 else f"{r['time']*1e3:7.2f}ms"
        bound = "  MEM" if (r['ai'] and r['ai'] < ridge) else "  cmp"
        print(f"{tag:<34}{r['levels']:>4}{r['leaf']:>6}{_h(r['tiles']*b*b):>7}"
              f"{_h(r['flops']):>9}{_h(r['bytes']):>8}{r['ai']:>7.0f}"
              f"{r['temp']/1e6:>8.1f}{tms:>10}{r['tflops']:>9.1f}{r['gbs']:>8.0f}{bound}")

    print("\nRead it like this:")
    print("  • On TPU the whole FFT is MEMORY-BOUND (AI << ridge for every row), so the")
    print("    winner is decided by TOTAL BYTES MOVED, not MXU occupancy.")
    print("  • Compare each SPLIT row with its leaf/DIRECT partner at the same N: the leaf")
    print("    moves far more bytes (dense s×s DFT is O(s²)); split is Cooley-Tukey O(s·b).")
    print("  • time tracks bytes: e.g. b²·8 leaf moves ~8× the bytes of split -> ~4.5× time.")
    print("  • Everything sits well below peak TFLOP/s & GB/s -> small-op/overhead bound;")
    print("    real headroom (bf16, fusion, fewer transposes) but orthogonal to LEAF choice.")

    # ── visual profiler trace for TensorBoard (op_profile / trace_viewer) ──
    print(f"\ncapturing jax.profiler trace -> {LOGDIR}  (open with TensorBoard PROFILE tab)")
    with jax.profiler.trace(LOGDIR):
        for tag, r in rows:
            f = make_fft(r["N"], r["LEAF"])
            x = jnp.arange(1, r["N"] + 1, dtype=jnp.float32).astype(CDTYPE)
            with jax.named_scope(tag.split()[0] + ("_split" if r["LEAF"] == b else "_leaf")):
                for _ in range(5):
                    f(x).block_until_ready()
    print(f"done.  In Colab:  %load_ext tensorboard  ;  %tensorboard --logdir {LOGDIR}")

if __name__ == "__main__":
    main()
