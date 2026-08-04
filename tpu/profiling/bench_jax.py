"""
bench_jax.py  —  WHEN TO SPLIT, and at which LEAF = k*b?   (op-count + REAL time)

One sweep: vary N across split-depths (0 -> 1 -> 2 -> 3 radix-b splits, many points
each) and, for every N, vary the LEAF threshold over LEAF = k*b, k = 1..32 (+ direct).
Splitting stops once a subproblem <= LEAF.  For each (N, LEAF) we report:

    work  = multiply-steps after rounding every matmul UP to the array width b
            (exact, hardware-independent; LOWER = fewer flops)
    time  = median of REPS real, block_until_ready'd runs on the current JAX backend

Goal: find the smallest LEAF = k*b at which splitting stops paying off — i.e. the
best place to stop.  Math says the leaf size wants to be s* = b/ln b ≈ 27, but on a
b=128 array anything <= 128 is padded to a full tile, so k=1 (LEAF=b) should already
be optimal; the time column tests whether a bigger k ever wins by cutting stages.

IMPORTANT:  CPU wall-clock LIES about the TPU 128x128 cliff (no systolic array,
BLAS-dominated) and can invert the ranking.  On CPU trust `work` for the trend;
run on TPU for the real `time`.  b=128: b^2=16,384  b^3=2,097,152
"""
import math
import time as _time

import jax
import jax.numpy as jnp

b = 128
CDTYPE = jnp.complex64
ACCUM  = jnp.complex64

BACKEND = jax.devices()[0].platform  # 'cpu' | 'gpu' | 'tpu'

# ── real JAX FFT, with LEAF as an explicit argument ────────────────────────
def dft_matrix(m):
    k = jnp.arange(m)
    W = jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m)
    return W.astype(CDTYPE)

def small_dft(vecs, m, LEAF):
    """m-point DFT of each length-m column of vecs (m, K); peel radix-b, recurse to LEAF."""
    if m <= LEAF or m % b != 0:
        return jnp.matmul(dft_matrix(m), vecs.astype(CDTYPE), preferred_element_type=ACCUM)
    N1 = m // b
    K = vecs.shape[1]
    Xb = vecs.reshape(b, N1, K).astype(CDTYPE)
    Cb = dft_matrix(b)
    Yb = jnp.einsum('br,rck->bck', Cb, Xb, preferred_element_type=ACCUM)
    r = jnp.arange(b)[:, None]
    c = jnp.arange(N1)[None, :]
    Tw = jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)
    Zb = Yb * Tw[:, :, None]
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, b * K)
    Wd = small_dft(Zc, N1, LEAF)
    out = Wd.reshape(N1, b, K)
    return out.reshape(m, K)

def make_fft(N, LEAF):
    """Jitted N-point DFT (N a multiple of b) using the given LEAF threshold."""
    @jax.jit
    def f(x):
        return small_dft(x[:, None], N, LEAF)[:, 0]
    return f

# ── analytic TPU-padded work model (matches the FFT above) ─────────────────
def tpu_matmul_cost(R, C):
    return math.ceil(R / b) * math.ceil(C / b) * b * b

def split_cost(N, LEAF):
    if N <= LEAF or N % b != 0:
        return tpu_matmul_cost(N, N)          # direct leaf: N x N matmul, padded
    return (N // b) * tpu_matmul_cost(b, b) + b * split_cost(N // b, LEAF)

def count_levels(N, LEAF):
    if N <= LEAF or N % b != 0:
        return 1
    return 1 + count_levels(N // b, LEAF)

def leaf_size(N, LEAF):
    """Size of the bottom (direct) leaf that this (N, LEAF) actually produces."""
    m = N
    while m > LEAF and m % b == 0:
        m //= b
    return m

# ── real timing (blocked) ──────────────────────────────────────────────────
MEM_LEAF_CAP = 4096          # skip actually running leaves bigger than this (matrix too big)
REPS = 25

def time_fft(N, LEAF, reps=REPS):
    """Median wall-clock of a blocked N-point FFT, or None if the leaf is too big to run."""
    if leaf_size(N, LEAF) > MEM_LEAF_CAP:
        return None
    f = make_fft(N, LEAF)
    x = jnp.arange(1, N + 1, dtype=jnp.float32).astype(CDTYPE)
    f(x).block_until_ready()                 # compile + warm up
    ts = []
    for _ in range(reps):
        t0 = _time.perf_counter()
        f(x).block_until_ready()
        ts.append(_time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]

def human(x):
    for u, s in [(1e18,"E"),(1e15,"P"),(1e12,"T"),(1e9,"G"),(1e6,"M"),(1e3,"K")]:
        if x >= u:
            return f"{x/u:.1f}{s}"
    return str(int(x))

def fmt_t(t):
    if t is None:
        return "   (too big)"
    if t >= 1e-3:
        return f"{t*1e3:8.2f} ms"
    return f"{t*1e6:8.1f} us"

# ── N values, grouped by how many radix-b splits they can take (LEAF=b) ─────
def gen_N_groups():
    """
    Lots of N, from 0-split (N<=b) up through 4-split (N~b^4).
    Each group is (label, [N, ...]).  Many points per group for a conclusive trend.
    """
    k_small = [2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 20, 24, 25, 26, 27, 28, 32, 40, 48, 64, 96, 128]
    return [
        ("0 splits  (N <= b, direct only)", [8, 16, 32, 48, 64, 96, 128]),
        ("1 split   (N = b   * k)",        [b * k       for k in k_small]),
        ("2 splits  (N = b^2 * k)",        [b * b * k   for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]]),
        ("3 splits  (N = b^3 * k)",        [b * b * b * k for k in [1, 2, 4, 8, 16, 32]]),
        ("4 splits  (N = b^4 * k)",        [b * b * b * b * k for k in [1, 2, 4, 8, 16, 32]]),
    ]

# leaf thresholds to try: LEAF = k*b, k = 1..32  (plus direct = no split)
LEAF_KS = list(range(1, 33))

def run():
    print("=" * 96)
    print(f"WHEN TO SPLIT / BEST LEAF   b={b}   backend={BACKEND.upper()}   "
          f"(time = median of {REPS} blocked runs)")
    print("  For each N we sweep LEAF = k*b (k=1..32) + direct.  Splitting stops when a")
    print("  subproblem <= LEAF.  work = MXU-padded flops (exact).  LOWER work/time = better.")
    print("  CPU time is indicative only; the 128x128 cliff is real only on TPU. Trust work.")
    print("=" * 96)

    for label, Ns in gen_N_groups():
        print(f"\n{'#'*96}\n#  {label}\n{'#'*96}")
        for N in Ns:
            # build candidate LEAFs (k*b <= N) plus 'direct' (LEAF=N, no split)
            cand = [k * b for k in LEAF_KS if k * b <= N] + [N]
            cand = sorted(set(cand))

            # dedup by resulting split STRUCTURE (levels, leaf_size): identical graph => time once
            tcache = {}
            rows = []
            best_w = (float("inf"), None)
            best_t = (float("inf"), None)
            for LEAF in cand:
                lvl = count_levels(N, LEAF)
                lsz = leaf_size(N, LEAF)
                w   = split_cost(N, LEAF)
                key = (lvl, lsz)
                if key not in tcache:
                    tcache[key] = time_fft(N, LEAF)      # only distinct graphs are timed
                t = tcache[key]
                kfac = "direct" if LEAF == N and lvl == 1 and lsz == N else f"{LEAF // b}b"
                rows.append((LEAF, kfac, lvl, lsz, w, t))
                if w < best_w[0]:
                    best_w = (w, LEAF)
                if t is not None and t < best_t[0]:
                    best_t = (t, LEAF)

            print(f"\n  N = {N:,}   (max splits @ LEAF=b: {count_levels(N, b) - 1})")
            print(f"    {'LEAF':>9} {'=k*b':>6} {'levels':>6} {'leaf_sz':>8} "
                  f"{'work':>9} {'time':>12}   best")
            for LEAF, kfac, lvl, lsz, w, t in rows:
                marks = []
                if LEAF == best_w[1]:
                    marks.append("min-WORK")
                if LEAF == best_t[1]:
                    marks.append("min-TIME")
                print(f"    {LEAF:>9,} {kfac:>6} {lvl:>6} {lsz:>8} "
                      f"{human(w):>9} {fmt_t(t):>12}   {' '.join(marks)}")

# ── parameterized work model (explicit b, LEAF) for plotting ───────────────
def work_of(N, LEAF, bb):
    """MXU-padded work for an N-point FFT, array width bb, leaf threshold LEAF."""
    if N <= LEAF or N % bb != 0:
        r = math.ceil(N / bb)
        return r * r * bb * bb                 # direct leaf: N x N matmul, padded to tiles
    return (N // bb) * bb * bb + bb * work_of(N // bb, LEAF, bb)

def plot(path="tpu/work_vs_N.png"):
    """One figure, two panels: work vs N, (left) varying LEAF at b=128, (right) varying b."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Okabe-Ito colorblind-safe categorical palette
    C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

    # dense N values: many multiples of 128 from b .. b^4 (log-spaced, so tails show up)
    lo, hi = b, b ** 4
    raw = [lo * (hi / lo) ** (i / 140) for i in range(141)]
    Ns = sorted({max(b, round(v / b) * b) for v in raw})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ── Panel A: WHICH LEAF is best?  work vs LEAF/b, one curve per N ───────
    # Many N (b^2 .. b^5, HUGE).  For each N we sweep LEAF = k*b and normalise by
    # that N's own minimum, so every curve bottoms at 1.0 and the RING sits at the
    # optimal LEAF for that N.  log-y so the 20x spikes don't crush the small ones.
    import matplotlib.colors as mcolors

    ks = list(range(1, 41))                                # LEAF/b = 1..40
    Ns_leaf = sorted({b ** p * m
                      for p in (2, 3, 4) for m in (2, 3, 4, 6, 8, 10, 12, 16, 20)}
                     | {b ** 5 * m for m in (4, 16)})      # + a couple HUGE ones
    norm = mcolors.Normalize(vmin=math.log10(min(Ns_leaf)),
                             vmax=math.log10(max(Ns_leaf)))
    cmap = plt.get_cmap("viridis")
    for N in Ns_leaf:
        ys = [work_of(N, k * b, b) for k in ks]
        ymin = min(ys)
        yr = [y / ymin for y in ys]                        # normalised (1.0 = best)
        col = cmap(norm(math.log10(N)))
        axL.plot(ks, yr, color=col, lw=1.3, alpha=0.85)
        j = min(range(len(ks)), key=lambda t: ys[t])       # argmin -> optimal LEAF/b
        axL.plot(ks[j], yr[j], marker="o", ms=8, mfc="white",
                 mec=col, mew=1.8, zorder=5)                # ring on the minimum
    axL.axvline(b ** 0.5, color="crimson", ls="--", lw=1.3, zorder=4)
    axL.set_yscale("log")
    axL.set_ylim(0.98, 3.0)                                 # zoom on the 1.0 region; big spikes run off-top
    axL.text(b ** 0.5 + 0.4, 2.85, r"$\sqrt{b}\approx12$  (ceiling on best LEAF)",
             color="crimson", fontsize=9, va="top", ha="left")
    axL.set_title("which LEAF is best?   work vs LEAF   (b = 128,  20 values of N)", fontsize=11)
    axL.set_xlabel("LEAF  (in units of b,  i.e. LEAF = k·b)")
    axL.set_ylabel("work / best work for that N   (1.0 = optimal;  zoomed, spikes off-top)")
    axL.grid(True, which="both", alpha=0.22, lw=0.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axL, pad=0.02)
    cb.set_label("N  (log₁₀)", fontsize=9)

    # ── Panel B: fix LEAF=b, vary array width b (unchanged) ─────────────────
    for i, bb in enumerate([64, 128, 256]):
        Nb = [N for N in Ns if N % bb == 0]
        ys = [work_of(N, bb, bb) for N in Nb]              # LEAF = b (full split)
        axR.plot(Nb, ys, color=C[i], lw=1.8, marker="o", ms=3, label=f"b = {bb}")
    axR.set_title("work vs N  —  varying array width b   (LEAF = b)", fontsize=11)
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("N  (transform length)")
    axR.set_ylabel("work  (MXU-padded multiply-steps)")
    axR.grid(True, which="major", alpha=0.25, lw=0.6)
    for p, lbl in [(2, "b²"), (3, "b³"), (4, "b⁴")]:
        axR.axvline(b ** p, color="0.6", ls=":", lw=0.8)
        axR.text(b ** p, axR.get_ylim()[0], f" {lbl}", color="0.45",
                 fontsize=8, va="bottom", ha="left")
    axR.legend(frameon=False, fontsize=9, loc="upper left")
    axR.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: human(v)))

    fig.suptitle("TPU FFT work: the best LEAF (left) and the effect of array width b (right)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    print(f"saved {path}")

def plot_time(path="tpu/time_vs_leaf.png"):
    """
    Same 'which LEAF is best?' question, but with MEASURED time instead of the
    analytic work model.  x = LEAF/b, y = time normalised to each N's own best,
    ring = the LEAF that was actually fastest.  N kept modest so CPU can run it
    (huge N / big leaves are skipped by the memory cap).  CPU time is indicative:
    it has NO 128x128 systolic cliff, so it will NOT reproduce the sqrt(b) ceiling
    the work model shows — that is exactly the point of comparing the two figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    ks = list(range(1, 21))                                # LEAF/b = 1..20
    Ns = [b * b * m for m in (2, 4, 8, 16)] + [b ** 3 * m for m in (1, 2)]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for i, N in enumerate(Ns):
        # measure once per DISTINCT split structure, then map every k to it
        tcache, pts = {}, []
        for k in ks:
            LEAF = k * b
            key = (count_levels(N, LEAF), leaf_size(N, LEAF))
            if key not in tcache:
                tcache[key] = time_fft(N, LEAF, reps=9)
            t = tcache[key]
            if t is not None:
                pts.append((k, t))
        if not pts:
            continue
        tmin = min(t for _, t in pts)
        xs = [k for k, _ in pts]
        ys = [t / tmin for _, t in pts]
        col = C[i % len(C)]
        ax.plot(xs, ys, color=col, lw=1.6, marker="o", ms=3.5,
                label=f"N = {human(N)}")
        j = min(range(len(pts)), key=lambda t: pts[t][1])  # fastest LEAF
        ax.plot(xs[j], ys[j], marker="o", ms=11, mfc="white",
                mec=col, mew=2, zorder=5)                  # ring on the minimum

    ax.axvline(b ** 0.5, color="crimson", ls="--", lw=1.3, zorder=4)
    ax.set_ylim(0.975, 1.4)                                 # zoom on the 1.0 region; big jumps run off-top
    ax.text(b ** 0.5 + 0.3, 1.39, r" $\sqrt{b}\approx12$ (work-model optimum)",
            color="crimson", fontsize=9, va="top", ha="left")
    ax.set_title(f"which LEAF is fastest?   MEASURED time vs LEAF   "
                 f"(b={b}, backend={BACKEND.upper()})", fontsize=11)
    ax.set_xlabel("LEAF  (in units of b,  i.e. LEAF = k·b)")
    ax.set_ylabel("time / fastest time for that N   (1.0 = fastest;  zoomed, jumps off-top)")
    ax.grid(True, which="both", alpha=0.22, lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right", title="ring = fastest LEAF")
    fig.suptitle("TIME view — compare against the work view (work_vs_N.png)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")

# A sub-DFT is one recursive call of small_dft: an s-point DFT it must either SPLIT
# (peel a radix-b stage) or make a LEAF (do directly as one padded matmul).  The whole
# tuning question is: at what sub-DFT size s = k*b should small_dft STOP splitting?
# That size is the LEAF threshold.  s and LEAF are the same thing, measured in b.

def _kstar(bb):
    """Exact crossover: a sub-DFT of size s = k*bb is cheaper to SPLIT once when k > k*."""
    return (1 + math.sqrt(1 + 4 * bb)) / 2

def _cap(bb):
    """Largest LEAF (in units of bb) that is still a LEAF rather than a split.
    A sub-DFT of size k*bb costs k^2 tiles direct vs (k+bb) tiles split-once, so the
    cap is the biggest k with k^2 <= k+bb  (below it: leaf; above it: split)."""
    k = 0
    while (k + 1) ** 2 <= (k + 1) + bb:
        k += 1
    return k

def prove(path="tpu/cap_vs_b.png"):
    """
    Proof of the two claims, WORK and TIME, over a large N range:
      (1) a LEAF below the cap (leaf the small sub-DFTs) beats LEAF=b (split every
          sub-DFT all the way down),
      (2) the cap is sqrt(b) — it MOVES with the array width b, so 12 is not magic.
    Claim 2's cap is a work-model result (exact, hardware-independent); CPU time has
    no 128x128 cliff so it can only confirm claim 1 here — the cap-by-time needs TPU.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    LEAF_CAP = _cap(b) * b                                  # = 11*b = 1408 for b=128

    # ── CLAIM 2 (WORK, exact): the cap = sqrt(b), across array widths b ─────
    print("=" * 84)
    print("CLAIM 2 — the LEAF cap is sqrt(b), NOT a fixed 12  (work model, exact):")
    print("=" * 84)
    print(f"  {'b':>6} {'sqrt(b)':>9} {'k* (exact)':>11} {'cap=floor(k*)':>13}   check")
    bs = [16, 32, 64, 128, 256, 512, 1024, 4096]
    caps = [_cap(x) for x in bs]
    for bb, cap in zip(bs, caps):
        tick = "cap == round(sqrt(b))" if cap == round(math.sqrt(bb)) else ""
        print(f"  {bb:>6} {math.sqrt(bb):>9.2f} {_kstar(bb):>11.2f} {cap:>13}   {tick}")

    # ── CLAIM 1 (WORK + TIME): LEAF=b (split all) vs LEAF=cap, LARGE N range ─
    print("\n" + "=" * 84)
    print(f"CLAIM 1 — LEAF=cap ({_cap(b)}·b) vs LEAF=b (split everything)   "
          f"b={b}  backend={BACKEND.upper()}")
    print("  ratio = LEAF=b / LEAF=cap  (>1 means splitting-everything is WORSE)")
    print("=" * 84)
    print(f"  {'N':>18} {'N=b^p·m':>9} {'work@cap':>9} {'work×':>7} "
          f"{'time@cap':>11} {'time×':>7}")
    TIME_MAX_N = 9_000_000                                  # only time what CPU can run fast
    combos = [(2, 1), (2, 2), (2, 4), (2, 8), (3, 1), (3, 2), (3, 4), (3, 8),
              (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]       # b^2 .. b^5  (16K .. 6.9e10)
    rows = []                                              # (N, work_ratio, time_ratio) for plotting
    for p, m in sorted(combos, key=lambda pm: b ** pm[0] * pm[1]):
        N = b ** p * m
        w_all, w_cap = work_of(N, b, b), work_of(N, LEAF_CAP, b)
        wr = w_all / w_cap
        if N <= TIME_MAX_N:
            t_all, t_cap = time_fft(N, b, reps=7), time_fft(N, LEAF_CAP, reps=7)
            tr = (t_all / t_cap) if (t_all and t_cap) else None
        else:
            t_cap = tr = None
        rows.append((N, wr, tr))
        ts = f"{tr:6.2f}x" if tr else "     —"
        tc = fmt_t(t_cap) if t_cap else "      —"
        print(f"  {N:>18,} {f'b^{p}·{m}':>9} {human(w_cap):>9} {wr:>6.2f}x "
              f"{tc:>11} {ts:>7}")

    # ── FIGURE: (L) WORK crossover -> cap=sqrt(b);  (R) cap vs b on sqrt curve
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    ks_axis = list(range(1, 40))
    axL.plot(ks_axis, [k * k for k in ks_axis], color="0.4", lw=2.2,
             label="LEAF / direct  (k² tiles)")
    for i, bb in enumerate([64, 128, 256]):
        col = C[i]
        axL.plot(ks_axis, [k + bb for k in ks_axis], color=col, lw=2, ls="--",
                 label=f"split once, b={bb}  (k+b tiles)")
        kc = _kstar(bb)
        axL.plot(kc, kc * kc, marker="o", ms=12, mfc="white", mec=col, mew=2.2, zorder=5)
    axL.set_yscale("log")
    axL.set_title("split the sub-DFT or make it a LEAF?   crossing = the cap  (WORK)", fontsize=11)
    axL.set_xlabel("sub-DFT size  s = LEAF  (in units of b)")
    axL.set_ylabel("cost  (tiles;  leaf = k²,  split = k+b)")
    axL.grid(True, which="both", alpha=0.22, lw=0.6)
    axL.legend(frameon=False, fontsize=8.5, loc="upper left")

    axR.plot(bs, [math.sqrt(x) for x in bs], color="crimson", lw=2,
             label=r"$\sqrt{b}$  (the law)", zorder=1)
    axR.scatter(bs, caps, s=80, color=C[0], zorder=3, label="LEAF cap (work model)")
    for x, y in zip(bs, caps):
        axR.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    axR.set_xscale("log", base=2); axR.set_yscale("log", base=2)
    axR.set_title("the cap IS √b — it moves with the array width  (WORK)", fontsize=11)
    axR.set_xlabel("array width  b")
    axR.set_ylabel("LEAF cap  (in units of b)")
    axR.grid(True, which="both", alpha=0.22, lw=0.6)
    axR.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("PROOF: leaf the small sub-DFTs (claim 1), and the LEAF cap is √b (claim 2)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    print(f"\nsaved {path}   (work proof of the cap; time column above is the claim-1 check)")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "plot":
        plot()
        plot_time()
    elif mode == "prove":
        prove()
    else:
        run()
