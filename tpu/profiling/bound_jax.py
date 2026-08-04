"""
bound_jax.py  —  is the derived cost model an UPPER BOUND on real v5e time?

We fix m = 1 (LEAF = b, split every sub-DFT all the way to a single 128-tile leaf),
which the TIME data already proved is the fastest regime, and sweep ~100 values of N
(multiples of b, log-spaced from hundreds to ~10^7).  For each N we:

    measure   median blocked wall-clock of the real FFT on this backend
    predict   two model curves for the same N:

      (A) A-PRIORI  φ=6 upper bound  (FIXED constants from the v5e roofline + profiler):
            T = 8·(4NL + Nb)/β_MXU   +   8·(2NL)/β_VPU   +   (2NbL + Nb)/π_MXU   +   T0
          two matmul HBM passes/level + leaf matmul on the MXU pipe (β_MXU),
          the separate twiddle HBM pass/level on the slow VPU pipe (β_VPU),
          matmul compute, and a fixed kernel-dispatch floor T0.  L = split levels.
          This is a SUM of memory+compute (they actually overlap on HW), so it is an
          honest OVER-estimate -> should sit ABOVE measured for all but tiny N.

      (B) FITTED   least-squares (NNLS) fit to YOUR measured points.  With b constant
          and m=1, every term above is a constant multiple of one of three shapes,
          so the only separable model is
                T = a·(N·log_b N)  +  c·N  +  T0
          a,c,T0 >= 0 fit from the data.  This is the curve that "fits it for it".

Output: bound_data.csv  (N, levels, leaf, flops, bytes, ai, time, apriori, fit)
        time_vs_formula.png  (log-log measured vs both curves + residual ratio)

Run on a Colab **TPU v5e**:   !python bound_jax.py
Re-fit/plot later WITHOUT a TPU, straight from the CSV:
        python bound_jax.py --from-csv bound_data.csv
"""
import csv
import math
import sys
import time as _time

import jax
import jax.numpy as jnp
import numpy as np

b = 128
CDTYPE = jnp.complex64
ACCUM = jnp.complex64
BACKEND = jax.devices()[0].platform  # 'cpu' | 'gpu' | 'tpu'

# ── FIXED a-priori constants (v5e roofline + your profiler) ─────────────────
BYTES_PER = 8.0          # complex64 = 8 bytes/element
BETA_MXU = 384e9         # GB/s of the matmul-fed HBM pipe (profiler)
BETA_VPU = 50e9          # GB/s of the low-intensity twiddle/VPU pass (profiler)
PI_MXU = 197e12          # v5e bf16 peak FLOP/s
T0_APRIORI = 12e-6       # kernel-dispatch floor (s); tiny-N latency

CSV_PATH = "bound_data.csv"
PNG_PATH = "time_vs_formula.png"
REPS = 15


# ── real JAX FFT (LEAF = b, i.e. m = 1, maximal split), self-contained ──────
def dft_matrix(m):
    k = jnp.arange(m)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m).astype(CDTYPE)


def small_dft(vecs, m):
    if m <= b or m % b != 0:                       # LEAF == b  (single 128-tile leaf)
        return jnp.matmul(dft_matrix(m), vecs.astype(CDTYPE), preferred_element_type=ACCUM)
    N1 = m // b
    K = vecs.shape[1]
    Xb = vecs.reshape(b, N1, K).astype(CDTYPE)
    Yb = jnp.einsum('br,rck->bck', dft_matrix(b), Xb, preferred_element_type=ACCUM)
    r = jnp.arange(b)[:, None]
    c = jnp.arange(N1)[None, :]
    Zb = Yb * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, b * K)
    return small_dft(Zc, N1).reshape(N1, b, K).reshape(m, K)


def make_fft(N):
    @jax.jit
    def f(x):
        return small_dft(x[:, None], N)[:, 0]
    return f


def split_levels(N):
    """Number of radix-b SPLIT levels at LEAF=b (0 for N<=b)."""
    L, m = 0, N
    while m > b and m % b == 0:
        m //= b
        L += 1
    return L


def leaf_elems(N):
    """Residual leaf size in ELEMENTS at LEAF=b (<= b in the m=1 set -> one tile)."""
    m = N
    while m > b and m % b == 0:
        m //= b
    return m


def time_fft(N, reps=REPS):
    f = make_fft(N)
    x = jnp.arange(1, N + 1, dtype=jnp.float32).astype(CDTYPE)
    f(x).block_until_ready()                       # compile + warm
    ts = []
    for _ in range(reps):
        t0 = _time.perf_counter()
        f(x).block_until_ready()
        ts.append(_time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def cost_of(N):
    """Real XLA flops & HBM bytes from the compiled executable (best-effort)."""
    try:
        f = make_fft(N)
        x = jnp.arange(1, N + 1, dtype=jnp.float32).astype(CDTYPE)
        ca = f.lower(x).compile().cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        ca = ca or {}
        return float(ca.get("flops", 0) or 0), float(ca.get("bytes accessed", 0) or 0)
    except Exception:
        return 0.0, 0.0


# ── the two model curves ────────────────────────────────────────────────────
def apriori_time(N):
    """FIXED-constant φ=6 upper bound (seconds)."""
    L = split_levels(N)
    mem_mxu = BYTES_PER * (4 * N * L + N * b)
    mem_vpu = BYTES_PER * (2 * N * L)
    flops_mxu = 2.0 * N * b * L + N * b
    return mem_mxu / BETA_MXU + mem_vpu / BETA_VPU + flops_mxu / PI_MXU + T0_APRIORI


def fit_basis(N):
    """The 3 separable shapes: [ N·log_b N , N , 1 ]."""
    L = split_levels(N)
    return [float(N) * L, float(N), 1.0]


def nnls_fit(X, y):
    """Non-negative least squares; scipy if present, else lstsq + clamp."""
    try:
        from scipy.optimize import nnls
        coef, _ = nnls(X, y)
        return coef
    except Exception:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.maximum(coef, 0.0)


# ── N set: ~100 log-spaced multiples of b, ALL in the m=1 (single-tile) leaf
#    regime.  snap each target to N = b^p · r with r in [1, b] so leaf <= b. ──
def gen_Ns(count=100, lo=b, hi=10_000_000):
    out = set()
    for i in range(count):
        T = lo * (hi / lo) ** (i / (count - 1))
        p = max(1, int(math.floor(math.log(T) / math.log(b))))
        r = max(1, round(T / b ** p))
        if r > b:                                  # keep the leaf a single tile
            p += 1
            r = max(1, round(T / b ** p))
        out.add(b ** p * r)
    return sorted(out)


def measure():
    Ns = gen_Ns()
    print("=" * 92)
    print(f"UPPER-BOUND CHECK   b={b}   backend={BACKEND.upper()}   m=1 (LEAF=b)   "
          f"{len(Ns)} values of N   (median of {REPS} runs)")
    if BACKEND != "tpu":
        print("!! not a TPU — the memory-bound bound is v5e-specific; run on Colab TPU v5e.")
    print("=" * 92)
    print(f"  {'N':>12} {'lvls':>4} {'leaf':>5} {'flops':>9} {'bytes':>9} {'AI':>6} "
          f"{'measured':>11} {'apriori':>11}")
    rows = []
    for N in Ns:
        t = time_fft(N)
        flops, byts = cost_of(N)
        ai = flops / byts if byts else 0.0
        ap = apriori_time(N)
        rows.append({"N": N, "levels": split_levels(N), "leaf": leaf_elems(N),
                     "flops": flops, "bytes": byts, "ai": ai,
                     "time": t, "apriori": ap})
        tms = f"{t*1e6:8.1f}us" if t < 1e-3 else f"{t*1e3:8.2f}ms"
        aps = f"{ap*1e6:8.1f}us" if ap < 1e-3 else f"{ap*1e3:8.2f}ms"
        print(f"  {N:>12,} {split_levels(N):>4} {leaf_elems(N):>5} "
              f"{flops:>9.2e} {byts:>9.2e} {ai:>6.1f} {tms:>11} {aps:>11}")
    return rows


def fit_and_report(rows):
    N = np.array([r["N"] for r in rows], float)
    y = np.array([r["time"] for r in rows], float)
    ap = np.array([r["apriori"] for r in rows], float)
    X = np.array([fit_basis(r["N"]) for r in rows], float)

    coef = nnls_fit(X, y)
    a, c, t0 = coef
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    cover = float(np.mean(y <= ap))                 # fraction under the a-priori bound
    tightness = float(np.max(y / ap))               # how close measured gets to the bound
    med_ratio = float(np.median(ap / y))            # how loose the bound is (median)

    print("\n" + "=" * 92)
    print("FITTED model   T = a·(N·log_b N) + c·N + T0   (NNLS on measured):")
    print(f"    a  = {a:.4e} s   per unit N·log_b N   (per-level twiddle+matmul passes)")
    print(f"    c  = {c:.4e} s   per unit N           (leaf matmul)")
    print(f"    T0 = {t0*1e6:.2f} us                    (dispatch floor)")
    print(f"    R^2 = {r2:.4f}")
    print("\nA-PRIORI φ=6 upper bound  (FIXED β_MXU=384, β_VPU=50 GB/s, π=197 TF/s, T0=12us):")
    print(f"    measured <= a-priori for {cover*100:.0f}% of N   "
          f"(bound holds where this is 100%)")
    print(f"    tightest point: measured = {tightness:.2f}× the bound   "
          f"(<=1.0 means never exceeded)")
    print(f"    median looseness: bound = {med_ratio:.2f}× measured")
    print("=" * 92)

    for r in rows:
        r["fit"] = float(np.array(fit_basis(r["N"])) @ coef)
    return {"a": a, "c": c, "t0": t0, "r2": r2, "cover": cover,
            "tightness": tightness, "med_ratio": med_ratio}


def write_csv(rows, path=CSV_PATH):
    cols = ["N", "levels", "leaf", "flops", "bytes", "ai", "time", "apriori", "fit"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nsaved {path}   ({len(rows)} rows)")


def plot(rows, stats, path=PNG_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array([r["N"] for r in rows], float)
    y = np.array([r["time"] for r in rows], float)
    ap = np.array([r["apriori"] for r in rows], float)
    ft = np.array([r["fit"] for r in rows], float)
    o = np.argsort(N)
    N, y, ap, ft = N[o], y[o], ap[o], ft[o]

    fig, (axT, axR) = plt.subplots(
        2, 1, figsize=(9, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    axT.fill_between(N, y, ap, where=ap >= y, color="#0072B2", alpha=0.10,
                     label="headroom (measured → bound)")
    axT.plot(N, ap, color="#D55E00", lw=2, label="a-priori φ=6 upper bound (fixed const)")
    axT.plot(N, ft, color="#009E73", lw=1.8, ls="--",
             label=f"fitted  a·N·log_bN + c·N + T0   (R²={stats['r2']:.3f})")
    axT.scatter(N, y, s=22, color="#0072B2", zorder=5, label="measured (v5e)")
    axT.set_xscale("log"); axT.set_yscale("log")
    axT.set_ylabel("time  (s)")
    axT.set_title(f"Is the cost model an upper bound on v5e time?   "
                  f"b={b}, m=1, backend={BACKEND.upper()}", fontsize=11)
    axT.grid(True, which="both", alpha=0.22, lw=0.6)
    axT.legend(frameon=False, fontsize=8.5, loc="upper left")
    for p, lbl in [(2, "b²"), (3, "b³")]:
        axT.axvline(b ** p, color="0.7", ls=":", lw=0.8)
        axT.text(b ** p, axT.get_ylim()[0], f" {lbl}", color="0.5",
                 fontsize=8, va="bottom")

    axR.axhline(1.0, color="0.5", lw=1)
    axR.scatter(N, ap / y, s=18, color="#D55E00", label="a-priori / measured")
    axR.scatter(N, ft / y, s=14, color="#009E73", marker="x", label="fit / measured")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_ylim(0.3, max(3.0, float(np.max(ap / y)) * 1.2))
    axR.set_ylabel("model / measured")
    axR.set_xlabel("N  (transform length)")
    axR.grid(True, which="both", alpha=0.22, lw=0.6)
    axR.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    axR.text(N[0], 1.02, "  above 1.0 = model over-predicts (upper bound holds)",
             fontsize=7.5, color="0.4", va="bottom")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def from_csv(path):
    rows = []
    with open(path) as fh:
        for d in csv.DictReader(fh):
            rows.append({"N": int(float(d["N"])), "levels": int(float(d["levels"])),
                         "leaf": int(float(d["leaf"])), "flops": float(d["flops"] or 0),
                         "bytes": float(d["bytes"] or 0), "ai": float(d["ai"] or 0),
                         "time": float(d["time"]), "apriori": float(d["apriori"])})
    stats = fit_and_report(rows)
    plot(rows, stats)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--from-csv":
        from_csv(sys.argv[2])
        return
    rows = measure()
    stats = fit_and_report(rows)
    write_csv(rows)
    plot(rows, stats)


if __name__ == "__main__":
    main()
