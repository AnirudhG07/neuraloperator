"""
traffic_model.py  —  offline HBM-traffic predictor for the radix-B FFT fusion study.

The v5e FFT is memory-bandwidth-bound (AI 10-28 << ridge 240), so wall-clock ~ HBM bytes.
This module PREDICTS bytes-across-HBM for each fusion strategy as a function of N, WITHOUT a
TPU, so variants can be ranked before spending Colab time.  It is calibrated against XLA's
*static* cost_analysis 'bytes accessed' (also computed with no TPU) via phi-fitting.

Model (one N-point transform, radix B, complex64 = DT bytes/element):

    L(N)  = HBM-touching levels = radix-B split stages + leaf   (<= 4 for N <= 2^24 !)
    phi   = HBM round-trips PER LEVEL, set by how well the level's 4 ops fuse:
              6 = fully unfused (matmul 2N + twiddle 2N + transpose 2N)
              4 = XLA-typical  (twiddle fuses into transpose; matmul separate)
              2 = within-level fused (read once, all ops in VMEM, write once)

    M_naive(N)    = phi * N * L(N) * DT
    M_fused(N)    = 2   * N * L(N) * DT            (phi -> 2)
    M_fourstep(N) = p4s * N * DT,  p4s ~ 6 (2 FFT passes + 1 transpose), 4 if transpose fuses
                    (four-step keeps HBM passes CONSTANT in N because the sqrt(N) sub-FFTs
                     stay VMEM-resident; only worth it when L is large — here L<=4, so the
                     byte-cutters below usually beat it.)

    stackable multipliers:  rfft -> x0.5 (half the spectrum);  bf16 -> x0.5 on the
    matmul-OPERAND share only (twiddle/transpose data stay f32).

Key predicted finding to validate: at B=128 (tiny L), stacking within-level fusion + rfft +
bf16 (24 -> 2 units of N*DT) beats chasing the textbook four-step pass-count cut (24 -> 6).

Usage:
    python tpu/traffic_model.py            # fit phi from baseline + print the ranking table
    python tpu/traffic_model.py plot       # + save tpu/traffic_model.png
"""
import sys

B = 128           # radix
DT = 8            # bytes per complex64 element


# ── structural size ─────────────────────────────────────────────────────────
def levels(N, b=B):
    """HBM-touching levels for split-all-the-way radix-b: peels + leaf."""
    L, m = 0, N
    while m > b and m % b == 0:
        m //= b
        L += 1
    return L + 1                                   # +1 for the leaf matmul


# ── the coarse predictor (the ranking instrument) ───────────────────────────
# fraction of a level's traffic that is a matmul OPERAND (bf16-shrinkable) rather than
# complex intermediate data (twiddle/transpose, stays f32).  Modeling choice — refine
# against measured bf16 bytes.  For naive-phi the matmul input read is ~1 of phi sub-passes
# (~1/phi); for the fused level the single read IS the matmul input (~1/2).
def _matmul_share(strategy, phi):
    if strategy == "fused":
        return 0.5
    if strategy in ("fourstep", "fourstep_fused"):
        return 1.0 / 3.0                           # 1 of the ~3 passes is a matmul input
    return 1.0 / max(phi, 1.0)                     # naive


# fixed O(B^2) DFT-matrix overhead per transform: the radix B*B matrix is generated
# (exp -> write B^2) and consumed (read B^2) once, CSE'd across levels.  Dominates the byte
# count at tiny N (where the data is < the matrix), negligible at large N.  This term is
# what made the naive per-point phi blow up to ~259 at N=128.
MATRIX_OVERHEAD = 2 * B * B * DT                   # ~262 KB


def traffic_model(N, strategy="naive", phi=4.0, rfft=False, bf16=False,
                  matmul_share=None, matrix=True):
    """Predicted HBM bytes for one N-point transform under `strategy`."""
    L = levels(N)
    passes = {
        "naive":          phi * L,
        "fused":          2.0 * L,
        "fourstep":       6.0,
        "fourstep_fused": 4.0,
    }[strategy]
    data = passes * N * DT                          # the streamed transform data
    if bf16:
        s = _matmul_share(strategy, phi) if matmul_share is None else matmul_share
        data *= (1.0 - s) + s * 0.5                 # halve only the operand share
    if rfft:
        data *= 0.5
    mtx = (MATRIX_OVERHEAD * (0.5 if bf16 else 1.0)) if matrix else 0.0
    return data + mtx


# ── calibration against XLA static cost_analysis (offline, no TPU) ──────────
def measured_bytes(N, variant="baseline"):
    """XLA 'bytes accessed' for a compiled variant from tpu/fft_variants.py (static)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tpu.fft_variants import fft_dispatch

    x = jnp.asarray(np.random.randn(N).astype(np.float32))
    comp = jax.jit(lambda x: fft_dispatch(x, variant)).lower(x).compile()
    ca = comp.cost_analysis() or {}
    return float(ca.get("bytes accessed", 0) or 0)


def fit_phi(Ns, variant="baseline"):
    """Back out the effective phi per level from measured bytes: phi = bytes / (N*DT*L)."""
    rows = []
    for N in Ns:
        by = measured_bytes(N, variant)
        L = levels(N)
        phi = by / (N * DT * L) if (N and L) else 0.0
        rows.append({"N": N, "L": L, "bytes": by, "phi": phi})
    return rows


# ── reporting ───────────────────────────────────────────────────────────────
def _h(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def report(fit_Ns=(128, 16384, 2097152), table_Ns=(128, 16384, 2097152, 16777216)):
    print("=" * 84)
    print("PHI FIT  — back out per-level HBM round-trips from measured baseline bytes")
    print("=" * 84)
    print(f"  {'N':>12} {'L':>3} {'measured bytes':>16} {'phi=bytes/(N*DT*L)':>20}")
    fit = fit_phi(fit_Ns)
    for r in fit:
        print(f"  {r['N']:>12,} {r['L']:>3} {_h(r['bytes']):>16} {r['phi']:>20.2f}")
    phi_hat = sum(r["phi"] for r in fit) / len(fit)
    print(f"\n  => effective phi ~ {phi_hat:.2f}  "
          f"(2=fused, 4=XLA-typical, 6=unfused).  Using it to predict strategies below.\n")

    print("=" * 84)
    print(f"PREDICTED HBM BYTES per strategy  (phi={phi_hat:.2f})   "
          f"— lower = fewer round-trips")
    print("=" * 84)
    cols = [
        ("naive",              {"strategy": "naive", "phi": phi_hat}),
        ("fused",              {"strategy": "fused"}),
        ("fused+rfft",         {"strategy": "fused", "rfft": True}),
        ("fused+rfft+bf16",    {"strategy": "fused", "rfft": True, "bf16": True}),
        ("fourstep",           {"strategy": "fourstep"}),
    ]
    print(f"  {'N':>12} {'L':>3} " + "".join(f"{name:>17}" for name, _ in cols))
    for N in table_Ns:
        row = f"  {N:>12,} {levels(N):>3} "
        base = traffic_model(N, strategy="naive", phi=phi_hat)
        for _, kw in cols:
            b = traffic_model(N, **kw)
            row += f"{_h(b) + f'({b/base:.2f}x)':>17}"
        print(row)
    print("\n  ratios are vs naive at the same N;  units are HBM bytes for ONE transform.")


def plot(path="tpu/traffic_model.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = fit_phi((128, 16384, 2097152))
    phi_hat = sum(r["phi"] for r in fit) / len(fit)
    Ns = [B ** p for p in (1, 2, 3)] + [B ** 3 * m for m in (2, 4, 8)]
    Ns = sorted(set(Ns))
    series = [
        ("naive",           {"strategy": "naive", "phi": phi_hat}, "#D55E00"),
        ("fused",           {"strategy": "fused"}, "#0072B2"),
        ("fused+rfft",      {"strategy": "fused", "rfft": True}, "#009E73"),
        ("fused+rfft+bf16", {"strategy": "fused", "rfft": True, "bf16": True}, "#56B4E9"),
        ("fourstep",        {"strategy": "fourstep"}, "#CC79A7"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for name, kw, col in series:
        ys = [traffic_model(N, **kw) for N in Ns]
        ax.plot(Ns, ys, marker="o", ms=4, lw=1.7, color=col, label=name)
    # overlay measured baseline bytes
    mb = [(N, measured_bytes(N)) for N in (128, 16384, 2097152)]
    ax.scatter([n for n, _ in mb], [b for _, b in mb], s=70, facecolors="none",
               edgecolors="black", zorder=6, label="measured baseline (cost_analysis)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("predicted HBM bytes / transform")
    ax.set_title(f"HBM-traffic model  (B={B}, phi_fit={phi_hat:.2f})", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


if __name__ == "__main__":
    report()
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
