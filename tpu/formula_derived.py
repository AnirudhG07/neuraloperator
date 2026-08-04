"""
TPU v5e FFT time model — piecewise (regime-aware), m=1 (always split to LEAF=b).

Model structure
---------------
    T(N) = overhead(N) + memory(N) + compute(N)

  where every term scales with the FRACTIONAL number of split levels
  L = log_b(N/b)  (fractional, so the curve rises smoothly through the
  N = b^k boundaries instead of stepping).

  - overhead = T0 + TL * L          one-time floor + per-level kernel dispatch
  - memory   = sum of HBM passes    (MXU data, VPU twiddle, copy/transpose, matrix)
  - compute  = MXU + VPU flops       (tiny, ~0.5%, kept for portability)

  Regimes (same physics, one formula):
    A  overhead-bound   N <= b^2         time ~ flat floor (~200 us)
    B  transition       b^2 < N < b^2.5  floor ~ memory
    C  memory-bound     N  > b^2.5       time ~ linear in N*L  (VPU-twiddle dominated)

Hardware constants are for TPU v5e (measured from the jax profiler trace).
Swap them for another TPU generation and the same structure holds.
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  Hardware constants  (TPU v5e)
# ─────────────────────────────────────────────────────────────────────────────
b = 128                 # MXU systolic-array width  (radix / tile size)

BETA_MXU  = 384.0       # GB/s  effective bandwidth of the DFT matmul (custom-call)
BETA_VPU  = 50.0        # GB/s  effective bandwidth of the twiddle/mask elementwise op
BETA_COPY = 788.0       # GB/s  effective bandwidth of transpose/reshape copies
PI_MXU    = 197e3       # GFLOP/s  MXU peak  (197 TFLOP/s bf16)
PI_VPU    = 7e3         # GFLOP/s  VPU peak  (~few TFLOP/s)

BYTES     = 8           # bytes per element (complex64)
T0        = 175.0       # us  one-time launch / dispatch floor
TL        = 55.0        # us  per-level dispatch overhead (≈ 4 kernels × ~14 us)
EPS_LEAF  = 1.1e-3      # us per leaf-matrix element (only matters when m > 1)


# unit helpers:  bytes -> us  and  flop -> us
def _mem_us(num_bytes, bw_gbs):
    return num_bytes / (bw_gbs * 1e3)        # 1 GB/s = 1e3 bytes/us

def _flop_us(num_flop, pi_gflops):
    return num_flop / (pi_gflops * 1e3)      # 1 GFLOP/s = 1e3 flop/us


# ─────────────────────────────────────────────────────────────────────────────
#  The model
# ─────────────────────────────────────────────────────────────────────────────
def fft_time_us(N, m=1, fractional_L=True):
    """Predicted wall-clock time (us) of an N-point FFT on TPU v5e.

    N            : transform length
    m            : leaf size in units of b  (m=1 => split fully to LEAF=b)
    fractional_L : use L = log_b(N/mb) directly (smooth) instead of rounding.
                   Fractional L removes the integer-step artifact at N = b^k.
    """
    if N <= b:
        # direct single-tile DFT, no splitting
        return T0 + _mem_us(N * N * BYTES, BETA_MXU) + _flop_us(N * N, PI_MXU)

    # number of radix-b split levels (fractional by default)
    L = math.log(N / (m * b), b)
    if not fractional_L:
        L = max(1, round(L))

    # ── overhead ──────────────────────────────────────────────────────────
    t = T0 + TL * L

    # ── memory (HBM passes, per level) ────────────────────────────────────
    t += _mem_us(4 * N * BYTES, BETA_MXU)  * L    # MXU: col-DFT + row-DFT data
    t += _mem_us(2 * N * BYTES, BETA_VPU)  * L    # VPU: twiddle+mask (DOMINANT)
    t += _mem_us(2 * N * BYTES, BETA_COPY) * L    # COPY: transpose / reshape
    t += _mem_us(b * b * BYTES, BETA_MXU)  * L    # MXU: b×b DFT matrix read/level

    # ── compute (tiny, kept for portability) ──────────────────────────────
    t += _flop_us(2 * N * b * L, PI_MXU)          # MXU: sub-DFT matmuls
    t += _flop_us(8 * N * L,     PI_VPU)          # VPU: twiddle mul+add + table gen

    # ── leaf matrix (only nonzero when m > 1) ─────────────────────────────
    t += EPS_LEAF * (m * b) ** 2

    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Measured data (TPU v5e, m = 1 / LEAF = b)
# ─────────────────────────────────────────────────────────────────────────────
DATA_M1 = [
    (256, 208.0), (384, 216.4), (512, 178.9), (640, 179.1), (768, 167.5),
    (1024, 232.0), (1280, 200.7), (1408, 216.4), (1536, 223.3), (1664, 204.9),
    (1792, 227.5), (1920, 225.3), (2048, 250.8), (2560, 193.6), (3072, 230.5),
    (3328, 164.5), (3456, 181.1), (3584, 174.9), (4096, 196.4), (5120, 211.6),
    (6144, 216.2), (8192, 225.4), (12288, 255.7), (16384, 225.6), (32768, 244.3),
    (49152, 291.8), (65536, 336.8), (98304, 337.5), (131072, 392.4),
    (196608, 433.5), (262144, 538.1), (393216, 642.7), (524288, 834.4),
    (2097152, 2880.0), (4194304, 4100.0), (8388608, 8370.0), (16777216, 19360.0),
    (33554432, 41940.0), (67108864, 83550.0),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(data):
    Ns = np.array([d[0] for d in data])
    Ts = np.array([d[1] for d in data])
    pred = np.array([fft_time_us(N) for N in Ns])

    rel = (pred - Ts) / Ts * 100.0
    abs_err = np.abs(rel)

    lt, lp = np.log(Ts), np.log(pred)
    r2_log = 1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)

    small = Ns <= b * b
    big = Ns > b * b

    print("=" * 60)
    print(f"GOODNESS OF FIT  ({len(Ns)} points, m=1)")
    print("=" * 60)
    print(f"  median |err|        : {np.median(abs_err):5.1f} %")
    print(f"  mean   |err|        : {np.mean(abs_err):5.1f} %")
    print(f"  90th pct |err|      : {np.percentile(abs_err, 90):5.1f} %")
    print(f"  max    |err|        : {np.max(abs_err):5.1f} %")
    print(f"  R^2 (log space)     : {r2_log:6.4f}")
    print(f"  small N (<=b^2)     : median |err| {np.median(abs_err[small]):5.1f} %")
    print(f"  large N (>b^2)      : median |err| {np.median(abs_err[big]):5.1f} %")
    for band in (10, 20, 30):
        print(f"  within +/-{band:<2d}%       : {100*np.mean(abs_err<band):3.0f} % of points")
    return Ns, Ts, pred, rel


# ─────────────────────────────────────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot(Ns, Ts, pred, rel, path="fft_time_model.png"):
    abs_err = np.abs(rel)
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # regime shading
    ax.axvspan(1e2, b * b, alpha=0.06, color="green")
    ax.axvspan(b * b, b ** 2.5, alpha=0.06, color="gold")
    ax.axvspan(b ** 2.5, 1e8, alpha=0.06, color="orange")
    ax.text(2e3, 5e4, "A: overhead\nbound", fontsize=9, color="green", ha="center")
    ax.text(7e4, 5e4, "B:\ntransition", fontsize=9, color="#B8860B", ha="center")
    ax.text(5e6, 5e4, "C: memory\nbound", fontsize=9, color="#C0552F", ha="center")

    # measured (dots) vs calculated (crosses)
    ax.loglog(Ns, Ts, "o", ms=8, color="#185FA5", label="MEASURED",
              zorder=5, mec="white", mew=0.8)
    ax.loglog(Ns, pred, "X", ms=10, color="#993C1D", label="CALCULATED",
              zorder=6, mew=1.8)
    for N, T, p in zip(Ns, Ts, pred):
        ax.plot([N, N], [T, p], "-", color="gray", alpha=0.3, lw=0.7, zorder=1)

    # level boundaries
    for x, c, lab in [(b * b, "green", "L:1→2"), (b ** 3, "orange", "L:2→3")]:
        ax.axvline(x, color=c, ls="--", alpha=0.5, lw=1)

    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("time (us)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"TPU v5e FFT (m=1): measured vs calculated\n"
                 f"median |err| {np.median(abs_err):.0f}%")

    # residuals
    axr.semilogx(Ns, rel, "o", ms=6, color="#7F77DD")
    axr.axhline(0, color="k", lw=0.8)
    axr.axhspan(-20, 20, alpha=0.12, color="green")
    axr.axvline(b * b, color="green", ls="--", alpha=0.5)
    axr.axvline(b ** 3, color="orange", ls="--", alpha=0.5)
    axr.set_xlabel("N")
    axr.set_ylabel("(calc - meas) / meas   %")
    axr.grid(alpha=0.3)
    axr.set_title("Residuals")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nplot saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Ns, Ts, pred, rel = evaluate(DATA_M1)

    print(f"\n{'N':>10} {'measured':>9} {'calculated':>11} {'err%':>6} {'L':>5}")
    for N, T, p in zip(Ns, Ts, pred):
        L = math.log(N / b, b) if N > b else 0
        print(f"{N:>10} {T:>9.1f} {p:>11.1f} {(p-T)/T*100:>+6.0f} {L:>5.2f}")

    plot(Ns, Ts, pred, rel)