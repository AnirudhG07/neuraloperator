"""
CODE WRITTEN BY AI but Verified by me, it's just a proof of concept...

Demonstrates:
  Part 1 — Three equivalent 1D DFT algorithms:
    (a) Naive O(N²)
    (b) Cooley-Tukey recursive (N = power of 2)
    (c) Factor N = p*q (row-column + twiddle)

  Part 2 — 2D DFT:
    (a) Naive O(N1² * N2²)
    (b) Separable row-then-column  ← "2D FFT = repeated 1D FFTs"
    Show they are equal, and that a flat 1D FFT of the data is NOT equal.
"""

import cmath
import math
import random

# ─── helpers ──────────────────────────────────────────────────────────────────

def allclose(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))

def allclose_2d(A, B, tol=1e-9):
    return all(abs(A[i][j] - B[i][j]) < tol
               for i in range(len(A)) for j in range(len(A[0])))

def fmt(c, w=9):
    return f"{c.real:+.3f}{c.imag:+.3f}j".rjust(w + 10)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — 1D DFT
# ══════════════════════════════════════════════════════════════════════════════

# ── (a) Naive DFT  O(N²) ─────────────────────────────────────────────────────
def naive_dft(x):
    """Ground-truth DFT.  X[k] = Σ_n x[n] · exp(-2πi·k·n/N)"""
    N = len(x)
    return [
        sum(x[n] * cmath.exp(-2j * math.pi * k * n / N) for n in range(N))
        for k in range(N)
    ]


# ── (b) Cooley-Tukey DIT (radix-2, N must be a power of 2) ───────────────────
#
#  Split x into even-indexed and odd-indexed halves, recurse, combine:
#
#    X[k]       = E[k] + w^k · O[k]      k = 0 … N/2-1
#    X[k + N/2] = E[k] - w^k · O[k]
#
#  where w^k = exp(-2πi·k/N)  (twiddle factor)
#        E   = DFT of even-indexed sub-sequence
#        O   = DFT of odd-indexed sub-sequence
#
def cooley_tukey_fft(x):
    N = len(x)
    if N == 1:
        return [x[0]]
    assert N & (N - 1) == 0, "N must be a power of 2"

    even = cooley_tukey_fft(x[::2])
    odd  = cooley_tukey_fft(x[1::2])

    half    = N // 2
    twiddle = [cmath.exp(-2j * math.pi * k / N) * odd[k] for k in range(half)]

    return [even[k] + twiddle[k] for k in range(half)] + \
           [even[k] - twiddle[k] for k in range(half)]


# ── (c) Factor N = p*q — using col-FFT → twiddle → transpose → col-FFT ────────
#
# Derivation (same as before, rewritten to use only column FFTs):
#   Write n = n1 + p·n2, k = k1·q + k2.
#
#   X[k1·q+k2] = Σ_{n1} W_p^{k1·n1} · [Σ_{n2} x[n1+p·n2] · W_q^{k2·n2}] · W_N^{k2·n1}
#                                         ↑ q-point col-DFT for each n1-column
#                                                                              ↑ twiddle
#
# Algorithm (only Cooley-Tukey column FFTs, mirrors fft_2d but WITH twiddle):
#   1. Arrange x as q×p matrix MT  MT[n2][n1] = x[n1 + p·n2]
#      (each column = x[n1], x[n1+p], x[n1+2p], …)
#   2. Column FFT: q-point CT FFT of every column of MT  → C[k2][n1]
#   3. Twiddle:    C[k2][n1] *= exp(-2πi·k2·n1/N)        ← key difference vs 2D FFT
#   4. Transpose:  C → Cᵀ[n1][k2]
#   5. Column FFT: p-point CT FFT of every column of Cᵀ  → Xᵀ[k1][k2]
#   6. Flatten row-major  X[k1·q + k2]
#
def factor_fft(x, p, q):
    """1D -> 2D for FFT -> 1D"""
    N = p * q
    assert len(x) == N

    # Step 1 — arrange as q×p matrix  MT[n2][n1] = x[n1 + p*n2]
    MT = [[x[n1 + p * n2] for n1 in range(p)] for n2 in range(q)]

    # Step 2 — q-point column FFT of each of the p columns of MT
    C = [[0j] * p for _ in range(q)]          # C[k2][n1]
    for n1 in range(p):
        col = [MT[n2][n1] for n2 in range(q)]
        col_fft = cooley_tukey_fft(col)
        for k2 in range(q):
            C[k2][n1] = col_fft[k2]

    # Step 3 — twiddle  exp(-2πi · k2 · n1 / N)
    for k2 in range(q):
        for n1 in range(p):
            C[k2][n1] *= cmath.exp(-2j * math.pi * k2 * n1 / N)

    # Step 4 — transpose  C → Cᵀ[n1][k2]
    CT = [[C[k2][n1] for k2 in range(q)] for n1 in range(p)]

    # Step 5 — p-point column FFT of each of the q columns of Cᵀ
    XT = [[0j] * q for _ in range(p)]         # Xᵀ[k1][k2]
    for k2 in range(q):
        col = [CT[n1][k2] for n1 in range(p)]
        col_fft = cooley_tukey_fft(col)
        for k1 in range(p):
            XT[k1][k2] = col_fft[k1]

    # Step 6 — flatten row-major  k = k1*q + k2
    return [XT[k1][k2] for k1 in range(p) for k2 in range(q)]


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — 2D DFT
# ══════════════════════════════════════════════════════════════════════════════

# ── (a) Naive 2D DFT  O(N1² · N2²) ──────────────────────────────────────────
#
#   X[k1,k2] = Σ_{n1} Σ_{n2} M[n1][n2] · exp(-2πi·k1·n1/N1) · exp(-2πi·k2·n2/N2)
#
def naive_dft_2d(M):
    N1, N2 = len(M), len(M[0])
    return [
        [
            sum(
                M[n1][n2]
                * cmath.exp(-2j * math.pi * k1 * n1 / N1)
                * cmath.exp(-2j * math.pi * k2 * n2 / N2)
                for n1 in range(N1)
                for n2 in range(N2)
            )
            for k2 in range(N2)
        ]
        for k1 in range(N1)
    ]


# ── (b) 2D FFT: column-FFT → transpose → column-FFT (NO twiddle) ──────────────
#
# Derivation:
#   X[k1,k2] = Σ_{n2} W_N2^{k2·n2} · [ Σ_{n1} M[n1][n2] · W_N1^{k1·n1} ]
#                                         ↑ column FFT (axis 0) for each n2
#
# The two dimensions are separable — their twiddle factors are independent.
# No cross-term twiddle exp(-2πi·k2·n1/N) appears here (contrast with
# factor_fft for 1D, which uses N = N1·N2 and DOES have that cross twiddle).
#
# Algorithm (only Cooley-Tukey column FFTs):
#   1. Column FFT: N1-point CT FFT of every column of M      → C[k1][n2]
#   2. Transpose C                                            → Cᵀ[n2][k1]
#   3. Column FFT: N2-point CT FFT of every column of Cᵀ     → Xᵀ[k2][k1]
#   4. Transpose back                                         → X[k1][k2]
#
def fft_2d(M):
    """2D FFT via column-FFT → transpose → column-FFT (no twiddle needed here)"""
    N1, N2 = len(M), len(M[0])

    # Step 1: N1-point column FFT (one per column of M)
    C = [[0j] * N2 for _ in range(N1)]           # C[k1][n2]
    for n2 in range(N2):
        col = [M[n1][n2] for n1 in range(N1)]
        fft_col = cooley_tukey_fft(col)
        for k1 in range(N1):
            C[k1][n2] = fft_col[k1]

    # Step 2: transpose  C → Cᵀ[n2][k1]
    CT = [[C[k1][n2] for k1 in range(N1)] for n2 in range(N2)]

    # Step 3: N2-point column FFT (one per column of Cᵀ)
    XT = [[0j] * N1 for _ in range(N2)]          # Xᵀ[k2][k1]
    for k1 in range(N1):
        col = [CT[n2][k1] for n2 in range(N2)]
        fft_col = cooley_tukey_fft(col)
        for k2 in range(N2):
            XT[k2][k1] = fft_col[k2]

    # Step 4: transpose back → X[k1][k2]
    return [[XT[k2][k1] for k2 in range(N2)] for k1 in range(N1)]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)
    SEP  = "═" * 60
    SEP2 = "─" * 60

    # ── shared data ───────────────────────────────────────────────────────────
    N       = 16
    N1, N2  = 4, 4
    x  = [complex(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(N)]
    M  = [[complex(random.gauss(0, 1), random.gauss(0, 1))
           for _ in range(N2)] for _ in range(N1)]

    # ── compute all variants ──────────────────────────────────────────────────
    # 1D methods (all should give identical output)
    r_naive  = naive_dft(x)
    r_ct     = cooley_tukey_fft(x)
    r_f28    = factor_fft(x, p=2, q=8)
    r_f44    = factor_fft(x, p=4, q=4)
    r_f82    = factor_fft(x, p=8, q=2)

    # 2D methods (both should give identical output)
    r_2d_naive = naive_dft_2d(M)
    r_2d_fast  = fft_2d(M)

    # 1D DFT of the same data as M, flattened column-major
    flat_cm      = [M[n1][n2] for n2 in range(N2) for n1 in range(N1)]
    r_1d_of_flat = naive_dft(flat_cm)

    # 2D DFT flattened to compare against 1D
    r_2d_flat = [r_2d_naive[k1][k2] for k1 in range(N1) for k2 in range(N2)]

    # ─────────────────────────────────────────────────────────────────────────
    print(SEP)
    print("GROUP 1 — all 1D methods agree with each other")
    print(SEP)
    print(f"  Input: {N}-point 1D signal")
    print()
    ok = [
        ("naive_dft  ==  cooley_tukey_fft",    allclose(r_naive, r_ct)),
        ("naive_dft  ==  factor_fft(p=2,q=8)", allclose(r_naive, r_f28)),
        ("naive_dft  ==  factor_fft(p=4,q=4)", allclose(r_naive, r_f44)),
        ("naive_dft  ==  factor_fft(p=8,q=2)", allclose(r_naive, r_f82)),
        ("cooley_tukey_fft  ==  factor_fft(p=4,q=4)", allclose(r_ct, r_f44)),
    ]
    for label, result in ok:
        tick = "PASS" if result else "FAIL"
        print(f"  [{tick}]  {label}")
    print()
    print(f"  Sample output (k=0..3):")
    print(f"  {'k':>3}  {'naive':>22}  {'cooley_tukey':>22}  {'factor(4x4)':>22}")
    for k in range(4):
        print(f"  {k:>3}  {fmt(r_naive[k])}  {fmt(r_ct[k])}  {fmt(r_f44[k])}")

    # ─────────────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("GROUP 2 — both 2D methods agree with each other")
    print(SEP)
    print(f"  Input: {N1}x{N2} 2D grid")
    print()
    ok2 = [
        ("naive_dft_2d  ==  fft_2d (col->T->col)", allclose_2d(r_2d_naive, r_2d_fast)),
    ]
    for label, result in ok2:
        tick = "PASS" if result else "FAIL"
        print(f"  [{tick}]  {label}")
    print()
    print(f"  Sample output (k1=0, k2=0..3):")
    print(f"  {'k2':>3}  {'naive_2d':>22}  {'fft_2d':>22}")
    for k2 in range(4):
        print(f"  {k2:>3}  {fmt(r_2d_naive[0][k2])}  {fmt(r_2d_fast[0][k2])}")

    # ─────────────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("GROUP 3 — 1D DFT  !=  2D DFT  (same raw data, different transform)")
    print(SEP)
    print(f"  Same {N1}x{N2} values, column-major flat -> 1D DFT  vs  2D DFT")
    print()
    cross = [
        ("1D naive_dft(flat)  ==  1D factor_fft(flat)",  allclose(r_1d_of_flat, factor_fft(flat_cm, p=N1, q=N2))),
        ("1D naive_dft(flat)  ==  2D naive_dft_2d flat", allclose(r_1d_of_flat, r_2d_flat)),
        ("1D factor_fft(flat) ==  2D fft_2d flat",       allclose(factor_fft(flat_cm, p=N1, q=N2), r_2d_flat)),
    ]
    for label, result in cross:
        tick = "PASS" if result else "FAIL"
        print(f"  [{tick}]  {label}")
    print()
    print(f"  First 4 values to see the difference:")
    print(f"  {'k':>3}  {'1D DFT of flat':>22}  {'2D DFT flat':>22}  {'same?':>6}")
    for k in range(4):
        same = abs(r_1d_of_flat[k] - r_2d_flat[k]) < 1e-9
        print(f"  {k:>3}  {fmt(r_1d_of_flat[k])}  {fmt(r_2d_flat[k])}  {'yes' if same else 'NO':>6}")
    print()
    print("  Why they differ:")
    print("    1D DFT: X[k] = sum_n x[n] * exp(-2pi*i * k*n / N)")
    print("            single k mixes ALL of n -> cross twiddle exp(-2pi*i*k2*n1/N)")
    print("    2D DFT: X[k1,k2] = sum_{n1,n2} M[n1][n2]")
    print("              * exp(-2pi*i*k1*n1/N1) * exp(-2pi*i*k2*n2/N2)")
    print("            k1 touches only n1, k2 touches only n2 -> no cross term")

