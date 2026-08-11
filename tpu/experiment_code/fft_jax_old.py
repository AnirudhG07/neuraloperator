import math

import jax
import jax.numpy as jnp

# ─── global hyperparameters ───────────────────────────────────────────────
B = 128  # base: size of the systolic array you "built". Must divide N3 and N4.

# Leaf threshold — DERIVED FROM B, not hardcoded, so it tracks the array width.
#
# The decision at every recursion step is: given a piece of size m = k*B, is it
# cheaper to SPLIT it once (peel a radix-B stage) or to do it DIRECT as one padded
# matmul?  On a B-wide systolic array a k*B x k*B DFT costs k^2 tiles, while one
# split costs (k + B) tiles, so splitting wins exactly when
#       k^2 > k + B   <=>   k > k* = (1 + sqrt(1 + 4B)) / 2 ≈ sqrt(B).
# So: DIRECT (leaf) when k <= floor(k*),  SPLIT when k >= floor(k*)+1.
# For B=128, k* ≈ 11.8, so we leaf pieces up to 11*B and split anything >= 12*B.
# `small_dft` applies this per piece via `m <= LEAF`, so a single LEAF = floor(k*)*B
# realises the rule for WHATEVER factor `factorize` produces — the leaf size always
# lands below 12*B.  Change B and this recomputes automatically.
K_STAR = int((1 + math.sqrt(1 + 4 * B)) / 2)   # 11 for B=128  (floor of the sqrt(B) crossover)
LEAF = K_STAR * B                              # 1408 for B=128  (= 11*B, just under 12*B)

CDTYPE = jnp.complex64   # fp32 components — the working precision for the transform
ACCUM  = jnp.complex64   # fp32 matmul accumulation


def dft_matrix(m):
    """m x m DFT coefficient matrix C, where C[r,c] = W_m^(r*c)."""
    k = jnp.arange(m)
    W = jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m)
    return W.astype(CDTYPE)  # shape (m, m), fp32 complex

def small_dft(vecs, m):
    """
    Apply an m-point DFT to each length-m column of `vecs` (shape (m, K)).

    ADAPTIVE DEPTH: peel one radix-B factor (m = B * N1), then handle the inner
    N1-point DFT by RECURSING — so a big N1 is split again, a small N1 (<= LEAF) is
    done as a single direct matmul. The depth therefore grows with m automatically:
      m <= LEAF (or not B-divisible)  ->  1 direct DFT matmul   (a leaf)
      m  > LEAF                       ->  split once, recurse on N1 = m/B
    For m <= B*LEAF this is byte-identical to the old fixed 2-step version.
    """
    if m <= LEAF or m % B != 0:
        # leaf: one direct DFT matmul, fp32 accumulation
        return jnp.matmul(dft_matrix(m), vecs.astype(CDTYPE),
                          preferred_element_type=ACCUM)

    N1 = m // B
    K = vecs.shape[1]

    # reshape each column into a (B, N1) block: index n = r*N1 + c  (r in [0,B) is
    # the HIGH digit, c in [0,N1) the LOW digit) — this is what reshape(B, N1) gives.
    Xb = vecs.reshape(B, N1, K).astype(CDTYPE)  # (B, N1, K)

    Cb = dft_matrix(B)                          # (B, B)   radix coefficients
    # fp32 accumulation on the MXU via preferred_element_type
    Yb = jnp.einsum('br,rck->bck', Cb, Xb,
                    preferred_element_type=ACCUM)  # (B, N1, K)  B-point DFT over r

    r = jnp.arange(B)[:, None]                  # row index
    c = jnp.arange(N1)[None, :]                 # inner-col index
    Tw = jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)  # twiddle W_m^(r*c), (B, N1)
    Zb = Yb * Tw[:, :, None]                    # element-wise twiddle -> (B, N1, K)

    # inner N1-point DFT over the c axis: RECURSE (adaptive depth) instead of a direct matmul.
    # flatten so each length-N1 column is a batch column: (N1, B*K).
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)   # column j <- (b, k), row <- c
    Wd = small_dft(Zc, N1)                       # (N1, B*K)  recurse
    out = Wd.reshape(N1, B, K)                   # (d, b, k)  frequency index = d*B + b
    return out.reshape(m, K)                     # (m, K)

def base_b_fft(x, N3, N4):
    """N = N3*N4 point DFT via the row-column (Cooley-Tukey) decomposition."""
    N = N3 * N4
    # Pack n = k + N4*i  ->  column index i is the HIGH digit (weight N4).
    # This must match the column-first (length-N3 DFT first) order below and the
    # W_N^(i*k) twiddle; the old `reshape(N4, N3).T` used the opposite convention.
    X = x.astype(CDTYPE).reshape(N3, N4)          # (N3, N4)  X[i, k] = x[k + N4*i]

    # stage 1: column DFTs (length N3, one per column)
    Xc = small_dft(X, N3)                        # (N3, N4)

    # level-1 twiddle: multiply element (i,k) by W_N^(i*k)
    i = jnp.arange(N3)[:, None]
    k = jnp.arange(N4)[None, :]
    Xc = Xc * jnp.exp(-2j * jnp.pi * (i * k) / N).astype(CDTYPE)

    # stage 2: row DFTs (length N4, one per row) -> transpose so rows become columns
    Xr = small_dft(Xc.T, N4)                     # (N4, N3)  Xr[q, p] = X(p + N3*q)
    # Output index m = p + N3*q  ->  row-major flatten of Xr (shape (N4, N3)) gives
    # exactly q*N3 + p = m. (The old `Xr.T.reshape(-1)` used m = p*N4 + q, wrong.)
    return Xr.reshape(-1)                          # normal-order output X(0..N-1)

from collections import namedtuple

# A plan is a *host-side*, static description of how to compute the N-point DFT.
# It is chosen from N (a Python int) before tracing, so the compiled TPU graph has
# NO data-dependent control flow — just fixed-shape matmuls. This is the whole point:
# we pay the "which factorization?" cost once on the host, never on the accelerator.
Plan = namedtuple("Plan", "method N N3 N4 L levels")


def _divisors(N):
    """All divisors of N, ascending."""
    ds = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            ds.append(i)
            if i != N // i:
                ds.append(N // i)
        i += 1
    return sorted(ds)


def _balanced_split(N, b):
    """
    Pick d | N (1 < d < N) so that N = d * (N/d) is a good top-level Cooley-Tukey
    split for the MXU:
      1. prefer BOTH factors divisible by b (so each small_dft factorises again and
         every matmul fills the systolic array),
      2. then prefer the most balanced split (d ~ N/d), which minimises the size of
         the largest single DFT matrix.
    Returns (N3, N4) or None if N is prime.
    """
    best = None
    for d in _divisors(N):
        if d == 1 or d == N:
            continue
        e = N // d
        n_mult = int(d % b == 0) + int(e % b == 0)     # 0, 1, or 2
        score = (n_mult, -abs(d - e))                   # more b-multiples, then balance
        if best is None or score > best[0]:
            best = (score, (min(d, e), max(d, e)))
    return best[1] if best else None


def _next_pow_b(target, b):
    """Smallest power of b that is >= target (used for the Bluestein workspace length)."""
    p = 1
    while p < target:
        p *= b
    return p


def factorize(N, b):
    """
    Smart, *static* factorisation picker for the TPU.

    Regimes (see the module-level table):
      - N <= b            : 'direct'     — one DFT-matrix matmul, no factorisation.
      - N composite       : 'two_factor' — N = N3 * N4 via `base_b_fft`; both factors
                            chosen near sqrt(N) and divisible by b when possible so the
                            inner `small_dft` factorises them again (the "b*b*something"
                            case). This is the exact four-step / Cooley-Tukey FFT.
      - N prime / awkward : 'bluestein'  — an exact N-point DFT is impossible to reach by
                            plain zero-padding (padding changes the frequency grid!), so we
                            reformulate it as a length-L circular convolution with
                            L = next power of b >= 2N-1, and run THAT with `base_b_fft`.
                            Here the padding is a valid convolution workspace, not a fake
                            transform length.

    Returns a Plan namedtuple. `levels` counts nested Cooley-Tukey stages
    (0 = direct, 2 = top split + one inner small_dft split per factor, ...).
    """
    if N <= b:
        return Plan("direct", N, None, None, None, 0)

    split = _balanced_split(N, b)
    if split is not None:
        N3, N4 = split
        # levels: 1 for the top split, +1 for each factor that small_dft can split again
        levels = 1 + int(N3 % b == 0) + int(N4 % b == 0)
        return Plan("two_factor", N, N3, N4, None, levels)

    # N is prime -> Bluestein over a b-smooth workspace length
    L = _next_pow_b(2 * N - 1, b)
    return Plan("bluestein", N, None, None, L, factorize(L, b).levels)


def base_b_ifft(X, N3, N4):
    """Inverse of base_b_fft, via ifft(X) = conj(fft(conj(X))) / N."""
    N = N3 * N4
    return jnp.conj(base_b_fft(jnp.conj(X), N3, N4)) / N


def bluestein_fft(x, L):
    """
    Exact N-point DFT of x (any N) as a length-L circular convolution (chirp-z /
    Bluestein). L must be b-smooth and >= 2N-1. Identity used:
        n*k = (n^2 + k^2 - (k-n)^2) / 2
      => X[k] = c[k] * sum_n (x[n] c[n]) * conj(c[k-n]),   c[m] = exp(-i*pi*m^2/N)
    the sum is a linear convolution, done with two length-L FFTs + one iFFT.
    """
    N = x.shape[0]
    split = _balanced_split(L, B)           # L is a power of B, so this is never None
    assert split is not None, f"Bluestein length {L} must be composite"
    L3, L4 = split

    n = jnp.arange(N)
    c = jnp.exp(-1j * jnp.pi * (n.astype(jnp.float64) ** 2) / N).astype(CDTYPE)  # chirp

    # a[n] = x[n]*c[n], zero-padded to length L
    a = jnp.zeros((L,), CDTYPE).at[:N].set(x.astype(CDTYPE) * c)

    # kernel h[m] = conj(c[|m|]) placed circularly on [-(N-1) .. N-1] over length L
    h = jnp.zeros((L,), CDTYPE)
    h = h.at[:N].set(jnp.conj(c))                       # m = 0 .. N-1
    tail = jnp.conj(c)[1:][::-1]                         # m = -(N-1) .. -1  ->  L-(N-1)..L-1
    h = h.at[L - (N - 1):].set(tail)

    A = base_b_fft(a, L3, L4)
    H = base_b_fft(h, L3, L4)
    conv = base_b_ifft(A * H, L3, L4)                   # circular conv, length L

    return c * conv[:N]                                  # de-chirp, keep first N


def fft(x):
    """Top-level dispatcher: pick a static plan from len(x), then run fixed-shape matmuls."""
    N = x.shape[0]
    plan = factorize(N, B)
    if plan.method == "direct":
        return jnp.matmul(dft_matrix(N), x.astype(CDTYPE), preferred_element_type=ACCUM)
    if plan.method == "two_factor":
        return base_b_fft(x, plan.N3, plan.N4)
    return bluestein_fft(x, plan.L)                      # 'bluestein'

def ifft(x):
    plan = factorize(x.shape[0], B)
    if plan.method == "direct":
        return jnp.conj(jnp.matmul(dft_matrix(plan.N), jnp.conj(x.astype(CDTYPE)),
                                   preferred_element_type=ACCUM)) / plan.N
    if plan.method == "two_factor":
        return base_b_ifft(x, plan.N3, plan.N4)
    return jnp.conj(bluestein_fft(jnp.conj(x.astype(CDTYPE)), plan.L)) / plan.N


if __name__ == "__main__":
    # fp64 chirp in bluestein needs x64 enabled; harmless for the rest.
    jax.config.update("jax_enable_x64", True)

    def small_dft_depth(m):
        """Report how many split levels small_dft(m) uses (leaf = 0)."""
        d = 0
        while m > LEAF and m % B == 0:
            m //= B
            d += 1
        return d

    # print(f"adaptive depth (LEAF={LEAF}, B={B}) — small_dft split levels vs size:")
    # for m in [4, 16, 32, 64, 128, 256, 1024, 4096]:
    #     print(f"  m={m:<5} split levels = {small_dft_depth(m)}  (leaf size = "
    #           f"{m // B ** small_dft_depth(m)})")

    # print("\naccuracy (unified fft dispatcher):")
    # for N in [3, 4, 7, 12, 16, 17, 60, 64, 100, 256, 1024, 4096]:
    #     x = jnp.arange(1, N + 1, dtype=jnp.float64)
    #     err = float(jnp.max(jnp.abs(fft(x) - jnp.fft.fft(x))))
    #     print(f"  N={N:<5} max abs error = {err:.2e}")
