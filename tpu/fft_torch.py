import torch

# ─── global hyperparameter ────────────────────────────────────────────────
B = 4  # base: size of the systolic array you "built". Must divide N3 and N4.

# Leaf threshold: largest DFT done as ONE direct matmul instead of splitting again.
# Single knob for recursion DEPTH — small_dft peels radix-B factors while size > LEAF,
# then stops. Set ~ the MXU dimension (~128 on a real TPU). LEAF=16 keeps the toy visible.
LEAF = 16

CDTYPE = torch.complex64   # fp32 components — the working precision for the transform

def dft_matrix(m):
    """m x m DFT coefficient matrix C, where C[r,c] = W_m^(r*c). Fallback."""
    k = torch.arange(m)
    W = torch.exp(-2j * torch.pi * torch.outer(k, k).to(CDTYPE) / m)
    return W.to(CDTYPE)  # (m, m), fp32 complex

def small_dft(vecs, m):
    """
    Apply an m-point DFT to each length-m column of `vecs` (shape (m, K)).

    ADAPTIVE DEPTH: peel one radix-B factor (m = B*N1), then RECURSE on the inner
    N1-point DFT — big N1 splits again, small N1 (<= LEAF) is a single direct matmul.
    Depth grows with m automatically:
      m <= LEAF (or not B-divisible) -> 1 direct DFT matmul (leaf)
      m  > LEAF                      -> split once, recurse on N1 = m/B
    """
    if m <= LEAF or m % B != 0:
        return dft_matrix(m) @ vecs.to(CDTYPE)      # leaf

    N1 = m // B
    K = vecs.shape[1]

    # reshape each column into a (B, N1) block: n = r*N1 + c  (r high, c low digit).
    Xb = vecs.reshape(B, N1, K).to(CDTYPE)          # (B, N1, K)

    Cb = dft_matrix(B)                              # (B, B)  radix coefficients
    Yb = torch.einsum('br,rck->bck', Cb, Xb)        # B-point DFT over r

    r = torch.arange(B).unsqueeze(1)                # (B,1)
    c = torch.arange(N1).unsqueeze(0)               # (1,N1)
    Tw = torch.exp(-2j * torch.pi * (r * c).to(CDTYPE) / m)  # (B,N1)
    Zb = Yb * Tw.unsqueeze(-1)                       # twiddle -> (B, N1, K)

    # inner N1-point DFT over the c axis: RECURSE instead of a direct matmul.
    Zc = Zb.permute(1, 0, 2).reshape(N1, B * K)     # column <- (b,k), row <- c
    Wd = small_dft(Zc, N1)                           # (N1, B*K) recurse
    out = Wd.reshape(N1, B, K)                       # (d, b, k), freq index = d*B + b
    return out.reshape(m, K)                         # (m, K)

def base_b_fft(x, N3, N4):
    """N = N3*N4 point DFT via row-column (Cooley-Tukey) decomposition."""
    N = N3 * N4
    # Pack n = k + N4*i  ->  column index i is the HIGH digit (weight N4).
    # This must match the column-first (length-N3 DFT first) order below and the
    # W_N^(i*k) twiddle; the old `reshape(N4, N3).T` used the opposite convention.
    X = x.to(CDTYPE).reshape(N3, N4)                 # (N3, N4)  X[i, k] = x[k + N4*i]

    Xc = small_dft(X, N3)                            # stage 1: column DFTs

    i = torch.arange(N3).unsqueeze(1)
    k = torch.arange(N4).unsqueeze(0)
    Xc = Xc * torch.exp(-2j * torch.pi * (i * k).to(CDTYPE) / N)  # level-1 twiddle

    Xr = small_dft(Xc.T, N4)                         # stage 2: row DFTs  Xr[q,p]=X(p+N3*q)
    # Output index m = p + N3*q  ->  row-major flatten of Xr (shape (N4, N3)) gives
    # exactly q*N3 + p = m. (The old `Xr.T.reshape(-1)` used m = p*N4 + q, wrong.)
    return Xr.reshape(-1)                            # normal-order output

from collections import namedtuple

# A Plan is a *host-side*, static description of how to compute the N-point DFT.
# It is chosen from N (a Python int) before any kernel runs, so the compiled graph has
# NO data-dependent control flow — just fixed-shape matmuls (the TPU-friendly property).
Plan = namedtuple("Plan", "method N N3 N4 L levels")


def _divisors(N):
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
    Pick d | N (1 < d < N) so N = d*(N/d) is a good top-level split:
    prefer BOTH factors divisible by b (each small_dft factorises again), then prefer
    the most balanced split. Returns (N3, N4) or None if N is prime.
    """
    best = None
    for d in _divisors(N):
        if d == 1 or d == N:
            continue
        e = N // d
        n_mult = int(d % b == 0) + int(e % b == 0)
        score = (n_mult, -abs(d - e))
        if best is None or score > best[0]:
            best = (score, (min(d, e), max(d, e)))
    return best[1] if best else None


def _next_pow_b(target, b):
    p = 1
    while p < target:
        p *= b
    return p


def factorize(N, b):
    """
    Static factorisation picker:
      - N <= b       : 'direct'     — one DFT-matrix matmul.
      - N composite  : 'two_factor' — N = N3*N4 via base_b_fft, factors near sqrt(N) and
                       divisible by b when possible (the "b*b*something" case).
      - N prime      : 'bluestein'  — plain zero-padding would change the frequency grid,
                       so reformulate as a length-L convolution, L = next power of b >= 2N-1.
    """
    if N <= b:
        return Plan("direct", N, None, None, None, 0)

    split = _balanced_split(N, b)
    if split is not None:
        N3, N4 = split
        levels = 1 + int(N3 % b == 0) + int(N4 % b == 0)
        return Plan("two_factor", N, N3, N4, None, levels)

    L = _next_pow_b(2 * N - 1, b)
    return Plan("bluestein", N, None, None, L, factorize(L, b).levels)


def base_b_ifft(X, N3, N4):
    """Inverse of base_b_fft: ifft(X) = conj(fft(conj(X))) / N."""
    N = N3 * N4
    return torch.conj(base_b_fft(torch.conj(X), N3, N4)) / N


def bluestein_fft(x, L):
    """
    Exact N-point DFT (any N) as a length-L circular convolution (Bluestein / chirp-z).
    c[m] = exp(-i*pi*m^2/N);  X[k] = c[k] * sum_n (x[n] c[n]) * conj(c[k-n]).
    """
    N = x.shape[0]
    split = _balanced_split(L, B)
    assert split is not None, f"Bluestein length {L} must be composite"
    L3, L4 = split

    n = torch.arange(N, dtype=torch.float64)
    c = torch.exp(-1j * torch.pi * (n ** 2) / N).to(CDTYPE)         # chirp

    a = torch.zeros(L, dtype=CDTYPE)
    a[:N] = x.to(CDTYPE) * c                                         # zero-padded a[n]

    h = torch.zeros(L, dtype=CDTYPE)                                 # circular kernel
    h[:N] = torch.conj(c)                                           # m = 0 .. N-1
    h[L - (N - 1):] = torch.conj(c)[1:].flip(0)                     # m = -(N-1) .. -1

    A = base_b_fft(a, L3, L4)
    H = base_b_fft(h, L3, L4)
    conv = base_b_ifft(A * H, L3, L4)

    return c * conv[:N]                                              # de-chirp, keep first N


def fft(x):
    """Top-level dispatcher: static plan from len(x), then fixed-shape matmuls."""
    N = x.shape[0]
    plan = factorize(N, B)
    if plan.method == "direct":
        return dft_matrix(N) @ x.to(CDTYPE)
    if plan.method == "two_factor":
        return base_b_fft(x, plan.N3, plan.N4)
    return bluestein_fft(x, plan.L)


if __name__ == "__main__":
    def small_dft_depth(m):
        d = 0
        while m > LEAF and m % B == 0:
            m //= B
            d += 1
        return d

    print(f"adaptive depth (LEAF={LEAF}, B={B}) — small_dft split levels vs size:")
    for m in [4, 16, 32, 64, 128, 256, 1024, 4096]:
        print(f"  m={m:<5} split levels = {small_dft_depth(m)}  (leaf size = "
              f"{m // B ** small_dft_depth(m)})")

    print("\naccuracy (unified fft dispatcher):")
    for N in [3, 4, 7, 12, 16, 17, 60, 64, 100, 256, 1024, 4096]:
        x = torch.arange(1, N + 1, dtype=torch.float64)
        err = (fft(x) - torch.fft.fft(x)).abs().max().item()
        print(f"  N={N:<5} max abs error = {err:.2e}")