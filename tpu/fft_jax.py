import jax
import jax.numpy as jnp

# ─── global hyperparameters ───────────────────────────────────────────────
B = 128  # base: size of the systolic array you "built". The radix of every split.

# Leaf threshold = B (split ALL the way down).  The benchmark data (bound_jax /
# bench_jax time sweeps on v5e) showed the fastest choice on the 128-wide MXU is to peel
# a radix-B stage at every level and never leave a fat leaf.  A bigger leaf lowers the
# padded-flop "work" count but moves more HBM bytes, and on a memory-bound TPU the bytes
# decide wall-clock.  So the factorisation is simply  N = B·B·…·B·m,  and the ONLY leaf is
# the final remainder m (the sub-DFT that is no longer divisible by B).

CDTYPE = jnp.complex64   # fp32 components — the working precision for the transform
ACCUM  = jnp.complex64   # fp32 matmul accumulation


def dft_matrix(m):
    """m x m DFT coefficient matrix C, where C[r,c] = W_m^(r*c)."""
    k = jnp.arange(m)
    W = jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m)
    return W.astype(CDTYPE)  # shape (m, m), fp32 complex


def _twiddle_mul(Y, ang):
    """Multiply complex Y by exp(i*ang) using REAL f32 arithmetic."""
    tr = jnp.cos(ang)
    ti = jnp.sin(ang)
    yr = jnp.real(Y)
    yi = jnp.imag(Y)
    return jax.lax.complex(yr * tr - yi * ti, yr * ti + yi * tr)


def small_dft(vecs, m):
    """
    Apply an m-point DFT to each length-m column of `vecs` (shape (m, K)).

    SPLIT ALL THE WAY: peel one radix-B factor (m = B * N1), do the B-point DFT as the
    only matmul, apply the twiddle, then RECURSE on the inner N1-point DFT.  Recursion
    bottoms out at the first sub-DFT that is <= B or not B-divisible — that remainder is
    the single direct leaf.  So depth grows with m automatically:
      m <= B (or not B-divisible)  ->  1 direct DFT matmul   (the leaf)
      m  > B                       ->  split once, recurse on N1 = m/B
    """
    if m <= B or m % B != 0:
        # leaf: one direct DFT matmul, fp32 accumulation
        return jnp.matmul(dft_matrix(m), vecs.astype(CDTYPE),
                          preferred_element_type=ACCUM)

    N1 = m // B
    K = vecs.shape[1]

    # reshape each column into a (B, N1) block: index n = r*N1 + c  (r in [0,B) is
    # the HIGH digit, c in [0,N1) the LOW digit) — this is what reshape(B, N1) gives.
    Xb = vecs.reshape(B, N1, K).astype(CDTYPE)  # (B, N1, K)

    Cb = dft_matrix(B)                          # (B, B)   radix coefficients, for the rows
    # fp32 accumulation on the MXU via preferred_element_type
    Yb = jnp.einsum('br,rck->bck', Cb, Xb,
                    preferred_element_type=ACCUM)  # (B, N1, K)  B-point DFT over r

    # twiddle W_m^(r*c): generate the angle real, multiply on real/imag components
    # (no complex exp, no is-finite/select guards).
    r = jnp.arange(B)[:, None]                  # row index
    c = jnp.arange(N1)[None, :]                 # inner-col index
    ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)  # phase of W_m^(r*c), (B, N1)
    Zb = _twiddle_mul(Yb, ang[:, :, None])      # (B, N1, K)  real-arith twiddle

    # inner N1-point DFT over the c axis: RECURSE (adaptive depth) instead of a direct matmul.
    # flatten so each length-N1 column is a batch column: (N1, B*K).
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)   # column j <- (b, k), row <- c
    Wd = small_dft(Zc, N1)                       # (N1, B*K)  recurse
    out = Wd.reshape(N1, B, K)                   # (d, b, k)  frequency index = d*B + b
    return out.reshape(m, K)                     # (m, K)


@jax.jit
def fft(x):
    """N-point DFT.  Radix-B Cooley-Tukey, split all the way down: N = B·B·…·B·m, one
    radix-B matmul + a real-arithmetic twiddle per level, then a direct leaf DFT on the
    final remainder m.  N is static (from the shape), so the compiled graph is pure
    fixed-shape matmuls with no data-dependent control flow."""
    N = x.shape[0]
    return small_dft(x.astype(CDTYPE)[:, None], N)[:, 0]


@jax.jit
def ifft(x):
    """Inverse via  ifft(x) = conj(fft(conj(x))) / N."""
    N = x.shape[0]
    return jnp.conj(fft(jnp.conj(x.astype(CDTYPE)))) / N


if __name__ == "__main__":
    import numpy as np

    def depth(m):
        d = 0
        while m > B and m % B == 0:
            m //= B
            d += 1
        return d, m                              # (split levels, leaf size)

    print(f"radix-B FFT   B={B}   (split all the way; leaf = final non-B remainder)")
    print("accuracy vs numpy.fft:")
    for N in [4, 16, 64, 128, 256, 640, 1024, 4096, 16384, 262144]:
        x = np.arange(1, N + 1, dtype=np.float32)
        X = np.asarray(fft(jnp.asarray(x)))
        ref = np.fft.fft(x)
        err = float(np.max(np.abs(X - ref)))
        rel = err / float(np.max(np.abs(ref)))
        d, leaf = depth(N)
        print(f"  N={N:<8} levels={d} leaf={leaf:<5} "
              f"max abs err = {err:.2e}   rel = {rel:.2e}")
