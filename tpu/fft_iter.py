"""
fft_iter.py  —  iterative (non-recursive) radix-B Cooley-Tukey FFT, batched.

  fft_iter(x) :  x complex (N, K)  ->  (N, K), an N-point DFT of each of the K columns.
                 x may also be 1-D (N,) -> (N,).
"""
import jax
import jax.numpy as jnp

B = 128
CDTYPE = jnp.complex64


def dft_matrix(m, dtype=CDTYPE):
    """m x m DFT coefficient matrix C[r,c] = W_m^(r*c)."""
    k = jnp.arange(m)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m).astype(dtype)


def fft_iter(x):
    """Iterative radix-B FFT of each column of x (N, K)  (or a 1-D length-N vector)."""
    one_d = (x.ndim == 1)
    if one_d:
        x = x[:, None]
    N, K = x.shape
    cur, m, batch = x.astype(CDTYPE), N, K
    stages = [] # (B, N1) per stage, for the rebuild

    # forward sweep: one radix-B stage per iteration
    while m > B and m % B == 0:
        N1 = m // B
        cur = cur.reshape(B, N1, batch) # split length -> (r, c, batch)
        cur = jnp.einsum('br,rck->bck', dft_matrix(B), cur,
                         preferred_element_type=CDTYPE)  # radix-B DFT over r (axis 0)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        cur = cur * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]  # twiddle
        cur = cur.transpose(1, 0, 2).reshape(N1, B * batch)   # rotate c to axis0; fold b in
        stages.append((B, N1))
        m, batch = N1, B * batch

    # leaf: one direct DFT over the remaining axis 0
    cur = dft_matrix(m) @ cur   # (m, batch)

    # rebuild: undo the folds in reverse order (the digit-reversal)
    out, length = cur, m
    for (Bi, _N1i) in reversed(stages):
        out = out.reshape(length, Bi, -1).reshape(length * Bi, -1)   # merge b_i back (outer)
        length = length * Bi
    return out[:, 0] if one_d else out


@jax.jit
def fft(x):
    return fft_iter(x)


if __name__ == "__main__":
    import numpy as np
    print(f"iterative radix-B FFT  B={B}   (vs numpy.fft)")
    for N in [128, 256, 1024, 16384, 16384 * 2, 128 ** 3]:
        for K in [1, 8]:
            xr = np.random.randn(N, K) + 1j * np.random.randn(N, K)
            got = np.asarray(fft_iter(jnp.asarray(xr)))
            ref = np.fft.fft(xr, axis=0)
            rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
            print(f"  N={N:<8} K={K:<3} rel_err={rel:.2e}  {'OK' if rel < 1e-4 else 'BAD'}")
