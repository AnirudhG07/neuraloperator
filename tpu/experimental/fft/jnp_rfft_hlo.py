"""
jnp_rfft_hlo.py — the FOUR-STEP (Bailey) matmul rfft that reconstructs jnp.rfft's TPU pipeline,
but carried entirely in REAL/IMAG (no complex64, no `c64` custom-call).

N = N1*N2.  Real input -> inner DFT_N2 (real -> 2 matmuls) -> twiddle (fused elementwise) ->
outer DFT_N1 (complex -> 3 real matmuls, Gauss/Karatsuba) -> digit-reversal reshape.  Returns
(yr, yi) as two f32 arrays — measured ~10-15% faster than the complex64 version because it drops
the final complex-assembly custom-call (see tpu/profiling/PROFILING_noc64.md).
"""
import jax.numpy as jnp

from ...fft.core import ACC, F, dft_matrix_reim as _CB_reim


def rfft_hlo(xr, N1=128, N2=8, dt=F):
    """Real (N=N1*N2, K) -> rfft (N//2+1, K) as (yr, yi).  Real/imag four-step; no complex64."""
    N, K = xr.shape

    # twiddle W_N^{n1*k2}, shape (N1, N2) -> as (N2, N1) to match B's [k2, n1] layout
    n1 = jnp.arange(N1)[:, None]
    k2 = jnp.arange(N2)[None, :]
    ang = -2.0 * jnp.pi * n1 * k2 / N
    twr = jnp.cos(ang).T[:, :, None].astype(F)          # (N2, N1, 1)
    twi = jnp.sin(ang).T[:, :, None].astype(F)

    A = xr.astype(dt).reshape(N2, N1, K)                 # (n2, n1, K)

    # inner DFT over n2 (size N2) — real input, so only 2 matmuls
    C2r, C2i, _ = _CB_reim(N2, dt)
    Br = jnp.einsum('ca,ank->cnk', C2r, A, preferred_element_type=ACC)   # (k2, n1, K)
    Bi = jnp.einsum('ca,ank->cnk', C2i, A, preferred_element_type=ACC)

    # twiddle multiply (elementwise, fuses into the matmul epilogue)
    Br, Bi = Br * twr - Bi * twi, Br * twi + Bi * twr

    # outer DFT over n1 (size N1) — complex x complex -> 3 real matmuls (Gauss)
    C1r, C1i, _ = _CB_reim(N1, dt)
    mm = lambda M, Z: jnp.einsum('cnk,nd->cdk', Z, M, preferred_element_type=ACC)
    P1, P2 = mm(C1r, Br), mm(C1i, Bi)
    P3 = mm((C1r + C1i).astype(dt), Br + Bi)
    Xpr, Xpi = P1 - P2, P3 - P1 - P2                     # (k2, k1, K)

    # reorder to natural bin X[N2*k1 + k2], keep the Hermitian half
    Xr = Xpr.transpose(1, 0, 2).reshape(N, K)
    Xi = Xpi.transpose(1, 0, 2).reshape(N, K)
    return Xr[:N // 2 + 1].astype(dt), Xi[:N // 2 + 1].astype(dt)


if __name__ == "__main__":
    import numpy as np
    print("real/imag four-step rfft (no complex64)  vs numpy.fft.rfft")
    for (a, b) in [(128, 8), (8, 128), (32, 32), (64, 16)]:
        N = a * b
        xr = np.random.randn(N, 4).astype(np.float32)
        yr, yi = rfft_hlo(jnp.asarray(xr), a, b)
        got = np.asarray(yr) + 1j * np.asarray(yi)
        ref = np.fft.rfft(xr, axis=0)
        rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        print(f"  N={N}=({a}x{b})  rel_err={rel:.2e}  {'OK' if rel < 1e-4 else 'BAD'}")
