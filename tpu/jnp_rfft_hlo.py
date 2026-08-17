"""
jnp_rfft_hlo.py — a from-scratch reconstruction of what jnp.fft.rfft compiles to ON TPU,
read straight off the captured xplane HLO of the `jnp.rfft` engine (N=1024, K=256 -> (513,256)).

It is a FOUR-STEP (Bailey) FFT done as matmuls: N = N1*N2 = 128*8.  Each JAX line is tagged with
the HLO op it corresponds to in that trace.  This is why jnp.rfft profiles ~72% MXU: on TPU the
FFT itself IS matmuls (`convolution fusion`), not a butterfly.

Standalone study of the reference pipeline — does NOT touch our kernels.  Complex here for
readability; on TPU each complex matmul splits into 2-3 REAL `convolution fusion`s (real*real,
real*imag, twiddle-fused), which is why the trace shows 6 of them for the 2 stages.
"""
import jax.numpy as jnp

C = jnp.complex64


def rfft_hlo(xr, N1=128, N2=8):
    """Real (N=N1*N2, K) -> rfft (N//2+1, K).  Reconstructs jnp.rfft's TPU four-step pipeline."""
    N, K = xr.shape  # N=1024, K=256

    # (0) twiddle grid W_N^{n1*k2}, shape (N1,N2)=(128,8)
    n1 = jnp.arange(N1)[:, None]
    k2 = jnp.arange(N2)[None, :]
    tw = jnp.exp(-2j * jnp.pi * n1 * k2 / N).astype(C)  # HLO: fusion.43(cos)+fusion.42(sin) f32[128,8]

    # (1) view the length-N signal as an (n2, n1) matrix, batched over K
    A = xr.astype(C).reshape(N2, N1, K)  # HLO: copy (data formatting) f32[256,128,8]

    # (2) inner DFT over n2 (size N2=8) — first radix stage
    F2 = jnp.exp(-2j * jnp.pi * jnp.outer(jnp.arange(N2), jnp.arange(N2)) / N2).astype(C)
    B = jnp.einsum('ca,anK->cnK', F2, A)  # HLO: fusion.11 (convolution fusion, DFT_8), B[k2,n1,K]

    # (3) twiddle multiply W_N^{n1*k2} — on TPU FUSED into the matmul epilogue
    B = B * tw.T[:, :, None]  # HLO: convolution_subtract_fusion + fusion.63 (twiddle fused into the conv)

    # (4) outer DFT over n1 (size N1=128) — second radix stage; the corner-turn is copy.7/copy.8
    F1 = jnp.exp(-2j * jnp.pi * jnp.outer(jnp.arange(N1), jnp.arange(N1)) / N1).astype(C)
    Xp = jnp.einsum('cnK,nd->cdK', B, F1)  # HLO: fusion.6/fusion.3/fusion.2 (conv, DFT_128) + copy.7/copy.8 (corner-turn {0,2,1})

    # (5) reorder to natural bin index X[N2*k1 + k2]
    Xf = Xp.transpose(1, 0, 2).reshape(N, K)  # HLO: custom-call (transpose / digit-reversal)

    # (6) keep the Hermitian half [0:N/2+1] (real input -> conjugate-symmetric spectrum)
    return Xf[:N // 2 + 1]  # HLO: slice_bitcast_fusion + slice_bitcast_fusion.1 f32[513,256]


if __name__ == "__main__":
    import numpy as np
    print("reconstruction of jnp.rfft (four-step matmul FFT)  vs numpy.fft.rfft")
    for (a, b) in [(128, 8), (8, 128), (32, 32), (64, 16)]:
        N = a * b
        xr = np.random.randn(N, 4).astype(np.float32)
        got = np.asarray(rfft_hlo(jnp.asarray(xr), a, b))
        ref = np.fft.rfft(xr, axis=0)
        rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        print(f"  N={N}=({a}x{b})  rel_err={rel:.2e}  {'OK' if rel < 1e-4 else 'BAD'}")
