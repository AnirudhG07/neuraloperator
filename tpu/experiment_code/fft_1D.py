"""
FFT variants for TPU A/B testing.
Radix-B Cooley-Tukey, split all the way. Three toggles:
  ORDER      : 'col_first' or 'row_first'  (Q1)
  FUSE_TWIDDLE : True/False  -- bake twiddle into the DFT matrix (Q2)
  USE_TRANSPOSE: True/False  -- explicit transpose vs in-place einsum (Q3)
Pure JAX, jax.jit on the top-level entry points.
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

B = 128
CDTYPE = jnp.complex64
ACCUM  = jnp.complex64


def dft_matrix(m):
    k = jnp.arange(m)
    W = jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / m)
    return W.astype(CDTYPE)

def small_dft_baseline(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)
    N1 = m // B
    K = vecs.shape[1]
    # reshaping
    Xb = vecs.reshape(B, N1, K)
    # radix coefficient matrix for B-DFT
    Cb = dft_matrix(B)
    # Column DFT of size B
    Yb = jnp.einsum('br,rck->bck', Cb, Xb, preferred_element_type=ACCUM)

    r = jnp.arange(B)[:, None]
    c = jnp.arange(N1)[None, :]
    ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)
    tw = jnp.exp(1j * ang).astype(CDTYPE)[:, :, None]
    # Twiddle multiplication: Y[b,c,k] * W_m^(b*c)
    Zb = Yb * tw
    # explicit transpose then recurse
    Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)
    # recurse on the row DFT of size N1
    Wd = small_dft_baseline(Zc, N1)
    out = Wd.reshape(N1, B, K)
    return out.reshape(m, K)


# NO explicit transpose. Do the inner DFT in-place via einsum
def small_dft_no_transpose(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)
    N1 = m // B
    K = vecs.shape[1]
    # reshaping
    Xb = vecs.reshape(B, N1, K)
    # radix coefficient matrix for B-DFT
    Cb = dft_matrix(B)
    # Column DFT of size B
    Yb = jnp.einsum('br,rck->bck', Cb, Xb, preferred_element_type=ACCUM)

    r = jnp.arange(B)[:, None]
    c = jnp.arange(N1)[None, :]
    ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)
    tw = jnp.exp(1j * ang).astype(CDTYPE)[:, :, None]
    # Twiddle multiplication: Y[b,c,k] * W_m^(b*c)
    Zb = Yb * tw 
    # inner N1-DFT WITHOUT transpose: contract the c (N1) axis in place.
    # only valid for a DIRECT inner DFT (N1 <= B). If N1 > B we must recurse,
    # which needs the (N1, ...) layout -> fall back to transpose there.
    if N1 <= B:
        C1 = dft_matrix(N1)                        # (N1, N1)
        Wd = jnp.einsum('bck,dc->bdk', Zb, C1, preferred_element_type=ACCUM)  # (B, N1, K)
        # freq index = d*B + b  -> need output[m] with d outer, b inner
        out = jnp.transpose(Wd, (1, 0, 2))         # (N1, B, K)  -- metadata swap for reshape
        return out.reshape(m, K)
    else:
        # deep case: must recurse, needs (N1, B*K) layout
        Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)
        Wd = small_dft_no_transpose(Zc, N1)
        out = Wd.reshape(N1, B, K)
        return out.reshape(m, K)


# ─────────────────────────────────────────────────────────────
#  Q2 variant: FUSE twiddle into the inner DFT matrix.
#  W[b,d] = sum_c Ctwid_b[d,c] * Y[b,c],  Ctwid_b[d,c] = C_N1[d,c] * W_m^(b*c)
#  (only for the direct inner DFT, N1 <= B)
# ─────────────────────────────────────────────────────────────
def small_dft_fused(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)
    N1 = m // B
    K = vecs.shape[1]
    Xb = vecs.reshape(B, N1, K)
    Cb = dft_matrix(B)
    Yb = jnp.einsum('br,rck->bck', Cb, Xb, preferred_element_type=ACCUM)  # (B,N1,K)
    if N1 <= B:
        # build the twiddled inner-DFT tensor: for each b, C_N1[d,c]*W_m^(b*c)
        d = jnp.arange(N1)[:, None]                # (N1,1)
        c = jnp.arange(N1)[None, :]                # (1,N1)
        C1 = jnp.exp(-2j * jnp.pi * (d * c) / N1)  # (N1,N1) inner DFT
        b = jnp.arange(B)[:, None, None]           # (B,1,1)
        cc = jnp.arange(N1)[None, None, :]         # (1,1,N1)
        tw = jnp.exp(-2j * jnp.pi * (b * cc).astype(jnp.float32) / m)  # (B,1,N1)
        Ctwid = (C1[None] * tw).astype(CDTYPE)     # (B, N1, N1)  twiddle baked in
        # contract c: W[b,d,k] = sum_c Ctwid[b,d,c] * Y[b,c,k]
        Wd = jnp.einsum('bdc,bck->bdk', Ctwid, Yb, preferred_element_type=ACCUM)
        out = jnp.transpose(Wd, (1, 0, 2))         # (N1,B,K)
        return out.reshape(m, K)
    else:
        # deep case: fall back to explicit twiddle + recurse
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)
        tw = jnp.exp(1j * ang).astype(CDTYPE)[:, :, None]
        Zb = Yb * tw
        Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)
        Wd = small_dft_fused(Zc, N1)
        out = Wd.reshape(N1, B, K)
        return out.reshape(m, K)




# ─────────────────────────────────────────────────────────────
#  COMBINED: fused twiddle + no transpose  (Q2 + Q3 together)
#  - twiddle baked into the inner DFT matrix  (no VPU twiddle pass)
#  - inner DFT contracted in place            (no transpose copy)
# ─────────────────────────────────────────────────────────────
def small_dft_fused_no_transpose(vecs, m):
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=ACCUM)
    N1 = m // B
    K = vecs.shape[1]
    Xb = vecs.reshape(B, N1, K)
    Cb = dft_matrix(B)
    Yb = jnp.einsum('br,rck->bck', Cb, Xb, preferred_element_type=ACCUM)  # (B,N1,K)
    if N1 <= B:
        # twiddle baked into inner DFT tensor, per row-group b
        d = jnp.arange(N1)[:, None]
        c = jnp.arange(N1)[None, :]
        C1 = jnp.exp(-2j * jnp.pi * (d * c) / N1)             # (N1,N1)
        b = jnp.arange(B)[:, None, None]
        cc = jnp.arange(N1)[None, None, :]
        tw = jnp.exp(-2j * jnp.pi * (b * cc).astype(jnp.float32) / m)  # (B,1,N1)
        Ctwid = (C1[None] * tw).astype(CDTYPE)                # (B,N1,N1)
        # contract c IN PLACE (no transpose): W[b,d,k] = sum_c Ctwid[b,d,c]*Y[b,c,k]
        Wd = jnp.einsum('bdc,bck->bdk', Ctwid, Yb, preferred_element_type=ACCUM)  # (B,N1,K)
        out = jnp.transpose(Wd, (1, 0, 2))                    # metadata-only for reshape
        return out.reshape(m, K)
    else:
        # deep case: explicit twiddle + recurse (needs (N1,...) layout)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        ang = (-2.0 * jnp.pi / m) * (r * c).astype(jnp.float32)
        tw = jnp.exp(1j * ang).astype(CDTYPE)[:, :, None]
        Zb = Yb * tw
        Zc = jnp.transpose(Zb, (1, 0, 2)).reshape(N1, B * K)
        Wd = small_dft_fused_no_transpose(Zc, N1)
        out = Wd.reshape(N1, B, K)
        return out.reshape(m, K)


# top-level entry points, jitted
@partial(jax.jit, static_argnums=(1,))
def fft_dispatch(x, variant):
    N = x.shape[0]
    v = x.astype(CDTYPE)[:, None]
    if variant == 'baseline':
        return small_dft_baseline(v, N)[:, 0]
    elif variant == 'no_transpose':
        return small_dft_no_transpose(v, N)[:, 0]
    elif variant == 'fused':
        return small_dft_fused(v, N)[:, 0]
    elif variant == 'fused_no_transpose':
        return small_dft_fused_no_transpose(v, N)[:, 0]


# ─────────────────────────────────────────────────────────────
#  correctness check vs numpy
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"radix-B FFT variants  B={B}")
    for variant in ['baseline', 'no_transpose', 'fused', 'fused_no_transpose']:
        print(f"\n=== variant: {variant} ===")
        for N in [4, 64, 128, 256, 1024, 16384, 16384*4]:
            x = np.arange(1, N + 1, dtype=np.float32)
            try:
                X = np.asarray(fft_dispatch(jnp.asarray(x), variant))
                ref = np.fft.fft(x)
                rel = float(np.max(np.abs(X - ref))) / float(np.max(np.abs(ref)))
                status = "OK " if rel < 1e-4 else "BAD"
                print(f"  N={N:<7} rel_err={rel:.2e}  {status}")
            except Exception as e:
                print(f"  N={N:<7} ERROR: {str(e)[:60]}")


# ─────────────────────────────────────────────────────────────
#  Benchmark harness (run on TPU). Times each variant, reports HBM-ish
#  proxy via jax cost analysis + wall clock.
# ─────────────────────────────────────────────────────────────
def benchmark(Ns=None, variants=None, reps=50):
    import time
    if Ns is None:
        Ns = [1024, 16384, 65536, 262144, 1048576, 4194304]
    if variants is None:
        variants = ['baseline', 'no_transpose', 'fused', 'fused_no_transpose']

    print(f"\n{'='*72}")
    print(f"BENCHMARK  (device: {jax.devices()[0].platform})")
    print(f"{'='*72}")
    header = f"{'N':>10}" + "".join(f"{v[:16]:>18}" for v in variants)
    print(header)

    for N in Ns:
        x = jnp.asarray(np.random.randn(N).astype(np.float32))
        row = f"{N:>10}"
        for variant in variants:
            try:
                f = lambda x: fft_dispatch(x, variant)
                r = f(x); r.block_until_ready()          # warmup/compile
                t0 = time.perf_counter()
                for _ in range(reps):
                    r = f(x)
                r.block_until_ready()
                us = (time.perf_counter() - t0) / reps * 1e6
                row += f"{us:>15.1f}us"
            except Exception as e:
                row += f"{'ERR':>18}"
        print(row)

    # cost analysis (FLOPs + bytes estimate from XLA) for one size
    print(f"\n--- XLA cost analysis (N={Ns[len(Ns)//2]}) ---")
    N = Ns[len(Ns)//2]
    x = jnp.asarray(np.random.randn(N).astype(np.float32))
    for variant in variants:
        try:
            f = jax.jit(lambda x: fft_dispatch(x, variant))
            c = f.lower(x).compile().cost_analysis()
            flops = c.get('flops', 0)
            bytes_acc = c.get('bytes accessed', 0)
            print(f"  {variant:20s} flops={flops:.3e}  bytes={bytes_acc:.3e}  AI={flops/max(bytes_acc,1):.1f}")
        except Exception as e:
            print(f"  {variant:20s} cost analysis failed: {str(e)[:40]}")


if __name__ == "__main__" and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'bench':
    benchmark()