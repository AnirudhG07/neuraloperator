# rfft / spectral-conv optimization — literature review (2026-08)

Research pass on how the real FFT (and the FNO spectral conv built on it) can be optimized
beyond what we've already tried (Trick A/B packing, Hermitian output truncation, Karatsuba,
bf16, VMEM-resident Pallas kernel). Grounded in recent papers + classic FFT theory, and
mapped to **our** setting: TPU v5e, **memory/VPU-bound** (AI 24–56 ≪ ridge 240; MXU ~4.6%
of time), FNO keeps only a small `n_modes` (~16), kernels in Pallas.

## TL;DR — for the FNO, the best rfft optimization is to NOT do the rfft

Two 2025 papers independently say: replace `rfft → truncate → weights → irfft` with a
**fused, all-matmul** `partialDFT(n_modes×N) → weights → partial-iDFT(N×n_modes)`, kept
entirely **on-chip**. It hits our exact bottleneck (memory: never materialize the discarded
modes) and is pure MXU matmul. See "Recommendation" at the bottom.

---

## Papers and what each contributes

### DRIFT — Direct Reduced Fourier Transforms (2026)
Link: https://arxiv.org/html/2607.14394
- **Core idea:** reverse the FNO pipeline. Instead of full-FFT-then-discard, compute **only
  the retained low modes** directly via a precomputed **DFT-basis matmul**
  `B[k,j] = W_N^{k·j}`, `k ∈ kept modes`. Applied as GEMMs (cuBLAS), one per dimension.
- **Progressive compression (the key extra trick):** apply the partial-DFT matmul **one
  dimension at a time**; each stage contracts that dim `N → K`, so *later* matmuls run on a
  much smaller tensor. Better cache locality, less memory.
- **Complexity:** `O(K·N)` per dim vs FFT's `O(N log N)`. Reported **13% fewer flops than 4
  cuFFT calls** for a 128×128×64 grid, `K=16` — modest on flops, but…
- **The real win is MEMORY + COMM:** it **never allocates the discarded `(N − 2k_max)`
  coefficients**, killing that global-memory traffic; useful data fraction `(2k_max/N)^d`
  → 0 with resolution, so the waste it removes grows. Communication 97%→<6% of forward time;
  38–64× forward speedup distributed.
- No explicit real-input trick, but the basis-matmul formulation makes real/imag + Hermitian
  trivial to add.

### TurboFNO — fused FFT-GEMM-iFFT on GPU (2025)
Link: https://arxiv.org/abs/2504.11681
- **First fully fused FFT → weight-GEMM → iFFT single kernel.** FFT, filtering, GEMM,
  zero-pad, iFFT consolidated into one kernel → no launch overhead, no HBM round-trips.
- **Mode truncation/pruning built INSIDE the FFT kernel** (not a separate slice/copy) — the
  unused high modes are never produced.
- **On-chip forwarding:** two shared-memory swizzle patterns forward FFT→GEMM→iFFT results
  through shared memory (100% bank utilization) — the GPU analog of our **VMEM residency**.
- **Up to 150% faster than PyTorch/cuBLAS/cuFFT** on A100.
- Custom FFT + GEMM from scratch (matches or beats cuFFT/cuBLAS) — validates hand-writing
  the kernel rather than calling a library.

### FNet — Fourier token mixing (2021)
Link: https://arxiv.org/pdf/2105.03824
- Measured directly: **on TPU, for sequences < ~4096, caching the DFT matrix and doing
  MATMULS is FASTER than FFT** ("TPUs are more highly optimized for matmul than GPUs").
- => Validates our entire **matmul-DFT** premise for the hardware.

### tcFFT — half-precision FFT on tensor cores (2021)
Link: https://arxiv.org/abs/2104.11471
- Half-precision FFT through tensor cores → validates our **bf16** work.
- Flags the exact pain we hit: complex element-wise ops are awkward on matmul-fragment
  layouts → confirms the **real/imag split** is the right representation for a matmul FFT.

### Pruned FFT (classic) — the honest caveat
Links: https://www.fftw.org/pruned.html , https://arxiv.org/pdf/1001.5272
- Pruning a **butterfly** FFT to K outputs is only **O(N log K)** — a *log-factor* saving.
  FFTW: "not recommended unless you want ≤1% of outputs."
- => Classic FFT-pruning is a **dead end**. The win is NOT pruning the butterfly; it's the
  **direct partial DFT as a matmul** (DRIFT), a different algorithm that is MXU-native and
  kills the wasted memory. For us it's a *double* win: our matmul-FFT is far more flops than
  a butterfly, so replacing it with a skinny `n_modes×N` matmul is large, and it cuts bytes.

---

## Techniques ranked for our stack (TPU, memory-bound, small n_modes, Pallas)

| technique | source | what it saves for us | status |
|---|---|---|---|
| **Fused partial-DFT spectral conv** — skip FFT; `n_modes×N` matmul → weights → `N×n_modes`, one VMEM kernel | DRIFT + TurboFNO | **bytes + flops** — never materialize discarded modes; all MXU | not built |
| **Progressive / dim-by-dim partial DFT** (2D/3D FNO) | DRIFT | shrinks tensor each stage → less VMEM, cache-local | not built |
| **Mode truncation built into the kernel** (not a post-slice) | TurboFNO | no extra HBM copy of the full spectrum | partial (pallas_half truncates output) |
| **bf16 partial-DFT matrix / operands** | tcFFT | ½ bytes on the (already skinny) matmul | done for FFT (`dt=BF`), extend to partial DFT |
| real/imag + Hermitian on the partial DFT | ours | real input → cos/sin basis, positive modes only | not built |

Cross-check with our own earlier findings (see `01_rfft.md`, `02_rfft_compare.md`,
`03_pallas_compare.md`):
- We're memory/VPU-bound → optimizing MXU flops (Karatsuba, Trick B) barely helps; the
  papers agree the lever is **memory** (kill discarded-mode traffic) and **fusion** (on-chip).
- Trick A/B, leaf tuning: all move flops, not bytes → confirmed second-order.
- The partial-DFT approach is the first that changes **bytes** at the algorithm level.

## Why the partial DFT is bigger for us than for DRIFT/cuFFT

DRIFT gets only 13% fewer flops vs cuFFT because a native **butterfly** rfft is already
flop-cheap. Our baseline is a **matmul-FFT** (O(N·B·L), far more flops than a butterfly),
and we're **memory-bound**. So the partial DFT wins on both axes for us: (a) it replaces the
expensive matmul-FFT with a cheap `n_modes×N` matmul; (b) it never allocates the discarded
`~N/2 − n_modes` modes — cutting the write, which is our bottleneck. For `n_modes=16`,
`N=16384`: partial DFT ≈ `16·N` vs full FFT `~N·B·L`.

## Recommendation / next step

Build the **fused partial-DFT spectral-conv Pallas kernel**:
```
read real x (N, K)            # once from HBM
  -> partialDFT:  Xhat = C_lo @ x        # C_lo is (n_modes, N) real/imag DFT-basis matrix
  -> weights:     Yhat = einsum('mio,bim->bom', W, Xhat)   # complex channel mix on n_modes
  -> partial-iDFT: y = C_lo^H @ Yhat     # back to (N, K) real
write real y (N, K)           # once to HBM
```
- All matmuls (MXU-native), all in VMEM (TurboFNO-style fusion), only `n_modes` ever
  materialized (DRIFT-style memory saving). Real/imag + bf16 stack on top.
- Measure bytes/flops/time vs the full `rfft → weights → irfft` path in `bench_jax.py`.
- Expected: dramatically fewer bytes (the memory-bound metric) at `n_modes ≪ N` — the
  paper-backed "better way," and simpler than the FFT tricks we've been fighting.

## Sources
- DRIFT: https://arxiv.org/html/2607.14394
- TurboFNO: https://arxiv.org/abs/2504.11681  (PDF: https://arxiv.org/pdf/2504.11681)
- FNet: https://arxiv.org/pdf/2105.03824
- tcFFT: https://arxiv.org/abs/2104.11471
- FFTW pruned FFTs: https://www.fftw.org/pruned.html
- Truncated Fourier transform: https://arxiv.org/pdf/1001.5272
