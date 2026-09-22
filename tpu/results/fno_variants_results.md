# FNO efficiency variants — v5e results (experimental/fno_variants.py)

All variants keep the spectral transform real/imag (no complex64).  Accuracy = Darcy-16 rel L2
after a short 60-step run (indicative, not converged).  Not production — production stays in
tpu/fno_reim_2d.py.

## Parameter efficiency + accuracy (Darcy-16, width 24, modes 8, 4 layers)

| variant | params | rel L2 | note |
|---|---|---|---|
| dense (baseline) | ~300k | ~0.20 | full (Cin,Cout,m,m) spectral weight |
| **separable (F-FNO)** | **78,745** | **0.172** | one 1-D weight per axis, summed — 4× fewer params, *best* accuracy here |
| **CP rank-8 (TFNO)** | **9,113** | 0.194 | rank-8 CP factorization of the spectral tensor — 33× fewer params |
| **CP rank-4 (TFNO)** | **7,065** | 0.207 | rank-4 — 42× fewer params, small accuracy cost |

→ Factorized spectral weights (F-FNO separable, TFNO CP) cut the spectral-conv parameters
**4–42×** with comparable accuracy — directly shrinks the channel-mix GEMM (params + bytes) on a
memory-bound model.

## bf16 ACTIVATIONS — the throughput lever (32×32, batch 512)

| forward | µs / sample | throughput | 
|---|---|---|
| f32 activations | 4.24 | 235,672 samples/s |
| **bf16 activations** | **3.34** | **299,671 samples/s** |

→ **+27% throughput** (3.34 vs 4.24 µs/sample) from carrying **activations** in bf16 — this is the
win that weights-only bf16 did NOT give (weights-only was ~0% because activations dominate the HBM
traffic).  Confirms the model is memory-bound: halving activation bytes ≈ proportional speedup.

## Techniques implemented (experimental/fno_variants.py)
- separable spectral weights (F-FNO), CP-factorized spectral weights (TFNO)
- full-bf16-activations forward (the throughput lever above)
- incremental-modes curriculum (mode-unmasking schedule — training-cost lever)
NOT implemented here: the fused FFT-GEMM-iFFT single kernel (being done separately).

## bf16 ACCURACY (the missing half) — Darcy-16, paired, same seed (40-step CPU)

| forward | rel L2 |
|---|---|
| f32 activations | 0.1965 |
| bf16 activations | 0.1973  (delta **+0.0008**) |

→ **bf16 costs essentially nothing on accuracy** (+0.0008 rel L2 ≈ 0.4% relative, within run noise) while
giving **+27% throughput**.  So on this memory-bound FNO, bf16 activations are close to a free lunch:
big throughput win, negligible accuracy change.

## Incremental modes — Darcy-16 (grow active modes 2→8 over training)

| training | rel L2 |
|---|---|
| fixed modes (m=8) | 0.1965 |
| **incremental (2→8)** | **0.1969** |

→ **Same accuracy** as fixed-mode, but early epochs run with far fewer active modes → cheaper
spectral work early in training (a training-cost lever, not an inference-time one).

## Layout / einsum-reorder — implemented + correct; device-time TBD

`forward_reorder` (reordered forward-DFT contractions + `optimize='optimal'` spectral mix) is
implemented and **numerically matches the baseline (rel_err 3e-7)**.  Whether it actually removes the
corner-turn copy is an **HLO/device-time question** that only shows on a TPU run — those runs kept
failing on VM issues (backend-init / single-tenant lock), so the reorder speed delta is not yet
measured.  (Also likely superseded by the fused FFT-GEMM-iFFT kernel being done separately.)
