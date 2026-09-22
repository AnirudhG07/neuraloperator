# Fused-kernel (fft+GEMM+ifft) vs plain-JAX FNO — measured on v5e (2026-09-07)

Config: FNO forward, 128², width 32, m 16, 4 layers. Real v5e (us-west1-c, jax 0.11.1).
Same math/weights at every level. `baseline` = plain-JAX `fno_proper.forward_proper` (XLA);
`A` = per-block Pallas kernel (JAX loops layers); `B` = all blocks in one kernel; `C` = lift+blocks+project in one kernel.

## Inference LATENCY (single instance, batch 1)
| level | µs/sample | relL2 |
|---|---|---|
| **baseline plain-JAX f32** | **372.98** | 0 |
| A per-block Pallas f32 | 787.44 | 4.5e-7 |
| A/B/C bf16 | Mosaic compile FAIL | — |
| B/C f32 | VMEM OOM | — |

## Inference THROUGHPUT (batched, batch 32)
| level | µs/sample | overall ms/32 | relL2 |
|---|---|---|---|
| **baseline plain-JAX f32** | **187.88** | **6.01** | 0 |
| A per-block Pallas f32 | 493.81 | 15.80 | 3.1e-7 |
| A/B/C bf16 | Mosaic compile FAIL | — | — |
| B/C f32 | VMEM OOM (stacked 4-layer weights > 128 MB) | — | — |

## Headline finding
**The hand-fused Pallas kernel LOSES to XLA.** Level A (correct: relL2 ~3e-7) is **2.1–2.6× slower**
than the plain-JAX baseline in both latency and throughput. The fusion hypothesis (TurboFNO-style
win) does **not** hold for this partial-DFT FNO on v5e.

## Why (diagnosis)
1. **Lane padding 8×.** The kept-mode dim `m=16` pads to the 128-lane MXU tile, so every mode-space
   matmul does ~8× wasted work. XLA's batched plain-JAX version amortizes this far better.
2. **B_TILE=1 → tiny per-sample matmuls.** The kernel processes one sample per grid step; XLA runs
   the whole batch at once with much better MXU utilization.
3. **Per-block kernel-launch overhead** ×4 layers.
4. **bf16 in Mosaic doesn't compile** here (all bf16 variants MosaicError) — the mixed bf16-operand /
   f32-accumulate einsums in the partial-DFT kernel aren't Mosaic-legal as written.
5. **B/C f32 OOM** — stacking all 4 layers' spectral weights (each `(Cin,Cout,2m,m)`, m padded to
   128) exceeds the **128 MB** VMEM (measured: `Used 148.12M of 128.00M`).

## VMEM fact (confirmed)
v5e scoped VMEM is **128 MB** (not 16 MB) — the earlier 16 MB was Mosaic's default scoped limit,
raised here via `vmem_limit_bytes`.

## What would be needed to make fusion win (unproven, each costs a TPU run)
- Kill the 8× lane-padding: lay the mode axes so the trailing dim is a multiple of 128.
- Raise B_TILE for MXU utilization (VMEM-bounded).
- Make bf16 Mosaic-legal (cast layout / accumulate handling).
Given XLA already wins by 2×, this is a real kernel-engineering effort, not a quick tweak.
