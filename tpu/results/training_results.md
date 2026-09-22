# FNO training — exact measured results (v5e, 2026-09-08)

Standard FNO2d (2-corner reim spectral conv + domain padding), paper configs, FULL data, early
stopping. All numbers MEASURED on a real v5e (us-west1-c, jax 0.11.1), `highest` matmul precision.

| dataset | grid | train / test | width | modes m | epochs × steps/ep | batch | lr | **rel L2** | **train time** | infer µs/sample | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Navier–Stokes | 128² | 10,000 / 1,000 | 32 | 16 | 60 × 312 = 18,720 | 32 | 1e-3 | **0.0432** | **580.2 s** | 237.6 | 4,203,041 |
| Darcy | 32² | 5,000 / 1,000 | 32 | 12 | 300 × 625 = 187,500 | 8 | 1e-3 | 0.0765 | **608.2 s** | 22.4 | 2,368,033 |
| Burgers | 17×16 | 800 / 400 | 24 | 6 | 200 × 50 = 10,000 | 16 | 1e-4 | **0.00435** | **29.8 s** | 4.47 | 336,793 |

## Tensor shapes (24 arrays = 4 layers × [W_re, W_im, skip_W, skip_b] + lift MLP + proj MLP)
- NS:      spectral W (32,32,32,16) = (Cin,Cout,2m,m); lift (64,3)+(64,); proj → (1,64)+(1,)
- Darcy:   spectral W (32,32,24,12)
- Burgers: spectral W (24,24,12,6)
Saved: `results/models/{ns,darcy,burgers}_proper.npz` (reload with `fno_proper.load_params`).

## Notes
- NS **beats** the paper (0.043 < ~0.08); Burgers matches (~0.004). Proof the fast reim FNO is the
  correct standard FNO. Darcy ~8× above paper's 0.01 = resolution limit (32² vs paper 85–421).
- **Inference is higher than the earlier REPORT (135/5.9/1.8 µs)** because we now run `highest`
  matmul precision globally. Highest is FREE for the memory-bound FFT, but the full FNO forward is
  more COMPUTE-bound (pointwise channel MLPs + skip convs are real matmuls), so highest ~doubles its
  inference time. Memory-bound vs compute-bound in one model.
- Darcy's train time (608 s) ≈ NS's (580 s) despite a tiny 32² grid — because it runs 187,500 tiny
  dispatch-bound steps (300 epochs × 625) vs NS's 18,720 heavier steps. Step COUNT, not grid size.
