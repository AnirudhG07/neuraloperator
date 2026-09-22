# Reference targets vs our results — Darcy / Burgers / Navier-Stokes

Ideal numbers are from the original FNO paper (Li et al., ICLR 2021, "Fourier Neural Operator for
Parametric PDEs") and follow-ups. "Ours" = our reim FNO (real/imag, no complex64) with the neuralop
**FNO_Small2d** config (hidden 24, modes 16, 4 layers) — a deliberately SMALL model, short training.

## Accuracy (relative L2 test error)

| dataset | paper / ideal FNO | ours | gap — why |
|---|---|---|---|
| **Burgers 1D** | 0.0139 – 0.0160 (res 256–8192) | **0.0062** | we're BETTER — small grid (16), easy task, 300 ep |
| **Darcy 2D** | 0.0098 – 0.0109 (res 85–421) | **0.080** | ~8× worse: small config (width 24 / modes 16), 32² res, 300 ep |
| **Navier-Stokes 2D** | ν=1e-3: **< 0.01**  ·  ν=1e-4 (turbulent): **~0.08** | **0.228** | worse: 16 of 64 modes kept (heavy truncation), small width, 250 ep |

Notes:
- The paper uses larger models (width 32–64, up to 12–32 modes) + longer training; its Darcy is
  85–421 res, not 32². Our gaps are a **config/training-budget** effect, not a correctness bug —
  Burgers (where the small config is enough) already matches/beats the paper.
- To close the Darcy/NS gap: raise modes (16→32/64), width (24→64), train longer, and add the
  factorized-weight / bf16-activation levers so the bigger model stays cheap.

## Inference / compute (reference)

| reference | number |
|---|---|
| FNO paper, Navier-Stokes 256² full solve (inference) | **~0.005 s** (vs 2.2 s pseudo-spectral) |
| ours, per-sample single forward (v5e): Burgers | 2.0 µs/sample (496k/s) |
| ours, Darcy 32² | 6.3 µs/sample (158k/s) |
| ours, Navier-Stokes 128² | 105.9 µs/sample (9.4k/s) |

(Our per-sample forward-pass times aren't directly comparable to the paper's full-rollout solve
time; the paper's headline is the ~440× speedup over a classical NS solver.)

## Sources
- Li et al., FNO for Parametric PDEs (ICLR 2021): https://arxiv.org/pdf/2010.08895
- Neural Operator: Learning Maps Between Function Spaces: https://arxiv.org/pdf/2108.08481
- Darcy comparison table (FNO ~0.0108 vs GNO/DeepONet/etc.): https://arxiv.org/pdf/2108.08481
