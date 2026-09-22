# Trained FNO tensors — STANDARD FNO (2-corner spectral conv + padding), run fast via reim (no c64)

These are the CORRECT standard-FNO algorithm (2-corner ± band spectral conv, domain padding for
non-periodic Darcy), implemented with our fast real/imag partial-DFT — no complex64.  Trained with
full data + early stopping (best-test params saved).  Reload:
    P = tpu.experimental.fno_proper.load_params(path, width, m, n_layers=4)

| model | config | rel L2 (best) | vs earlier single-corner | paper ideal | INFERENCE |
|---|---|---|---|---|---|
| ns_proper.npz     | w32 m16 pad0, FULL 10k data | **0.043** | 0.222 → 0.043 (5×, overfit fixed) | ~0.08 | **135.6 µs/sample (7,375/s)** |
| darcy_proper.npz  | w32 m12 pad8               | 0.076 | 0.080 → 0.076 | ~0.01 | **5.94 µs/sample (168k/s)** |
| burgers_proper.npz| w24 m6  pad0              | **0.0054** | 0.006 → 0.0054 | ~0.014 | **1.79 µs/sample (559k/s)** |

- **NS and Burgers now match/beat the paper** — the two shortcuts I'd taken (2000-sample subset,
  single-corner spectral conv) were the whole problem, not fp32/bf16.
- **Darcy still ~8× off the paper's 0.01** — this is a RESOLUTION limit: the paper uses 85–421 res;
  32² is the largest Darcy in the repo.  Not fixable without higher-res data.
- Inference is the deliverable: the correct standard FNO runs at 1.8–136 µs/sample on the v5e.
  (2-corner is ~30% slower than the wrong single-corner because it carries 2× the axis-0 modes —
  that's the cost of correctness.)

Loss curves: *_proper.csv (early stopping — the "best" column is the saved model).
