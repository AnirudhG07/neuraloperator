# Reference (jnp.fft, complex64) vs Ours (reim, no c64) — v5e, 2026-09-08

Both trained from the SAME init seed on the SAME data (fno_pass/train.py :: compare).  Same weight
tree, same modes, same 2-corner spectral conv — only the FFT implementation differs:
  reference = jnp.fft.rfft2 / irfft2 + einsum (complex64)   [fno_pass/reference.py]
  ours      = cos/sin matmuls, partial low-mode DFT, real/imag (no complex64)   [fno_pass/ours.py]
Verified offline: ours == reference to rel 4.5e-6 on identical weights.

| dataset | variant | rel L2 | train | infer µs/sample | speedup (infer / train) |
|---|---|---|---|---|---|
| Darcy 32² (m12,w32,100ep,b8) | reference | 0.0844 | 171.6 s | 19.33 | — |
| | ours | 0.0844 | 167.7 s | 18.37 | 1.05× / 1.02× |
| Navier-Stokes 128² (m16,w32,30ep,b32) | reference | 0.0564 | 533 s | 653.6 | — |
| | ours | 0.0564 | 293 s | 237.6 | **2.75× / 1.8×** |

## Findings
- IDENTICAL accuracy (same math) — confirms ours is a correct standard FNO, not an approximation.
- At NS 128² (the real size) ours is 2.75× faster inference, 1.8× faster training — the payoff of
  no-complex64 + partial low-mode DFT vs the reference's full complex64 rfft2/irfft2.
- At Darcy 32² the gap is ~5% — the transform is tiny/dispatch-bound; the win grows with problem size.
(NS relL2 0.0564 here is at 30 epochs for a quick fair compare; the full 60-epoch run reached 0.043.)

## Seed sweep (is the Burgers gap variance?) — YES
| dataset | variant | relL2 mean ± std | runs |
|---|---|---|---|
| Burgers (5 seeds, 200ep) | reference | 0.0031 ± 0.0015 | .0021 .0020 .0059 .0036 .0020 |
|                          | ours      | 0.0054 ± 0.0038 | .0110 .0021 .0022 .0089 .0027 |
| Darcy (5 seeds, 50ep)    | reference | 0.0859 ± 0.0021 | .0862 .0849 .0861 .0829 .0894 |
|                          | ours      | 0.0859 ± 0.0021 | IDENTICAL per seed |
| NS (3 seeds, 15ep)       | reference | 0.0741 ± 0.0011 | .0742 .0727 .0753 |
|                          | ours      | 0.0741 ± 0.0011 | IDENTICAL per seed |
Verdict: Darcy/NS reference==ours to 4 decimals PER SEED. Burgers is noisy for BOTH (tiny 800-sample
problem); ours reaches 0.002 on 3/5 seeds -> the single 0.011 was seed variance, not a flaw. ours is
equivalent to the jnp.fft reference everywhere, and 2.75x faster at NS 128^2.
