# Fast FFT + FNO on a single TPU v5e

Running the **standard** FNO and its FFT as fast as one v5e allows, with a real/imag matmul FFT
instead of `complex64` (the TPU has no native complex type, so assembling one is pure overhead).
This is a speed project, not an algorithm change — see `results/REPORT.md` for the measurements.

## Layout

```
tpu/
  tpu.sh              interactive helper for the Cloud TPU VM (create / upload / run / copyback)
  fft/                the FFT engines that won the bench
    core.py             shared radix-128 pass, real/imag, used by every engine below
    rfft1d.py           rfft_full / rfft_half / rfft_low_modes
    rfft2d.py           rfft2_full / rfft2_full_staged_cols / rfft2_low_modes  (the FNO transform)
  fno/                one FNO, the fast one
    spectral.py         THE spectral conv: 2-corner, partial DFT, real/imag
    model.py            forward pass, save/load params
    layers.py           lift / skip / project, weight init, Adam, losses
    data.py             Darcy / Navier-Stokes / Burgers loaders
    train.py            trainer with early stopping, and inference timing
  tests/              CPU correctness net (the real/imag path vs a complex64 jnp.fft reference)
  results/            measured numbers, reports and trained tensors
  traces/             xprof captures — gitignored, never committed
  experimental/       everything tried and not adopted, kept for the record
    fft/                Pallas kernels, packing tricks, the complex64 reference FFTs
    fno/               the jnp.fft baseline, spectral variants, factorized (F-FNO / TFNO), 1-D
    bench/             the benchmark and profiling scripts
    notes/             derivations and working notes
    data/              dataset summaries
```

Nothing under `experimental/` is imported by `fft/` or `fno/` — the dependency only runs the other
way, so the main tree stays small.

## Use

```bash
python -m pytest tpu/tests -q                 # correctness, CPU, no TPU needed
./tpu/tpu.sh                                  # menu: create a VM, upload, run, copy traces back
```

On the VM (`PYTHONPATH=~/nop`):

```python
from tpu.fno import train
train.run("darcy")                            # train + report rel-L2 and us/sample

import tpu.experimental.bench.bench_jax as bj
bj.compare(case="rfft_1D", Ns=(256, 1024, 4096, 8192, 16384))
```
