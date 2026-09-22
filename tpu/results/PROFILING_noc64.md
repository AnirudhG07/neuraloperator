# TPU FFT — does dropping the complex64 assembly help the tricks? (v5e, xprof device time)

The c64 tricks (`fft-iter_trick*`) run a complex64 FFT and assemble a `complex64` output (a `c64` custom-call).  The `reimtrick_*` variants use the real/imag core `_fft_c` and return `(re, im)` — **no complex64, no c64 custom-call**.  Same math, same N//2 output.

| N | trick | c64 µs | c64 custom% | reim µs | reim custom% | reim/c64 |
|---|---|---|---|---|---|---|
| 256 | trick A | 6.2 | 13% | 6.5 | 0% | 1.04x |
| 256 | trick B | 4.9 | 17% | 4.6 | 0% | 0.94x |
| 1024 | trick A | 15.0 | 12% | 15.4 | 0% | 1.02x |
| 1024 | trick B | 16.8 | 11% | 17.8 | 0% | 1.06x |
| 4096 | trick A | 53.9 | 12% | 46.5 | 0% | 0.86x |
| 4096 | trick B | 60.3 | 11% | 49.1 | 0% | 0.81x |
| 8192 | trick A | 107.8 | 12% | 92.8 | 0% | 0.86x |
| 8192 | trick B | 117.1 | 11% | 93.2 | 0% | 0.80x |
| 16384 | trick A | 214.9 | 12% | 195.5 | 0% | 0.91x |
| 16384 | trick B | 246.7 | 10% | 203.1 | 0% | 0.82x |

