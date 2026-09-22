"""1D radix-B FFT.  Production (fastest-correct): jax_rfft (reim real/imag rfft).
Everything else — the c64 reference FFTs, the packing tricks, int8, and the VMEM-resident
Pallas kernels — lives under fft1d/experimental/."""
from .jax_fft import jax_rfft
