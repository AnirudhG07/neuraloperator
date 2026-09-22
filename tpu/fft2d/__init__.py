"""2D radix-B rfft.  Production (fastest-correct): jax_rfft2 (full reim rfft2) and rfft2_partial
(low-mode partial DFT — the FNO transform).  The Pallas kernel lives under fft2d/experimental/."""
from .jax_fft import jax_rfft2, rfft2_partial
