"""
fno_pass — a small, readable FNO2d package with two interchangeable spectral convs:

  reference.py  the plain-JAX baseline: jnp.fft.rfft2 / irfft2 + jnp.einsum (complex64)
  ours.py       our version: the same 2-corner spectral conv done entirely in real/imag
                (cos/sin matmuls, NO complex64)
  layers.py     shared pieces (lift/skip/project MLPs, weight init, Adam, losses)
  data.py       dataset loaders (Darcy / Navier-Stokes / Burgers)
  train.py      baseline trainer (Adam + MSE); trains either forward on any dataset

reference.forward and ours.forward take the SAME weight tree (from layers.init_fno) and the same
modes `m`, so they are directly comparable — run both on one input and the outputs should match.
"""
