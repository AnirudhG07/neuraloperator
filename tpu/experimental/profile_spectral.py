"""
profile_spectral.py — run ON the v5e.  Captures xprof device traces of the two 2-corner spectral
convs so we can see WHERE the time goes: the (2m×H) matmul (current) vs the (m+1×H) matmul + the
flip/concatenate conjugation (hermitian).  Traces saved to ~/prof_{current,herm} and pushed to GCS.
"""
import os
import jax, jax.numpy as jnp, numpy as np

import tpu.fft_core  # noqa: F401
from tpu.experimental.fno_proper import spectral_2corner, spectral_2corner_herm, init_2corner

print("jax", jax.__version__, "|", jax.devices(), flush=True)
B, C, H, W, m = 32, 32, 128, 128, 16                    # one spectral-conv layer at NS width/grid
w = init_2corner(jax.random.PRNGKey(0), C, C, m)
x = jnp.asarray(np.random.randn(B, C, H, W).astype(np.float32))

for name, spec in [("current", spectral_2corner), ("herm", spectral_2corner_herm)]:
    fn = jax.jit(lambda z, spec=spec: spec(w, z, m))
    jax.block_until_ready(fn(x))                        # compile OUTSIDE the trace
    ld = os.path.expanduser(f"~/prof_{name}")
    with jax.profiler.trace(ld):
        for _ in range(30):
            jax.block_until_ready(fn(x))
    # also dump the compiled HLO + a flops/bytes summary (cheap, offline-parseable)
    comp = fn.lower(x).compile()
    ca = comp.cost_analysis()
    fl = ca.get("flops", 0) if isinstance(ca, dict) else 0
    by = ca.get("bytes accessed", 0) if isinstance(ca, dict) else 0
    print(f"PROFILED {name}: flops={fl:.3e}  bytes_accessed={by:.3e}  -> {ld}", flush=True)
print("PROFILE_DONE", flush=True)
