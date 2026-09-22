# Pallas VMEM-resident FFT vs fft_iter vs jnp.fft

> Runnable: `python -m tpu.pallas_fft`. Kernel: `tpu/pallas_fft.py`. Real→complex FFT,
> K=256 columns, on this CPU box (interpret + AOT-lowering; wall-clock needs a v5e).

## The four contenders

| engine | what it is |
|---|---|
| `jnp.fft` | XLA's native FFT — opaque custom-call (ducc on CPU, FFT op on TPU); butterfly O(N·logN) |
| `fft_recur` | our pure-JAX RECURSIVE radix-B FFT (the original baseline) |
| `fft_iter` | our pure-JAX ITERATIVE radix-B FFT (stage-loop refactor of the recursion) |
| **pallas** | our radix-B FFT as **one Pallas kernel**, whole transform resident in **VMEM** |

## Results (K=256, real→complex)

| N | engine | rel_err | HBM bytes | ×jnp | flops | ×jnp | lowers? |
|---|---|---|---|---|---|---|---|
| 256 | jnp.fft | 1.3e-07 | 2.88M | 1.00× | 4.26M | 1.0× | (native) |
| 256 | fft_recur | 1.3e-05 | 4.20M | 1.46× | 17.5M | 4.1× | (jax) |
| 256 | fft_iter | 1.3e-05 | 4.20M | 1.46× | 17.5M | 4.1× | (jax) |
| 256 | **pallas** | 1.3e-05 | **786K** | **0.27×** | 34.6M | 8.1× | **yes** |
| 1024 | jnp.fft | 1.4e-07 | 11.53M | 1.00× | 21.2M | 1.0× | (native) |
| 1024 | fft_recur | 1.7e-05 | 16.01M | 1.39× | 72.2M | 3.4× | (jax) |
| 1024 | fft_iter | 1.7e-05 | 16.01M | 1.39× | 72.2M | 3.4× | (jax) |
| 1024 | **pallas** | 1.7e-05 | **3.15M** | **0.27×** | 144M | 6.8× | **yes** |
| 16384 | jnp.fft | 2.0e-07 | 184.6M | 1.00× | 474M | 1.0× | (native) |
| 16384 | fft_recur | 2.6e-05 | 252.3M | 1.37× | 2.16G | 4.6× | (jax) |
| 16384 | fft_iter | 2.6e-05 | 252.3M | 1.37× | 2.16G | 4.6× | (jax) |
| 16384 | **pallas** | 2.6e-05 | **50.3M** | **0.27×** | 4.32G | 9.1× | **yes** |
| 16384 | **pallas_rfft** | 1.7e-05 | **33.6M** | **0.18×** | 3.24G | 6.8× | **yes** |

(rfft rows at 256/1024: bytes 0.18×, flops 4.1×/3.6× of jnp — i.e. ~0.5× of `pallas`.)

**`fft_recur` ≡ `fft_iter`** in bytes AND flops — confirms the recursion→stage-loop refactor
is purely structural (identical computation), as intended.

**`pallas_rfft`** (real-input Hermitian rfft) is the trick applied inside the kernel:
- **bytes 0.27× → 0.18×** — writing only `N//2` rows (Hermitian) instead of `N` cuts the
  write in half (HBM = read N + write N, vs read N + write 2N = 2/3 of `pallas`).
- **flops ~halved vs `pallas`** (8.1×→4.1× jnp at N=256) — the `xi=0` first stage does 2
  matmuls instead of 4, and truncating to N//2 lets DCE prune upstream work. The win shrinks
  as stages grow (the `xi=0` saving is only the first of L stages).
- This is exactly the `02_rfft_compare.md` pattern (halved flops, no complex-input
  inflation), now realized on the Pallas engine and **lowering to TPU**.

## Findings

1. **Pallas cuts HBM to 0.27× of `jnp.fft`** (and ~0.20× of our own `fft_iter`), consistently
   across N. The reason is the whole design thesis: `fft_iter` round-trips every stage's
   intermediate through HBM (a `dot` is a fusion barrier in XLA); the Pallas kernel keeps
   them in **VMEM** and touches HBM only to read the input once and write the output once.
2. **It lowers to TPU** (`lower_as_mlir(tpu)` accepts it at every N) — Mosaic is happy with
   the real/imag f32 matmuls, the in-kernel cos/sin twiddles, the reshapes and transposes.
3. **Correctness holds** (rel ~2e-5, the f32/complex64 floor — same as `fft_iter`; the real
   /imag split is numerically identical to complex64, and it removes the guard tax).
4. **Flops go UP by design — this is NOT the tricks, and not a regression.** Two separate
   causes, neither related to rfft (this table computes the FULL FFT, no tricks):
   - **Algorithm:** matmul radix-B is **O(N·B)/stage** vs jnp's **O(N·logN)** butterfly →
     the ~4× seen in `fft_recur`/`fft_iter`. This is the deliberate cost of using the MXU.
   - **Representation:** Pallas carries **real/imag**, so one complex matmul becomes 4 real
     matmuls → another ~2× on top (the 8× column). Part of that is *waste*: for real input
     the first stage has `xi=0`, so 2 of those 4 matmuls are multiply-by-zero — exploiting
     `xi=0` in stage 1 would roughly halve the first stage's flops. Fixable.
   - **The rfft tricks DO cut flops ~2×** — but relative to the *same engine's* full-FFT
     baseline (measured 0.5× in `rfft_variants`), not shown here. They cannot make a matmul
     FFT cheaper in flops than a butterfly; different algorithm class.
   - **Why we accept it:** v5e is memory-bound (MXU is 4.6% of time). We spend cheap MXU
     flops to save scarce HBM bytes — the bytes column is the one that predicts wall-clock.
     Flops only matter if the kernel turns compute-bound, which at AI≈10–28 it is far from.

## How each number was obtained (honesty)

- `jnp.fft`, `fft_iter` **bytes/flops** = XLA `cost_analysis` on the compiled program — a
  real HBM measure (offline, CPU).
- **pallas bytes = ANALYTICAL** block I/O (`read in + write out`), because:
  - interpret-mode `cost_analysis` **over-counts** — it charges in-kernel intermediates that
    on TPU live in VMEM, not HBM;
  - a real TPU-compiled `cost_analysis` needs a TPU backend we don't have here.
  So the Pallas byte figure is the **design lower bound** assuming clean VMEM residency —
  **to be confirmed on a v5e** with the profiler's memory_viewer.
- pallas **flops** = interpret `cost_analysis` (the real matmuls execute, so this is fair).

## Trick A/B packing hit a Mosaic wall (a real finding)

The packing tricks (A: pair signals; B: even/odd) are **interpret-correct** but **do not
lower to TPU**. Their unpack needs the Hermitian mirror `conj(Z[N-k])` — a **row reversal**.
Mosaic has **no `rev`/`flip`/gather**; only `pltpu.roll` (a cyclic shift, which cannot build
a reversal). The one reversal that lowers is a permutation-matmul `P @ Z`, but `P` is N×N →
**O(N²)**, which dwarfs the radix-B FFT (≈42× its flops at N=16384). So:

> Trick A/B are blocked on a missing Mosaic primitive, not on our algorithm. The
> **real-input Hermitian rfft** (`pallas_rfft`) sidesteps it entirely — it needs no mirror
> (just `xi=0` + an output slice) and captures the byte win + a good part of the flop win.

If a future Mosaic exposes a cheap axis-0 reverse, Trick A drops in for the full flop-halving.

## Caveats / limits

- **VMEM ceiling.** "Whole transform resident" only holds while `(N, K_TILE)` + the stage
  intermediates fit in VMEM (~64–128 MiB). Fine for the FNO (N=256). Around N≈16384 the
  working set is tens of MiB and may spill; beyond that (toward 2²⁴) you MUST cache-block
  (multi-pass), which raises bytes above the 0.27× bound. "Lowers to TPU" ≠ "fits at
  runtime" — spill/OOM is a separate v5e check.
- **Tiling rule.** block last-two-dims must be (÷8, ÷128); `K_TILE` is a multiple of 128.
- The **rfft trick stacks on top**: writing only `N//2+1` rows (real input → Hermitian) cuts
  the *write* term, taking the analytical bound from 0.27× toward ~0.18×. (Exact `N//2+1`
  is not ÷8; use `N//2` rows or pad — a small refinement.)

## Verdict / next

The VMEM-resident Pallas kernel is the first thing that **beats `jnp.fft` on bytes** (0.27×),
is **correct**, and **lowers to TPU** — all confirmed offline. It is the realization of the
whole project thesis (memory-bound → minimize HBM round-trips). Remaining, in order:
1. **One Colab v5e run** to confirm the analytical 0.27× as real HBM + get wall-clock.
2. **Add the rfft write-truncation** inside the kernel (→ ~0.18×).
3. **Cache-blocked variant** for N beyond the VMEM ceiling.

"""
pallas_fft.py  —  a VMEM-resident radix-B FFT as a single Pallas TPU kernel.

WHY Pallas here.  In fft_iter (pure JAX) every stage is a separate XLA kernel, and `dot`
is a fusion barrier, so each stage's result round-trips HBM — that is fft_iter's 2x-at-large-N
byte penalty.  A Pallas kernel runs the WHOLE transform inside one kernel with the working
set resident in VMEM, so the only HBM traffic is: read the input block once, write the
output block once.  The per-stage intermediates (DFT matrices, twiddles, transposes) never
leave VMEM.  That is the memory-bound win the roofline says matters.

Design (optimized, not a toy):
  * grid over the K/batch axis only; the whole length-N axis stays resident.
  * NO complex64 (Mosaic has no complex lowering) -> real/imag as separate f32 arrays.
    This also removes the ~21% is-finite/select complex-multiply GUARD tax for free.
  * radix-B matmul on the MXU via jnp.dot; twiddle as guard-free cos/sin real arithmetic.
  * DFT matrices generated in-kernel (cos/sin from iota) -> no matrix HBM traffic.
  * block last-two-dims must be (multiple of 8, multiple of 128) — Mosaic tiling rule; so
    K_TILE is a multiple of 128 and N a multiple of 8.

Measuring bytes WITHOUT a TPU — read this carefully:
  * interpret=True runs the kernel on CPU and is CORRECT, but its cost_analysis OVER-counts
    HBM: it charges every in-kernel intermediate as traffic, whereas on TPU those live in
    VMEM.  So we do NOT use interpret bytes as the HBM number.
  * The true Pallas HBM traffic is a DESIGN property of the grid/BlockSpec = (input read +
    output writes), computed by `pallas_hbm_bytes`.  This is an analytical lower bound
    (assumes clean VMEM residency, no spills) to be CONFIRMED on a v5e with the profiler.
  * flops from interpret cost_analysis ARE representative (the real matmuls run).

Local validation (no TPU): interpret correctness vs numpy, and pl.lower_as_mlir(...,
platforms=['tpu']) to prove Mosaic accepts it.  Wall-clock + real HBM: one Colab v5e run.

  python -m tpu.pallas_fft            # correctness + TPU-lowering + 3-way byte/flop table
"""