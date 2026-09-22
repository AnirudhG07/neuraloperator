# Many batches on ONE TPU core — the math (2026-08)

The FNO spectral conv hands us a DFT with a **huge batch**: `C[N'×N'] @ X[N', K]`, where
`N'` = grid resolution (64–128) and `K = batch·width·(other spatial dims) ≈ 10⁴–10⁵`
(NS-128: K = 65 536; MHD-64: K = 131 072).  The worry was "many batches will fail on one
core."  It doesn't — **on a TPU, large K is the *best* case**, provided we feed it as the
*streaming dimension of one matmul*, not as many small kernels.  This note derives why, and
the tiling/roofline math for doing it nicely on a single v5e core.

## 0. The mental-model flip (GPU vs TPU)

| | GPU | TPU (one core) |
|---|---|---|
| how batch is expressed | many thread-blocks / small FFTs in parallel across SMs | the **free (column) dimension of one big matmul** streamed through the MXU |
| what "more batch" costs | more blocks (more scheduling) | **more cycles on the same stationary weight** — nearly free |
| bad case | — | *small* batch (weight-load not amortized) |
| good case | large batch | **large batch** |

So the fix isn't "find a trick for many batches" — it's "**don't chop K into little
kernels; make K the long axis of one weight-stationary matmul**." The DFT matrix `C` is the
*weight* (loaded once, reused for all K); the `K` signal columns are the *activations*
(streamed). This is precisely the DNN-inference pattern the MXU is designed for.

## 1. Systolic amortization — why big K wins

The MXU is a `B×B` (B=128) weight-stationary systolic array. To run `C @ X` it:
1. loads `C` into the array — a fill latency of ~`B` cycles,
2. streams the `K` columns of `X` through at **1 column/cycle** (steady state),
3. drains — ~`B` cycles.

MXU utilization for one weight-load over a `K`-column stream:
$$\eta(K)=\frac{K}{K+2B}\qquad(B=128)$$

| K per matmul | η |
|---|---|
| 128 | 33 % |
| 512 | 67 % |
| 1024 | 80 % |
| 8192 | 97 % |
| 65 536 | **99.6 %** |

**Takeaway:** feed the MXU *at least ~1024 columns per weight-load* and it is essentially
saturated; at the FNO's K ≈ 10⁵ it is ~100 %. Small batches are what starve a TPU — and we
have the opposite problem, which is the good one. (Confirmed by Google's own guidance: the
large MXUs "rely on large batch sizes, which amortize memory accesses for weights.")

## 2. Roofline — batching does NOT change memory-bound-ness

Bare batched DFT `C[N'×N'] @ X[N', K]`, real/imag, f32:
- flops ≈ `2·N'²·K` (per real matmul; ×2–3 for complex/Karatsuba)
- bytes ≈ read `X` + write `Y` = `2·N'·K·dt`  (the weight `C` is `N'²·dt`, amortized → 0 as K↑)

$$\text{AI}=\frac{2N'^2K}{2N'K\cdot dt}=\frac{N'}{dt}\ \Big(\text{f32}\Rightarrow \tfrac{N'}{4}\Big)$$

For N'=128, f32: **AI ≈ 32 flop/byte ≪ ridge 240** → memory-bound. Crucially **AI is
independent of K** (flops and bytes both scale with K). So:

> Adding batches keeps the MXU busy (§1) but **cannot** move you toward compute-bound. The
> job on one core is therefore to **saturate HBM bandwidth (819 GB/s)**, not the MXU. The
> MXU will *idle waiting on HBM* — that is expected and fine; HBM is the meter.

The only things that raise AI are algorithmic (see §5), not more batching.

## 3. VMEM tiling — the actual "how" for one core

`X` at full K is too big for VMEM (NS-128: 128·65536·4 B = 33 MB ≫ VMEM). So **tile K by the
Pallas grid** and stream each tile from HBM — which the kernel already does
(`grid=(K//K_TILE,)`, `BlockSpec((N', K_TILE), …)`). The per-core working set that must fit
VMEM `V`:
$$\underbrace{3N'^2\,dt}_{C_r,C_i,C_s}+\underbrace{c\,(2N'\!\cdot\!K_{\text{tile}}\,dt)}_{\text{in+out re/im}}\cdot\underbrace{2}_{\text{double buffer}}\ \le\ V$$

Solve for the tile:
$$\boxed{\,K_{\text{tile}}\ \le\ \frac{V-3N'^2 dt}{4c\,N'\,dt}\,}\quad(c\!=\!\text{live re/im buffers})$$

Then pick `K_tile` as the **largest multiple of 128** under that bound (128-lane packing),
and **never below ~512** (else §1 util drops). The grid length is `⌈K/K_tile⌉`. Because the
DFT matrix `C` is shared across *every* tile, keep it VMEM-resident and reuse — the weight is
loaded once conceptually, activations stream forever.

**Pipeline it:** `dimension_semantics=("parallel",)` lets Mosaic double-buffer — DMA tile
`k+1` from HBM while the MXU/VPU works tile `k`. Since we're HBM-bound, the goal is to keep
the DMA engines saturated; `K_tile` only needs to be big enough to (a) hide the ~fixed
kernel/weight-load latency and (b) keep the DMA pipeline full. Bigger past that buys nothing
(you're already at peak BW) and costs VMEM.

## 4. Where batching *comes from* inside a large-N′ FFT (Bailey four-step)

Relevant when N' itself is large (NS-1024): the **four-step / Bailey FFT** factors
`N' = N₁·N₂`, reshapes the vector into an `N₁×N₂` matrix, and does
1. `N₂` batched length-`N₁` FFTs (columns), 2. twiddle multiply, 3. transpose,
4. `N₁` batched length-`N₂` FFTs (rows).

I.e. **a big FFT is itself a pile of batched small FFTs** — the exact `C@X` matmul shape of
§1, now with batch `= N₂` (or `N₁`) *times* the real K. This is why the radix-B kernel's
stage loop is the right structure: each stage is a batched-matmul DFT, and the batch only
grows (`batch ×= B` per stage), keeping the MXU ever better amortized as it goes deeper. The
"batch" and "transform-length" axes are interchangeable bookkeeping on the same matmul.

## 5. The only real levers to beat memory-bound (not more batching)

Since §2 says batching can't raise AI, the wins are algorithmic — and they help *most* at
large K because they cut the K-scaled bytes:

1. **Partial DFT** — compute only the `n_modes` kept rows: `C_lo[n_modes×N'] @ X[N', K]` →
   output `(n_modes, K)`. Write bytes drop `N' / n_modes` (NS-128, modes 32: **4× less
   write**). Memory-bound + write-scaled-by-K ⇒ this is the biggest single lever. See
   `04_rfft_research.md`.
2. **Fuse the whole spectral conv** (DFT → weight-mult → iDFT) in one VMEM-resident kernel:
   read `x` once, write `y` once, the `(n_modes, K)` spectrum never touches HBM. AI rises
   because you do 3× the flops per HBM byte. TurboFNO/DRIFT.
3. **bf16 / int8 operands** (`dt=BF`/`I8`): halve/quarter the K-scaled bytes directly.

## 6. Recipe — many batches on one v5e core, cleanly

1. **One matmul, K as the streaming free-dim.** `C` (stationary weight) `@ X[N', K_tile]`.
   Never emit per-batch kernels.
2. **Tile K by the grid**, `K_tile` = largest mult-of-128 fitting VMEM (§3), ≥ 512 for MXU
   amortization (§1). `grid = ⌈K/K_tile⌉`.
3. **`("parallel",)` semantics** → double-buffer HBM↔VMEM; overlap DMA(k+1) with compute(k).
4. **Aim at HBM BW, not MXU.** You are memory-bound (§2); success = bytes-moved / 819 GB/s.
5. **Cut the K-scaled bytes** with partial DFT + fusion + low precision (§5) — the actual
   speedups.
6. **Multi-core, if ever:** shard K across cores (embarrassingly parallel, batches are
   independent). One core just serializes the same tiles — identical total bytes, 1/n the
   bandwidth. No math is lost going single-core; you're simply bandwidth-limited.

**Bottom line:** "many batches" is not a failure mode on TPU — it is the regime the MXU is
built for. K is free in compute (§1) and only costs HBM traffic (§2), which the grid-tiled,
double-buffered VMEM-resident kernel already streams at peak bandwidth. The engineering knob
is `K_tile` (VMEM ∩ ≥512); the *speed* knob is cutting K-scaled bytes (partial DFT, fusion,
bf16/int8) — never concatenating batches into the transform axis (that breaks the DFT).

## Sources
- TPU weight-stationary / batch amortization: https://jax-ml.github.io/scaling-book/tpus/ ,
  https://telesens.co/2018/07/30/systolic-architectures/
- TPU architecture / MXU: https://blog.bytebytego.com/p/how-googles-tensor-processing-unit
- Bailey four-step FFT: https://en.wikipedia.org/wiki/Bailey%27s_FFT_algorithm ,
  https://link.springer.com/chapter/10.1007/3-540-48228-8_58
- Partial-DFT / fusion levers: `tpu/math/04_rfft_research.md` (DRIFT, TurboFNO)
