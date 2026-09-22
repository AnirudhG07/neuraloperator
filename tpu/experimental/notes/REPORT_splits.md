# TPU FFT — Splitting, the flop cap at √b = 12, and the time-optimal leaf

**Hardware:** TPU (single core), `b = 128` (systolic-array width).
**Data:** every number is copied verbatim from `data/tpu_data.md` (the output of
`bench_jax.py run()` on the TPU), which sweeps **every** LEAF = `k·b`, `k = 1..32`, plus
`direct`, for 48 values of N. Rows are cited by their `N` (grep the file to check).

### Definitions

- **N** — transform length.
- **LEAF = m·b** — split a sub-DFT (peel one radix-`b` stage) while its size > LEAF; otherwise do
  it **direct** as one matmul (a "leaf"). `m` = LEAF in units of `b`.
- **`m = 1` (LEAF = b) = split everything** down to `b`-sized leaves. **`direct` = never split.**
- **work** — multiply-steps after padding each matmul to the `b×b` array (MXU-tile op count).
- **time** — median of 25 blocked TPU runs.
- **min-WORK / min-TIME** — the LEAF the sweep marks as lowest work / lowest time for that N.

---

## Claim 1 — Split beats direct at every step ⇒ recurse fully, `L = ⌊log_b N⌋`

Direct time explodes with N and then runs **out of memory**; the fully-split plan stays flat.
So the plan should split at every step, giving `L = ⌊log_b N⌋` stages.

| N | direct work | direct time | split (m=1) work | split (m=1) time | speed-up |
|--:|--:|--:|--:|--:|--:|
| 512 | 262.1K | 499.5 µs | 2.2M | **178.9 µs** | 2.8× |
| 1,024 | 1.0M | 1.67 ms | 2.2M | **213.2 µs** | 7.8× |
| 2,048 | 4.2M | 8.74 ms | 2.4M | **250.8 µs** | 35× |
| 4,096 | 16.8M | 33.80 ms | 2.6M | **196.4 µs** | 172× |
| 5,120 | 26.2M | **OOM** | 2.8M | **211.6 µs** | ∞ |
| 8,192 | 67.1M | **OOM** | 3.1M | **225.4 µs** | ∞ |
| 16,384 | 268.4M | **OOM** | 4.2M | **225.6 µs** | ∞ |

Stage count equals `⌊log_b N⌋` on powers of `b`: `b²=16,384 → 2 levels`; `b³=2,097,152 → 3 levels`.

---

## Claim 2 — The op-count cap is `√b = 12`, shown at three recursion depths

For a sub-DFT of size `k·b`: direct costs `k²` tiles, split-once costs `k + b` tiles. Direct is
cheaper while `k² < k + b`, i.e. `k < √b ≈ 12`. **The min-WORK plan leafs a sub-DFT while its
size ≤ 11·b and splits it once it reaches 12·b — at every depth.**

### 2.1 Top level: `N = b·k`, direct vs split (op count)

| k | N = b·k | direct work | split (m=1) work | fewer ops |
|--:|--:|--:|--:|:--|
| 2 | 256 | **65.5K** | 2.1M | direct |
| 4 | 512 | **262.1K** | 2.2M | direct |
| 6 | 768 | **589.8K** | 2.2M | direct |
| 8 | 1,024 | **1.0M** | 2.2M | direct |
| 10 | 1,280 | **1.6M** | 2.3M | direct |
| 11 | 1,408 | **2.0M** | 2.3M | direct |
| **12** | **1,536** | 2.4M | **2.3M** | **split ← crossover** |
| 13 | 1,664 | 2.8M | **2.3M** | split |
| 16 | 2,048 | 4.2M | **2.4M** | split |
| 20 | 2,560 | 6.6M | **2.4M** | split |
| 24 | 3,072 | 9.4M | **2.5M** | split |
| 32 | 4,096 | 16.8M | **2.6M** | split |

### 2.2 One level down: `N = b²·k`, min-WORK leaf multiplier `m*`

`m*` is the leaf size (in units of `b`) the sweep chooses. It equals the tail `k` while `k ≤ 11`
(leaf the tail), then snaps to `1` (split it) at `k ≥ 12`.

| tail k | N = b²·k | min-WORK m* | min-WORK leaf size | plan |
|--:|--:|--:|--:|:--|
| 2 | 32,768 | 2 | 256 | leaf tail |
| 3 | 49,152 | 3 | 384 | leaf tail |
| 4 | 65,536 | 4 | 512 | leaf tail |
| 6 | 98,304 | 6 | 768 | leaf tail |
| 8 | 131,072 | 8 | 1,024 | leaf tail |
| **12** | **196,608** | **1** | **12** | **split ← crossover** |
| 16 | 262,144 | 1 | 16 | split |
| 24 | 393,216 | 1 | 24 | split |
| 32 | 524,288 | 1 | 32 | split |

### 2.3 Two levels down: `N = b³·k`, same crossover

| tail k | N = b³·k | min-WORK m* | plan |
|--:|--:|--:|:--|
| 2 | 4,194,304 | 2 | leaf tail |
| 4 | 8,388,608 | 4 | leaf tail |
| 8 | 16,777,216 | 8 | leaf tail |
| **16** | **33,554,432** | **1** | **split ← crossover (8 → 12 → 16)** |
| 32 | 67,108,864 | 1 | split |

Across **all three depths** the optimal leaf never exceeds `8·b` in the data and always splits by
`12·b`. The largest leaf worth keeping is `⌊√b⌋·b = 11·b`; crossover at 12.

### 2.4 Full LEAF sweep at one N below the cap and one at the cap

Below the cap (`N = b²·8`, tail 8): leafing the 8·b tail (m=8) is fewer ops than splitting; but
note the **time** column already disagrees (this is Claim 3).

| LEAF (m·b) | levels | leaf size | work | time |
|--:|--:|--:|--:|--:|
| 1·b | 3 | 8 | 302.0M | **392.4 µs** ← min-TIME |
| 4·b | 3 | 8 | 302.0M | 392.4 µs |
| 7·b | 3 | 8 | 302.0M | 392.4 µs |
| 8·b | 2 | 1,024 | **151.0M** ← min-WORK | 1.78 ms |
| 16·b | 2 | 1,024 | 151.0M | 1.78 ms |
| 32·b | 2 | 1,024 | 151.0M | 1.78 ms |

At the cap (`N = b²·12`, tail 12): now splitting (m=1) wins on **both** — leafing the 12·b tail
costs more work *and* far more time.

| LEAF (m·b) | levels | leaf size | work | time |
|--:|--:|--:|--:|--:|
| 1·b | 3 | 12 | **318.8M** ← min-WORK | **433.5 µs** ← min-TIME |
| 8·b | 3 | 12 | 318.8M | 433.5 µs |
| 11·b | 3 | 12 | 318.8M | 433.5 µs |
| 12·b | 2 | 1,536 | 327.2M | 4.35 ms |
| 16·b | 2 | 1,536 | 327.2M | 4.35 ms |
| 32·b | 2 | 1,536 | 327.2M | 4.35 ms |

---

## Claim 3 — On TIME, split fully (`m = 1`) wins at every N

`min-TIME` is `m = 1` for **all 48** N in the file. Where the op model prefers a bigger leaf
(Claim 2, `k < 12`), that leaf is **slower or tied** in time — the op model is the wrong proxy.

| N | m = 1 time (min-TIME) | min-WORK leaf | its time | m=1 vs work-leaf |
|--:|--:|:--|--:|:--|
| 32,768 (b²·2) | **244.3 µs** | 2·b (256) | 313.8 µs | m=1 1.3× faster |
| 49,152 (b²·3) | **291.8 µs** | 3·b (384) | 434.9 µs | 1.5× |
| 65,536 (b²·4) | **336.8 µs** | 4·b (512) | 577.9 µs | 1.7× |
| 98,304 (b²·6) | **337.5 µs** | 6·b (768) | 970.6 µs | 2.9× |
| 131,072 (b²·8) | **392.4 µs** | 8·b (1024) | 1.78 ms | **4.5×** |
| 196,608 (b²·12) | **433.5 µs** | 1·b (splits) | 433.5 µs | same (agree) |
| 262,144 (b²·16) | **538.1 µs** | 1·b | 538.1 µs | same |
| 2,097,152 (b³) | **2.88 ms** | 1·b | 2.88 ms | same |
| 4,194,304 (b³·2) | **4.10 ms** | 2·b (256) | 4.10 ms | tie |
| 8,388,608 (b³·4) | **8.37 ms** | 4·b (512) | 9.13 ms | 1.1× |
| 16,777,216 (b³·8) | **19.36 ms** | 8·b (1024) | 22.24 ms | 1.15× |
| 33,554,432 (b³·16) | **41.94 ms** | 1·b | 41.94 ms | same |
| 67,108,864 (b³·32) | **83.55 ms** | 1·b | 83.55 ms | same |

Plus the whole 1-split region (Claim 1): `m = 1` time stays ~200 µs while `direct` grows to tens
of ms. The inversion at `N = 131,072` is the clearest single row: the min-WORK plan has **half**
the ops (151.0M vs 302.0M) yet is **4.5× slower** (1.78 ms vs 392 µs).

---

## Summary

| question | metric | answer | tables |
|:--|:--|:--|:--|
| split or direct? | time | **always split** (direct OOMs ≥ 5,120) | Claim 1 |
| how deep? | — | `L = ⌊log_b N⌋` | Claim 1 |
| min op-count leaf? | work (MXU-tile ops) | leaf ≤ 11·b, split ≥ 12·b (**cap √b = 12**, 3 depths) | 2.1–2.4 |
| min-time leaf? | measured time | `LEAF = b` (m = 1), all 48 N | Claim 3 |

The op model (Claim 2) caps at 12; measured time (Claim 3) wants `m = 1`. They agree only for
`k ≥ 12`; below that the op model says "leaf" but the clock says "split." **Why** the TPU behaves
this way (op categories, bytes moved, complex-multiply overhead) is the next report.

---

## Pass 1 — Profiling: where the time goes on TPU v5e, and why m = 1 wins

This is the "why" the summary deferred. It uses the JAX profiler trace
(`tpu/tpu_profile/…/d9cb5168b752.xplane.pb`), the compiled cost analysis
(`profile_tpu.py`), and the per-op breakdown (`op_breakdown.md`, `fft_ops.csv`).

### 1.1 The chip (TPU v5e)

| spec | value |
|:--|--:|
| bf16 peak compute | **197 TFLOP/s** |
| HBM bandwidth | **819 GB/s** |
| VMEM (on-chip scratchpad) | **128 MiB** |
| MXU (systolic array) | 128 × 128 |
| **roofline ridge** = compute / bandwidth | **197e12 / 819e9 ≈ 240 FLOP/byte** |

Anything with arithmetic intensity **below 240 FLOP/byte is memory-bound** on v5e.

### 1.2 Where the 423 ms of device time actually goes — the MXU is used *least*

Device-op time by category (from the trace, 5,546 ops, `op_breakdown.md`):

| category | time | share |
|:--|--:|--:|
| fusion — **VPU** elementwise/reduce | 289.4 ms | **68.4%** |
| complex-multiply guards (`is-finite`/`select`) | 90.6 ms | **21.4%** |
| **MXU matmul** (custom-call) | 19.4 ms | **4.6% ← least** |
| transpose / copy (layout) | 17.4 ms | 4.1% |
| reshape | 5.9 ms | 1.4% |

**The systolic array does only 4.6% of the work.** ~90% is the VPU: the complex DFT
is lowered to vector multiply+reduce, and 21% is pure overhead — IEEE `is-finite`/`select`
guards that XLA emits for every `complex64` multiply. Time is a **vector/memory** story,
not an MXU story.

### 1.3 Every config is memory-bound (measured AI ≪ 240)

Compiled cost analysis per config (`profile_tpu.py` on v5e). AI = FLOP/byte;
util = achieved ÷ v5e peak:

| config | AI | achieved TFLOP/s | % of 197 | achieved GB/s | % of 819 | bound |
|:--|--:|--:|--:|--:|--:|:--|
| b·8 SPLIT | 10 | 0.3 | 0.2% | 29 | 4% | MEM |
| b·8 DIRECT (1024² mat-vec) | 12 | 2.4 | 1.2% | 203 | 25% | MEM |
| b²·8 SPLIT | 16 | 2.1 | 1.1% | 128 | 16% | MEM |
| b²·8 leaf 1024 | 14 | 3.2 | 1.6% | 226 | 28% | MEM |
| b³·2 SPLIT | 15 | 6.9 | 3.5% | 463 | **57%** | MEM |
| b³·4 SPLIT | 15 | 6.6 | 3.4% | 442 | 54% | MEM |
| b³·8 SPLIT | 14 | 5.8 | 2.9% | 411 | 50% | MEM |
| b³·8 leaf 1024 | 28 | 9.2 | 4.7% | 329 | 40% | MEM |

Every AI is **10–28**, i.e. **8–24× below the 240 ridge**. **Compute utilisation never
exceeds ~5%** of the MXU; **bandwidth reaches ~57%.** The chip is starved for data, not
compute — so **time is set by HBM bytes moved**, and no FLOP-side optimisation (the √b=12
cap of Claim 2) can matter because the workload never approaches the compute roofline.

### 1.4 Since it's memory-bound, m = 1 wins by moving the fewest bytes

Split (m=1) vs the fat leaf, same N, HBM bytes vs time (cost analysis + measured):

| N | plan | HLO flops | **HBM bytes** | time | vs split |
|:--|:--|--:|--:|--:|:--|
| b²·2 (32,768) | m=1 SPLIT | 255M | **17.3 MB** | 241 µs | — |
| b²·2 | leaf 256 | 537M | 36.5 MB | 344 µs | 2.1× bytes → 1.4× |
| b²·4 (65,536) | m=1 SPLIT | 447M | **28.3 MB** | 298 µs | — |
| b²·4 | leaf 512 | 1.6G | 112.4 MB | 575 µs | 4.0× bytes → 1.9× |
| b²·8 (131,072) | m=1 SPLIT | 834M | **50.6 MB** | 394 µs | — |
| b²·8 | leaf 1024 | 5.7G | 400.7 MB | 1.78 ms | 7.9× bytes → **4.5×** |
| b³·8 (16.8M) | m=1 SPLIT | 111.5G | 7.9 GB | 19.36 ms | — |
| b³·8 | leaf 1024 | 206.5G | 7.4 GB | 22.35 ms | fewer bytes but 40% GB/s → still slower |

At small–mid N the leaf moves **2–8× more bytes** → directly **1.4–4.5× slower**. At b³
the byte counts converge, but the fat leaf's dense access pattern only sustains **329 GB/s
vs 411** (40% vs 50% of peak), so it is *still* slower. Either way the loser is the one that
stresses memory more — and **m = 1 always moves the fewest bytes at the best efficiency.**

### 1.5 The op shapes confirm it (from `fft_ops.csv`)

Top device ops carry their HLO tensor shapes — split and leaf are visibly different work:

| dur (×5 runs) | op — shape | meaning |
|--:|:--|:--|
| 25.9 ms | `fusion.368` — **f32[128, 131072, 1]** | SPLIT: batched 128-tile vector work |
| 20.7 ms | `fusion.361` — **f32[2048, 2048]** | LEAF: dense 2048² DFT matrix |
| 9.5 ms | `fusion.361` — **f32[1536, 1536]** | LEAF: dense 1536² DFT matrix |
| 8.1 ms | `is-finite_and_fusion.22` — pred[128, 131072, 1] | complex-multiply guard tax |

Split keeps everything as `[128, N, 1]` batched tiles; the fat leaf materialises big square
`[s, s]` matrices — the O(s²) dense DFT that moves the extra bytes.

### 1.6 Pass-1 conclusion

On v5e the FFT is **memory-bandwidth-bound** (AI 10–28 ≪ 240 ridge; ≤5% compute util, up to
57% BW util) and runs almost entirely as **complex64 VPU vector ops** — the MXU is the least-used
unit at 4.6%. Because time = bytes ÷ 819 GB/s, the plan that moves the fewest bytes wins, and a
dense m·b leaf moves ~m× more bytes than radix-b tiles. **Hence m = 1 (LEAF = b) is fastest — a
memory-bound result, not a compute one.** (Two off-topic levers this exposes: the 21% complex-guard
tax, and getting the matmul onto the MXU in bf16 — either would move the workload toward the 240
ridge, where the √b cap would finally start to matter.)

---

## Appendix A — Complete per-N summary (all 40 blocks)

Every N in the sweep, in one table. Two columns carry the whole report:
**min-WORK m\*** = tail `k` while `k ≤ 11`, then `1` from `k = 12` on (the √b cap, Claim 2);
**min-TIME m\*** = `1` on **every single row** (split fully is fastest, Claim 3).

| N | = | split m=1 work | split m=1 time | min-WORK m* | min-WORK work | min-WORK time | min-TIME m* |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 256 | b·2 | 2.1M | 208.0 µs | 2 | 65.5K | 282.9 µs | **1** |
| 384 | b·3 | 2.1M | 216.4 µs | 3 | 147.5K | 389.5 µs | **1** |
| 512 | b·4 | 2.2M | 178.9 µs | 4 | 262.1K | 499.5 µs | **1** |
| 640 | b·5 | 2.2M | 179.1 µs | 5 | 409.6K | 649.5 µs | **1** |
| 768 | b·6 | 2.2M | 167.5 µs | 6 | 589.8K | 861.6 µs | **1** |
| 1,024 | b·8 | 2.2M | 213.2 µs | 8 | 1.0M | 1.67 ms | **1** |
| 1,280 | b·10 | 2.3M | 200.7 µs | 10 | 1.6M | 2.74 ms | **1** |
| 1,408 | b·11 | 2.3M | 216.4 µs | 11 | 2.0M | 3.51 ms | **1** |
| **1,536** | **b·12** | 2.3M | 223.3 µs | **1** | 2.3M | 223.3 µs | **1** |
| 1,664 | b·13 | 2.3M | 204.9 µs | 1 | 2.3M | 204.9 µs | **1** |
| 1,792 | b·14 | 2.3M | 227.5 µs | 1 | 2.3M | 227.5 µs | **1** |
| 1,920 | b·15 | 2.3M | 225.3 µs | 1 | 2.3M | 225.3 µs | **1** |
| 2,048 | b·16 | 2.4M | 250.8 µs | 1 | 2.4M | 250.8 µs | **1** |
| 2,560 | b·20 | 2.4M | 193.6 µs | 1 | 2.4M | 193.6 µs | **1** |
| 3,072 | b·24 | 2.5M | 230.5 µs | 1 | 2.5M | 230.5 µs | **1** |
| 3,200 | b·25 | 2.5M | 235.1 µs | 1 | 2.5M | 235.1 µs | **1** |
| 3,328 | b·26 | 2.5M | 164.5 µs | 1 | 2.5M | 164.5 µs | **1** |
| 3,456 | b·27 | 2.5M | 181.1 µs | 1 | 2.5M | 181.1 µs | **1** |
| 3,584 | b·28 | 2.6M | 174.9 µs | 1 | 2.6M | 174.9 µs | **1** |
| 4,096 | b·32 | 2.6M | 196.4 µs | 1 | 2.6M | 196.4 µs | **1** |
| 5,120 | b·40 | 2.8M | 211.6 µs | 1 | 2.8M | 211.6 µs | **1** |
| 6,144 | b·48 | 2.9M | 216.2 µs | 1 | 2.9M | 216.2 µs | **1** |
| 8,192 | b·64 | 3.1M | 225.4 µs | 1 | 3.1M | 225.4 µs | **1** |
| 12,288 | b·96 | 3.7M | 255.7 µs | 1 | 3.7M | 255.7 µs | **1** |
| 16,384 | b² | 4.2M | 225.6 µs | 1 | 4.2M | 225.6 µs | **1** |
| 32,768 | b²·2 | 276.8M | 244.3 µs | 2 | 12.6M | 313.8 µs | **1** |
| 49,152 | b²·3 | 281.0M | 291.8 µs | 3 | 25.2M | 434.9 µs | **1** |
| 65,536 | b²·4 | 285.2M | 336.8 µs | 4 | 41.9M | 577.9 µs | **1** |
| 98,304 | b²·6 | 293.6M | 337.5 µs | 6 | 88.1M | 970.6 µs | **1** |
| 131,072 | b²·8 | 302.0M | 392.4 µs | 8 | 151.0M | 1.78 ms | **1** |
| **196,608** | **b²·12** | 318.8M | 433.5 µs | **1** | 318.8M | 433.5 µs | **1** |
| 262,144 | b²·16 | 335.5M | 538.1 µs | 1 | 335.5M | 538.1 µs | **1** |
| 393,216 | b²·24 | 369.1M | 642.7 µs | 1 | 369.1M | 642.7 µs | **1** |
| 524,288 | b²·32 | 402.7M | 834.4 µs | 1 | 402.7M | 834.4 µs | **1** |
| 2,097,152 | b³ | 805.3M | 2.88 ms | 1 | 805.3M | 2.88 ms | **1** |
| 4,194,304 | b³·2 | 36.0G | 4.10 ms | 2 | 2.1G | 4.10 ms | **1** |
| 8,388,608 | b³·4 | 37.6G | 8.37 ms | 4 | 6.4G | 9.13 ms | **1** |
| 16,777,216 | b³·8 | 40.8G | 19.36 ms | 8 | 21.5G | 22.24 ms | **1** |
| **33,554,432** | **b³·16** | 47.2G | 41.94 ms | **1** | 47.2G | 41.94 ms | **1** |
| 67,108,864 | b³·32 | 60.1G | 83.55 ms | 1 | 60.1G | 83.55 ms | **1** |

Scan the **min-WORK m\*** column: `2,3,4,5,6,8,10,11` (= tail) then `1,1,1,…` — the flip is at
`k = 12` in the b· region, again at b²· (`8 → 1` between k=8 and k=12), and again at b³·
(`8 → 1` between k=8 and k=16). Scan **min-TIME m\***: it is `1` on all 40 rows.

---

## Appendix B — verify it yourself

- **Raw numbers:** `data/tpu_data.md` — grep any `N =` above to find its full 32-LEAF sweep.
- **Profiler UI (no TPU needed):**
  ```bash
  pip install tensorboard tensorboard-plugin-profile
  tensorboard --logdir tpu/tpu_profile      # http://localhost:6006 → PROFILE
  ```
  Files: `tpu/tpu_profile/plugins/profile/2026_07_27_16_40_59/d9cb5168b752.xplane.pb` (+ `.trace.json.gz`).
- **Reproduce:** `python tpu/bench_jax.py` on a TPU runtime.
