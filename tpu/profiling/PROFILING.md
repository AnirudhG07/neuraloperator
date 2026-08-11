# TPU FFT profiling — tpu/profiling/tpu_prof/ (xprof /device:TPU:0) + timings.csv

## methodology
- capture: `bench_jax.profile()` = `jax.profiler.trace()` around 15 `block_until_ready(fn(x))` per (engine, N).
- device µs = Σ `XLA Ops` event `device_duration_ps` on /device:TPU:0 ÷ runs.  wall µs = host `perf_counter` (= device + dispatch).
- HBM = `memory_access_breakdown` mem-space-1 bytes ÷ runs.  AI = `flops` ÷ `bytes_accessed`.  category = XLA `hlo_category`.

## verify in TensorBoard (open this trace; each number's source)
| number here | TensorBoard tool → field |
|---|---|
| device µs/run | Trace Viewer → 'XLA Modules' block duration; or Op Profile → total self-time ÷ runs |
| wall µs/run | not in xprof — host `perf_counter` (= device + ~180µs dispatch) |
| HBM MB/run | Memory Viewer → peak/bytes; or Op Profile → 'Bytes accessed' (HBM), ÷ runs |
| AI (flop/byte) | Op Profile → 'FLOPs' ÷ 'Bytes accessed' |
| MXU/VPU/memops/custom % | Op Profile → group by Category (rolled up from hlo_category) |
| the RAW category rows below | Op Profile → the 'Category' column, ungrouped |
| a single op (fusion.N) | Op Profile → expand a category; or Trace Viewer → click the op |

## summary  (sorted by N;  ★ = fastest device µs at that N;  memops = copy+reshape+transpose+slice+gather+reverse+DMA)

| N | engine | device µs | wall µs | gap µs | HBM MB | AI | MXU% | VPU% | memops% | custom% |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | ★ pallas_half | 5.4 | 292.8 | 287.4 | — | 0 | 0 | 0 | 0 | 100 |
| 256 | trickA | 6.2 | 178.3 | 172.1 | 0.26 | 4 | 48 | 20 | 19 | 13 |
| 256 | trickB | 7.6 | 167.1 | 159.6 | 0.66 | 3 | 24 | 35 | 29 | 11 |
| 256 | jnp.rfft | 8.8 | 180.3 | 171.6 | 0.26 | 154 | 67 | 23 | 0 | 9 |
| 512 | ★ pallas_half | 6.0 | 240.7 | 234.7 | — | 0 | 0 | 0 | 0 | 100 |
| 512 | trickA | 8.4 | 175.3 | 166.9 | 0.52 | 4 | 42 | 21 | 23 | 14 |
| 512 | jnp.rfft | 16.6 | 186.0 | 169.4 | 0.52 | 43 | 75 | 1 | 16 | 7 |
| 512 | trickB | 17.6 | 182.0 | 164.5 | 0.79 | 3 | 20 | 42 | 32 | 7 |
| 1024 | ★ pallas_half | 7.8 | 257.2 | 249.4 | — | 0 | 0 | 0 | 0 | 100 |
| 1024 | trickA | 13.3 | 175.4 | 162.1 | 1.05 | 4 | 40 | 22 | 24 | 14 |
| 1024 | jnp.rfft | 23.0 | 191.4 | 168.4 | 1.05 | 47 | 73 | 1 | 17 | 8 |
| 1024 | trickB | 25.7 | 195.3 | 169.6 | 1.57 | 3 | 16 | 42 | 33 | 7 |
| 4096 | ★ trickA | 50.7 | 212.0 | 161.3 | 4.19 | 5 | 44 | 19 | 24 | 13 |
| 4096 | trickB | 82.2 | 240.3 | 158.0 | 6.29 | 3 | 16 | 39 | 37 | 8 |
| 4096 | jnp.rfft | 88.3 | 337.8 | 249.5 | 4.19 | 55 | 75 | 1 | 16 | 7 |
| 8192 | ★ trickA | 101.4 | 309.0 | 207.6 | 8.39 | 5 | 42 | 21 | 24 | 13 |
| 8192 | trickB | 157.8 | 355.3 | 197.5 | 12.58 | 3 | 16 | 38 | 38 | 8 |
| 8192 | jnp.rfft | 175.9 | 427.5 | 251.6 | 8.39 | 67 | 74 | 1 | 17 | 7 |
| 16384 | ★ trickA | 201.2 | 422.0 | 220.8 | 16.78 | 7 | 44 | 19 | 24 | 13 |
| 16384 | jnp.rfft | 311.2 | 547.5 | 236.2 | 16.78 | 89 | 72 | 1 | 19 | 8 |
| 16384 | trickB | 329.9 | 560.9 | 231.1 | 25.17 | 4 | 15 | 35 | 43 | 8 |

## per (engine, N) — RAW `hlo_category` (xprof Op-Profile categories, no rollup)

**jnp.rfft  N=256** — device 8.8 us/run, 15 runs
```
convolution fusion      66.9%  ███████████████████·········
loop fusion             23.5%  ███████·····················
custom-call              9.4%  ███·························
copy-start               0.2%  ····························
copy-done                0.0%  ····························
```
**pallas_half  N=256** — device 5.4 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**trickA  N=256** — device 6.2 us/run, 15 runs
```
convolution fusion      47.6%  █████████████···············
loop fusion             19.9%  ██████······················
custom-call             13.5%  ████························
reverse                  9.3%  ███·························
data formatting          8.8%  ██··························
slice                    0.9%  ····························
```
**trickB  N=256** — device 7.6 us/run, 15 runs
```
custom fusion           28.3%  ████████····················
convolution fusion      24.3%  ███████·····················
slice                   13.7%  ████························
copy-done               11.2%  ███·························
custom-call             11.0%  ███·························
loop fusion              6.8%  ██··························
reverse                  4.4%  █···························
copy-start               0.1%  ····························
async-start              0.1%  ····························
async-done               0.0%  ····························
```
**jnp.rfft  N=512** — device 16.6 us/run, 15 runs
```
convolution fusion      75.0%  █████████████████████·······
data formatting         16.4%  █████·······················
custom-call              7.1%  ██··························
loop fusion              1.5%  ····························
```
**pallas_half  N=512** — device 6.0 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**trickA  N=512** — device 8.4 us/run, 15 runs
```
convolution fusion      42.1%  ████████████················
loop fusion             21.0%  ██████······················
custom-call             14.0%  ████························
reverse                 11.1%  ███·························
data formatting         10.5%  ███·························
slice                    1.2%  ····························
```
**trickB  N=512** — device 17.6 us/run, 15 runs
```
custom fusion           36.0%  ██████████··················
convolution fusion      19.5%  █████·······················
data formatting         12.9%  ████························
slice                   12.7%  ████························
custom-call              6.7%  ██··························
reverse                  6.2%  ██··························
loop fusion              5.7%  ██··························
copy-done                0.2%  ····························
copy-start               0.0%  ····························
```
**jnp.rfft  N=1024** — device 23.0 us/run, 15 runs
```
convolution fusion      73.3%  █████████████████████·······
data formatting         17.0%  █████·······················
custom-call              8.2%  ██··························
loop fusion              1.5%  ····························
```
**pallas_half  N=1024** — device 7.8 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**trickA  N=1024** — device 13.3 us/run, 15 runs
```
convolution fusion      40.2%  ███████████·················
loop fusion             21.7%  ██████······················
custom-call             14.1%  ████························
reverse                 12.3%  ███·························
data formatting         10.3%  ███·························
slice                    1.4%  ····························
```
**trickB  N=1024** — device 25.7 us/run, 15 runs
```
custom fusion           35.5%  ██████████··················
convolution fusion      16.2%  █████·······················
slice                   13.6%  ████························
data formatting         12.8%  ████························
custom-call              7.3%  ██··························
reverse                  7.0%  ██··························
loop fusion              6.1%  ██··························
async-done               1.3%  ····························
async-start              0.1%  ····························
```
**jnp.rfft  N=4096** — device 88.3 us/run, 15 runs
```
convolution fusion      75.4%  █████████████████████·······
data formatting         16.2%  █████·······················
custom-call              7.3%  ██··························
loop fusion              1.1%  ····························
```
**trickA  N=4096** — device 50.7 us/run, 15 runs
```
convolution fusion      44.0%  ████████████················
loop fusion             19.2%  █████·······················
custom-call             12.8%  ████························
reverse                 11.6%  ███·························
data formatting         11.0%  ███·························
slice                    1.4%  ····························
```
**trickB  N=4096** — device 82.2 us/run, 15 runs
```
custom fusion           31.5%  █████████···················
convolution fusion      16.4%  █████·······················
slice                   15.7%  ████························
data formatting         10.5%  ███·························
custom-call              7.9%  ██··························
reverse                  7.5%  ██··························
loop fusion              7.1%  ██··························
copy-done                3.5%  █···························
copy-start               0.0%  ····························
```
**jnp.rfft  N=8192** — device 175.9 us/run, 15 runs
```
convolution fusion      74.3%  █████████████████████·······
data formatting         17.2%  █████·······················
custom-call              7.3%  ██··························
loop fusion              1.1%  ····························
```
**trickA  N=8192** — device 101.4 us/run, 15 runs
```
convolution fusion      42.4%  ████████████················
loop fusion             20.9%  ██████······················
custom-call             12.7%  ████························
reverse                 11.4%  ███·························
data formatting         10.9%  ███·························
slice                    1.7%  ····························
```
**trickB  N=8192** — device 157.8 us/run, 15 runs
```
custom fusion           30.5%  █████████···················
slice                   16.1%  █████·······················
convolution fusion      15.9%  ████························
data formatting         10.9%  ███·························
custom-call              8.2%  ██··························
reverse                  7.5%  ██··························
loop fusion              7.2%  ██··························
copy-done                3.6%  █···························
copy-start               0.0%  ····························
```
**jnp.rfft  N=16384** — device 311.2 us/run, 15 runs
```
convolution fusion      71.8%  ████████████████████········
data formatting         18.7%  █████·······················
custom-call              8.3%  ██··························
loop fusion              1.2%  ····························
```
**trickA  N=16384** — device 201.2 us/run, 15 runs
```
convolution fusion      43.7%  ████████████················
loop fusion             19.3%  █████·······················
custom-call             12.8%  ████························
reverse                 11.4%  ███·························
data formatting         11.0%  ███·························
slice                    1.8%  █···························
```
**trickB  N=16384** — device 329.9 us/run, 15 runs
```
custom fusion           28.2%  ████████····················
slice                   16.7%  █████·······················
data formatting         16.4%  █████·······················
convolution fusion      14.6%  ████························
custom-call              7.8%  ██··························
reverse                  7.1%  ██··························
loop fusion              6.7%  ██··························
copy-done                2.4%  █···························
copy-start               0.0%  ····························
```

## sources
- xprof / JAX profiling: https://openxla.org/xprof/jax_profiling
- xplane proto (device_duration_ps / hlo_category / bytes_accessed / flops / memory_access_breakdown): `tensorflow.tsl.profiler.protobuf.xplane_pb2`
