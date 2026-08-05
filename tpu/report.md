# Analysis of `fft_variants.py`

## Goal

The goal of this implementation is to investigate whether different formulations of the radix-`B` Cooley–Tukey FFT can reduce memory traffic on TPUs, where the workload is strongly memory-bound.

The investigated questions are:

1. Can explicit transposes be removed?
2. Can the twiddle multiplication be fused into the inner DFT?
3. Will these reduce HBM traffic?

Overall, the implementation is well structured and makes it easy to isolate individual design decisions. However, after analyzing the algorithms (rather than only the generated JAX code), several observations emerged.

---

# 1. The "No Transpose" variant does not actually eliminate the transpose

Current implementation

```
Yb
 ↓
Twiddle
 ↓
einsum over N1
 ↓
transpose
 ↓
reshape
```

Baseline

```
Yb
 ↓
Twiddle
 ↓
transpose
 ↓
reshape
 ↓
matmul
```

The transpose has simply moved.

Instead of

```python
transpose(...)
matmul(...)
```

the implementation performs

```python
einsum(...)
transpose(...)
```

The output ordering of Cooley–Tukey requires the frequency index ordering

```
d * B + b
```

while the einsum naturally produces

```
b * N1 + d
```

Therefore a transpose is still necessary before returning.

### Consequence

No memory movement has actually been removed.

The implementation is therefore testing

> "transpose early vs transpose late"

rather than

> "transpose vs no transpose"

---

# 2. Twiddle fusion is algebraically correct but computationally expensive

Current implementation constructs

```
Ctwid
shape

(B, N1, N1)
```

instead of

```
twiddle
shape

(B, N1)
```

Suppose

```
B = 128
N1 = 128
```

Then

Old operand

```
128 × 128
```

New operand

```
128 × 128 × 128
```

which is

```
128× larger
```

before any multiplication begins.

Additionally,

```
DFT matrix
```

is shared across all `b`

whereas

```
Ctwid
```

is different for every `b`.

This prevents reuse.

### Recommendation

Do **not** fold the twiddle into the DFT matrix.

Instead fuse

```
DFT
↓

twiddle multiply

↓

next FFT stage
```

inside one kernel.

This is the kind of fusion that reduces HBM traffic.

---

# 3. The optimization target is not actually the transpose

The implementation focuses on

```
transpose
```

but the dominant cost is

```
materialization
```

Current pipeline

```
FFT

↓

store

↓

reload

↓

twiddle

↓

store

↓

reload

↓

transpose

↓

store

↓

reload
```

The expensive operations are

- HBM writes
- HBM reads
- intermediate tensors

rather than the transpose instruction itself.

### Recommendation

Optimize for

> "How many FFT stages can stay in VMEM?"

instead of

> "Can one transpose disappear?"

---

# 4. Recursive decomposition creates many intermediate tensors

Each recursive level performs

```
reshape

↓

FFT

↓

twiddle

↓

transpose

↓

reshape

↓

recursive call
```

Every level creates multiple arrays.

Even if XLA fuses some operations, recursion naturally increases the number of live tensors.

### Recommendation

Prefer stage-oriented kernels

```
stage 0

↓

stage 1

↓

stage 2
```

rather than recursively building a very large HLO graph.

This is especially important for Pallas.

---

# 5. Recursive flatten → transpose → reshape is not TPU-friendly

Current recursion repeatedly performs

```
(B,N1,K)

↓

transpose

↓

(N1,B,K)

↓

reshape

↓

recursive FFT
```

This repeatedly changes memory layout.

On TPUs, layout changes are expensive if they leave VMEM.

### Recommendation

Keep tensors multidimensional for as long as possible.

Instead of repeatedly flattening,

operate directly on

```
(B,N1,K)
```

and interpret different axes as FFT dimensions.

---

# 6. Large-leaf GEMM is not necessarily beneficial

One possible optimization is

```
stop recursion early

↓

large dense DFT
```

However benchmarking indicates that

```
256×256
```

or larger dense DFTs become slower.

This likely indicates that

- the larger matrices exceed efficient VMEM usage,
- the additional operand size outweighs reduced recursion,
- XLA can no longer keep everything on-chip.

### Recommendation

Continue splitting recursively until a relatively small radix.

Current experimental evidence suggests that deeper recursion is preferable to larger dense leaves.

---

# 7. DFT matrices are recomputed

Each recursive level constructs

```python
dft_matrix(...)
```

Although XLA will likely constant-fold many of these,

they still increase HLO size.

### Recommendation

Cache

```
DFT16
DFT32
DFT64
DFT128
```

and reuse them.

---

# 8. Current implementation relies heavily on transpose between recursion levels

The recursion assumes

```
FFT always acts on axis 0
```

Therefore

every recursive call requires

```
transpose

↓

reshape

↓

FFT
```

### Recommendation

Redesign recursion around

```
FFT(axis=i)
```

rather than

```
FFT(always axis 0)
```

This would allow recursion to operate on arbitrary tensor dimensions.

Whether XLA ultimately removes the layout conversion is another question, but this structure is significantly closer to what a Pallas implementation would naturally express.

---

# 9. Pure JAX may hide transposes inside `dot_general`

Even if explicit

```python
transpose(...)
```

disappears,

operations like

```python
einsum(...)
```

may lower internally to

```
transpose

↓

dot_general
```

to satisfy preferred layouts.

Therefore

absence of

```python
jnp.transpose
```

does **not** imply

absence of memory movement.

### Recommendation

Do not rely solely on HLO textual inspection.

Validate with

- TPU profiler
- memory viewer
- HLO after optimizations
- kernel timeline

---

# 10. The implementation is testing algorithmic changes, not kernel fusion

The term

```
fusion
```

currently refers to

```
twiddle folded into matrix
```

This is algebraic fusion.

However

kernel fusion means

```
load

↓

FFT

↓

twiddle

↓

FFT

↓

store
```

inside a single compiled kernel.

These are completely different.

### Recommendation

Keep these concepts separate.

The latter is generally the one that reduces memory traffic.

---

# 11. Biggest missing optimization

The implementation still writes intermediate results between FFT stages.

Ideally one would like

```
Load tile into VMEM

↓

FFT stage

↓

Twiddle

↓

FFT stage

↓

Store once
```

rather than

```
Load

↓

FFT

↓

Store

↓

Load

↓

Twiddle

↓

Store

↓

Load

↓

FFT

↓

Store
```

This is likely where the largest TPU speedups will come from.

---

# 12. Main opportunity for Pallas

The current implementation is constrained by generic XLA operations.

Pallas provides explicit control over

- tile shape,
- VMEM usage,
- indexing,
- synchronization,
- load/store placement.

Rather than trying to remove transposes algebraically, the Pallas kernel can instead

- load tiles directly in the desired logical layout,
- perform multiple FFT stages while data remains in VMEM,
- apply twiddles immediately,
- store only the final result.

This directly attacks the HBM bottleneck.

---

# Summary

## Good aspects of the implementation

- Clean separation of variants.
- Correct Cooley–Tukey decomposition.
- Excellent benchmarking infrastructure.
- Useful HLO inspection utilities.
- Easy comparison of algorithmic choices.

---

## Weaknesses

- "No transpose" delays rather than removes the transpose.
- Twiddle folding significantly increases operand size.
- Focuses on transpose rather than materialization.
- Recursive structure creates many intermediates.
- Recursion assumes FFT always acts on axis 0.
- Large dense leaves appear to exceed efficient working-set sizes.
- Pure JAX cannot fully control layout or VMEM residency.

---

# Overall recommendation

For the pure JAX implementation, keep the code primarily as a correctness reference and benchmarking harness.

The primary optimization effort should move to the Pallas implementation, where the focus should be on **memory residency rather than algebraic reformulation**. Specifically:

1. Keep FFT tiles resident in VMEM across multiple stages.
2. Fuse FFT → twiddle → FFT within a single kernel.
3. Minimize HBM reads/writes rather than eliminating explicit `transpose` operations.
4. Use small recursive radices (e.g., 64 or 128) if benchmarking confirms they outperform larger dense leaves.
5. Treat transposes as an indexing problem inside the kernel whenever possible, rather than as standalone tensor operations.

Given that the workload is demonstrably memory-bound, reducing HBM traffic is expected to provide significantly greater benefits than reducing floating-point operations alone.