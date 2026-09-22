# rfft — halving the bytes by exploiting real input

> **How to read this file:** open VS Code's built-in Markdown preview
> (`Ctrl/Cmd+Shift+V`). It renders `$…$` / `$$…$$` with KaTeX natively — no LaTeX
> install, no plugin. This is the math for variant **F3 (rfft)** in the plan.

Goal of F3: the FNO inputs are **real**, but `fft_jax.py` casts real→`complex64` on line 1,
doubling every byte from the start. A real-input DFT has a symmetry that lets us do **half
the work / move half the bytes** — the single biggest offline byte-cut in the catalog
(`traffic_model`: rfft = ×0.5). This file derives (a) *why* half is redundant, and (b) two
ways to actually cash that in, with a recommendation for our radix-B / FNO setting.

---

## 0. Notation

Length-$N$ DFT of $x[0..N{-}1]$, with the twiddle root

$$
W_N \;=\; e^{-2\pi i / N},
\qquad
X[k] \;=\; \sum_{n=0}^{N-1} x[n]\, W_N^{\,nk},
\quad k = 0,\dots,N-1 .
$$

Everything below assumes $x[n]\in\mathbb{R}$.

---

## 1. Hermitian symmetry — half the spectrum is redundant

Claim: for real $x$,

$$
\boxed{\,X[N-k] \;=\; \overline{X[k]}\,}\qquad(\text{overbar} = \text{complex conjugate}).
$$

**Proof.** Use $W_N^{\,nN}=1$, so $W_N^{\,n(N-k)} = W_N^{-nk} = \overline{W_N^{\,nk}}$, and
$x[n]$ real ($x[n]=\overline{x[n]}$):

$$
X[N-k]
= \sum_n x[n]\,W_N^{\,n(N-k)}
= \sum_n x[n]\,\overline{W_N^{\,nk}}
= \overline{\sum_n x[n]\,W_N^{\,nk}}
= \overline{X[k]} . \qquad\blacksquare
$$

**Consequences.**

- Only the first half is independent: knowing $X[0],\dots,X[N/2]$ gives the rest by
  conjugation. That is $\tfrac N2 + 1$ complex numbers, not $N$.
- The two "self-conjugate" bins are purely **real**:
  $$
  X[0]=\sum_n x[n]\ \ (\text{DC}),\qquad
  X[N/2]=\sum_n (-1)^n x[n]\ \ (\text{Nyquist}).
  $$
- Storage drops from $N$ complex $=16N$ bytes (re+im, f32) to $(\tfrac N2{+}1)$ complex
  $\approx 8N$ bytes. **That is the ×0.5.**

**Why this is *extra* free for the FNO:** the spectral conv keeps only the lowest
`n_modes` frequencies and discards the rest anyway. The high half we're dropping via
Hermitian symmetry is *already thrown away* downstream — so rfft removes bytes the FNO
never wanted. Perfect alignment.

The remaining question is purely *how to compute* those $\tfrac N2+1$ bins without paying
for the full complex transform. Two standard tricks.

---

## 2. Trick A — "two reals for one complex" (pack two signals)

Best when you have **many** real signals of the same length — which is exactly the FNO
case: $K=\text{batch}\cdot\text{width}\approx 1024$ real rows of length $N$.

Take two real signals $a,b$ and pack them into one complex signal

$$
z[n] \;=\; a[n] + i\,b[n].
$$

One length-$N$ **complex** FFT gives $Z=\text{DFT}(z)=A + iB$ where $A=\text{DFT}(a)$,
$B=\text{DFT}(b)$ — but linearly entangled. Unpack using Hermitian symmetry of $A,B$
(both are real-input transforms, so $A[N{-}k]=\overline{A[k]}$, likewise $B$):

$$
\boxed{\;
A[k] = \tfrac12\big(Z[k] + \overline{Z[N-k]}\big),
\qquad
B[k] = \tfrac{1}{2i}\big(Z[k] - \overline{Z[N-k]}\big).
\;}
$$

**Check** (linearity + the two symmetries):
$Z[k]=A[k]+iB[k]$ and $\overline{Z[N-k]}=\overline{A[N-k]}+\overline{iB[N-k]}
=A[k]-iB[k]$; add/subtract to isolate $A[k],B[k]$. $\blacksquare$

**Payoff.** $K$ real FFTs $\to$ $K/2$ complex FFTs + a cheap $O(N)$ unpack. The transform
count — and thus the dominant bytes — **halves**, and the **FFT kernel is untouched**: pack
before, unpack after. This is the pragmatic F3 for us.

Cost of the unpack: one pass reading $Z$ and $\overline{Z[N-k]}$ (a reversed view) →
$O(N)$ VPU, guard-free (real arithmetic). Fold it into the mode-selection step so it is not
a separate HBM pass.

---

## 3. Trick B — "half-length + twist" (one signal, half the length)

Best for a **single** real signal (or when you can't pair): compute a length-$N$ real FFT
via a length-$N/2$ **complex** FFT.

Split $x$ into even/odd interleaved halves and pack them as the re/im of a half-length
signal:

$$
z[j] \;=\; x[2j] + i\,x[2j+1], \qquad j=0,\dots,\tfrac N2-1 .
$$

Let $Z=\text{DFT}_{N/2}(z)$. Define the even/odd sub-DFTs
$E=\text{DFT}_{N/2}(x_{\text{even}})$, $O=\text{DFT}_{N/2}(x_{\text{odd}})$; unpack them from
$Z$ exactly as in Trick A:

$$
E[k]=\tfrac12\big(Z[k]+\overline{Z[\frac N2-k]}\big),\qquad
O[k]=\tfrac{1}{2i}\big(Z[k]-\overline{Z[\frac N2-k]}\big).
$$

Then the standard radix-2 recombination (the "twist" by $W_N^k$) rebuilds the length-$N$
spectrum:

$$
\boxed{\,X[k] = E[k] + W_N^{\,k}\,O[k],\qquad
X[k+\tfrac N2] = E[k] - W_N^{\,k}\,O[k]\,}\qquad k=0,\dots,\tfrac N2-1 .
$$

**Payoff.** Each transform runs at **half length** ($N/2$), halving its bytes, at the cost
of one extra $O(N)$ twiddle-recombination pass. Since we're memory-bound and $L$ is tiny,
halving $N$ is a clean ×0.5 — but it *does* touch the transform length (and, in our radix-B
world, the factorization), so it's more invasive than Trick A.

---

## 4. Which one for us

| | Trick A (pair channels) | Trick B (half-length) |
|---|---|---|
| needs | ≥2 real signals (FNO: 1024) | single signal ok |
| FFT kernel | **unchanged** | length/factorization changes |
| extra pass | $O(N)$ unpack | $O(N)$ unpack + twist |
| byte cut | ×0.5 (half the transforms) | ×0.5 (half the length) |
| risk | trivial | interacts with radix-B split |

**Recommendation: Trick A.** For the FNO, $K\approx1024$ real rows pair perfectly into
$512$ complex transforms; the FFT engine you already have runs *as-is*, and the only new
code is a pack (before) and an unpack (after, folded into mode-selection). It banks the full
×0.5 with essentially zero kernel risk. Keep Trick B in reserve for the single-signal /
large-$N$ study where pairing isn't available.

Also, independent of A vs B: **stop casting real→complex64 on input.** Carry the real
signal as real until the first complex operation actually needs it. That alone removes the
up-front 2× inflation the current code eats on line 1.

---

## 5. What you'll write, and how we verify (offline)

Yours to implement (per our split — you write the FFT logic):

1. `pack`: `z = a + 1j*b` over paired channels (real input `(K, N)` → complex `(K/2, N)`).
2. run the existing complex FFT on `z`.
3. `unpack`: the boxed $A[k],B[k]$ formulas, keeping only $k=0..N/2$ (rfft truncation).

Verification (my side — tooling), all on this CPU box, **no TPU**:

- **correctness:** `unpack(fft(pack(a,b)))` vs `np.fft.rfft(a)` and `np.fft.rfft(b)`,
  rel-err < 1e-5, swept $N$.
- **the byte claim:** `inspect()` / `traffic_model.measured_bytes` on the rfft variant vs
  `baseline` — we want to *see* `bytes accessed` drop ≈ ×0.5. That number is the whole
  point of F3, and it's visible offline.

Then it slots into the ranking table next to `fused`, and we stack it with bf16 (F4) for
the ×0.25 the model predicts.
