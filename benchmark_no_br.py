"""
Benchmark: standard SpectralConv vs no_br (bit-reversal-free) SpectralConv.

Measures wall-clock time for each sub-operation inside a single forward pass:
  FFT  |  R (weight contraction)  |  iFFT

Runs across multiple spatial resolutions and reports per-layer breakdown.
Produces a grouped bar chart saved to benchmark_no_br.png.

Usage:
    python benchmark_no_br.py                  # 2-D, batch=4, channels=32
    python benchmark_no_br.py --dim 1          # 1-D
    python benchmark_no_br.py --batch 8 --ch 64
"""

import argparse
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralop.layers.spectral_convolution import SpectralConv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Instrumented SpectralConv subclass
# ---------------------------------------------------------------------------

class TimedSpectralConv(SpectralConv):
    """Wraps the two forward paths to record sub-operation durations."""

    def _sync(self):
        if self._dev.type == "cuda":
            torch.cuda.synchronize()

    def _clock(self):
        self._sync()
        return time.perf_counter()

    def _timed_forward_standard(self, x, output_shape=None):
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        # --- FFT ---
        t0 = self._clock()
        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)
        t_fft = self._clock() - t0

        # --- R contraction ---
        out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        if self.complex_data:
            slices_w += [slice(s // 2, -s // 2) if s else slice(s, None) for s in starts]
        else:
            slices_w += [slice(s // 2, -s // 2) if s else slice(s, None) for s in starts[:-1]]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]

        weight_start_idx = 1 if self.separable else 2
        slices_x = [slice(None), slice(None)]
        for all_modes, kept_modes in zip(fft_size, list(weight.shape[weight_start_idx:])):
            center = all_modes // 2
            neg = kept_modes // 2
            pos = kept_modes // 2 + kept_modes % 2
            slices_x += [slice(center - neg, center + pos)]
        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)
        slices_x = tuple(slices_x)

        t1 = self._clock()
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=self.separable)
        t_R = self._clock() - t1

        # --- iFFT ---
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
        if output_shape is not None:
            mode_sizes = output_shape
        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

        t2 = self._clock()
        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            if self.enforce_hermitian_symmetry:
                out_fft = torch.fft.ifftn(out_fft, s=mode_sizes[:-1], dim=fft_dims[:-1], norm=self.fft_norm)
                out_fft[..., 0].imag.zero_()
                if mode_sizes[-1] % 2 == 0:
                    out_fft[..., -1].imag.zero_()
                x = torch.fft.irfft(out_fft, n=mode_sizes[-1], dim=fft_dims[-1], norm=self.fft_norm)
            else:
                x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        t_ifft = self._clock() - t2

        if self.bias is not None:
            x = x + self.bias
        return x, t_fft, t_R, t_ifft

    def _timed_forward_no_br(self, x, output_shape=None):
        if self.complex_data or self.fno_block_precision != "full":
            out, *_ = self._timed_forward_standard(x, output_shape)
            return out, 0.0, 0.0, 0.0

        batchsize, channels, *mode_sizes = x.shape
        fft_dims = list(range(-self.order, 0))

        # --- FFT (butterfly only) ---
        t0 = self._clock()
        x = x.to(torch.cfloat)
        for d in fft_dims:
            x = self._fft_no_br(x, dim=d)
        t_fft = self._clock() - t0

        fft_size = list(mode_sizes)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat
        )
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        slices_w += [slice(None, -s) if s else slice(None) for s in starts]
        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]
        weight_start_idx = 1 if self.separable else 2
        slices_x = [slice(None), slice(None)]
        for kept_modes in weight.shape[weight_start_idx:]:
            slices_x += [slice(None, kept_modes)]
        slices_x = tuple(slices_x)

        # --- R contraction ---
        t1 = self._clock()
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=self.separable)
        t_R = self._clock() - t1

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
        if output_shape is not None:
            mode_sizes = output_shape

        # --- iFFT (butterfly only) ---
        t2 = self._clock()
        y = out_fft
        for d in fft_dims:
            y = self._ifft_no_br(y, dim=d, n=mode_sizes[d])
        x = y.real
        t_ifft = self._clock() - t2

        if self.bias is not None:
            x = x + self.bias
        return x, t_fft, t_R, t_ifft

    def timed_forward(self, x, output_shape=None):
        if self.no_br:
            return self._timed_forward_no_br(x, output_shape)
        return self._timed_forward_standard(x, output_shape)

    @property
    def _dev(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def make_conv(n_modes, in_ch, out_ch, no_br, device):
    conv = TimedSpectralConv(
        in_channels=in_ch,
        out_channels=out_ch,
        n_modes=n_modes,
        no_br=no_br,
    ).to(device)
    return conv


def run_one(conv, x, n_warmup=5, n_trials=30):
    """Warm up then measure n_trials forward passes. Returns (fft, R, ifft) mean seconds."""
    device = next(conv.parameters()).device

    for _ in range(n_warmup):
        conv.timed_forward(x)

    t_fft_all, t_R_all, t_ifft_all = [], [], []
    for _ in range(n_trials):
        _, tf, tr, ti = conv.timed_forward(x)
        t_fft_all.append(tf)
        t_R_all.append(tr)
        t_ifft_all.append(ti)

    return (
        np.mean(t_fft_all) * 1e3,   # ms
        np.mean(t_R_all)  * 1e3,
        np.mean(t_ifft_all) * 1e3,
    )


def benchmark(dim, batch, channels, resolutions, n_modes_fraction=0.25,
              n_warmup=5, n_trials=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  dim={dim}  batch={batch}  channels={channels}")
    print(f"{'Resolution':<14}{'Mode':>10}  {'FFT (ms)':>10}  {'R (ms)':>10}  {'iFFT (ms)':>10}  {'Total (ms)':>11}")
    print("-" * 70)

    results = {}   # resolution -> {standard: (fft,R,ifft), no_br: (...)}

    for res in resolutions:
        n_modes = tuple([max(2, int(res * n_modes_fraction))] * dim)
        spatial = tuple([res] * dim)
        x = torch.randn(batch, channels, *spatial, device=device)

        for label, no_br in [("standard", False), ("no_br", True)]:
            conv = make_conv(n_modes, channels, channels, no_br=no_br, device=device)
            tf, tr, ti = run_one(conv, x, n_warmup=n_warmup, n_trials=n_trials)
            results.setdefault(res, {})[label] = (tf, tr, ti)
            total = tf + tr + ti
            print(f"  {res:<12}  {label:>10}  {tf:10.3f}  {tr:10.3f}  {ti:10.3f}  {total:11.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, dim, output_path="benchmark_no_br.png"):
    resolutions = sorted(results.keys())
    labels = ["standard", "no_br"]
    colors = {
        "standard": {"FFT": "#4C72B0", "R": "#55A868", "iFFT": "#C44E52"},
        "no_br":    {"FFT": "#8172B2", "R": "#CCB974", "iFFT": "#64B5CD"},
    }
    hatches = {"standard": "", "no_br": "///"}

    x_pos = np.arange(len(resolutions))
    width = 0.35
    offsets = {"standard": -width / 2, "no_br": width / 2}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"SpectralConv timing: standard vs no_br  ({dim}D spatial)\n"
        "no_br skips bit-reversal permutations; R is permuted once offline",
        fontsize=12,
    )

    # --- Left: stacked bar (FFT | R | iFFT breakdown) ---
    ax = axes[0]
    for mode in labels:
        offset = offsets[mode]
        bottoms = np.zeros(len(resolutions))
        for part, key in [("FFT", 0), ("R", 1), ("iFFT", 2)]:
            vals = np.array([results[r][mode][key] for r in resolutions])
            ax.bar(
                x_pos + offset, vals, width,
                bottom=bottoms,
                label=f"{mode} – {part}",
                color=colors[mode][part],
                hatch=hatches[mode],
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.set_xlabel("Spatial resolution (per dim)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-operation breakdown (stacked)")
    ax.legend(fontsize=7, ncol=2)

    # --- Right: speedup line (total no_br / total standard) ---
    ax2 = axes[1]
    speedups = []
    for r in resolutions:
        std_total = sum(results[r]["standard"])
        br_total  = sum(results[r]["no_br"])
        speedups.append(std_total / br_total if br_total > 0 else float("nan"))

    ax2.plot(resolutions, speedups, "o-", color="#C44E52", linewidth=2, markersize=7)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.fill_between(resolutions, 1.0, speedups,
                     where=[s > 1 for s in speedups], alpha=0.15, color="#C44E52")
    ax2.set_xlabel("Spatial resolution (per dim)")
    ax2.set_ylabel("Speedup  (standard total / no_br total)")
    ax2.set_title("Overall speedup of no_br vs standard")
    for r, s in zip(resolutions, speedups):
        ax2.annotate(f"{s:.2f}×", (r, s), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",   type=int, default=2, choices=[1, 2, 3],
                        help="Spatial dimensionality (default: 2)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ch",    type=int, default=32, dest="channels")
    parser.add_argument("--warmup",  type=int, default=5)
    parser.add_argument("--trials",  type=int, default=30)
    parser.add_argument("--out",   type=str, default="benchmark_no_br.png")
    args = parser.parse_args()

    dim = args.dim
    if dim == 1:
        resolutions = [64, 128, 256, 512, 1024]
    elif dim == 2:
        resolutions = [16, 32, 64, 128, 256]
    else:
        resolutions = [8, 16, 32, 64]

    results = benchmark(
        dim=dim,
        batch=args.batch,
        channels=args.channels,
        resolutions=resolutions,
        n_warmup=args.warmup,
        n_trials=args.trials,
    )

    plot_results(results, dim=dim, output_path=args.out)
