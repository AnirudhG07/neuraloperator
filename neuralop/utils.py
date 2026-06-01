from typing import List, Optional, Union
from math import prod
from pathlib import Path
from contextlib import contextmanager, nullcontext
import contextvars
import time
import torch

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False


def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )


def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2 * n_params
    return n_params


def wandb_login(api_key_file="../config/wandb_api_key.txt", key=None):
    if key is None:
        key = get_wandb_api_key(api_key_file)

    wandb.login(key=key)


def set_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Parameters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT.
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``).
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular
        coefficient is the coefficient's location along the diagonal, indexed
        from the top-left corner of the 2d FFT output.
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.sqrt(k_x**2 + k_y**2)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].abs() ** 2).sum(dim=1)

    spectrum = spectrum.mean(dim=0)
    return spectrum


Number = Union[float, int]


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None


def compute_rank(tensor):
    # Compute the matrix rank of a tensor
    rank = torch.matrix_rank(tensor)
    return rank


def compute_stable_rank(tensor):
    # Compute the stable rank of a tensor
    tensor = tensor.detach()
    fro_norm = torch.linalg.norm(tensor, ord="fro") ** 2
    l2_norm = torch.linalg.norm(tensor, ord=2) ** 2
    rank = fro_norm / l2_norm
    rank = rank
    return rank


def compute_explained_variance(frequency_max, s):
    # Compute the explained variance based on frequency_max and singular
    # values (s)
    s_current = s.clone()
    s_current[frequency_max:] = 0
    return 1 - torch.var(s - s_current) / torch.var(s)


def get_project_root():
    root = Path(__file__).parent.parent
    return root


_ACTIVE_PROFILER = contextvars.ContextVar("neuralop_active_profiler", default=None)


class TimingStat:
    def __init__(self):
        self.total_s = 0.0
        self.count = 0

    def update(self, duration_s: float):
        self.total_s += duration_s
        self.count += 1


class TimingProfiler:
    def __init__(self):
        self._stats = {}

    def reset(self):
        self._stats = {}

    def record(self, name: str, duration_s: float):
        if name not in self._stats:
            self._stats[name] = TimingStat()
        self._stats[name].update(duration_s)

    def section(self, name: str, device: Optional[torch.device] = None):
        return _TimingSection(self, name, device)

    def summary(self, title: Optional[str] = None, sort_by: str = "total", group_depth: Optional[int] = None):
        if group_depth is not None:
            stats = self._group_stats(group_depth)
        else:
            stats = self._stats

        total_s = sum(stat.total_s for stat in stats.values())
        items = list(stats.items())
        if sort_by == "name":
            items.sort(key=lambda item: item[0])
        else:
            items.sort(key=lambda item: item[1].total_s, reverse=True)

        lines = []
        if title:
            lines.append(title)
        header = "section | total(s) | avg(ms) | pct | count"
        lines.append(header)
        lines.append("-" * len(header))
        for name, stat in items:
            avg_ms = (stat.total_s / stat.count * 1000.0) if stat.count else 0.0
            pct = (stat.total_s / total_s * 100.0) if total_s else 0.0
            lines.append(
                f"{name} | {stat.total_s:.4f} | {avg_ms:.3f} | {pct:5.1f}% | {stat.count}"
            )
        return "\n".join(lines)

    def simple_summary(self, title: Optional[str] = None, top_k: int = 3, group_depth: Optional[int] = None):
        if group_depth is not None:
            stats = self._group_stats(group_depth)
        else:
            stats = self._stats

        total_s = sum(stat.total_s for stat in stats.values())
        items = sorted(stats.items(), key=lambda item: item[1].total_s, reverse=True)

        lines = []
        if title:
            lines.append(title)
        if not items or total_s == 0.0:
            lines.append("No timing data collected.")
            return "\n".join(lines)

        lines.append("Kid summary:")
        for name, stat in items[:top_k]:
            pct = (stat.total_s / total_s * 100.0) if total_s else 0.0
            lines.append(f"- {name} is a big eater: {pct:.1f}% of the measured time.")
        return "\n".join(lines)

    def _group_stats(self, group_depth: int):
        grouped = {}
        for name, stat in self._stats.items():
            parts = name.split("/")
            key = "/".join(parts[:group_depth]) if len(parts) >= group_depth else name
            if key not in grouped:
                grouped[key] = TimingStat()
            grouped[key].total_s += stat.total_s
            grouped[key].count += stat.count
        return grouped


class _TimingSection:
    def __init__(self, profiler: TimingProfiler, name: str, device: Optional[torch.device]):
        self._profiler = profiler
        self._name = name
        self._device = device
        self._start_time = None
        self._start_event = None
        self._end_event = None

    def __enter__(self):
        if self._device is not None and self._device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_event is not None:
            self._end_event.record()
            self._end_event.synchronize()
            duration_s = self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            duration_s = time.perf_counter() - self._start_time
        self._profiler.record(self._name, duration_s)
        return False


@contextmanager
def profiling(profiler: TimingProfiler):
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


def get_active_profiler() -> Optional[TimingProfiler]:
    return _ACTIVE_PROFILER.get()


def maybe_profile(profiler: Optional[TimingProfiler], name: str, device: Optional[torch.device] = None):
    if profiler is None:
        return nullcontext()
    return profiler.section(name, device)
