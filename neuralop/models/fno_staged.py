from typing import List, Tuple, Union, Literal, Optional

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued, CGELU, ctanh
from ..layers.skip_connections import skip_connection
from ..layers.normalization_layers import AdaIN, InstanceNorm, BatchNorm
from ..utils import get_active_profiler, maybe_profile, validate_scaling_factor
from .base_model import BaseModel


class FNOStaged(BaseModel, name="FNO_Staged"):
    """FNO variant with per-layer Fourier mode counts.

    This allows shrinking the number of spectral parameters in later layers by
    providing per-layer n_modes (and optionally per-layer max_n_modes).
    """

    def __init__(
        self,
        n_modes_per_layer: List[Tuple[int, ...]],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: Optional[int] = None,
        max_n_modes_per_layer: Optional[List[Tuple[int, ...]]] = None,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm", "batch_norm"] = None,
        norm_groups: int = 1,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
        no_br: bool = False,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()

        if n_layers is not None and n_layers != len(n_modes_per_layer):
            raise ValueError(
                f"Got n_layers={n_layers} but n_modes_per_layer has length {len(n_modes_per_layer)}"
            )

        self.n_layers = len(n_modes_per_layer)
        self.n_dim = len(n_modes_per_layer[0])
        self.n_modes_per_layer = [list(m) for m in n_modes_per_layer]

        if max_n_modes_per_layer is None:
            max_n_modes_per_layer = self.n_modes_per_layer
        if len(max_n_modes_per_layer) != self.n_layers:
            raise ValueError(
                "max_n_modes_per_layer must be None or have the same length as n_modes_per_layer"
            )
        self.max_n_modes_per_layer = [list(m) for m in max_n_modes_per_layer]

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision
        self.stabilizer = stabilizer
        self.no_br = no_br

        if self.complex_data:
            self.non_linearity = CGELU

        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNO positional embedding with {positional_embedding}, "
                "expected one of 'grid', GridEmbeddingND"
            )

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = validate_scaling_factor(
            resolution_scaling_factor, self.n_dim, self.n_layers
        )

        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.hidden_channels,
                    self.hidden_channels,
                    n_modes=self.n_modes_per_layer[i],
                    max_n_modes=self.max_n_modes_per_layer[i],
                    resolution_scaling_factor=(
                        self.resolution_scaling_factor[i]
                        if self.resolution_scaling_factor is not None
                        else None
                    ),
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    fno_block_precision=fno_block_precision,
                    decomposition_kwargs=decomposition_kwargs,
                    complex_data=complex_data,
                    **(
                        {"enforce_hermitian_symmetry": enforce_hermitian_symmetry}
                        if issubclass(conv_module, SpectralConv)
                        else {}
                    ),
                )
                for i in range(self.n_layers)
            ]
        )
        for i, conv in enumerate(self.convs):
            setattr(conv, "profile_prefix", f"fno_blocks/{i}/spectral_conv")

        if fno_skip is not None:
            self.fno_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.hidden_channels,
                        self.hidden_channels,
                        skip_type=fno_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(self.n_layers)
                ]
            )
        else:
            self.fno_skips = None
        if self.complex_data and self.fno_skips is not None:
            self.fno_skips = nn.ModuleList([ComplexValued(x) for x in self.fno_skips])

        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        if self.use_channel_mlp:
            self.channel_mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.hidden_channels,
                        hidden_channels=round(self.hidden_channels * channel_mlp_expansion),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(self.n_layers)
                ]
            )
            if self.complex_data:
                self.channel_mlp = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp]
                )
            if channel_mlp_skip is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [
                        skip_connection(
                            self.hidden_channels,
                            self.hidden_channels,
                            skip_type=channel_mlp_skip,
                            n_dim=self.n_dim,
                        )
                        for _ in range(self.n_layers)
                    ]
                )
            else:
                self.channel_mlp_skips = None
            if self.complex_data and self.channel_mlp_skips is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp_skips]
                )

        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [InstanceNorm() for _ in range(self.n_layers * self.n_norms)]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=norm_groups, num_channels=self.hidden_channels)
                    for _ in range(self.n_layers * self.n_norms)
                ]
            )
        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    BatchNorm(n_dim=self.n_dim, num_features=self.hidden_channels)
                    for _ in range(self.n_layers * self.n_norms)
                ]
            )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(None, self.hidden_channels)
                    for _ in range(self.n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, batch_norm, ada_in]"
            )
        if self.complex_data and self.norm is not None:
            self.norm = nn.ModuleList([ComplexValued(x) for x in self.norm])

        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        self.lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    @staticmethod
    def _bitrev_spatial(x: torch.Tensor) -> torch.Tensor:
        """Permute each spatial dimension of x in bit-reversed order (P²=I).

        Applied once before and once after the Fourier layers. Dims that are
        not a power-of-2 are left unchanged (no-op, fallback to cuFFT path).
        """
        for dim in range(2, x.ndim):
            n = x.shape[dim]
            if n > 1 and (n & (n - 1)) == 0:
                bits = int(n).bit_length() - 1
                idx = torch.arange(n, dtype=torch.long, device=x.device)
                rev = torch.zeros_like(idx)
                for i in range(bits):
                    rev = (rev << 1) | ((idx >> i) & 1)
                x = torch.index_select(x, dim, rev)
        return x

    def _forward_block_postactivation(self, x, index=0, output_shape=None):
        profiler = get_active_profiler()
        prefix = f"fno_blocks/{index}"

        if self.fno_skips is not None:
            with maybe_profile(profiler, f"{prefix}/fno_skip", x.device):
                x_skip_fno = self.fno_skips[index](x)
            with maybe_profile(profiler, f"{prefix}/fno_skip_transform", x.device):
                x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            with maybe_profile(profiler, f"{prefix}/channel_mlp_skip", x.device):
                x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            with maybe_profile(profiler, f"{prefix}/channel_mlp_skip_transform", x.device):
                x_skip_channel_mlp = self.convs[index].transform(
                    x_skip_channel_mlp, output_shape=output_shape
                )

        if self.stabilizer == "tanh":
            with maybe_profile(profiler, f"{prefix}/stabilizer", x.device):
                if self.complex_data:
                    x = ctanh(x)
                else:
                    x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.norm is not None:
            with maybe_profile(profiler, f"{prefix}/norm_0", x.device):
                x_fno = self.norm[self.n_norms * index](x_fno)

        with maybe_profile(profiler, f"{prefix}/fno_skip_add", x.device):
            x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            with maybe_profile(profiler, f"{prefix}/activation_0", x.device):
                x = self.non_linearity(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                with maybe_profile(profiler, f"{prefix}/channel_mlp", x.device):
                    x = self.channel_mlp[index](x)
                with maybe_profile(profiler, f"{prefix}/channel_mlp_skip_add", x.device):
                    x = x + x_skip_channel_mlp
            else:
                with maybe_profile(profiler, f"{prefix}/channel_mlp", x.device):
                    x = self.channel_mlp[index](x)

        if self.norm is not None:
            with maybe_profile(profiler, f"{prefix}/norm_1", x.device):
                x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            with maybe_profile(profiler, f"{prefix}/activation_1", x.device):
                x = self.non_linearity(x)

        return x

    def _forward_block_preactivation(self, x, index=0, output_shape=None):
        profiler = get_active_profiler()
        prefix = f"fno_blocks/{index}"

        with maybe_profile(profiler, f"{prefix}/activation_0", x.device):
            x = self.non_linearity(x)

        if self.norm is not None:
            with maybe_profile(profiler, f"{prefix}/norm_0", x.device):
                x = self.norm[self.n_norms * index](x)

        if self.fno_skips is not None:
            with maybe_profile(profiler, f"{prefix}/fno_skip", x.device):
                x_skip_fno = self.fno_skips[index](x)
            with maybe_profile(profiler, f"{prefix}/fno_skip_transform", x.device):
                x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            with maybe_profile(profiler, f"{prefix}/channel_mlp_skip", x.device):
                x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            with maybe_profile(profiler, f"{prefix}/channel_mlp_skip_transform", x.device):
                x_skip_channel_mlp = self.convs[index].transform(
                    x_skip_channel_mlp, output_shape=output_shape
                )

        if self.stabilizer == "tanh":
            with maybe_profile(profiler, f"{prefix}/stabilizer", x.device):
                if self.complex_data:
                    x = ctanh(x)
                else:
                    x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        with maybe_profile(profiler, f"{prefix}/fno_skip_add", x.device):
            x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            with maybe_profile(profiler, f"{prefix}/activation_1", x.device):
                x = self.non_linearity(x)

        if self.norm is not None:
            with maybe_profile(profiler, f"{prefix}/norm_1", x.device):
                x = self.norm[self.n_norms * index + 1](x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                with maybe_profile(profiler, f"{prefix}/channel_mlp", x.device):
                    x = self.channel_mlp[index](x)
                with maybe_profile(profiler, f"{prefix}/channel_mlp_skip_add", x.device):
                    x = x + x_skip_channel_mlp
            else:
                with maybe_profile(profiler, f"{prefix}/channel_mlp", x.device):
                    x = self.channel_mlp[index](x)

        return x

    def forward(self, x, output_shape=None, **kwargs):
        if kwargs:
            warnings.warn(
                f"FNOStaged.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        profiler = get_active_profiler()

        if self.positional_embedding is not None:
            with maybe_profile(profiler, "fno/positional_embedding", x.device):
                x = self.positional_embedding(x)

        with maybe_profile(profiler, "fno/lifting", x.device):
            x = self.lifting(x)

        if self.domain_padding is not None:
            with maybe_profile(profiler, "fno/domain_padding/pad", x.device):
                x = self.domain_padding.pad(x)

        if self.no_br:
            x = self._bitrev_spatial(x)

        for layer_idx in range(self.n_layers):
            if self.preactivation:
                x = self._forward_block_preactivation(
                    x, layer_idx, output_shape=output_shape[layer_idx]
                )
            else:
                x = self._forward_block_postactivation(
                    x, layer_idx, output_shape=output_shape[layer_idx]
                )

        if self.no_br:
            x = self._bitrev_spatial(x)  # P² = I, restores natural order

        if self.domain_padding is not None:
            with maybe_profile(profiler, "fno/domain_padding/unpad", x.device):
                x = self.domain_padding.unpad(x)

        with maybe_profile(profiler, "fno/projection", x.device):
            x = self.projection(x)

        return x
