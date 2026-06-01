from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNOStagedConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class DarcyOptConfig(OptimizationConfig):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5


class DarcyDatasetConfig(ConfigBase):
    folder: str = "~/data/darcy/"
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 16
    n_tests: List[int] = [100, 50]
    test_resolutions: List[int] = [16, 32]
    test_batch_sizes: List[int] = [16, 16]
    encode_input: bool = True
    encode_output: bool = True
    download: bool = True

class DarcyDatasetHeavyConfig(ConfigBase):
    folder: str = "~/data/darcy/"
    batch_size: int = 16
    n_train: int = 1000
    train_resolution: int = 32
    n_tests: List[int] = [100]
    test_resolutions: List[int] = [32]
    test_batch_sizes: List[int] = [16]
    encode_input: bool = True
    encode_output: bool = True
    download: bool = True

class FNOStagedConfigDefault(FNOStagedConfig):
    data_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 24
    n_modes_per_layer: List[List[int]] = [
            [16, 16],
            [16, 16],
            [12, 12],
            [8, 8],
        ]
    max_n_modes_per_layer: Optional[List[List[int]]] = None
    projection_channel_ratio: int = 2

class FNOStagedConfigHeavy(FNOStagedConfig):
    data_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 128
    n_modes_per_layer: List[List[int]] = [
            [12, 12],
            [12, 12],
            [12, 12],
            [12, 12],
        ]
    max_n_modes_per_layer: Optional[List[List[int]]] = None
    projection_channel_ratio: int = 2

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno_staged"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNOStagedConfigDefault()
    opt: OptimizationConfig = DarcyOptConfig()
    data: DarcyDatasetConfig = DarcyDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()

class Heavy(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno_staged"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNOStagedConfigHeavy()
    opt: OptimizationConfig = DarcyOptConfig()
    data: DarcyDatasetHeavyConfig = DarcyDatasetHeavyConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
