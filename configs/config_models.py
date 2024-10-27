from typing import Literal, Optional
from pydantic import BaseModel


class ConfigDINO(BaseModel):
    patch_size: int = 16
    backbone_model: str = "vit_tiny"
    img_size: Optional[int] = None
    out_dim: Optional[int] = 192
    optimizer: str = "adamw"
    lr_scheduler_type: Literal["cosine", "linear", "plateau", "step"] = "linear"
    warmup_epochs: int = 10
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    student_temp: float = 0.1
    teacher_momentum_start: float = 0.996
    teacher_momentum_end: float = 1
    center_momentum: float = 0.9
    epochs: int = 100
    batch_size: int = 128
    min_lr: float = 1e-6
    start_lr: float = 5e-5
    seed: int = 42
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    lr_step_size: int = 10


class ConfigDINO_Head(BaseModel):
    out_dim: int = 1024
    hidden_dim: int = 256
    bottleneck_dim: int = 128
    use_bn: bool = False


class ConfigDataset(BaseModel):
    name: str = "CIFAR10"
    root: str = "./data"
    img_size: int = 32
    num_classes: int = 10
    global_crop_ratio: tuple[float, float] = (0.32, 1.0)
    local_crop_ratio: tuple[float, float] = (0.05, 0.32)
    local_crop_size: int = 8
    nb_local_crops: int = 8
    dataset_means: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    dataset_stds: tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
