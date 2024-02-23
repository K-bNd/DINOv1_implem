from pydantic import BaseModel


class ConfigDINO(BaseModel):
    patch_size: int = 16
    backbone_model: str = "vit_tiny"
    out_dim: int = 1024
    optimizer: str = "adamw"
    warmup_epochs: int = 5
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    teacher_momemtum: float = 0.996
    center_momentum: float = 0.9
    epochs: int = 100
    batch_size: int = 128
    min_lr: float = 1e-6
    start_lr: float = 5e-5
    seed: int = 42
    checkpoint_freq: int = 10
    log_path: str = "../logs"
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4


class ConfigDINO_Head(BaseModel):
    out_dim: int = 1024
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    use_bn: bool = False


class ConfigDataset(BaseModel):
    name: str = "CIFAR10"
    root: str = "../data"
    img_size: int = 32
    num_classes: int = 10
    global_crop_ratio: tuple[float, float] = (0.32, 1.0)
    local_crop_ratio: tuple[float, float] = (0.05, 0.32)
    local_crop_size: int = 14
    nb_local_crops: int = 8
    dataset_means: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    dataset_stds: tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
