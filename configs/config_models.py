from pydantic import BaseModel


class ConfigDINO(BaseModel):
    backbone_model: str
    scheduler: str
    teacher_temp: float
    student_temp: float
    center_momentum: float
    epochs: int
    batch_size: int
    seed: int


class ConfigDINO_Head(BaseModel):
    out_dim: int
    hidden_dim: int
    bottleneck_dim: int
    use_bn: bool


class ConfigDataset(BaseModel):
    img_size: int
    num_classes: int
    global_crop_ratio: tuple[float, float]
    local_crop_ratio: tuple[float, float]
    local_crop_size: int
    nb_local_crops: int
    dataset_means: tuple[float, float, float]
    dataset_stds: tuple[float, float, float]
