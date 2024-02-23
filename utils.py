from pydantic import ValidationError
import torch.nn as nn
from torch import manual_seed as set_torch_seed
from torch.cuda import manual_seed as set_cuda_seed
import torchvision.transforms.v2 as v2_transforms
from yaml import load, FullLoader

from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset


def get_configs(args: dict, options: list[str]):
    """Get configs from argparse and use pydantic for validation"""
    configs = {}
    config_types = [ConfigDINO, ConfigDINO_Head, ConfigDataset]

    for idx, option in enumerate(options):
        with open(args[option]) as f:
            try:
                configs[option] = config_types[idx](**load(f, Loader=FullLoader))
            except ValidationError as err:
                print(err)

    return configs


def get_random_apply(transforms: list[v2_transforms.Transform], prob=0.5):
    """Apply RandomApply transformations with a given probability"""
    return v2_transforms.RandomApply(nn.ModuleList(transforms), p=prob)


def set_seeds(seed: int):
    """Set random seeds"""
    set_torch_seed(seed)
    set_cuda_seed(seed)


class DataAugmentationDINO:
    def __init__(
        self,
        config: ConfigDataset,
    ):
        flip_and_color_jitter = v2_transforms.Compose(
            [
                v2_transforms.RandomHorizontalFlip(p=0.5),
                v2_transforms.RandomApply(
                    [
                        v2_transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                v2_transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = v2_transforms.Compose(
            [
                v2_transforms.ToTensor(),
                v2_transforms.Normalize(config.dataset_means, config.dataset_stds),
            ]
        )

        blur_kernel_size = int(config.img_size * 0.1)

        # first global crop
        self.global_transfo1 = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    config.img_size,
                    scale=config.global_crop_ratio,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                v2_transforms.GaussianBlur(blur_kernel_size),
                normalize,
            ]
        )

        # second global crop
        self.global_transfo2 = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    config.img_size,
                    scale=config.global_crop_ratio,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                get_random_apply(
                    [v2_transforms.GaussianBlur(blur_kernel_size)], prob=0.1
                ),
                v2_transforms.RandomSolarize(128, p=0.2),
                normalize,
            ]
        )

        # transformation for the local small crops
        self.local_crops_number = config.nb_local_crops
        self.local_transfo = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    config.local_crop_size,
                    scale=config.local_crop_ratio,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                get_random_apply(
                    [v2_transforms.GaussianBlur(blur_kernel_size)], prob=0.5
                ),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
