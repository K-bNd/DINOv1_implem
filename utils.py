from typing import Callable
from pydantic import BaseModel, ValidationError
import torch
import timm
import torch.nn as nn
import torchvision.transforms.v2 as v2_transforms
import torchvision
import warnings
from yaml import load, FullLoader
import os
import json
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset


def save_config(path: str, model: BaseModel):
    """Saves Pydantic config to disk"""
    with open(os.path.join(path, f"{model.__repr_name__()}.json"), "w") as f:
        json.dump(model.model_dump_json(), f)


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


def init_dataloader(
    dataset_name: str,
    root: str,
    batch_size: int,
    device: str,
    transforms: Callable | None = None,
) -> list[torch.utils.data.DataLoader]:
    """Initialize dataloader
    Default values for training dataset are:
    - CIFAR10
    - CIFAR100
    - ImageNet

    WARNING: If the name of the dataset is neither of those, we will try to load from the given root
    """
    train_dataset = None
    test_dataset = None
    match dataset_name:
        case "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                root,
                train=True,
                download=True,
                transform=transforms,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root,
                train=False,
                download=True,
                transform=transforms,
            )
        case "CIFAR100":
            train_dataset = torchvision.datasets.CIFAR100(
                root,
                train=True,
                download=True,
                transform=transforms,
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root,
                train=False,
                download=True,
                transform=transforms,
            )
        case "ImageNet":
            train_dataset = torchvision.datasets.ImageNet(
                root,
                split="train",
                transform=transforms,
            )
            test_dataset = torchvision.datasets.ImageNet(
                root, split="val", transform=transforms
            )
        case _:
            warnings.warn(
                f"Unsupported dataset detected, will try to load it from disk"
            )
            train_dataset = torchvision.datasets.ImageFolder(
                root,
                transform=transforms,
            )

    return [
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        ),
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        ),
    ]


def load_model(filename: str) -> nn.Module:
    model = timm.models.VisionTransformer(num_classes=0)
    model.load_state_dict(torch.load(filename))
    return model


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
                v2_transforms.ToImage(),
                v2_transforms.ToDtype(torch.float32, scale=True),
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
