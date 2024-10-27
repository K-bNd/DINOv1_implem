from typing import Callable
from pydantic import BaseModel, ValidationError
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2_transforms
import torchvision
import warnings
import os
import math
from yaml import load, FullLoader
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset
from torch.utils.data import DataLoader


def save_config(path: str, model: BaseModel) -> str:
    """
    Saves a Pydantic configuration model to a JSON file on disk.

    Args:
        path (str): The path to the directory where the JSON file will be saved.
        model (BaseModel): The Pydantic model instance containing the configuration data to be saved.

    Returns:
        str: The path to the created config file
    """
    config_path = os.path.join(path, f"{model.__repr_name__()}.json")
    with open(config_path, "w") as f:
        f.write(model.model_dump_json())

    return config_path


def get_configs(args: dict, options: list[str]):
    """
    Get configs from argparse and use pydantic for validation

    Args:
        args (dict): Dictionary containing arguments parsed from command line using argparse.
        options (list[str]): List of strings representing option names used in argparse.
            These options should correspond to keys in the `args` dictionary and filenames
            containing configuration data.

    Returns:
        dict: A dictionary containing validated configurations loaded from the specified files.
            Keys in the dictionary correspond to the option names, and values are instances
            of the appropriate Pydantic configuration models (ConfigDINO, ConfigDINO_Head, ConfigDataset).

    Raises:
        ValidationError: If any configuration file fails Pydantic validation, the corresponding
            error message is printed, but the function continues processing other files.
    """
    configs = {}
    config_types = [ConfigDINO, ConfigDINO_Head, ConfigDataset]

    for idx, option in enumerate(options):
        with open(args[option]) as f:
            try:
                configs[option] = config_types[idx](**load(f, Loader=FullLoader))
            except ValidationError as err:
                print(err)

    return configs


def get_latest_file(folder_path):
    """
    Gets the latest file from a folder based on modification time.

    Args:
        folder_path: The path to the folder containing the files.

    Returns:
        The path to the latest file, or None if the folder is empty.
    """
    files = os.listdir(folder_path)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found at given path")
    latest_file = max(
        files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f))
    )
    return os.path.join(folder_path, latest_file)


def cosine_scheduler(
    iters: int, total_iterations: int, initial_value: float, final_value: float
):
    """
    Calculates a schedule based on a cosine decay function.

    The schedule starts with an initial value (initial_value) and gradually
    converge to a final value (final_value) over the course of a specified
    number of iterations (total_iterations). The schedule follows a cosine decay pattern.

    Args:
        iters (int): The current iteration number.
        total_iterations (int): The total number of iterations for the schedule.
        initial_value (float): The initial value.
        final_value (float): The final value.

    Returns:
        float: The value for the current iteration.
    """
    return (
        initial_value
        + (final_value - initial_value)
        * (1 + math.cos(math.pi * iters / total_iterations))
        / 2
    )


def get_random_apply(transforms: list[v2_transforms.Transform], prob=0.5):
    """
    Creates a `v2_transforms.RandomApply` instance that applies a random subset of
    the provided transformations with a given probability.

    Args:
        transforms (List[Transform]): A list of `v2_transforms.Transform` objects representing
            the available transformations.
        prob (float, optional): The probability (between 0.0 and 1.0) of applying any
            individual transformation within the RandomApply instance. Defaults to 0.5.

    Returns:
        RandomApply: A `v2_transforms.RandomApply` instance configured with the provided
            transformations and probability.
    """
    return v2_transforms.RandomApply(nn.ModuleList(transforms), p=prob)


def init_dataloader(
    dataset_name: str,
    root: str,
    batch_size: int,
    device: str,
    transforms: Callable | None = None,
) -> list[torch.utils.data.DataLoader]:
    """
    Initializes dataloaders for training and testing datasets.

    Args:
        dataset_name (str): Name of the dataset to load. Supported options include:
            "CIFAR10", "CIFAR100", "ImageNet" or any dataset from the given root.
        root (str): The path to the directory where the dataset is stored.
        batch_size (int): The number of samples per batch for loading data.
        device (str): The device to use for training ("cuda" or "cpu").
        transforms (Callable, optional): A function or callable object that performs
            transformations on the loaded data samples. Defaults to None.

    Returns:
        List[DataLoader]: A list containing two DataLoaders: one for the training set
            and another for the testing set.
    """
    train_dataset = None
    test_dataset = None
    match dataset_name:
        case "CIFAR10":
            warnings.warn(
                f"Current transformations are adapted to the ImageNet dataset"
            )
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
            )
        case "CIFAR100":
            warnings.warn(
                f"Current transformations are adapted to the ImageNet dataset"
            )
            train_dataset = torchvision.datasets.CIFAR100(
                root,
                train=True,
                download=True,
                transform=transforms,
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root, train=False, download=True
            )
        case "ImageNet":
            train_dataset = torchvision.datasets.ImageNet(
                root,
                split="train",
                transform=transforms,
            )
            test_dataset = torchvision.datasets.ImageNet(root, split="val")
        case "Imagenette":
            train_dataset = torchvision.datasets.Imagenette(
                root,
                download=True,
                split="train",
                size="160px",
                transform=transforms,
            )
            test_dataset = torchvision.datasets.Imagenette(
                root, download=True, split="val", size="160px"
            )
        case "STL10":
            train_dataset = torchvision.datasets.STL10(
                root,
                split="unlabeled",
                download=True,
                transform=transforms,
            )
            test_dataset = torchvision.datasets.STL10(root, split="test", download=True)
        case _:
            warnings.warn(
                f"Unsupported dataset detected, will try to load it from disk"
            )
            train_dataset = torchvision.datasets.ImageFolder(
                root,
                transform=transforms,
            )

    return [
        DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        ),
        DataLoader(
            test_dataset,
            batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        ),
    ]


class DataTransformDINO:
    def __init__(
        self,
        dataset_config: ConfigDataset,
        model_config: ConfigDINO,
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
                v2_transforms.Normalize(
                    dataset_config.dataset_means, dataset_config.dataset_stds
                ),
            ]
        )

        # first global crop
        self.global_transfo1 = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    dataset_config.img_size,
                    scale=dataset_config.global_crop_ratio,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                v2_transforms.GaussianBlur(7),
                normalize,
            ]
        )

        # second global crop
        self.global_transfo2 = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    dataset_config.img_size,
                    scale=dataset_config.global_crop_ratio,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                get_random_apply([v2_transforms.GaussianBlur(7)], prob=0.1),
                v2_transforms.RandomSolarize(128, p=0.2),
                normalize,
            ]
        )

        # transformation for the local small crops
        self.local_crops_number = dataset_config.nb_local_crops
        self.local_transfo = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    dataset_config.local_crop_size,
                    scale=dataset_config.local_crop_ratio,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                get_random_apply([v2_transforms.GaussianBlur(7)], prob=0.5),
                normalize,
            ]
        )

        self.model_config = model_config

    def __call__(self, image):
        if self.model_config.img_size:
            image = v2_transforms.Resize(self.model_config.img_size)(image)
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
