import torch
from torch.cuda import manual_seed_all as set_cuda_seed
import argparse
import wandb
from utils import get_configs
from Trainer import Trainer


def train(configs: dict):
    set_cuda_seed(configs["dino_config"].seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    wandb.init(
        project="DINOv1",
        config=configs["dino_config"].model_dump()
        | configs["dino_head_config"].model_dump()
        | configs["dataset_config"].model_dump()
        | {"device": device},
    )

    trainer = Trainer(
        configs["dino_config"],
        configs["dino_head_config"],
        configs["dataset_config"],
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO")

    parser.add_argument(
        "--dino-config",
        type=str,
        default="configs/dino.yml",
        help="Config YAML file for DINO hyperparameters",
    )
    parser.add_argument(
        "--dino_head-config",
        type=str,
        default="configs/dino_head.yml",
        help="Config YAML file for DINO head hyperparameters",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/cifar10.yml",
        help="Config YAML file for the dataset",
    )

    args = vars(parser.parse_args())

    train_configs: dict = get_configs(
        args, ["dino_config", "dino_head_config", "dataset_config"]
    )

    train(configs=train_configs)
