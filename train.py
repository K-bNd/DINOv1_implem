import torch
from torch.cuda import manual_seed_all as set_cuda_seed
import argparse
import wandb
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset
from utils import get_configs
from Trainer import Trainer


def train(configs: dict, save_path: str, checkpoint_path: str | None):
    set_cuda_seed(configs["dino_config"].seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    run = wandb.init(
        project="DINOv1",
        config=configs["dino_config"].model_dump()
        | configs["dino_head_config"].model_dump()
        | configs["dataset_config"].model_dump()
        | {"device": device},
        resume=True,
    )

    trainer = Trainer(
        configs["dino_config"],
        configs["dino_head_config"],
        configs["dataset_config"],
        device=device,
    )

    if not run.resumed:
        trainer.train()
    else:
        checkpoint = torch.load(wandb.restore(checkpoint_path))
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scaler.load_state_dict(checkpoint["scaler"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        trainer.amp_enabled = checkpoint["amp_enabled"]
        trainer.dataset_config = ConfigDataset(**checkpoint["dataset_config"])
        trainer.dino_config = ConfigDINO(**checkpoint["dino_config"])
        trainer.dino_head_config = ConfigDINO_Head(**checkpoint["dino_head_config"])
        trainer.loss_fn.center = checkpoint["loss_center"]
        trainer.train(warmup=False, start_epoch=checkpoint["epoch"])

    torch.save(trainer.model.state_dict(), save_path)
    # Log the model to the W&B run
    run.log_model(path=save_path, name=configs["dataset_config"].name)


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
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_weights/model.pt",
        help="Where to save the model after training",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Where we can resume training"
    )

    args = vars(parser.parse_args())

    train_configs: dict = get_configs(
        args, ["dino_config", "dino_head_config", "dataset_config"]
    )

    train(
        configs=train_configs,
        save_path=args["save_path"],
        checkpoint_path=args["checkpoint_path"],
    )
