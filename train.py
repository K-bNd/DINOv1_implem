import datetime
import os
import time
import torch
from torch.cuda import manual_seed_all as set_cuda_seed
import argparse
import wandb
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset
from utils import get_configs, save_config
from Trainer import Trainer


def train(configs: dict, save_path: str, checkpoint_path: str, checkpoint_freq: int):
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
        checkpoint_path,
        checkpoint_freq,
        device=device,
    )

    start_time = time.time()

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

    model_dir = os.path.join(
        save_path, f"{configs['dataset_config'].name}_{run._run_id}"
    )
    os.makedirs(model_dir, exist_ok=True)
    save_config(model_dir, trainer.dataset_config)
    save_config(model_dir, trainer.dino_config)
    save_config(model_dir, trainer.dino_head_config)
    student_path = os.path.join(model_dir, "student_backbone.pt")
    teacher_path = os.path.join(model_dir, "teacher_backbone.pt")
    torch.save(trainer.model.student_backbone.state_dict(), student_path)
    torch.save(trainer.model.teacher_backbone.state_dict(), teacher_path)

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    print(f"Training took {total_time_str} !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv1 training script")

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
        default="training_output",
        help="Where to save the model after training",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Where we can resume training",
    )

    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="How often we save our checkpoints",
    )

    args = vars(parser.parse_args())

    train_configs: dict = get_configs(
        args, ["dino_config", "dino_head_config", "dataset_config"]
    )

    os.makedirs(args["save_path"], exist_ok=True)
    os.makedirs(args["checkpoint_path"], exist_ok=True)

    train(
        configs=train_configs,
        save_path=args["save_path"],
        checkpoint_path=args["checkpoint_path"],
        checkpoint_freq=args["checkpoint_freq"],
    )
