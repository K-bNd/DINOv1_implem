import datetime
import os
import time
import torch
from torch.cuda import manual_seed_all as set_cuda_seed
import argparse
import wandb
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset
from utils import get_configs, get_latest_file, save_config
from Trainer import Trainer


def save_training_data(trainer: Trainer, save_path: str) -> None:
    """
    Saves the training configurations and model weights from a Trainer object to disk.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the training configurations
            and model state.
        save_path (str): The path to the directory where the training data will be saved.

    Raises:
        OSError: If there's an error creating the directory or saving the files.
    """
    os.makedirs(save_path, exist_ok=True)
    save_config(save_path, trainer.dataset_config)
    save_config(save_path, trainer.dino_config)
    save_config(save_path, trainer.dino_head_config)
    student_path = os.path.join(save_path, "student_backbone.pt")
    teacher_path = os.path.join(save_path, "teacher_backbone.pt")

    try:
        torch.save(trainer.model.student_backbone.state_dict(), student_path)
        torch.save(trainer.model.teacher_backbone.state_dict(), teacher_path)
    except OSError as err:
        raise OSError(f"Error saving model weights: {err}") from err


def train_from_checkpoint(
    configs: dict,
    save_path: str,
    checkpoint_path: str,
    checkpoint_freq: int,
    device: str,
) -> None:
    """
    Resumes training from a checkpoint loaded from the specified path.

    Args:
        configs (dict[str, Any]): A dictionary containing the training configurations.
        save_path (str): The path to the directory where the resumed training data will be saved.
        checkpoint_path (str): The path to the checkpoint file containing the previously trained model state.
        checkpoint_freq (int): The frequency (in epochs) at which to save checkpoints during training.
        device (str): The device to use for training ("cuda" or "cpu").

    Raises:
        OSError: If there's an error loading the checkpoint file.
    """
    try:
        checkpoint = torch.load(get_latest_file(checkpoint_path))
    except OSError as err:
        raise OSError(f"Error loading checkpoint file: {err}") from err

    run = wandb.init(
        project="DINOv1",
        config=configs["dino_config"].model_dump()
        | configs["dino_head_config"].model_dump()
        | configs["dataset_config"].model_dump()
        | {"device": device},
        resume="allow",
        id=checkpoint["run_id"],
    )

    trainer = Trainer(
        run._run_id,
        configs["dino_config"],
        configs["dino_head_config"],
        configs["dataset_config"],
        checkpoint_path,
        checkpoint_freq,
        device=device,
    )

    trainer.model.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.scaler.load_state_dict(checkpoint["scaler"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    trainer.training_dtype = checkpoint["training_dtype"]
    trainer.amp_enabled = checkpoint["amp_enabled"]
    trainer.dataset_config = ConfigDataset(**checkpoint["dataset_config"])
    trainer.dino_config = ConfigDINO(**checkpoint["dino_config"])
    trainer.dino_head_config = ConfigDINO_Head(**checkpoint["dino_head_config"])
    trainer.loss_fn.center = checkpoint["loss_center"]
    trainer.checkpoint_path = checkpoint["checkpoint_path"]
    trainer.checkpoint_freq = checkpoint["checkpoint_freq"]

    start_time = time.time()

    trainer.train(warmup=False, start_epoch=checkpoint["epoch"])

    model_dir = os.path.join(
        save_path, f"{configs['dataset_config'].name}_{run._run_id}"
    )

    save_training_data(trainer, model_dir)

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))

    print(f"New training phase took {total_time_str} !")


def train(
    configs: dict,
    save_path: str,
    checkpoint_path: str,
    checkpoint_freq: int,
    start_from_checkpoint: bool,
):
    """
    Trains a DINO model based on the provided configurations.

    Args:
        configs (dict[str, Any]): A dictionary containing the training configurations,
            including model architectures, dataset settings, and training hyperparameters.
        save_path (str): The path to the directory where the training data (configurations, model weights)
            will be saved.
        checkpoint_path (str): The path to a directory containing a previous training checkpoint.
            If provided and `start_from_checkpoint` is True, training resumes from this checkpoint.
        checkpoint_freq (int): The frequency (in epochs) at which to save checkpoints during training.
        start_from_checkpoint (bool, optional): If True, training resumes from the latest checkpoint
            in the `checkpoint_path` directory. Defaults to False (start fresh training).

    Raises:
        OSError: If `start_from_checkpoint` is True but no checkpoint is found.
    """

    set_cuda_seed(configs["dino_config"].seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    if start_from_checkpoint:
        train_from_checkpoint(
            configs, save_path, checkpoint_path, checkpoint_freq, device
        )
        return

    run = wandb.init(
        project="DINOv1",
        config=configs["dino_config"].model_dump()
        | configs["dino_head_config"].model_dump()
        | configs["dataset_config"].model_dump()
        | {"device": device},
        resume=start_from_checkpoint,
    )

    trainer = Trainer(
        run._run_id,
        configs["dino_config"],
        configs["dino_head_config"],
        configs["dataset_config"],
        checkpoint_path,
        checkpoint_freq,
        device=device,
    )

    start_time = time.time()

    trainer.train()

    model_dir = os.path.join(
        save_path, f"{configs['dataset_config'].name}_{run._run_id}"
    )

    save_training_data(trainer, model_dir)

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
        default="./configs/stl10.yml",
        help="Config YAML file for the dataset",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="training_output",
        help="Where to save the model after training",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints",
        help="Path to checkpoint values",
    )

    parser.add_argument(
        "--start-from-checkpoint",
        type=bool,
        default=False,
        help="Resume training from latest checkpoint",
    )

    parser.add_argument(
        "--checkpoint-freq",
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
        train_configs,
        args["save_path"],
        args["checkpoint_path"],
        args["checkpoint_freq"],
        args["start_from_checkpoint"],
    )
