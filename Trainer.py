import warnings
import torch
import torchvision
import wandb
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset
from torch.optim import Optimizer
from tqdm import tqdm
from models.DINO import DINO
from models.DINO_loss import DINO_Loss
from utils import DataAugmentationDINO


class Trainer:
    def __init__(
        self,
        dino_config: ConfigDINO,
        dino_head_config: ConfigDINO_Head,
        dataset_config: ConfigDataset,
        device: str = "cuda",
    ):
        self.device = device
        self.dino_config = dino_config
        self.dino_head_config = dino_head_config
        self.dataset_config = dataset_config

        self.model = DINO(dino_config, dino_head_config, dataset_config)
        wandb.watch(self.model)

        lr = dino_config.start_lr * dino_config.batch_size / 256
        self.optimizer = self._init_optimizer(dino_config.optimizer, lr)
        self.dataloader = self._init_dataloader()
        self._set_schedulers(lr)

        self.loss_fn = DINO_Loss(dino_config)
        self.amp_enabled = True if self.device != "cpu" else False
        self.training_dtype = torch.float16 if self.amp_enabled else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _init_optimizer(self, optimizer_type: str, lr: float) -> Optimizer:
        """Initialize optimizer"""
        match optimizer_type:
            case "adamw":
                return torch.optim.AdamW(self.model.parameters(), lr=lr)
            case "adam":
                return torch.optim.Adam(self.model.parameters(), lr=lr)
            case "sgd":
                return torch.optim.SGD(self.model.parameters())
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _init_dataloader(self) -> torch.utils.data.DataLoader:
        """Initialize dataloader
        Default values for training dataset are:
        - CIFAR10
        - CIFAR100
        - ImageNet

        WARNING: If the name of the dataset is neither of those, we will try to load from the given root
        """
        dataset = None
        match self.dataset_config.name:
            case "CIFAR10":
                dataset = torchvision.datasets.CIFAR10(
                    self.dataset_config.root,
                    train=True,
                    download=True,
                    transform=DataAugmentationDINO(self.dataset_config),
                )
            case "CIFAR100":
                dataset = torchvision.datasets.CIFAR100(
                    self.dataset_config.root,
                    train=True,
                    download=True,
                    transform=DataAugmentationDINO(self.dataset_config),
                )
            case "ImageNet":
                dataset = torchvision.datasets.ImageNet(
                    self.dataset_config.root,
                    train=True,
                    download=True,
                    transform=DataAugmentationDINO(self.dataset_config),
                )
            case _:
                warnings.warn(
                    f"Unsupported dataset detected, will try to load it from disk"
                )
                dataset = torchvision.datasets.ImageFolder(
                    self.dataset_config.root,
                    transform=DataAugmentationDINO(self.dataset_config),
                )

        return torch.utils.data.DataLoader(
            dataset,
            self.dino_config.batch_size,
            shuffle=True,
            generator=torch.Generator(device=self.device),
        )

    def _set_schedulers(self, lr) -> None:
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-5,
            end_factor=lr,
            total_iters=self.dino_config.warmup_epochs * len(self.dataloader),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.dino_config.epochs * len(self.dataloader),
            eta_min=lr,
        )

    def train_one_epoch(self, epoch: int, loop: tqdm) -> None:
        """Train for one epoch"""
        for _, (crops, _) in loop:
            with torch.autocast(
                device_type=self.device,
                dtype=self.training_dtype,
                enabled=self.amp_enabled,
            ):
                student_out, teacher_out = self.model(crops, training=True)
                loss = self.loss_fn(student_out, teacher_out)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.update_teacher(self.dino_config.teacher_momemtum)
            self.optimizer.zero_grad()
            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
            )

            loop.set_description(f"Epoch [{epoch} / {self.dino_config.epochs}]")
            loop.set_postfix(
                {
                    "Loss": loss.item(),
                    "Learning rate": self.scheduler.get_last_lr()[0],
                }
            )

    def warmup_train(self):
        """Warmup training run with linear lr scheduler"""
        for warmup_epoch in range(1, self.dino_config.warmup_epochs + 1):
            loop = tqdm(
                enumerate(self.dataloader),
                desc="Warmup training",
                total=len(self.dataloader),
                ascii="=",
            )
            self.train_one_epoch(self.scaler, warmup_epoch, loop)
            self.warmup_scheduler.step(warmup_epoch)
        print("Warmup done !\n")

    def train(self, warmup=True, start_epoch=1):
        """
        Train the model using given parameters and cosine scheduler.
        """
        self.model.train()
        if warmup:
            self.warmup_train()
        for epoch in range(start_epoch, self.dino_config.epochs + 1):
            loop = tqdm(
                enumerate(self.dataloader), desc="Training", total=len(self.dataloader)
            )
            self.train_one_epoch(epoch, loop)

            if epoch % self.dino_config.checkpoint_freq == 0:
                self.save_checkpoint(epoch)
            self.scheduler.step()

        print("Training over !\n")

    def save_checkpoint(self, epoch: int) -> None:
        """Save current trainer state to disk"""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp_enabled": self.amp_enabled,
            "epoch": epoch,
            "dataset_config": self.dataset_config.model_dump(),
            "dino_config": self.dino_config.model_dump(),
            "dino_head_config": self.dino_head_config.model_dump(),
            "loss_center": self.loss_fn.center,
        }

        torch.save(
            state_dict, self.dino_config.checkpoint_dir + f"{epoch}_checkpoint.pt"
        )
