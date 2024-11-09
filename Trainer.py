import logging
import os
import torch
import wandb
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset
from torch.optim import Optimizer
from tqdm import tqdm
from models.DINO import DINO
from models.DINO_loss import DINO_Loss
from utils import DataTransformDINO, cosine_scheduler, init_dataloader


class Trainer:
    def __init__(
        self,
        run_id: str,
        dino_config: ConfigDINO,
        dino_head_config: ConfigDINO_Head,
        dataset_config: ConfigDataset,
        checkpoint_path: str,
        checkpoint_freq: int,
        device: str = "cuda",
    ):
        self.run_id = run_id
        self.device = device
        self.dino_config = dino_config
        self.dino_head_config = dino_head_config
        self.dataset_config = dataset_config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq

        self.model = DINO(dino_config, dino_head_config, dataset_config)
        wandb.watch(self.model)

        self.optimizer = self._init_optimizer(dino_config.optimizer)
        self.dataloader, _ = init_dataloader(
            self.dataset_config.name,
            self.dataset_config.root,
            self.dino_config.batch_size,
            self.device,
            transforms=DataTransformDINO(dataset_config, dino_config),
        )

        self.scheduler = self._set_scheduler(
            self.dino_config.lr_scheduler_type, len(self.dataloader)
        )

        self.loss_fn = DINO_Loss(dino_config, dino_head_config.out_dim)
        self.amp_enabled = True if self.device != "cpu" else False
        self.training_dtype = torch.float16 if self.amp_enabled else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _init_optimizer(self, optimizer_type: str) -> Optimizer:
        """Initialize optimizer"""
        match optimizer_type:
            case "adamw":
                return torch.optim.AdamW(
                    self.model.parameters(),
                    weight_decay=self.dino_config.weight_decay_start,
                )
            case "adam":
                return torch.optim.Adam(
                    self.model.parameters(),
                    weight_decay=self.dino_config.weight_decay_start,
                )
            case "sgd":
                return torch.optim.SGD(
                    self.model.parameters(),
                    weight_decay=self.dino_config.weight_decay_start,
                )
            case _:
                logging.error(f"Unsupported optimizer: {optimizer_type}")
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _set_scheduler(
        self, scheduler_type: str, nb_iters: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            total_iters=self.dino_config.warmup_epochs * nb_iters,
        )
        match scheduler_type:
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.dino_config.epochs * nb_iters,
                    eta_min=self.dino_config.min_lr,
                )
            case "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=self.dino_config.batch_size / 256,
                    total_iters=self.dino_config.epochs * nb_iters,
                )
            case "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.1,
                    patience=5 * nb_iters,
                    min_lr=self.dino_config.min_lr
                )
                return scheduler
            case "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.dino_config.lr_step_size,
                )
            case _:
                logging.error(f"Unsupported scheduler: {scheduler_type}")
                raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        return torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, scheduler])

    def train_one_epoch(self, epoch: int, loop: tqdm, warmup=False) -> None:
        """Train for one epoch"""
        self.loss_fn.update_teacher_temp(epoch)
        for it, (crops, _) in loop:
            self.optimizer.zero_grad()
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
            self.model.update_teacher(
                epoch * loop.total + it, loop.total * self.dino_config.epochs
            )
            self.optimizer.param_groups[0]["weight_decay"] = cosine_scheduler(
                epoch * loop.total + it,
                loop.total * self.dino_config.epochs,
                self.dino_config.weight_decay_start,
                self.dino_config.weight_decay_end,
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                    "lr": self.scheduler.get_last_lr()[0] if self.dino_config.lr_scheduler_type != "plateau" else [group['lr'] for group in self.optimizer.param_groups][0],
                }
            )

            if self.dino_config.lr_scheduler_type != "plateau":
                self.scheduler.step()
            else:
                self.scheduler.step(loss)

            if warmup:
                loop.set_description(
                    f"(Warmup) Epoch [{epoch + 1} / {self.dino_config.warmup_epochs}]"
                )
            else:
                loop.set_description(f"Epoch [{epoch + 1} / {self.dino_config.epochs}]")
            loop.set_postfix(
                {
                    "Loss": loss.item(),
                }
            )

    def warmup_train(self):
        """Warmup training run with linear lr scheduler"""
        for warmup_epoch in range(self.dino_config.warmup_epochs):
            loop = tqdm(
                enumerate(self.dataloader),
                desc="Warmup training",
                total=len(self.dataloader),
                ascii=True,
            )
            self.train_one_epoch(warmup_epoch, loop, warmup=True)
        logging.info("Warmup done !\n")

    def train(self, warmup=True, start_epoch=0):
        """
        Train the model using given parameters and cosine scheduler.
        """
        self.model.train()
        if warmup:
            self.warmup_train()
        for epoch in range(start_epoch, self.dino_config.epochs):
            loop = tqdm(
                enumerate(self.dataloader), desc="Training", total=len(self.dataloader)
            )
            self.train_one_epoch(epoch, loop)

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int) -> None:
        """Save current trainer state to disk"""
        state_dict = {
            "run_id": self.run_id,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "training_dtype": self.training_dtype,
            "amp_enabled": self.amp_enabled,
            "epoch": epoch,
            "dataset_config": self.dataset_config.model_dump(),
            "dino_config": self.dino_config.model_dump(),
            "dino_head_config": self.dino_head_config.model_dump(),
            "loss_center": self.loss_fn.center,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_freq": self.checkpoint_freq,
            "teacher_backbone": self.model.teacher_backbone.state_dict(),
            "teacher_head": self.model.teacher_head.state_dict(),
            "student_backbone": self.model.student_backbone.state_dict(),
            "student_head": self.model.student_head.state_dict(),
        }

        torch.save(
            state_dict,
            os.path.join(self.checkpoint_path, f"{self.run_id}_{epoch}_checkpoint.pt"),
        )
