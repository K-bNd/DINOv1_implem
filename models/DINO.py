import torch
import timm
import torch.nn as nn
from models.DINO_head import DINO_Head
from configs.config_models import ConfigDINO, ConfigDINO_Head, ConfigDataset


class DINO(nn.Module):
    """DINOv1 model implementation"""

    def __init__(
        self,
        dino_config: ConfigDINO,
        dino_head_config: ConfigDINO_Head,
        dataset_config: ConfigDataset,
    ):
        super(DINO, self).__init__()

        self.model_config = dino_config
        self.dataset_config = dataset_config
        self.backbone_type = dino_config.backbone_model

        self.student_backbone, self.teacher_backbone = self._init_backbone()

        self.student_head = DINO_Head(self.embed_dim, dino_head_config)
        self.teacher_head = DINO_Head(self.embed_dim, dino_head_config)

        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    def _init_backbone(
        self,
    ) -> tuple[timm.models.VisionTransformer, timm.models.VisionTransformer]:
        """Initialize backbone model"""
        match self.backbone_type:
            case "vit_tiny":
                self.embed_dim = 192
                return timm.models.VisionTransformer(
                    img_size=self.dataset_config.img_size,
                    num_classes=0,
                    embed_dim=self.embed_dim,
                    num_heads=3,
                    patch_size=self.model_config.patch_size,
                    dynamic_img_size=True,
                ), timm.models.VisionTransformer(
                    img_size=self.dataset_config.img_size,
                    num_classes=0,
                    embed_dim=self.embed_dim,
                    num_heads=3,
                    patch_size=self.model_config.patch_size,
                )
            case "vit_small":
                self.embed_dim = 384
                return timm.models.VisionTransformer(
                    img_size=self.dataset_config.img_size,
                    num_classes=0,
                    embed_dim=self.embed_dim,
                    num_heads=6,
                    patch_size=self.model_config.patch_size,
                    dynamic_img_size=True,
                ), timm.models.VisionTransformer(
                    img_size=self.dataset_config.img_size,
                    num_classes=0,
                    embed_dim=self.embed_dim,
                    num_heads=6,
                    patch_size=self.model_config.patch_size,
                )
            case "vit_base":
                self.embed_dim = 768
                return timm.models.VisionTransformer(
                    img_size=self.dataset_config.img_size,
                    num_classes=0,
                    embed_dim=self.embed_dim,
                    num_heads=12,
                    patch_size=self.model_config.patch_size,
                    dynamic_img_size=True,
                ), timm.models.VisionTransformer(
                    img_size=self.dataset_config.img_size,
                    num_classes=0,
                    embed_dim=self.embed_dim,
                    num_heads=12,
                    patch_size=self.model_config.patch_size,
                )
            case _:
                raise ValueError(f"Unsupported backbone model: {self.backbone_type}")

    def update_teacher(self, teacher_momentum: float):
        with torch.no_grad():
            for (student_ps_backbone, teacher_ps_backbone), (
                student_ps_head,
                teacher_ps_head,
            ) in zip(
                zip(
                    self.student_backbone.parameters(),
                    self.teacher_backbone.parameters(),
                ),
                zip(self.student_head.parameters(), self.teacher_head.parameters()),
            ):
                teacher_ps_backbone.data = (
                    teacher_ps_backbone.data * (1 - teacher_momentum)
                    + student_ps_backbone.data * teacher_momentum
                )
                teacher_ps_head.data = (
                    teacher_ps_head.data * (1 - teacher_momentum)
                    + student_ps_head.data * teacher_momentum
                )

    def _student_forward(self, crops: list) -> torch.Tensor:
        # Extract and group image sizes for efficient forward passes
        image_sizes = torch.tensor([crop.shape[-1] for crop in crops])

        nb_global_crops, _ = torch.unique_consecutive(image_sizes, return_counts=True)[
            1
        ]

        global_crops = torch.cat(crops[:nb_global_crops])
        local_crops = torch.cat(crops[nb_global_crops:])

        global_crops_output = self.student_backbone(global_crops)
        local_crops_output = self.student_backbone(local_crops)

        output = torch.cat((global_crops_output, local_crops_output))

        return self.student_head(output)

    def _teacher_forward(self, global_crops: torch.Tensor) -> torch.Tensor:
        x = self.teacher_backbone(global_crops)
        x = self.teacher_head(x)

        return x

    def forward(self, x: list, training=False):
        if not training:
            return self.student_backbone(x)

        global_crops = torch.cat(x[:2], dim=0)

        student_out = self._student_forward(x)
        teacher_out = self._teacher_forward(global_crops)

        student_out = student_out.chunk(len(x))
        teacher_out = teacher_out.chunk(2)

        return student_out, teacher_out
