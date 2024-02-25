import torch.nn as nn
import torch.nn.functional as F
from configs.config_models import ConfigDINO_Head


class DINO_Head(nn.Module):
    def __init__(self, in_dim: int, config: ConfigDINO_Head) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = config.hidden_dim
        self.bottleneck_dim = config.bottleneck_dim
        self.out_dim = config.out_dim

        layers = [
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        ]

        if config.use_bn:
            layers.insert(nn.BatchNorm1d(self.hidden_dim), 1)
            layers.insert(nn.BatchNorm1d(self.hidden_dim), 4)

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(self.bottleneck_dim, self.out_dim)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x
