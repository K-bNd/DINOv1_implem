import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config_models import ConfigDINO


class DINO_Loss(nn.Module):
    def __init__(self, config: ConfigDINO):
        super(DINO_Loss, self).__init__()

        self.out_dim = config.out_dim
        self.teacher_temp = config.teacher_temp
        self.student_temp = config.student_temp
        self.center_momentum = config.center_momentum

        self.register_buffer("center", torch.zeros(1, config.out_dim))

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    def forward(self, _student_out, _teacher_out):
        student_out = [s / self.student_temp for s in _student_out]
        teacher_out = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in _teacher_out
        ]

        loss = 0
        count = 0

        for t_ix, t in enumerate(teacher_out):
            for s_ix, s in enumerate(student_out):
                if t_ix == s_ix:
                    continue

                tmp_loss = torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1)
                loss += tmp_loss.mean()
                count += 1

        loss /= count
        self.update_center(_teacher_out)

        return loss
