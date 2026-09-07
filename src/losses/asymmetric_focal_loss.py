from __future__ import annotations
import torch
import torch.nn as nn

class AsymmetricFocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, label_smoothing=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        loss_pos = targets * torch.pow(1 - p, self.gamma_pos) * torch.log(p)
        p_neg = p
        if self.clip is not None and self.clip > 0:
            p_neg = torch.clamp(p_neg - self.clip, min=0.0)
        loss_neg = (1 - targets) * torch.pow(p_neg, self.gamma_neg) * torch.log(1 - p_neg + self.eps)
        return -(loss_pos + loss_neg).mean()

def build_loss(cfg):
    if cfg.loss.type != 'asymmetric_focal':
        raise ValueError(f'Unknown loss type: {cfg.loss.type}')
    return AsymmetricFocalLossWithSmoothing(
        gamma_pos=cfg.loss.gamma_pos,
        gamma_neg=cfg.loss.gamma_neg,
        clip=cfg.loss.clip,
        label_smoothing=cfg.loss.label_smoothing,
        eps=cfg.loss.eps,
    )
