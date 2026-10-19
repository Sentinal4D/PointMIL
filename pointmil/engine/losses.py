"""Classification losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SmoothCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing as implemented in OpenPoints.

    The target distribution puts ``1 - eps`` on the true class and ``eps / (C - 1)``
    on each other class (this differs slightly from ``torch.nn.CrossEntropyLoss``,
    which uses ``eps / C`` for every class).
    """

    def __init__(self, label_smoothing: float = 0.2, weight: torch.Tensor | None = None):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = gt.contiguous().view(-1)
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                return -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            return -(one_hot * log_prb).sum(dim=1).mean()
        return F.cross_entropy(pred, gt, weight=self.weight)


def build_loss(loss_cfg: dict) -> nn.Module:
    """``{name: cross_entropy, label_smoothing: 0.0}``."""
    name = loss_cfg.get("name", "cross_entropy")
    if name != "cross_entropy":
        raise KeyError(f"Unknown loss '{name}'")
    ls = float(loss_cfg.get("label_smoothing", 0.0))
    weight = loss_cfg.get("class_weights")
    weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else None
    if ls > 0 or weight is not None:
        return SmoothCrossEntropy(ls, weight)
    return nn.CrossEntropyLoss()
