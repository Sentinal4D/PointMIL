"""The PointMIL model: a per-point feature extractor followed by an MIL pooling head."""

from __future__ import annotations

import torch
from torch import nn


class PointMIL(nn.Module):
    """Compose a backbone and a pooling head.

    The backbone maps ``[B, N, C_in]`` points to ``(xyz [B, N, 3], features [B, d, N])``.
    The pooling head maps ``(xyz, features)`` to a dict with ``bag_logits`` ``[B, C]``
    and ``interpretation`` ``[B, C, N]`` (plus head-specific extras).
    """

    def __init__(self, backbone: nn.Module, pooling: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling

    def forward(self, points: torch.Tensor) -> dict:
        xyz, features = self.backbone(points)
        out = self.pooling(xyz, features)
        out["xyz"] = xyz[..., :3]  # some backbones pass all input channels on as coordinates
        return out

    @staticmethod
    def interpret(output: dict) -> torch.Tensor:
        """Point-level interpretation ``[B, C, N]`` from a forward output."""
        return output["interpretation"]

    @torch.no_grad()
    def point_importance(self, points: torch.Tensor, target_class: torch.Tensor | int | None = None) -> torch.Tensor:
        """Per-point importance ``[B, N]`` for ``target_class`` (default: predicted class); see ``importance_from_output``."""
        return importance_from_output(self.forward(points), target_class)


def importance_from_output(out: dict, target_class: torch.Tensor | int | None = None) -> torch.Tensor:
    """Per-point importance ``[B, N]`` from a forward output.

    Attention pooling scores every point with one attention weight that does not depend on the
    class (its output has ``attention`` but no ``instance_logits``); that weight is returned as is.
    For the class-specific heads (instance, additive, conjunctive, gap) the interpretation is
    softmax-normalised over the class axis (Section 3.4 of the paper) and the row of
    ``target_class`` is returned (default: the predicted class).
    """
    if "attention" in out and "instance_logits" not in out:
        return out["attention"].squeeze(-1)
    probs = torch.softmax(out["interpretation"], dim=1)
    batch = probs.shape[0]
    if target_class is None:
        target_class = out["bag_logits"].argmax(dim=1)
    if isinstance(target_class, int):
        target_class = torch.full((batch,), target_class, dtype=torch.long)
    return probs[torch.arange(batch, device=probs.device), target_class.to(probs.device)]

