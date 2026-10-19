"""Point cloud saliency maps (Zheng et al., ICCV 2019), the PSM baseline of Table 1.

The saliency of a point is minus the gradient of the classification loss with
respect to the point, projected onto the direction from the cloud's median centre
and scaled by the distance to the centre (dropping a point is approximated by
moving it to the centre). Higher values mean the point supports the prediction
more; the paper ranked points by this saliency and removed the highest first.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def point_cloud_saliency(model: nn.Module, points: torch.Tensor, labels: torch.Tensor, criterion: nn.Module | None = None) -> torch.Tensor:
    """Saliency scores ``[B, N]`` for ``points`` ``[B, N, C]`` (xyz first) and class ``labels`` ``[B]``."""
    criterion = criterion or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    pts = points.clone().detach().requires_grad_(True)
    out = model(pts)
    loss = criterion(out["bag_logits"], labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad = pts.grad.detach()[..., :3].cpu().numpy()
    xyz = pts.detach()[..., :3].cpu().numpy()
    centre = np.median(xyz, axis=1, keepdims=True)
    radius = np.sqrt(np.sum(np.square(xyz - centre), axis=2))
    axis = xyz - centre
    sphere_map = np.sum(grad * axis, axis=2) * radius
    if was_training:
        model.train()
    return torch.from_numpy(-sphere_map)


def rank_scores(scores: torch.Tensor) -> torch.Tensor:
    """Convert raw scores to ranks (the highest score gets the largest rank), as used for the PSM baseline."""
    order = torch.argsort(scores, dim=-1)
    ranks = torch.zeros_like(scores)
    ranks.scatter_(-1, order, torch.arange(scores.shape[-1], dtype=scores.dtype).expand_as(scores))
    return ranks
