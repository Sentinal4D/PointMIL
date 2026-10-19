"""Contextual attention (Eq. 6 of the paper).

The attention weight of every point is replaced by the mean attention weight of
its ``k`` nearest neighbours in Euclidean space. ``k = 0`` disables the update.
"""

from __future__ import annotations

import torch

from pointmil.backbones.knn import knn_point


def contextual_attention(attention: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Smooth per-point attention weights over their k-nearest neighbourhood.

    Args:
        attention: ``[B, N, 1]`` attention weights in ``[0, 1]``.
        xyz: ``[B, N, 3]`` point coordinates used to build the k-NN graph.
        k: neighbourhood size including the point itself. ``k <= 0`` returns
            the input unchanged.

    Returns:
        ``[B, N, 1]`` smoothed attention weights.
    """
    if k is None or k <= 0:
        return attention
    neighbour_idx = knn_point(k, xyz, xyz)  # [B, N, k]
    B, N, K = neighbour_idx.shape
    expanded = attention.expand(B, N, K)
    neighbour_attention = torch.gather(expanded, 1, neighbour_idx.long())
    return neighbour_attention.mean(dim=-1).unsqueeze(-1)
