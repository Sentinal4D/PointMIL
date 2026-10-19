"""Shared point-cloud neighbourhood utilities (pure PyTorch).

These functions follow the widely used implementations from
Pointnet_Pointnet2_pytorch (MIT licence, Xu Yan) that were copied into the
original PointMIL code base. They are kept numerically identical so that
checkpoints trained with the original code reproduce the same outputs.
"""

from __future__ import annotations

import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between every pair of points.

    Args:
        src: ``[B, N, C]`` source points.
        dst: ``[B, M, C]`` target points.

    Returns:
        ``[B, N, M]`` squared distances.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Indices of the ``k`` nearest points in ``xyz`` for every query in ``new_xyz``.

    The query set normally equals the support set, in which case every point is
    its own nearest neighbour (the k-NN graph has a self loop).

    Args:
        k: number of neighbours.
        xyz: ``[B, N, C]`` support points.
        new_xyz: ``[B, S, C]`` query points.

    Returns:
        ``[B, S, k]`` long tensor of neighbour indices (unsorted).
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: ``[B, N, C]``.
        idx: ``[B, S]`` or ``[B, S, K]`` long indices into the ``N`` axis.

    Returns:
        ``[B, S, C]`` or ``[B, S, K, C]`` gathered points.
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int, start_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Iterative farthest point sampling.

    Args:
        xyz: ``[B, N, 3]`` points.
        npoint: number of points to sample.
        start_idx: optional ``[B]`` long tensor with the first sampled index per
            cloud. When ``None`` a random start point is drawn (the behaviour of
            the original implementation).

    Returns:
        ``[B, npoint]`` long tensor of sampled indices.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    if start_idx is None:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    else:
        farthest = start_idx.to(device=device, dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_graph_channels_first(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN indices for channels-first features ``[B, C, N]`` (DGCNN convention).

    Uses the negative squared distance and ``topk`` exactly as the DGCNN
    reference implementation does, so the resulting graph (including the self
    loop) is identical.
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def gather_neighbours(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbour features for channels-first input.

    Pure PyTorch replacement for the OpenPoints CUDA ``grouping_operation``.

    Args:
        features: ``[B, C, N]``.
        idx: ``[B, N, K]`` long or int indices.

    Returns:
        ``[B, C, N, K]`` grouped features.
    """
    B, C, N = features.shape
    _, S, K = idx.shape
    idx = idx.long().reshape(B, 1, S * K).expand(-1, C, -1)
    out = torch.gather(features, 2, idx)
    return out.reshape(B, C, S, K)
