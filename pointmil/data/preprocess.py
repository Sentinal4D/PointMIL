"""Preprocess a single point cloud for inference.

``prepare_cloud`` reproduces the test-time processing of the IntrA, RBC and ATLAS-1 loaders
(subsample to ``num_points``, unit-ball normalisation of the coordinates). ModelNet40 and
ScanObjectNN models expect the first ``num_points`` points without re-normalisation, and the
ScanObjectNN models an extra height channel (``pointmil.data.transforms.PointCloudCenterAndNormalize``).
"""

from __future__ import annotations

import numpy as np
import torch

from pointmil.backbones.knn import farthest_point_sample
from pointmil.data.transforms import unit_ball_normalise


def prepare_cloud(points: np.ndarray, num_points: int = 1024, with_normals: bool | None = None, sampling: str = "random", seed: int = 0, normalise: bool = True) -> torch.Tensor:
    """Turn an ``[M, 3]`` or ``[M, 6]`` array into a model input ``[1, num_points, C]``.

    Args:
        points: xyz (and optionally normals) per point.
        num_points: number of points to keep (sampled with replacement when ``M`` is smaller).
        with_normals: keep the normal columns (default: keep them if present).
        sampling: ``random`` (as in training) or ``fps`` (deterministic farthest point sampling from point 0).
        seed: seed of the random subsampling.
        normalise: unit-ball normalisation of the coordinates after sampling.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be [M, 3] or [M, 6]")
    if with_normals is None:
        with_normals = pts.shape[1] >= 6
    elif with_normals and pts.shape[1] < 6:
        raise ValueError("with_normals=True needs [M, 6] input (xyz + normals)")
    pts = pts[:, :6] if with_normals else pts[:, :3]
    m = pts.shape[0]
    if sampling == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(m, num_points, replace=m < num_points)
    elif sampling == "fps":
        if m <= num_points:
            idx = np.resize(np.arange(m), num_points)
        else:
            idx = farthest_point_sample(torch.from_numpy(pts[:, :3]).unsqueeze(0), num_points, start_idx=torch.zeros(1, dtype=torch.long))[0].numpy()
    else:
        raise ValueError("sampling must be 'random' or 'fps'")
    pts = pts[idx]
    if normalise:
        pts[:, :3] = unit_ball_normalise(pts[:, :3])
    return torch.from_numpy(np.ascontiguousarray(pts)).unsqueeze(0)
