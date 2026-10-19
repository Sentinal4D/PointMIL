"""Point cloud preprocessing and augmentation.

Two families are provided because the paper used two training code bases:

* NumPy functions applied inside the datasets of the original PointMIL code
  (random subsampling, unit-ball normalisation, anisotropic scale and shift).
* Torch transforms on ``{"pos": ..., ...}`` dictionaries reproducing the
  OpenPoints transforms used for ATLAS-1 and ScanObjectNN (scale and translate,
  scaling, centring and normalisation with height channel, random rotation).

Transforms are configured as a list of ``{name: ..., <kwargs>}`` entries and
composed with :func:`build_transforms`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- numpy helpers (original code path)
def random_subsample(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Random subset of ``num_points`` rows (with replacement when there are fewer points)."""
    n = points.shape[0]
    idx = rng.choice(n, num_points, replace=n < num_points)
    return idx


def unit_ball_normalise(xyz: np.ndarray) -> np.ndarray:
    """Centre on the mean and scale so that the farthest point lies on the unit sphere."""
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    return xyz / dist


class RandomScaleTranslateNumpy:
    """Anisotropic scaling followed by a random shift on xyz (original IntrA and RBC pipelines)."""

    def __init__(self, scale: Sequence[float] = (0.8, 1.2), shift: float = 0.2):
        self.scale = scale
        self.shift = shift

    def __call__(self, data: dict, rng: np.random.Generator) -> dict:
        scale = rng.uniform(self.scale[0], self.scale[1], size=[3]).astype(np.float32)
        shift = rng.uniform(-self.shift, self.shift, size=[3]).astype(np.float32)
        data["pos"] = (data["pos"] * scale + shift).astype(np.float32)
        return data


class DGCNNScaleTranslateNumpy:
    """DGCNN-style augmentation used for ModelNet40: scale U(2/3, 3/2) and shift U(-0.2, 0.2)."""

    def __init__(self, scale: Sequence[float] = (2.0 / 3.0, 3.0 / 2.0), shift: float = 0.2):
        self.scale = scale
        self.shift = shift

    def __call__(self, data: dict, rng: np.random.Generator) -> dict:
        xyz1 = rng.uniform(self.scale[0], self.scale[1], size=[3])
        xyz2 = rng.uniform(-self.shift, self.shift, size=[3])
        data["pos"] = np.add(np.multiply(data["pos"], xyz1), xyz2).astype(np.float32)
        return data


class ShufflePointsNumpy:
    """Random permutation of the points (applied to all per-point arrays)."""

    def __call__(self, data: dict, rng: np.random.Generator) -> dict:
        n = data["pos"].shape[0]
        perm = rng.permutation(n)
        for key in ("pos", "normals", "seg", "heights"):
            if key in data and data[key] is not None:
                data[key] = data[key][perm]
        return data


# --------------------------------------------------------------------------- torch transforms (OpenPoints code path)
class PointsToTensor:
    """Convert every numpy array in the sample to a float tensor (start of the OpenPoints transforms)."""

    def __call__(self, data: dict, rng=None) -> dict:
        for key in list(data.keys()):
            v = data[key]
            if isinstance(v, np.ndarray) and key != "y":
                data[key] = torch.from_numpy(v)
        return data


class PointCloudScaleAndTranslate:
    """OpenPoints ``PointCloudScaleAndTranslate``: anisotropic scale and shift (torch RNG)."""

    def __init__(self, scale: Sequence[float] = (2.0 / 3.0, 3.0 / 2.0), shift: Sequence[float] = (0.2, 0.2, 0.2), anisotropic: bool = True):
        self.scale_min, self.scale_max = float(scale[0]), float(scale[1])
        self.shift = torch.tensor(shift, dtype=torch.float32)
        self.anisotropic = anisotropic

    def __call__(self, data: dict, rng=None) -> dict:
        pos = data["pos"]
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32) * (self.scale_max - self.scale_min) + self.scale_min
        translation = (torch.rand(3, dtype=torch.float32) - 0.5) * 2 * self.shift
        data["pos"] = pos * scale + translation
        return data


class PointCloudScaling:
    """OpenPoints ``PointCloudScaling``."""

    def __init__(self, scale: Sequence[float] = (2.0 / 3.0, 3.0 / 2.0), anisotropic: bool = True):
        self.scale_min, self.scale_max = float(scale[0]), float(scale[1])
        self.anisotropic = anisotropic

    def __call__(self, data: dict, rng=None) -> dict:
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32) * (self.scale_max - self.scale_min) + self.scale_min
        data["pos"] = data["pos"] * scale
        return data


class PointCloudCenterAndNormalize:
    """OpenPoints ``PointCloudCenterAndNormalize``: height channel, centring, unit-sphere scaling."""

    def __init__(self, centering: bool = True, normalize: bool = True, gravity_dim: int = 2):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim

    def __call__(self, data: dict, rng=None) -> dict:
        pos = data["pos"]
        height = pos[:, self.gravity_dim : self.gravity_dim + 1]
        data["heights"] = height - torch.min(height)
        if self.centering:
            pos = pos - torch.mean(pos, dim=0, keepdim=True)
        if self.normalize:
            m = torch.max(torch.sqrt(torch.sum(pos**2, dim=-1, keepdim=True)), dim=0, keepdim=True)[0]
            pos = pos / m
        data["pos"] = pos
        return data


def _axis_rotation(axis: int, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class PointCloudRotation:
    """OpenPoints ``PointCloudRotation``: random rotation about each axis by up to ``angle[i] * pi``, random order."""

    def __init__(self, angle: Sequence[float] = (0.0, 1.0, 0.0)):
        self.angle = np.array(angle, dtype=np.float64) * np.pi

    def __call__(self, data: dict, rng=None) -> dict:
        mats = []
        for axis, bound in enumerate(self.angle):
            theta = np.random.uniform(-bound, bound) if bound else 0.0
            mats.append(_axis_rotation(axis, theta))
        np.random.shuffle(mats)
        rot = torch.tensor(mats[0] @ mats[1] @ mats[2], dtype=torch.float32)
        data["pos"] = data["pos"] @ rot.T
        if "normals" in data and data["normals"] is not None:
            data["normals"] = data["normals"] @ rot.T
        return data


TRANSFORMS = {
    "random_scale_translate": RandomScaleTranslateNumpy,
    "dgcnn_scale_translate": DGCNNScaleTranslateNumpy,
    "shuffle_points": ShufflePointsNumpy,
    "points_to_tensor": PointsToTensor,
    "scale_and_translate": PointCloudScaleAndTranslate,
    "scaling": PointCloudScaling,
    "center_and_normalize": PointCloudCenterAndNormalize,
    "rotation": PointCloudRotation,
}


class Compose:
    """Apply transforms in order; each takes and returns the sample dict."""

    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, data: dict, rng: np.random.Generator | None = None) -> dict:
        for t in self.transforms:
            data = t(data, rng)
        return data

    def __repr__(self) -> str:
        return "Compose(" + ", ".join(type(t).__name__ for t in self.transforms) + ")"


def build_transforms(specs: Sequence[dict[str, Any]] | None) -> Compose:
    """Build a ``Compose`` from config entries like ``[{name: scaling, scale: [0.9, 1.1]}]``."""
    transforms = []
    for spec in specs or []:
        spec = dict(spec)
        name = spec.pop("name")
        if name not in TRANSFORMS:
            raise KeyError(f"Unknown transform '{name}'. Available: {sorted(TRANSFORMS)}")
        transforms.append(TRANSFORMS[name](**spec))
    return Compose(transforms)
