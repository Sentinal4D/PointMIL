"""Common dataset machinery.

Every dataset yields a dict with ``pos`` ``[N, 3]`` (float32), optionally
``normals`` ``[N, 3]``, ``seg`` ``[N]`` (per-point labels) and ``heights`` ``[N, 1]``,
the class label ``y``, the item ``index`` and ``name``, and the model input ``x``
``[N, C_in]`` assembled from ``feature_keys``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from pointmil.backbones.knn import farthest_point_sample
from pointmil.data.transforms import Compose, random_subsample, unit_ball_normalise


class PointCloudDataset(Dataset):
    """Base class handling subsampling, normalisation, transforms and feature assembly.

    Args:
        num_points: points per cloud fed to the model.
        sampling: ``random`` (random subset each time, with replacement when the
            cloud is smaller), ``fps`` (farthest point sampling from the first
            point), ``first`` (the first ``num_points`` rows) or ``none``.
        normalise: unit-ball normalisation of the coordinates after sampling.
        transforms: ``Compose`` applied after normalisation (train augmentation
            or OpenPoints-style preprocessing).
        feature_keys: which per-point arrays are concatenated into ``x``.
        cache: keep parsed files in memory.
        fps_random_start: start farthest point sampling at a random point (training) instead of
            point 0 (evaluation, deterministic).
    """

    classes: Sequence[str] = ()

    def __init__(
        self,
        num_points: int = 1024,
        sampling: str = "random",
        normalise: bool = True,
        transforms: Compose | None = None,
        feature_keys: Sequence[str] = ("pos",),
        cache: bool = True,
        fps_random_start: bool = False,
    ):
        if sampling not in {"random", "fps", "first", "none"}:
            raise ValueError(f"unknown sampling '{sampling}'")
        self.num_points = num_points
        self.sampling = sampling
        self.normalise = normalise
        self.transforms = transforms or Compose([])
        self.feature_keys = list(feature_keys)
        self.cache = cache
        self.fps_random_start = fps_random_start  # True in training: random seed point as in the original loader
        self._cache: dict = {}
        self.items: list[dict] = []

    # ------------------------------------------------------------------ to implement
    def _load(self, item: dict) -> dict:
        """Return ``{"pos": [M, 3] float32, "normals": optional, "seg": optional}``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def _get_raw(self, index: int) -> dict:
        if self.cache and index in self._cache:
            return self._cache[index]
        raw = self._load(self.items[index])
        if self.cache:
            self._cache[index] = raw
        return raw

    def _subsample(self, raw: dict, rng: np.random.Generator) -> dict:
        n = raw["pos"].shape[0]
        if self.sampling == "none" or (self.sampling == "first" and n <= self.num_points):
            idx = np.arange(n) if self.sampling == "none" else np.arange(min(n, self.num_points))
        elif self.sampling == "first":
            idx = np.arange(self.num_points)
        elif self.sampling == "random":
            idx = random_subsample(raw["pos"], self.num_points, rng)
        else:  # fps
            if n <= self.num_points:
                idx = rng.choice(n, self.num_points, replace=True) if n < self.num_points else np.arange(n)
            else:
                pts = torch.from_numpy(np.ascontiguousarray(raw["pos"])).unsqueeze(0)
                start = int(rng.integers(0, n)) if self.fps_random_start else 0
                idx = farthest_point_sample(pts, self.num_points, start_idx=torch.tensor([start]))[0].numpy()
        out = {"pos": raw["pos"][idx].astype(np.float32)}
        for key in ("normals", "seg"):
            if raw.get(key) is not None:
                out[key] = raw[key][idx]
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        data = self._subsample(self._get_raw(index), rng)
        if self.normalise:
            data["pos"] = unit_ball_normalise(data["pos"]).astype(np.float32)
        data = self.transforms(data, rng)
        for key in ("pos", "normals", "heights"):
            if key in data and isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(np.ascontiguousarray(data[key])).float()
        if "seg" in data:
            data["seg"] = torch.as_tensor(np.asarray(data["seg"]), dtype=torch.long)
        feats = []
        for key in self.feature_keys:
            if key not in data:
                raise KeyError(f"feature key '{key}' not produced by {type(self).__name__} (available: {sorted(data)})")
            feats.append(data[key].float())
        data["x"] = torch.cat(feats, dim=1)
        data["y"] = torch.tensor(int(self.items[index]["label"]), dtype=torch.long)
        data["index"] = index
        data["name"] = str(self.items[index].get("name", index))
        return data

    @property
    def labels(self) -> np.ndarray:
        return np.array([it["label"] for it in self.items], dtype=np.int64)


def load_ad_file(path: str) -> dict:
    """Read an IntrA-style ``.ad`` text file: ``x y z nx ny nz label`` per row."""
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    seg = arr[:, -1].astype(np.int64)
    seg[seg == 2] = 1  # boundary points count as aneurysm, as in the original code
    return {"pos": arr[:, :3].copy(), "normals": arr[:, 3:6].copy(), "seg": seg}
