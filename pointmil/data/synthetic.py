"""Tiny synthetic dataset for smoke tests and CI (two classes of random shapes)."""

from __future__ import annotations

import numpy as np

from pointmil.data.base import PointCloudDataset


class Synthetic(PointCloudDataset):
    """Class 0: points on a sphere; class 1: points on an elongated ellipsoid with a bump.

    Per-point ``seg`` labels mark the bump (used to exercise NDCG@n).
    """

    classes = ("sphere", "ellipsoid_with_bump")
    num_classes = 2

    def __init__(self, root: str = "", split: str = "train", num_samples: int = 16, seed: int = 0, **kwargs):
        kwargs.setdefault("normalise", True)
        super().__init__(**kwargs)
        self.root = root
        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2, "annotated": 3}.get(split, 4))
        self.items = [{"index": i, "name": f"{split}_{i}", "label": int(i % 2)} for i in range(num_samples)]
        self._data = []
        for item in self.items:
            n = 1200
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            seg = np.zeros(n, dtype=np.int64)
            if item["label"] == 1:
                v[:, 0] *= 2.0
                bump = v[:, 2] > 0.8
                v[bump] *= 1.4
                seg[bump] = 1
            self._data.append({"pos": v.astype(np.float32), "normals": v.astype(np.float32), "seg": seg})

    def _load(self, item: dict) -> dict:
        return self._data[item["index"]]
