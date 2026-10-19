"""ScanObjectNN, hardest variant (PB_T50_RS), from the official ``h5_files/main_split``.

Training clouds: the first ``num_points`` of the 2048 stored points (the
OpenPoints fork default), followed by the configured transforms. Test clouds:
1024 points obtained by farthest point sampling once and cached next to the h5
file (``test_objectdataset_augmentedrot_scale75_1024_fps.pkl``), as in OpenPoints.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from pointmil.backbones.knn import farthest_point_sample, index_points
from pointmil.data.base import PointCloudDataset


class ScanObjectNN(PointCloudDataset):
    classes = ("bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet")
    num_classes = 15

    def __init__(self, root: str, split: str = "train", uniform_sample: bool = True, **kwargs):
        kwargs.setdefault("sampling", "first")
        kwargs.setdefault("normalise", False)  # handled by the center_and_normalize transform
        super().__init__(**kwargs)
        self.root = Path(root)
        name = "training" if split in ("train", "trainval") else "test"
        h5_name = self.root / f"{name}_objectdataset_augmentedrot_scale75.h5"
        if not h5_name.is_file():
            raise FileNotFoundError(f"{h5_name} does not exist")
        with h5py.File(h5_name, "r") as f:
            points = np.array(f["data"]).astype(np.float32)
            labels = np.array(f["label"]).astype(np.int64)
        if name == "test" and uniform_sample:
            cache = self.root / f"{name}_objectdataset_augmentedrot_scale75_{self.num_points}_fps.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    points = pickle.load(f)
            else:
                pts = torch.from_numpy(points)
                idx = farthest_point_sample(pts, self.num_points, start_idx=torch.zeros(pts.shape[0], dtype=torch.long))
                points = index_points(pts, idx).numpy()
                try:
                    with open(cache, "wb") as f:
                        pickle.dump(points, f)
                except OSError:
                    pass
        self.points = points
        self.items = [{"index": i, "name": f"{name}_{i}", "label": int(labels[i])} for i in range(len(labels))]

    def _load(self, item: dict) -> dict:
        return {"pos": self.points[item["index"]]}
