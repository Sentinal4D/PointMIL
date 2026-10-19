"""ModelNet40 from the 2048-point HDF5 release (``ply_data_{train,test}*.h5``)."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import numpy as np

from pointmil.data.base import PointCloudDataset


class ModelNet40(PointCloudDataset):
    """ModelNet40 from any folder of ``*train*.h5`` / ``*test*.h5`` files (official or repackaged release)."""

    num_classes = 40

    def __init__(self, root: str, split: str = "train", **kwargs):
        kwargs.setdefault("sampling", "first")
        kwargs.setdefault("normalise", False)  # the h5 clouds are already normalised
        super().__init__(**kwargs)
        self.root = Path(root)
        partition = "train" if split in ("train", "trainval") else "test"
        files = sorted(glob.glob(str(self.root / "**" / f"*{partition}*.h5"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no ModelNet40 h5 files for partition '{partition}' under {self.root}")
        data, labels = [], []
        for h5_name in files:
            with h5py.File(h5_name, "r") as f:
                data.append(f["data"][:].astype(np.float32))
                labels.append(f["label"][:].astype(np.int64).reshape(-1))
        self.data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        self.items = [{"index": i, "name": f"{partition}_{i}", "label": int(labels[i])} for i in range(len(labels))]
        names_file = os.path.join(self.root, "shape_names.txt")
        self.classes = tuple(open(names_file).read().split()) if os.path.exists(names_file) else tuple(str(i) for i in range(40))

    def _load(self, item: dict) -> dict:
        return {"pos": self.data[item["index"]]}
