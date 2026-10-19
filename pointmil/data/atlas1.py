"""ATLAS-1: 3D drug-treated cancer cells (released with the paper).

``source`` chooses the input under ``root``:

* ``points``: ``pointclouds_4096/<serial>.npy``, the 4096-point clouds used for the paper,
  subsampled to ``num_points`` (``random`` as in the original PointMIL loader, ``fps`` as in
  the OpenPoints loader);
* ``points1024``: ``pointclouds_1024/<serial>.npy``, a fixed farthest-point subsample;
* ``mesh``: the vertices of the public surface meshes ``data/<serial>.obj``.

The split file (``splits/atlas1/split0.csv``: ``serial,treatment,split``) gives the labels and
the train/val/test split of the ATLAS-1 experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pointmil.data.base import PointCloudDataset


def read_obj_vertices(path: str) -> np.ndarray:
    """Vertex coordinates ``[V, 3]`` of a Wavefront ``.obj`` file."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(verts, dtype=np.float32)


class ATLAS1(PointCloudDataset):
    """ATLAS-1 cells; ``split`` is ``train``, ``val``, ``test``, ``trainval`` or ``all``."""

    classes = ("Blebbistatin", "No Treatment", "Nocodazole")  # alphabetical, as the original label encoder
    num_classes = 3

    def __init__(self, root: str, split_file: str, split: str = "train", source: str = "points", **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        self.split = split
        self.source = source
        df = pd.read_csv(split_file)
        if split == "train":
            df = df[df["split"] == "train"]
        elif split in ("val", "validation"):
            df = df[df["split"] == "val"]
        elif split == "test":
            df = df[df["split"] == "test"]
        elif split == "trainval":
            df = df[df["split"].isin(["train", "val"])]
        elif split != "all":
            raise ValueError(f"unknown split '{split}'")
        label_of = {c: i for i, c in enumerate(self.classes)}
        self.items = []
        for serial, treatment in zip(df["serial"].tolist(), df["treatment"].tolist()):
            if source == "points":
                path = self.root / "pointclouds_4096" / f"{serial}.npy"
            elif source == "points1024":
                path = self.root / "pointclouds_1024" / f"{serial}.npy"
            elif source == "mesh":
                path = self.root / "data" / f"{serial}.obj"
            else:
                raise ValueError(f"unknown source '{source}'")
            self.items.append({"path": str(path), "name": serial, "label": label_of[treatment]})
        missing = [it["path"] for it in self.items if not os.path.exists(it["path"])]
        if missing:
            raise FileNotFoundError(f"{len(missing)} ATLAS-1 files not found under {self.root}, e.g. {missing[0]}")

    def _load(self, item: dict) -> dict:
        if item["path"].endswith(".npy"):
            pos = np.load(item["path"]).astype(np.float32)[:, :3]
        else:
            pos = read_obj_vertices(item["path"])
        return {"pos": pos}
