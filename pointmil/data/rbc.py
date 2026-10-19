"""3D red blood cell dataset (Simionato et al., 2021) as point clouds.

Expected layout under ``root``: ``ad/*.ad`` files (``x y z nx ny nz 0`` rows), built
from the public 3D masks with ``scripts/prepare_rbc.py`` (see ``mask_to_point_cloud``).
Fold files (``splits/rbc/fold_{k}.csv``) have ``Path``, ``Split`` and ``Label`` columns.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pointmil.data.base import PointCloudDataset, load_ad_file


class RBC(PointCloudDataset):
    classes = (
        "stomatocyte", "cell cluster", "keratocyte", "knizocyte", "multilobate cell",
        "acanthocyte", "spherocyte", "discocyte", "echinocyte",
    )
    num_classes = 9

    def __init__(self, root: str, split_file: str, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        self.split = split
        df = pd.read_csv(split_file)
        if split == "train":
            df = df[df["Split"].isin(["train", "validation"])]
        elif split in ("test", "val"):
            df = df[df["Split"] == "test"]
        elif split == "validation":
            df = df[df["Split"] == "validation"]
        elif split == "train_only":
            df = df[df["Split"] == "train"]
        elif split != "all":
            raise ValueError(f"unknown split '{split}'")
        self.items = []
        for path, label in zip(df["Path"].tolist(), df["Label"].tolist()):
            rel = path.replace("\\", "/")
            rel = "ad/" + rel.split("ad/")[-1] if "ad/" in rel else rel
            self.items.append({"path": str(self.root / rel), "name": Path(rel).stem, "label": int(label)})
        missing = [it["path"] for it in self.items if not os.path.exists(it["path"])]
        if missing:
            raise FileNotFoundError(f"{len(missing)} RBC files not found under {self.root}, e.g. {missing[0]}")

    def _load(self, item: dict) -> dict:
        return load_ad_file(item["path"])


def mask_to_point_cloud(mask) -> np.ndarray:
    """Point cloud of one 3D mask exactly as used for the paper: ``[V, 7]`` = ``x y z nx ny nz 0``.

    The points are the vertices of the marching-cubes surface of the mask
    (``skimage.measure.marching_cubes`` at its default iso-level, halfway between the
    smallest and largest voxel value, no smoothing) and the columns 3 to 5 are the vertex
    normals. Coordinates are in voxel units; the loader normalises each cloud to the unit
    ball and subsamples it to ``num_points``. Requires scikit-image (``pip install -e .[data]``).
    """
    from skimage import measure

    verts, _faces, normals, _values = measure.marching_cubes(np.asarray(mask).astype(np.float32))
    return np.concatenate([verts, normals, np.zeros((len(verts), 1), dtype=verts.dtype)], axis=1)


def write_ad_file(path: str | os.PathLike, points) -> None:
    """Write ``[V, 7]`` rows in the ``.ad`` text format (6 decimals, integer label column)."""
    np.savetxt(path, points, fmt=["%f"] * 6 + ["%d"])

