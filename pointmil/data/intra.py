"""IntrA intracranial aneurysm dataset (Yang et al., CVPR 2020).

Expected layout under ``root`` (the official release):

    annotated/ad/*.ad            116 annotated aneurysm segments (per-point labels)
    generated/vessel/ad/*.ad     1694 healthy vessel segments (class 0)
    generated/aneurysm/ad/*.ad   215 aneurysm segments (class 1)

Fold files (``splits/intra/fold_{k}.csv``) list relative paths with a ``Split``
column (train, validation, test). As in the original code the training set is
``train + validation`` and model selection used the ``test`` rows.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

from pointmil.data.base import PointCloudDataset, load_ad_file


class IntrA(PointCloudDataset):
    classes = ("healthy", "aneurysm")
    num_classes = 2

    def __init__(self, root: str, split_file: str | None, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        self.split = split
        if split == "annotated":
            paths = sorted(glob.glob(str(self.root / "annotated" / "ad" / "*.ad")))
            rel = [os.path.relpath(p, self.root) for p in paths]
        else:
            if split_file is None:
                raise ValueError("split_file is required unless split='annotated'")
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
            rel = [self._relative(p) for p in df["Path"].tolist()]
        self.items = [{"path": str(self.root / r), "name": Path(r).stem, "label": self._label(r)} for r in rel]
        missing = [it["path"] for it in self.items if not os.path.exists(it["path"])]
        if missing:
            raise FileNotFoundError(f"{len(missing)} IntrA files not found under {self.root}, e.g. {missing[0]}")

    @staticmethod
    def _relative(path: str) -> str:
        path = path.replace("\\", "/")
        if "IntrA/" in path:
            path = path.split("IntrA/")[-1]
        return path.lstrip("/")

    @staticmethod
    def _label(rel: str) -> int:
        return 0 if rel.startswith("generated/vessel/") else 1

    def _load(self, item: dict) -> dict:
        return load_ad_file(item["path"])
