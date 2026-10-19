"""Build datasets and data loaders from the ``data`` section of a config.

``data`` keys: ``name``, ``root``, ``num_points``, ``batch_size``, ``num_workers``,
``feature_keys``, ``sampling``, ``normalise``, ``train_transforms``,
``test_transforms``, ``split_dir`` (fold files), ``split_file`` (fixed split),
plus dataset-specific keys (``source`` for ATLAS-1, ``uniform_sample`` for ScanObjectNN).
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pointmil.data.transforms import build_transforms
from pointmil.utils.seed import seed_worker

FOLD_DATASETS = {"intra", "rbc"}


def _split_file(data_cfg: dict, fold: int | None) -> str | None:
    if data_cfg.get("split_file"):
        return data_cfg["split_file"]
    if data_cfg.get("split_dir"):
        if fold is None:
            return None  # the dataset raises if the split actually needs a fold file
        return f"{data_cfg['split_dir']}/fold_{fold}.csv"
    return None


def build_dataset(data_cfg: dict, split: str, train: bool, fold: int | None = None):
    """Instantiate the dataset named in ``data_cfg`` for ``split``."""
    from pointmil.data import DATASETS

    cfg = copy.deepcopy(data_cfg)
    name = cfg.pop("name")
    if name not in DATASETS and ":" in name:  # your own dataset class, e.g. "mypackage.cells:MyCells"
        module_name, _, class_name = name.partition(":")
        DATASETS[name] = getattr(importlib.import_module(module_name), class_name)
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)} (or 'module.path:ClassName')")
    transforms = build_transforms(cfg.pop("train_transforms", None) if train else cfg.pop("test_transforms", None))
    cfg.pop("train_transforms", None), cfg.pop("test_transforms", None)
    common: dict[str, Any] = dict(
        num_points=cfg.pop("num_points"),
        feature_keys=cfg.pop("feature_keys", ["pos"]),
        sampling=cfg.pop("sampling", "random"),
        transforms=transforms,
        cache=cfg.pop("cache", True),
        fps_random_start=train,
    )
    if "normalise" in cfg:
        common["normalise"] = cfg.pop("normalise")
    for key in ("batch_size", "num_workers", "split_dir", "split_file", "val_batch_size"):
        cfg.pop(key, None)
    root = cfg.pop("root")
    split_file = _split_file(data_cfg, fold)
    cls = DATASETS[name]
    if name in FOLD_DATASETS or name == "atlas1":
        if split_file is None and not (name == "intra" and split == "annotated"):
            need = "--fold (fold files in data.split_dir)" if name in FOLD_DATASETS else "data.split_file"
            raise ValueError(f"dataset '{name}' needs {need}")
        return cls(root=root, split_file=split_file, split=split, **common, **cfg)
    if split_file is not None:  # a user dataset with a split file (data.split_file or data.split_dir + --fold)
        return cls(root=root, split_file=split_file, split=split, **common, **cfg)
    return cls(root=root, split=split, **common, **cfg)


def build_dataloaders(cfg: dict, fold: int | None, seed: int) -> dict[str, DataLoader]:
    """Train, validation and test loaders according to ``cfg['data']`` and ``cfg['train']['val_split']``.

    ``val_split`` is one of:

    * ``test``: model selection on the test split (the convention of the paper runs for
      IntrA, RBC, ModelNet40 and ScanObjectNN);
    * ``validation``: the validation rows of the split file (ATLAS-1 in the paper; for IntrA and
      RBC the model is then trained on the ``train`` rows only, so validation rows are not seen);
    * ``holdout``: a stratified ``val_fraction`` of the training items is held out.
    """
    data_cfg = cfg["data"]
    val_split = cfg["train"]["val_split"]
    g = torch.Generator()
    g.manual_seed(seed)
    # IntrA and RBC: "train" = train + validation rows of the fold file (as in the paper runs)
    train_split = "train_only" if val_split == "validation" and data_cfg["name"] in FOLD_DATASETS else "train"
    train_ds = build_dataset(data_cfg, train_split, train=True, fold=fold)
    test_ds = build_dataset(data_cfg, "test", train=False, fold=fold)
    if val_split == "test":
        val_ds = test_ds
    elif val_split == "holdout":
        frac = float(cfg["train"].get("val_fraction", 0.1))
        rng = np.random.default_rng(seed)
        labels = train_ds.labels
        val_idx = []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            val_idx.extend(idx[: max(1, int(round(frac * len(idx))))].tolist())
        train_idx = sorted(set(range(len(train_ds))) - set(val_idx))
        val_eval = build_dataset(data_cfg, train_split, train=False, fold=fold)  # no augmentation
        val_ds = Subset(val_eval, sorted(val_idx))
        train_ds = Subset(train_ds, train_idx)
    elif val_split == "validation":
        val_ds = build_dataset(data_cfg, "validation", train=False, fold=fold)
    else:
        raise ValueError(f"unknown val_split '{val_split}'")
    bs = int(data_cfg["batch_size"])
    val_bs = int(data_cfg.get("val_batch_size", bs))
    nw = int(data_cfg["num_workers"])
    drop_last = bool(cfg["train"].get("drop_last", True))
    loaders = {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=drop_last, num_workers=nw, worker_init_fn=seed_worker, generator=g, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=val_bs, shuffle=False, drop_last=False, num_workers=nw, worker_init_fn=seed_worker, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=val_bs, shuffle=False, drop_last=False, num_workers=nw, worker_init_fn=seed_worker, pin_memory=True),
    }
    return loaders
