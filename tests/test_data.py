"""Transforms and dataset builder tests on the synthetic dataset."""

import numpy as np
import pytest
import torch

from pointmil.data import build_dataset
from pointmil.data.transforms import build_transforms, unit_ball_normalise


def test_unit_ball():
    pts = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32) * 5 + 3
    out = unit_ball_normalise(pts)
    assert abs(np.linalg.norm(out, axis=1).max() - 1.0) < 1e-5 and np.allclose(out.mean(0), 0, atol=1e-5)


def test_openpoints_transforms_add_heights_and_rotate():
    t = build_transforms([{"name": "points_to_tensor"}, {"name": "center_and_normalize", "gravity_dim": 1}, {"name": "rotation", "angle": [0, 1, 0]}])
    data = t({"pos": np.random.default_rng(0).normal(size=(30, 3)).astype(np.float32)}, None)
    assert data["heights"].shape == (30, 1) and torch.all(data["heights"] >= 0)
    assert torch.allclose(data["pos"].norm(dim=1).max(), torch.tensor(1.0), atol=1e-5)


def test_synthetic_dataset_batches():
    cfg = {"name": "synthetic", "root": "", "num_samples": 6, "num_points": 64, "sampling": "random", "feature_keys": ["pos", "normals"], "batch_size": 2, "num_workers": 0,
           "train_transforms": [{"name": "random_scale_translate"}], "test_transforms": []}
    ds = build_dataset(cfg, "train", train=True)
    item = ds[0]
    assert item["x"].shape == (64, 6) and item["seg"].shape == (64,) and item["y"].dtype == torch.long
    cfg_fps = dict(cfg, sampling="fps")
    item = build_dataset(cfg_fps, "test", train=False)[1]
    assert item["x"].shape == (64, 6) and item["seg"].sum() > 0


def test_rbc_mask_to_point_cloud_round_trip(tmp_path):
    pytest.importorskip("skimage")
    from pointmil.data.base import load_ad_file
    from pointmil.data.rbc import mask_to_point_cloud, write_ad_file

    grid = np.indices((32, 32, 32)).transpose(1, 2, 3, 0) - 15.5
    mask = ((grid ** 2).sum(-1) < 10.0 ** 2).astype(np.uint8) * 255
    pts = mask_to_point_cloud(mask)
    assert pts.shape[1] == 7 and len(pts) > 500
    radius = np.linalg.norm(pts[:, :3] - 15.5, axis=1)
    assert abs(radius.mean() - 10.0) < 0.5
    assert np.allclose(np.linalg.norm(pts[:, 3:6], axis=1), 1.0, atol=1e-3)
    path = tmp_path / "cell.tif.ad"
    write_ad_file(path, pts)
    first = path.read_text().splitlines()[0].split(" ")
    assert len(first) == 7 and first[-1] == "0"
    loaded = load_ad_file(str(path))
    assert loaded["pos"].shape == (len(pts), 3)


def test_dataset_by_import_path():
    cfg = {"name": "pointmil.data.synthetic:Synthetic", "root": "", "num_samples": 4, "num_points": 32, "sampling": "random",
           "feature_keys": ["pos"], "batch_size": 2, "num_workers": 0, "train_transforms": [], "test_transforms": []}
    ds = build_dataset(cfg, "train", train=True)
    assert len(ds) == 4 and ds[0]["x"].shape == (32, 3)
