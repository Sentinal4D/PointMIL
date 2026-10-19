#!/usr/bin/env python
"""Export the ATLAS-1 point clouds used for the paper under their public serial names.

Authors-only provenance script. It reads the internal 4096-point ``.ply`` clouds
(``<internal>/Plate{n}/stacked_pointcloud_4096/<Treatment>/<old_serial>.ply``),
renames them with ``ATLAS-1_serial_number_mapping.csv`` and writes

    <out>/pointclouds_4096/<serial>.npy   float32 [4096, 3] (the training inputs)
    <out>/pointclouds_1024/<serial>.npy   float32 [1024, 3] (farthest point sampling from point 0, unit ball)

The 1024-point version is a deterministic subsample for users who want fixed
inputs; training in the paper subsampled the 4096 points on the fly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointmil.backbones.knn import farthest_point_sample  # noqa: E402
from pointmil.data.transforms import unit_ball_normalise  # noqa: E402


def read_ply_xyz(path: Path) -> np.ndarray:
    try:
        from pyntcloud import PyntCloud

        return PyntCloud.from_file(str(path)).points[["x", "y", "z"]].values.astype(np.float32)
    except ImportError:
        import trimesh

        return np.asarray(trimesh.load(str(path)).vertices, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internal", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="the original fold0.csv (Treatment, serialNumber, PlateNumber, Splits)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    mapping = pd.read_csv(args.mapping)
    split = pd.read_csv(args.split)
    plate_of = dict(zip(split["serialNumber"], split["PlateNumber"]))
    treat_of = dict(zip(split["serialNumber"], split["Treatment"]))
    (args.out / "pointclouds_4096").mkdir(parents=True, exist_ok=True)
    (args.out / "pointclouds_1024").mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for new, old in zip(mapping["new_serial"], mapping["old_serial"]):
        plate, treat = plate_of.get(old), treat_of.get(old)
        if plate is None:
            print("no split entry for", old)
            continue
        src = args.internal / f"Plate{plate}" / "stacked_pointcloud_4096" / treat / f"{old}.ply"
        pts = read_ply_xyz(src)
        np.save(args.out / "pointclouds_4096" / f"{new}.npy", pts)
        idx = farthest_point_sample(torch.from_numpy(pts).unsqueeze(0), 1024, start_idx=torch.zeros(1, dtype=torch.long))[0].numpy()
        np.save(args.out / "pointclouds_1024" / f"{new}.npy", unit_ball_normalise(pts[idx]).astype(np.float32))
        n_ok += 1
    print(f"wrote {n_ok} clouds to {args.out}")


if __name__ == "__main__":
    main()
