#!/usr/bin/env python
"""Build the 825 red blood cell point clouds used in the paper from the public 3D masks.

The cells were imaged and classified by Simionato et al. (PLOS Comput. Biol. 17(5), 2021,
raw data: https://doi.org/10.5281/zenodo.4670205). Waibel et al. thresholded them into
64 x 64 x 64 masks and released them on Zenodo under CC BY 4.0
(https://doi.org/10.5281/zenodo.7031924, folder ``dataset_for_3D_reconstruction/obj``).
PointMIL does not redistribute the cells; this script rebuilds the point clouds locally.

The ``.ad`` files used for the paper were obtained ready-made. They are the vertex sets (with
vertex normals) of the marching-cubes surfaces of these masks, so this script recreates them with
``pointmil.data.rbc.mask_to_point_cloud`` and writes ``x y z nx ny nz 0`` rows to
``<root>/RBC/ad/<name>.tif.ad``. The output is byte-identical to the files used for the paper
(checked on all 825 cells with scikit-image 0.23).

Mask sources, in order of preference:
  --obj_dir DIR   an extracted ``dataset_for_3D_reconstruction/obj`` folder
  --zip FILE      the downloaded ``Datasets_for_3D_reconstruction.zip`` (3.6 GB)
  (default)       read only the 825 masks from the Zenodo zip with HTTP range requests
                  (no need to download the whole archive)

Usage:
  pip install -e .[data]
  python scripts/prepare_rbc.py --root $POINTMIL_DATA
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointmil.data.rbc import mask_to_point_cloud, write_ad_file  # noqa: E402

ZENODO_ZIP = "https://zenodo.org/records/7031924/files/Datasets_for_3D_reconstruction.zip?download=1"
OBJ_PREFIX = "dataset_for_3D_reconstruction/obj/"
SPLIT_DIR = Path(__file__).resolve().parents[1] / "splits" / "rbc"


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP range requests (enough for ``zipfile``)."""

    def __init__(self, url: str, retries: int = 5):
        self.url, self.pos, self.retries = url, 0, retries
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def wanted_names() -> set[str]:
    """Mask file names referenced by the committed fold files (``<name>.tif``)."""
    import pandas as pd

    df = pd.read_csv(SPLIT_DIR / "fold_0.csv")
    return {Path(p).name[: -len(".ad")] for p in df["Path"]}


def iter_masks(args, names: set[str]):
    """Yield ``(file name, mask array)`` for every wanted mask from the chosen source."""
    import tifffile

    if args.obj_dir:
        for name in sorted(names):
            yield name, tifffile.imread(Path(args.obj_dir) / name)
        return
    if args.zip:
        handle = open(args.zip, "rb")
    else:
        print(f"reading the masks from {ZENODO_ZIP.split('?')[0]} (range requests, no full download)")
        handle = io.BufferedReader(HTTPRangeFile(ZENODO_ZIP), buffer_size=1 << 20)
    with zipfile.ZipFile(handle) as z:
        members = {Path(m).name: m for m in z.namelist() if m.startswith(OBJ_PREFIX) and m.endswith(".tif")}
        missing = sorted(names - set(members))
        if missing:
            raise SystemExit(f"{len(missing)} masks not found in the zip, e.g. {missing[:3]}")
        for name in sorted(names):
            yield name, tifffile.imread(io.BytesIO(z.read(members[name])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("POINTMIL_DATA", "data"), help="dataset root; files go to <root>/RBC/ad/")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--obj_dir", help="extracted dataset_for_3D_reconstruction/obj folder")
    src.add_argument("--zip", help="downloaded Datasets_for_3D_reconstruction.zip")
    ap.add_argument("--limit", type=int, default=None, help="only build the first N clouds (for a quick test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.root) / "RBC" / "ad"
    out.mkdir(parents=True, exist_ok=True)
    names = wanted_names()
    if args.limit:
        names = set(sorted(names)[: args.limit])
    if not args.overwrite:
        names = {n for n in names if not (out / f"{n}.ad").exists()}
    print(f"building {len(names)} point clouds into {out}")
    for i, (name, mask) in enumerate(iter_masks(args, names), 1):
        write_ad_file(out / f"{name}.ad", mask_to_point_cloud(mask))
        if i % 100 == 0 or i == len(names):
            print(f"  {i}/{len(names)}")
    print("done. Splits: splits/rbc/fold_{0..4}.csv; set POINTMIL_DATA to", Path(args.root).resolve())


if __name__ == "__main__":
    main()
