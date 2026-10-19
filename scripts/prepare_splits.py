#!/usr/bin/env python
"""Write the committed split files from the original fold CSVs and the ATLAS-1 serial mapping.

This is a provenance script: it was run once by the authors. The outputs are
committed under ``splits/``. Paths in the original CSVs were absolute; they are
made relative to the dataset roots here.

Usage (authors only)::

    python scripts/prepare_splits.py --intra-folds <dir> --rbc-folds <dir> \
        --atlas1-split <fold0.csv> --atlas1-mapping <mapping.csv> --atlas1-metadata <metadata.csv> --out splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_intra(folds_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        df = pd.read_csv(folds_dir / f"fold_{k}.csv")
        df["Path"] = df["Path"].str.replace("\\", "/", regex=False).str.split("IntrA/").str[-1]
        df[["Path", "Split", "Fold"]].to_csv(out_dir / f"fold_{k}.csv", index=False)
        print("intra", k, df["Split"].value_counts().to_dict())


def convert_rbc(folds_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        df = pd.read_csv(folds_dir / f"fold_{k}.csv")
        df["Path"] = "ad/" + df["Path"].str.replace("\\", "/", regex=False).str.split("ad/").str[-1]
        df[["Path", "Split", "Fold", "Label"]].to_csv(out_dir / f"fold_{k}.csv", index=False)
        print("rbc", k, df["Split"].value_counts().to_dict())


def convert_atlas1(split_csv: Path, mapping_csv: Path, metadata_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(split_csv)
    df = df[df["Treatment"].isin(["No Treatment", "Nocodazole", "Blebbistatin"])]
    mapping = pd.read_csv(mapping_csv)
    meta = pd.read_csv(metadata_csv)
    merged = df.merge(mapping, left_on="serialNumber", right_on="old_serial", how="left")
    unmapped = merged[merged["new_serial"].isna()]
    merged = merged.dropna(subset=["new_serial"]).merge(meta, left_on="new_serial", right_on=meta.columns[0], how="inner")
    assert (merged["Treatment"] == merged["treatment"]).all(), "treatment mismatch between split file and metadata"
    out = pd.DataFrame({
        "serial": merged["new_serial"],
        "treatment": merged["treatment"],
        "split": merged["Splits"].map({"train": "train", "val": "val", "test": "test"}),
        "plate": merged["PlateNumber"],
    })
    # One original cell maps to two public serials (a duplicate in the public release); keep one row per public serial.
    dup_old = mapping[mapping["old_serial"].duplicated(keep=False)]
    out = out.drop_duplicates(subset=["serial"]).sort_values(["split", "treatment", "serial"])
    out.to_csv(out_dir / "split0.csv", index=False)
    print("atlas1", out.groupby(["treatment", "split"]).size().unstack(fill_value=0).to_dict())
    with open(out_dir / "README.md", "w") as f:
        counts = out["split"].value_counts()
        f.write("# ATLAS-1 split\n\n`split0.csv` is the train/validation/test split used for the ATLAS-1 experiments.\n\n"
                f"Cells: {len(out)} in total, {counts.get('train', 0)} train, {counts.get('val', 0)} val and {counts.get('test', 0)} test.")
        if len(unmapped):
            f.write(f" {unmapped['serialNumber'].nunique()} cell(s) of the original split have no public serial number and are not part of the release.")
        f.write("\n\n")
        if len(dup_old):
            pairs = dup_old.groupby("old_serial")["new_serial"].apply(lambda s: " and ".join(sorted(s)))
            f.write("Note: the following public serials are the same cell released twice under two names: "
                    + "; ".join(pairs.tolist()) + ". Both are kept in the training split.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intra-folds", type=Path)
    ap.add_argument("--rbc-folds", type=Path)
    ap.add_argument("--atlas1-split", type=Path)
    ap.add_argument("--atlas1-mapping", type=Path)
    ap.add_argument("--atlas1-metadata", type=Path)
    ap.add_argument("--out", type=Path, default=Path("splits"))
    args = ap.parse_args()
    if args.intra_folds:
        convert_intra(args.intra_folds, args.out / "intra")
    if args.rbc_folds:
        convert_rbc(args.rbc_folds, args.out / "rbc")
    if args.atlas1_split:
        convert_atlas1(args.atlas1_split, args.atlas1_mapping, args.atlas1_metadata, args.out / "atlas1")


if __name__ == "__main__":
    main()
