#!/usr/bin/env python
"""Aggregate results into paper-style tables (mean and standard deviation over folds).

Two modes::

    python scripts/aggregate_results.py runs outputs/table1_intra_transformer_conjunctive --out results/table.md
    python scripts/aggregate_results.py legacy results/legacy --out docs/reproducibility_tables.md

``runs`` reads the ``summary.json`` files written by ``scripts/train.py`` (one per fold
and seed) and, where present, ``interpretability.json`` (``scripts/interpret.py metrics --out``);
``legacy`` reads the JSON files of the authors' ``scripts/evaluate_legacy.py``.
"Max over epochs" is the largest value of the validation metric during training, which is the
paper's convention when the validation split is the test split.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def fmt(values, digits=1):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} +/- {np.std(values, ddof=1):.{digits}f}"


def aggregate_runs(dirs: list[str]) -> str:
    lines = ["| experiment | folds | test oACC (best ckpt) | test mACC | test F1 | max over epochs: mACC | max over epochs: F1 | test F1 (last ckpt) | AOPCR | NDCG@n |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for d in dirs:
        summaries = sorted(glob.glob(os.path.join(d, "*", "summary.json")))
        if not summaries:
            continue
        rows = [json.load(open(s))["summary"] for s in summaries]
        interp = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*", "interpretability.json")))]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            os.path.basename(d.rstrip("/")), len(rows),
            fmt([r["test_best"]["oacc"] for r in rows]), fmt([r["test_best"]["macc"] for r in rows]), fmt([r["test_best"]["f1"] for r in rows]),
            fmt([r["highest_val"]["macc"] for r in rows]), fmt([r["highest_val"]["f1"] for r in rows]), fmt([r["test_last"]["f1"] for r in rows]),
            fmt([i.get("aopcr") for i in interp], 3), fmt([i.get("ndcg") for i in interp], 3)))
    return "\n".join(lines)


def aggregate_legacy(result_dir: str) -> str:
    lines = ["| paper row (config) | pooling | folds found | paper | release code: mACC | oACC | F1 | AOPCR | NDCG@n |", "|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        r = json.load(open(path))
        folds = [f for f in r["folds"] if "test" in f]
        pv = r.get("paper_values") or {}
        paper = ", ".join(f"{k} {v}" for k, v in pv.items()) or "-"
        aopcr = [f["interpretability"]["aopcr"] for f in folds if "interpretability" in f]
        ndcg = [f["interpretability"].get("ndcg") for f in folds if "interpretability" in f]
        errors = [f for f in r["folds"] if "error" in f]
        note = f" ({len(errors)} fold(s) failed to load)" if errors else ""
        lines.append("| {} | {} | {}/{}{} | {} | {} | {} | {} | {} | {} |".format(
            r["id"], r["pooling"], len(folds), len(r["folds"]), note, paper,
            fmt([f["test"]["macc"] for f in folds]), fmt([f["test"]["oacc"] for f in folds]), fmt([f["test"]["f1"] for f in folds]),
            fmt(aopcr, 3), fmt(ndcg, 3)))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["runs", "legacy"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "runs":
        dirs = []
        for p in args.paths:
            dirs.extend(sorted(d for d in glob.glob(p) if os.path.isdir(d)))
        table = aggregate_runs(dirs)
    else:
        table = aggregate_legacy(args.paths[0])
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")


if __name__ == "__main__":
    main()
