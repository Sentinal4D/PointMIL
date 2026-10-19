#!/usr/bin/env python
"""Write REPRODUCIBILITY.md (and the authors' HF_EXPORT_NOTES.md) from the evaluation and re-training outputs.

Authors' tool: its inputs are not part of the public repository.
Inputs: ``results/legacy/*.json`` (from scripts/evaluate_legacy.py), ``outputs/*/*/summary.json``
(from scripts/train.py) and the authors' manifest ``docs/legacy_checkpoints.yaml`` (not public;
see scripts/evaluate_legacy.py). Without the manifest only REPRODUCIBILITY.md is written.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml


def mean_sd(vals, d=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{d}f}"
    return f"{np.mean(vals):.{d}f} +/- {np.std(vals, ddof=1):.{d}f}"


def legacy_rows(result_dir: str, extra_notes: dict | None = None) -> tuple[str, list[dict]]:
    rows, records = [], []
    extra_notes = extra_notes or {}
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        r = json.load(open(path))
        folds = [f for f in r["folds"] if "test" in f]
        pv = r.get("paper_values") or {}
        paper = "; ".join(f"{k} {v}" for k, v in pv.items()) or "-"
        aopcr = [f["interpretability"]["aopcr"] for f in folds if "interpretability" in f]
        ndcg = [f["interpretability"].get("ndcg") for f in folds if "interpretability" in f]
        errs = [f for f in r["folds"] if "error" in f]
        missing = [f for f in r["folds"] if f.get("n_matches", 0) == 0]
        notes = []
        if r["id"] in extra_notes:
            notes.append(extra_notes[r["id"]])
        if errs:
            notes.append(f"{len(errs)} fold(s) failed to load")
        if missing:
            notes.append(f"{len(missing)} fold(s) without checkpoint")
        rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["id"], r["pooling"], f"{len(folds)}/{len(r['folds'])}", paper,
            mean_sd([f["test"]["macc"] for f in folds]), mean_sd([f["test"]["oacc"] for f in folds]), mean_sd([f["test"]["f1"] for f in folds]),
            mean_sd(aopcr, 3) + (" / " + mean_sd(ndcg, 3) if ndcg else ""), "; ".join(notes) or "-"))
        for f in r["folds"]:
            records.append({"row": r["id"], "paper": r["paper"], "pooling": r["pooling"], "format": r["format"], "config": r["config"], **f})
    header = ("| paper row | pooling | folds | paper value | mACC (release code) | oACC | F1 | AOPCR / NDCG@n | notes |\n"
              "|---|---|---|---|---|---|---|---|---|\n")
    return header + "\n".join(rows), records


def paper_values_by_config(manifest_path: str) -> dict:
    """Map experiment config path -> published values, from the authors' manifest when present."""
    if not os.path.exists(manifest_path):
        return {}
    out = {}
    for row in yaml.safe_load(open(manifest_path)).get("rows", []):
        for pooling in row.get("poolings", []):
            pv = (row.get("paper_values") or {}).get(pooling)
            if pv:
                out.setdefault(row["config"].format(pooling=pooling), pv)
    return out


def retrain_rows(out_dir: str, manifest_path: str = "") -> str:
    paper = paper_values_by_config(manifest_path)
    lines = ["| experiment | folds | paper value | test mACC (best ckpt) | test oACC | test F1 | max over epochs: mACC | max over epochs: F1 | AOPCR / NDCG@n | epochs run | train time (h) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(glob.glob(os.path.join(out_dir, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in ("smoke", "sanity"):  # test runs of scripts/smoke_test.sh
            continue
        sums = sorted(glob.glob(os.path.join(d, "*", "summary.json")))
        if not sums:
            continue
        rows = [json.load(open(s))["summary"] for s in sums]
        cfg_path = ""
        cfg_file = os.path.join(os.path.dirname(sums[0]), "config.yaml")
        if os.path.exists(cfg_file):
            cfg_path = (yaml.safe_load(open(cfg_file)) or {}).get("_config_path", "")
        if not cfg_path:
            name = os.path.basename(d)
            group, _, rest = name.partition("_")
            cfg_path = f"configs/{group}/{rest}.yaml"
        pv = paper.get(cfg_path, {})
        interp = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*", "interpretability.json")))]
        interp_cell = (mean_sd([i["aopcr"] for i in interp], 3) + " / " + mean_sd([i["ndcg"] for i in interp], 3)) if interp else "-"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            os.path.basename(d), len(rows), "; ".join(f"{k} {v}" for k, v in pv.items()) or "-",
            mean_sd([r["test_best"]["macc"] for r in rows]), mean_sd([r["test_best"]["oacc"] for r in rows]),
            mean_sd([r["test_best"]["f1"] for r in rows]), mean_sd([r["highest_val"]["macc"] for r in rows]), mean_sd([r["highest_val"]["f1"] for r in rows]),
            interp_cell, mean_sd([r["epochs_run"] for r in rows], 0), mean_sd([r["train_time_s"] / 3600 for r in rows])))
    return "\n".join(lines)


def noise_rows(noise_dir: str) -> str:
    """Mean F1 and mACC over folds per number of noisy points, grouped by model (from scripts/noise_robustness.py outputs)."""
    files = sorted(glob.glob(os.path.join(noise_dir, "*.json")))
    if not files:
        return "_No noise robustness results found._"
    groups: dict[str, list[dict]] = {}
    for f in files:
        r = json.load(open(f))
        key = os.path.basename(f).rsplit("_fold", 1)[0]
        groups.setdefault(key, []).append(r)
    levels = sorted({x["noise_points"] for rs in groups.values() for r in rs for x in r["results"]})
    lines = ["| model (sigma = 1.0 in unit-ball coordinates) | folds | " + " | ".join(f"F1 / mACC at {n} noisy points" for n in levels) + " |",
             "|---|---|" + "---|" * len(levels)]
    for key, rs in groups.items():
        cells = []
        for n in levels:
            f1 = [x["f1"] for r in rs for x in r["results"] if x["noise_points"] == n]
            macc = [x["macc"] for r in rs for x in r["results"] if x["noise_points"] == n]
            cells.append(f"{np.mean(f1):.1f} / {np.mean(macc):.1f}" if f1 else "-")
        lines.append(f"| {key} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hf_notes(records: list[dict], manifest: dict) -> str:
    lines = ["# Checkpoints found for the Hugging Face release", "",
             "Every checkpoint of the original code bases that corresponds to a paper result, with the fold, the metrics obtained when the",
             "release code re-evaluates it (test split of the fold, random 1024-point subsample, seed 42) and whether it loads cleanly.",
             "Paths are on the authors' workstation. `unexpected keys` are tensors of the original file that the ported model does not use",
             "(e.g. unused classifier heads); `missing` keys would mean a failed load and are reported in the notes.", "",
             "The Hugging Face repositories are built from these checkpoints with `scripts/export_hf.py` (layout in `pointmil/hub.py`).", "",
             "| paper row | pooling | fold | checkpoint | format | epoch | loads | test mACC | oACC | F1 | AOPCR | NDCG@n |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec in records:
        if rec.get("n_matches", 0) == 0:
            lines.append(f"| {rec['row']} | {rec['pooling']} | {rec.get('fold')} | (no checkpoint found: {rec['pattern']}) | {rec['format']} | - | - | - | - | - | - | - |")
            continue
        test = rec.get("test", {})
        interp = rec.get("interpretability", {})
        loads = "error: " + rec["error"][:60] if "error" in rec else f"yes ({rec.get('unexpected_keys', 0)} unexpected keys)"
        lines.append("| {} | {} | {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            rec["row"], rec["pooling"], rec.get("fold"), rec.get("checkpoint", ""), rec["format"], rec.get("epoch"), loads,
            f"{test['macc']:.2f}" if test else "-", f"{test['oacc']:.2f}" if test else "-", f"{test['f1']:.2f}" if test else "-",
            f"{interp['aopcr']:.3f}" if interp else "-", f"{interp['ndcg']:.3f}" if interp.get("ndcg") is not None else "-"))
    non_portable = manifest.get("non_portable", [])
    if non_portable:
        lines += ["", "## Checkpoints of the other code bases (not loadable by the release code)", "",
                  "ModelNet40 CurveNet dagger rows were trained with the authors' CurveNet fork and PointMLPElite dagger rows with the PointMLP fork;",
                  "their checkpoints use those repositories' model classes and formats and were not ported.", "",
                  "| paper row | folder (authors' workstation) | format | metric from the run log |", "|---|---|---|---|"]
        for item in non_portable:
            lines.append("| {} | `{}` | {} | {} |".format(item["row"], item["path"], item["format"], item["metric"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", default="results/legacy")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--noise", default="results/noise")
    ap.add_argument("--manifest", default="docs/legacy_checkpoints.yaml")
    ap.add_argument("--template", default="docs/reproducibility_template.md")
    ap.add_argument("--out", default="REPRODUCIBILITY.md")
    ap.add_argument("--hf_out", default="HF_EXPORT_NOTES.md")
    ap.add_argument("--fresh_env", default="results/fresh_env.md", help="markdown written after the fresh environment test; inserted verbatim")
    ap.add_argument("--gpu_determinism", default="results/determinism_gpu.md", help="markdown with the GPU determinism measurement; inserted verbatim when present")
    args = ap.parse_args()
    manifest_notes = (yaml.safe_load(open(args.manifest)) or {}).get("notes", {}) if os.path.exists(args.manifest) else {}
    legacy_table, records = legacy_rows(args.legacy, manifest_notes)
    retrain_table = retrain_rows(args.outputs, args.manifest)
    template = Path(args.template).read_text() if os.path.exists(args.template) else "# Reproducibility\n\n{{LEGACY_TABLE}}\n\n{{RETRAIN_TABLE}}\n"
    fresh = Path(args.fresh_env).read_text().strip() if os.path.exists(args.fresh_env) else "Not run yet (see `install.sh`)."
    text = (template.replace("{{LEGACY_TABLE}}", legacy_table).replace("{{RETRAIN_TABLE}}", retrain_table)
            .replace("{{NOISE_TABLE}}", noise_rows(args.noise)).replace("{{FRESH_ENV}}", fresh)
            .replace("{{GPU_DETERMINISM}}", Path(args.gpu_determinism).read_text().strip() if os.path.exists(args.gpu_determinism) else ""))
    Path(args.out).write_text(text)
    if os.path.exists(args.manifest):
        Path(args.hf_out).write_text(hf_notes(records, yaml.safe_load(open(args.manifest))))
        print(f"wrote {args.out} and {args.hf_out}")
    else:
        print(f"wrote {args.out} ({args.manifest} not found, {args.hf_out} skipped)")


if __name__ == "__main__":
    main()
