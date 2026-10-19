#!/usr/bin/env python
"""Build the Hugging Face model repositories (authors' tool: it needs the original checkpoints and
``results/legacy``, which are not part of the public repository).

Writes one folder per dataset repository under ``--out`` (see ``pointmil.hub.REPOS``):

    pointmil-<dataset>/
        README.md                      model card: layout, how to load, metrics, citation, licence
        LICENSE                        the PointMIL licence
        index.json                     every model with its path and metrics
        <backbone>/<pooling>/<fold>/   checkpoints behind the paper (original runs)
        retrained/<backbone>/<pooling>/<fold>/   models trained with this repository's code

Each model folder has ``model.safetensors``, ``config.yaml`` and ``metrics.json``.

Inputs: ``results/legacy/*.json`` (written by ``scripts/evaluate_legacy.py``; it records
which original checkpoint was evaluated and its metrics) and ``outputs/<experiment>/<run>/``
(written by ``scripts/train.py``). Every exported model is reloaded from the written files
and must reproduce the source model's outputs; the script also refuses to write any file
that contains a local absolute path.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pointmil  # noqa: E402
from pointmil.checkpoints import load_checkpoint  # noqa: E402
from pointmil.hub import REPOS, fold_dir, load_local, save_pretrained  # noqa: E402

CV_DATASETS = ("intra", "rbc")  # 5-fold cross-validation; the others have a single split
DATASET_TITLES = {"intra": "IntrA", "rbc": "3D red blood cells (RBC)", "atlas1": "ATLAS-1", "modelnet40": "ModelNet40", "scanobjectnn": "ScanObjectNN (PB_T50_RS)"}
BACKBONE_TITLES = {"transformer": "Transformer", "dgcnn": "DGCNN", "pointnet": "PointNet", "curvenet": "CurveNet (no downsampling)", "pointmlpelite": "PointMLPElite (no downsampling)"}
FORBIDDEN = re.compile(r"/home/|/mnt/|/Users/|\\\\Users\\\\|rds\.icr|wandb\.ai/[a-z]")
BIBTEX = """@InProceedings{De_Vries_2025_ICCV,
  author    = {De Vries, Matt and Naidoo, Reed and Fourkioti, Olga and Dent, Lucas G. and Curry, Nathan and Dunsby, Chris and Bakal, Chris},
  title     = {Interpretable point cloud classification using multiple instance learning},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {22209-22220}
}"""
VARIANT_NOTES = {
    "gap": "global average pooling with class activation maps (the CLAIM baseline of Table 1; the dagger baseline of Table 2 for PointMLPElite and CurveNet on ATLAS-1)",
    "baseline": "the original CurveNet classifier without MIL (CurveNet dagger baseline of Table 2); no point-level interpretation",
    "conjunctive-nocontext": "conjunctive pooling trained without contextual attention (Supp. Table 3, k = 0)",
    "attention-nocontext": "attention pooling trained without contextual attention (Supp. Table 3, k = 0)",
    "additive-nocontext": "additive pooling trained without contextual attention (Supp. Table 3, k = 0)",
    "conjunctive-k20": "conjunctive pooling trained with contextual attention k = 20 (Fig. 6)",
}


def split_name(name: str) -> tuple[str, str, str]:
    """``intra_transformer_conjunctive_nocontext`` -> ``(intra, transformer, conjunctive-nocontext)``."""
    dataset, backbone, rest = name.split("_", 2)
    variant = rest.replace("_nocontext", "-nocontext").replace("_k20", "-k20")
    variant = {"instance-nocontext": "instance", "original": "baseline"}.get(variant, variant)
    return dataset, backbone, variant


def raw_config(path: str) -> dict:
    """Experiment config as committed (``${POINTMIL_DATA}`` placeholders kept)."""
    return yaml.safe_load(open(ROOT / path))


def _portable_config(cfg: dict) -> dict:
    """Training config of a run with the data root replaced by ``${POINTMIL_DATA}/<folder>``."""
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    data = dict(cfg["data"])
    data["root"] = "${POINTMIL_DATA}/" + Path(str(data["root"])).name
    cfg["data"] = data
    return cfg


def example_input(config: dict, n: int = 1024, seed: int = 0) -> torch.Tensor:
    width = {"pos": 3, "normals": 3, "heights": 1}
    c = sum(width[k] for k in config["data"].get("feature_keys", ["pos"]))
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, n, c, generator=g)
    x[..., :3] = x[..., :3] / x[..., :3].norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True)
    return x


def check_clean(folder: Path) -> None:
    for f in folder.rglob("*"):
        if f.suffix in (".yaml", ".json", ".md") and FORBIDDEN.search(f.read_text()):
            raise SystemExit(f"refusing to export {f}: it contains a local path")


def verify(model: torch.nn.Module, folder: Path, config: dict) -> float:
    """Reload the written files and compare bag logits with the source model."""
    reloaded, _ = load_local(folder)
    model = model.cpu().eval()
    x = example_input(config)
    with torch.no_grad():
        a, b = model(x)["bag_logits"], reloaded(x)["bag_logits"]
    diff = float((a - b).abs().max())
    if diff > 1e-4:
        raise SystemExit(f"{folder}: reloaded model differs from the source (max abs diff {diff:.2e})")
    return diff


def round_metrics(d: dict | None, keys=("oacc", "macc", "f1", "loss", "aopcr", "aopcr_std", "ndcg", "ndcg_std")) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if k in keys and v is not None and np.isfinite(float(v)):
            out[k] = round(float(v), 4)
    return out


def export_original(args, entries: dict) -> None:
    from evaluate_legacy import build_from_checkpoint

    from pointmil.config import load_config

    for path in sorted(glob.glob(os.path.join(args.legacy, "*.json"))):
        row = json.load(open(path))
        dataset, backbone, variant = split_name(row["id"])
        if args.only and args.only not in row["id"]:
            continue
        for fo in row["folds"]:
            if "test" not in fo:
                continue
            fold = fo["fold"] if dataset in CV_DATASETS else None  # single-split datasets have no fold folder
            cfg = load_config(row["config"], data_root="${POINTMIL_DATA}")
            model, ck, report, model_cfg = build_from_checkpoint(fo["checkpoint"], cfg, row["format"])
            config = raw_config(row["config"])
            config["model"] = model_cfg
            config["fold"] = fold
            metrics = {
                "source": "original run behind the paper",
                "dataset": dataset, "backbone": backbone, "pooling": variant, "fold": fold,
                "epoch": fo.get("epoch"),
                "test": round_metrics(fo["test"]),
                "interpretability": dict(round_metrics(fo.get("interpretability")), **({"split": "annotated (116 IntrA segments)", "target_class": "aneurysm"} if fo.get("interpretability") else {})),
                "paper_value": row.get("paper_values") or {},
                "evaluated_with": f"pointmil {pointmil.__version__}, scripts/evaluate_legacy.py (test split of the fold, random 1024-point subsample, seed 42)",
            }
            sub = f"{backbone}/{variant}/{fold_dir(dataset, fold)}"
            out = Path(args.out) / f"pointmil-{dataset}" / sub
            save_pretrained(model, config, metrics, out)
            diff = verify(model, out, config)
            entries[dataset].append(dict(path=sub, retrained=False, paper_value=metrics["paper_value"], inputs=config["data"].get("feature_keys", ["pos"]),
                                         **{k: metrics[k] for k in ("backbone", "pooling", "fold", "test", "interpretability")}))
            print(f"  {dataset:12s} {sub:45s} ok (reload diff {diff:.1e})")


def export_retrained(args, entries: dict) -> None:
    for summary_path in sorted(glob.glob(os.path.join(args.outputs, "table*", "*", "summary.json"))):
        run = Path(summary_path).parent
        exp = run.parent.name
        group, _, name = exp.partition("_")
        dataset, backbone, variant = split_name(name)
        if args.only and args.only not in name:
            continue
        cfg_rel = f"configs/{group}/{name}.yaml"
        ck = load_checkpoint(run / "best.pt")
        # the config the model was trained with (configs/ may have changed since), local paths removed
        config = _portable_config(ck["config"])
        summary = json.load(open(summary_path))["summary"]
        m = re.match(r"fold(\d+)_seed(\d+)", run.name)
        fold = int(m.group(1)) if m and dataset in CV_DATASETS else None
        config["fold"] = fold
        interp_path = run / "interpretability.json"
        interp = json.load(open(interp_path)) if interp_path.exists() else None
        metrics = {
            "source": "trained with the PointMIL release code",
            "dataset": dataset, "backbone": backbone, "pooling": variant, "fold": fold,
            "config_file": cfg_rel, "seed": int(m.group(2)) if m else 42,
            "epoch": summary.get("best_epoch"), "epochs_run": summary.get("epochs_run"),
            "test": round_metrics(summary.get("test_best")),
            "best_over_epochs": round_metrics(summary.get("highest_val")),
            "interpretability": dict(round_metrics(interp), **({"split": "annotated (116 IntrA segments)", "context_k": interp.get("context_k"), "target_class": "aneurysm"} if interp else {})),
            "train_time_h": round(summary.get("train_time_s", 0) / 3600, 2),
        }
        sub = f"retrained/{backbone}/{variant}/{fold_dir(dataset, fold)}"
        out = Path(args.out) / f"pointmil-{dataset}" / sub
        save_pretrained(ck["model"], config, metrics, out)
        diff = verify(ck["model"], out, config)
        entries[dataset].append(dict(path=sub, retrained=True, paper_value={}, inputs=config["data"].get("feature_keys", ["pos"]),
                                     **{k: metrics[k] for k in ("backbone", "pooling", "fold", "test", "interpretability")}))
        print(f"  {dataset:12s} {sub:45s} ok (reload diff {diff:.1e})")


def mean_sd(values: list[float], digits: int = 1) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def summary_table(models: list[dict], dataset: str) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in models:
        groups[(m["backbone"], m["pooling"])].append(m)
    has_interp = any(m["interpretability"] for m in models)
    head = "| backbone | pooling | inputs | folds | mACC | oACC | F1 |" + (" AOPCR | NDCG@n |" if has_interp else "") + " paper | folder |"
    sep = "|---|---|---|---|---|---|---|" + ("---|---|" if has_interp else "") + "---|---|"
    lines = [head, sep]
    for (bb, pool), ms in sorted(groups.items()):
        folds = sorted(str(m["fold"]) if m["fold"] is not None else "-" for m in ms)
        inputs = " + ".join({"pos": "xyz", "normals": "normals", "heights": "height"}[k] for k in ms[0]["inputs"])
        row = f"| {BACKBONE_TITLES.get(bb, bb)} | {pool} | {inputs} | {', '.join(folds)} | {mean_sd([m['test']['macc'] for m in ms])} | {mean_sd([m['test']['oacc'] for m in ms])} | {mean_sd([m['test']['f1'] for m in ms])} |"
        if has_interp:
            a = [m["interpretability"]["aopcr"] for m in ms if m["interpretability"]]
            n = [m["interpretability"]["ndcg"] for m in ms if m["interpretability"]]
            row += f" {mean_sd(a, 2)} | {mean_sd(n, 3)} |"
        pv = ms[0].get("paper_value") or {}
        row += " " + (", ".join(f"{k} {v}" for k, v in pv.items()) or "-") + " |"
        prefix = "retrained/" if ms[0]["retrained"] else ""
        row += f" `{prefix}{bb}/{pool}/` |"
        lines.append(row)
    return "\n".join(lines)


def model_card(dataset: str, models: list[dict]) -> str:
    repo = REPOS[dataset]
    original = [m for m in models if not m["retrained"]]
    retrained = [m for m in models if m["retrained"]]
    # prefer a model whose inputs prepare_cloud can build (xyz or xyz + normals, no height channel)
    example = sorted(original or retrained, key=lambda m: ("heights" in m["inputs"], m["backbone"] != "transformer", "-" in m["pooling"], m["pooling"] != "conjunctive", m["path"]))[0]
    ex_fold = "" if example["fold"] is None else f", fold={example['fold']}"
    ex_retrained = ", retrained=True" if example["retrained"] else ""
    folds = "folds `fold0` to `fold4` (5-fold cross-validation, splits in the code repository under `splits/`)" if dataset in ("intra", "rbc") else (
        "one folder `split0` (single train/validation/test split, `splits/atlas1/split0.csv`)" if dataset == "atlas1" else "one folder `official` (official train/test split)")
    variants = sorted({m["pooling"] for m in models if m["pooling"] in VARIANT_NOTES})
    front = ["---", "license: other", "license_name: sentinal4d-non-commercial-eula", "license_link: LICENSE", "library_name: pytorch",
             "tags:", "  - point-cloud", "  - point-cloud-classification", "  - multiple-instance-learning", "  - interpretability", "  - pointmil"]
    if dataset == "atlas1":
        front += ["datasets:", "  - Sentinal4D/ATLAS-1"]
    front += ["---", ""]
    body = f"""# PointMIL models: {DATASET_TITLES[dataset]}

Trained models from **Interpretable point cloud classification using multiple instance learning**
(De Vries et al., ICCV 2025). PointMIL classifies a point cloud and, in the same forward pass,
scores how much every point contributed to the prediction.

- Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/De_Vries_Interpretable_point_cloud_classification_using_multiple_instance_learning_ICCV_2025_paper.pdf
- Code: https://github.com/Sentinal4D/PointMIL (install with `pip install -e .[hub]`)
- Project page: https://sentinal4d.github.io/PointMIL

## Load a model

```python
from pointmil.hub import load_pretrained, list_pretrained

from pointmil import prepare_cloud

model, info = load_pretrained("{dataset}", "{example['backbone']}", "{example['pooling']}"{ex_fold}{ex_retrained})
x = prepare_cloud(points, num_points=1024)  # points: numpy [M, 3] (xyz) or [M, 6] (xyz + normals)
out = model(x)
pred = out["bag_logits"].argmax(1)          # class
importance = model.point_importance(x)      # [1, 1024] point-level importance for the predicted class
print(info["metrics"]["test"])

print([m["path"] for m in list_pretrained("{dataset}")])   # everything in this repository
```

Or download a folder directly with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
from pointmil.hub import load_local

folder = snapshot_download("{repo}", allow_patterns="{example['path']}/*")
model, info = load_local(f"{{folder}}/{example['path']}")
```

The input features of each model (column "inputs" below, and `data.feature_keys` in its
`config.yaml`) must match: xyz, xyz + normals, or xyz + height above the lowest point for
ScanObjectNN (added by `pointmil.data.transforms.PointCloudCenterAndNormalize`).

## Layout

```
<backbone>/<pooling>/<fold>/            checkpoints behind the paper (original training runs)
retrained/<backbone>/<pooling>/<fold>/  models trained with the released code (see below)
    model.safetensors                   weights
    config.yaml                         full experiment config; config["model"] builds the network
    metrics.json                        test metrics, interpretability metrics and provenance
index.json                              every model in this repository with its metrics
```

Each `<backbone>/<pooling>/` holds {folds}. Backbones: {", ".join(sorted({BACKBONE_TITLES.get(m['backbone'], m['backbone']) for m in models}))}.
Pooling heads: instance, attention, additive and conjunctive (Eq. 1 to 4 of the paper){"; " + "; ".join(f"`{v}`: {VARIANT_NOTES[v]}" for v in variants) if variants else ""}.

## Metrics

Test metrics of each checkpoint re-evaluated with the released code (mean ± standard deviation
over folds; mACC = mean class accuracy, oACC = overall accuracy, F1 = macro F1, in percent).
Each cloud is subsampled to 1024 random points at evaluation. The paper reports the best value
over training epochs on a validation split that was the test split, so its numbers are
slightly higher; `REPRODUCIBILITY.md` in the code repository explains every difference.
{"AOPCR and NDCG@n are the Table 1 interpretability metrics on the 116 annotated IntrA segments." if dataset == "intra" else ""}

### Checkpoints behind the paper

{summary_table(original, dataset) if original else "None in this repository."}
"""
    if retrained:
        body += f"""
### Models trained with the released code

These were trained with `scripts/train.py` (seed 42); each folder's `config.yaml` is the exact
config of its run. They fill folds whose original checkpoint was not found and provide 5-fold
sets where the paper used a single fold.

{summary_table(retrained, dataset)}
"""
    body += f"""
## Citation

```bibtex
{BIBTEX}
```

## Licence

The models are released under the Sentinal4D Non-Commercial End User License Agreement
(`LICENSE`), the licence of the PointMIL code: free for non-commercial research, academic and
educational use.{" The ATLAS-1 dataset itself is released under CC BY 4.0." if dataset == "atlas1" else ""}
"""
    return "\n".join(front) + body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="staging folder, e.g. ../PointMIL-hf")
    ap.add_argument("--legacy", default="results/legacy")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--only", default=None, help="substring filter on model names (for testing)")
    ap.add_argument("--skip_original", action="store_true")
    ap.add_argument("--skip_retrained", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    entries: dict[str, list[dict]] = defaultdict(list)
    if not args.skip_original:
        print("original checkpoints:")
        export_original(args, entries)
    if not args.skip_retrained:
        print("retrained models:")
        export_retrained(args, entries)
    for dataset, models in entries.items():
        repo_dir = Path(args.out) / f"pointmil-{dataset}"
        index = {"repo_id": REPOS[dataset], "pointmil_version": pointmil.__version__,
                 "models": sorted(models, key=lambda m: m["path"])}
        (repo_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
        (repo_dir / "README.md").write_text(model_card(dataset, models))
        shutil.copy(ROOT / "LICENSE", repo_dir / "LICENSE")
        check_clean(repo_dir)
        size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file()) / 1e6
        print(f"{REPOS[dataset]}: {len(models)} models, {size:.0f} MB -> {repo_dir}")


if __name__ == "__main__":
    main()
