#!/usr/bin/env python
"""Re-evaluate the original checkpoints with the release code (authors' tool).

Needs the authors' manifest ``docs/legacy_checkpoints.yaml``, which maps every paper
result to its original run folder. It is not part of the public repository because it
lists paths on the authors' workstation; released checkpoints are evaluated with
``scripts/eval.py`` instead. Reads the manifest, loads every listed checkpoint into the
ported modules, evaluates it on the test split of its fold (and optionally the
interpretability metrics on the IntrA annotated segments) and writes one JSON per
row under ``--out_dir``. ``scripts/aggregate_results.py legacy`` turns them into
the tables of REPRODUCIBILITY.md.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from interpret import run_metrics, set_context_k  # noqa: E402

from pointmil.checkpoints import load_checkpoint, openpoints_attention_classifier, openpoints_head_layout, torch_load_tolerant  # noqa: E402
from pointmil.config import load_config  # noqa: E402
from pointmil.data import build_dataset  # noqa: E402
from pointmil.engine import evaluate  # noqa: E402
from pointmil.models import build_model  # noqa: E402
from pointmil.utils import save_json, seed_everything  # noqa: E402


def resolve_ckpt(root: str, pattern: str) -> list[str]:
    """Checkpoints matching ``pattern`` under ``root``, sorted by path (run folders carry a timestamp)."""
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))


def saved_feature_extractor(path: str) -> str | None:
    """``args.feature_extractor`` stored in a Lightning checkpoint (None for other formats)."""
    try:
        args = torch_load_tolerant(path).get("hyper_parameters", {}).get("args")
    except Exception:  # noqa: BLE001
        return None
    return getattr(args, "feature_extractor", None) if args is not None else None


def pick_checkpoint(paths: list[str], expected: str | None) -> tuple[str | None, list[str]]:
    """Latest matching checkpoint whose saved feature extractor equals ``expected``.

    Some run folders were named after one backbone while the code that ran used another
    (the backbone was hard coded at the time), so the folder name alone is not trusted.
    Returns the chosen path and the paths skipped because of a different backbone.
    """
    if expected is None:
        return paths[-1], []
    skipped = []
    for path in reversed(paths):
        found = saved_feature_extractor(path)
        if found is None or found == expected:
            return path, skipped
        skipped.append(f"{path} (feature_extractor={found})")
    return None, skipped


def build_from_checkpoint(ck_path: str, cfg: dict, fmt: str):
    ck = load_checkpoint(ck_path, model=None, build=False)
    if fmt == "openpoints":
        model_cfg = dict(cfg["model"])
        model_cfg["pooling"] = dict(model_cfg["pooling"], instance_head=openpoints_head_layout(ck["state_dict"]))
        if model_cfg["pooling"].get("pooling") == "attention" and "attention_classifier" not in model_cfg["pooling"]:
            model_cfg["pooling"]["attention_classifier"] = openpoints_attention_classifier(ck["state_dict"])
    else:
        model_cfg = ck["config"]["model"]
    model = build_model(model_cfg)
    report = load_checkpoint(ck_path, model=model, build=False)["load_report"]
    return model, ck, report, model_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="docs/legacy_checkpoints.yaml")
    ap.add_argument("--out_dir", default="results/legacy")
    ap.add_argument("--data_root", default=os.environ.get("POINTMIL_DATA"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="substring filter on row ids")
    ap.add_argument("--skip_interpretability", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    manifest = yaml.safe_load(open(args.manifest))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest["rows"]:
        for pooling in row["poolings"]:
            row_id = row["id"].format(pooling=pooling)
            if args.only and args.only not in row_id:
                continue
            out_path = out_dir / f"{row_id}.json"
            if out_path.exists():
                logging.info("skip %s (exists)", row_id)
                continue
            cfg = load_config(row["config"].format(pooling=pooling), data_root=args.data_root)
            root = manifest["roots"][row["root"]]
            result = {"id": row_id, "paper": row["paper"], "config": row["config"].format(pooling=pooling), "pooling": pooling, "format": row["format"],
                      "paper_values": (row.get("paper_values") or {}).get(pooling, {}), "folds": []}
            for fold in row["folds"]:
                pattern = row["ckpt"].format(pooling=pooling, fold=fold if fold is not None else "")
                paths = resolve_ckpt(root, pattern)
                entry = {"fold": fold, "pattern": pattern, "n_matches": len(paths)}
                if not paths:
                    logging.warning("%s fold %s: no checkpoint for %s", row_id, fold, pattern)
                    result["folds"].append(entry)
                    continue
                ck_path, skipped = pick_checkpoint(paths, row.get("feature_extractor"))
                if skipped:
                    entry["skipped"] = skipped
                    logging.warning("%s fold %s: skipped %d checkpoint(s) trained with another backbone: %s", row_id, fold, len(skipped), "; ".join(skipped))
                if ck_path is None:
                    entry["error"] = "every matching checkpoint was trained with another backbone"
                    result["folds"].append(entry)
                    continue
                entry["checkpoint"] = ck_path
                try:
                    seed_everything(42)
                    model, ck, report, model_cfg = build_from_checkpoint(ck_path, cfg, row["format"])
                    model.to(device).eval()
                    entry.update(epoch=ck.get("epoch"), unexpected_keys=len(report["unexpected"]), model_cfg=model_cfg)
                    ds = build_dataset(cfg["data"], "test", train=False, fold=fold)
                    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
                    t0 = time.time()
                    res = evaluate(model, loader, device, int(cfg["model"]["pooling"]["num_classes"]))
                    entry["test"] = {k: v for k, v in res.items() if k != "per_class_acc"}
                    entry["eval_time_s"] = time.time() - t0
                    logging.info("%s fold %s: oacc %.2f macc %.2f f1 %.2f (epoch %s)", row_id, fold, res["oacc"], res["macc"], res["f1"], ck.get("epoch"))
                    if row.get("interpretability") and not args.skip_interpretability:
                        icfg = row["interpretability"]
                        set_context_k(model, icfg.get("context_k"))
                        iargs = argparse.Namespace(split=icfg["split"], fold=fold, batch_size=args.batch_size, method="model", target_class=1,
                                                   stop=0.5, step=0.05, n_random=3, aopcr_seed=72, context_k=icfg.get("context_k"))
                        t0 = time.time()
                        ires = run_metrics(iargs, model, cfg, device)
                        ires["time_s"] = time.time() - t0
                        entry["interpretability"] = ires
                        logging.info("%s fold %s: aopcr %.3f ndcg %.3f", row_id, fold, ires["aopcr"], ires.get("ndcg", float("nan")))
                except Exception as exc:  # keep going, record the failure
                    logging.exception("%s fold %s failed", row_id, fold)
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                result["folds"].append(entry)
            save_json(result, out_path)


if __name__ == "__main__":
    main()
