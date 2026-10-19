#!/usr/bin/env python
"""Train a PointMIL model from a config file.

Example::

    python scripts/train.py --config configs/table1/intra_transformer_conjunctive.yaml --fold 0 --seed 42 \
        --data_root /path/to/datasets --out_dir outputs

The data root replaces ``${POINTMIL_DATA}`` in the config (or set the
``POINTMIL_DATA`` environment variable). Every hyperparameter lives in the config;
``--override key.path=value`` changes single entries.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointmil.config import load_config  # noqa: E402
from pointmil.data import build_dataloaders  # noqa: E402
from pointmil.engine import Trainer  # noqa: E402
from pointmil.models import build_model  # noqa: E402
from pointmil.utils import save_yaml, seed_everything  # noqa: E402


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--override expects key=value (e.g. train.epochs=10), got '{item}'")
        out[key] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None, help="cross-validation fold (IntrA, RBC)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default=os.environ.get("POINTMIL_DATA"), help="dataset root (replaces ${POINTMIL_DATA})")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action="store_true", help="also log to Weights and Biases (needs `wandb login`)")
    ap.add_argument("--wandb_project", default="PointMIL")
    ap.add_argument("--override", nargs="*", default=[], help="config overrides, e.g. train.epochs=2 data.batch_size=4")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, overrides=parse_overrides(args.override), data_root=args.data_root)
    cfg["seed"] = args.seed
    cfg["fold"] = args.fold
    seed_everything(args.seed, deterministic=bool(cfg["train"].get("deterministic", True)))
    device = torch.device(args.device)

    run_name = args.run_name or (f"fold{args.fold}_seed{args.seed}" if args.fold is not None else f"seed{args.seed}")
    out_dir = Path(args.out_dir) / cfg["experiment"]["name"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config.yaml")

    loaders = build_dataloaders(cfg, fold=args.fold, seed=args.seed)
    model = build_model(cfg["model"])
    logging.info("model: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    logging.info("train/val/test sizes: %d/%d/%d", len(loaders["train"].dataset), len(loaders["val"].dataset), len(loaders["test"].dataset))

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=f"{cfg['experiment']['name']}/{run_name}", config=cfg, dir=str(out_dir))
    trainer = Trainer(cfg, model, loaders, out_dir, device, wandb_run=wandb_run)
    summary = trainer.fit()
    logging.info("done. best epoch %s, test (best ckpt): oacc %.2f macc %.2f f1 %.2f", summary["best_epoch"], summary["test_best"]["oacc"], summary["test_best"]["macc"], summary["test_best"]["f1"])
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
