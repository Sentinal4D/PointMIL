#!/usr/bin/env python
"""Evaluate a checkpoint on a dataset split.

Works with model folders from the Hugging Face Hub (``model.safetensors`` + ``config.yaml``),
checkpoints written by ``scripts/train.py`` (``best.pt``) and the checkpoints of the original
code bases (Lightning ``.ckpt`` files and OpenPoints ``*_ckpt_best.pth`` files). All but the
OpenPoints checkpoints carry enough information to rebuild the model; those need ``--config``.

Example::

    python scripts/eval.py --checkpoint outputs/table1_intra_transformer_conjunctive/fold0_seed42/best.pt --fold 0 --split test
    python scripts/eval.py --checkpoint hub/transformer/conjunctive/fold0          # a downloaded Hub folder
    python scripts/eval.py --checkpoint original.ckpt --config configs/table1/intra_transformer_conjunctive.yaml --fold 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointmil.checkpoints import load_checkpoint, openpoints_attention_classifier, openpoints_head_layout  # noqa: E402
from pointmil.config import load_config  # noqa: E402
from pointmil.data import build_dataset  # noqa: E402
from pointmil.engine import evaluate  # noqa: E402
from pointmil.engine.losses import build_loss  # noqa: E402
from pointmil.models import build_model  # noqa: E402
from pointmil.utils import save_json, seed_everything  # noqa: E402


def load_model_and_config(args) -> tuple[torch.nn.Module, dict, dict]:
    """Build the model of ``args.checkpoint`` and return ``(model, config, checkpoint info)``.

    ``args.checkpoint`` can be a model folder from the Hugging Face Hub, a checkpoint written by
    ``scripts/train.py``, or an original Lightning / OpenPoints checkpoint. ``args.config`` (an
    experiment YAML) supplies the data section and, for OpenPoints checkpoints, the model; with
    ``args.model_from_checkpoint`` the model is always rebuilt from the checkpoint. The returned
    config's ``model`` section is the one that built the returned model. Shared by
    ``interpret.py``, ``noise_robustness.py`` and ``convert_checkpoint.py``.
    """
    if Path(args.checkpoint).is_dir():
        # a model folder downloaded from the Hugging Face Hub (model.safetensors + config.yaml + metrics.json)
        from pointmil.hub import CONFIG, load_local

        model, info = load_local(args.checkpoint)
        cfg = load_config(args.config or Path(args.checkpoint) / CONFIG, data_root=args.data_root)
        if getattr(args, "fold", None) is None and info["config"].get("fold") is not None:
            args.fold = info["config"]["fold"]
        logging.info("loaded %s (%s)", args.checkpoint, info["metrics"].get("source", "hub model"))
        return model, cfg, {"format": "hub", "epoch": info["metrics"].get("epoch"), "metrics": info["metrics"], "config": info["config"]}
    cfg = load_config(args.config, data_root=args.data_root) if args.config else None
    ck = load_checkpoint(args.checkpoint, model=None, build=False)
    if ck["format"] == "openpoints":
        if cfg is None:
            raise SystemExit("--config is required for OpenPoints checkpoints")
        model_cfg = cfg["model"]
        if model_cfg["pooling"]["name"] == "openpoints":
            model_cfg["pooling"]["instance_head"] = openpoints_head_layout(ck["state_dict"])
            if model_cfg["pooling"].get("pooling") == "attention" and "attention_classifier" not in model_cfg["pooling"]:
                model_cfg["pooling"]["attention_classifier"] = openpoints_attention_classifier(ck["state_dict"])
    elif cfg is not None and not args.model_from_checkpoint:
        model_cfg = cfg["model"]
    else:
        if ck.get("config") is None:
            raise SystemExit("checkpoint has no model config; pass --config")
        model_cfg = ck["config"]["model"]
        if cfg is None:
            cfg = ck["config"] if ck["format"] == "pointmil" else None
    model = build_model(model_cfg)
    report = load_checkpoint(args.checkpoint, model=model, build=False)["load_report"]
    logging.info("loaded %d tensors (%d unexpected keys ignored)", report["loaded"], len(report["unexpected"]))
    if cfg is not None:
        cfg = dict(cfg, model=model_cfg)
    return model.eval(), cfg, ck


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="experiment config (data section is used; required for OpenPoints checkpoints)")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_root", default=os.environ.get("POINTMIL_DATA"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42, help="seed for the random point subsampling of the test clouds")
    ap.add_argument("--out", default=None, help="write metrics (and predictions) to this JSON file")
    ap.add_argument("--model_from_checkpoint", action="store_true", help="build the model from the checkpoint even when --config is given")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    seed_everything(args.seed)
    model, cfg, ck = load_model_and_config(args)
    if cfg is None:
        raise SystemExit("no data configuration available; pass --config")
    dataset = build_dataset(cfg["data"], args.split, train=False, fold=args.fold)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 4)))
    device = torch.device(args.device)
    model.to(device)
    loss_cfg = cfg.get("train", {}).get("loss")
    criterion = build_loss(loss_cfg).to(device) if loss_cfg else None  # without a training config the loss is reported as null
    res = evaluate(model, loader, device, int(cfg["model"]["pooling"]["num_classes"]), criterion=criterion, return_predictions=True)
    if criterion is None:
        res["loss"] = None
    printable = {k: v for k, v in res.items() if k != "predictions"}
    logging.info("%s split %s (%d samples): %s", cfg["data"]["name"], args.split, res["num_samples"], json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in printable.items() if k != "per_class_acc"}))
    if args.out:
        save_json({"checkpoint": str(args.checkpoint), "format": ck["format"], "epoch": ck.get("epoch"), "split": args.split, "fold": args.fold, "metrics": printable, "predictions": res["predictions"]}, args.out)


if __name__ == "__main__":
    main()
