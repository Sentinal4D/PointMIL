#!/usr/bin/env python
"""Point-level interpretability: AOPCR and NDCG@n (Table 1), or per-cloud importance maps.

Metrics mode (IntrA annotated segments, as in the paper)::

    python scripts/interpret.py metrics --checkpoint best.pt --config configs/table1/intra_transformer_conjunctive.yaml \
        --split annotated --out results/interp.json

Importance mode (write per-point importance for every cloud of a split)::

    python scripts/interpret.py importance --checkpoint best.pt --config ... --fold 0 --split test --out_dir results/importance

For the Transformer, DGCNN and PointNet models of the paper, contextual attention
with ``k = 12`` was applied at inference time; the context k of the loaded model
can be overridden with ``--context_k``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointmil.data import build_dataset  # noqa: E402
from pointmil.interpret import calculate_aopcr, calculate_ndcg_at_n, point_cloud_saliency  # noqa: E402
from pointmil.interpret.psm import rank_scores  # noqa: E402
from pointmil.models.pointmil import importance_from_output  # noqa: E402
from pointmil.utils import save_json, seed_everything  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import load_model_and_config  # noqa: E402


def set_context_k(model: torch.nn.Module, k: int | None) -> None:
    """Override the contextual attention neighbourhood ``k`` of every pooling head (``k = 0`` disables it)."""
    if k is None:
        return
    for m in model.modules():
        if hasattr(m, "contextual_attention") and hasattr(m, "context_k"):
            m.contextual_attention = k > 0
            m.context_k = k


def run_metrics(args, model, cfg, device) -> dict:
    """AOPCR (no labels needed) and NDCG@n (needs per-point ``seg`` labels) over a split."""
    dataset = build_dataset(cfg["data"], args.split, train=False, fold=args.fold)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    aopcrs, ndcgs = [], []
    target = args.target_class
    for batch in loader:
        x = batch["x"].to(device)
        tgt = None if target is None else torch.full((x.shape[0],), target, dtype=torch.long)
        if args.method == "psm":
            labels = (tgt if tgt is not None else batch["y"]).to(device)
            scores = rank_scores(point_cloud_saliency(model, x, labels))
            aopcr, _, _ = calculate_aopcr(model, x, stop=args.stop, step=args.step, n_random=args.n_random, seed=args.aopcr_seed, batch_interpretations=scores, target_class=tgt)
        else:
            with torch.no_grad():
                aopcr, _, _ = calculate_aopcr(model, x, stop=args.stop, step=args.step, n_random=args.n_random, seed=args.aopcr_seed, target_class=tgt)
                out = model(x)
                scores = model.interpret(out).cpu()
                cls = tgt if tgt is not None else out["bag_logits"].argmax(dim=1).cpu()
                scores = scores[torch.arange(x.shape[0]), cls]
        aopcrs.extend(aopcr.tolist())
        if "seg" in batch:
            for i in range(x.shape[0]):
                seg = batch["seg"][i]
                if (seg == 1).sum() > 0:
                    ndcgs.append(calculate_ndcg_at_n(scores[i], seg))
    res = {"split": args.split, "fold": args.fold, "num_clouds": len(aopcrs), "aopcr": float(np.mean(aopcrs)), "aopcr_std": float(np.std(aopcrs)),
           "method": args.method, "context_k": args.context_k, "target_class": target}
    if ndcgs:
        res.update(ndcg=float(np.mean(ndcgs)), ndcg_std=float(np.std(ndcgs)), num_ndcg=len(ndcgs))
    return res


def run_importance(args, model, cfg, device) -> None:
    """Write one ``<name>.npz`` per cloud (``xyz``, ``importance`` [N], raw ``interpretation`` [C, N], label, prediction)."""
    dataset = build_dataset(cfg["data"], args.split, train=False, fold=args.fold)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            pred = out["bag_logits"].argmax(dim=1)
            importance = importance_from_output(out, args.target_class).cpu()
            for i in range(x.shape[0]):
                imp = importance[i].numpy()
                name = batch["name"][i]
                np.savez(out_dir / f"{name}.npz", xyz=batch["pos"][i].numpy(), importance=imp, interpretation=out["interpretation"][i].cpu().numpy(), label=int(batch["y"][i]), pred=int(pred[i]))
                rows.append({"name": name, "label": int(batch["y"][i]), "pred": int(pred[i])})
    save_json(rows, out_dir / "index.json")
    logging.info("wrote %d importance maps to %s", len(rows), out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["metrics", "importance"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", default="annotated")
    ap.add_argument("--data_root", default=os.environ.get("POINTMIL_DATA"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context_k", type=int, default=None, help="override contextual attention k at inference (0 disables)")
    ap.add_argument("--method", choices=["model", "psm"], default="model", help="model interpretation or point cloud saliency baseline")
    ap.add_argument("--target_class", type=int, default=None, help="class whose interpretation is scored, for AOPCR and NDCG@n alike (default: the predicted class); the Table 1 scripts use 1, the aneurysm class")
    ap.add_argument("--stop", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--aopcr_seed", type=int, default=72)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out_dir", default="results/importance")
    ap.add_argument("--model_from_checkpoint", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    seed_everything(args.seed)
    model, cfg, _ = load_model_and_config(args)
    device = torch.device(args.device)
    model.to(device)
    set_context_k(model, args.context_k)
    if args.mode == "metrics":
        res = run_metrics(args, model, cfg, device)
        res["checkpoint"] = str(args.checkpoint)
        logging.info("%s", res)
        if args.out:
            save_json(res, args.out)
    else:
        run_importance(args, model, cfg, device)


if __name__ == "__main__":
    main()
