#!/usr/bin/env python
"""Robustness to noisy points (Fig. 7 and Supplementary Fig. 12).

Gaussian noise (``--sigma``) is added to ``n`` randomly chosen points of every test
cloud of a fold, for ``n`` in ``--noise_points``; F1 and mACC of the loaded
checkpoint are reported per ``n``. Clouds are otherwise processed exactly as at
test time (random 1024-point subsample, unit ball normalisation); the subsample is
drawn once, so every noise level is evaluated on the same clouds.

Example::

    python scripts/noise_robustness.py --checkpoint outputs/table1_intra_dgcnn_additive/fold0_seed42/best.pt \
        --fold 0 --noise_points 0 10 50 100 200 --out results/noise_dgcnn_additive_fold0.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import load_model_and_config  # noqa: E402

from pointmil.data import build_dataset  # noqa: E402
from pointmil.metrics import ConfusionMatrix  # noqa: E402
from pointmil.utils import save_json, seed_everything  # noqa: E402


def add_noise(x: torch.Tensor, n: int, sigma: float, gen: torch.Generator) -> torch.Tensor:
    if n <= 0:
        return x
    x = x.clone()
    B, N, _ = x.shape
    for b in range(B):
        idx = torch.randperm(N, generator=gen)[:n]
        x[b, idx, :3] += sigma * torch.randn((n, 3), generator=gen)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_root", default=os.environ.get("POINTMIL_DATA"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--noise_points", type=int, nargs="+", default=[0, 10, 50, 100, 200, 500])
    ap.add_argument("--sigma", type=float, default=1.0, help="standard deviation of the Gaussian noise (unit-ball coordinates)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model_from_checkpoint", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    seed_everything(args.seed)
    model, cfg, _ = load_model_and_config(args)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = build_dataset(cfg["data"], args.split, train=False, fold=args.fold)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 4)))
    # draw the random 1024-point subsample once, so that every noise level sees the same clouds
    batches = [(batch["x"], batch["y"]) for batch in loader]
    num_classes = int(cfg["model"]["pooling"]["num_classes"])
    results = []
    for n in args.noise_points:
        gen = torch.Generator().manual_seed(args.seed)
        cm = ConfusionMatrix(num_classes)
        with torch.no_grad():
            for x, y in batches:
                cm.update(model(add_noise(x, n, args.sigma, gen).to(device))["bag_logits"].argmax(dim=1), y.to(device))
        s = cm.summary()
        results.append({"noise_points": n, "oacc": s["oacc"], "macc": s["macc"], "f1": s["f1"]})
        logging.info("noise points %4d: oacc %.2f macc %.2f f1 %.2f", n, s["oacc"], s["macc"], s["f1"])
    if args.out:
        save_json({"checkpoint": args.checkpoint, "fold": args.fold, "sigma": args.sigma, "results": results}, args.out)


if __name__ == "__main__":
    main()
