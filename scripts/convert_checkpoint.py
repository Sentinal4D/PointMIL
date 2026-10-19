#!/usr/bin/env python
"""Convert a checkpoint of the original code bases into the release format.

Release checkpoints hold ``model_state_dict``, the experiment ``config`` (so that
``pointmil.load_checkpoint`` can rebuild the model), ``pointmil_version``,
``metrics`` and ``epoch``. Lightning checkpoints carry their training arguments and
need no config; OpenPoints checkpoints need the matching config file.

Example::

    python scripts/convert_checkpoint.py --checkpoint epoch=172-step=34946.ckpt \
        --config configs/table1/intra_dgcnn_additive.yaml --fold 0 --out release/intra_dgcnn_additive_fold0.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import load_model_and_config  # noqa: E402

from pointmil.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="experiment config; required for OpenPoints checkpoints, recommended otherwise")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", required=True)
    # keeping the literal placeholder (when POINTMIL_DATA is unset) keeps the saved config portable
    ap.add_argument("--data_root", default=os.environ.get("POINTMIL_DATA", "${POINTMIL_DATA}"))
    ap.add_argument("--model_from_checkpoint", action="store_true", help="build the model from the checkpoint arguments even when --config is given")
    ap.add_argument("--metrics", default=None, help="optional JSON string with metrics to store, e.g. '{\"test_macc\": 97.3}'")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model, cfg, ck = load_model_and_config(args)
    if cfg is None:
        cfg = ck["config"]
    cfg = dict(cfg)
    cfg["fold"] = args.fold
    cfg["converted_from"] = {"path": os.path.basename(args.checkpoint), "format": ck["format"], "epoch": ck.get("epoch")}
    metrics = json.loads(args.metrics) if args.metrics else dict(ck.get("metrics") or {})
    save_checkpoint(args.out, model, cfg, metrics=metrics, epoch=ck.get("epoch"))
    check = load_checkpoint(args.out)
    logging.info("wrote %s (%d tensors, format %s -> pointmil)", args.out, len(check["state_dict"]), ck["format"])


if __name__ == "__main__":
    main()
