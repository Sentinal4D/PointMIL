"""Plain PyTorch training loop with CSV/JSON logging, checkpointing and early stopping."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
from torch import nn

from pointmil.checkpoints import save_checkpoint
from pointmil.engine.evaluate import evaluate
from pointmil.engine.losses import build_loss
from pointmil.engine.optim import build_optimizer, build_scheduler
from pointmil.metrics import ConfusionMatrix
from pointmil.utils.io import save_json

log = logging.getLogger("pointmil")


class Trainer:
    """Train a PointMIL model according to the ``train`` section of a config.

    Args:
        cfg: full experiment config (already validated).
        model: the model.
        loaders: ``{"train", "val", "test"}`` data loaders.
        out_dir: run directory (``log.csv``, ``best.pt``, ``last.pt``, ``summary.json``).
        device: torch device.
        wandb_run: optional ``wandb`` run object for online logging.
    """

    def __init__(self, cfg: dict, model: nn.Module, loaders: dict, out_dir: str | Path, device: torch.device, wandb_run=None):
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.model = model.to(device)
        self.loaders = loaders
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.wandb = wandb_run
        self.num_classes = int(cfg["model"]["pooling"]["num_classes"])
        self.criterion = build_loss(self.tcfg["loss"]).to(device)
        self.optimizer = build_optimizer(self.model, self.tcfg["optimizer"])
        self.epochs = int(self.tcfg["epochs"])
        self.scheduler = build_scheduler(self.optimizer, self.tcfg["scheduler"], self.epochs)
        self.monitor = self.tcfg["checkpoint_monitor"]  # e.g. val_f1, val_macc, val_oacc, val_loss
        if self.monitor not in {"val_f1", "val_macc", "val_oacc", "val_loss"}:
            raise ValueError(f"train.checkpoint_monitor must be val_f1, val_macc, val_oacc or val_loss, not '{self.monitor}'")
        self.mode = "min" if self.monitor.endswith("loss") else "max"
        self.patience = int(self.tcfg["early_stopping_patience"])
        self.grad_clip = self.tcfg.get("grad_clip")
        self.log_path = self.out_dir / "log.csv"
        self.history: list[dict] = []

    # ------------------------------------------------------------------ epochs
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        cm = ConfusionMatrix(self.num_classes)
        total, n = 0.0, 0
        for batch in self.loaders["train"]:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            out = self.model(x)
            loss = self.criterion(out["bag_logits"], y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip), norm_type=2)
            self.optimizer.step()
            total += float(loss.item()) * x.shape[0]
            n += x.shape[0]
            cm.update(out["bag_logits"].argmax(dim=1), y)
        res = cm.summary()
        res["loss"] = total / max(n, 1)
        return res

    def fit(self) -> dict:
        """Train for ``train.epochs`` epochs with checkpoint selection on the validation loader; returns the summary."""
        best_value = -float("inf") if self.mode == "max" else float("inf")
        best_epoch, bad_epochs = 0, 0
        best_val: dict = {}
        start = time.time()
        for epoch in range(self.epochs):
            self.scheduler.step(epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            t0 = time.time()
            train_res = self.train_epoch(epoch)
            val_res = evaluate(self.model, self.loaders["val"], self.device, self.num_classes, self.criterion)
            row = {"epoch": epoch + 1, "lr": lr, "time_s": time.time() - t0}
            row.update({f"train_{k}": v for k, v in train_res.items() if k != "per_class_acc"})
            row.update({f"val_{k}": v for k, v in val_res.items() if k not in ("per_class_acc", "predictions")})
            self._log_row(row)
            value = row[self.monitor]
            improved = value > best_value if self.mode == "max" else value < best_value
            if improved:
                best_value, best_epoch, bad_epochs = value, epoch + 1, 0
                best_val = val_res
                save_checkpoint(self.out_dir / "best.pt", self.model, self.cfg, metrics={"val": _strip(val_res), "train": _strip(train_res)}, epoch=epoch + 1)
            else:
                bad_epochs += 1
            save_checkpoint(self.out_dir / "last.pt", self.model, self.cfg, metrics={"val": _strip(val_res), "train": _strip(train_res)}, epoch=epoch + 1)
            log.info(
                "epoch %d/%d lr %.2e train loss %.4f oacc %.2f | val loss %.4f oacc %.2f macc %.2f f1 %.2f | best %s %.4f @%d",
                epoch + 1, self.epochs, lr, train_res["loss"], train_res["oacc"], val_res["loss"], val_res["oacc"], val_res["macc"], val_res["f1"], self.monitor, best_value, best_epoch,
            )
            if self.patience and bad_epochs >= self.patience:
                log.info("early stopping after %d epochs without improvement", bad_epochs)
                break
        summary = {
            "best_epoch": best_epoch,
            "best_monitor": self.monitor,
            "best_monitor_value": best_value,
            "best_val": _strip(best_val),
            "epochs_run": len(self.history),
            "train_time_s": time.time() - start,
        }
        # Test the best and the last checkpoint.
        for tag in ("best", "last"):
            ckpt = torch.load(self.out_dir / f"{tag}.pt", map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            res = evaluate(self.model, self.loaders["test"], self.device, self.num_classes, self.criterion, return_predictions=True)
            summary[f"test_{tag}"] = _strip(res)
            save_json(res["predictions"], self.out_dir / f"predictions_test_{tag}.json")
            log.info("test (%s checkpoint, epoch %s): oacc %.2f macc %.2f f1 %.2f", tag, ckpt.get("epoch"), res["oacc"], res["macc"], res["f1"])
        summary["highest_val"] = {k: max(r[f"val_{k}"] for r in self.history) for k in ("oacc", "macc", "f1")} if self.history else {}
        save_json({"config": self.cfg, "summary": summary}, self.out_dir / "summary.json")
        if self.wandb is not None:
            self.wandb.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        return summary

    def _log_row(self, row: dict) -> None:
        self.history.append(row)
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        if self.wandb is not None:
            self.wandb.log(row, step=row["epoch"])


def _strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "predictions"}
