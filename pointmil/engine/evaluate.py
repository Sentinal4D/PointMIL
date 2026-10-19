"""Evaluation loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pointmil.metrics import ConfusionMatrix


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int, criterion: nn.Module | None = None, return_predictions: bool = False) -> dict:
    """Return ``oacc``, ``macc``, ``f1`` (percent), ``loss`` and optionally per-sample predictions."""
    model.eval()
    cm = ConfusionMatrix(num_classes)
    losses, n = 0.0, 0
    preds, probs, targets, names = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        out = model(x)
        logits = out["bag_logits"]
        if criterion is not None:
            losses += float(criterion(logits, y).item()) * x.shape[0]
            n += x.shape[0]
        pred = logits.argmax(dim=1)
        cm.update(pred, y)
        if return_predictions:
            preds.append(pred.cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(y.cpu().numpy())
            names.extend(batch.get("name", [""] * x.shape[0]))
    result = cm.summary()
    result["loss"] = losses / max(n, 1) if criterion is not None else float("nan")
    result["num_samples"] = int(cm.value.sum())
    if return_predictions:
        result["predictions"] = {"pred": np.concatenate(preds).tolist(), "target": np.concatenate(targets).tolist(), "probs": np.concatenate(probs).tolist(), "names": list(names)}
    return result
