"""Classification metrics computed from a confusion matrix.

Definitions follow the paper: overall accuracy (oACC), mean per-class accuracy
(mACC, the macro average of per-class recall) and macro F1. These coincide with
the torchmetrics (original code) and OpenPoints ``ConfusionMatrix`` definitions
used to produce the published numbers.
"""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrix:
    """Accumulate predictions and compute oACC, mACC and macro F1 (all in percent)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.value = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        pred = np.asarray(pred.detach().cpu() if torch.is_tensor(pred) else pred).reshape(-1)
        target = np.asarray(target.detach().cpu() if torch.is_tensor(target) else target).reshape(-1)
        idx = target * self.num_classes + pred
        self.value += np.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.value[:] = 0

    @property
    def tp(self) -> np.ndarray:
        return np.diag(self.value)

    @property
    def count(self) -> np.ndarray:
        return self.value.sum(axis=1)

    @property
    def predicted(self) -> np.ndarray:
        return self.value.sum(axis=0)

    def per_class_accuracy(self) -> np.ndarray:
        return 100.0 * self.tp / np.maximum(self.count, 1)

    def overall_accuracy(self) -> float:
        return float(100.0 * self.tp.sum() / max(self.value.sum(), 1))

    def mean_class_accuracy(self) -> float:
        """Macro average of the per-class recall; a class absent from the data contributes 0 to the mean."""
        return float(self.per_class_accuracy().mean())

    def macro_f1(self) -> float:
        tp = self.tp.astype(np.float64)
        precision = tp / np.maximum(self.predicted, 1)
        recall = tp / np.maximum(self.count, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        return float(100.0 * f1.mean())

    def summary(self) -> dict:
        return {
            "oacc": self.overall_accuracy(),
            "macc": self.mean_class_accuracy(),
            "f1": self.macro_f1(),
            "per_class_acc": self.per_class_accuracy().tolist(),
        }
