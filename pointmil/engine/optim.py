"""Optimisers and learning-rate schedules."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(model_or_params, opt_cfg: dict) -> torch.optim.Optimizer:
    """``{name: adamw|adam|sgd, lr, weight_decay, momentum, nesterov, filter_bias_and_bn}``.

    With ``filter_bias_and_bn: true`` (the OpenPoints default, used by the OpenPoints recipe)
    biases and all other 1-D parameters, i.e. the normalisation weights, get no weight decay;
    this needs a model rather than a parameter list.
    """
    name = opt_cfg["name"].lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if isinstance(model_or_params, nn.Module):
        if wd and opt_cfg.get("filter_bias_and_bn", False):
            decay, no_decay = [], []
            for pname, param in model_or_params.named_parameters():
                if param.requires_grad:
                    (no_decay if param.ndim == 1 or pname.endswith(".bias") else decay).append(param)
            params = [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": wd}]
        else:
            params = model_or_params.parameters()
    else:
        if opt_cfg.get("filter_bias_and_bn", False):
            raise ValueError("filter_bias_and_bn needs the model, not a parameter list")
        params = model_or_params
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name in ("sgd", "momentum"):
        return torch.optim.SGD(params, lr=lr, momentum=float(opt_cfg.get("momentum", 0.9)), weight_decay=wd, nesterov=bool(opt_cfg.get("nesterov", False)))
    raise KeyError(f"Unknown optimizer '{name}'")


class CosineSchedule:
    """Per-epoch cosine decay from the base learning rate to ``min_lr`` over ``t_max`` epochs.

    ``lr(t) = min_lr + 0.5 (lr0 - min_lr) (1 + cos(pi t / t_max))`` for ``t < t_max``
    and ``min_lr`` afterwards; optional linear warm-up for ``warmup_epochs``.
    This is the closed form of ``CosineAnnealingLR`` (original code) and of the
    cosine scheduler used by OpenPoints when stepped once per epoch.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, t_max: int, min_lr: float = 0.0, warmup_epochs: int = 0):
        self.optimizer = optimizer
        self.t_max = max(int(t_max), 1)
        self.min_lr = float(min_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = 0

    def lr_at(self, base_lr: float, epoch: int) -> float:
        if self.warmup_epochs and epoch < self.warmup_epochs:
            return base_lr * (epoch + 1) / self.warmup_epochs
        t = epoch - self.warmup_epochs
        if t >= self.t_max:
            return self.min_lr
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t / self.t_max))

    def step(self, epoch: int) -> None:
        """Set the learning rate for ``epoch`` (0-indexed epoch about to run)."""
        self.last_epoch = epoch
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = self.lr_at(base, epoch)

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = state["last_epoch"]
        self.base_lrs = state["base_lrs"]


class StepSchedule:
    """Multiply the learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer, step_size: int, gamma: float):
        self.optimizer, self.step_size, self.gamma = optimizer, int(step_size), float(gamma)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * self.gamma ** (epoch // self.step_size)

    def state_dict(self):
        return {"base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.base_lrs = state["base_lrs"]


class ConstantSchedule:
    """Keep the learning rate fixed."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, epoch: int) -> None:
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def build_scheduler(optimizer, sched_cfg: dict, epochs: int):
    """``{name: cosine, t_max, min_lr, warmup_epochs}`` | ``{name: step, step_size, gamma}`` | ``{name: none}``."""
    name = sched_cfg.get("name", "cosine").lower()
    if name == "cosine":
        return CosineSchedule(optimizer, t_max=int(sched_cfg.get("t_max", epochs)), min_lr=float(sched_cfg.get("min_lr", 0.0)), warmup_epochs=int(sched_cfg.get("warmup_epochs", 0)))
    if name == "step":
        return StepSchedule(optimizer, sched_cfg["step_size"], sched_cfg.get("gamma", 0.5))
    if name in ("none", "constant"):
        return ConstantSchedule(optimizer)
    raise KeyError(f"Unknown scheduler '{name}'")
