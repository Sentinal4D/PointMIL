"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    With ``deterministic=True`` cuDNN autotuning is disabled and deterministic
    convolution algorithms are requested. Operations that remain
    non-deterministic on GPU (notably ``torch.topk`` tie-breaking in the k-NN
    graphs and scatter-style gathers) are documented in REPRODUCIBILITY.md.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses (e.g. DataLoader workers started later), not this interpreter
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader`` so that each worker gets a derived seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
