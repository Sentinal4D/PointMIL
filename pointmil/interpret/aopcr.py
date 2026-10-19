"""Area Over the Perturbation Curve relative to Random (AOPCR).

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification). Changes for PointMIL:
points are removed from the cloud (the bag shrinks) instead of being masked, and
the random baseline uses a fixed seed. Models whose classifier normalises over a
fixed number of points (``non_linear`` heads, OpenPoints ``mlp_bn_points`` heads)
cannot take a smaller cloud; for them each removed point is replaced by a copy of
a randomly chosen remaining point, so the cloud keeps its size but loses the
removed geometry. All Table 1 models accept smaller clouds and use removal.

Given a model and a batch of point clouds, the most important points according
to the interpretation are removed in steps of ``step`` (fraction of the cloud) up
to ``stop``; the drop of the predicted-class logit is integrated over the steps
and compared with the same procedure under ``n_random`` random orderings.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _interpret(model: nn.Module, output: dict) -> torch.Tensor:
    if hasattr(model, "interpret"):
        return model.interpret(output)
    return output["interpretation"]


def calculate_aopcr(
    model: nn.Module,
    bags: torch.Tensor,
    stop: float = 0.5,
    step: float = 0.05,
    n_random: int = 3,
    seed: int = 72,
    batch_interpretations: torch.Tensor | None = None,
    target_class: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AOPCR for a batch of clouds ``bags`` ``[B, N, C_in]``.

    Returns ``(aopcr [B], perturbation_curve [B, steps], random_curve [B, steps])``.
    Interpretations default to the model's interpretation of the predicted class
    (or of ``target_class`` when given).
    """
    keep_size = not accepts_fewer_points(model, bags)
    with torch.no_grad():
        out = model(bags)
    bag_logits = out["bag_logits"].detach().cpu()
    pred = bag_logits.argmax(dim=1) if target_class is None else target_class.cpu()
    orig_logits = bag_logits[torch.arange(len(bags)), pred]
    if batch_interpretations is None:
        interp = _interpret(model, out).detach().cpu()
        batch_interpretations = interp[torch.arange(len(bags)), pred, :]
    aopc, pc = _calculate_aopc(model, bags, orig_logits, pred, batch_interpretations, stop, step, keep_size, seed)
    r_aopc, r_pc = _calculate_random_aopc(model, bags, orig_logits, pred, stop, step, n_random, seed, keep_size)
    return aopc - r_aopc, pc, r_pc


def accepts_fewer_points(model: nn.Module, bags: torch.Tensor) -> bool:
    """Whether ``model`` runs on a cloud with fewer points than ``bags`` has.

    False for heads that normalise over a fixed number of points (BatchNorm over the point axis).
    """
    try:
        with torch.no_grad():
            model(bags[:, : max(1, bags.shape[1] - 8)])
        return True
    except (RuntimeError, IndexError):
        return False


def perturbation_curve(model: nn.Module, bags: torch.Tensor, interpretations: torch.Tensor, target_class: torch.Tensor, stop: float = 0.5, step: float = 0.05) -> torch.Tensor:
    """Perturbation curve ``[B, steps]`` (logit drop relative to the unperturbed logit) for given scores."""
    keep_size = not accepts_fewer_points(model, bags)
    with torch.no_grad():
        out = model(bags)
    orig = out["bag_logits"].detach().cpu()[torch.arange(len(bags)), target_class.cpu()]
    _, pc = _calculate_aopc(model, bags, orig, target_class.cpu(), interpretations.cpu(), stop, step, keep_size)
    return pc


def _steps(stop: float, step: float) -> np.ndarray:
    return np.linspace(1, stop, num=math.ceil((1 - stop) / step + 1))


def _calculate_aopc(model, bags, orig_logits, clzs, scores, stop, step, keep_size=False, seed=72) -> tuple[torch.Tensor, torch.Tensor]:
    n_bags, n_instances = bags.shape[0], bags.shape[1]
    steps = _steps(stop, step)
    batch_pc = torch.zeros((n_bags, len(steps)))
    batch_pc[:, 0] = orig_logits
    morf = torch.argsort(scores.cpu(), descending=True, stable=True)  # most relevant first
    gen = torch.Generator().manual_seed(seed)  # only used to pick replacement points when keep_size is True
    for step_idx, frac in enumerate(steps[1:]):
        n_to_remove = int((1 - frac) * n_instances)
        perturbed = torch.stack([_create_perturbed_bag(bags[i], morf[i], n_to_remove, keep_size, gen) for i in range(n_bags)])
        with torch.no_grad():
            new_logits = model(perturbed)["bag_logits"].detach().cpu()
        batch_pc[:, step_idx + 1] = new_logits[torch.arange(n_bags), clzs]
    batch_aopc = torch.zeros(n_bags)
    for k in range(1, len(steps)):
        batch_aopc += batch_pc[:, 0] - batch_pc[:, k]
    batch_aopc /= len(steps)
    batch_pc -= orig_logits.unsqueeze(1)
    return batch_aopc, batch_pc


def _calculate_random_aopc(model, bags, orig_logits, clzs, stop, step, n_repeats, seed, keep_size=False) -> tuple[torch.Tensor, torch.Tensor]:
    n_bags, n_instances = bags.shape[0], bags.shape[1]
    gen = torch.Generator().manual_seed(seed)
    aopcs, pcs = [], []
    for _ in range(n_repeats):
        random_scores = torch.rand((n_bags, n_instances), generator=gen)
        r_aopc, r_pc = _calculate_aopc(model, bags, orig_logits, clzs, random_scores, stop, step, keep_size, seed)
        aopcs.append(r_aopc)
        pcs.append(r_pc)
    return torch.stack(aopcs).mean(dim=0), torch.stack(pcs).mean(dim=0)


def _create_perturbed_bag(bag: torch.Tensor, bag_morf: torch.Tensor, n_to_remove: int, keep_size: bool = False,
                          gen: torch.Generator | None = None) -> torch.Tensor:
    """Remove the ``n_to_remove`` most relevant points from ``bag`` ``[N, C]``.

    With ``keep_size`` the removed points are replaced by copies of randomly chosen
    remaining points, so the result still has ``N`` points.
    """
    mask = torch.ones(len(bag), dtype=torch.bool, device=bag.device)
    mask[bag_morf[:n_to_remove].to(bag.device)] = False
    kept = bag[mask]
    if not keep_size or n_to_remove == 0:
        return kept
    fill = torch.randint(len(kept), (n_to_remove,), generator=gen).to(bag.device)
    return torch.cat([kept, kept[fill]], dim=0)
