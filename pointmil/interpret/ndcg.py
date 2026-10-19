"""Normalised Discounted Cumulative Gain at n (NDCG@n).

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).

From MILLET (Early et al., ICLR 2024). ``n`` is the number of positive points in
the ground truth annotation; the ``n`` highest-scoring points are ranked and the
gain is discounted by log2 of the rank.
"""

from __future__ import annotations

import math

import torch


def calculate_ndcg_at_n(instance_importance_scores: torch.Tensor, instance_labels: torch.Tensor) -> float:
    """NDCG@n for one cloud: ``scores`` ``[N]`` and binary ``labels`` ``[N]`` (1 = discriminatory point)."""
    n = int((instance_labels == 1).sum().item())
    if n == 0:
        raise ValueError("Trying to assess interpretability with no discriminatory instances")
    top_n = torch.topk(instance_importance_scores.detach().cpu(), n)[1]
    dcg = 0.0
    norm = 0.0
    labels = instance_labels.detach().cpu()
    for i, order_idx in enumerate(top_n):
        rel = labels[order_idx].item()
        dcg += rel / math.log2(i + 2)
        norm += 1.0 / math.log2(i + 2)
    return dcg / norm
