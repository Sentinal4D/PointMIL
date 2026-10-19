"""Sinusoidal 3D positional encoding added to point features.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: encodes 3D point coordinates instead of time-step indices.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding3D(nn.Module):
    """Sinusoidal positional encoding of ``(x, y, z)`` coordinates.

    For each axis the standard transformer sine/cosine encoding is computed and
    the three encodings are summed. The result is added to the point features.

    Args:
        num_features: feature dimension ``d`` (must be even).
    """

    def __init__(self, num_features: int):
        super().__init__()
        if num_features % 2:
            raise ValueError(f"positional encoding needs an even feature width, got {num_features}")
        self.num_features = num_features
        div_term = torch.exp(torch.arange(0, num_features, 2) * -(math.log(10000.0) / num_features))
        self.register_buffer("div_term", div_term)

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Add the encoding of ``coords`` ``[B, N, 3]`` to ``features`` ``[B, d, N]``."""
        batch_size, num_points, _ = coords.shape
        pe = torch.zeros(batch_size, num_points, self.num_features, device=coords.device, dtype=features.dtype)
        for i in range(3):
            coord = coords[:, :, i].unsqueeze(-1)
            pe[:, :, 0::2] += torch.sin(coord * self.div_term)
            pe[:, :, 1::2] += torch.cos(coord * self.div_term)
        return features + pe.transpose(2, 1)
