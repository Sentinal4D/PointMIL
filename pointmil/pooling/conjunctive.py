"""Conjunctive pooling (Eq. 4): attention weights and point logits computed independently.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: pools points instead of time steps, with 3D positional encoding,
contextual attention (Eq. 6), the PointMIL classifier variants and optional dimension
raising.
"""

from __future__ import annotations

import torch

from pointmil.pooling._common import (
    MILPoolingBase,
    build_attention_head,
    build_instance_classifier,
)


class ConjunctivePooling(MILPoolingBase):
    """Mean over points of (attention weight x per-point logits).

    Interpretation = attention-weighted per-point logits (class specific).
    """

    supports_raise_dimension = True

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)
        self.node_out = build_instance_classifier(
            self.feature_dim, num_classes, self.dropout, self.non_linear, self.non_linear_v1, self.num_points,
            narrow=self.non_linear_narrow or self.raise_dimension,
            narrow_first_bias=self.narrow_first_bias and not self.raise_dimension,
        )
        self.attention_head = build_attention_head(self.feature_dim, self.attention_hidden)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
        features = self.encode(xyz, features)
        features_t = features.transpose(2, 1)  # [B, N, d]
        attn = self.attend(xyz, features_t)  # [B, N, 1]
        instance_logits = self.node_out(features_t)  # [B, N, C]
        weighted = instance_logits * attn
        bag_logits = torch.mean(weighted, dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted.transpose(2, 1),
            "instance_logits": instance_logits.transpose(2, 1),
            "attention": attn,
        }
