"""Attention pooling (Eq. 2): attention-weighted mean of the point features.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: pools points instead of time steps, with 3D positional encoding
and contextual attention (Eq. 6).
"""

from __future__ import annotations

import torch

from pointmil.pooling._common import MILPoolingBase, build_attention_head, build_bag_classifier


class AttentionPooling(MILPoolingBase):
    """Gated attention over points, then a classifier on the pooled embedding.

    The interpretation is the attention weight of each point, repeated over the
    class axis (it is not class specific).
    """

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)
        self.attention_head = build_attention_head(num_features, self.attention_hidden)
        self.bag_out = build_bag_classifier(
            num_features, num_classes, self.dropout, self.non_linear, self.non_linear_v1
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
        features = self.encode(xyz, features)
        features_t = features.transpose(2, 1)  # [B, N, d]
        attn = self.attend(xyz, features_t)  # [B, N, 1]
        bag_embedding = torch.mean(features_t * attn, dim=1)
        bag_logits = self.bag_out(bag_embedding)
        return {
            "bag_logits": bag_logits,
            "interpretation": attn.repeat(1, 1, self.num_classes).transpose(2, 1),
            "attention": attn,
        }
