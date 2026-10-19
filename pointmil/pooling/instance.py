"""Instance pooling (Eq. 1): classify every point, average the predictions.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: classifies points instead of time steps, with optional 3D
positional encoding and the PointMIL classifier variants.
"""

from __future__ import annotations

import torch

from pointmil.pooling._common import MILPoolingBase, build_instance_classifier


class InstancePooling(MILPoolingBase):
    """Mean of per-point class logits. Interpretation = per-point logits."""

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)
        self.node_out = build_instance_classifier(
            num_features, num_classes, self.dropout, self.non_linear, self.non_linear_v1, self.num_points,
            narrow=self.non_linear_narrow, narrow_first_bias=self.narrow_first_bias,
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
        features = self.encode(xyz, features)
        instance_logits = self.node_out(features.transpose(2, 1))  # [B, N, C]
        bag_logits = torch.mean(instance_logits, dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": instance_logits.transpose(2, 1),
            "instance_logits": instance_logits.transpose(2, 1),
        }
