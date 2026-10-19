"""Global average pooling with class activation maps (the CLAIM / CAM baseline).

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: class activation maps over points instead of time steps.
"""

from __future__ import annotations

import torch
from torch import nn

from pointmil.pooling._common import MILPoolingBase, build_bag_classifier


class GlobalAveragePooling(MILPoolingBase):
    """Mean-pool the point features, then a linear classifier.

    The interpretation is the class activation map: classifier weights applied to
    every point feature (Huang et al., CLAIM). It requires the linear classifier
    (``non_linear=False``).
    """

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)
        if self.non_linear or self.non_linear_v1:
            raise ValueError("gap pooling needs the linear classifier (non_linear=False) to compute class activation maps")
        self.bag_out = build_bag_classifier(
            num_features, num_classes, self.dropout, self.non_linear, self.non_linear_v1
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
        features = self.encode(xyz, features)
        if isinstance(self.bag_out, nn.Linear):
            cam = self.bag_out.weight @ features  # [B, C, N]
        else:
            raise RuntimeError("Class activation maps require the linear classifier (non_linear=False).")
        bag_embedding = features.mean(dim=-1)
        bag_logits = self.bag_out(bag_embedding)
        return {"bag_logits": bag_logits, "interpretation": cam}
