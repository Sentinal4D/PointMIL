"""MIL head used by the OpenPoints training path (ATLAS-1 and ScanObjectNN results).

This reproduces the ``ClsHead`` MIL branch added to the OpenPoints fork. It has
no positional encoding and no contextual attention, uses dropout 0.5, and comes
in two instance-classifier layouts that were both used for paper numbers:

* ``linear``: ``Dropout(0.5) -> Linear(in, C)``.
* ``mlp_bn_points``: ``Linear(in, 512) -> BatchNorm1d(N) -> ReLU -> Dropout ->
  Linear(512, 256) -> BatchNorm1d(N) -> ReLU -> Dropout -> Linear(256, C)``,
  where the BatchNorm runs over the point axis (``N`` points), a quirk of the
  original code that ties the head to a fixed number of points.

Submodule names follow the original so that OpenPoints checkpoints load.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: re-implemented as the MIL branch of the OpenPoints classification
head (OpenPoints, MIT) for point clouds.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class OpenPointsMILHead(nn.Module):
    """OpenPoints-style MIL pooling head.

    Args:
        num_features: per-point feature dimension of the encoder output.
        num_classes: number of classes.
        pooling: one of ``instance``, ``additive``, ``conjunctive``, ``attention``, ``gap``.
        instance_head: ``linear`` or ``mlp_bn_points``.
        num_points: points per cloud (needed by ``mlp_bn_points``).
        mlp_dims: hidden widths of the MLP heads.
        dropout: dropout rate.
        attention_hidden: hidden width of the gated attention MLP.
        attention_classifier: bag classifier of attention pooling. ``mlp`` (default, current
            OpenPoints fork): MLP on the concatenated mean and max of the attention-weighted
            features. ``linear``: dropout and one linear layer (``mil_head_att``) on their mean,
            the layout of the ATLAS-1 DGCNN, PointNet and PointMLPElite and the ScanObjectNN
            PointMLPElite attention checkpoints.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        pooling: str = "additive",
        instance_head: str = "linear",
        num_points: int = 1024,
        mlp_dims: Sequence[int] = (512, 256),
        dropout: float = 0.5,
        attention_hidden: int = 8,
        attention_classifier: str = "mlp",
    ):
        super().__init__()
        if pooling not in {"instance", "additive", "conjunctive", "attention", "gap"}:
            raise ValueError(f"Unsupported pooling '{pooling}'")
        if attention_classifier not in {"mlp", "linear"}:
            raise ValueError(f"Unknown attention_classifier '{attention_classifier}'")
        self.attention_classifier = attention_classifier
        self.num_features = num_features
        self.num_classes = num_classes
        self.pooling = pooling
        self.instance_head = instance_head
        self.num_points = num_points

        if instance_head == "linear":
            self.mil_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_features, num_classes, bias=True))
        elif instance_head == "mlp_bn_points":
            dims = [num_features, *mlp_dims]
            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(
                    nn.Sequential(
                        nn.Linear(dims[i], dims[i + 1], bias=False),
                        nn.BatchNorm1d(num_points),
                        nn.ReLU(inplace=True),
                    )
                )
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Sequential(nn.Linear(dims[-1], num_classes, bias=True)))
            self.mil_head = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown instance_head '{instance_head}'")

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
            nn.Sigmoid(),
        )
        if pooling == "gap":
            self.bag_out = nn.Linear(num_features, num_classes)
        if pooling == "attention" and attention_classifier == "linear":
            self.mil_head_att = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_features, num_classes, bias=True))
        if pooling == "attention" and attention_classifier == "mlp":
            # Bag classifier on the concatenated attention-weighted mean and max embedding.
            dims = [2 * num_features, *mlp_dims]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(
                    nn.Sequential(nn.Linear(dims[i], dims[i + 1], bias=False), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True))
                )
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Sequential(nn.Linear(dims[-1], num_classes, bias=True)))
            self.head = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
        """``features`` is ``[B, d, N]``; ``xyz`` is unused but kept for a uniform interface."""
        features_t = features.transpose(2, 1)  # [B, N, d]
        if self.pooling == "gap":
            cam = self.bag_out.weight @ features
            return {"bag_logits": self.bag_out(features.mean(dim=-1)), "interpretation": cam}
        if self.pooling == "instance":
            instance_logits = self.mil_head(features_t)
            return {
                "bag_logits": instance_logits.mean(dim=1),
                "interpretation": instance_logits.transpose(2, 1),
                "instance_logits": instance_logits.transpose(2, 1),
            }
        attention = self.attention_head(features_t)  # [B, N, 1]
        if self.pooling == "additive":
            instance_logits = self.mil_head(features_t * attention)
            return {
                "bag_logits": instance_logits.mean(dim=1),
                "interpretation": (instance_logits * attention).transpose(2, 1),
                "instance_logits": instance_logits.transpose(2, 1),
                "attention": attention,
            }
        if self.pooling == "conjunctive":
            instance_logits = self.mil_head(features_t)
            weighted = instance_logits * attention
            return {
                "bag_logits": weighted.mean(dim=1),
                "interpretation": weighted.transpose(2, 1),
                "instance_logits": instance_logits.transpose(2, 1),
                "attention": attention,
            }
        # attention pooling
        weighted = features_t * attention
        if self.attention_classifier == "linear":
            bag_logits = self.mil_head_att(weighted.mean(dim=1))
        else:
            bag_logits = self.head(torch.cat([weighted.mean(dim=1), weighted.max(dim=1)[0]], dim=1))
        return {
            "bag_logits": bag_logits,
            "interpretation": attention.repeat(1, 1, self.num_classes).transpose(2, 1),
            "attention": attention,
        }
