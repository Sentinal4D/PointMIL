"""Building blocks shared by the MIL pooling heads.

The layer layouts (including the BatchNorm over the point axis used by the
non-linear instance classifiers of the original code) are reproduced exactly so
that checkpoints trained with the original PointMIL implementation load into
these modules without any weight conversion.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Changes for PointMIL: the shared layers work on per-point features ``[B, d, N]``;
classifier variants of the original PointMIL code (including BatchNorm over the point
axis), 3D positional encoding and contextual attention hooks were added.
"""

from __future__ import annotations

from torch import nn

from pointmil.pooling.contextual import contextual_attention
from pointmil.pooling.positional import PositionalEncoding3D


def build_attention_head(num_features: int, hidden: int) -> nn.Sequential:
    """Gated attention MLP producing one weight in ``[0, 1]`` per point."""
    return nn.Sequential(
        nn.Linear(num_features, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )


def build_instance_classifier(
    num_features: int,
    num_classes: int,
    dropout: float,
    non_linear: bool,
    non_linear_v1: bool,
    num_points: int,
    narrow: bool = False,
    narrow_first_bias: bool = True,
) -> nn.Module:
    """Per-point classifier applied to ``[B, N, d]`` features.

    ``narrow`` selects the layout d -> d/2 -> d/4 -> C of the ModelNet40 PointNet conjunctive
    checkpoint (also used after ``raise_dimension``, with ``narrow_first_bias=False``); it takes
    precedence over ``non_linear``.

    ``non_linear_v1`` takes precedence over ``non_linear``, as in the original
    code. The ``non_linear`` variant normalises over the point axis
    (``BatchNorm1d(num_points)``), which ties the module to a fixed number of
    points; this quirk is preserved for checkpoint compatibility.
    """
    if non_linear_v1:
        return nn.Sequential(
            nn.Linear(num_features, num_features // 2, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_classes, bias=True),
        )
    if narrow:
        return nn.Sequential(
            nn.Linear(num_features, num_features // 2, bias=narrow_first_bias),
            nn.BatchNorm1d(num_points),
            nn.ReLU(True),
            nn.Linear(num_features // 2, num_features // 4),
            nn.BatchNorm1d(num_points),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 4, num_classes),
        )
    if non_linear:
        return nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.BatchNorm1d(num_points),
            nn.ReLU(True),
            nn.Linear(num_features * 2, num_features // 2),
            nn.BatchNorm1d(num_points),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_classes),
        )
    return nn.Linear(num_features, num_classes)


def build_bag_classifier(
    num_features: int,
    num_classes: int,
    dropout: float,
    non_linear: bool,
    non_linear_v1: bool,
) -> nn.Module:
    """Classifier applied to a pooled bag embedding ``[B, d]``."""
    if non_linear_v1:
        return nn.Sequential(
            nn.Linear(num_features, num_features // 2, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_classes, bias=True),
        )
    if non_linear:
        return nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.BatchNorm1d(num_features * 2),
            nn.ReLU(True),
            nn.Linear(num_features * 2, num_features // 2),
            nn.BatchNorm1d(num_features // 2),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_classes),
        )
    return nn.Linear(num_features, num_classes)


class MILPoolingBase(nn.Module):
    """Common state and helpers for the MIL pooling heads.

    Args:
        num_features: dimension ``d`` of the per-point features.
        num_classes: number of classes ``C``.
        dropout: dropout rate inside the classifiers.
        attention_hidden: hidden size of the gated attention MLP.
        non_linear: use the non-linear classifier variant.
        non_linear_v1: use the two-layer non-linear classifier variant.
        positional_encoding: add a sinusoidal 3D positional encoding to the
            features before pooling.
        contextual_attention: smooth attention weights over the ``context_k``
            nearest neighbours (Eq. 6).
        context_k: neighbourhood size for contextual attention. ``0`` disables it.
        num_points: number of points per cloud (only needed by the
            ``non_linear`` instance classifiers).
        non_linear_narrow: use the narrowing instance classifier (d -> d/2 -> d/4 -> C).
        narrow_first_bias: bias on the first layer of the narrowing classifier.
        raise_dimension: raise the features from d to 4d with two 1x1 convolutions before
            attention and the instance classifier (ModelNet40 Transformer runs; conjunctive
            pooling only, as in the original code).
    """

    supports_raise_dimension = False

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.1,
        attention_hidden: int = 8,
        non_linear: bool = False,
        non_linear_v1: bool = False,
        positional_encoding: bool = True,
        contextual_attention: bool = False,
        context_k: int = 12,
        num_points: int = 1024,
        non_linear_narrow: bool = False,
        narrow_first_bias: bool = True,
        raise_dimension: bool = False,
    ):
        super().__init__()
        if raise_dimension and not self.supports_raise_dimension:
            raise ValueError(f"raise_dimension is only supported by conjunctive pooling, not {type(self).__name__}")
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_hidden = attention_hidden
        self.non_linear = non_linear
        self.non_linear_v1 = non_linear_v1
        self.positional_encoding = positional_encoding
        self.contextual_attention = contextual_attention
        self.context_k = context_k
        self.num_points = num_points
        self.non_linear_narrow = non_linear_narrow
        self.narrow_first_bias = narrow_first_bias
        self.raise_dimension = raise_dimension
        # width of the features seen by the attention head and the instance classifier
        self.feature_dim = 4 * num_features if raise_dimension else num_features
        if positional_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)
        if raise_dimension:
            self.conv_raise = nn.Sequential(
                nn.Conv1d(num_features, num_features * 2, kernel_size=1, bias=False),
                nn.BatchNorm1d(num_features * 2),
                nn.ReLU(True),
                nn.Conv1d(num_features * 2, num_features * 4, kernel_size=1, bias=False),
                nn.BatchNorm1d(num_features * 4),
                nn.ReLU(True),
            )

    def encode(self, xyz, features):
        """Optionally add the positional encoding and raise the dimension. ``features`` is ``[B, d, N]``."""
        if self.positional_encoding:
            features = self.pos_enc(xyz, features)
        if self.raise_dimension:
            features = self.conv_raise(features)
        return features

    def attend(self, xyz, features_t):
        """Attention weights ``[B, N, 1]`` for ``features_t`` ``[B, N, d]``."""
        attn = self.attention_head(features_t)
        if self.contextual_attention:
            attn = contextual_attention(attn, xyz, self.context_k)
        return attn

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, num_classes={self.num_classes}, "
            f"positional_encoding={self.positional_encoding}, "
            f"contextual_attention={self.contextual_attention}, context_k={self.context_k}"
        )
