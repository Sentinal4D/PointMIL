"""MIL pooling heads.

All heads take ``(xyz [B, N, 3], features [B, d, N])`` and return a dict with at
least ``bag_logits`` ``[B, C]`` and ``interpretation`` ``[B, C, N]``.
"""

from __future__ import annotations

from pointmil.pooling.additive import AdditivePooling
from pointmil.pooling.attention import AttentionPooling
from pointmil.pooling.conjunctive import ConjunctivePooling
from pointmil.pooling.contextual import contextual_attention
from pointmil.pooling.gap import GlobalAveragePooling
from pointmil.pooling.instance import InstancePooling
from pointmil.pooling.openpoints_head import OpenPointsMILHead
from pointmil.pooling.positional import PositionalEncoding3D

POOLINGS = {
    "instance": InstancePooling,
    "attention": AttentionPooling,
    "additive": AdditivePooling,
    "conjunctive": ConjunctivePooling,
    "gap": GlobalAveragePooling,
    "openpoints": OpenPointsMILHead,
}


def build_pooling(name: str, **kwargs):
    """Instantiate a pooling head by registry name."""
    if name not in POOLINGS:
        raise KeyError(f"Unknown pooling '{name}'. Available: {sorted(POOLINGS)}")
    return POOLINGS[name](**kwargs)


__all__ = [
    "POOLINGS",
    "build_pooling",
    "InstancePooling",
    "AttentionPooling",
    "AdditivePooling",
    "ConjunctivePooling",
    "GlobalAveragePooling",
    "OpenPointsMILHead",
    "PositionalEncoding3D",
    "contextual_attention",
]
