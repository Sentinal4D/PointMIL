"""Build a PointMIL model from a configuration dictionary.

Expected ``model`` section::

    model:
      backbone: {name: transformer, in_channels: 6, ...}
      pooling: {name: conjunctive, num_classes: 2, ...}

``pooling.num_features`` defaults to the backbone's ``out_channels``. The fused
CurveNet model of the original code (``backbone.name: curvenet_mil``) carries
its own pooling and takes ``pooling.name`` as its ``pooling`` argument.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from torch import nn

from pointmil.backbones import build_backbone
from pointmil.backbones.curvenet import CurveNetMIL
from pointmil.models.pointmil import PointMIL
from pointmil.pooling import build_pooling


def build_model(model_cfg: Mapping[str, Any]) -> nn.Module:
    """Instantiate the model described by ``model_cfg``."""
    cfg = copy.deepcopy(dict(model_cfg))
    backbone_cfg = dict(cfg["backbone"])
    pooling_cfg = dict(cfg["pooling"])
    backbone_name = backbone_cfg.pop("name")
    pooling_name = pooling_cfg.pop("name")

    if backbone_name == "curvenet_mil":
        # Original fused CurveNet + MIL head (pooling is part of the backbone class).
        return CurveNetMIL(pooling=pooling_name, num_classes=pooling_cfg["num_classes"], **backbone_cfg)

    if backbone_name == "curvenet":
        raise KeyError("use backbone 'curvenet_mil' (fused CurveNet + MIL head, IntrA/RBC) or 'openpoints_curvenet' (ATLAS-1)")
    backbone = build_backbone(backbone_name, **backbone_cfg)
    pooling_cfg.setdefault("num_features", backbone.out_channels)
    pooling = build_pooling(pooling_name, **pooling_cfg)
    return PointMIL(backbone, pooling)
