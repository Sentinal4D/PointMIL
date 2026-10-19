"""Per-point feature extractors.

Every backbone maps ``[B, N, C_in]`` points (xyz first) to a tuple
``(coords [B, N, *], features [B, d, N])`` and exposes ``out_channels``.
"""

from __future__ import annotations

from pointmil.backbones.curvenet import CurveNetMIL, OpenPointsCurveNet
from pointmil.backbones.dgcnn import DGCNNExtractor, OpenPointsDGCNN
from pointmil.backbones.pointmlp import PointMLPEncoder, pointmlp_elite
from pointmil.backbones.pointnet import OpenPointsPointNet, PointNetExtractor
from pointmil.backbones.transformer import TransformerBlock, TransformerExtractor

BACKBONES = {
    # Original PointMIL implementations (IntrA, RBC, ModelNet40 rows, Table 1)
    "transformer": TransformerExtractor,
    "pointnet": PointNetExtractor,
    "dgcnn": DGCNNExtractor,
    # OpenPoints implementations (ATLAS-1 and ScanObjectNN rows)
    "openpoints_pointnet": OpenPointsPointNet,
    "openpoints_dgcnn": OpenPointsDGCNN,
    "openpoints_curvenet": OpenPointsCurveNet,
    "pointmlp": PointMLPEncoder,
    "pointmlp_elite": pointmlp_elite,
}


def build_backbone(name: str, **kwargs):
    """Instantiate a backbone by registry name."""
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)


__all__ = [
    "BACKBONES",
    "build_backbone",
    "TransformerBlock",
    "TransformerExtractor",
    "PointNetExtractor",
    "OpenPointsPointNet",
    "DGCNNExtractor",
    "OpenPointsDGCNN",
    "CurveNetMIL",
    "OpenPointsCurveNet",
    "PointMLPEncoder",
    "pointmlp_elite",
]
