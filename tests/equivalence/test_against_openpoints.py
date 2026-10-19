"""Numerical equivalence of the ported OpenPoints modules against the OpenPoints fork (GPU only)."""

from __future__ import annotations

import pytest
import torch

from pointmil.backbones.curvenet import OpenPointsCurveNet
from pointmil.backbones.dgcnn import OpenPointsDGCNN
from pointmil.backbones.pointmlp import pointmlp_elite
from pointmil.backbones.pointnet import OpenPointsPointNet
from pointmil.backbones.transformer import TransformerExtractor
from pointmil.pooling.openpoints_head import OpenPointsMILHead

torch.manual_seed(0)
DEV = "cuda"


def _inputs(in_channels: int, B: int = 2, N: int = 1024):
    pos = torch.randn(B, N, 3, device=DEV)
    if in_channels == 4:
        heights = pos[..., 1:2] - pos[..., 1:2].min(dim=1, keepdim=True)[0]
        x = torch.cat([pos, heights], -1)
    else:
        x = pos
    return pos, x


@pytest.mark.parametrize("in_channels", [3, 4])
def test_dgcnn(openpoints, in_channels):
    from openpoints.models.backbone.dgcnn import DGCNN

    original = DGCNN(in_channels=in_channels, channels=64, embed_dim=1024, n_blocks=5, k=20, is_mil=True).to(DEV).eval()
    ported = OpenPointsDGCNN(in_channels=in_channels).to(DEV).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    pos, x = _inputs(in_channels)
    with torch.no_grad():
        f_a = original.forward_cls_feat(pos, x.transpose(1, 2).contiguous())
        _, f_p = ported(x)
    assert torch.allclose(f_a, f_p, atol=1e-4), (f_a - f_p).abs().max()


@pytest.mark.parametrize("in_channels", [3, 4])
def test_pointnet(openpoints, in_channels):
    from openpoints.models.backbone.pointnet import PointNetEncoder

    original = PointNetEncoder(in_channels=in_channels).to(DEV).eval()
    ported = OpenPointsPointNet(in_channels=in_channels).to(DEV).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    pos, x = _inputs(in_channels)
    with torch.no_grad():
        f_a = original.forward_cls_feat(pos, x.transpose(1, 2).contiguous())
        _, f_p = ported(x)
    assert torch.allclose(f_a, f_p, atol=1e-4)


def test_pointmlp_elite(openpoints):
    from openpoints.models.backbone.pointmlp import PointMLPEncoder

    kw = dict(in_channels=3, embed_dim=32, groups=1, res_expansion=0.25, activation="relu", bias=False, use_xyz=False,
              normalize="anchor", dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
              k_neighbors=[24, 24, 24, 24], reducers=[1, 1, 1, 1])
    original = PointMLPEncoder(**kw).to(DEV).eval()
    ported = pointmlp_elite(in_channels=3).to(DEV).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    pos, x = _inputs(3)
    with torch.no_grad():
        f_a = original.forward_cls_feat(pos, x.transpose(1, 2).contiguous())  # points in CUDA-FPS order
        p_p, f_p = ported(x)  # points in input order
        # Recover the CUDA farthest point sampling permutation (starts at index 0, same for every stage).
        from openpoints.models.layers import furthest_point_sample

        perm = furthest_point_sample(pos.contiguous(), pos.shape[1]).long()
    f_p_perm = torch.gather(f_p, 2, perm.unsqueeze(1).expand(-1, f_p.shape[1], -1))
    assert torch.allclose(f_a, f_p_perm, atol=1e-3), (f_a - f_p_perm).abs().max()
    assert torch.allclose(f_a.mean(-1), f_p.mean(-1), atol=1e-4)


def test_curvenet(openpoints):
    from openpoints.models.backbone.pointnext import CurveNetNoDown

    original = CurveNetNoDown(num_classes=3, is_mil=True).to(DEV).eval()
    ported = OpenPointsCurveNet(num_classes=3).to(DEV).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    pos, x = _inputs(3)
    with torch.no_grad():
        f_a = original.forward_cls_feat({"pos": pos, "x": x.transpose(1, 2).contiguous()})
        _, f_p = ported(x)
    assert torch.allclose(f_a, f_p, atol=1e-3), (f_a - f_p).abs().max()


def test_transformer(openpoints):
    from openpoints.models.backbone.pointnext import MedPTFeatureExtractor

    original = MedPTFeatureExtractor(in_channels=3).to(DEV).eval()
    ported = TransformerExtractor(in_channels=3, coord_dims=3).to(DEV).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    pos, x = _inputs(3)
    with torch.no_grad():
        f_a = original.forward_cls_feat({"pos": pos, "x": x.transpose(1, 2).contiguous()})
        _, f_p = ported(x)
    assert torch.allclose(f_a, f_p, atol=1e-3), (f_a - f_p).abs().max()


@pytest.mark.parametrize("pooling,enc_type,layout", [
    ("additive", "other", "linear"), ("instance", "other", "linear"), ("conjunctive", "other", "linear"),
    ("gap", "other", "linear"), ("attention", "other", "linear"),
    ("additive", "dgcnn", "mlp_bn_points"), ("conjunctive", "dgcnn", "mlp_bn_points"), ("instance", "dgcnn", "mlp_bn_points"),
])
def test_mil_head(openpoints, pooling, enc_type, layout):
    from openpoints.models.classification.cls_base import ClsHead

    d = 1024 if enc_type == "dgcnn" else 256
    C, N, B = 3, 1024, 2
    in_channels = 2 * d if pooling == "attention" else d
    original = ClsHead(num_classes=C, in_channels=in_channels, mlps=[512, 256], norm_args={"norm": "bn1d"},
                       is_mil=True, pooling=pooling, enc_type=enc_type, inference=True).to(DEV).eval()
    ported = OpenPointsMILHead(num_features=d, num_classes=C, pooling=pooling, instance_head=layout, num_points=N).to(DEV).eval()
    sd = {k: v for k, v in original.state_dict().items() if k in ported.state_dict()}
    missing, unexpected = ported.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    feats = torch.randn(B, d, N, device=DEV)
    with torch.no_grad():
        out_a = original(feats)
        out_p = ported(None, feats)
    for key in ("bag_logits", "interpretation"):
        assert torch.allclose(out_a[key], out_p[key], atol=1e-4), key
