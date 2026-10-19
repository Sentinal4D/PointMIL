"""Numerical equivalence of the ported modules against the original PointMIL code (repo A, HEAD)."""

from __future__ import annotations

import argparse
import importlib

import pytest
import torch

from pointmil.backbones.dgcnn import DGCNNExtractor
from pointmil.backbones.pointnet import PointNetExtractor
from pointmil.backbones.transformer import TransformerExtractor
from pointmil.pooling import build_pooling

torch.manual_seed(0)
ATOL = 1e-5


def _args(**kw):
    base = dict(
        contextual_attention=False, context_k=12, raise_dimension=False, non_lin_1=False,
        k=20, emb_dims=1024, use_norm=True, num_K=[32, 64], dim_k=32, num_heads=8,
        channel_dim=[128, 256], channel_raise=[64, 256], dropout=0.1, num_points=1024,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("pooling", ["instance", "attention", "additive", "conjunctive", "gap"])
@pytest.mark.parametrize("variant", ["linear_ctx", "linear_noctx_nope", "non_linear", "non_lin_1"])
def test_pooling_heads(repo_a_models, pooling, variant):
    mp = importlib.import_module("models.mil_pooling")
    d, C, N, B = 256, 3, 1024, 2
    cfg = dict(linear_ctx=dict(non_linear=False, non_lin_1=False, pe=True, ctx=True),
               linear_noctx_nope=dict(non_linear=False, non_lin_1=False, pe=False, ctx=False),
               non_linear=dict(non_linear=True, non_lin_1=False, pe=True, ctx=True),
               non_lin_1=dict(non_linear=False, non_lin_1=True, pe=True, ctx=False))[variant]
    if pooling == "gap" and (cfg["non_linear"] or cfg["non_lin_1"]):
        pytest.skip("CAM needs the linear classifier")
    a_cls = dict(instance=mp.InstancePooling, attention=mp.AttentionPooling, additive=mp.AdditivePooling,
                 conjunctive=mp.ConjunctivePooling, gap=mp.GlobalAveragePooling)[pooling]
    args = _args(contextual_attention=cfg["ctx"], context_k=12, non_lin_1=cfg["non_lin_1"])
    original = a_cls(num_features=d, num_classes=C, dropout=0.1, heads=8, non_linear=cfg["non_linear"],
                     apply_pos_encoding=cfg["pe"], args=args).eval()
    ported = build_pooling(pooling, num_features=d, num_classes=C, dropout=0.1, attention_hidden=8,
                           non_linear=cfg["non_linear"], non_linear_v1=cfg["non_lin_1"],
                           positional_encoding=cfg["pe"], contextual_attention=cfg["ctx"], context_k=12, num_points=N).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    xyz = torch.randn(B, N, 3)
    feats = torch.randn(B, d, N)
    with torch.no_grad():
        out_a = original((xyz, feats))
        out_p = ported(xyz, feats)
    for key in ("bag_logits", "interpretation"):
        assert torch.allclose(out_a[key], out_p[key], atol=ATOL), key


@pytest.mark.parametrize("use_norm", [True, False])
def test_transformer_extractor(repo_a_models, use_norm):
    mu = importlib.import_module("models.model_utils")
    original = mu.MedPTFeatureExtractor(use_norm=use_norm, num_K=[32, 64], dim_k=32, head=8,
                                        channel_dim=[128, 256], channel_raise=[64, 256], num_points=1024).eval()
    ported = TransformerExtractor(in_channels=6 if use_norm else 3, coord_dims=3).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 1024, 6 if use_norm else 3)
    with torch.no_grad():
        xyz_a, f_a = original(x)
        xyz_p, f_p = ported(x)
    assert torch.allclose(xyz_a, xyz_p) and torch.allclose(f_a, f_p, atol=1e-4), (f_a - f_p).abs().max()


def test_pointnet_extractor(repo_a_models):
    pn = importlib.import_module("models.backbones.pointnet")
    original = pn.PointNetEncoder(normal_channel=False).eval()
    ported = PointNetExtractor().eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 1024, 6)
    with torch.no_grad():
        xyz_a, f_a = original(x)
        xyz_p, f_p = ported(x)
    assert torch.allclose(xyz_a, xyz_p) and torch.allclose(f_a, f_p, atol=1e-4)


def test_dgcnn_extractor(repo_a_models):
    dg = importlib.import_module("models.backbones.dgcnn")
    original = dg.DGCNN(_args(k=20, emb_dims=1024)).eval()
    ported = DGCNNExtractor(k=20, emb_dims=1024).eval()
    missing, unexpected = ported.load_state_dict(original.state_dict(), strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 1024, 6)
    with torch.no_grad():
        xyz_a, f_a = original(x)
        xyz_p, f_p = ported(x)
    assert torch.allclose(xyz_a, xyz_p) and torch.allclose(f_a, f_p, atol=1e-4)
