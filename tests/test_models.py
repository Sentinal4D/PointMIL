"""Shape tests for every backbone x pooling combination and the contextual attention identity."""

import pytest
import torch

from pointmil.models import build_model
from pointmil.pooling import build_pooling
from pointmil.pooling.contextual import contextual_attention

torch.manual_seed(0)
N = 128
PTS6 = torch.randn(2, N, 6)
PTS3 = PTS6[..., :3]
PTS4 = torch.cat([PTS3, PTS3[..., 1:2]], -1)

A_BACKBONES = {
    "transformer": (dict(name="transformer", in_channels=6, coord_dims=3, num_k=[8, 8], channel_dim=[16, 32], channel_raise=[16, 32], dim_k=4, heads=4), PTS6, 32),
    "pointnet": (dict(name="pointnet"), PTS6, 1024),
    "dgcnn": (dict(name="dgcnn", k=8, emb_dims=64), PTS6, 64),
}
OP_BACKBONES = {
    "op_transformer4": (dict(name="transformer", in_channels=4, coord_dims=4, num_k=[8, 8], channel_dim=[16, 32], channel_raise=[16, 32], dim_k=4, heads=4), PTS4, 32),
    "op_dgcnn": (dict(name="openpoints_dgcnn", in_channels=4, channels=16, embed_dim=64, n_blocks=5, k=8), PTS4, 64),
    "op_pointnet": (dict(name="openpoints_pointnet", in_channels=4), PTS4, 1024),
    "op_curvenet": (dict(name="openpoints_curvenet", num_classes=3, k=8, num_points=N), PTS3, 1024),
    "pointmlp_elite": (dict(name="pointmlp_elite", in_channels=3), PTS3, 256),
}


@pytest.mark.parametrize("pooling", ["instance", "attention", "additive", "conjunctive", "gap"])
@pytest.mark.parametrize("backbone", list(A_BACKBONES))
def test_original_recipe_shapes(backbone, pooling):
    bcfg, x, d = A_BACKBONES[backbone]
    pcfg = dict(name=pooling, num_classes=3, dropout=0.1, attention_hidden=8, non_linear=False, non_linear_v1=False,
                positional_encoding=True, contextual_attention=pooling != "gap", context_k=4, num_points=N)
    model = build_model({"backbone": bcfg, "pooling": pcfg}).eval()
    with torch.no_grad():
        out = model(x)
    assert out["bag_logits"].shape == (2, 3)
    assert out["interpretation"].shape == (2, 3, N)
    assert model.backbone.out_channels == d


@pytest.mark.parametrize("pooling", ["instance", "attention", "additive", "conjunctive", "gap"])
@pytest.mark.parametrize("backbone", list(OP_BACKBONES))
@pytest.mark.parametrize("layout", ["linear", "mlp_bn_points"])
def test_openpoints_recipe_shapes(backbone, pooling, layout):
    bcfg, x, d = OP_BACKBONES[backbone]
    pcfg = dict(name="openpoints", pooling=pooling, num_classes=3, instance_head=layout, num_points=N, mlp_dims=[16, 8], dropout=0.5)
    model = build_model({"backbone": bcfg, "pooling": pcfg}).eval()
    with torch.no_grad():
        out = model(x)
    assert out["bag_logits"].shape == (2, 3)
    assert out["interpretation"].shape == (2, 3, N)


@pytest.mark.parametrize("backbone", list(OP_BACKBONES))
def test_openpoints_attention_linear_classifier(backbone):
    bcfg, x, d = OP_BACKBONES[backbone]
    pcfg = dict(name="openpoints", pooling="attention", num_classes=3, instance_head="linear", num_points=N,
                mlp_dims=[16, 8], dropout=0.5, attention_classifier="linear")
    model = build_model({"backbone": bcfg, "pooling": pcfg}).eval()
    assert not hasattr(model.pooling, "head") and model.pooling.mil_head_att[1].in_features == d
    with torch.no_grad():
        out = model(x)
        feats = model.backbone(x)[1].transpose(2, 1)
        att = model.pooling.attention_head(feats)
        expected = model.pooling.mil_head_att((feats * att).mean(dim=1))
    assert torch.allclose(out["bag_logits"], expected, atol=1e-5)


@pytest.mark.parametrize("pooling", ["instance", "attention", "additive", "conjunctive", "gap", "original"])
@pytest.mark.parametrize("instance_input", ["features", "features_with_global"])
def test_curvenet_mil_shapes(pooling, instance_input):
    model = build_model({"backbone": dict(name="curvenet_mil", k=8, instance_input=instance_input, num_points=N), "pooling": dict(name=pooling, num_classes=3)}).eval()
    with torch.no_grad():
        out = model(PTS3)
    assert out["bag_logits"].shape == (2, 3) and out["interpretation"].shape == (2, 3, N)


def test_raise_dimension_and_narrow_classifier_shapes():
    feats, xyz = torch.randn(2, 16, N), torch.randn(2, N, 3)
    raised = build_pooling("conjunctive", num_features=16, num_classes=5, non_linear=True, raise_dimension=True,
                           positional_encoding=False, num_points=N).eval()
    assert raised.node_out[0].in_features == 64 and raised.node_out[0].bias is None and raised.node_out[-1].in_features == 16
    narrow = build_pooling("conjunctive", num_features=16, num_classes=5, non_linear=True, non_linear_narrow=True,
                           positional_encoding=False, num_points=N).eval()
    assert [m.out_features for m in narrow.node_out if isinstance(m, torch.nn.Linear)] == [8, 4, 5]
    with torch.no_grad():
        for head in (raised, narrow):
            out = head(xyz, feats)
            assert out["bag_logits"].shape == (2, 5) and out["interpretation"].shape == (2, 5, N)
    with pytest.raises(ValueError):
        build_pooling("additive", num_features=16, num_classes=5, raise_dimension=True)


def test_contextual_attention_k0_is_identity():
    attn = torch.rand(2, N, 1)
    assert torch.equal(contextual_attention(attn, PTS3, 0), attn)
    head_ctx = build_pooling("conjunctive", num_features=16, num_classes=2, contextual_attention=True, context_k=0, positional_encoding=False)
    head_no = build_pooling("conjunctive", num_features=16, num_classes=2, contextual_attention=False, positional_encoding=False)
    head_no.load_state_dict(head_ctx.state_dict())
    head_ctx.eval(), head_no.eval()
    feats = torch.randn(2, 16, N)
    with torch.no_grad():
        a, b = head_ctx(PTS3, feats), head_no(PTS3, feats)
    assert torch.allclose(a["bag_logits"], b["bag_logits"]) and torch.allclose(a["interpretation"], b["interpretation"])


def test_contextual_attention_k1_is_identity_with_self_loop():
    attn = torch.rand(2, N, 1)
    assert torch.allclose(contextual_attention(attn, PTS3, 1), attn)


def test_contextual_attention_smooths_towards_neighbour_mean():
    xyz = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]])
    attn = torch.tensor([[[1.0], [0.0], [0.5]]])
    out = contextual_attention(attn, xyz, 2)
    assert torch.allclose(out[0, 0], torch.tensor([0.5])) and torch.allclose(out[0, 1], torch.tensor([0.5])) and torch.allclose(out[0, 2], torch.tensor([0.25]))


def test_point_importance_shape():
    model = build_model({"backbone": dict(name="dgcnn", k=8, emb_dims=64), "pooling": dict(name="conjunctive", num_classes=3, positional_encoding=False)}).eval()
    imp = model.point_importance(PTS3)
    assert imp.shape == (2, N) and torch.all(imp >= 0) and torch.all(imp <= 1)
