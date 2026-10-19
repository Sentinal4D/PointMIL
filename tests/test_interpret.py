"""Hand-computed checks of AOPCR and NDCG@n."""

import math

import torch
from torch import nn

from pointmil.interpret import calculate_aopcr, calculate_ndcg_at_n
from pointmil.interpret.psm import rank_scores


def test_ndcg_perfect_and_worst():
    labels = torch.tensor([1, 1, 0, 0, 0])
    assert calculate_ndcg_at_n(torch.tensor([0.9, 0.8, 0.1, 0.2, 0.3]), labels) == 1.0
    assert calculate_ndcg_at_n(torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3]), labels) == 0.0


def test_ndcg_partial():
    # n = 2 positives; top-2 = [idx 2 (rel 0), idx 0 (rel 1)] -> dcg = 1/log2(3); norm = 1 + 1/log2(3)
    labels = torch.tensor([1, 1, 0])
    scores = torch.tensor([0.5, 0.1, 0.9])
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert abs(calculate_ndcg_at_n(scores, labels) - expected) < 1e-9


class CountModel(nn.Module):
    """Logit of class 1 = number of points with x > 0 (so removing them in order drops the logit linearly)."""

    def forward(self, bags):
        pos = (bags[..., 0] > 0).float()
        logit1 = pos.sum(dim=1)
        return {"bag_logits": torch.stack([torch.zeros_like(logit1), logit1], dim=1), "interpretation": torch.stack([-pos, pos], dim=1)}

    @staticmethod
    def interpret(out):
        return out["interpretation"]


def test_aopcr_counting_model():
    torch.manual_seed(0)
    n = 20
    bag = torch.zeros(1, n, 3)
    bag[0, :10, 0] = 1.0  # 10 "important" points
    model = CountModel()
    aopcr, pc, r_pc = calculate_aopcr(model, bag, stop=0.5, step=0.25, n_random=1, seed=0)
    # steps: keep 100%, 75%, 50% -> remove 5 then 10 important points: logit 10 -> 5 -> 0; drops 5 and 10; mean over 3 steps = 5
    assert torch.allclose(pc[0], torch.tensor([0.0, -5.0, -10.0]))
    aopc = (5 + 10) / 3
    assert aopcr[0] <= aopc and aopcr[0] > 0  # random ordering removes fewer important points


def test_rank_scores():
    s = torch.tensor([[0.2, 0.9, 0.5]])
    assert torch.equal(rank_scores(s), torch.tensor([[0.0, 2.0, 1.0]]))


def test_attention_importance_is_the_attention_not_uniform():
    from pointmil.models import build_model

    torch.manual_seed(0)
    model = build_model({"backbone": {"name": "pointnet"}, "pooling": {"name": "attention", "num_classes": 3, "positional_encoding": False}}).eval()
    x = torch.randn(2, 64, 3)
    imp = model.point_importance(x)
    out = model(x)
    assert imp.shape == (2, 64)
    assert torch.allclose(imp, out["attention"].squeeze(-1))
    assert imp.std(dim=1).min() > 0  # not one constant value per cloud


def test_aopcr_runs_on_heads_with_point_axis_batchnorm():
    from pointmil.interpret import calculate_aopcr
    from pointmil.interpret.aopcr import accepts_fewer_points
    from pointmil.models import build_model

    torch.manual_seed(0)
    cfg = {"backbone": {"name": "pointnet"}, "pooling": {"name": "conjunctive", "num_classes": 3, "non_linear": True, "num_points": 64, "positional_encoding": False}}
    model = build_model(cfg).eval()
    x = torch.randn(2, 64, 3)
    assert not accepts_fewer_points(model, x)
    aopcr, curve, random_curve = calculate_aopcr(model, x, n_random=1)
    assert aopcr.shape == (2,) and torch.isfinite(aopcr).all()
    assert curve.shape == random_curve.shape
