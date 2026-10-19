"""Training with the release code matches training with PointNeXt / OpenPoints, step for step (GPU only).

The ScanObjectNN DGCNN additive model of the paper is built twice from the same weights: with the
OpenPoints fork exactly as its run config describes it, and with the release config. Both then take
the same optimisation steps as the PointNeXt classification trainer (AdamW through
``build_optimizer_from_cfg``, OpenPoints ``SmoothCrossEntropy``, gradient clipping at 10) on the
same batches. Dropout is switched off in both so that the steps are comparable; everything else
(BatchNorm in training mode, weight decay groups, loss, clipping) runs as in training.

The first step is identical to float precision. Afterwards both runs drift apart slowly through
GPU floating-point non-determinism, exactly as two copies of the OpenPoints model do (measured:
losses still agree to 1e-5 after five steps), so later steps are compared by their loss.
"""

from __future__ import annotations

import copy

import pytest
import torch
import yaml

DEV = "cuda"

OPENPOINTS_MODEL = {  # model section of the original ScanObjectNN DGCNN additive run config
    "NAME": "BaseCls",
    "encoder_args": {"NAME": "DGCNN", "in_channels": 4, "channels": 64, "emb_dims": 1024, "n_blocks": 5, "k": 20,
                     "conv": "edge", "dropout": 0.5, "n_classes": 40,
                     "act_args": {"act": "leakyrelu", "negative_slope": 0.2}, "norm_args": {"norm": "bn"}},
    "cls_args": {"NAME": "ClsHead", "in_channels": 1024, "is_mil": True, "mlps": [512, 256],
                 "norm_args": {"norm": "bn1d"}, "num_classes": 15, "pooling": "additive"},
}


def _no_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def test_scanobjectnn_training_steps_match_pointnext(openpoints):
    from openpoints.loss.build import SmoothCrossEntropy as OPSmoothCE
    from openpoints.models import build_model_from_cfg
    from openpoints.optim import build_optimizer_from_cfg
    from openpoints.utils import EasyConfig

    from pointmil.checkpoints import convert_openpoints_state_dict
    from pointmil.engine.losses import build_loss
    from pointmil.engine.optim import build_optimizer
    from pointmil.models import build_model

    torch.manual_seed(0)
    model_cfg = EasyConfig()
    model_cfg.update(copy.deepcopy(OPENPOINTS_MODEL))
    original = build_model_from_cfg(model_cfg).to(DEV)
    cfg = yaml.safe_load(open("configs/table2/scanobjectnn_dgcnn_additive.yaml"))
    ported = build_model(cfg["model"]).to(DEV)
    missing, unexpected = ported.load_state_dict(convert_openpoints_state_dict(original.state_dict()), strict=False)
    assert not missing, missing
    _no_dropout(original), _no_dropout(ported)
    original.train(), ported.train()

    t = cfg["train"]
    opt_a = build_optimizer_from_cfg(original, lr=t["optimizer"]["lr"], NAME="adamw", weight_decay=t["optimizer"]["weight_decay"])
    opt_b = build_optimizer(ported, t["optimizer"])
    loss_a_fn = OPSmoothCE(label_smoothing=t["loss"]["label_smoothing"])
    loss_b_fn = build_loss(t["loss"])

    gen = torch.Generator(device="cpu").manual_seed(1)
    for step in range(4):
        pos = torch.randn(8, 1024, 3, generator=gen).to(DEV)
        heights = pos[..., 1:2] - pos[..., 1:2].min(dim=1, keepdim=True)[0]
        x = torch.cat([pos, heights], dim=-1)
        y = torch.randint(0, 15, (8,), generator=gen).to(DEV)

        logits_a = original({"pos": pos, "x": x.transpose(1, 2).contiguous()})
        loss_a = loss_a_fn(logits_a, y)
        opt_a.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(original.parameters(), t["grad_clip"], norm_type=2)
        opt_a.step()

        logits_b = ported(x)["bag_logits"]
        loss_b = loss_b_fn(logits_b, y)
        opt_b.zero_grad()
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(ported.parameters(), t["grad_clip"], norm_type=2)
        opt_b.step()

        assert abs(loss_a.item() - loss_b.item()) < 1e-4, (step, loss_a.item(), loss_b.item())
        if step == 0:
            assert torch.allclose(logits_a, logits_b, atol=1e-6), (logits_a - logits_b).abs().max()
            after_a = convert_openpoints_state_dict(original.state_dict())
            after_b = ported.state_dict()
            worst = max((after_a[k].float() - after_b[k].float()).abs().max().item() for k in after_b if k in after_a)
            assert worst < 1e-6, worst  # same parameters and BatchNorm statistics after one training step


@pytest.mark.parametrize("filter_bias_and_bn", [True, False])
def test_weight_decay_groups_match_openpoints(openpoints, filter_bias_and_bn):
    from openpoints.optim import build_optimizer_from_cfg

    from pointmil.engine.optim import build_optimizer
    from pointmil.models import build_model

    cfg = yaml.safe_load(open("configs/table2/scanobjectnn_dgcnn_additive.yaml"))
    model = build_model(cfg["model"])
    a = build_optimizer_from_cfg(model, lr=0.002, NAME="adamw", weight_decay=0.05, filter_bias_and_bn=filter_bias_and_bn)
    b = build_optimizer(model, dict(cfg["train"]["optimizer"], filter_bias_and_bn=filter_bias_and_bn))

    def groups(opt):
        return sorted((g["weight_decay"], sorted(id(p) for p in g["params"])) for g in opt.param_groups if g["params"])

    assert groups(a) == groups(b)
