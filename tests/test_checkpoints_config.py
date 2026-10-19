"""Checkpoint round trip, legacy key conversion, config loading and metrics."""

import argparse

import numpy as np
import pytest
import torch
import yaml

from pointmil.checkpoints import (
    convert_lightning_state_dict,
    convert_openpoints_state_dict,
    lightning_args_to_model_config,
    load_checkpoint,
    openpoints_attention_classifier,
    save_checkpoint,
)
from pointmil.config import load_config, validate_config
from pointmil.metrics import ConfusionMatrix
from pointmil.models import build_model

CFG = {
    "experiment": {"name": "t"},
    "data": {"name": "synthetic", "root": "${POINTMIL_DATA}/x", "num_points": 64, "batch_size": 2, "num_workers": 0, "train_transforms": [], "test_transforms": []},
    "model": {"backbone": {"name": "dgcnn", "k": 8, "emb_dims": 32}, "pooling": {"name": "additive", "num_classes": 2, "positional_encoding": False, "contextual_attention": False}},
    "train": {"epochs": 1, "optimizer": {"name": "adamw", "lr": 1e-3}, "scheduler": {"name": "cosine", "t_max": 1, "min_lr": 0}, "loss": {"name": "cross_entropy"}, "grad_clip": None, "checkpoint_monitor": "val_f1", "early_stopping_patience": 0, "val_split": "test"},
}


def test_checkpoint_round_trip(tmp_path):
    model = build_model(CFG["model"])
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, CFG, metrics={"val": {"f1": 1.0}}, epoch=3)
    res = load_checkpoint(path)
    assert res["format"] == "pointmil" and res["epoch"] == 3 and res["metrics"]["val"]["f1"] == 1.0
    assert res["config"]["model"]["backbone"]["name"] == "dgcnn"
    for k, v in model.state_dict().items():
        assert torch.equal(res["model"].state_dict()[k], v)


def test_lightning_key_conversion_and_args():
    sd = {"model.feature_extractor.conv1.0.weight": torch.zeros(1), "model.pooling.node_out.weight": torch.zeros(1)}
    out = convert_lightning_state_dict(sd)
    assert set(out) == {"backbone.conv1.0.weight", "pooling.node_out.weight"}
    args = argparse.Namespace(n_classes=2, feature_extractor="dgcnn", pooling="additive", dropout=0.1, num_heads=8, non_linear_cls=False,
                              apply_positional_encoding=True, contextual_attention=True, k=20, emb_dims=1024, num_points=1024)
    cfg = lightning_args_to_model_config(args)
    assert cfg["backbone"] == {"name": "dgcnn", "k": 20, "emb_dims": 1024}
    assert cfg["pooling"]["context_k"] == 12 and cfg["pooling"]["name"] == "additive"
    model = build_model(cfg)
    assert model.pooling.contextual_attention


def test_openpoints_key_conversion():
    sd = {"module.encoder.head.gconv.nn.0.weight": torch.zeros(1), "prediction.mil_head.1.weight": torch.zeros(1)}
    out = convert_openpoints_state_dict(sd)
    assert set(out) == {"backbone.head.gconv.nn.0.weight", "pooling.mil_head.1.weight"}


def test_openpoints_attention_classifier_detection():
    att = torch.zeros(8, 256)
    mlp = {"pooling.attention_head.0.weight": att, "pooling.head.0.0.weight": torch.zeros(512, 512), "pooling.mil_head_att.1.weight": torch.zeros(3, 256)}
    lin = {"pooling.attention_head.0.weight": att, "pooling.head.0.0.weight": torch.zeros(512, 256), "pooling.mil_head_att.1.weight": torch.zeros(3, 256)}
    assert openpoints_attention_classifier(mlp) == "mlp"
    assert openpoints_attention_classifier(lin) == "linear"
    assert openpoints_attention_classifier({"pooling.mil_head.1.weight": torch.zeros(3, 256)}) == "mlp"


def test_lightning_layout_inference_from_weights():
    args = {"n_classes": 40, "feature_extractor": "pointnet", "pooling": "conjunctive", "non_linear_cls": True}
    narrow = {"pooling.node_out.0.weight": torch.zeros(512, 1024), "pooling.node_out.0.bias": torch.zeros(512),
              "pooling.node_out.7.weight": torch.zeros(40, 256)}
    cfg = lightning_args_to_model_config(args, narrow)["pooling"]
    assert cfg["non_linear_narrow"] and cfg["narrow_first_bias"]
    raised = {"pooling.conv_raise.0.weight": torch.zeros(512, 256, 1), "pooling.node_out.0.weight": torch.zeros(512, 1024)}
    assert lightning_args_to_model_config(args, raised)["pooling"]["raise_dimension"]
    wide = {"pooling.node_out.0.weight": torch.zeros(2048, 1024), "pooling.node_out.7.weight": torch.zeros(40, 512)}
    assert "non_linear_narrow" not in lightning_args_to_model_config(args, wide)["pooling"]


def test_config_loading(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(CFG))
    monkeypatch.setenv("POINTMIL_DATA", "/data")
    cfg = load_config(path, overrides={"train.epochs": 5})
    assert cfg["data"]["root"] == "/data/x" and cfg["train"]["epochs"] == 5
    cfg2 = load_config(path, data_root="/other")
    assert cfg2["data"]["root"] == "/other/x"
    bad = {k: dict(v) for k, v in CFG.items()}
    del bad["train"]["val_split"]
    with pytest.raises(KeyError):
        validate_config(bad)


def test_confusion_matrix_metrics():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 2, 2, 1]), np.array([0, 1, 1, 2, 2, 2]))
    s = cm.summary()
    assert abs(s["oacc"] - 100 * 4 / 6) < 1e-9
    # per-class recall: c0 1/1, c1 1/2, c2 2/3
    assert abs(s["macc"] - 100 * (1 + 0.5 + 2 / 3) / 3) < 1e-9
    # f1: c0 p=1/2 r=1 -> 2/3 ; c1 p=1/2 r=1/2 -> 1/2 ; c2 p=1 r=2/3 -> 4/5
    assert abs(s["f1"] - 100 * (2 / 3 + 1 / 2 + 4 / 5) / 3) < 1e-9


def test_optimizer_filters_bias_and_norm_weights_from_weight_decay():
    from pointmil.engine.optim import build_optimizer

    model = build_model({"backbone": {"name": "pointnet"}, "pooling": {"name": "instance", "num_classes": 2, "positional_encoding": False}})
    opt = build_optimizer(model, {"name": "sgd", "lr": 0.1, "weight_decay": 2e-4, "momentum": 0.9, "filter_bias_and_bn": True})
    no_decay, decay = opt.param_groups
    assert no_decay["weight_decay"] == 0.0 and decay["weight_decay"] == 2e-4
    assert all(p.ndim == 1 for p in no_decay["params"]) and all(p.ndim > 1 for p in decay["params"])
    assert len(no_decay["params"]) + len(decay["params"]) == len(list(model.parameters()))
    plain = build_optimizer(model, {"name": "sgd", "lr": 0.1, "weight_decay": 2e-4})
    assert len(plain.param_groups) == 1 and plain.param_groups[0]["weight_decay"] == 2e-4
