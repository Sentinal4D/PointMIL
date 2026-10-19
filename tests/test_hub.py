"""Hugging Face helpers: local round trip and the download path (network mocked)."""

import json

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("huggingface_hub")

from pointmil import hub  # noqa: E402
from pointmil.models import build_model  # noqa: E402

MODEL_CFG = {"backbone": {"name": "pointnet"}, "pooling": {"name": "conjunctive", "num_classes": 3, "positional_encoding": False}}


def _save(tmp_path):
    torch.manual_seed(0)
    model = build_model(MODEL_CFG).eval()
    config = {"data": {"feature_keys": ["pos"]}, "model": MODEL_CFG}
    folder = tmp_path / "pointnet" / "conjunctive" / "fold0"
    hub.save_pretrained(model, config, {"test": {"macc": 90.0}}, folder)
    return model, folder


def test_save_and_load_local_round_trip(tmp_path):
    model, folder = _save(tmp_path)
    assert {p.name for p in folder.iterdir()} == {"model.safetensors", "config.yaml", "metrics.json"}
    reloaded, info = hub.load_local(folder)
    x = torch.randn(2, 64, 3)
    with torch.no_grad():
        assert torch.allclose(model(x)["bag_logits"], reloaded(x)["bag_logits"])
    assert info["metrics"]["test"]["macc"] == 90.0


def test_paths():
    assert hub.model_path("intra", "transformer", "conjunctive", 0) == "transformer/conjunctive/fold0"
    assert hub.model_path("rbc", "dgcnn", "instance", 3, retrained=True) == "retrained/dgcnn/instance/fold3"
    assert hub.model_path("atlas1", "dgcnn", "instance") == "dgcnn/instance/split0"
    assert hub.model_path("modelnet40", "pointnet", "instance") == "pointnet/instance/official"


def test_load_pretrained_with_mocked_download(tmp_path, monkeypatch):
    model, _ = _save(tmp_path)
    (tmp_path / "index.json").write_text(json.dumps({"models": [{"path": "pointnet/conjunctive/fold0"}]}))
    calls = []

    def fake_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename))
        path = tmp_path / filename
        if not path.exists():
            from huggingface_hub.utils import EntryNotFoundError
            raise EntryNotFoundError(filename)
        return str(path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    loaded, info = hub.load_pretrained("intra", "pointnet", "conjunctive", fold=0)
    assert calls[0][0] == "Sentinal4D/pointmil-intra" and info["path"] == "pointnet/conjunctive/fold0"
    with pytest.raises(FileNotFoundError, match="Available"):
        hub.load_pretrained("intra", "pointnet", "conjunctive", fold=4)


def test_round_trip_with_aliased_modules(tmp_path):
    """DGCNN registers its batch norms under two names; the saved file keeps one copy and reloads strictly."""
    cfg = {"backbone": {"name": "dgcnn", "k": 8, "emb_dims": 64}, "pooling": {"name": "additive", "num_classes": 2, "positional_encoding": False}}
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    folder = tmp_path / "dgcnn"
    hub.save_pretrained(model, {"data": {"feature_keys": ["pos"]}, "model": cfg}, {}, folder)
    reloaded, _ = hub.load_local(folder)
    x = torch.randn(2, 64, 3)
    with torch.no_grad():
        assert torch.allclose(model(x)["bag_logits"], reloaded(x)["bag_logits"], atol=1e-6)
