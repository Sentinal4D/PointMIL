"""Pretrained PointMIL models on the Hugging Face Hub.

One model repository per dataset (see ``REPOS``). Inside a repository every model is a
folder ``<backbone>/<pooling>/<fold>/`` holding

* ``model.safetensors``: the weights,
* ``config.yaml``: the full experiment config (``model`` builds the network),
* ``metrics.json``: test metrics, interpretability metrics and provenance.

Checkpoints behind the paper sit at the top level; models trained with this repository's
code (missing folds of the original runs, new 5-fold sets) sit under ``retrained/``.
``index.json`` at the repository root lists every model.

Example::

    from pointmil.hub import load_pretrained, list_pretrained

    model, info = load_pretrained("intra", "transformer", "conjunctive", fold=0)
    print(info["metrics"]["test"])
    print([m["path"] for m in list_pretrained("atlas1")])

Requires ``pip install -e .[hub]`` (``huggingface_hub`` and ``safetensors``).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from torch import nn

REPOS = {
    "intra": "Sentinal4D/pointmil-intra",
    "rbc": "Sentinal4D/pointmil-rbc",
    "atlas1": "Sentinal4D/pointmil-atlas1",
    "modelnet40": "Sentinal4D/pointmil-modelnet40",
    "scanobjectnn": "Sentinal4D/pointmil-scanobjectnn",
}
WEIGHTS, CONFIG, METRICS, INDEX = "model.safetensors", "config.yaml", "metrics.json", "index.json"


def fold_dir(dataset: str, fold: int | None) -> str:
    """Folder name of a split: ``fold<k>`` for cross-validation, ``split0`` (ATLAS-1) or ``official``."""
    if fold is not None:
        return f"fold{int(fold)}"
    return "split0" if dataset == "atlas1" else "official"


def model_path(dataset: str, backbone: str, pooling: str, fold: int | None = None, retrained: bool = False) -> str:
    """Path of one model inside its dataset repository."""
    path = f"{backbone}/{pooling}/{fold_dir(dataset, fold)}"
    return f"retrained/{path}" if retrained else path


def _canonical_keys(model: nn.Module) -> set[str]:
    """State-dict keys without the aliases of modules registered under two names (DGCNN's conv/bn)."""
    names = {n for n, _ in model.named_parameters(remove_duplicate=True)}
    names |= {n for n, _ in model.named_buffers(remove_duplicate=True)}
    return {k for k in model.state_dict() if k in names}


def save_pretrained(model: nn.Module, config: dict, metrics: dict, out_dir: str | Path) -> Path:
    """Write ``model.safetensors``, ``config.yaml`` and ``metrics.json`` into ``out_dir``."""
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    keep = _canonical_keys(model)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items() if k in keep}, str(out / WEIGHTS), metadata={"format": "pt"})
    (out / CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
    (out / METRICS).write_text(json.dumps(metrics, indent=2) + "\n")
    return out


def load_local(folder: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Build a model from a folder written by ``save_pretrained`` (strict weight loading)."""
    from safetensors.torch import load_file

    from pointmil.models.build import build_model

    folder = Path(folder)
    config = yaml.safe_load((folder / CONFIG).read_text())
    model = build_model(config["model"])
    missing, unexpected = model.load_state_dict(load_file(str(folder / WEIGHTS), device="cpu"), strict=False)
    aliases = set(model.state_dict()) - _canonical_keys(model)
    if unexpected or set(missing) - aliases:
        raise RuntimeError(f"weights in {folder} do not match the config: missing {sorted(set(missing) - aliases)[:5]}, unexpected {unexpected[:5]}")
    metrics = json.loads((folder / METRICS).read_text()) if (folder / METRICS).exists() else {}
    return model.to(device).eval(), {"config": config, "metrics": metrics, "folder": str(folder)}


def _repo(dataset: str, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    if dataset not in REPOS:
        raise KeyError(f"unknown dataset '{dataset}', choose from {sorted(REPOS)}")
    return REPOS[dataset]


def list_pretrained(dataset: str, repo_id: str | None = None, revision: str | None = None) -> list[dict]:
    """Entries of ``index.json``: one dict per model with ``path``, ``backbone``, ``pooling``, ``fold``, metrics."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_repo(dataset, repo_id), INDEX, revision=revision)
    return json.loads(Path(path).read_text())["models"]


def load_pretrained(dataset: str, backbone: str, pooling: str, fold: int | None = None, retrained: bool = False,
                    repo_id: str | None = None, revision: str | None = None, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Download one model and return ``(model in eval mode, info)``.

    Args:
        dataset: ``intra``, ``rbc``, ``atlas1``, ``modelnet40`` or ``scanobjectnn``.
        backbone: ``transformer``, ``dgcnn``, ``pointnet``, ``curvenet`` or ``pointmlpelite``.
        pooling: ``instance``, ``attention``, ``additive``, ``conjunctive``, ``gap`` or a variant
            listed in the repository (for example ``conjunctive-nocontext``).
        fold: cross-validation fold (IntrA, RBC); ``None`` for single-split datasets.
        retrained: load the model trained with this repository's code instead of the paper checkpoint.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    repo = _repo(dataset, repo_id)
    sub = model_path(dataset, backbone, pooling, fold, retrained)
    try:
        files = [hf_hub_download(repo, f"{sub}/{name}", revision=revision) for name in (CONFIG, WEIGHTS, METRICS)]
    except EntryNotFoundError as err:
        available = sorted(m["path"] for m in list_pretrained(dataset, repo_id, revision))
        raise FileNotFoundError(f"{repo} has no model at '{sub}'. Available: {available}") from err
    model, info = load_local(Path(files[0]).parent, device=device)
    info.update(repo_id=repo, path=sub)
    return model, info
