"""Checkpoint saving and loading, including the formats of the original code bases.

New checkpoints are plain dictionaries::

    {"model_state_dict": ..., "config": <full experiment config>, "pointmil_version": "x.y.z",
     "metrics": {...}, "epoch": int}

Legacy formats that can be converted:

* ``lightning``: Lightning ``.ckpt`` files of the original PointMIL code (keys
  ``model.feature_extractor.*`` and ``model.pooling.*``, or ``model.lpfa...`` for
  the fused CurveNet model). Their ``hyper_parameters`` pickle the model object,
  so a tolerant unpickler is used that replaces unimportable classes by stubs.
* ``openpoints``: ``*_ckpt_best.pth`` files of the OpenPoints training path
  (``model`` state dict with ``encoder.*`` and ``prediction.*`` keys).
"""

from __future__ import annotations

import argparse
import pickle
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

import pointmil


# --------------------------------------------------------------------------- new format
def save_checkpoint(path: str | Path, model: nn.Module, config: dict, metrics: dict | None = None, epoch: int | None = None, extra: dict | None = None) -> None:
    """Save ``{model_state_dict, config, pointmil_version, metrics, epoch}`` (the release checkpoint format)."""
    payload = {
        "model_state_dict": model.state_dict(),
        "config": _plain(config),
        "pointmil_version": pointmil.__version__,
        "metrics": _plain(metrics or {}),
        "epoch": epoch,
    }
    if extra:
        payload.update(_plain(extra))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


# --------------------------------------------------------------------------- tolerant unpickling
class _StubUnpickler(pickle.Unpickler):
    """Unpickler that replaces classes it cannot import by inert stand-ins."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return type(name, (), {"__module__": module, "__setstate__": _stub_setstate})


def _stub_setstate(self, state):
    if isinstance(state, dict):
        self.__dict__.update(state)
    elif isinstance(state, tuple) and state and isinstance(state[0], dict):
        self.__dict__.update(state[0])


_STUB_PICKLE = types.SimpleNamespace(Unpickler=_StubUnpickler, load=pickle.load, loads=pickle.loads, __name__="pointmil_stub_pickle")


def torch_load_tolerant(path: str | Path) -> Any:
    """``torch.load`` on CPU that survives pickled references to the original code."""
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", pickle_module=_STUB_PICKLE, weights_only=False)


# --------------------------------------------------------------------------- format detection / conversion
def detect_format(payload: dict) -> str:
    """``pointmil`` (release), ``lightning`` (original PointMIL code) or ``openpoints`` (PointNeXt fork)."""
    if "model_state_dict" in payload and "pointmil_version" in payload:
        return "pointmil"
    if "state_dict" in payload and "hyper_parameters" in payload:
        return "lightning"
    if "model" in payload and isinstance(payload["model"], dict) and any(k.startswith(("encoder.", "prediction.", "module.")) for k in payload["model"]):
        return "openpoints"
    raise ValueError("unrecognised checkpoint format")


def convert_lightning_state_dict(sd: dict) -> dict:
    """Map ``model.feature_extractor.*`` / ``model.pooling.*`` to ``backbone.*`` / ``pooling.*``."""
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("feature_extractor."):
            k = "backbone." + k[len("feature_extractor."):]
        out[k] = v
    return out


def convert_openpoints_state_dict(sd: dict) -> dict:
    """Map ``encoder.*`` / ``prediction.*`` to ``backbone.*`` / ``pooling.*``."""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("encoder."):
            k = "backbone." + k[len("encoder."):]
        elif k.startswith("prediction."):
            k = "pooling." + k[len("prediction."):]
        out[k] = v
    return out


def _instance_classifier_layout(state_dict: dict) -> dict:
    """Instance-classifier options that the saved ``args`` do not record, read from the weights.

    Two layouts are used only by ModelNet40 conjunctive checkpoints: features raised from d to 4d
    before pooling (``conv_raise``, Transformer) and a narrowing classifier d -> d/2 -> d/4 -> C
    (PointNet).
    """
    def get(key):
        return state_dict.get(f"pooling.{key}", state_dict.get(f"model.pooling.{key}"))

    w0, w7 = get("node_out.0.weight"), get("node_out.7.weight")
    if get("conv_raise.0.weight") is not None:
        return {"raise_dimension": True}
    if w0 is not None and w7 is not None and w0.dim() == 2 and 2 * w0.shape[0] == w0.shape[1]:
        return {"non_linear_narrow": True, "narrow_first_bias": get("node_out.0.bias") is not None}
    return {}


def lightning_args_to_model_config(args: argparse.Namespace | dict, state_dict: dict | None = None) -> dict:
    """Infer the ``model`` config section from the ``args`` stored in a Lightning checkpoint."""
    a = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    n_classes = int(a["n_classes"])
    fe = a.get("feature_extractor", "transformer")
    pooling = a.get("pooling", "conjunctive")
    head = dict(
        name=pooling,
        num_classes=n_classes,
        dropout=float(a.get("dropout", 0.1)),
        attention_hidden=int(a.get("num_heads", 8)),
        non_linear=bool(a.get("non_linear_cls", False)),
        non_linear_v1=bool(a.get("non_lin_1", False)),
        positional_encoding=bool(a.get("apply_positional_encoding", True)),
        contextual_attention=bool(a.get("contextual_attention", False)),
        # Older runs predate the ``context_k`` option; their code used k = 12.
        context_k=int(a.get("context_k", 12) or 12),
        num_points=int(a.get("num_points", 1024)),
    )
    if state_dict is not None:
        head.update(_instance_classifier_layout(state_dict))
    if fe == "curvenet":
        instance_input = "features"
        if state_dict is not None:
            w = state_dict.get("conv1.weight", state_dict.get("model.conv1.weight"))
            if w is not None and w.shape[1] == 3072:
                instance_input = "features_with_global"
        return {"backbone": dict(name="curvenet_mil", k=20, instance_input=instance_input, num_points=head["num_points"]),
                "pooling": dict(name=pooling, num_classes=n_classes)}
    if fe == "transformer":
        use_norm = bool(a.get("use_norm", True))
        backbone = dict(name="transformer", in_channels=6 if use_norm else 3, coord_dims=3,
                        num_k=list(a.get("num_K", [32, 64])), channel_dim=list(a.get("channel_dim", [128, 256])),
                        channel_raise=list(a.get("channel_raise", [64, 256])), dim_k=int(a.get("dim_k", 32)), heads=int(a.get("num_heads", 8)))
    elif fe == "dgcnn":
        backbone = dict(name="dgcnn", k=int(a.get("k", 20)), emb_dims=int(a.get("emb_dims", 1024)))
    elif fe == "pointnet":
        backbone = dict(name="pointnet")
    else:
        raise ValueError(f"cannot infer a backbone for feature_extractor='{fe}'")
    return {"backbone": backbone, "pooling": head}


def load_state_dict_report(model: nn.Module, state_dict: dict, strict: bool = False) -> dict:
    """Load with a report of missing and unexpected keys (unexpected keys are tolerated when ``strict`` is False)."""
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    unexpected = sorted(set(state_dict) - model_keys)
    missing = sorted(model_keys - set(filtered))
    if strict and (missing or unexpected):
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]}... unexpected={unexpected[:5]}...")
    if missing:
        raise RuntimeError(f"state dict is missing {len(missing)} keys, e.g. {missing[:5]}")
    model.load_state_dict(filtered, strict=False)
    return {"missing": missing, "unexpected": unexpected, "loaded": len(filtered)}


def load_checkpoint(path: str | Path, model: nn.Module | None = None, build: bool = True) -> dict:
    """Load a checkpoint of any supported format.

    Returns a dict with ``format``, ``state_dict`` (already in the new key
    layout), ``config`` (when available), ``metrics`` and, when ``build`` is True
    and the format carries enough information, a ``model`` with the weights loaded.
    For OpenPoints checkpoints a model must be passed in (the yaml config of the
    run is not stored in the file).
    """
    payload = torch_load_tolerant(path)
    fmt = detect_format(payload)
    result: dict[str, Any] = {"format": fmt, "path": str(path)}
    if fmt == "pointmil":
        result["state_dict"] = payload["model_state_dict"]
        result["config"] = payload.get("config")
        result["metrics"] = payload.get("metrics", {})
        result["epoch"] = payload.get("epoch")
        if model is None and build and result["config"] is not None:
            from pointmil.models import build_model

            model = build_model(result["config"]["model"])
    elif fmt == "lightning":
        result["state_dict"] = convert_lightning_state_dict(payload["state_dict"])
        args = payload["hyper_parameters"].get("args")
        result["legacy_args"] = vars(args) if isinstance(args, argparse.Namespace) else args
        result["epoch"] = payload.get("epoch")
        result["metrics"] = {}
        if args is not None:
            result["config"] = {"model": lightning_args_to_model_config(args, result["state_dict"])}
            if model is None and build:
                from pointmil.models import build_model

                model = build_model(result["config"]["model"])
    else:  # openpoints
        result["state_dict"] = convert_openpoints_state_dict(payload["model"])
        result["epoch"] = payload.get("epoch")
        result["metrics"] = {"best_val": payload.get("best_val")}
    if model is not None:
        result["load_report"] = load_state_dict_report(model, result["state_dict"], strict=False)
        result["model"] = model
    return result


def openpoints_head_layout(state_dict: dict) -> str:
    """Infer the instance-classifier layout of an OpenPoints checkpoint (``linear`` or ``mlp_bn_points``)."""
    if any(k.startswith("pooling.mil_head.1.") for k in state_dict):
        return "linear"
    return "mlp_bn_points"


def openpoints_attention_classifier(state_dict: dict) -> str:
    """Infer the attention-pooling bag classifier of an OpenPoints checkpoint (``mlp`` or ``linear``).

    ``mlp``: an MLP ``head`` on the concatenated mean and max embedding (2 x feature width).
    ``linear``: ``mil_head_att`` on the mean embedding; these checkpoints also contain an unused
    ``head`` of 1 x feature width, which is how they are recognised. The width test cannot tell the
    two apart for the ATLAS-1 DGCNN attention checkpoint, so its config sets the classifier
    explicitly. Non-attention checkpoints return ``mlp`` (unused).
    """
    att = state_dict.get("pooling.attention_head.0.weight")
    head = state_dict.get("pooling.head.0.0.weight")
    if att is None or head is None or "pooling.mil_head_att.1.weight" not in state_dict:
        return "mlp"
    return "mlp" if head.shape[1] == 2 * att.shape[1] else "linear"
