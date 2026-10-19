"""Experiment configuration loading and validation.

A configuration is a single YAML file. The configs under ``configs/`` state every
hyperparameter explicitly; only the keys in ``REQUIRED`` are mandatory, and the code
falls back to documented defaults for the rest. The data root may be given
as ``${POINTMIL_DATA}`` and is expanded from the environment or from the
``--data_root`` command line option.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

from pointmil.utils.io import load_yaml

REQUIRED = {
    "experiment": ["name"],
    "data": ["name", "root", "num_points", "batch_size", "num_workers", "train_transforms", "test_transforms"],
    "model": ["backbone", "pooling"],
    "train": ["epochs", "optimizer", "scheduler", "loss", "grad_clip", "checkpoint_monitor", "early_stopping_patience", "val_split"],
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        def repl(m):
            name = m.group(1)
            if name not in env:
                raise KeyError(f"environment variable {name} is not set (needed by the config)")
            return env[name]

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None, data_root: str | None = None) -> dict:
    """Load, expand and validate a configuration file.

    Args:
        path: YAML file.
        overrides: optional ``{"a.b.c": value}`` dotted overrides.
        data_root: replaces ``${POINTMIL_DATA}`` when given.
    """
    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} does not contain a mapping")
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    env = dict(os.environ)
    if data_root is not None:
        env["POINTMIL_DATA"] = str(data_root)
    cfg = _expand(cfg, env)
    validate_config(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def validate_config(cfg: dict) -> None:
    """Raise ``KeyError`` if a required section or key is missing."""
    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise KeyError(f"config is missing the '{section}' section")
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"config section '{section}' is missing '{key}'")
    for part in ("backbone", "pooling"):
        if "name" not in cfg["model"][part]:
            raise KeyError(f"model.{part}.name is required")
