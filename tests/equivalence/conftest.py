"""Fixtures for the equivalence tests against the original code bases.

These tests only run on the machine that holds the original repositories. They
are skipped unless the environment variables ``POINTMIL_REPO_A`` (original
PointMIL code) and ``POINTMIL_REPO_B`` (PointNeXt fork whose ``openpoints`` folder
is the OpenPoints fork) point at those repositories.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_A = Path(os.environ.get("POINTMIL_REPO_A", ""))
REPO_B = Path(os.environ.get("POINTMIL_REPO_B", ""))


@pytest.fixture(scope="session")
def repo_a_models():
    """Import the ``models`` package of the original code at its committed HEAD.

    The working copy of the original repository contains uncommitted debugging
    edits that never produced a paper number, so the committed ``mil_pooling.py``
    is used instead of the file on disk.
    """
    if not str(REPO_A) or not (REPO_A / "models" / "mil_pooling.py").exists():
        pytest.skip("original PointMIL repository not available")
    tmp = Path(tempfile.mkdtemp(prefix="pointmil_repo_a_"))
    shutil.copytree(REPO_A / "models", tmp / "models", ignore=shutil.ignore_patterns("__pycache__"))
    head = subprocess.run(["git", "-C", str(REPO_A), "show", "HEAD:models/mil_pooling.py"], check=True, capture_output=True, text=True).stdout
    (tmp / "models" / "mil_pooling.py").write_text(head)
    # Avoid importing the lightning-based package __init__ (heavy, optional deps).
    (tmp / "models" / "__init__.py").write_text("")
    (tmp / "models" / "backbones" / "__init__.py").write_text("")
    sys.path.insert(0, str(tmp))
    for mod in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
        del sys.modules[mod]
    import importlib

    pkg = importlib.import_module("models")
    yield pkg
    sys.path.remove(str(tmp))
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def openpoints():
    """Import the OpenPoints fork (needs its compiled CUDA extension and a GPU)."""
    import torch

    if not str(REPO_B) or not (REPO_B / "openpoints" / "models").exists() or not torch.cuda.is_available():
        pytest.skip("OpenPoints fork or GPU not available")
    sys.path.insert(0, str(REPO_B))
    try:
        import openpoints  # noqa: F401
        import openpoints.models  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import openpoints: {exc}")
    yield openpoints
