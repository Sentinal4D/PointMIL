"""PointMIL: interpretable point cloud classification with multiple instance learning."""

__version__ = "1.0.0"

from pointmil.checkpoints import load_checkpoint  # noqa: E402
from pointmil.data.preprocess import prepare_cloud  # noqa: E402
from pointmil.models import PointMIL, build_model  # noqa: E402

__all__ = ["__version__", "PointMIL", "build_model", "load_checkpoint", "prepare_cloud"]
