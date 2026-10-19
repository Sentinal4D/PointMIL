"""Datasets and loaders."""

from pointmil.data.atlas1 import ATLAS1
from pointmil.data.build import build_dataloaders, build_dataset
from pointmil.data.intra import IntrA
from pointmil.data.modelnet40 import ModelNet40
from pointmil.data.preprocess import prepare_cloud
from pointmil.data.rbc import RBC
from pointmil.data.scanobjectnn import ScanObjectNN
from pointmil.data.synthetic import Synthetic

DATASETS = {"intra": IntrA, "rbc": RBC, "atlas1": ATLAS1, "modelnet40": ModelNet40, "scanobjectnn": ScanObjectNN, "synthetic": Synthetic}

__all__ = ["DATASETS", "IntrA", "RBC", "ATLAS1", "ModelNet40", "ScanObjectNN", "Synthetic", "build_dataset", "build_dataloaders", "prepare_cloud"]
