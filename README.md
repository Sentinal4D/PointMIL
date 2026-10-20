# PointMIL: Interpretable point cloud classification using multiple instance learning

[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/De_Vries_Interpretable_point_cloud_classification_using_multiple_instance_learning_ICCV_2025_paper.pdf) ·
[Supplementary](https://openaccess.thecvf.com/content/ICCV2025/supplemental/De_Vries_Interpretable_point_cloud_ICCV_2025_supplemental.pdf) ·
[Project page](https://sentinal4d.github.io/PointMIL) ·
[ATLAS-1 dataset](https://huggingface.co/datasets/Sentinal4D/ATLAS-1) ·
[Pretrained models](https://huggingface.co/Sentinal4D)

Official code for *Interpretable point cloud classification using multiple instance learning*
(De Vries et al., ICCV 2025). PointMIL treats a point cloud as a bag of points: a feature
extractor gives every point a feature vector, and a multiple instance learning pooling head
classifies the cloud while scoring how much each point contributed to the prediction.

<p align="center"><img src="assets/pointmil_pipeline.png" width="85%" alt="PointMIL pipeline"></p>

## Installation

```bash
git clone https://github.com/Sentinal4D/PointMIL.git && cd PointMIL
pip install -e ".[dev,data,hub]"        # or: bash install.sh (conda environment with the paper's versions)
```

## Quickstart

```python
import numpy as np
from pointmil import prepare_cloud
from pointmil.hub import load_pretrained

model, info = load_pretrained("intra", "transformer", "conjunctive", fold=0)
points = np.loadtxt("IntrA/annotated/ad/AN1-_norm.ad")[:, :6]   # an IntrA cloud: xyz + normals
x = prepare_cloud(points, num_points=1024)
pred = model(x)["bag_logits"].argmax(1)                         # predicted class
importance = model.point_importance(x)                          # [1, 1024] importance of every point
```

The [quickstart notebook](notebooks/quickstart.ipynb) plots the point importance of an ATLAS-1 cell.

## Data

```bash
export POINTMIL_DATA=/path/to/datasets
bash scripts/download_data.sh all
```

This downloads ModelNet40 and ATLAS-1, builds the RBC point clouds from the public masks, and
prints instructions for IntrA and ScanObjectNN, which need their authors' download pages.
See the [guide](docs/GUIDE.md#data) for folder layouts and sources.

## Training and evaluation

```bash
python scripts/train.py --config configs/table1/intra_transformer_conjunctive.yaml --fold 0 --seed 42
python scripts/eval.py --checkpoint outputs/table1_intra_transformer_conjunctive/fold0_seed42/best.pt --split test
python scripts/interpret.py metrics --checkpoint outputs/table1_intra_transformer_conjunctive/fold0_seed42/best.pt \
    --split annotated --target_class 1 --context_k 12
```

Every config lists all its hyperparameters ([config to paper map](docs/EXPERIMENTS.md)). Two
training recipes are included: the original PointMIL recipe, and the
[PointNeXt](https://github.com/guochengqian/PointNeXt) recipe as implemented in
[OpenPoints](https://github.com/guochengqian/openpoints) (ATLAS-1, ScanObjectNN,
PointMLPElite, and recommended for ModelNet40 via `configs/table2/modelnet40_openpoints_*`).

## Reproducing the paper

| Result | Command |
|---|---|
| Table 1 and the IntrA rows of Table 2 | `bash scripts/reproduce_table1.sh` |
| Table 2 (RBC, ModelNet40, ATLAS-1, ScanObjectNN) | `bash scripts/reproduce_table2.sh rbc` (or `modelnet40`, `atlas1`, `scanobjectnn`) |
| Supp. Table 3 and Fig. 6 | `bash scripts/reproduce_supp_table3_fig6.sh` |

[Run times](docs/GUIDE.md#run-times) · [Reproducibility](REPRODUCIBILITY.md)

## Pretrained models

| Dataset | Models |
|---|---|
| IntrA | [Sentinal4D/pointmil-intra](https://huggingface.co/Sentinal4D/pointmil-intra) |
| 3D red blood cells | [Sentinal4D/pointmil-rbc](https://huggingface.co/Sentinal4D/pointmil-rbc) |
| ATLAS-1 | [Sentinal4D/pointmil-atlas1](https://huggingface.co/Sentinal4D/pointmil-atlas1) |
| ModelNet40 | [Sentinal4D/pointmil-modelnet40](https://huggingface.co/Sentinal4D/pointmil-modelnet40) |
| ScanObjectNN | [Sentinal4D/pointmil-scanobjectnn](https://huggingface.co/Sentinal4D/pointmil-scanobjectnn) |

`load_pretrained(dataset, backbone, pooling, fold)` loads any of them; each model card lists
every model with its metrics ([details](docs/GUIDE.md#pretrained-models)).

## Citation

```bibtex
@InProceedings{De_Vries_2025_ICCV,
  author    = {De Vries, Matt and Naidoo, Reed and Fourkioti, Olga and Dent, Lucas G. and Curry, Nathan and Dunsby, Chris and Bakal, Chris},
  title     = {Interpretable point cloud classification using multiple instance learning},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {22209-22220}
}
```

If you use the PointNeXt / OpenPoints recipe or encoders, please also cite:

```bibtex
@InProceedings{qian2022pointnext,
  title     = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author    = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```

## Licence

Code and models: [Sentinal4D Non-Commercial EULA](LICENSE). ATLAS-1: CC BY 4.0.
Third-party components keep their own licences ([NOTICE.md](NOTICE.md)).

## Acknowledgements

PointMIL builds on [PointNeXt](https://github.com/guochengqian/PointNeXt) and
[OpenPoints](https://github.com/guochengqian/openpoints),
[CurveNet](https://github.com/tiangexiang/CurveNet),
[PointMLP](https://github.com/ma-xu/pointMLP-pytorch),
[DGCNN](https://github.com/antao97/dgcnn.pytorch),
[PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch),
[3DMedPT](https://github.com/crane-papercode/3DMedPT) and
[MILLET](https://github.com/JAEarly/MILTimeSeriesClassification). We thank the authors of
[IntrA](https://github.com/intra3d2019/IntrA), the
[3D red blood cell dataset](https://doi.org/10.1371/journal.pcbi.1008934) and its
[3D masks](https://doi.org/10.5281/zenodo.7031924),
[ModelNet](https://modelnet.cs.princeton.edu/) and
[ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/) for making their data available.
