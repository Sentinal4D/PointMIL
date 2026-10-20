# PointMIL guide

Details behind the [README](../README.md). [EXPERIMENTS.md](EXPERIMENTS.md) maps every config
to the paper; [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) describes the evaluation protocol and determinism.

## Installation

Pure PyTorch, no CUDA extensions to compile: Python 3.10 or newer, PyTorch 2.0 or newer
(the paper used Python 3.10, PyTorch 2.2.2 and CUDA 12.1).

```bash
bash install.sh                 # conda environment from environment.yml, package and unit tests
pip install -e ".[dev]"         # or into an existing environment
```

`install.sh` uses mamba or micromamba when they are installed (the classic conda solver can
take hours on `environment.yml`). Optional extras: `data` (dataset download, RBC
preparation), `hub` (pretrained models), `notebook` (quickstart notebook), `wandb`
(Weights & Biases logging), e.g. `pip install -e ".[data,hub,notebook]"`.
`bash scripts/smoke_test.sh` trains every backbone and pooling head for two epochs on a
synthetic dataset on CPU and is what the CI runs.

## Data

All configs read their data from `${POINTMIL_DATA}/<dataset>`.
`bash scripts/download_data.sh [intra|rbc|modelnet40|scanobjectnn|atlas1|all]` downloads
what can be downloaded automatically, prints instructions for the rest and carries on
with the next dataset if one step fails.

| Dataset | Folder | Source | Split |
|---|---|---|---|
| IntrA | `IntrA/annotated/ad`, `IntrA/generated/{vessel,aneurysm}/ad` | [IntrA](https://github.com/intra3d2019/IntrA) (Google Drive folder of its authors) | `splits/intra/fold_{0..4}.csv` |
| 3D RBC | `RBC/ad/*.ad` | built with `scripts/prepare_rbc.py` from the [public 3D masks](https://doi.org/10.5281/zenodo.7031924) | `splits/rbc/fold_{0..4}.csv` |
| ModelNet40 | `ModelNet40/**/*.h5` | official `modelnet40_ply_hdf5_2048.zip`, downloaded by the script | official |
| ScanObjectNN (PB_T50_RS) | `ScanObjectNN/{training,test}_objectdataset_augmentedrot_scale75.h5` | [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/) (terms-of-use form) | official |
| ATLAS-1 | `ATLAS-1/metadata.csv`, `ATLAS-1/data/*.obj`, `ATLAS-1/pointclouds_4096/*.npy` | [ATLAS-1](https://huggingface.co/datasets/Sentinal4D/ATLAS-1) | `splits/atlas1/split0.csv` |

- **`.ad` files** are text files with one point per line (`x y z nx ny nz label`). The
  IntrA, RBC and ATLAS-1 loaders subsample 1024 points and normalise each cloud to the unit
  ball; ModelNet40 and ScanObjectNN take the first 1024 points of their pre-normalised clouds.
- **RBC** point clouds are not redistributed. `scripts/prepare_rbc.py` reads only the 825
  masks from the Zenodo archive (about 7 minutes) and writes the vertices and normals of each
  mask's marching-cubes surface (`pointmil.data.rbc.mask_to_point_cloud`), byte-identical to
  the files used for the paper.
- **ModelNet40** comes from the copy of the official archive on the Hugging Face Hub
  (`Msun/modelnet40`), checked against its SHA-256, so everyone trains and evaluates on the
  same file; the original Stanford host is the fallback.
- **ATLAS-1** configs read `pointclouds_4096/`. With only the meshes, add
  `--override data.source=mesh` to sample points from the mesh vertices.

## Training

```bash
python scripts/train.py --config configs/table1/intra_transformer_conjunctive.yaml --fold 0 --seed 42
python scripts/train.py --config configs/table2/atlas1_dgcnn_instance.yaml --seed 42 --wandb
```

Outputs go to `outputs/<experiment>/fold<k>_seed<s>/` (or `seed<s>/` for datasets without
folds): `log.csv`, `best.pt`, `last.pt`, `summary.json` (test metrics of the selected and of
the last checkpoint, best validation values) and the test predictions.
`--override key=value` changes single entries, e.g. `--override train.epochs=50 train.val_split=holdout`.
Checkpoints hold `model_state_dict`, the full `config`, `pointmil_version`, `metrics` and
`epoch`; `pointmil.load_checkpoint` also reads the checkpoints of the original code bases.

**Recipes.** `pointmil` is the original PointMIL training code. `openpoints` (ATLAS-1,
ScanObjectNN, PointMLPElite) is the training strategy of
[PointNeXt](https://github.com/guochengqian/PointNeXt) as implemented in
[OpenPoints](https://github.com/guochengqian/openpoints): its optimiser and weight-decay
groups, cosine schedule, label-smoothed cross-entropy, gradient clipping, transforms and
encoders, re-implemented here in plain PyTorch. `tests/equivalence/` checks against
OpenPoints that the encoders give the same outputs and that a training step gives the same
parameters. ModelNet40 has configs for both recipes; the PointNeXt recipe
(`configs/table2/modelnet40_openpoints_<backbone>_<pooling>.yaml`) converges much faster and
is the recommended one for new ModelNet40 experiments.

## Evaluation and interpretability

```bash
python scripts/eval.py --checkpoint <run>/best.pt --fold 0 --split test
python scripts/interpret.py metrics --checkpoint <run>/best.pt --split annotated --target_class 1 --context_k 12   # AOPCR, NDCG@n
python scripts/interpret.py importance --checkpoint <run>/best.pt --fold 0 --split test --out_dir results/importance  # per-point maps
python scripts/noise_robustness.py --checkpoint <run>/best.pt --fold 0 --noise_points 0 50 100 200 500
python scripts/aggregate_results.py runs "outputs/table1_intra_*"     # mean and standard deviation over folds
```

`out["interpretation"]` is the raw `[B, C, N]` map (per-point class logits scaled by attention
for Additive and Conjunctive, per-point logits for Instance, the attention weights for
Attention); `model.point_importance(x)` turns it into one score per point.

## Run times

One Quadro RTX 6000 (24 GB), measured while sharing the GPU with a second run.

| Result | Command | Cost |
|---|---|---|
| Table 1, Table 2 IntrA | `bash scripts/reproduce_table1.sh` | 1 to 6 GPU hours per run (Transformer 5.3 h), about two GPU weeks in total; `reproduce_table1.sh transformer conjunctive` for one cell |
| Table 2 RBC | `bash scripts/reproduce_table2.sh rbc` | 1 to 2.5 GPU hours per fold and model |
| Table 2 ModelNet40 | `bash scripts/reproduce_table2.sh modelnet40` | 8 to 14 GPU hours per model |
| ModelNet40, PointNeXt recipe | `bash scripts/reproduce_table2.sh modelnet40_openpoints` | about 12 GPU hours per model |
| Table 2 ATLAS-1 | `bash scripts/reproduce_table2.sh atlas1` | about 1 GPU hour per model |
| Table 2 ScanObjectNN | `bash scripts/reproduce_table2.sh scanobjectnn` | 3 to 8 GPU hours per model |
| Table 2 IntrA PointMLPElite | `bash scripts/reproduce_table2.sh intra_pointmlpelite` | about 3.3 GPU hours per fold and head |
| Supp. Table 3, Fig. 6 | `bash scripts/reproduce_supp_table3_fig6.sh` | 95 runs of about 6.5 h |
| Supp. Fig. 12 | `scripts/noise_robustness.py` | minutes |

## Pretrained models

One Hub repository per dataset (`Sentinal4D/pointmil-<dataset>`). Each model is a folder
`<backbone>/<pooling>/<fold>/` with `model.safetensors`, `config.yaml` and `metrics.json`;
models trained with this code for additional folds sit under `retrained/`. Backbones are
`transformer`, `dgcnn`, `pointnet`, `curvenet` and `pointmlpelite`; folds are `fold0` to
`fold4` (IntrA, RBC), `split0` (ATLAS-1) or `official` (ModelNet40, ScanObjectNN).

```python
from pointmil.hub import load_pretrained, list_pretrained
model, info = load_pretrained("rbc", "dgcnn", "instance", fold=3, retrained=True)
print([m["path"] for m in list_pretrained("atlas1")])
```

The scripts also take a downloaded folder as `--checkpoint`:

```bash
hf download Sentinal4D/pointmil-intra --include "transformer/conjunctive/fold0/*" --local-dir hub
python scripts/eval.py --checkpoint hub/transformer/conjunctive/fold0 --split test
```

## Your own data

[cellshape-helper](https://github.com/Sentinal4D/cellshape-helper) converts 3D segmentation
masks to meshes and point clouds.

1. Sub-class `pointmil.data.base.PointCloudDataset`: fill `self.items` with one
   `{"path", "name", "label"}` dict per cloud of the requested `split` and implement
   `_load(item)` to return `{"pos": [M, 3] array}` (plus `"normals"`).
   `pointmil/data/rbc.py` is a short example.
2. Copy a config (e.g. `configs/table2/atlas1_transformer_additive.yaml` for xyz clouds) and
   set `data.name` to the import path of your class (`mypackage.cells:MyCells`), `data.root`,
   `data.split_file`, `model.pooling.num_classes` and `train.val_split`.
3. `python scripts/train.py --config my.yaml --seed 42`, then `scripts/interpret.py importance`.
   For a different number of points set `data.num_points` and `model.pooling.num_points`
   together.

Backbones: `transformer`, `pointnet`, `dgcnn`, `curvenet_mil`, `openpoints_dgcnn`,
`openpoints_pointnet`, `openpoints_curvenet`, `pointmlp_elite`. Pooling heads: `instance`,
`attention`, `additive`, `conjunctive`, `gap` and `openpoints`. All run on CPU.

## Tests

`pytest tests --ignore=tests/equivalence` runs the CPU unit tests and `ruff check .` lints.
`tests/equivalence/` compares the ported modules with the original code bases and runs only
when they are available (`POINTMIL_REPO_A`: the original PointMIL code, `POINTMIL_REPO_B`:
the PointNeXt fork).

## Repository structure

```
pointmil/     the package: backbones, pooling, models, data, interpret, engine, hub
configs/      one YAML per experiment (table1/, table2/, fig6_k_sweep/, smoke/)
splits/       fold and split files
scripts/      training, evaluation, interpretation, data preparation and reproduction scripts
tests/        unit tests and equivalence tests
docs/         this guide, EXPERIMENTS.md, reproducibility_template.md
notebooks/    quickstart.ipynb
```
