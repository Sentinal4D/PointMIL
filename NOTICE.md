# Third-party code and licences

PointMIL is released under the Sentinal4D Non-Commercial End User License Agreement
(see `LICENSE`). That agreement covers the code written by the PointMIL authors.
The components below were adapted from other projects and remain under their own
licences, whose full texts are in `third_party_licenses/`. Where files were
modified, the modifications are listed.

| Component | Where in this repository | Origin | Licence | Modifications |
|---|---|---|---|---|
| MIL pooling formulations (Instance, Attention, Additive, Conjunctive) and the AOPCR / NDCG@n metrics | `pointmil/pooling/*` (except `contextual.py`, which is PointMIL's own), the MIL heads in `pointmil/backbones/curvenet.py`, `pointmil/interpret/aopcr.py`, `pointmil/interpret/ndcg.py` | MILLET, Early et al., ICLR 2024 ([JAEarly/MILTimeSeriesClassification](https://github.com/JAEarly/MILTimeSeriesClassification)), Copyright Amazon.com, Inc. or its affiliates | Apache-2.0 | Pooling heads re-implemented for point clouds (positional encoding, contextual attention, point-axis normalisation); AOPCR removes points instead of masking them; function signatures adapted. Every adapted file keeps the Amazon copyright and SPDX header and states its changes. |
| OpenPoints encoders and MIL head (`OpenPointsDGCNN`, `OpenPointsPointNet`, `OpenPointsCurveNet`, `PointMLPEncoder`, `OpenPointsMILHead`), point cloud transforms, `SmoothCrossEntropy` | `pointmil/backbones/dgcnn.py` (second half), `pointnet.py` (second half), `curvenet.py` (`OpenPointsCurveNet`), `pointmlp.py`, `pointmil/pooling/openpoints_head.py`, `pointmil/data/transforms.py`, `pointmil/engine/losses.py` | OpenPoints / PointNeXt, Guocheng Qian et al. ([guochengqian/openpoints](https://github.com/guochengqian/openpoints)) | MIT | Rewritten without the OpenPoints registry and CUDA extensions (gather instead of `grouping_operation`, pure PyTorch farthest point sampling); the MIL branch added by the PointMIL authors to `ClsHead` is reproduced as `OpenPointsMILHead`. Submodule names kept for checkpoint compatibility. |
| CurveNet layers (`Walk`, `LPFA`, `CIC`, `CurveAggregation`, ...) | `pointmil/backbones/_curvenet_layers.py`, `pointmil/backbones/curvenet.py` | CurveNet, Tiange Xiang et al., ICCV 2021 ([tiangexiang/CurveNet](https://github.com/tiangexiang/CurveNet)) | MIT | Hard-coded CUDA devices replaced by the input device; cache clearing removed; downsampling disabled and MIL head added (the paper's "CurveNet dagger"). |
| PointMLP / PointMLPElite | `pointmil/backbones/pointmlp.py` | PointMLP, Xu Ma et al., ICLR 2022 ([ma-xu/pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)) via OpenPoints | Apache-2.0 | Reducers set to 1 (no sampling) for the paper's "PointMLPElite dagger"; farthest point sampling in pure PyTorch; input point order kept when no reduction is applied. |
| DGCNN | `pointmil/backbones/dgcnn.py` (`DGCNNExtractor`, `get_graph_feature`) | DGCNN, Yue Wang et al., 2019; PyTorch port by An Tao ([antao97/dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch)) | MIT | Classification head removed so that per-point features are returned. |
| PointNet encoder, spatial transformers, k-NN utilities | `pointmil/backbones/pointnet.py`, `pointmil/backbones/knn.py` | Pointnet_Pointnet2_pytorch, Xu Yan ([yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)) | MIT | Global max pooling removed so that per-point features are returned; identity matrices created on the input device. |
| Transformer block of the PointMIL feature extractor | `pointmil/backbones/transformer.py` | Adapted from the transformer block of 3DMedPT, Jianhui Yu et al., 2021 ([crane-papercode/3DMedPT](https://github.com/crane-papercode/3DMedPT)) | No licence file is provided by that repository | Point sampling and multi-graph reasoning removed; relative positional MLP, EdgeConv-style query and two-layer stacking as described in the paper. The PointMIL authors acknowledge 3DMedPT as the origin of this design. |

## NOTICE text of the Apache-2.0 components

MILLET ([JAEarly/MILTimeSeriesClassification](https://github.com/JAEarly/MILTimeSeriesClassification)) ships the following NOTICE,
reproduced here as required by section 4(d) of the Apache License 2.0 (also in
`third_party_licenses/MILLET_NOTICE`):

```
MILTimeSeriesClassification
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
```

PointMLP ([ma-xu/pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)) has no NOTICE file.

## Datasets

Datasets are not redistributed with the code except for the split files under
`splits/`. IntrA (Yang et al., 2020), the 3D red blood cell dataset (Simionato et
al., 2021, [raw data](https://doi.org/10.5281/zenodo.4670205); [3D masks](https://doi.org/10.5281/zenodo.7031924)
by Waibel et al., CC BY 4.0), ModelNet40 (Wu et al., 2015) and
ScanObjectNN (Uy et al., 2019) are subject to their own terms. The RBC point clouds
are not redistributed: `scripts/prepare_rbc.py` builds them from the public masks.
ATLAS-1 is released by Sentinal4D under the Creative Commons Attribution 4.0 licence
(CC BY 4.0), as stated in the paper ([ATLAS-1 dataset](https://huggingface.co/datasets/Sentinal4D/ATLAS-1)).
