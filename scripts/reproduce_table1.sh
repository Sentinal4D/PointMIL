#!/usr/bin/env bash
# Table 1 (IntrA interpretability) and the IntrA rows of Table 2: train 4 backbones x 5 pooling heads x 5 folds,
# then compute AOPCR / NDCG@n on the annotated aneurysm segments with contextual attention k = 12 at inference
# (as in the paper) and aggregate.
# With the CurveNet backbone it also trains the CurveNet dagger baseline (CurveNet classifier without MIL).
# Runtime on one Quadro RTX 6000 (24 GB): Transformer about 5.3 h per fold (250 epochs), DGCNN and PointNet 1 to 3 h,
# CurveNet up to 6 h (300 epochs); about two GPU weeks for the full table. Pass a subset to run less, e.g.
#   bash scripts/reproduce_table1.sh transformer conjunctive
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
BACKBONES=${1:-"transformer dgcnn pointnet curvenet"}
POOLINGS=${2:-"instance attention additive conjunctive gap"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
OUT=${OUT:-outputs}
for bb in $BACKBONES; do for pool in $POOLINGS; do
  cfg=configs/table1/intra_${bb}_${pool}.yaml
  for fold in $FOLDS; do
    run=$OUT/table1_intra_${bb}_${pool}/fold${fold}_seed42
    [ -f "$run/summary.json" ] || $PY scripts/train.py --config "$cfg" --fold "$fold" --seed 42 --out_dir "$OUT"
    k=12; [ "$bb" = curvenet ] && k=0; [ "$pool" = gap ] && k=0
    [ -f "$run/interpretability.json" ] || $PY scripts/interpret.py metrics --checkpoint "$run/best.pt" --split annotated --target_class 1 --context_k $k --out "$run/interpretability.json"
  done
done; done
if [[ " $BACKBONES " == *" curvenet "* ]]; then
  for fold in $FOLDS; do
    [ -f "$OUT/table2_intra_curvenet_dagger_baseline/fold${fold}_seed42/summary.json" ] || \
      $PY scripts/train.py --config configs/table2/intra_curvenet_dagger_baseline.yaml --fold "$fold" --seed 42 --out_dir "$OUT"
  done
fi
# classification metrics and the mean AOPCR / NDCG@n over folds
$PY scripts/aggregate_results.py runs "$OUT"/table1_intra_* "$OUT"/table2_intra_curvenet_dagger_baseline --out results/table1.md
