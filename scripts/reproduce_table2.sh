#!/usr/bin/env bash
# Table 2 rows other than IntrA (those are produced by reproduce_table1.sh).
#   RBC:          5 folds x {transformer, pointnet, dgcnn, curvenet, pointmlpelite} x poolings (1 to 2.5 h per fold)
#   ModelNet40:   {transformer, dgcnn, pointnet} x poolings, official split, original PointMIL recipe (8 to 14 h each)
#   ModelNet40 (modelnet40_openpoints): {transformer, dgcnn, pointnet, pointmlpelite} x poolings with the PointNeXt /
#                 OpenPoints recipe (about 12 h each); not part of "all"
#   ATLAS-1:      {transformer, dgcnn, pointnet, curvenet, pointmlpelite} x poolings, OpenPoints recipe, single split (about 1 h each)
#   ScanObjectNN: 5 models, OpenPoints recipe (3 to 8 h each)
#   IntrA PointMLPElite: 5 folds x poolings, OpenPoints recipe (about 3 to 8 h per fold)
# Times measured on one Quadro RTX 6000 (24 GB).
# Usage: bash scripts/reproduce_table2.sh [rbc|modelnet40|modelnet40_openpoints|atlas1|scanobjectnn|intra_pointmlpelite|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${OUT:-outputs}
what=${1:-all}
case "$what" in
  rbc|modelnet40|modelnet40_openpoints|atlas1|scanobjectnn|intra_pointmlpelite|all) ;;
  *) echo "unknown argument '$what'; use rbc, modelnet40, modelnet40_openpoints, atlas1, scanobjectnn, intra_pointmlpelite or all" >&2; exit 1 ;;
esac
run_folds() { local cfg=$1; local name; name=$(basename "$cfg" .yaml); for fold in ${FOLDS:-0 1 2 3 4}; do
  [ -f "$OUT/table2_${name}/fold${fold}_seed42/summary.json" ] || $PY scripts/train.py --config "$cfg" --fold "$fold" --seed 42 --out_dir "$OUT"; done; }
run_single() { local cfg=$1; local name; name=$(basename "$cfg" .yaml); [ -f "$OUT/table2_${name}/seed42/summary.json" ] || $PY scripts/train.py --config "$cfg" --seed 42 --out_dir "$OUT"; }
case "$what" in
  rbc|all) for cfg in configs/table2/rbc_*.yaml; do run_folds "$cfg"; done ;;&
  modelnet40|all) for cfg in configs/table2/modelnet40_*.yaml; do [[ $cfg == *_openpoints_* ]] || run_single "$cfg"; done ;;&
  modelnet40_openpoints) for cfg in configs/table2/modelnet40_openpoints_*.yaml; do run_single "$cfg"; done ;;&
  atlas1|all) for cfg in configs/table2/atlas1_*.yaml; do run_single "$cfg"; done ;;&
  scanobjectnn|all) for cfg in configs/table2/scanobjectnn_*.yaml; do run_single "$cfg"; done ;;&
  intra_pointmlpelite|all) for cfg in configs/table2/intra_pointmlpelite_*.yaml; do run_folds "$cfg"; done ;;
esac
$PY scripts/aggregate_results.py runs "$OUT/table2_*" --out results/table2.md
