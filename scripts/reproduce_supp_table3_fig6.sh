#!/usr/bin/env bash
# Supplementary Table 3 and Fig. 6: Transformer on IntrA with contextual attention k in {0, 4, 8, 12, 16, 20}
# for additive, attention and conjunctive pooling (plus instance without context), 5 folds each.
# Classification metrics come from training; AOPCR / NDCG@n from scripts/interpret.py with the same k at inference.
# About 6.5 h per run (300 epochs) on one Quadro RTX 6000.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${OUT:-outputs}
for cfg in configs/fig6_k_sweep/*.yaml; do
  name=$(basename "$cfg" .yaml); k=${name##*_k}
  for fold in ${FOLDS:-0 1 2 3 4}; do
    run=$OUT/fig6_${name}/fold${fold}_seed42
    [ -f "$run/summary.json" ] || $PY scripts/train.py --config "$cfg" --fold "$fold" --seed 42 --out_dir "$OUT"
    [ -f "$run/interpretability.json" ] || $PY scripts/interpret.py metrics --checkpoint "$run/best.pt" --split annotated --target_class 1 --context_k "$k" --out "$run/interpretability.json"
  done
done
$PY scripts/aggregate_results.py runs "$OUT"/fig6_* --out results/fig6.md
