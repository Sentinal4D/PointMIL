#!/usr/bin/env bash
# Train every backbone x one pooling head for two epochs on the synthetic dataset (CPU), then evaluate and interpret.
# Expected runtime: a few minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${SMOKE_OUT:-outputs/smoke}
PY=${PYTHON:-python}
rm -rf "$OUT"
run() {  # name backbone-yaml-fragment pooling-yaml-fragment feature_keys [transforms]
  local name=$1 backbone=$2 pooling=$3 fk=$4 tf=${5:-"[]"}
  local train_tf=${6:-"[{name: random_scale_translate, scale: [0.8, 1.2], shift: 0.2}]"}
  echo "=== smoke: $name"
  $PY scripts/train.py --config configs/smoke/synthetic_transformer_conjunctive.yaml --device cpu --out_dir "$OUT" --seed 0 --run_name "$name" \
     --override "model.backbone=$backbone" "model.pooling=$pooling" "data.feature_keys=$fk" "data.test_transforms=$tf" "data.train_transforms=$train_tf" "experiment.name=smoke" >/dev/null
  $PY scripts/eval.py --checkpoint "$OUT/smoke/$name/best.pt" --split test --device cpu --batch_size 4 >/dev/null
}
run transformer_conjunctive '{name: transformer, in_channels: 6, coord_dims: 3, num_k: [16, 16], channel_dim: [32, 64], channel_raise: [32, 64], dim_k: 8, heads: 8}' \
    '{name: conjunctive, num_classes: 2, dropout: 0.1, attention_hidden: 8, non_linear: false, non_linear_v1: false, positional_encoding: true, contextual_attention: true, context_k: 4, num_points: 256}' '[pos, normals]'
run pointnet_additive '{name: pointnet}' \
    '{name: additive, num_classes: 2, dropout: 0.1, attention_hidden: 8, non_linear: false, non_linear_v1: false, positional_encoding: true, contextual_attention: true, context_k: 4, num_points: 256}' '[pos, normals]'
run dgcnn_instance '{name: dgcnn, k: 8, emb_dims: 64}' \
    '{name: instance, num_classes: 2, dropout: 0.1, attention_hidden: 8, non_linear: true, non_linear_v1: false, positional_encoding: false, contextual_attention: false, context_k: 0, num_points: 256}' '[pos]'
run dgcnn_attention '{name: dgcnn, k: 8, emb_dims: 64}' \
    '{name: attention, num_classes: 2, dropout: 0.1, attention_hidden: 8, non_linear: false, non_linear_v1: true, positional_encoding: true, contextual_attention: true, context_k: 4, num_points: 256}' '[pos]'
run dgcnn_gap '{name: dgcnn, k: 8, emb_dims: 64}' \
    '{name: gap, num_classes: 2, dropout: 0.1, attention_hidden: 8, non_linear: false, non_linear_v1: false, positional_encoding: true, contextual_attention: false, context_k: 0, num_points: 256}' '[pos]'
run curvenet_mil_instance '{name: curvenet_mil, k: 8, instance_input: features_with_global, num_points: 256}' \
    '{name: instance, num_classes: 2}' '[pos]'
run op_transformer_additive '{name: transformer, in_channels: 4, coord_dims: 4, num_k: [16, 16], channel_dim: [32, 64], channel_raise: [32, 64], dim_k: 8, heads: 8}' \
    '{name: openpoints, pooling: additive, num_classes: 2, instance_head: mlp_bn_points, num_points: 256, mlp_dims: [32, 16], dropout: 0.5, attention_hidden: 8}' '[pos, heights]' \
    '[{name: points_to_tensor}, {name: center_and_normalize, gravity_dim: 1}]' '[{name: points_to_tensor}, {name: center_and_normalize, gravity_dim: 1}]'
run op_dgcnn_conjunctive '{name: openpoints_dgcnn, in_channels: 3, channels: 16, embed_dim: 64, n_blocks: 5, k: 8}' \
    '{name: openpoints, pooling: conjunctive, num_classes: 2, instance_head: linear, num_points: 256, mlp_dims: [32, 16], dropout: 0.5, attention_hidden: 8}' '[pos]'
run op_pointnet_gap '{name: openpoints_pointnet, in_channels: 3, input_transform: true, feature_transform: true}' \
    '{name: openpoints, pooling: gap, num_classes: 2, instance_head: linear, num_points: 256, mlp_dims: [32, 16], dropout: 0.5, attention_hidden: 8}' '[pos]'
run op_curvenet_instance '{name: openpoints_curvenet, num_classes: 2, k: 8, num_points: 256}' \
    '{name: openpoints, pooling: instance, num_classes: 2, instance_head: mlp_bn_points, num_points: 256, mlp_dims: [32, 16], dropout: 0.5, attention_hidden: 8}' '[pos]'
run pointmlp_elite_attention '{name: pointmlp_elite, in_channels: 3, reducers: [1, 1, 1, 1]}' \
    '{name: openpoints, pooling: attention, num_classes: 2, instance_head: linear, num_points: 256, mlp_dims: [32, 16], dropout: 0.5, attention_hidden: 8}' '[pos]'
echo "=== smoke: interpretability metrics"
$PY scripts/interpret.py metrics --checkpoint "$OUT/smoke/transformer_conjunctive/best.pt" --split annotated --device cpu --batch_size 4 --step 0.25 --n_random 1 --out "$OUT/smoke_interp.json"
echo "smoke test passed"
