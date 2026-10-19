#!/usr/bin/env python
"""Generate the experiment configs under ``configs/`` (one YAML per paper experiment).

Every file states every hyperparameter explicitly. Values were recovered from
the training arguments stored in the checkpoints, the wandb configs and the
OpenPoints yaml files of the runs behind the paper (see docs/EXPERIMENTS.md). Re-run this script after changing a recipe here; do not
edit the generated files by hand.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"

POOLINGS = ["instance", "attention", "additive", "conjunctive", "gap"]

# --------------------------------------------------------------------------- data sections
INTRA_DATA = dict(
    name="intra", root="${POINTMIL_DATA}/IntrA", split_dir="splits/intra", num_points=1024, sampling="random",
    normalise=True, feature_keys=["pos", "normals"], batch_size=8, val_batch_size=8, num_workers=4,
    train_transforms=[dict(name="random_scale_translate", scale=[0.8, 1.2], shift=0.2)], test_transforms=[],
)
RBC_DATA = dict(INTRA_DATA, name="rbc", root="${POINTMIL_DATA}/RBC", split_dir="splits/rbc")
MN40_DATA = dict(
    name="modelnet40", root="${POINTMIL_DATA}/ModelNet40", num_points=1024, sampling="first", normalise=False,
    feature_keys=["pos"], batch_size=32, val_batch_size=32, num_workers=6,
    train_transforms=[dict(name="dgcnn_scale_translate", scale=[0.6666666666666666, 1.5], shift=0.2), dict(name="shuffle_points")],
    test_transforms=[],
)
ATLAS1_DATA = dict(
    name="atlas1", root="${POINTMIL_DATA}/ATLAS-1", split_file="splits/atlas1/split0.csv", source="points",
    num_points=1024, sampling="fps", normalise=True, feature_keys=["pos"], batch_size=16, val_batch_size=16,
    num_workers=6, train_transforms=[], test_transforms=[],
)
SONN_DATA = dict(
    name="scanobjectnn", root="${POINTMIL_DATA}/ScanObjectNN", uniform_sample=True, num_points=1024, sampling="first",
    normalise=False, feature_keys=["pos", "heights"], batch_size=32, val_batch_size=64, num_workers=6,
    # the OpenPoints ScanObjectNN loader shuffles the training points before its transforms
    train_transforms=[dict(name="shuffle_points"), dict(name="points_to_tensor"), dict(name="scaling", scale=[0.9, 1.1], anisotropic=True),
                      dict(name="center_and_normalize", gravity_dim=1), dict(name="rotation", angle=[0.0, 1.0, 0.0])],
    test_transforms=[dict(name="points_to_tensor"), dict(name="center_and_normalize", gravity_dim=1)],
)

# --------------------------------------------------------------------------- train sections
def train_a(epochs, patience, monitor="val_f1", opt="adamw", lr=1e-3, wd=1e-4, min_lr=1e-6, momentum=None, t_max=None):
    o = dict(name=opt, lr=lr, weight_decay=wd)
    if momentum is not None:
        o["momentum"] = momentum
    return dict(
        epochs=epochs, optimizer=o, scheduler=dict(name="cosine", t_max=t_max or epochs, min_lr=min_lr, warmup_epochs=0),
        loss=dict(name="cross_entropy", label_smoothing=0.0), grad_clip=None, checkpoint_monitor=monitor,
        early_stopping_patience=patience, val_split="test", drop_last=True, deterministic=True,
    )


def train_b(epochs=100, lr=0.1, opt="sgd", wd=2e-4, t_max=100, min_lr=0.005, label_smoothing=0.2, grad_clip=1, val_split="test"):
    """OpenPoints recipe (ATLAS-1, ScanObjectNN, PointMLPElite): as the PointNeXt classification trainer runs it."""
    o = dict(name=opt, lr=lr, weight_decay=wd, filter_bias_and_bn=True)  # OpenPoints: no weight decay on biases and norm weights
    if opt == "sgd":
        o["momentum"] = 0.9
    return dict(
        epochs=epochs, optimizer=o, scheduler=dict(name="cosine", t_max=t_max, min_lr=min_lr, warmup_epochs=0),
        loss=dict(name="cross_entropy", label_smoothing=label_smoothing), grad_clip=grad_clip, checkpoint_monitor="val_oacc",
        early_stopping_patience=0, val_split=val_split, drop_last=True, deterministic=True,
    )


# --------------------------------------------------------------------------- model sections
def transformer(in_channels=6, coord_dims=3, num_k=(32, 64)):
    return dict(name="transformer", in_channels=in_channels, coord_dims=coord_dims, num_k=list(num_k), channel_dim=[128, 256], channel_raise=[64, 256], dim_k=32, heads=8)


def head_a(pooling, num_classes, dropout=0.1, non_linear=False, non_linear_v1=False, pe=True, ctx=True, k=12, **layout):
    head = dict(name=pooling, num_classes=num_classes, dropout=dropout, attention_hidden=8, non_linear=non_linear,
                non_linear_v1=non_linear_v1, positional_encoding=pe, contextual_attention=ctx, context_k=k, num_points=1024)
    head.update(layout)  # raise_dimension / non_linear_narrow: ModelNet40 conjunctive checkpoints
    return head


def head_op(pooling, num_classes, layout="linear", attention_classifier="mlp"):
    head = dict(name="openpoints", pooling=pooling, num_classes=num_classes, instance_head=layout, num_points=1024, mlp_dims=[512, 256], dropout=0.5, attention_hidden=8)
    if pooling == "attention":
        # "linear" for the ATLAS-1 DGCNN / PointNet / PointMLPElite and ScanObjectNN PointMLPElite checkpoints
        head["attention_classifier"] = attention_classifier
    return head


N_WRITTEN = 0


def write(path: Path, exp: dict, data: dict, model: dict, train: dict):
    global N_WRITTEN
    N_WRITTEN += 1
    cfg = {"experiment": exp, "seed": 42, "data": copy.deepcopy(data), "model": copy.deepcopy(model), "train": copy.deepcopy(train)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Generated by scripts/make_configs.py. Every hyperparameter is explicit; see docs/EXPERIMENTS.md.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    # ----------------------------------------------------------------- Table 1 and Table 2 IntrA (original recipe)
    intra_backbones = {
        "transformer": (transformer(6), dict(epochs=250, patience=250), dict(ctx=True)),
        "dgcnn": (dict(name="dgcnn", k=20, emb_dims=1024), dict(epochs=250, patience=250), dict(ctx=True)),
        "pointnet": (dict(name="pointnet"), dict(epochs=250, patience=250), dict(ctx=True)),
        "curvenet": (dict(name="curvenet_mil", k=20, instance_input="features", num_points=1024), dict(epochs=300, patience=300), dict(ctx=False)),
    }
    for bb, (bcfg, tcfg, hcfg) in intra_backbones.items():
        for pool in POOLINGS:
            if bb == "transformer" and pool == "gap":
                # the CLAIM baseline run (PointMIL_IntrA_positional_encoding_gap) trained for 300 epochs, no context
                tcfg, hcfg = dict(epochs=300, patience=300), dict(ctx=False)
            else:
                tcfg, hcfg = intra_backbones[bb][1], intra_backbones[bb][2]
            model_bb = copy.deepcopy(bcfg)
            if bb == "curvenet" and pool in ("instance", "conjunctive"):
                model_bb["instance_input"] = "features_with_global"
            exp = dict(name=f"table1_intra_{bb}_{pool}", recipe="pointmil",
                       paper=f"Table 1 ({bb}, {pool}); Table 2 IntrA PointMIL({bb})" + ("; Supp. Table 3 (+ context 12)" if bb == "transformer" else "") + ("; CLAIM baseline" if pool == "gap" else ""))
            write(CFG / "table1" / f"intra_{bb}_{pool}.yaml", exp, INTRA_DATA, {"backbone": model_bb, "pooling": head_a(pool, 2, **hcfg)}, train_a(**tcfg))
    # CurveNet dagger baseline (CurveNet classifier without MIL) on IntrA and RBC
    write(CFG / "table2" / "intra_curvenet_dagger_baseline.yaml", dict(name="table2_intra_curvenet_dagger_baseline", recipe="pointmil", paper="Table 2 IntrA, CurveNet dagger (no MIL)"),
          INTRA_DATA, {"backbone": dict(name="curvenet_mil", k=20, instance_input="features", num_points=1024), "pooling": dict(name="original", num_classes=2)}, train_a(300, 300))

    # ----------------------------------------------------------------- Fig. 6 and Supp. Table 3: contextual k sweep (Transformer, IntrA)
    for pool in ("additive", "attention", "conjunctive"):
        for k in (0, 4, 8, 12, 16, 20):
            tcfg = dict(epochs=300, patience=300) if k == 0 else dict(epochs=300, patience=100)
            exp = dict(name=f"fig6_intra_transformer_{pool}_k{k}", recipe="pointmil",
                       paper=f"Fig. 6 k sweep (k={k}); Supp. Table 3 " + ("(no context)" if k == 0 else f"(+ context {k})"))
            write(CFG / "fig6_k_sweep" / f"intra_transformer_{pool}_k{k}.yaml", exp, INTRA_DATA,
                  {"backbone": transformer(6), "pooling": head_a(pool, 2, ctx=k > 0, k=k if k > 0 else 12)}, train_a(**tcfg))
    write(CFG / "fig6_k_sweep" / "intra_transformer_instance_k0.yaml", dict(name="fig6_intra_transformer_instance_k0", recipe="pointmil", paper="Supp. Table 3 Instance (no attention, hence no context)"),
          INTRA_DATA, {"backbone": transformer(6), "pooling": head_a("instance", 2, ctx=False)}, train_a(300, 300))

    # ----------------------------------------------------------------- Table 2 RBC (original recipe)
    rbc_tf = dict(RBC_DATA, batch_size=16, val_batch_size=16)
    for pool in ("instance", "attention", "additive", "conjunctive"):
        write(CFG / "table2" / f"rbc_transformer_{pool}.yaml", dict(name=f"table2_rbc_transformer_{pool}", recipe="pointmil", paper="Table 2 RBC PointMIL(Trans.)"),
              rbc_tf, {"backbone": transformer(6), "pooling": head_a(pool, 9, non_linear=True)}, train_a(250, 250))
    for pool in ("instance", "attention", "additive", "conjunctive"):
        write(CFG / "table2" / f"rbc_pointnet_{pool}.yaml", dict(name=f"table2_rbc_pointnet_{pool}", recipe="pointmil", paper="Table 2 RBC PointMIL(PointNet)"),
              dict(RBC_DATA, feature_keys=["pos"]), {"backbone": dict(name="pointnet"), "pooling": head_a(pool, 9, dropout=0.5, non_linear=True)}, train_a(250, 100))
        write(CFG / "table2" / f"rbc_dgcnn_{pool}.yaml", dict(name=f"table2_rbc_dgcnn_{pool}", recipe="pointmil", paper="Table 2 RBC PointMIL(DGCNN)"),
              dict(RBC_DATA, feature_keys=["pos"]), {"backbone": dict(name="dgcnn", k=20, emb_dims=1024), "pooling": head_a(pool, 9, dropout=0.5, non_linear=True)}, train_a(250, 250))
    for pool in ("instance", "attention", "additive", "conjunctive", "original"):
        write(CFG / "table2" / f"rbc_curvenet_{pool}.yaml", dict(name=f"table2_rbc_curvenet_{pool}", recipe="pointmil", paper="Table 2 RBC PointMIL(CurveNet dagger)" if pool != "original" else "Table 2 RBC CurveNet dagger baseline"),
              dict(RBC_DATA, feature_keys=["pos"]), {"backbone": dict(name="curvenet_mil", k=20, instance_input="features", num_points=1024), "pooling": dict(name=pool, num_classes=9)}, train_a(250, 100))

    # ----------------------------------------------------------------- Table 2 ModelNet40 (original recipe, SGD)
    sgd = dict(opt="sgd", lr=0.1, wd=1e-4, min_lr=1e-3, momentum=0.9, monitor="val_oacc")
    for pool in ("instance", "attention", "additive", "conjunctive"):
        write(CFG / "table2" / f"modelnet40_transformer_{pool}.yaml", dict(name=f"table2_modelnet40_transformer_{pool}", recipe="pointmil", paper="Table 2 ModelNet40 PointMIL(Trans.)"),
              MN40_DATA, {"backbone": transformer(3, num_k=(20, 20)), "pooling": head_a(pool, 40, dropout=0.5, non_linear=True, pe=False, ctx=False, **({"raise_dimension": True} if pool == "conjunctive" else {}))},
              train_a(250, 250, **sgd) if pool == "conjunctive" else train_a(200, 200, **sgd))
        write(CFG / "table2" / f"modelnet40_dgcnn_{pool}.yaml", dict(name=f"table2_modelnet40_dgcnn_{pool}", recipe="pointmil", paper="Table 2 ModelNet40 PointMIL(DGCNN)"),
              MN40_DATA, {"backbone": dict(name="dgcnn", k=20, emb_dims=1024), "pooling": head_a(pool, 40, dropout=0.5, non_linear=True, pe=False, ctx=pool != "instance", k=12)}, train_a(250, 250, **sgd))
        write(CFG / "table2" / f"modelnet40_pointnet_{pool}.yaml", dict(name=f"table2_modelnet40_pointnet_{pool}", recipe="pointmil", paper="Table 2 ModelNet40 PointMIL(PointNet)"),
              MN40_DATA, {"backbone": dict(name="pointnet"), "pooling": head_a(pool, 40, dropout=0.5, non_linear=True, pe=False, ctx=False, **({"non_linear_narrow": True} if pool == "conjunctive" else {}))}, train_a(250, 250, **sgd))

    # ----------------------------------------------------------------- ModelNet40 with the PointNeXt / OpenPoints recipe
    # The recipe of the authors' PointNeXt ModelNet40 runs: OpenPoints ModelNet40Ply2048 loader (first 1024 points,
    # shuffled in training, scale and shift), AdamW 1e-3 / wd 0.05 without decay on biases and norm weights, cosine to
    # 1e-6 over 250 epochs, label smoothing 0.2, gradient clipping 1, selection on test overall accuracy.
    # The OpenPoints MIL heads are those of the ScanObjectNN runs.
    mn40_op_data = dict(MN40_DATA, val_batch_size=64,
                        train_transforms=[dict(name="shuffle_points"), dict(name="points_to_tensor"),
                                          dict(name="scale_and_translate", scale=[2.0 / 3.0, 1.5], shift=[0.2, 0.2, 0.2])],
                        test_transforms=[dict(name="points_to_tensor")])
    mn40_op_train = train_b(epochs=250, lr=1e-3, opt="adamw", wd=0.05, t_max=250, min_lr=1e-6, label_smoothing=0.2, grad_clip=1)
    mn40_op_backbones = {
        "transformer": (transformer(3, 3), "mlp_bn_points"),
        "dgcnn": (dict(name="openpoints_dgcnn", in_channels=3, channels=64, embed_dim=1024, n_blocks=5, k=20), "mlp_bn_points"),
        "pointnet": (dict(name="openpoints_pointnet", in_channels=3, input_transform=True, feature_transform=True), "linear"),
        "pointmlpelite": (dict(name="pointmlp_elite", in_channels=3, reducers=[1, 1, 1, 1]), "linear"),
    }
    for bb, (bcfg, layout) in mn40_op_backbones.items():
        for pool in POOLINGS:
            write(CFG / "table2" / f"modelnet40_openpoints_{bb}_{pool}.yaml",
                  dict(name=f"table2_modelnet40_openpoints_{bb}_{pool}", recipe="openpoints",
                       paper=f"ModelNet40 PointMIL({bb}) with the PointNeXt / OpenPoints training recipe"),
                  mn40_op_data, {"backbone": bcfg, "pooling": head_op(pool, 40, layout)}, mn40_op_train)

    # ----------------------------------------------------------------- Table 2 ATLAS-1 (OpenPoints recipe, single split)
    atlas_backbones = {
        "transformer": (transformer(3, 3), "linear", 16, "mlp"),
        "dgcnn": (dict(name="openpoints_dgcnn", in_channels=3, channels=64, embed_dim=1024, n_blocks=5, k=20), "linear", 16, "linear"),
        "pointnet": (dict(name="openpoints_pointnet", in_channels=3, input_transform=True, feature_transform=True), "linear", 16, "linear"),
        "curvenet": (dict(name="openpoints_curvenet", num_classes=3, k=20, num_points=1024), "mlp_bn_points", 16, "mlp"),
        "pointmlpelite": (dict(name="pointmlp_elite", in_channels=3, reducers=[1, 1, 1, 1]), "linear", 8, "linear"),
    }
    for bb, (bcfg, layout, bs, att_clf) in atlas_backbones.items():
        for pool in POOLINGS:
            write(CFG / "table2" / f"atlas1_{bb}_{pool}.yaml", dict(name=f"table2_atlas1_{bb}_{pool}", recipe="openpoints", paper=f"Table 2 ATLAS-1 PointMIL({bb})" + (" / dagger baseline (GAP)" if pool == "gap" else "")),
                  dict(ATLAS1_DATA, batch_size=bs, val_batch_size=bs), {"backbone": bcfg, "pooling": head_op(pool, 3, layout, att_clf)}, train_b())

    # ----------------------------------------------------------------- Table 2 ScanObjectNN (OpenPoints recipe)
    sonn = dict(epochs=250, lr=0.002, opt="adamw", wd=0.05, t_max=200, min_lr=1e-4, label_smoothing=0.3, grad_clip=10, val_split="test")
    sonn_models = {
        "transformer_additive": (transformer(4, 4), head_op("additive", 15, "mlp_bn_points"), 4, ["pos", "heights"]),
        "dgcnn_additive": (dict(name="openpoints_dgcnn", in_channels=4, channels=64, embed_dim=1024, n_blocks=5, k=20), head_op("additive", 15, "mlp_bn_points"), 32, ["pos", "heights"]),
        "pointnet_additive": (dict(name="openpoints_pointnet", in_channels=4, input_transform=True, feature_transform=True), head_op("additive", 15, "linear"), 32, ["pos", "heights"]),
        "pointmlpelite_attention": (dict(name="pointmlp_elite", in_channels=3, reducers=[1, 1, 1, 1]), head_op("attention", 15, "linear", "linear"), 8, ["pos"]),
        "pointmlpelite_additive": (dict(name="pointmlp_elite", in_channels=3, reducers=[1, 1, 1, 1]), head_op("additive", 15, "linear"), 8, ["pos"]),
    }
    for name, (bcfg, hcfg, bs, fk) in sonn_models.items():
        write(CFG / "table2" / f"scanobjectnn_{name}.yaml", dict(name=f"table2_scanobjectnn_{name}", recipe="openpoints", paper="Table 2 ScanObjectNN PointMIL"),
              dict(SONN_DATA, batch_size=bs, val_batch_size=max(bs, 32), feature_keys=fk), {"backbone": bcfg, "pooling": hcfg}, train_b(**sonn))

    # ----------------------------------------------------------------- PointMLPElite on IntrA and RBC (OpenPoints recipe, 5 folds)
    # Schedules of the original single-fold runs (epochs, cosine t_max); 100 / 100 where no original run exists.
    pointmlp_schedule = {("intra", "additive"): (250, 100), ("rbc", "additive"): (250, 250), ("rbc", "conjunctive"): (250, 250), ("rbc", "gap"): (100, 250)}
    for ds, data, C in (("intra", INTRA_DATA, 2), ("rbc", RBC_DATA, 9)):
        # no augmentation: the OpenPoints IntrA and RBC loaders do not apply the configured transforms
        d = dict(data, feature_keys=["pos"], batch_size=8, val_batch_size=8, train_transforms=[], sampling="random")
        for pool in POOLINGS:
            epochs, t_max = pointmlp_schedule.get((ds, pool), (100, 100))
            write(CFG / "table2" / f"{ds}_pointmlpelite_{pool}.yaml", dict(name=f"table2_{ds}_pointmlpelite_{pool}", recipe="openpoints", paper=f"Table 2 {ds.upper()} PointMIL(PointMLPElite dagger)" + (" / dagger baseline (GAP)" if pool == "gap" else "")),
                  d, {"backbone": dict(name="pointmlp_elite", in_channels=3, reducers=[1, 1, 1, 1]), "pooling": head_op(pool, C, "linear")}, train_b(epochs=epochs, t_max=t_max))

    print(f"wrote {N_WRITTEN} configs under {CFG}")


if __name__ == "__main__":
    main()
