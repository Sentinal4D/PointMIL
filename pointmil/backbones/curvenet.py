"""CurveNet without point downsampling (Supplementary Section 10).

Two classes are provided because the paper used two code paths:

* ``CurveNetMIL`` is the fused CurveNet + MIL model of the original PointMIL
  code (IntrA and RBC "CurveNet dagger" rows and the CurveNet column of Table 1).
  The pooling head is part of the class, as in the checkpoints. Two layouts of
  the instance classifier input exist in the trained models and are selected
  with ``instance_input``: ``features`` (1024-d per-point features) and
  ``features_with_global`` (per-point features concatenated with the global max
  and mean feature, 3072-d).
* ``OpenPointsCurveNet`` is the ``CurveNetNoDown`` encoder of the OpenPoints
  fork (ATLAS-1 rows); it returns 1024-d per-point features for a separate
  ``OpenPointsMILHead``.

Both keep every one of the 1024 input points (the CIC blocks sample ``npoint``
equal to the number of input points, so farthest point sampling is a no-op
permutation) and use the CurveNet layers in ``_curvenet_layers.py``.

Adapted from MILLET (Early et al., ICLR 2024,
https://github.com/JAEarly/MILTimeSeriesClassification, ``millet/model/pooling.py``).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 (see ``third_party_licenses/MILLET_LICENSE_Apache-2.0``).
Applies to the MIL pooling heads of ``CurveNetMIL``; the CurveNet layers are MIT (see
``_curvenet_layers.py``). Changes for PointMIL: the heads pool points instead of time
steps and are fused with the CurveNet classifier layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pointmil.backbones._curvenet_layers import CIC, LPFA, curve_config


def _build_trunk(module: nn.Module, k: int, setting: str, npoint: int = 1024) -> None:
    """Attach the CurveNet encoder layers to ``module`` (names match the original code)."""
    if setting not in curve_config:
        raise ValueError(f"Unknown CurveNet setting '{setting}'")
    additional_channel = 32
    cfg = curve_config[setting]
    module.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)
    module.cic11 = CIC(npoint=npoint, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=cfg[0])
    module.cic12 = CIC(npoint=npoint, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=cfg[0])
    module.cic21 = CIC(npoint=npoint, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=cfg[1])
    module.cic22 = CIC(npoint=npoint, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=cfg[1])
    module.cic31 = CIC(npoint=npoint, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=cfg[2])
    module.cic32 = CIC(npoint=npoint, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=cfg[2])
    module.cic41 = CIC(npoint=npoint, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=cfg[3])
    module.cic42 = CIC(npoint=npoint, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=cfg[3])
    module.conv0 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))


def _run_trunk(module: nn.Module, xyz_cf: torch.Tensor) -> torch.Tensor:
    """Run the encoder on channels-first coordinates ``[B, 3, N]``; returns ``[B, 512, N]``."""
    l0_points = module.lpfa(xyz_cf, xyz_cf)
    l1_xyz, l1_points = module.cic11(xyz_cf, l0_points)
    l1_xyz, l1_points = module.cic12(l1_xyz, l1_points)
    l2_xyz, l2_points = module.cic21(l1_xyz, l1_points)
    l2_xyz, l2_points = module.cic22(l2_xyz, l2_points)
    l3_xyz, l3_points = module.cic31(l2_xyz, l2_points)
    l3_xyz, l3_points = module.cic32(l3_xyz, l3_points)
    l4_xyz, l4_points = module.cic41(l3_xyz, l3_points)
    l4_xyz, l4_points = module.cic42(l4_xyz, l4_points)
    return l4_points


class CurveNetMIL(nn.Module):
    """Fused CurveNet encoder and MIL head of the original PointMIL code.

    Args:
        num_classes: number of classes.
        pooling: ``instance``, ``additive``, ``conjunctive``, ``attention``,
            ``gap`` or ``original`` (the CurveNet classifier, no interpretation).
        k: neighbourhood size of the CurveNet layers.
        setting: CurveNet curve configuration.
        instance_input: ``features`` or ``features_with_global`` (see module docstring).
        num_points: points per cloud (the instance classifier normalises over the point axis).
        attention_hidden: hidden width of the gated attention MLP.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pooling: str = "instance",
        k: int = 20,
        setting: str = "default",
        instance_input: str = "features",
        num_points: int = 1024,
        attention_hidden: int = 8,
    ):
        super().__init__()
        if pooling not in {"instance", "additive", "conjunctive", "attention", "gap", "original"}:
            raise ValueError(f"Unsupported pooling '{pooling}'")
        if instance_input not in {"features", "features_with_global"}:
            raise ValueError(f"Unknown instance_input '{instance_input}'")
        self.num_classes = num_classes
        self.pooling = pooling
        self.instance_input = instance_input
        self.num_points = num_points
        self.out_channels = 1024
        _build_trunk(self, k, setting, npoint=num_points)
        if pooling != "gap":
            # The per-point classifier is not used by the gap (CLAIM) baseline; its checkpoints carry
            # one with yet another input width (2048), which is ignored as unexpected keys.
            in_dim = 1024 * 3 if instance_input == "features_with_global" else 1024
            self.conv1 = nn.Linear(in_dim, 512, bias=False)
            self.conv2 = nn.Linear(512, num_classes)
            self.bn1 = nn.BatchNorm1d(num_points)
            self.dp1 = nn.Dropout(p=0.5)
        self.attention_head = nn.Sequential(nn.Linear(1024, attention_hidden), nn.Tanh(), nn.Linear(attention_hidden, 1), nn.Sigmoid())
        self.conv1_att = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2_att = nn.Linear(512, num_classes)
        self.bn1_att = nn.BatchNorm1d(512)
        self.dp1_att = nn.Dropout(p=0.5)
        if pooling == "gap":
            self.gap_cls = nn.Linear(512, num_classes)

    def _bag_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """CurveNet-style classifier on the concatenated max and mean pooled features ``[B, 1024, N]``."""
        pooled = torch.cat((F.adaptive_max_pool1d(x, 1), F.adaptive_avg_pool1d(x, 1)), dim=1).squeeze(-1)
        h = F.relu(self.bn1_att(self.conv1_att(pooled).unsqueeze(-1)), inplace=True).squeeze(-1)
        return self.conv2_att(self.dp1_att(h))

    def _instance_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Per-point logits ``[B, N, C]`` from per-point features ``x`` ``[B, 1024, N]``."""
        if self.instance_input == "features_with_global":
            n = x.shape[-1]
            x_max = torch.max(x, dim=-1)[0].unsqueeze(-1).repeat(1, 1, n)
            x_mean = torch.mean(x, dim=-1).unsqueeze(-1).repeat(1, 1, n)
            x = torch.cat([x_max, x_mean, x], dim=1)
        h = F.relu(self.bn1(self.conv1(x.transpose(2, 1))), inplace=True)
        return self.conv2(self.dp1(h))

    def forward(self, points: torch.Tensor) -> dict:
        xyz = points[..., :3]
        l4_points = _run_trunk(self, xyz.permute(0, 2, 1).contiguous())  # [B, 512, N]
        x = self.conv0(l4_points)  # [B, 1024, N]
        out: dict = {"xyz": xyz}
        if self.pooling == "original":
            logits = self._bag_classifier(x)
            out.update(bag_logits=logits, interpretation=torch.zeros(x.shape[0], self.num_classes, x.shape[-1], device=x.device))
            return out
        if self.pooling == "gap":
            cam = self.gap_cls.weight @ l4_points
            out.update(bag_logits=self.gap_cls(l4_points.mean(dim=-1)), interpretation=cam)
            return out
        if self.pooling == "instance":
            instance_logits = self._instance_logits(x)
            out.update(bag_logits=instance_logits.mean(dim=1), interpretation=instance_logits.transpose(2, 1), instance_logits=instance_logits.transpose(2, 1))
            return out
        attention = self.attention_head(x.transpose(2, 1))  # [B, N, 1]
        if self.pooling == "attention":
            weighted = x * attention.transpose(2, 1)
            out.update(bag_logits=self._bag_classifier(weighted), interpretation=attention.repeat(1, 1, self.num_classes).transpose(2, 1), attention=attention)
            return out
        if self.pooling == "additive":
            instance_logits = self._instance_logits(x * attention.transpose(2, 1))
            out.update(bag_logits=instance_logits.mean(dim=1), interpretation=(instance_logits * attention).transpose(2, 1), instance_logits=instance_logits.transpose(2, 1), attention=attention)
            return out
        # conjunctive
        instance_logits = self._instance_logits(x)
        weighted = instance_logits * attention
        out.update(bag_logits=weighted.mean(dim=1), interpretation=weighted.transpose(2, 1), instance_logits=instance_logits.transpose(2, 1), attention=attention)
        return out

    @staticmethod
    def interpret(output: dict) -> torch.Tensor:
        """Point-level interpretation ``[B, C, N]`` from a forward output."""
        return output["interpretation"]

    @torch.no_grad()
    def point_importance(self, points: torch.Tensor, target_class: torch.Tensor | int | None = None) -> torch.Tensor:
        """Per-point importance ``[B, N]``, as ``PointMIL.point_importance``."""
        from pointmil.models.pointmil import importance_from_output

        return importance_from_output(self.forward(points), target_class)


class OpenPointsCurveNet(nn.Module):
    """``CurveNetNoDown`` encoder of the OpenPoints fork; returns ``(xyz, features [B, 1024, N])``.

    The unused classifier layers of the original class are kept so that its
    checkpoints load without key mismatches.
    """

    def __init__(self, num_classes: int = 40, k: int = 20, setting: str = "default", num_points: int = 1024):
        super().__init__()
        self.out_channels = 1024
        _build_trunk(self, k, setting, npoint=num_points)
        self.conv1 = nn.Linear(1024, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = points[..., :3].contiguous()
        l4_points = _run_trunk(self, xyz.permute(0, 2, 1).contiguous())
        return xyz, self.conv0(l4_points)
