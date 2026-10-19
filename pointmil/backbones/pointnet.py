"""PointNet per-point feature extractors (Qi et al., 2017).

Both variants derive from the PyTorch port by Xu Yan
(https://github.com/yanx27/Pointnet_Pointnet2_pytorch, MIT licence) with the
global max pooling removed so that the 1024-d per-point features are returned.

* ``PointNetExtractor``: the version used by the original PointMIL code (IntrA,
  RBC, ModelNet40 rows). Only xyz enters the network; the full input (which may
  include normals) is passed on as the "coordinates" for the pooling head, as in
  the original runs.
* ``OpenPointsPointNet``: the OpenPoints encoder (two extra 64-channel
  convolutions, arbitrary input channels) used for the ATLAS-1 and ScanObjectNN rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class STN3d(nn.Module):
    """Spatial transformer predicting a 3x3 matrix from ``[B, C, N]`` input."""

    def __init__(self, channel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 9).repeat(batchsize, 1)
        return (x + iden).view(-1, 3, 3)


class STNkd(nn.Module):
    """Feature transformer predicting a ``k x k`` matrix."""

    def __init__(self, k: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(batchsize, 1)
        return (x + iden).view(-1, self.k, self.k)


class PointNetExtractor(nn.Module):
    """PointNet encoder of the original PointMIL code; returns ``(input, features [B, 1024, N])``."""

    def __init__(self, feature_transform: bool = True):
        super().__init__()
        channel = 3
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=64)
        self.out_channels = 1024

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coords = x  # the original code hands the full input to the pooling head
        x = x.transpose(2, 1)[:, :3, :]
        trans = self.stn(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return coords, x


class OpenPointsPointNet(nn.Module):
    """OpenPoints PointNet encoder; returns ``(xyz, features [B, 1024, N])``.

    Args:
        in_channels: input channels per point (xyz first).
        input_transform: apply the 3x3 spatial transformer to the coordinates.
        feature_transform: apply the 64x64 feature transformer.
    """

    def __init__(self, in_channels: int = 3, input_transform: bool = True, feature_transform: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.stn = STN3d(in_channels) if input_transform else None
        self.conv0_1 = nn.Conv1d(in_channels, 64, 1)
        self.conv0_2 = nn.Conv1d(64, 64, 1)
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k=64) if feature_transform else None
        self.out_channels = 1024

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = points[:, :, :3].contiguous()
        x = points[:, :, : self.in_channels].transpose(1, 2).contiguous()
        _, D, _ = x.size()
        if self.stn is not None:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = F.relu(self.bn0_2(self.conv0_2(x)))
        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return xyz, x
