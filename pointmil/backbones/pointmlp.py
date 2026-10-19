"""PointMLP encoder (Ma et al., 2022) as used through OpenPoints.

Adapted from https://github.com/ma-xu/pointMLP-pytorch (Apache-2.0, Xu Ma) via
the OpenPoints library (MIT, Guocheng Qian). The PointMLPElite configuration with
``reducers=[1, 1, 1, 1]`` keeps every point (no farthest point sampling), which
is the "PointMLPElite dagger" model of the paper. The CUDA farthest point
sampling of the original is replaced by a pure PyTorch implementation that
starts from the first point like the CUDA kernel; when no reduction is applied
the original point order is kept.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pointmil.backbones.knn import farthest_point_sample, index_points, knn_point


def get_activation(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "gelu":
        return nn.GELU()
    if activation == "rrelu":
        return nn.RReLU(inplace=True)
    if activation == "selu":
        return nn.SELU(inplace=True)
    if activation == "silu":
        return nn.SiLU(inplace=True)
    if activation == "hardswish":
        return nn.Hardswish(inplace=True)
    if activation == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"unknown activation '{activation}'")


class LocalGrouper(nn.Module):
    """Sample ``N // sample_ratio`` centres and group ``kneighbors`` neighbours around each."""

    def __init__(self, channel: int, sample_ratio: int, kneighbors: int, use_xyz: bool = True, normalize: str | None = "center"):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor"]:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        S = N // self.sample_ratio
        xyz = xyz.contiguous()
        if S == N:
            fps_idx = torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)
        else:
            fps_idx = farthest_point_sample(xyz, S, start_idx=torch.zeros(B, dtype=torch.long, device=xyz.device))
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:  # anchor
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, bias: bool = True, activation: str = "relu"):
        super().__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel: int, kernel_size: int = 1, groups: int = 1, res_expansion: float = 1.0, bias: bool = True, activation: str = "relu"):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels: int, out_channels: int, blocks: int = 1, groups: int = 1, res_expansion: float = 1, bias: bool = True, activation: str = "relu", use_xyz: bool = True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.operation = nn.Sequential(
            *[ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation) for _ in range(blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.transfer(x)
        batch_size = x.size(0)
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x.reshape(b, n, -1).permute(0, 2, 1)


class PosExtraction(nn.Module):
    def __init__(self, channels: int, blocks: int = 1, groups: int = 1, res_expansion: float = 1, bias: bool = True, activation: str = "relu"):
        super().__init__()
        self.operation = nn.Sequential(
            *[ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation) for _ in range(blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.operation(x)


class PointMLPEncoder(nn.Module):
    """PointMLP encoder returning per-point features ``(xyz, features [B, d, N'])``.

    With ``reducers`` all equal to 1 the output has one feature vector per input
    point in the input order. With reduction the returned ``xyz`` are the sampled
    centres that the features belong to.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        groups: int = 1,
        res_expansion: float = 1.0,
        activation: str = "relu",
        bias: bool = False,
        use_xyz: bool = False,
        normalize: str = "anchor",
        dim_expansion: Sequence[int] = (2, 2, 2, 2),
        pre_blocks: Sequence[int] = (2, 2, 2, 2),
        pos_blocks: Sequence[int] = (2, 2, 2, 2),
        k_neighbors: Sequence[int] = (24, 24, 24, 24),
        reducers: Sequence[int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(in_channels, embed_dim, bias=bias, activation=activation)
        if not (len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion)):
            raise ValueError("pre_blocks, pos_blocks, k_neighbors, reducers and dim_expansion must have the same length")
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            self.local_grouper_list.append(LocalGrouper(last_channel, reducers[i], k_neighbors[i], use_xyz, normalize))
            self.pre_blocks_list.append(
                PreExtraction(last_channel, out_channel, pre_blocks[i], groups=groups, res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz)
            )
            self.pos_blocks_list.append(
                PosExtraction(out_channel, pos_blocks[i], groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
            last_channel = out_channel
        self.out_channels = last_channel
        self.act = get_activation(activation)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = points[:, :, :3].contiguous()
        x = points[:, :, : self.in_channels].transpose(1, 2).contiguous()
        x = self.embedding(x)
        for i in range(self.stages):
            p, x = self.local_grouper_list[i](p, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
        return p, x


def pointmlp_elite(in_channels: int = 3, reducers: Sequence[int] = (1, 1, 1, 1)) -> PointMLPEncoder:
    """PointMLPElite configuration (``reducers=(1, 1, 1, 1)`` is the paper's no-sampling variant)."""
    return PointMLPEncoder(
        in_channels=in_channels, embed_dim=32, groups=1, res_expansion=0.25, activation="relu", bias=False,
        use_xyz=False, normalize="anchor", dim_expansion=(2, 2, 2, 1), pre_blocks=(1, 1, 2, 1), pos_blocks=(1, 1, 2, 1),
        k_neighbors=(24, 24, 24, 24), reducers=tuple(reducers),
    )
