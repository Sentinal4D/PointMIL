"""DGCNN per-point feature extractors (Wang et al., 2019).

Two implementations are kept because the paper used both:

* ``DGCNNExtractor`` follows the reference PyTorch implementation by Yue Wang and
  An Tao (MIT licence) that the original PointMIL code used for IntrA, RBC and
  ModelNet40. The classification head is removed so that the 1024-d per-point
  features are returned.
* ``OpenPointsDGCNN`` reproduces the DGCNN encoder of the OpenPoints library
  (Guocheng Qian, MIT licence) that produced the ATLAS-1 and ScanObjectNN
  results, with the CUDA grouping replaced by a pure PyTorch gather.
"""

from __future__ import annotations

import torch
from torch import nn

from pointmil.backbones.knn import gather_neighbours, knn_graph_channels_first


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor | None = None) -> torch.Tensor:
    """Edge features ``[x_j - x_i, x_i]`` for channels-first ``x`` ``[B, C, N]``."""
    batch_size, _, num_points = x.shape
    if idx is None:
        idx = knn_graph_channels_first(x, k=k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNNExtractor(nn.Module):
    """Reference DGCNN without the classifier; returns ``(xyz, features [B, emb_dims, N])``.

    Only the xyz coordinates are used as input, as in the original PointMIL runs.
    """

    def __init__(self, k: int = 20, emb_dims: int = 1024):
        super().__init__()
        self.k = k
        self.out_channels = emb_dims
        channel = 6  # 2 x 3 coordinates after the edge feature construction
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(channel, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = x[:, :, :3]
        x = xyz.transpose(2, 1)
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1, keepdim=False)[0]
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1, keepdim=False)[0]
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1, keepdim=False)[0]
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1, keepdim=False)[0]
        x = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        return xyz, x


def _knn_cdist(points: torch.Tensor, k: int) -> torch.Tensor:
    """Sorted k nearest neighbour indices for ``points`` ``[B, N, C]`` (OpenPoints convention)."""
    dist = torch.cdist(points, points)
    return dist.topk(k=k, dim=-1, largest=False, sorted=True).indices


class _EdgeConvBlock(nn.Module):
    """OpenPoints ``GraphConv(EdgeConv)``: conv-norm-act on ``[x_i, x_j - x_i]`` then max over neighbours."""

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        # Module names mirror OpenPoints (``gconv.nn``) so that state dicts match.
        self.gconv = nn.Module()
        self.gconv.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, 1], edge_index: [B, N, K]
        x_j = gather_neighbours(x.squeeze(-1), edge_index)  # [B, C, N, K]
        k = edge_index.shape[-1]
        out = self.gconv.nn(torch.cat([x.expand(-1, -1, -1, k), x_j - x], dim=1))
        return torch.max(out, -1, keepdim=True)[0]


class OpenPointsDGCNN(nn.Module):
    """OpenPoints DGCNN encoder returning per-point features ``[B, embed_dim, N]``.

    Args:
        in_channels: input feature channels (3 for xyz, 4 for xyz plus height).
        channels: width of the first graph convolution.
        embed_dim: output embedding size.
        n_blocks: number of graph convolution blocks (head plus ``n_blocks - 2`` dynamic blocks).
        k: neighbourhood size.
    """

    def __init__(self, in_channels: int = 3, channels: int = 64, embed_dim: int = 1024, n_blocks: int = 5, k: int = 20):
        super().__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.k = k
        self.out_channels = embed_dim
        self.head = _EdgeConvBlock(in_channels, channels)
        out_channels = [channels]
        c_in = channels
        backbone = []
        for _ in range(n_blocks - 2):
            backbone.append(_EdgeConvBlock(c_in, channels))
            out_channels.append(channels)
            c_in = channels
            channels *= 2
        self.backbone = nn.Sequential(*backbone)
        self.fusion_block = nn.Sequential(
            nn.Conv1d(int(sum(out_channels)), embed_dim, 1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` is ``[B, N, in_channels]`` with xyz first; returns ``(xyz, features)``."""
        xyz = x[:, :, :3].contiguous()
        features = x[:, :, : self.in_channels].transpose(1, 2).contiguous().unsqueeze(-1)  # [B, C, N, 1]
        feats = [self.head(features, _knn_cdist(xyz, self.k))]
        for block in self.backbone:
            edge_index = _knn_cdist(feats[-1].squeeze(-1).transpose(1, 2), self.k)
            feats.append(block(feats[-1], edge_index))
        feats = torch.cat(feats, dim=1).squeeze(-1)
        return xyz, self.fusion_block(feats)
