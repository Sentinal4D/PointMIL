"""Transformer point feature extractor (Section 3.1 and Supplementary Section 9).

The block groups every point with its k nearest neighbours (self loop
included), augments the neighbour features with a learned relative positional
embedding, forms the query with an EdgeConv-style MLP followed by max pooling
over the neighbourhood, and attends over the neighbourhood with multi-head
attention. Two blocks are stacked (3 -> 128 -> 256 channels).

The design follows the transformer block of 3DMedPT (Yu et al., 2021,
https://github.com/crane-papercode/3DMedPT), without point sampling and without
multi-graph reasoning. The 3DMedPT repository does not carry a licence file; the
authors of PointMIL acknowledge it as the origin of this block.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from pointmil.backbones.knn import index_points, knn_point


class TransformerBlock(nn.Module):
    """One k-NN attention block.

    Args:
        c_in: number of input feature channels per point.
        c_out: number of output channels (must be divisible by ``heads``).
        k: neighbourhood size of the k-NN graph (self loop included).
        dim_k: key dimension per head.
        heads: number of attention heads.
        ch_raise: hidden width of the EdgeConv-style query MLP.
        coord_dims: number of leading channels of the input that hold the
            coordinates used for the k-NN graph and relative positions (3 for
            xyz; 4 when a height channel is appended, as in the ScanObjectNN
            configuration of the OpenPoints training path).
        features_from_coords: if True the coordinate channels themselves are the
            input features (first block without normals); otherwise the channels
            after the coordinates are the features.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 20,
        dim_k: int = 32,
        heads: int = 8,
        ch_raise: int = 64,
        coord_dims: int = 3,
        features_from_coords: bool = False,
    ):
        super().__init__()
        if c_out % heads != 0:
            raise ValueError("c_out must be divisible by heads")
        dim_v = c_out // heads
        self.d = dim_k  # key / query width per head
        self.K = k  # number of neighbours of each point
        self.heads = heads
        self.coord_dims = coord_dims
        self.features_from_coords = features_from_coords

        c_total = c_in * 2 + dim_v
        self.mlp = nn.Sequential(
            nn.Conv2d(c_total, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True),
            nn.Conv2d(ch_raise, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True),
        )
        self.mlp_v = nn.Conv1d(c_total, dim_v, 1, bias=False)
        self.mlp_k = nn.Conv1d(c_total, dim_k, 1, bias=False)
        self.mlp_q = nn.Conv1d(ch_raise, heads * dim_k, 1, bias=False)
        self.mlp_h = nn.Conv2d(coord_dims, dim_v, 1, bias=False)
        self.bn_value = nn.BatchNorm1d(dim_v)
        self.bn_query = nn.BatchNorm1d(heads * dim_k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one block.

        Args:
            x: ``[B, N, coord_dims + F]`` points with coordinates first.

        Returns:
            ``(coords [B, N, coord_dims], features [B, c_out, N])``.
        """
        n_samples = x.shape[1]
        coords = x[..., : self.coord_dims]
        feature = coords if self.features_from_coords else x[..., self.coord_dims :]
        bs = coords.shape[0]

        knn_idx = knn_point(self.K, coords, coords)  # [B, N, K]
        neighbor_xyz = index_points(coords, knn_idx)  # [B, N, K, D]
        grouped_features = index_points(feature, knn_idx).permute(0, 3, 1, 2).contiguous()  # [B, C, N, K]
        grouped_points_norm = grouped_features - feature.transpose(2, 1).unsqueeze(-1).contiguous()
        relative_pos = (neighbor_xyz - coords.unsqueeze(-2).repeat(1, 1, self.K, 1)).permute(0, 3, 1, 2).contiguous()
        pos_encoder = self.mlp_h(relative_pos)  # [B, dim_v, N, K]

        feature = torch.cat(
            [
                grouped_points_norm,
                feature.transpose(2, 1).unsqueeze(-1).repeat(1, 1, 1, self.K),
                pos_encoder,
            ],
            dim=1,
        )  # [B, 2C + dim_v, N, K]

        feature_q = self.mlp(feature).max(-1)[0]  # [B, ch_raise, N]
        query = F.relu(self.bn_query(self.mlp_q(feature_q)))  # [B, heads * d, N]
        query = rearrange(query, "b (h d) n -> b h d n", b=bs, h=self.heads, d=self.d)

        feature = feature.permute(0, 2, 1, 3).contiguous().view(bs * n_samples, -1, self.K)  # [B*N, 2C + dim_v, K]
        value = self.bn_value(self.mlp_v(feature)).view(bs, n_samples, -1, self.K)  # [B, N, v, K]
        key = self.mlp_k(feature).softmax(dim=-1).view(bs, n_samples, -1, self.K)  # [B, N, d, K]
        k_v_attn = einsum("b n d k, b n v k -> b d v n", key, value)
        out = einsum("b h d n, b d v n -> b h v n", query, k_v_attn.contiguous())
        out = rearrange(out.contiguous(), "b h v n -> b (h v) n")  # [B, c_out, N]
        return coords, out


class TransformerExtractor(nn.Module):
    """Two stacked ``TransformerBlock``s producing per-point features.

    Two input conventions are supported, both used in the paper:

    * ``coord_dims=3`` with ``in_channels=6``: xyz plus normals (IntrA, RBC);
      the first block uses the normals as features. With ``in_channels=3`` the
      first block uses the coordinates as features (ModelNet40).
    * ``coord_dims=in_channels`` (3 or 4): the OpenPoints training path, where
      all input channels (xyz, or xyz plus height for ScanObjectNN) are both the
      coordinates for the k-NN graph and the first-block features.

    Args:
        in_channels: channels per input point.
        coord_dims: how many leading channels are coordinates.
        num_k: neighbourhood size per block.
        channel_dim: output channels per block.
        channel_raise: query MLP width per block.
        dim_k: key dimension per head.
        heads: attention heads.
    """

    def __init__(
        self,
        in_channels: int = 6,
        coord_dims: int = 3,
        num_k: Sequence[int] = (32, 64),
        channel_dim: Sequence[int] = (128, 256),
        channel_raise: Sequence[int] = (64, 256),
        dim_k: int = 32,
        heads: int = 8,
    ):
        super().__init__()
        if in_channels < coord_dims:
            raise ValueError("in_channels must be at least coord_dims")
        first_from_coords = in_channels == coord_dims
        first_c_in = coord_dims if first_from_coords else in_channels - coord_dims
        self.in_channels = in_channels
        self.coord_dims = coord_dims
        self.out_channels = channel_dim[1]
        self.tf1 = TransformerBlock(
            first_c_in, channel_dim[0], k=num_k[0], dim_k=dim_k, heads=heads, ch_raise=channel_raise[0],
            coord_dims=coord_dims, features_from_coords=first_from_coords,
        )
        self.tf2 = TransformerBlock(
            channel_dim[0], channel_dim[1], k=num_k[1], dim_k=dim_k, heads=heads, ch_raise=channel_raise[1],
            coord_dims=coord_dims, features_from_coords=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` is ``[B, N, in_channels]``; returns ``(xyz [B, N, 3], features [B, channel_dim[1], N])``."""
        coords, feature = self.tf1(x)
        coords, feature = self.tf2(torch.cat([coords, feature.transpose(2, 1)], dim=2))
        return coords[..., :3], feature
