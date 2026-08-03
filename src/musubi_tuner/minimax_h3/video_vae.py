# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from Hugging Face Diffusers PR #14355 at revision
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    r"""
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses `spatial_padding_mode` (`"reflect"` in the released checkpoint); temporal
    padding is causal, i.e. `kernel_size_t - 1` zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode)
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    r"""
    Group normalization applied to each latent frame in isolation (`use_t_isolated_gn` in the original config): the
    temporal axis is folded into the batch axis so statistics never mix across frames.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    r"""
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly `ceil(size / 2)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    r"""
    Causal 3D CNN encoder. `block_out_channels` gives the channel count of every level; the per-level
    `spatial_downsample_factors` / `temporal_downsample_factors` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class MiniMaxH3VideoEncoderModel(nn.Module):
    """Encoder-only MiniMax H3 video VAE used for deterministic training caches."""

    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    clip_length = 17
    token_drop = 3

    def __init__(self, latents_mean: tuple[float, ...], latents_std: tuple[float, ...]) -> None:
        super().__init__()
        if len(latents_mean) != 24 or len(latents_std) != 24:
            raise ValueError("MiniMax H3 video latent statistics must contain 24 channels")
        self.encoder = MiniMaxH3VideoEncoder3d()
        self.quant_conv = nn.Conv3d(48, 48, kernel_size=1)
        self.register_buffer("latents_mean", torch.tensor(latents_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.tensor(latents_std, dtype=torch.float32), persistent=False)
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(left: torch.Tensor, right: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(left.shape[dim], right.shape[dim], extent)
        if extent == 0:
            return right
        positions = torch.arange(extent, device=right.device, dtype=right.dtype)
        shape = [1] * left.ndim
        shape[dim] = extent
        left_weight = (1 - positions / extent).view(shape)
        right_weight = (positions / extent).view(shape)
        left_slice = [slice(None)] * left.ndim
        left_slice[dim] = slice(-extent, None)
        right_slice = [slice(None)] * right.ndim
        right_slice[dim] = slice(0, extent)
        blended = left[tuple(left_slice)] * left_weight + right[tuple(right_slice)] * right_weight
        if extent == right.shape[dim]:
            return blended
        tail_slice = [slice(None)] * right.ndim
        tail_slice[dim] = slice(extent, None)
        return torch.cat((blended, right[tuple(tail_slice)]), dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode_clip(self, pixels: torch.Tensor) -> torch.Tensor:
        height, width = pixels.shape[-2:]
        y_starts, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_starts, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        rows = []
        for y_start, y_length in zip(y_starts, y_lengths):
            row = []
            for x_start, x_length in zip(x_starts, x_lengths):
                tile = pixels[..., y_start : y_start + y_length, x_start : x_start + x_length]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)
        latent_y_overlaps = [value // self.spatial_compression_ratio for value in y_overlaps]
        latent_x_overlaps = [value // self.spatial_compression_ratio for value in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 5 or pixels.shape[1] != 3:
            raise ValueError(f"video pixels must have shape [B, 3, F, H, W], got {tuple(pixels.shape)}")
        num_frames = pixels.shape[2]
        if num_frames % self.clip_length:
            pad = pixels[:, :, -1:].repeat(1, 1, (-num_frames) % self.clip_length, 1, 1)
            pixels = torch.cat((pixels, pad), dim=2)
        moments = torch.cat(
            [
                self._encode_clip(pixels[:, :, start : start + self.clip_length])
                for start in range(0, pixels.shape[2], self.clip_length)
            ],
            dim=2,
        )
        return moments[:, :, : -self.token_drop] if self.token_drop else moments

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        moments = self.encode_moments(pixels)
        latent = moments.chunk(2, dim=1)[0].float()
        mean = self.latents_mean.view(1, -1, 1, 1, 1)
        std = self.latents_std.view(1, -1, 1, 1, 1)
        return (latent - mean) / std
