# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax H3 T2VA and Ref2VA packed-sequence geometry.

The layout follows the released Diffusers H3 port at revision
``abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc``. Musubi keeps the
batch dimension on latent rows so the transforms are directly invertible at
the training boundary; H3 training itself remains batch-size one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag

_AUDIO_CHANNELS = 2
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32.0


@dataclass(frozen=True)
class MiniMaxH3PackedSequence:
    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0


@dataclass(frozen=True)
class MiniMaxH3ReferenceGeometry:
    kind: int
    num_latent_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    def num_video_rows(self, patch_size: tuple[int, int, int]) -> int:
        if self.kind == 2:
            return 0
        _, patch_h, patch_w = patch_size
        return self.num_latent_frames * (self.latent_height // patch_h) * (self.latent_width // patch_w)

    @property
    def num_audio_rows(self) -> int:
        return _AUDIO_CHANNELS * self.num_audio_latents


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Convert ``[B, C, T, H, W]`` latents to frame-major H3 video rows."""
    if latents.ndim != 5:
        raise ValueError(f"H3 video latents must have shape [B, C, T, H, W], got {tuple(latents.shape)}")
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, frames, height, width = latents.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"H3 video latents {tuple(latents.shape)} are not divisible by patch {patch_size}")
    latents = latents.reshape(
        batch_size,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(batch_size, -1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    *,
    latent_shape: tuple[int, int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Invert :func:`patchify_video_latents` to ``[B, C, T, H, W]``."""
    if rows.ndim != 3:
        raise ValueError(f"H3 video rows must have shape [B, rows, width], got {tuple(rows.shape)}")
    channels, frames, height, width = latent_shape
    patch_t, patch_h, patch_w = patch_size
    expected_rows = (frames // patch_t) * (height // patch_h) * (width // patch_w)
    expected_width = channels * patch_t * patch_h * patch_w
    if rows.shape[1:] != (expected_rows, expected_width):
        raise ValueError(f"H3 video rows have shape {tuple(rows.shape[1:])}, expected {(expected_rows, expected_width)}")
    rows = rows.reshape(
        rows.shape[0],
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(rows.shape[0], channels, frames, height, width).contiguous()


def pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Convert stereo-major ``[B, 2, 32, T]`` latents to channel-major rows."""
    if latents.ndim != 4 or latents.shape[1] != _AUDIO_CHANNELS:
        raise ValueError(f"H3 audio latents must have shape [B, 2, C, T], got {tuple(latents.shape)}")
    return latents.permute(0, 1, 3, 2).reshape(latents.shape[0], -1, latents.shape[2]).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, *, num_audio_latents: int) -> torch.Tensor:
    """Invert :func:`pack_audio_latents` to stereo-major audio latents."""
    if rows.ndim != 3 or rows.shape[1] != _AUDIO_CHANNELS * num_audio_latents:
        raise ValueError(f"H3 audio rows must have shape [B, {_AUDIO_CHANNELS * num_audio_latents}, C], got {tuple(rows.shape)}")
    rows = rows.reshape(rows.shape[0], _AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 1, 3, 2).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)] for index in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def _temporal_position_span(num_latent_frames: int) -> float:
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)] for index in range(num_latent_frames)
    )


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack(
        [grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )
    return frame_grid, width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        (
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        )
    )


def build_t2va_packed_sequence(
    text_token_tags: torch.Tensor,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    """Build FL2VA's ``[text | keyframes | target audio | target video]`` layout."""
    if text_token_tags.ndim != 1 or text_token_tags.numel() == 0:
        raise ValueError("H3 text token tags must be a non-empty one-dimensional tensor")
    if bool(((text_token_tags < 0) | (text_token_tags > int(MiniMaxH3TokenTag.AUDIO))).any()):
        raise ValueError("H3 text token tags must be valid video, text, or audio modality tags")
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError(f"the released H3 transformer requires temporal patch size 1, got {patch_t}")
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError("H3 latent height and width must be divisible by the spatial patch")
    if num_latent_frames < 0 or num_audio_latents < 0:
        raise ValueError("H3 target latent lengths cannot be negative")
    if num_latent_frames == 0 and num_audio_latents == 0:
        raise ValueError("H3 packed sequences require at least one target modality")
    if keyframe_anchors and num_latent_frames == 0:
        raise ValueError("H3 keyframe conditioning requires a video target")

    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    if any(anchor not in ("first", "last") for anchor in keyframe_anchors):
        raise ValueError("H3 keyframe anchors must be 'first' or 'last'")
    if len(set(keyframe_anchors)) != len(keyframe_anchors):
        raise ValueError("H3 keyframe anchors must be unique")
    num_text_rows = int(text_token_tags.shape[0])
    num_condition_video_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = _AUDIO_CHANNELS * num_audio_latents
    num_video_rows = num_latent_frames * rows_per_frame
    condition_start = num_text_rows
    audio_start = condition_start + num_condition_video_rows
    video_start = audio_start + num_audio_rows
    sequence_length = video_start + num_video_rows

    text_indices = torch.arange(num_text_rows)
    audio_indices = torch.arange(audio_start, video_start)
    video_indices = torch.cat((torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)))
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(device="cpu", dtype=torch.long)
    token_tags[audio_indices] = int(MiniMaxH3TokenTag.AUDIO)
    token_tags[video_indices] = int(MiniMaxH3TokenTag.VIDEO)

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[text_indices, 0] = torch.arange(num_text_rows, dtype=torch.float64)
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_rows)
        else:
            anchor_time = float(num_text_rows) + _temporal_position_span(num_latent_frames) - _ROPE_FRAME_RESCALE
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_rows) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_indices, 0] = audio_time.repeat(_AUDIO_CHANNELS)
    position_ids[audio_indices, 2] = torch.cat(
        (
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        )
    )

    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_rows))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_video_rows,
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: tuple[MiniMaxH3ReferenceGeometry, ...],
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    """Build Ref2VA's ordered ``[text | references | target audio | target video]`` layout."""
    if text_token_tags.ndim != 1 or text_token_tags.numel() == 0:
        raise ValueError("H3 Ref2VA text token tags must be a non-empty vector")
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError(f"the released H3 transformer requires temporal patch size 1, got {patch_t}")
    for reference in references:
        if reference.kind not in (0, 1, 2):
            raise ValueError(f"invalid H3 reference kind: {reference.kind}")
        if reference.kind != 2 and (
            reference.num_latent_frames < 1 or reference.latent_height % patch_h or reference.latent_width % patch_w
        ):
            raise ValueError(f"invalid H3 visual reference geometry: {reference}")
        if reference.num_audio_latents < 0:
            raise ValueError(f"invalid H3 audio reference geometry: {reference}")

    num_text_rows = int(text_token_tags.numel())
    num_reference_video_rows = sum(reference.num_video_rows(patch_size) for reference in references)
    num_reference_audio_rows = sum(reference.num_audio_rows for reference in references)
    target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    target_audio_rows = _AUDIO_CHANNELS * num_audio_latents
    sequence_length = num_text_rows + num_reference_video_rows + num_reference_audio_rows + target_audio_rows + target_video_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_rows, 0] = torch.arange(num_text_rows, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    cursor = num_text_rows
    rotary_time = float(num_text_rows)
    for reference in references:
        if reference.kind == 0:
            rows = slice(cursor, cursor + reference.num_video_rows(patch_size))
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == 2:
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids,
                rows,
                reference.num_audio_latents,
                rotary_time,
                target_width_grid,
            )
            rotary_time += float(reference.num_audio_latents)
        else:
            audio_rows = slice(cursor, cursor + reference.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.num_video_rows(patch_size))
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            _fill_audio_positions(
                position_ids,
                audio_rows,
                reference.num_audio_latents,
                rotary_time,
                width_grid,
            )
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames, 1)
            rotary_time += max(
                float(reference.num_audio_latents),
                _temporal_position_span(reference.num_latent_frames),
            )

    target_audio_start = cursor
    target_video_start = target_audio_start + target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(target_audio_start, target_video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    target_frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[target_video_start:, 0] = target_frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[target_video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices.append(torch.arange(target_video_start, sequence_length))
    audio_indices.append(torch.arange(target_audio_start, target_video_start))
    packed_video_indices = torch.cat(video_indices)
    packed_audio_indices = torch.cat(audio_indices)
    text_indices = torch.arange(num_text_rows)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(device="cpu", dtype=torch.long)
    token_tags[packed_audio_indices] = int(MiniMaxH3TokenTag.AUDIO)
    token_tags[packed_video_indices] = int(MiniMaxH3TokenTag.VIDEO)
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=packed_video_indices,
        audio_indices=packed_audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: torch.Tensor,
    audio_timestep: torch.Tensor,
    condition_video_timestep: torch.Tensor | None = None,
    condition_audio_timestep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign synchronized video/audio times and return unique values plus row indices."""
    if video_timestep.numel() != 1 or audio_timestep.numel() != 1:
        raise ValueError("H3 T2VA packed forward requires one video and audio timestep")
    video_timestep = video_timestep.reshape(1).to(torch.float32)
    audio_timestep = audio_timestep.reshape(1).to(device=video_timestep.device, dtype=torch.float32)
    row_timesteps = video_timestep.expand(layout.sequence_length).clone()
    video_indices = layout.video_indices.to(row_timesteps.device)
    audio_indices = layout.audio_indices.to(row_timesteps.device)
    if layout.num_condition_video_rows:
        if condition_video_timestep is None or condition_video_timestep.numel() != 1:
            raise ValueError("H3 visual conditioning requires one video timestep")
        row_timesteps[video_indices[: layout.num_condition_video_rows]] = condition_video_timestep.reshape(1).to(
            device=row_timesteps.device,
            dtype=torch.float32,
        )[0]
    row_timesteps[audio_indices[layout.num_condition_audio_rows :]] = audio_timestep[0]
    if layout.num_condition_audio_rows:
        if condition_audio_timestep is None or condition_audio_timestep.numel() != 1:
            raise ValueError("H3 reference-audio conditioning requires one timestep")
        row_timesteps[audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep.reshape(1).to(
            device=row_timesteps.device,
            dtype=torch.float32,
        )[0]
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)
