"""RVQ encoder operating on the stitched MiniMax Music 3 latent timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


CHUNK_FRAMES = 200
CHUNK_HOP_FRAMES = 100
STITCH_HOP_LATENTS = 345
OWNED_FROM_FRAME = 25
LATENT_RATIO_NUMERATOR = 441
LATENT_RATIO_DENOMINATOR = 128


def stitched_frame_boundaries(frame_count: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return latent indices for every 25 Hz frame boundary after chunk stitching."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    frame = torch.arange(frame_count + 1, device=device, dtype=torch.int64)
    chunk_count = max(1, (frame_count - 1) // CHUNK_HOP_FRAMES)
    chunk = torch.div(frame - OWNED_FROM_FRAME, CHUNK_HOP_FRAMES, rounding_mode="floor")
    chunk.clamp_(0, chunk_count - 1)
    local_frame = frame - chunk * CHUNK_HOP_FRAMES
    chunk_frames = torch.minimum(
        torch.full_like(chunk, CHUNK_FRAMES), frame_count - chunk * CHUNK_HOP_FRAMES
    )
    chunk_latents = torch.div(
        chunk_frames * LATENT_RATIO_NUMERATOR, LATENT_RATIO_DENOMINATOR, rounding_mode="floor"
    )
    local_latent = torch.div(
        local_frame * chunk_latents + chunk_frames - 1, chunk_frames, rounding_mode="floor"
    )
    return chunk * STITCH_HOP_LATENTS + local_latent


def frame_pool(boundaries: torch.Tensor, padded_latents: int) -> torch.Tensor:
    """Build exact per-frame mean pooling weights for one contiguous latent window."""
    relative = boundaries - boundaries[0]
    frame_count = relative.numel() - 1
    pool = torch.zeros(frame_count, padded_latents, dtype=torch.float32)
    for index in range(frame_count):
        start, end = int(relative[index]), int(relative[index + 1])
        if end <= start:
            raise ValueError("A frame owns no latent samples")
        pool[index, start:end] = 1.0 / (end - start)
    return pool


@dataclass(frozen=True)
class StitchedRVQEncoderConfig:
    width: int = 512
    layers: int = 8
    heads: int = 8
    feedforward_multiplier: int = 4
    window_frames: int = 128
    padded_latents: int = 448
    semantic_vocab: int = 16384
    depth_vocab: int = 1024
    depth_codebooks: int = 7


class ResidualLatentBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.temporal = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.output = nn.Conv1d(width, width, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.temporal(F.gelu(self.norm(hidden)))
        return hidden + self.output(F.gelu(value))


class StitchedRVQEncoder(nn.Module):
    """Map exact stitched Flow-VAE latent windows to eight independent RVQ streams."""

    format = "minimax_music3_stitched_rvq_encoder"

    def __init__(self, config: StitchedRVQEncoderConfig | None = None):
        super().__init__()
        self.config = config or StitchedRVQEncoderConfig()
        cfg = self.config
        self.input = nn.Conv1d(128, cfg.width, 7, padding=3)
        self.latent_blocks = nn.ModuleList(ResidualLatentBlock(cfg.width, dilation) for dilation in (1, 3, 9))
        self.position = nn.Parameter(torch.empty(1, cfg.window_frames, cfg.width))
        layer = nn.TransformerEncoderLayer(
            cfg.width,
            cfg.heads,
            cfg.width * cfg.feedforward_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.layers)
        self.output_norm = nn.LayerNorm(cfg.width)
        self.semantic_head = nn.Linear(cfg.width, cfg.semantic_vocab)
        self.depth_heads = nn.ModuleList(
            nn.Linear(cfg.width, cfg.depth_vocab) for _ in range(cfg.depth_codebooks)
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, latents: torch.Tensor, pool: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if latents.ndim != 3 or latents.shape[1] != 128:
            raise ValueError(f"Expected latents [B,128,L], got {tuple(latents.shape)}")
        if pool.ndim != 3 or pool.shape[0] != latents.shape[0] or pool.shape[2] != latents.shape[2]:
            raise ValueError(f"Pool shape {tuple(pool.shape)} is incompatible with {tuple(latents.shape)}")
        if pool.shape[1] > self.config.window_frames:
            raise ValueError("Pooling window exceeds the configured frame count")
        hidden = self.input(latents)
        for block in self.latent_blocks:
            hidden = block(hidden)
        hidden = torch.bmm(pool.to(hidden), hidden.transpose(1, 2))
        hidden = hidden + self.position[:, : hidden.shape[1]].to(hidden)
        hidden = self.output_norm(self.transformer(hidden))
        return self.semantic_head(hidden), [head(hidden) for head in self.depth_heads]

    @torch.inference_mode()
    def encode(self, latents: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.shape[0] != 1 or latents.shape[1] != 128:
            raise ValueError("Track encoding expects latents [1,128,T]")
        boundaries = stitched_frame_boundaries(frame_count)
        while frame_count and int(boundaries[frame_count]) > latents.shape[-1]:
            frame_count -= 1
        usable_frames = frame_count - frame_count % self.config.window_frames
        if usable_frames < self.config.window_frames:
            raise ValueError("Audio is shorter than one encoder window")
        codes, confidences = [], []
        for start_frame in range(0, usable_frames, self.config.window_frames):
            frame_boundaries = boundaries[start_frame : start_frame + self.config.window_frames + 1]
            start_latent, end_latent = int(frame_boundaries[0]), int(frame_boundaries[-1])
            length = end_latent - start_latent
            window = F.pad(latents[..., start_latent:end_latent], (0, self.config.padded_latents - length))
            pool = frame_pool(frame_boundaries, self.config.padded_latents).unsqueeze(0).to(latents.device)
            semantic, depth = self(window, pool)
            streams = [semantic, *depth]
            codes.append(torch.stack([stream.argmax(-1) for stream in streams], dim=-1)[0])
            confidences.append(
                torch.stack([stream.float().softmax(-1).amax(-1) for stream in streams], dim=-1)[0]
            )
        return torch.cat(codes), torch.cat(confidences)

    def save(self, path: str | Path, metadata: dict[str, str] | None = None) -> None:
        checkpoint_metadata = {"format": self.format, "config": json.dumps(asdict(self.config), sort_keys=True)}
        checkpoint_metadata.update(metadata or {})
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()},
            str(path),
            checkpoint_metadata,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "StitchedRVQEncoder":
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
        if metadata.get("format") != cls.format:
            raise ValueError(f"Not a stitched RVQ encoder checkpoint: {path}")
        model = cls(StitchedRVQEncoderConfig(**json.loads(metadata["config"])))
        model.load_state_dict(load_file(str(path), device=str(device)), strict=True)
        return model.to(device)
