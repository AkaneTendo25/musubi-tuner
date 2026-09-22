"""Distilled audio-to-RVQ encoder for MiniMax Music 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass(frozen=True)
class RVQEncoderConfig:
    input_channels: int = 128
    acoustic_channels: int = 0
    width: int = 512
    layers: int = 8
    kernel_size: int = 5
    semantic_vocab: int = 16384
    depth_vocab: int = 1024
    depth_codebooks: int = 7
    code_dim: int = 256
    logit_scale: float = 16.0
    classifier: str = "cosine"


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(16, width)
        self.conv = nn.Conv1d(width, width * 2, kernel_size, padding=padding, dilation=dilation)
        self.output = nn.Conv1d(width, width, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self.conv(F.silu(self.norm(hidden))).chunk(2, dim=1)
        return hidden + self.output(value * torch.sigmoid(gate))


class MiniMaxMusic3RVQEncoder(nn.Module):
    """Map folded stereo DAV latents to one semantic and seven depth codes."""

    def __init__(self, config: RVQEncoderConfig | None = None):
        super().__init__()
        self.config = config or RVQEncoderConfig()
        cfg = self.config
        primary_channels = cfg.input_channels - cfg.acoustic_channels
        if primary_channels < 1:
            raise ValueError("input_channels must exceed acoustic_channels")
        self.input = nn.Conv1d(primary_channels, cfg.width, cfg.kernel_size, padding=cfg.kernel_size // 2)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(cfg.width, cfg.kernel_size, 2 ** (index % 4)) for index in range(cfg.layers)
        )
        if cfg.acoustic_channels:
            self.acoustic_input = nn.Conv1d(
                cfg.acoustic_channels, cfg.width, cfg.kernel_size, padding=cfg.kernel_size // 2
            )
            self.acoustic_blocks = nn.ModuleList(
                ResidualTemporalBlock(cfg.width, cfg.kernel_size, 2 ** (index % 4)) for index in range(cfg.layers)
            )
            self.acoustic_norm = nn.GroupNorm(16, cfg.width)
            self.depth_fusion = nn.Conv1d(cfg.width * 2, cfg.width, 1)
        self.output_norm = nn.GroupNorm(16, cfg.width)
        self.semantic_head = nn.Conv1d(cfg.width, cfg.code_dim, 1)
        self.depth_heads = nn.ModuleList(nn.Conv1d(cfg.width, cfg.code_dim, 1) for _ in range(cfg.depth_codebooks))
        if cfg.classifier == "direct":
            self.semantic_classifier = nn.Conv1d(cfg.code_dim, cfg.semantic_vocab, 1, bias=False)
            self.depth_classifiers = nn.ModuleList(
                nn.Conv1d(cfg.code_dim, cfg.depth_vocab, 1, bias=False) for _ in range(cfg.depth_codebooks)
            )
        elif cfg.classifier != "cosine":
            raise ValueError(f"Unknown RVQ classifier: {cfg.classifier}")
        self.semantic_conditioner = nn.Conv1d(cfg.code_dim, cfg.width, 1)
        self.depth_conditioners = nn.ModuleList(
            nn.Conv1d(cfg.code_dim, cfg.width, 1) for _ in range(cfg.depth_codebooks - 1)
        )
        self.register_buffer("semantic_codebook", torch.zeros(cfg.semantic_vocab, cfg.code_dim))
        self.register_buffer("depth_codebooks", torch.zeros(cfg.depth_codebooks, cfg.depth_vocab, cfg.code_dim))

    def features(
        self, latents: torch.Tensor, frame_count: int, target_codes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if latents.ndim != 3 or latents.shape[1] != self.config.input_channels:
            raise ValueError(f"Expected DAV latents [B,{self.config.input_channels},T], got {tuple(latents.shape)}")
        primary = latents[:, : self.config.input_channels - self.config.acoustic_channels]
        hidden = self.input(primary)
        hidden = F.interpolate(hidden, size=frame_count, mode="linear", align_corners=False)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.silu(self.output_norm(hidden))
        semantic_features = self.semantic_head(hidden)
        depth_hidden = hidden
        if self.config.acoustic_channels:
            acoustic = self.acoustic_input(latents[:, -self.config.acoustic_channels :])
            acoustic = F.interpolate(acoustic, size=frame_count, mode="linear", align_corners=False)
            for block in self.acoustic_blocks:
                acoustic = block(acoustic)
            acoustic = F.silu(self.acoustic_norm(acoustic))
            depth_hidden = self.depth_fusion(torch.cat((hidden, acoustic), dim=1))
        if target_codes is None:
            semantic_logits = torch.einsum(
                "bdt,vd->bvt", F.normalize(semantic_features.float(), dim=1), self.semantic_codebook.float()
            )
            semantic_codes = semantic_logits.argmax(1)
        else:
            semantic_codes = target_codes[..., 0].clamp_min(0)
        semantic_embedding = self.semantic_codebook[semantic_codes].transpose(1, 2).to(hidden.dtype)
        depth_hidden = depth_hidden + self.semantic_conditioner(semantic_embedding)

        depth_features = []
        for index, head in enumerate(self.depth_heads):
            features = head(depth_hidden)
            depth_features.append(features)
            if index < len(self.depth_conditioners):
                if target_codes is None:
                    logits = torch.einsum(
                        "bdt,vd->bvt",
                        F.normalize(features.float(), dim=1),
                        self.depth_codebooks[index].float(),
                    )
                    depth_codes = logits.argmax(1)
                else:
                    depth_codes = target_codes[..., index + 1].clamp_min(0)
                embedding = self.depth_codebooks[index, depth_codes].transpose(1, 2).to(hidden.dtype)
                depth_hidden = depth_hidden + self.depth_conditioners[index](embedding)
        return semantic_features, depth_features

    @torch.no_grad()
    def set_codebooks(self, semantic: torch.Tensor, depth: torch.Tensor) -> None:
        expected_semantic = (self.config.semantic_vocab, self.config.code_dim)
        expected_depth = (self.config.depth_codebooks, self.config.depth_vocab, self.config.code_dim)
        if tuple(semantic.shape) != expected_semantic or tuple(depth.shape) != expected_depth:
            raise ValueError(
                f"Expected codebooks {expected_semantic} and {expected_depth}, got "
                f"{tuple(semantic.shape)} and {tuple(depth.shape)}"
            )
        self.semantic_codebook.copy_(F.normalize(semantic.float(), dim=-1).to(self.semantic_codebook))
        self.depth_codebooks.copy_(F.normalize(depth.float(), dim=-1).to(self.depth_codebooks))
        if self.config.classifier == "direct":
            self.semantic_classifier.weight.copy_(
                (self.semantic_codebook * self.config.logit_scale).unsqueeze(-1).to(self.semantic_classifier.weight)
            )
            for classifier, codebook in zip(self.depth_classifiers, self.depth_codebooks):
                classifier.weight.copy_(
                    (codebook * self.config.logit_scale).unsqueeze(-1).to(classifier.weight)
                )

    def classify(
        self, semantic_features: torch.Tensor, depth_features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.config.classifier == "direct":
            semantic = self.semantic_classifier(F.normalize(semantic_features, dim=1))
            depth = [
                classifier(F.normalize(features, dim=1))
                for classifier, features in zip(self.depth_classifiers, depth_features)
            ]
            return semantic, depth
        scale = self.config.logit_scale
        semantic = torch.einsum(
            "bdt,vd->bvt", F.normalize(semantic_features.float(), dim=1), self.semantic_codebook.float()
        ).mul_(scale)
        depth = [
            torch.einsum("bdt,vd->bvt", F.normalize(features.float(), dim=1), codebook.float()).mul_(scale)
            for features, codebook in zip(depth_features, self.depth_codebooks)
        ]
        return semantic, depth

    def forward(self, latents: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.classify(*self.features(latents, frame_count))

    @torch.inference_mode()
    def encode(self, latents: torch.Tensor, frame_count: int) -> torch.Tensor:
        semantic, depth = self(latents, frame_count)
        return torch.stack([semantic.argmax(dim=1), *(logits.argmax(dim=1) for logits in depth)], dim=-1)

    def save(self, path: str | Path, metadata: dict[str, str] | None = None) -> None:
        checkpoint_metadata = {"config": json.dumps(asdict(self.config), sort_keys=True)}
        checkpoint_metadata.update(metadata or {})
        save_file({key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}, str(path), checkpoint_metadata)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "MiniMaxMusic3RVQEncoder":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
        config = RVQEncoderConfig(**json.loads(metadata.get("config", "{}")))
        model = cls(config)
        model.load_state_dict(load_file(str(path), device=str(device)), strict=True)
        return model.to(device)
