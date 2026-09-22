"""Cache helpers for MiniMax Music 3 RVQ encoder distillation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DISTILL_CACHE_VERSION = "1"


def distill_item_id(caption: str, lyrics: str) -> str:
    digest = hashlib.sha256((caption + "\0" + lyrics).encode("utf-8")).hexdigest()[:16]
    return digest


def save_distill_cache(
    path: str | Path,
    latents: torch.Tensor,
    codes: torch.Tensor,
    *,
    caption: str,
    lyrics: str,
    ar_seed: int,
    diffusion_seeds: list[int],
) -> None:
    if latents.ndim != 3 or latents.shape[1] != 128:
        raise ValueError(f"Expected latents [variants,128,T], got {tuple(latents.shape)}")
    if codes.ndim != 2 or codes.shape[1] != 8:
        raise ValueError(f"Expected RVQ codes [frames,8], got {tuple(codes.shape)}")
    metadata = {
        "format": "minimax_music3_rvq_distill",
        "version": DISTILL_CACHE_VERSION,
        "caption": caption,
        "lyrics": lyrics,
        "ar_seed": str(ar_seed),
        "diffusion_seeds": json.dumps(diffusion_seeds),
    }
    tensors = {
        "dav_latents": latents.detach().cpu().to(torch.float16).contiguous(),
        "rvq_codes": codes.detach().cpu().to(torch.int32).contiguous(),
    }
    save_file(tensors, str(path), metadata=metadata)


def load_distill_cache(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("format") != "minimax_music3_rvq_distill":
        raise ValueError(f"Not a Music 3 RVQ distillation cache: {path}")
    if metadata.get("version") != DISTILL_CACHE_VERSION:
        raise ValueError(f"Unsupported distillation cache version in {path}")
    return tensors, metadata
