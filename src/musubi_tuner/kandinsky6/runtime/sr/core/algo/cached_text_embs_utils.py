"""Utils to work with cached on disk text embeddings for "" prompt."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig


def load_cached_empty_text_embeds(
    conf: DictConfig,
    local_rank: int,
) -> dict[str, torch.Tensor] | None:
    """Load cached empty-caption embeddings from config path if available.

    Args:
        conf: Metrics config section (may contain ``cached_empty_text_emb``).
        local_rank: Local GPU rank (logging gated to rank 0).

    Returns:
        Dict with ``text_embeds``, ``pooled_embed``, ``cu_seqlens`` or ``None``.
    """
    cached_emb_conf = getattr(conf, "cached_empty_text_emb", None)
    if cached_emb_conf is None:
        return None
    emb_path = getattr(cached_emb_conf, "video", None)
    if emb_path is None or not Path(emb_path).exists():
        return None
    cached = torch.load(emb_path, map_location="cpu", weights_only=True)
    if local_rank == 0:
        logger.info("Loaded cached empty-caption embeddings from {}", emb_path)
    return cached


def replicate_cached_embeds(
    cached: dict[str, torch.Tensor],
    bs: int,
    device: str | int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Replicate bs=1 cached text embeddings for an arbitrary batch size.

    Args:
        cached: Dict with ``text_embeds``, ``pooled_embed``, ``cu_seqlens``
            produced by ``save_empty_text_embeddings.py`` for a single sample.
        bs: Target batch size.
        device: Target device.

    Returns:
        Tuple of (text_embeds dict, cu_seqlens tensor) for the full batch.
    """
    single_text = cached["text_embeds"]
    single_pooled = cached["pooled_embed"]
    seq_len = int(cached["cu_seqlens"][-1].item())

    text_embeds = single_text.repeat(bs, 1).to(device)
    pooled_embed = single_pooled.repeat(bs, 1).to(device)
    cu_seqlens = seq_len * torch.arange(bs + 1, dtype=torch.int32, device=device)

    return {"text_embeds": text_embeds, "pooled_embed": pooled_embed}, cu_seqlens
