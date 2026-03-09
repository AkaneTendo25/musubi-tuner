from __future__ import annotations

from typing import Dict

import torch


def select_video_text_embeds_for_av_no_audio(
    text_embeds: torch.Tensor,
    conditions: Dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Pick video-side text conditioning for AV training batches without audio latents.

    Priority:
    1. Use modality-specific ``conditions['video_prompt_embeds']`` when available.
    2. Fallback to legacy first-half split of concatenated AV embeds.
    3. Otherwise return ``text_embeds`` unchanged.
    """
    if conditions is not None:
        video_prompt_embeds = conditions.get("video_prompt_embeds")
        if isinstance(video_prompt_embeds, torch.Tensor):
            return video_prompt_embeds

    if text_embeds.shape[-1] % 2 == 0:
        half = text_embeds.shape[-1] // 2
        return text_embeds[..., :half]

    return text_embeds

