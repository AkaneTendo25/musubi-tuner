from __future__ import annotations

from typing import Dict

import torch


def split_combined_prompt_embeds(
    prompt_embeds: torch.Tensor,
    *,
    expected_video_dim: int | None = None,
    expected_audio_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Split concatenated [video|audio] embeddings into modality tensors.

    Uses explicit expected dims when available; otherwise falls back to legacy
    half split for even hidden sizes.
    """
    hidden = int(prompt_embeds.shape[-1])
    video_dim = int(expected_video_dim or 0)
    audio_dim = int(expected_audio_dim or 0)

    if video_dim > 0 and audio_dim > 0:
        if hidden == (video_dim + audio_dim):
            return (
                prompt_embeds[..., :video_dim],
                prompt_embeds[..., video_dim : video_dim + audio_dim],
            )
        return None

    if hidden % 2 == 0:
        half = hidden // 2
        return prompt_embeds[..., :half], prompt_embeds[..., half:]

    return None


def select_audio_text_embeds_for_audio_mode(
    text_embeds: torch.Tensor,
    conditions: Dict[str, torch.Tensor] | None,
    *,
    expected_audio_dim: int | None = None,
    expected_video_dim: int | None = None,
) -> torch.Tensor:
    """Pick audio-side text conditioning for ``--ltx_mode audio``."""
    audio_dim = int(expected_audio_dim or 0)
    preferred_audio = None
    if conditions is not None:
        audio_prompt_embeds = conditions.get("audio_prompt_embeds")
        if isinstance(audio_prompt_embeds, torch.Tensor):
            if audio_dim <= 0 or int(audio_prompt_embeds.shape[-1]) == audio_dim:
                return audio_prompt_embeds
            preferred_audio = audio_prompt_embeds

    source = text_embeds
    if conditions is not None:
        prompt_embeds = conditions.get("prompt_embeds")
        if isinstance(prompt_embeds, torch.Tensor):
            source = prompt_embeds

    if audio_dim > 0 and int(source.shape[-1]) == audio_dim:
        return source

    split = split_combined_prompt_embeds(
        source,
        expected_video_dim=expected_video_dim,
        expected_audio_dim=expected_audio_dim,
    )
    if split is not None:
        _video, audio = split
        return audio

    if isinstance(preferred_audio, torch.Tensor):
        return preferred_audio
    return source


def select_video_text_embeds_for_video_mode(
    text_embeds: torch.Tensor,
    *,
    expected_video_dim: int | None = None,
    expected_audio_dim: int | None = None,
) -> torch.Tensor:
    """Pick video-side text conditioning from a possibly concatenated tensor."""
    video_dim = int(expected_video_dim or 0)
    if video_dim > 0 and int(text_embeds.shape[-1]) == video_dim:
        return text_embeds
    split = split_combined_prompt_embeds(
        text_embeds,
        expected_video_dim=expected_video_dim,
        expected_audio_dim=expected_audio_dim,
    )
    if split is not None:
        video, _audio = split
        return video
    return text_embeds


def select_video_text_embeds_for_av_no_audio(
    text_embeds: torch.Tensor,
    conditions: Dict[str, torch.Tensor] | None,
    *,
    expected_video_dim: int | None = None,
    expected_audio_dim: int | None = None,
) -> torch.Tensor:
    """Pick video-side text conditioning for AV training batches without audio latents.

    Priority:
    1. Use modality-specific ``conditions['video_prompt_embeds']`` when available
       and dimension-compatible.
    2. Try splitting ``conditions['prompt_embeds']`` (or ``text_embeds``) using
       expected dims.
    3. Fallback to legacy first-half split when dims are unknown.
    4. Otherwise return source unchanged.
    """
    video_dim = int(expected_video_dim or 0)
    preferred_video = None
    if conditions is not None:
        video_prompt_embeds = conditions.get("video_prompt_embeds")
        if isinstance(video_prompt_embeds, torch.Tensor):
            if video_dim <= 0 or int(video_prompt_embeds.shape[-1]) == video_dim:
                return video_prompt_embeds
            preferred_video = video_prompt_embeds

    source = text_embeds
    if conditions is not None:
        prompt_embeds = conditions.get("prompt_embeds")
        if isinstance(prompt_embeds, torch.Tensor):
            source = prompt_embeds

    if video_dim > 0 and int(source.shape[-1]) == video_dim:
        return source

    split = split_combined_prompt_embeds(
        source,
        expected_video_dim=expected_video_dim,
        expected_audio_dim=expected_audio_dim,
    )
    if split is not None:
        video, _audio = split
        return video

    if isinstance(preferred_video, torch.Tensor):
        return preferred_video
    return source
