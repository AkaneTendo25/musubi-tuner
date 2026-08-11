"""Checkpoint capability resolution for the LTX-2.5 model family."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import torch


@dataclass(frozen=True)
class LTX25Capabilities:
    """Architecture switches observed in an LTX transformer checkpoint."""

    ff_bias: bool = True
    audio_ff_bias: bool = True
    use_prompt_adaln_single: bool = True
    use_keyframes_abs_pos_embedding: bool = False


def capabilities_from_config(config: Mapping[str, object]) -> LTX25Capabilities:
    """Resolve capabilities while preserving the legacy LTX-2 defaults."""

    ff_bias = bool(config.get("ff_bias", True))
    return LTX25Capabilities(
        ff_bias=ff_bias,
        # The released 2.5 checkpoint is intentionally asymmetric: video FFNs
        # are bias-free while audio FFNs retain biases. Keep the historical
        # independent audio default when metadata omits audio_ff_bias.
        audio_ff_bias=bool(config.get("audio_ff_bias", True)),
        use_prompt_adaln_single=bool(config.get("use_prompt_adaln_single", True)),
        use_keyframes_abs_pos_embedding=bool(config.get("use_keyframes_abs_pos_embedding", False)),
    )


def capabilities_from_state_dict_keys(
    capabilities: LTX25Capabilities,
    keys: Iterable[str],
    *,
    explicit_config: Mapping[str, object] | None = None,
) -> LTX25Capabilities:
    """Fill capabilities that upstream detects from checkpoint tensor names."""

    key_set = set(keys)
    explicit = explicit_config or {}
    resolved = capabilities

    if "use_keyframes_abs_pos_embedding" not in explicit:
        has_keyframe_embedding = any(key.endswith("keyframes_abs_pos_embedding") for key in key_set)
        resolved = replace(resolved, use_keyframes_abs_pos_embedding=has_keyframe_embedding)

    # Quantized checkpoints may not expose ordinary Linear bias keys, so only
    # infer bias-free blocks when their corresponding dense weight is present.
    if "ff_bias" not in explicit:
        has_video_ff_weight = any(key.endswith(".ff.net.0.proj.weight") for key in key_set)
        has_video_ff_bias = any(key.endswith(".ff.net.0.proj.bias") for key in key_set)
        if has_video_ff_weight:
            resolved = replace(resolved, ff_bias=has_video_ff_bias)

    if "audio_ff_bias" not in explicit:
        has_audio_ff_weight = any(key.endswith(".audio_ff.net.0.proj.weight") for key in key_set)
        has_audio_ff_bias = any(key.endswith(".audio_ff.net.0.proj.bias") for key in key_set)
        if has_audio_ff_weight:
            resolved = replace(resolved, audio_ff_bias=has_audio_ff_bias)

    return resolved


def enrich_config_from_state_dict(config: dict, keys: Iterable[str]) -> dict:
    """Return a copied model config with state-dict-detectable capabilities."""

    enriched = dict(config)
    transformer = dict(enriched.get("transformer", {}))
    capabilities = capabilities_from_state_dict_keys(
        capabilities_from_config(transformer),
        keys,
        explicit_config=transformer,
    )
    transformer.setdefault("ff_bias", capabilities.ff_bias)
    transformer.setdefault("audio_ff_bias", capabilities.audio_ff_bias)
    transformer.setdefault("use_prompt_adaln_single", capabilities.use_prompt_adaln_single)
    transformer.setdefault("use_keyframes_abs_pos_embedding", capabilities.use_keyframes_abs_pos_embedding)
    enriched["transformer"] = transformer
    return enriched


def build_generated_keyframe_mask(batch_size: int, sequence_length: int, keyframe_tokens: int, device) -> torch.Tensor:
    """Mark the appended standalone-keyframe suffix in a token sequence."""

    if keyframe_tokens < 0 or keyframe_tokens > sequence_length:
        raise ValueError(f"Invalid generated-keyframe token count {keyframe_tokens} for sequence length {sequence_length}")
    mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    if keyframe_tokens:
        mask[:, sequence_length - keyframe_tokens :] = True
    return mask


def apply_generated_keyframe_embedding(
    tokens: torch.Tensor,
    keyframe_mask: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Apply the checkpoint-owned embedding to standalone keyframe tokens."""

    if keyframe_mask.shape != tokens.shape[:2]:
        raise ValueError(f"keyframe_mask shape {tuple(keyframe_mask.shape)} does not match tokens {tuple(tokens.shape[:2])}")
    value = embedding.to(device=tokens.device, dtype=tokens.dtype)
    return tokens + keyframe_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1) * value
