from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


COMFYUI_LEARNED_CONTEXT_KEY = "qwen3vl_32b"
H3_TEXT_HIDDEN_SIZE = 5120


def validate_learned_context(weight: torch.Tensor) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("H3 learned context must be a tensor")
    if weight.ndim != 2:
        raise ValueError(f"H3 learned context must have shape [tokens, {H3_TEXT_HIDDEN_SIZE}]")
    if weight.shape[0] < 1 or weight.shape[1] != H3_TEXT_HIDDEN_SIZE:
        raise ValueError(f"H3 learned context must have shape [tokens, {H3_TEXT_HIDDEN_SIZE}]")
    if not weight.is_floating_point():
        raise TypeError("H3 learned context must use a floating-point dtype")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("H3 learned context contains non-finite values")
    return weight


def load_learned_context(path: str | Path) -> torch.Tensor:
    state = load_file(str(path), device="cpu")
    if COMFYUI_LEARNED_CONTEXT_KEY not in state:
        raise ValueError(f"H3 learned-context file must contain {COMFYUI_LEARNED_CONTEXT_KEY!r}")
    if len(state) != 1:
        extras = sorted(set(state) - {COMFYUI_LEARNED_CONTEXT_KEY})
        raise ValueError("H3 learned-context file contains unexpected tensors: " + ", ".join(extras))
    return validate_learned_context(state[COMFYUI_LEARNED_CONTEXT_KEY])


def load_learned_context_sequence(
    paths: tuple[Path, ...] | list[Path], multipliers: tuple[float, ...] | list[float] = ()
) -> torch.Tensor | None:
    if len(multipliers) > len(paths):
        raise ValueError("more H3 learned-context multipliers than context files were provided")
    contexts = []
    for index, path in enumerate(paths):
        multiplier = multipliers[index] if index < len(multipliers) else 1.0
        if not torch.isfinite(torch.tensor(multiplier)):
            raise ValueError("H3 learned-context multipliers must be finite")
        if multiplier != 0:
            contexts.append(load_learned_context(path) * multiplier)
    return torch.cat(contexts, dim=0) if contexts else None


def prepend_learned_context(
    text_hidden: torch.Tensor,
    text_token_tags: torch.Tensor,
    learned_context: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if learned_context is None:
        return text_hidden, text_token_tags
    validate_learned_context(learned_context)
    if text_hidden.ndim != 2 or text_hidden.shape[1] != H3_TEXT_HIDDEN_SIZE:
        raise ValueError(f"H3 text hidden states must have shape [tokens, {H3_TEXT_HIDDEN_SIZE}]")
    if text_token_tags.ndim != 1 or text_token_tags.shape[0] != text_hidden.shape[0]:
        raise ValueError("H3 text token tags must contain one row per hidden state")
    context = learned_context.to(device=text_hidden.device, dtype=text_hidden.dtype)
    tags = torch.ones(context.shape[0], device=text_token_tags.device, dtype=text_token_tags.dtype)
    return torch.cat((context, text_hidden), dim=0), torch.cat((tags, text_token_tags), dim=0)


def compose_learned_context(
    text_hidden: torch.Tensor,
    text_token_tags: torch.Tensor,
    learned_context: torch.Tensor | None,
    composition: str = "prepend",
) -> tuple[torch.Tensor, torch.Tensor]:
    if composition == "prepend":
        return prepend_learned_context(text_hidden, text_token_tags, learned_context)
    if composition != "replace":
        raise ValueError("H3 learned-context composition must be prepend or replace")
    if learned_context is None:
        return text_hidden, text_token_tags
    validate_learned_context(learned_context)
    context = learned_context.to(device=text_hidden.device, dtype=text_hidden.dtype)
    tags = torch.ones(context.shape[0], device=text_token_tags.device, dtype=text_token_tags.dtype)
    return context, tags


def apply_learned_context(
    conditioning: dict[str, torch.Tensor],
    learned_context: torch.Tensor | None,
    composition: str = "prepend",
) -> dict[str, torch.Tensor]:
    if learned_context is None:
        return conditioning
    from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY

    hidden, tags = compose_learned_context(
        conditioning[H3_TEXT_HIDDEN_KEY], conditioning[H3_TEXT_TOKEN_TAGS_KEY], learned_context, composition
    )
    return {**conditioning, H3_TEXT_HIDDEN_KEY: hidden, H3_TEXT_TOKEN_TAGS_KEY: tags}
