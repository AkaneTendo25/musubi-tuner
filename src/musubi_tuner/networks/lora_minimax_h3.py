from __future__ import annotations

import ast

import torch
from torch import nn

from musubi_tuner.networks import lora

MINIMAX_H3_TARGET_REPLACE_MODULES = ["MiniMaxH3TransformerBlock"]
MINIMAX_H3_TOKEN_REFINER_REPLACE_MODULES = ["MiniMaxH3TokenRefinerBlock"]


def _block_pattern(blocks: tuple[int, ...] | None) -> str:
    return r"\d+" if blocks is None else "(?:" + "|".join(str(index) for index in blocks) + ")"


def _named_target_patterns(
    targets: tuple[str, ...],
    attention_blocks: tuple[int, ...] | None,
    mlp_blocks: tuple[int, ...] | None,
) -> list[str]:
    patterns: list[str] = []
    if "attention" in targets:
        patterns.append(rf"blocks\.{_block_pattern(attention_blocks)}\.attn\..*")
    if "mlp" in targets:
        patterns.append(rf"blocks\.{_block_pattern(mlp_blocks)}\.mlp\..*")
    if "audio" in targets:
        patterns.extend((r"audio_patch_proj", r"final_layer\.audio_out"))
    if "video" in targets:
        patterns.extend((r"video_patch_proj", r"final_layer\.video_out"))
    if "token_refiner" in targets:
        patterns.append(r"token_refiner\..*")
    return patterns


def create_arch_network(
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
    vae: nn.Module,
    text_encoders: list[nn.Module],
    unet: nn.Module,
    neuron_dropout: float | None = None,
    **kwargs,
):
    train_token_refiner = str(kwargs.pop("h3_lora_token_refiner", "false")).lower() in {"1", "true", "yes", "on"}
    named_targets_raw = kwargs.pop("h3_target_modules", None)
    named_blocks_raw = kwargs.pop("h3_target_blocks", None)
    attention_blocks_raw = kwargs.pop("h3_attention_blocks", None)
    mlp_blocks_raw = kwargs.pop("h3_mlp_blocks", None)
    exclude_patterns = kwargs.get("exclude_patterns")
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Keep timestep and modality calibration frozen. Attention and feed-forward
    # projections inside each transformer block remain adapter targets.
    if named_targets_raw:
        if kwargs.get("include_patterns") is not None or exclude_patterns:
            raise ValueError("named H3 LoRA targets cannot be combined with raw include_patterns/exclude_patterns")
        targets = tuple(piece.strip() for piece in str(named_targets_raw).split(",") if piece.strip())
        unknown = sorted(set(targets) - {"attention", "mlp", "audio", "video", "token_refiner"})
        if unknown:
            raise ValueError("unknown H3 LoRA target groups: " + ", ".join(unknown))
        shared_blocks = tuple(int(piece) for piece in str(named_blocks_raw).split(",")) if named_blocks_raw else None
        attention_blocks = (
            tuple(int(piece) for piece in str(attention_blocks_raw).split(",")) if attention_blocks_raw else shared_blocks
        )
        mlp_blocks = tuple(int(piece) for piece in str(mlp_blocks_raw).split(",")) if mlp_blocks_raw else shared_blocks
        for blocks in (attention_blocks, mlp_blocks):
            if blocks is not None and any(index < 0 for index in blocks):
                raise ValueError("H3 LoRA block indices must be non-negative")
        include_patterns = _named_target_patterns(targets, attention_blocks, mlp_blocks)
        if not include_patterns:
            raise ValueError("named H3 LoRA targeting selected no modules")
        kwargs["exclude_patterns"] = [r".*"]
        kwargs["include_patterns"] = include_patterns
        # Named input/output projections live outside transformer blocks, so
        # walk the complete H3 module tree. The allow-list above prevents any
        # unrelated Linear from being wrapped.
        target_modules = None
    else:
        exclude_patterns.extend((r".*(adaln_proj|modulation).*", r".*norm.*"))
        kwargs["exclude_patterns"] = exclude_patterns
        target_modules = list(MINIMAX_H3_TARGET_REPLACE_MODULES)
        if train_token_refiner:
            target_modules.extend(MINIMAX_H3_TOKEN_REFINER_REPLACE_MODULES)

    return lora.create_network(
        target_modules,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: dict[str, torch.Tensor],
    text_encoders: list[nn.Module] | None = None,
    unet: nn.Module | None = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        None,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
