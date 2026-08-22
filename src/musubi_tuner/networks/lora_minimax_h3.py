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


#: Prefixes community adapters put in front of the module path. Files exported
#: for ComfyUI carry one; files saved straight from training carry none.
_FOREIGN_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")

#: Matrix names in the wild mapped to the names this network reads. PEFT writes
#: ``lora_A``/``lora_B``; some ComfyUI exports already use ``lora_down``/``lora_up``.
_FOREIGN_SUFFIXES = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
    "lora_down.weight": "lora_down.weight",
    "lora_up.weight": "lora_up.weight",
    "alpha": "alpha",
}

#: Module paths this network never adapts, mapped to why an adapter carrying
#: them cannot simply be renamed.
_UNADAPTED = (
    ("adaln_proj", "AdaLN projections"),
    ("modulation", "modulation projections"),
    ("final_layer", "the final layer"),
)


def _strip_prefix(key: str) -> str | None:
    """Module path without the export prefix, or ``None`` if the key is ours."""
    if key.startswith("lora_unet_"):
        return None
    for prefix in _FOREIGN_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _split_key(key: str) -> tuple[str, str] | None:
    """Module path and matrix name, or ``None`` when the key is not an adapter key."""
    path = _strip_prefix(key)
    if path is None:
        return None
    for suffix, mapped in _FOREIGN_SUFFIXES.items():
        if path.endswith("." + suffix):
            return path[: -len(suffix) - 1], mapped
    return None


def is_foreign_lora(weights_sd: dict[str, torch.Tensor]) -> bool:
    """True when the file uses community adapter naming instead of ours.

    Our own files name every module ``lora_unet_<path>``; community files keep
    the dotted module path, with or without an export prefix.
    """
    if any(key.startswith("lora_unet_") for key in weights_sd):
        return False
    return any(_split_key(key) is not None for key in weights_sd)


def convert_foreign_lora(weights_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a community H3 adapter to the keys this network expects.

    Speed adapters are published in several shapes: with or without an export
    prefix, with ``lora_A``/``lora_B`` or ``lora_down``/``lora_up``, and with or
    without ``alpha``. The tensors are the same either way, so this renames the
    keys and supplies ``alpha`` per module when the file omits it.

    ``alpha`` matters because this network scales an adapter by ``alpha / rank``
    while a file without ``alpha`` is meant to apply as ``W + B @ A``. Setting
    ``alpha`` to that module's own rank leaves the scale at one and reproduces
    the published behaviour. It is set per module rather than once, because some
    adapters carry a different rank in every module.

    Adapters reaching modules this network does not adapt are rejected instead
    of being trimmed: dropping part of an adapter changes what it does, and for
    distilled speed adapters the dropped part is what does the distilling.
    """
    hit = {label for marker, label in _UNADAPTED if any(marker in key for key in weights_sd)}
    if hit:
        raise ValueError(
            "H3 adapter reaches modules this network does not adapt: "
            + ", ".join(sorted(hit))
            + ". Renaming it would silently drop those modules and change what the adapter does; "
            "use a build converted for this checkpoint instead"
        )

    converted: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    given_alpha: set[str] = set()
    for key, value in weights_sd.items():
        split = _split_key(key)
        if split is None:
            raise ValueError(f"unrecognized key in an H3 adapter: {key}")
        path, matrix = split
        name = "lora_unet_" + path.replace(".", "_")
        converted[f"{name}.{matrix}"] = value
        if matrix == "lora_down.weight":
            ranks[name] = value.shape[0]
        elif matrix == "alpha":
            given_alpha.add(name)

    missing = sorted(set(ranks) - given_alpha)
    for name in missing:
        converted[f"{name}.alpha"] = torch.tensor(float(ranks[name]))
    return converted


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
