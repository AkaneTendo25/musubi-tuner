from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import init_empty_weights

from musubi_tuner.minimax_h3.adaln_lowrank import (
    DEFAULT_TABLE_POINTS,
    build_adaln_basis,
    build_timestep_table,
    make_adaln_split_hook,
    read_time_embedder,
    reconstruction_error,
)
from musubi_tuner.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3TimeEmbedder, MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.training import H3TrainingMode
from musubi_tuner.minimax_h3.weights import CheckpointInspectionError, tensor_metadata
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight, quantize_int8_convrot_weight
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.modules.custom_offloading_utils import compute_offload_block_indices
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import WeightTransformHooks

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAMES: dict[H3TrainingMode, dict[bool, str]] = {
    "fl2va": {
        False: "minimax_h3_fl2va_bf16.safetensors",
        True: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    },
    "ref2va": {
        False: "minimax_h3_ref2va_bf16.safetensors",
        True: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    },
    "ref2va_omni": {
        False: "minimax_h3_ref2va_bf16.safetensors",
        True: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    },
}

_FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
    "rope.",
    "adaln_t_table",
)

# Quantize only the 50 repeated denoising blocks. ``blocks.`` also occurs in
# ``token_refiner.blocks.*``, so that path must be excluded explicitly. Norms
# are excluded because the shared state-dict quantizer matches tensor names,
# while its forward patch applies only to nn.Linear modules. AdaLN remains in
# scope intentionally: it dominates the block parameter mass, and the public H3
# weight-only quantization implementations also convert the main-block AdaLN linears.
H3_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
H3_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["token_refiner", "norm"]


def _lora_name_for_model_weight(model_weight_key: str) -> str:
    return "lora_unet_" + model_weight_key[: -len(".weight")].replace(".", "_")


def _validate_base_lora_matches(
    model_keys: set[str],
    lora_weights_list: list[dict[str, torch.Tensor]],
) -> list[int]:
    matched_counts = []
    model_weight_keys = [key for key in model_keys if key.endswith(".weight")]
    for index, lora_weights in enumerate(lora_weights_list):
        matched = 0
        for model_weight_key in model_weight_keys:
            lora_name = _lora_name_for_model_weight(model_weight_key)
            down_key = f"{lora_name}.lora_down.weight"
            up_key = f"{lora_name}.lora_up.weight"
            has_down = down_key in lora_weights
            has_up = up_key in lora_weights
            if has_down != has_up:
                missing = up_key if has_down else down_key
                raise ValueError(f"MiniMax H3 base LoRA #{index + 1} is missing paired tensor {missing!r}")
            matched += int(has_down)
        if matched == 0:
            raise ValueError(
                f"MiniMax H3 base LoRA #{index + 1} did not match any transformer weights; expected Musubi lora_unet_* keys"
            )
        matched_counts.append(matched)
    return matched_counts


@torch.no_grad()
def _merge_base_loras_into_int8_state_dict(
    state_dict: dict[str, torch.Tensor],
    lora_weights_list: list[dict[str, torch.Tensor]],
    lora_multipliers: list[float] | None,
    *,
    calc_device: torch.device,
) -> tuple[list[int], int]:
    """Merge generic base LoRAs and requantize each affected ConvRot layer once."""
    matched_counts = _validate_base_lora_matches(set(state_dict), lora_weights_list)
    multipliers = list(lora_multipliers or [])
    multipliers.extend([1.0] * (len(lora_weights_list) - len(multipliers)))
    multipliers = multipliers[: len(lora_weights_list)]
    remaining_keys = [set(weights) for weights in lora_weights_list]
    requantized_layers = 0

    for model_weight_key, model_weight in tuple(state_dict.items()):
        if not model_weight_key.endswith(".weight"):
            continue
        lora_name = _lora_name_for_model_weight(model_weight_key)
        matches = []
        for index, lora_weights in enumerate(lora_weights_list):
            down_key = f"{lora_name}.lora_down.weight"
            up_key = f"{lora_name}.lora_up.weight"
            if down_key in lora_weights and up_key in lora_weights:
                matches.append((index, down_key, up_key, f"{lora_name}.alpha"))
        if not matches:
            continue

        if model_weight.dtype is not torch.int8:
            raise TypeError(f"MiniMax H3 INT8 base LoRA target {model_weight_key!r} is not INT8")
        base = model_weight_key[: -len(".weight")]
        scale_key = f"{base}.scale_weight"
        group_size_key = f"{base}.int8_convrot_groupsize"
        if scale_key not in state_dict or group_size_key not in state_dict:
            raise ValueError(f"INT8 ConvRot base LoRA target {model_weight_key!r} is missing scale or group size")
        original_device = model_weight.device
        group_size = int(state_dict[group_size_key].detach().cpu().item())
        merged_weight = dequantize_int8_convrot_weight(
            model_weight.to(calc_device),
            state_dict[scale_key].to(calc_device),
            group_size,
        ).float()

        for index, down_key, up_key, alpha_key in matches:
            lora_weights = lora_weights_list[index]
            down = lora_weights[down_key].to(device=calc_device, dtype=torch.float32)
            up = lora_weights[up_key].to(device=calc_device, dtype=torch.float32)
            if down.ndim == 4 and down.shape[2:] == (1, 1):
                down = down.squeeze(3).squeeze(2)
            if up.ndim == 4 and up.shape[2:] == (1, 1):
                up = up.squeeze(3).squeeze(2)
            if down.ndim != 2 or up.ndim != 2:
                raise ValueError(
                    f"MiniMax H3 base LoRA supports linear weights, got {down_key}={tuple(down.shape)} "
                    f"and {up_key}={tuple(up.shape)}"
                )
            if up.shape[1] != down.shape[0] or (up.shape[0], down.shape[1]) != tuple(merged_weight.shape):
                raise ValueError(
                    f"MiniMax H3 base LoRA shape mismatch for {model_weight_key}: base={tuple(merged_weight.shape)}, "
                    f"down={tuple(down.shape)}, up={tuple(up.shape)}"
                )
            rank = int(down.shape[0])
            alpha_tensor = lora_weights.get(alpha_key)
            alpha = float(alpha_tensor.detach().float().cpu().item()) if alpha_tensor is not None else float(rank)
            merged_weight.addmm_(up, down, beta=1.0, alpha=multipliers[index] * alpha / rank)
            remaining_keys[index].difference_update((down_key, up_key, alpha_key))

        quantized, scale = quantize_int8_convrot_weight(merged_weight, group_size)
        state_dict[model_weight_key] = quantized.to(original_device)
        state_dict[scale_key] = scale.to(state_dict[scale_key].device)
        requantized_layers += 1

    for index, unused in enumerate(remaining_keys):
        if unused:
            logger.warning(
                "MiniMax H3 base LoRA #%d has %d unused tensor(s); examples: %s",
                index + 1,
                len(unused),
                sorted(unused)[:8],
            )
    return matched_counts, requantized_layers


def build_block_swap_placement(
    *,
    target_device: torch.device,
    num_blocks: int,
    blocks_to_swap: int,
    h2d_only: bool,
):
    """Place resident H3 blocks on GPU and initial swap blocks on CPU while loading."""
    offloaded = set(
        compute_offload_block_indices(
            num_blocks,
            blocks_to_swap,
            h2d_only=h2d_only,
        )
    )
    block_pattern = re.compile(r"^blocks\.(\d+)\.")
    cpu_device = torch.device("cpu")

    def placement(key: str, default_device: torch.device) -> torch.device:
        del default_device
        match = block_pattern.match(key)
        if match is not None and int(match.group(1)) in offloaded:
            return cpu_device
        return target_device

    return placement, tuple(sorted(offloaded))


def resolve_transformer_checkpoint(source: Path, mode: H3TrainingMode, *, int8_convrot: bool = False) -> Path:
    source = Path(source)
    expected_name = _CHECKPOINT_FILENAMES[mode][int8_convrot]
    if source.is_file():
        other_mode = "ref2va" if mode == "fl2va" else "fl2va"
        if other_mode in source.name.lower():
            raise ValueError(f"H3 mode {mode!r} cannot load {source.name!r}")
        return source

    candidates = (
        source / "diffusion_models" / expected_name,
        source / expected_name,
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        searched = ", ".join(str(path) for path in candidates)
        checkpoint_kind = "pruned INT8 ConvRot" if int8_convrot else "BF16"
        raise FileNotFoundError(f"could not resolve the {mode} {checkpoint_kind} transformer; searched: {searched}")
    return matches[0]


def _expected_checkpoint_dtype(name: str, *, pruned: bool = False) -> str:
    if pruned and ".adaln_proj.linear." in name:
        return "F16"
    return "F32" if name.startswith(_FP32_PREFIXES) else "BF16"


def infer_transformer_config(
    checkpoint_path: Path,
    config: MiniMaxH3TransformerConfig | None = None,
) -> MiniMaxH3TransformerConfig:
    config = config or MiniMaxH3TransformerConfig()
    actual = {tensor.name: tensor for tensor in tensor_metadata(checkpoint_path)}
    table = actual.get("adaln_t_table")
    if table is None:
        if config.adaln_t_table_size is not None:
            raise CheckpointInspectionError("MiniMax H3 checkpoint is missing its configured AdaLN timestep table")
        return config
    if len(table.shape) != 2 or table.shape[0] < 2 or table.shape[1] < 1:
        raise CheckpointInspectionError(f"invalid MiniMax H3 AdaLN timestep table shape: {table.shape}")
    inferred = (table.shape[0], table.shape[1])
    configured = (config.adaln_t_table_size, config.time_embed_dim)
    if config.adaln_t_table_size is not None and configured != inferred:
        raise CheckpointInspectionError(f"configured H3 AdaLN timestep table {configured} does not match {inferred}")
    return replace(config, adaln_t_table_size=inferred[0], time_embed_dim=inferred[1])


def _int8_checkpoint_bases(actual: dict[str, object]) -> tuple[set[str], set[str], set[str]]:
    markers = {name[: -len(".comfy_quant")] for name in actual if name.endswith(".comfy_quant")}
    scales = {name[: -len(".weight_scale")] for name in actual if name.endswith(".weight_scale")}
    weights = {name[: -len(".weight")] for name, tensor in actual.items() if name.endswith(".weight") and tensor.dtype == "I8"}
    return markers, scales, weights


def validate_transformer_checkpoint(
    checkpoint_path: Path,
    config: MiniMaxH3TransformerConfig | None = None,
) -> tuple[int, int]:
    """Validate every released key, shape, and mixed-precision dtype without allocating weights."""
    config = infer_transformer_config(checkpoint_path, config)
    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer(config)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    actual_tensors = tensor_metadata(checkpoint_path)
    actual = {tensor.name: tensor for tensor in actual_tensors}

    marker_bases, scale_bases, quantized_bases = _int8_checkpoint_bases(actual)
    if (marker_bases or scale_bases or quantized_bases) and (
        not marker_bases or marker_bases != scale_bases or marker_bases != quantized_bases
    ):
        raise CheckpointInspectionError(
            "MiniMax H3 INT8 checkpoint has inconsistent marker, scale, or weight sets: "
            f"markers={len(marker_bases)}, scales={len(scale_bases)}, weights={len(quantized_bases)}"
        )
    auxiliary = {name for name in actual if name.endswith((".comfy_quant", ".weight_scale"))}
    actual_model_keys = set(actual) - auxiliary

    missing = sorted(set(expected) - actual_model_keys)
    unexpected = sorted(actual_model_keys - set(expected))
    if missing or unexpected:
        raise CheckpointInspectionError(
            f"MiniMax H3 key mismatch: {len(missing)} missing, {len(unexpected)} unexpected; "
            f"missing examples={missing[:5]}, unexpected examples={unexpected[:5]}"
        )

    disagreements = []
    for name, expected_shape in expected.items():
        tensor = actual[name]
        if tensor.shape != expected_shape:
            disagreements.append(f"{name}: shape {tensor.shape}, expected {expected_shape}")
        base = name[: -len(".weight")] if name.endswith(".weight") else ""
        expected_dtype = (
            "I8" if base in quantized_bases else _expected_checkpoint_dtype(name, pruned=config.adaln_t_table_size is not None)
        )
        if tensor.dtype != expected_dtype:
            disagreements.append(f"{name}: dtype {tensor.dtype}, expected {expected_dtype}")
    for base in sorted(marker_bases):
        marker = actual[f"{base}.comfy_quant"]
        scale = actual[f"{base}.weight_scale"]
        weight = actual[f"{base}.weight"]
        if marker.dtype != "U8" or len(marker.shape) != 1 or marker.shape[0] == 0:
            disagreements.append(f"{base}.comfy_quant: expected a non-empty U8 vector")
        if scale.dtype != "F32" or scale.shape != (weight.shape[0], 1):
            disagreements.append(f"{base}.weight_scale: expected F32 shape {(weight.shape[0], 1)}")
    if disagreements:
        raise CheckpointInspectionError(
            f"MiniMax H3 checkpoint has {len(disagreements)} shape/dtype disagreement(s): {disagreements[:8]}"
        )
    return len(actual), sum(tensor.parameters for tensor in actual_tensors)


def load_transformer(
    source: Path,
    *,
    mode: H3TrainingMode,
    loading_device: str | torch.device,
    fp8_scaled: bool = False,
    quantization_device: str | torch.device | None = None,
    int8_convrot: bool = False,
    adaln_rank: int | None = None,
    attention_mode: str = "torch",
    fp8_quantization_mode: str = "block",
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    convrot_int8_fwd: str = "int8",
    target_device: str | torch.device | None = None,
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    low_ram_load: bool = True,
    base_lora_weights: list[dict[str, torch.Tensor]] | None = None,
    base_lora_multipliers: list[float] | None = None,
) -> MiniMaxH3Transformer:
    checkpoint_path = resolve_transformer_checkpoint(source, mode, int8_convrot=int8_convrot)
    config = infer_transformer_config(checkpoint_path)
    actual = {tensor.name: tensor for tensor in tensor_metadata(checkpoint_path)}
    marker_bases, _, _ = _int8_checkpoint_bases(actual)
    detected_int8 = bool(marker_bases)
    if detected_int8 != int8_convrot:
        expected = "an INT8 ConvRot" if int8_convrot else "a non-INT8"
        raise CheckpointInspectionError(f"MiniMax H3 loader expected {expected} checkpoint: {checkpoint_path.name}")
    if int8_convrot and fp8_scaled:
        raise ValueError("MiniMax H3 INT8 ConvRot cannot be combined with scaled FP8")
    tensor_count, parameter_count = validate_transformer_checkpoint(checkpoint_path, config)
    logger.info(
        "Validated MiniMax H3 %s checkpoint: %d tensors, %.3fB parameters",
        mode,
        tensor_count,
        parameter_count / 1e9,
    )

    # AdaLN reads only the timestep, so its 13.0B parameters describe a smooth
    # one-dimensional curve that a rank-r basis reproduces far below BF16's own
    # rounding floor. Reducing it as the weights stream in keeps the released
    # checkpoint as the only artifact anyone needs.
    weight_transform_hooks = None
    if adaln_rank is not None:
        if config.adaln_t_table_size is not None:
            raise ValueError("MiniMax H3 checkpoint is already pruned; --h3_adaln_rank cannot apply again")
        if int8_convrot:
            raise ValueError("MiniMax H3 INT8 ConvRot checkpoints ship pre-pruned; --h3_adaln_rank cannot apply")
        embedder = read_time_embedder(checkpoint_path, embedder_factory=MiniMaxH3TimeEmbedder)
        factor_device = torch.device(quantization_device) if quantization_device is not None else torch.device(loading_device)
        embedder.to(factor_device)
        # Uncentered so the projections' biases pass through untouched.
        basis = build_adaln_basis(embedder, adaln_rank, center=False)
        table = build_timestep_table(embedder, basis, DEFAULT_TABLE_POINTS)
        logger.info(
            "Reducing AdaLN to rank %d (curve relative RMS error %.3e)",
            adaln_rank,
            reconstruction_error(embedder, basis),
        )
        config = replace(config, adaln_t_table_size=DEFAULT_TABLE_POINTS, time_embed_dim=adaln_rank)
        weight_transform_hooks = WeightTransformHooks(split_hook=make_adaln_split_hook(basis, table, compute_device=factor_device))

    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer(config, attention_mode=attention_mode)

    device = torch.device(loading_device)
    calc_device = torch.device(quantization_device) if quantization_device is not None else device
    base_lora_weights = list(base_lora_weights or [])
    if base_lora_weights:
        _validate_base_lora_matches(set(model.state_dict()), base_lora_weights)
    placement_fn = None
    offloaded_blocks: tuple[int, ...] = ()
    resolved_target_device = torch.device(target_device) if target_device is not None else calc_device
    if low_ram_load and blocks_to_swap > 0 and device.type == "cpu" and resolved_target_device.type != "cpu":
        placement_fn, offloaded_blocks = build_block_swap_placement(
            target_device=resolved_target_device,
            num_blocks=len(model.blocks),
            blocks_to_swap=blocks_to_swap,
            h2d_only=block_swap_h2d_only,
        )
        logger.info(
            "MiniMax H3 low-RAM load: streaming %d initial swap blocks to CPU and resident tensors to %s",
            len(offloaded_blocks),
            resolved_target_device,
        )
    if int8_convrot:
        state_dict, quantized_layers = load_comfy_int8_convrot_state_dict(
            checkpoint_path,
            device=device,
            placement_fn=placement_fn,
        )
        if base_lora_weights:
            matched_counts, requantized_layers = _merge_base_loras_into_int8_state_dict(
                state_dict,
                base_lora_weights,
                base_lora_multipliers,
                calc_device=calc_device,
            )
            logger.info(
                "Merged %d MiniMax H3 base LoRA(s) into %d module(s), requantizing %d INT8 ConvRot layer(s)",
                len(base_lora_weights),
                sum(matched_counts),
                requantized_layers,
            )
        registered_layers = prepare_int8_convrot_modules(model, state_dict)
        if registered_layers != quantized_layers:
            raise RuntimeError(f"prepared {registered_layers} INT8 modules for {quantized_layers} checkpoint layers")
    else:
        # ConvRot quantizes the released BF16 weights as they stream, so it sees
        # the same reduced AdaLN the transform hook produces rather than the
        # full-rank projections a pre-quantized checkpoint carries.
        quantizer = (
            ConvRotInt8Quantizer(
                H3_FP8_OPTIMIZATION_TARGET_KEYS,
                H3_FP8_OPTIMIZATION_EXCLUDE_KEYS + (["adaln_proj"] if adaln_rank is not None else []),
            )
            if convrot_int8
            else None
        )
        state_dict = load_safetensors_with_lora_and_fp8(
            model_files=str(checkpoint_path),
            lora_weights_list=base_lora_weights,
            lora_multipliers=base_lora_multipliers,
            fp8_optimization=fp8_scaled,
            quantizer=quantizer,
            calc_device=calc_device,
            move_to_device=device.type != "cpu" and device == calc_device,
            dit_weight_dtype=None,
            target_keys=H3_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=(
                H3_FP8_OPTIMIZATION_EXCLUDE_KEYS + (["adaln_proj"] if adaln_rank is not None else []) if fp8_scaled else None
            ),
            weight_transform_hooks=weight_transform_hooks,
            quantization_mode=fp8_quantization_mode,
            placement_fn=placement_fn,
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
        elif convrot_int8:
            apply_convrot_int8_monkey_patch(model, state_dict, bwd_mode=convrot_int8_bwd, fwd_mode=convrot_int8_fwd)
            # int8 tensors cannot carry requires_grad, and load_state_dict(assign=True)
            # re-applies the meta parameters' requires_grad to whatever arrives. The base
            # is frozen for LoRA training regardless, so clear it before the load.
            model.requires_grad_(False)
    if placement_fn is None and not int8_convrot and device.type != "cpu" and device != calc_device:
        state_dict = {key: value.to(device) for key, value in state_dict.items()}
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"strict H3 load failed: {info}")
    if int8_convrot:
        enable_int8_convrot(model)
    del state_dict
    logger.info(
        "Loaded MiniMax H3 %s transformer from %s on %s%s",
        mode,
        checkpoint_path,
        f"split CPU/{resolved_target_device}" if placement_fn is not None else device,
        " with pruned INT8 ConvRot weights" if int8_convrot else " with scaled FP8 block weights" if fp8_scaled else "",
    )
    return model
