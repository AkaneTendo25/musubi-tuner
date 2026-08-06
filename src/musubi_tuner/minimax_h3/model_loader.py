from __future__ import annotations

import logging
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
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, fp8_linear_forward_patch
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


# torch._scaled_mm computes in FP8 rather than dequantizing to bf16 first, but it
# charges a per-call activation quantization proportional to the input, while the
# matmul saving grows with the output width. Measured on the released shapes, only
# the expansion projections come out ahead: qkv_proj 1.61x and the feed-forward
# gate 1.55x, against 0.97x and 1.01x for the two contracting projections, which
# hand the entire gain back. So the fast path is applied to those two alone.
FP8_FAST_MATMUL_SUFFIXES = ("attn.qkv_proj", "mlp.fc1")


def enable_fp8_fast_matmul(model: torch.nn.Module) -> int:
    """Re-bind the expansion projections to the FP8 matmul path.

    ``apply_fp8_monkey_patch`` captures its ``use_scaled_mm`` choice in a closure
    per module, so the selection is made by rebinding afterwards rather than by
    changing the shared patch.
    """
    patched = 0
    for name, module in model.named_modules():
        if not name.endswith(FP8_FAST_MATMUL_SUFFIXES) or not hasattr(module, "scale_weight"):
            continue
        if module.scale_weight.ndim != 1:
            raise ValueError("H3 fast FP8 matmul needs a per-tensor weight scale; load with fp8_quantization_mode='tensor'")

        def fast_forward(self, x):
            return fp8_linear_forward_patch(self, x, True, None)

        module.forward = fast_forward.__get__(module, type(module))
        patched += 1
    if patched == 0:
        raise RuntimeError("H3 fast FP8 matmul found no quantized expansion projections to patch")
    logger.info("Enabled FP8 fast matmul on %d expansion projections", patched)
    return patched


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
    fp8_fast: bool = False,
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
        # Uncentered so the projections' biases pass through untouched.
        basis = build_adaln_basis(embedder, adaln_rank, center=False)
        table = build_timestep_table(embedder, basis, DEFAULT_TABLE_POINTS)
        logger.info(
            "Reducing AdaLN to rank %d (curve relative RMS error %.3e)",
            adaln_rank,
            reconstruction_error(embedder, basis),
        )
        config = replace(config, adaln_t_table_size=DEFAULT_TABLE_POINTS, time_embed_dim=adaln_rank)
        weight_transform_hooks = WeightTransformHooks(split_hook=make_adaln_split_hook(basis, table))

    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer(config, attention_mode=attention_mode)

    device = torch.device(loading_device)
    calc_device = torch.device(quantization_device) if quantization_device is not None else device
    if int8_convrot:
        state_dict, quantized_layers = load_comfy_int8_convrot_state_dict(checkpoint_path, device=device)
        registered_layers = prepare_int8_convrot_modules(model, state_dict)
        if registered_layers != quantized_layers:
            raise RuntimeError(f"prepared {registered_layers} INT8 modules for {quantized_layers} checkpoint layers")
    else:
        state_dict = load_safetensors_with_lora_and_fp8(
            model_files=str(checkpoint_path),
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=fp8_scaled,
            calc_device=calc_device,
            move_to_device=device.type != "cpu" and device == calc_device,
            dit_weight_dtype=None,
            target_keys=H3_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=(
                H3_FP8_OPTIMIZATION_EXCLUDE_KEYS + (["adaln_proj"] if adaln_rank is not None else []) if fp8_scaled else None
            ),
            weight_transform_hooks=weight_transform_hooks,
            quantization_mode=fp8_quantization_mode,
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
            if fp8_fast:
                enable_fp8_fast_matmul(model)
    if not int8_convrot and device.type != "cpu" and device != calc_device:
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
        device,
        " with pruned INT8 ConvRot weights" if int8_convrot else " with scaled FP8 block weights" if fp8_scaled else "",
    )
    return model
