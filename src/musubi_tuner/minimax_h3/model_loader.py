from __future__ import annotations

import logging
from pathlib import Path

import torch
from accelerate import init_empty_weights

from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.training import H3TrainingMode
from musubi_tuner.minimax_h3.weights import CheckpointInspectionError, tensor_metadata
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAMES: dict[H3TrainingMode, str] = {
    "fl2va": "minimax_h3_fl2va_bf16.safetensors",
    "ref2va": "minimax_h3_ref2va_bf16.safetensors",
}

_FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
    "rope.",
)


def resolve_transformer_checkpoint(source: Path, mode: H3TrainingMode) -> Path:
    source = Path(source)
    expected_name = _CHECKPOINT_FILENAMES[mode]
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
        raise FileNotFoundError(f"could not resolve the {mode} BF16 transformer; searched: {searched}")
    return matches[0]


def _expected_checkpoint_dtype(name: str) -> str:
    return "F32" if name.startswith(_FP32_PREFIXES) else "BF16"


def validate_transformer_checkpoint(
    checkpoint_path: Path,
    config: MiniMaxH3TransformerConfig | None = None,
) -> tuple[int, int]:
    """Validate every released key, shape, and mixed-precision dtype without allocating weights."""
    config = config or MiniMaxH3TransformerConfig()
    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer(config)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    actual_tensors = tensor_metadata(checkpoint_path)
    actual = {tensor.name: tensor for tensor in actual_tensors}

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
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
        expected_dtype = _expected_checkpoint_dtype(name)
        if tensor.dtype != expected_dtype:
            disagreements.append(f"{name}: dtype {tensor.dtype}, expected {expected_dtype}")
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
) -> MiniMaxH3Transformer:
    checkpoint_path = resolve_transformer_checkpoint(source, mode)
    config = MiniMaxH3TransformerConfig()
    tensor_count, parameter_count = validate_transformer_checkpoint(checkpoint_path, config)
    logger.info(
        "Validated MiniMax H3 %s checkpoint: %d tensors, %.3fB parameters",
        mode,
        tensor_count,
        parameter_count / 1e9,
    )

    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer(config)

    device = torch.device(loading_device)
    state_dict = load_safetensors_with_lora_and_fp8(
        model_files=str(checkpoint_path),
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=False,
        calc_device=device,
        move_to_device=device.type != "cpu",
        # Preserve the checkpoint's BF16 blocks and required FP32 islands.
        dit_weight_dtype=None,
    )
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"strict H3 load failed: {info}")
    del state_dict
    logger.info("Loaded MiniMax H3 %s transformer from %s on %s", mode, checkpoint_path, device)
    return model
