"""LTX-2 model loading and configuration detection utilities."""

import os
import re
import logging
from collections import Counter

import torch
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
from musubi_tuner.modules.nf4_optimization_utils import (
    apply_nf4_monkey_patch,
    load_safetensors_with_nf4_optimization,
    DEFAULT_NF4_BLOCK_SIZE,
)
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.modules.w8a8_optimization_utils import (
    apply_w8a8_monkey_patch,
    apply_quanto_int8_monkey_patch,
    register_quanto_int8_scale_buffers,
)
from musubi_tuner.modules.int8_convrot_utils import (
    best_int8_convrot_groupsize,
    parse_comfy_quant_tensor,
    parse_int8_convrot_groupsizes,
    quantize_int8_convrot_weight,
    summarize_quality,
    write_quality_report,
)
from musubi_tuner.modules.int4_convrot_utils import (
    INT4_CONVROT_GROUP_SCALE_RATIO_SUFFIX,
    INT4_CONVROT_GROUP_SCALE_SIZE_SUFFIX,
    INT4_CONVROT_STABILIZER_L1_SUFFIX,
    INT4_CONVROT_STABILIZER_L2_SUFFIX,
    apply_int4_convrot_monkey_patch,
    best_int4_convrot_groupsize,
    compare_int4_convrot_group_scales,
    compute_int4_convrot_stabilizer,
    parse_comfy_quant_tensor as parse_int4_comfy_quant_tensor,
    parse_int4_convrot_groupsizes,
    parse_int4_convrot_scale_group_candidates,
    quantize_int4_convrot_weight,
    quantize_int4_convrot_weight_grouped,
    register_int4_convrot_buffers,
    summarize_quality as summarize_int4_quality,
    validate_int4_convrot_scale_group_size,
    write_quality_report as write_int4_quality_report,
)
from musubi_tuner.modules.int4_convrot_awq import (
    INT4_CONVROT_AWQ_SCALE_SUFFIX,
    apply_int4_convrot_awq_scale_to_weight,
    compute_int4_convrot_awq_scale,
    default_int4_convrot_awq_scales_path,
    load_int4_convrot_awq_scales,
    save_int4_convrot_awq_scales,
    summarize_int4_convrot_awq_scales,
)
from musubi_tuner.modules.convrot_policy import ConvRotPolicy, load_convrot_policy, resolve_int4_policy_parameters

logger = logging.getLogger(__name__)

_TRANSFORMER_BLOCK_KEY_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")

# Modules to keep in high precision for FP8 quantization.
# Excludes sensitive projection, conditioning, and normalization layers.
KEEP_FP8_HIGH_PRECISION_TOKENS = (
    # --- General layer-component exclusions ---
    "norm",
    "bias",
    "scale_shift_table",
    "layer_norm",
    # --- Video projection/conditioning layers ---
    "patchify_proj",
    "proj_out",
    "adaln_single",
    "caption_projection",
    # --- Audio projection/conditioning layers ---
    "audio_patchify_proj",
    "audio_proj_out",
    "audio_adaln_single",
    "audio_caption_projection",
    # --- AV cross-attention gate layers ---
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    # --- Gated attention ---
    "to_gate_logits",
)


def parse_fp8_keep_blocks(value: Any) -> List[int]:
    """Parse a comma-separated transformer-block list for FP8 exclusion."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        parsed: List[int] = []
        for item in value:
            parsed.extend(parse_fp8_keep_blocks(item))
        return sorted(set(parsed))
    text = str(value).strip()
    if not text:
        return []

    parsed: List[int] = []
    for raw_part in text.replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = [p.strip() for p in part.split("-", 1)]
            if not start_s or not end_s:
                raise ValueError(f"Invalid --fp8_keep_blocks range: {part!r}")
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid --fp8_keep_blocks range: {part!r}")
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(int(part))
    return sorted(set(parsed))


def build_fp8_keep_block_exclude_keys(value: Any, num_blocks: Optional[int]) -> List[str]:
    block_indices = parse_fp8_keep_blocks(value)
    if not block_indices:
        return []
    if num_blocks is None or num_blocks <= 0:
        raise ValueError("--fp8_keep_blocks requires a model with transformer_blocks")
    invalid = [idx for idx in block_indices if idx < 0 or idx >= num_blocks]
    if invalid:
        raise ValueError(f"--fp8_keep_blocks contains invalid block indices {invalid}; valid range is 0..{num_blocks - 1}")
    return [f"transformer_blocks.{idx}." for idx in block_indices]


def should_load_ltx2_transformer_block_key(key: str, *, keep_start: int, keep_end: int) -> bool:
    """Return whether a checkpoint key belongs to the requested transformer block range.

    Non-transformer-block tensors are shared by every stage and are always kept.
    """

    match = _TRANSFORMER_BLOCK_KEY_RE.search(str(key))
    if match is None:
        return True
    block_index = int(match.group(1))
    return int(keep_start) <= block_index < int(keep_end)


def _resolve_transformer_block_load_range(
    transformer_block_load_range: tuple[int, int | None] | None,
    *,
    num_transformer_blocks: int,
) -> tuple[int, int] | None:
    if transformer_block_load_range is None:
        return None
    keep_start, keep_end = transformer_block_load_range
    keep_start = int(keep_start)
    keep_end = num_transformer_blocks if keep_end is None else int(keep_end)
    if keep_start < 0 or keep_end < keep_start or keep_end > num_transformer_blocks:
        raise ValueError(
            f"invalid transformer_block_load_range {keep_start}:{keep_end} for {num_transformer_blocks} transformer blocks"
        )
    return keep_start, keep_end


def _prune_transformer_blocks_to_range(
    model: torch.nn.Module,
    *,
    keep_start: int,
    keep_end: int,
) -> int:
    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("LTX-2 range-aware loading requires transformer_blocks on the base model")
    replaced = 0
    for idx in range(len(blocks)):
        if keep_start <= idx < keep_end:
            continue
        blocks[idx] = torch.nn.Identity()
        replaced += 1
    if replaced:
        logger.info(
            "LTX-2 range-aware load: kept transformer blocks %d:%d, replaced %d unloaded blocks",
            keep_start,
            keep_end,
            replaced,
        )
    return replaced


def detect_ltx2_dtype(model_path: str) -> torch.dtype:
    """Detect the data type of LTX-2 model weights"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LTX-2 weights must be a .safetensors file. Got: {model_path}")

    # NVFP4 checkpoints store weight_scale tensors as float8_e4m3fn which would
    # be misidentified as fp8 model weights.  Detect NVFP4 early and skip those
    # quantization-internal keys so we return the true model dtype (bf16),
    # mirroring the behaviour of NF4 checkpoints.
    from musubi_tuner.modules.nvfp4_utils import detect_nvfp4_checkpoint

    _is_nvfp4 = detect_nvfp4_checkpoint(model_path)

    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"Unable to detect LTX-2 dtype; no tensors found in {model_path}")

        floating_dtypes: list[torch.dtype] = []
        fp8_dtype: torch.dtype | None = None

        # Avoid loading tensors: inspect header dtype for each key.
        for key in keys:
            # Skip NVFP4 quantization metadata keys — these are not model weights.
            if _is_nvfp4 and (key.endswith(".weight_scale") or key.endswith(".weight_scale_2")):
                continue
            meta = handle.header.get(key)
            if not isinstance(meta, dict) or "dtype" not in meta:
                continue
            dt = handle._get_torch_dtype(meta["dtype"])  # noqa: SLF001
            if not isinstance(dt, torch.dtype):
                continue
            if dt.is_floating_point:
                floating_dtypes.append(dt)
                if dt.itemsize == 1 and key.endswith(".weight"):
                    fp8_dtype = dt
                    break

        dtype = fp8_dtype or (
            Counter(floating_dtypes).most_common(1)[0][0] if floating_dtypes else handle.get_tensor(keys[0]).dtype
        )

    logger.info("Detected LTX-2 dtype: %s", dtype)
    return dtype


def detect_ltx2_config(model_path: str) -> Dict[str, Any]:
    """Infer LTX-2 model configuration from weights."""
    keys: List[str]
    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = list(handle.keys())

        def find_key(suffix: str) -> Optional[str]:
            for key in keys:
                if key.endswith(suffix):
                    return key
            return None

        def get_shape(suffix: str) -> Optional[Tuple[int, ...]]:
            key = find_key(suffix)
            if key is None:
                return None
            return tuple(handle.get_tensor(key).shape)

        config: Dict[str, Any] = {}

        # Count transformer blocks
        block_indices = set()
        for key in keys:
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_indices.add(int(match.group(1)))
        if block_indices:
            config["num_layers"] = max(block_indices) + 1

        # Infer attention dimensions
        attn2_shape = get_shape("transformer_blocks.0.attn2.to_k.weight")
        if attn2_shape is not None and len(attn2_shape) == 2:
            inner_dim, cross_dim = attn2_shape
            config["cross_attention_dim"] = cross_dim
            config["num_attention_heads"] = 32
            if inner_dim % config["num_attention_heads"] == 0:
                config["attention_head_dim"] = inner_dim // config["num_attention_heads"]
            else:
                logger.warning("Unable to evenly infer attention_head_dim from %s", attn2_shape)

        patchify_shape = get_shape("patchify_proj.weight")
        if patchify_shape is not None and len(patchify_shape) == 2:
            config["in_channels"] = patchify_shape[1]

        caption_shape = get_shape("caption_projection.linear_1.weight")
        if caption_shape is not None and len(caption_shape) == 2:
            config["caption_channels"] = caption_shape[1]

        # Audio-video specific fields
        audio_patchify_shape = get_shape("audio_patchify_proj.weight")
        audio_attn2_shape = get_shape("transformer_blocks.0.audio_attn2.to_k.weight")
        audio_caption_shape = get_shape("audio_caption_projection.linear_1.weight")
        if audio_patchify_shape is not None:
            config["audio_in_channels"] = audio_patchify_shape[1]
        if audio_attn2_shape is not None and len(audio_attn2_shape) == 2:
            audio_inner_dim, audio_cross_dim = audio_attn2_shape
            config["audio_cross_attention_dim"] = audio_cross_dim
            config["audio_num_attention_heads"] = 32
            if audio_inner_dim % config["audio_num_attention_heads"] == 0:
                config["audio_attention_head_dim"] = audio_inner_dim // config["audio_num_attention_heads"]
            else:
                logger.warning("Unable to evenly infer audio_attention_head_dim from %s", audio_attn2_shape)
        if audio_caption_shape is not None and len(audio_caption_shape) == 2:
            config["caption_channels"] = audio_caption_shape[1]

    return config


def infer_ltx2_transformer_config_from_weights(model_path: str) -> Dict[str, Any]:
    """Reconstruct the nested transformer config for checkpoints that carry no
    config metadata (e.g. Optimum-Quanto exports).

    Structural dims fall back to the configurator's defaults; checkpoint markers
    that change the architecture are auto-detected from weight keys/shapes:
      - apply_gated_attention      <- presence of ``to_gate_logits``
      - cross_attention_adaln      <- adaln_single.linear out-dim / inner-dim == 9
      - caption_proj_before_connector <- absence of ``caption_projection``
    """
    from musubi_tuner.ltx_2.model.transformer.adaln import (
        ADALN_BASE_PARAMS_COUNT,
        ADALN_CROSS_ATTN_PARAMS_COUNT,
    )

    def _strip_quanto(key: str) -> str:
        return key[: -len("._data")] if key.endswith("._data") else key

    with MemoryEfficientSafeOpen(model_path) as handle:
        raw_keys = list(handle.keys())
        norm_keys = {_strip_quanto(k) for k in raw_keys}

        def shape_of(name: str) -> Optional[Tuple[int, ...]]:
            for key in raw_keys:
                if _strip_quanto(key) == name:
                    return tuple(handle.get_tensor(key).shape)
            return None

        # Architecture constants asserted by the configurators' check_config_value calls.
        tcfg: Dict[str, Any] = {
            "dropout": 0.0,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
        }

        block_indices = {int(m.group(1)) for k in norm_keys if (m := re.search(r"transformer_blocks\.(\d+)\.", k))}
        if block_indices:
            tcfg["num_layers"] = max(block_indices) + 1

        attn1_shape = shape_of("transformer_blocks.0.attn1.to_q.weight")
        inner_dim = attn1_shape[0] if attn1_shape and len(attn1_shape) == 2 else None

        adaln_shape = shape_of("adaln_single.linear.weight")
        if adaln_shape and inner_dim and inner_dim > 0 and adaln_shape[0] % inner_dim == 0:
            coeff = adaln_shape[0] // inner_dim
            tcfg["cross_attention_adaln"] = coeff >= (ADALN_BASE_PARAMS_COUNT + ADALN_CROSS_ATTN_PARAMS_COUNT)

        tcfg["caption_proj_before_connector"] = shape_of("caption_projection.linear_1.weight") is None
        tcfg["apply_gated_attention"] = any(k.endswith("to_gate_logits.weight") or k.endswith("to_gate_logits") for k in norm_keys)

        # The released LTX-2.5 transformer is distributed as a standalone
        # safetensors file without its JSON config. Its checkpoint-owned
        # keyframe embedding is an unambiguous 2.5 marker, so restore the
        # official positional/audio constants instead of silently inheriting
        # the legacy configurator defaults. These values affect both inference
        # and training forwards.
        is_ltx25 = any(k.endswith("keyframes_abs_pos_embedding") for k in norm_keys)
        if is_ltx25:
            tcfg.update(
                {
                    "positional_embedding_theta": 10000.0,
                    "positional_embedding_max_pos": [20, 2048, 2048],
                    "audio_positional_embedding_max_pos": [20],
                    "rope_type": "split",
                    "frequencies_precision": "float64",
                    "timestep_scale_multiplier": 1000,
                    "av_ca_timestep_scale_multiplier": 1000.0,
                    "causal_temporal_positioning": True,
                }
            )

        attn2_shape = shape_of("transformer_blocks.0.attn2.to_k.weight")
        if attn2_shape and len(attn2_shape) == 2:
            tcfg["cross_attention_dim"] = attn2_shape[1]

    logger.info("Inferred LTX-2 transformer config from weights (no config metadata): %s", tcfg)
    return {"transformer": tcfg}


def infer_ltx_version_from_checkpoint_config(config: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Infer the LTX checkpoint generation from architecture metadata."""
    markers: List[str] = []
    transformer_cfg = config.get("transformer", {})
    vocoder_cfg = config.get("vocoder", {})

    if bool(transformer_cfg.get("use_keyframes_abs_pos_embedding", False)):
        markers.append("transformer.ltx25_keyframes_abs_pos_embedding=True")

    if bool(transformer_cfg.get("cross_attention_adaln", False)):
        markers.append("transformer.cross_attention_adaln=True")
    if isinstance(vocoder_cfg.get("bwe"), dict):
        markers.append("vocoder.bwe")

    # Additional soft markers used by newer text/audio connector configs.
    connector_keys = (
        "audio_connector_num_attention_heads",
        "audio_connector_attention_head_dim",
        "audio_connector_num_layers",
    )
    if any(k in transformer_cfg for k in connector_keys):
        markers.append("transformer.audio_connector_*")
    if bool(transformer_cfg.get("caption_proj_before_connector", False)):
        markers.append("transformer.caption_proj_before_connector=True")

    detected_version = "2.5" if any("ltx25" in marker for marker in markers) else ("2.3" if markers else "2.0")
    return detected_version, markers


def _apply_memory_optimization_settings(
    model: torch.nn.Module,
    ffn_chunk_target: Optional[str] = None,
    ffn_chunk_size: int = 0,
    split_attn_target: Optional[str] = None,
    split_attn_mode: Optional[str] = None,
    split_attn_chunk_size: int = 0,
) -> None:
    """Apply FFN chunking and split attention settings to transformer blocks.

    Args:
        model: LTXModel or similar with transformer_blocks
        ffn_chunk_target: Which FFN modules to apply chunking to (none/all/video/audio)
        ffn_chunk_size: Chunk size for FFN (0 = disabled)
        split_attn_target: Which attention modules to apply split attention to
                          (none/all/self/cross/text_cross/av_cross/video/audio)
        split_attn_mode: Split attention mode (batch/query)
        split_attn_chunk_size: Chunk size for query-based split attention (0 = default 1024)
    """

    if not hasattr(model, "transformer_blocks"):
        logger.warning("Model does not have transformer_blocks; skipping memory optimization settings")
        return

    # Apply FFN chunking
    if ffn_chunk_target and ffn_chunk_target != "none" and ffn_chunk_size > 0:
        ffn_count = 0
        for block in model.transformer_blocks:
            # Video FFN
            if ffn_chunk_target in ("all", "video") and hasattr(block, "ff"):
                block.ff.chunk_size = ffn_chunk_size
                ffn_count += 1
            # Audio FFN
            if ffn_chunk_target in ("all", "audio") and hasattr(block, "audio_ff"):
                block.audio_ff.chunk_size = ffn_chunk_size
                ffn_count += 1
        if ffn_count > 0:
            logger.info(
                "Applied FFN chunking (chunk_size=%d) to %d FeedForward modules (target=%s)",
                ffn_chunk_size,
                ffn_count,
                ffn_chunk_target,
            )

    # Apply split attention settings
    if split_attn_target and split_attn_target != "none" and split_attn_mode:
        attn_count = 0
        for block in model.transformer_blocks:
            # Video self-attention (attn1)
            if split_attn_target in ("all", "self", "video") and hasattr(block, "attn1"):
                block.attn1.split_attn_mode = split_attn_mode
                block.attn1.split_attn_chunk_size = split_attn_chunk_size
                attn_count += 1

            # Video text cross-attention (attn2)
            if split_attn_target in ("all", "cross", "text_cross", "video") and hasattr(block, "attn2"):
                block.attn2.split_attn_mode = split_attn_mode
                block.attn2.split_attn_chunk_size = split_attn_chunk_size
                attn_count += 1

            # Audio self-attention
            if split_attn_target in ("all", "self", "audio") and hasattr(block, "audio_attn1"):
                block.audio_attn1.split_attn_mode = split_attn_mode
                block.audio_attn1.split_attn_chunk_size = split_attn_chunk_size
                attn_count += 1

            # Audio text cross-attention
            if split_attn_target in ("all", "cross", "text_cross", "audio") and hasattr(block, "audio_attn2"):
                block.audio_attn2.split_attn_mode = split_attn_mode
                block.audio_attn2.split_attn_chunk_size = split_attn_chunk_size
                attn_count += 1

            # Audio-to-video cross-attention
            if split_attn_target in ("all", "cross", "av_cross") and hasattr(block, "audio_to_video_attn"):
                block.audio_to_video_attn.split_attn_mode = split_attn_mode
                block.audio_to_video_attn.split_attn_chunk_size = split_attn_chunk_size
                attn_count += 1

            # Video-to-audio cross-attention
            if split_attn_target in ("all", "cross", "av_cross") and hasattr(block, "video_to_audio_attn"):
                block.video_to_audio_attn.split_attn_mode = split_attn_mode
                block.video_to_audio_attn.split_attn_chunk_size = split_attn_chunk_size
                attn_count += 1

        if attn_count > 0:
            logger.info(
                "Applied split attention (mode=%s, chunk_size=%d) to %d Attention modules (target=%s)",
                split_attn_mode,
                split_attn_chunk_size,
                attn_count,
                split_attn_target,
            )


def load_quanto_int8_state_dict(
    model_files: List[str],
    *,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> dict[str, torch.Tensor]:
    """Load an Optimum-Quanto qint8 (weight-only) checkpoint into a plain state dict.

    Quanto stores each quantized Linear as ``<name>.weight._data`` (int8 [out, in])
    plus ``<name>.weight._scale`` (per-row [out, 1]); the scalar ``input_scale`` /
    ``output_scale`` entries are activation-quant artifacts (activations are not
    quantized here) and are dropped. Non-quantized tensors (bias, excluded blocks)
    are cast to ``non_quant_dtype``. Output keys are the standard module names:
    ``<name>.weight`` (int8), ``<name>.scale_weight`` (float32 [out, 1]).
    """
    sd: dict[str, torch.Tensor] = {}
    dropped = 0
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if key.endswith(".input_scale") or key.endswith(".output_scale"):
                    dropped += 1
                    continue
                if key.endswith(".weight._data"):
                    base = key[: -len("._data")]  # -> <name>.weight
                    if key_filter is not None and not key_filter(base):
                        continue
                    sd[base] = f.get_tensor(key)  # int8, keep as-is
                elif key.endswith(".weight._scale"):
                    base = key[: -len(".weight._scale")] + ".scale_weight"
                    if key_filter is not None and not key_filter(base):
                        continue
                    scale = f.get_tensor(key).float()
                    if scale.ndim == 1:
                        scale = scale.reshape(-1, 1)
                    sd[base] = scale
                else:
                    if key_filter is not None and not key_filter(key):
                        continue
                    value = f.get_tensor(key)
                    if value.is_floating_point() and non_quant_dtype is not None:
                        value = value.to(non_quant_dtype)
                    sd[key] = value
    logger.info("Quanto int8: loaded %d tensors (dropped %d input/output_scale artifacts)", len(sd), dropped)
    return sd


def load_safetensors_dynamic_int8(
    model_files: List[str],
    *,
    target_keys: List[str],
    exclude_keys: List[str],
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    calc_device: Union[str, torch.device] = "cpu",
    key_filter: Optional[Callable[[str], bool]] = None,
) -> dict[str, torch.Tensor]:
    """Stream a standard checkpoint and quantize the targeted Linear weights to per-row
    int8 on the fly (no fp8 intermediate; one tensor at a time, so the full bf16 model is
    never resident).

    Keys are renamed to model names during the stream so target/exclude matching uses the
    model's naming. Each targeted ``<name>.weight`` becomes int8 plus a per-row
    ``<name>.scale_weight`` (float32 [out, 1]); other tensors are cast to ``non_quant_dtype``.
    Output keys are final (the caller must not rename again).
    """
    from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP

    calc_device = torch.device(calc_device)
    sd: dict[str, torch.Tensor] = {}
    quantized = 0
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
                mkey = renamed if renamed is not None else key
                if key_filter is not None and not key_filter(mkey):
                    continue
                shape = f.header[key]["shape"]
                is_target = (
                    mkey.endswith(".weight")
                    and len(shape) == 2
                    and any(t in mkey for t in target_keys)
                    and not any(e in mkey for e in exclude_keys)
                )
                if is_target:
                    # Avoid staging each large BF16 matrix through numpy.fromfile and
                    # pageable CPU memory when quantization is performed on CUDA.
                    # MemoryEfficientSafeOpen uses its mmap transfer path when a CUDA
                    # device is supplied, keeping the stream one tensor at a time.
                    w = f.get_tensor(key, device=calc_device, dtype=torch.float32)
                    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
                    q = (w / scale).round_().clamp_(-127, 127).to(torch.int8)
                    sd[mkey] = q
                    sd[mkey[: -len(".weight")] + ".scale_weight"] = scale.to(torch.float32)
                    quantized += 1
                    del w
                else:
                    value = f.get_tensor(key)
                    if value.is_floating_point() and non_quant_dtype is not None:
                        value = value.to(non_quant_dtype)
                    if calc_device.type == "cuda":
                        value = value.to(calc_device)
                    sd[mkey] = value
    logger.info("int8 dynamic: quantized %d Linear weights to per-row int8 (%d tensors total)", quantized, len(sd))
    return sd


def load_comfy_int8_convrot_state_dict(
    model_files: List[str],
    *,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> dict[str, torch.Tensor]:
    """Load an INT8 ConvRot checkpoint with Comfy-compatible metadata.

    The compatible on-disk layout stores each quantized layer as:
      - ``<name>.weight`` int8 [out, in]
      - ``<name>.weight_scale`` fp32 [out, 1]
      - ``<name>.comfy_quant`` uint8 JSON metadata

    Output uses Musubi's internal keys:
      - ``<name>.weight``
      - ``<name>.scale_weight``
      - ``<name>.int8_convrot_groupsize`` when metadata declares ConvRot.
    """
    sd: dict[str, torch.Tensor] = {}
    quantized = 0
    convrot = 0
    passthrough = 0
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = list(f.keys())
            scale_bases = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
            comfy_cfg: dict[str, dict[str, Any]] = {}
            for key in keys:
                if not key.endswith(".comfy_quant"):
                    continue
                base = key[: -len(".comfy_quant")]
                try:
                    comfy_cfg[base] = parse_comfy_quant_tensor(f.get_tensor(key))
                except Exception as exc:
                    logger.warning("INT8 ConvRot: failed to parse %s in %s: %s", key, model_file, exc)

            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if key.endswith(".comfy_quant"):
                    continue
                if key.endswith(".weight_scale"):
                    base = key[: -len(".weight_scale")]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    scale = f.get_tensor(key).float()
                    if scale.ndim == 1:
                        scale = scale.reshape(-1, 1)
                    sd[base + ".scale_weight"] = scale
                    cfg = comfy_cfg.get(base) or {}
                    if bool(cfg.get("convrot", False)):
                        group_size = int(cfg.get("convrot_groupsize", 256))
                        sd[base + ".int8_convrot_groupsize"] = torch.tensor(group_size, dtype=torch.int32)
                        convrot += 1
                    continue

                if key_filter is not None and not key_filter(key):
                    continue
                value = f.get_tensor(key)
                if key.endswith(".weight") and value.dtype == torch.int8:
                    base = key[: -len(".weight")]
                    if base not in scale_bases:
                        raise ValueError(f"INT8 ConvRot checkpoint has int8 weight without .weight_scale: {key}")
                    sd[key] = value
                    quantized += 1
                    continue
                if (
                    key.endswith(".weight")
                    and key[: -len(".weight")] in scale_bases
                    and value.is_floating_point()
                    and value.dtype.itemsize == 1
                ):
                    raise ValueError(
                        f"{key} looks like a scaled FP8 weight, not an INT8 ConvRot weight. "
                        "Use --int8_convrot_dynamic to convert FP8/BF16 sources, or pass an INT8 ConvRot checkpoint "
                        "with Comfy-compatible .comfy_quant metadata."
                    )
                if value.is_floating_point() and non_quant_dtype is not None:
                    value = value.to(non_quant_dtype)
                sd[key] = value
                passthrough += 1
    logger.info(
        "INT8 ConvRot: loaded %d quantized layers (%d ConvRot), %d passthrough tensors, %d tensors total",
        quantized,
        convrot,
        passthrough,
        len(sd),
    )
    return sd


def load_safetensors_dynamic_int8_convrot(
    model_files: List[str],
    *,
    target_keys: List[str],
    exclude_keys: List[str],
    groupsizes: str | int | Iterable[int] | None = None,
    mse_clip: bool = True,
    quality_report: Optional[str] = None,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    calc_device: Union[str, torch.device] = "cpu",
    key_filter: Optional[Callable[[str], bool]] = None,
    policy: ConvRotPolicy | None = None,
) -> dict[str, torch.Tensor]:
    """Stream a standard checkpoint and quantize targeted Linear weights to INT8 ConvRot."""
    from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP

    calc_device = torch.device(calc_device)
    group_candidates = parse_int8_convrot_groupsizes(groupsizes)
    collect_quality = bool(quality_report)
    sd: dict[str, torch.Tensor] = {}
    quality_layers = []
    quantized = 0
    skipped_groupsize = 0
    policy_kept = 0

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            all_keys = list(f.keys())
            fp8_scale_keys = {k for k in all_keys if k.endswith(".weight_scale") or k.endswith(".input_scale")}
            if fp8_scale_keys:
                logger.info(
                    "INT8 ConvRot dynamic: detected %d FP8 scale tensors; FP8 weights will be dequantized before ConvRot",
                    len(fp8_scale_keys),
                )
            for key in tqdm(all_keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if key in fp8_scale_keys:
                    continue
                renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
                mkey = renamed if renamed is not None else key
                if key_filter is not None and not key_filter(mkey):
                    continue
                value = f.get_tensor(key)
                if value.is_floating_point() and value.dtype.itemsize == 1 and key.endswith(".weight"):
                    scale_key = key.replace(".weight", ".weight_scale")
                    if scale_key not in fp8_scale_keys:
                        raise ValueError(
                            f"INT8 ConvRot dynamic source has FP8 weight without weight_scale: {key}. "
                            "Use a bf16/fp16 checkpoint or a scaled FP8 checkpoint with matching scale tensors."
                        )
                    value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)
                is_candidate = (
                    mkey.endswith(".weight")
                    and value.ndim == 2
                    and value.shape[0] >= 8
                    and any(t in mkey for t in target_keys)
                    and not any(e in mkey for e in exclude_keys)
                )
                group_size = best_int8_convrot_groupsize(value.shape[1], group_candidates) if is_candidate else None
                decision = policy.resolve(mkey) if policy is not None and is_candidate else None
                if decision is not None and not decision.quantize:
                    is_candidate = False
                    policy_kept += 1
                if is_candidate and group_size is None:
                    skipped_groupsize += 1
                    is_candidate = False

                if is_candidate:
                    q, scale, quality = quantize_int8_convrot_weight(
                        value,
                        group_size=int(group_size),
                        calc_device=calc_device,
                        mse_clip=mse_clip,
                        collect_quality=collect_quality,
                        key=mkey,
                    )
                    base = mkey[: -len(".weight")]
                    sd[mkey] = q
                    sd[base + ".scale_weight"] = scale
                    sd[base + ".int8_convrot_groupsize"] = torch.tensor(int(group_size), dtype=torch.int32, device=q.device)
                    if quality is not None:
                        quality_layers.append(quality)
                    quantized += 1
                else:
                    if value.is_floating_point() and non_quant_dtype is not None:
                        value = value.to(non_quant_dtype)
                    if calc_device.type == "cuda":
                        value = value.to(calc_device)
                    sd[mkey] = value

    logger.info(
        "INT8 ConvRot dynamic: quantized %d Linear weights, kept %d policy-selected weights in floating point "
        "(%d skipped: no valid group size), %d tensors total",
        quantized,
        policy_kept,
        skipped_groupsize,
        len(sd),
    )
    if quality_layers:
        summary = summarize_quality(quality_layers)
        logger.info(
            "INT8 ConvRot quality: min_cosine=%.6f mean_cosine=%.6f weighted_sqnr=%.2f dB max_abs_error=%.6g",
            summary["min_cosine"],
            summary["mean_cosine"],
            summary["weighted_sqnr_db"],
            summary["max_abs_error"],
        )
        if quality_report:
            write_quality_report(
                quality_report,
                source=", ".join(model_files),
                options={
                    "mode": "dynamic",
                    "groupsizes": list(group_candidates),
                    "mse_clip": bool(mse_clip),
                    "target_keys": target_keys,
                    "exclude_keys": exclude_keys,
                    "calc_device": str(calc_device),
                },
                layers=quality_layers,
            )
            logger.info("INT8 ConvRot quality report written to %s", quality_report)
    elif quality_report:
        write_quality_report(
            quality_report,
            source=", ".join(model_files),
            options={"mode": "dynamic", "groupsizes": list(group_candidates), "mse_clip": bool(mse_clip)},
            layers=[],
        )
        logger.warning("INT8 ConvRot quality report requested, but no layers were quantized.")
    return sd


def load_comfy_int4_convrot_state_dict(
    model_files: List[str],
    *,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> dict[str, torch.Tensor]:
    """Load a packed INT4 ConvRot checkpoint with Comfy-compatible metadata.

    Expected quantized layer layout:
      - ``<name>.weight`` uint8 [out, ceil(padded_in / 2)] signed-int4 nibbles
      - ``<name>.weight_scale`` fp32 [out, 1]
      - ``<name>.int4_shape`` int32 [out, in, padded_in] or metadata fields in ``<name>.comfy_quant``
      - ``<name>.comfy_quant`` uint8 JSON with ``bits=4`` and ``convrot_groupsize``
    """

    sd: dict[str, torch.Tensor] = {}
    quantized = 0
    convrot = 0
    passthrough = 0
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = list(f.keys())
            scale_bases = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
            shape_bases = {k[: -len(".int4_shape")] for k in keys if k.endswith(".int4_shape")}
            group_bases = {k[: -len(".int4_convrot_groupsize")] for k in keys if k.endswith(".int4_convrot_groupsize")}
            comfy_cfg: dict[str, dict[str, Any]] = {}
            for key in keys:
                if not key.endswith(".comfy_quant"):
                    continue
                base = key[: -len(".comfy_quant")]
                try:
                    comfy_cfg[base] = parse_int4_comfy_quant_tensor(f.get_tensor(key))
                except Exception as exc:
                    logger.warning("INT4 ConvRot: failed to parse %s in %s: %s", key, model_file, exc)

            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if key.endswith(".comfy_quant"):
                    continue
                if key.endswith(".weight_scale"):
                    base = key[: -len(".weight_scale")]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    scale = f.get_tensor(key).float()
                    if scale.ndim == 1:
                        scale = scale.reshape(-1, 1)
                    sd[base + ".scale_weight"] = scale
                    continue
                if key.endswith(".int4_shape"):
                    base = key[: -len(".int4_shape")]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.int32)
                    continue
                if key.endswith(INT4_CONVROT_AWQ_SCALE_SUFFIX):
                    base = key[: -len(INT4_CONVROT_AWQ_SCALE_SUFFIX)]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.float32)
                    continue
                if key.endswith(INT4_CONVROT_GROUP_SCALE_RATIO_SUFFIX):
                    base = key[: -len(INT4_CONVROT_GROUP_SCALE_RATIO_SUFFIX)]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    ratio = f.get_tensor(key)
                    sd[key] = ratio if ratio.dtype == torch.int16 else ratio.to(torch.float32)
                    continue
                if key.endswith(INT4_CONVROT_GROUP_SCALE_SIZE_SUFFIX):
                    base = key[: -len(INT4_CONVROT_GROUP_SCALE_SIZE_SUFFIX)]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.int32)
                    continue
                if key.endswith(INT4_CONVROT_STABILIZER_L1_SUFFIX) or key.endswith(INT4_CONVROT_STABILIZER_L2_SUFFIX):
                    suffix = (
                        INT4_CONVROT_STABILIZER_L1_SUFFIX
                        if key.endswith(INT4_CONVROT_STABILIZER_L1_SUFFIX)
                        else INT4_CONVROT_STABILIZER_L2_SUFFIX
                    )
                    base = key[: -len(suffix)]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.bfloat16)
                    continue
                if key.endswith(".int4_convrot_groupsize"):
                    base = key[: -len(".int4_convrot_groupsize")]
                    weight_key = base + ".weight"
                    if key_filter is not None and not key_filter(weight_key):
                        continue
                    sd[key] = f.get_tensor(key).to(torch.int32)
                    continue

                if key_filter is not None and not key_filter(key):
                    continue
                value = f.get_tensor(key)
                if key.endswith(".weight") and key[: -len(".weight")] in scale_bases and value.dtype in (torch.uint8, torch.int8):
                    base = key[: -len(".weight")]
                    cfg = comfy_cfg.get(base) or {}
                    cfg_bits = int(cfg.get("bits", 0) or 0)
                    if cfg_bits not in (0, 4):
                        raise ValueError(f"{key} has unsupported comfy_quant bits={cfg_bits}; expected INT4 bits=4")
                    if base not in shape_bases and not cfg:
                        raise ValueError(
                            f"INT4 ConvRot checkpoint has packed uint8 weight without .int4_shape or .comfy_quant metadata: {key}"
                        )
                    group_size = int(cfg.get("convrot_groupsize", 0) or 0)
                    if group_size <= 0 and base in group_bases:
                        group_size = int(f.get_tensor(base + ".int4_convrot_groupsize").reshape(-1)[0].item())
                    if group_size <= 0:
                        raise ValueError(f"INT4 ConvRot checkpoint is missing convrot_groupsize metadata for {key}")
                    in_features = int(cfg.get("in_features", 0) or 0)
                    padded_features = int(cfg.get("padded_in_features", 0) or 0)
                    if padded_features <= 0:
                        padded_features = int(value.shape[1]) * 2
                    if in_features <= 0:
                        in_features = padded_features

                    # Comfy's convrot_w4a4 layout stores the packed nibbles in an
                    # int8 tensor. The trainer runtime uses uint8 for the same raw
                    # byte layout, so convert modulo 256 without unpacking.
                    sd[key] = value.to(torch.uint8) if value.dtype == torch.int8 else value
                    if base + ".int4_shape" not in sd:
                        sd[base + ".int4_shape"] = torch.tensor(
                            [int(value.shape[0]), in_features, padded_features],
                            dtype=torch.int32,
                        )
                    if group_size > 0 and base + ".int4_convrot_groupsize" not in sd:
                        sd[base + ".int4_convrot_groupsize"] = torch.tensor(group_size, dtype=torch.int32)
                    quantized += 1
                    if bool(cfg.get("convrot", True)):
                        convrot += 1
                    else:
                        # No-rotation layer: mark it so the runtime skips the online Hadamard
                        # rotation. Absence of this buffer retains the established rotated default.
                        sd[base + ".int4_rotation"] = torch.tensor(0, dtype=torch.int32)
                    continue
                if (
                    key.endswith(".weight")
                    and key[: -len(".weight")] in scale_bases
                    and value.is_floating_point()
                    and value.dtype.itemsize == 1
                ):
                    raise ValueError(
                        f"{key} looks like a scaled FP8 weight, not an INT4 ConvRot weight. "
                        "Pass a bf16/fp16 checkpoint (--w4a4g4/--w4a4g8/--w4a8 auto-detect the on-the-fly dynamic path), "
                        "or a converter-produced INT4 ConvRot checkpoint with .int4_shape/.comfy_quant metadata."
                    )
                if value.is_floating_point() and non_quant_dtype is not None:
                    value = value.to(non_quant_dtype)
                sd[key] = value
                passthrough += 1
    logger.info(
        "INT4 ConvRot: loaded %d quantized layers (%d ConvRot), %d passthrough tensors, %d tensors total",
        quantized,
        convrot,
        passthrough,
        len(sd),
    )
    return sd


def load_safetensors_dynamic_int4_convrot(
    model_files: List[str],
    *,
    target_keys: List[str],
    exclude_keys: List[str],
    groupsizes: str | int | Iterable[int] | None = None,
    mse_clip: bool = True,
    quality_report: Optional[str] = None,
    awq_calibration: bool = False,
    awq_alpha: float = 0.25,
    awq_scales: Optional[dict[str, torch.Tensor]] = None,
    awq_save_path: Optional[str] = None,
    stabilizer_rank: int = 0,
    non_quant_dtype: Optional[torch.dtype] = torch.bfloat16,
    calc_device: Union[str, torch.device] = "cpu",
    key_filter: Optional[Callable[[str], bool]] = None,
    policy: ConvRotPolicy | None = None,
    scale_refine_steps: int = 0,
    group_scales: int = 0,
    group_ratio_q8: bool = False,
    compare_group_scales: str | Iterable[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Stream a standard checkpoint and quantize targeted Linear weights to packed INT4 ConvRot.

    When ``stabilizer_rank > 0`` each targeted weight also gets a frozen rank-``stabilizer_rank``
    low-rank SVD outlier branch (shared ``compute_int4_convrot_stabilizer`` implementation), so
    the produced state dict matches what ``ltx2_quantize_int4_convrot --stabilizer_rank`` would
    write (packed residual + scales + shape + ``.int4_stabilizer_l1/l2``).
    """
    from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP

    calc_device = torch.device(calc_device)
    group_scales = validate_int4_convrot_scale_group_size(group_scales)
    if group_ratio_q8 and not group_scales:
        raise ValueError("group_ratio_q8 requires group_scales")
    comparison_candidates = parse_int4_convrot_scale_group_candidates(compare_group_scales)
    if comparison_candidates and not quality_report:
        raise ValueError("compare_group_scales requires quality_report")
    group_candidates = parse_int4_convrot_groupsizes(groupsizes)
    collect_quality = bool(quality_report)
    sd: dict[str, torch.Tensor] = {}
    quality_layers = []
    group_scale_comparisons = []
    applied_int4_parameters = []
    generated_awq_scales: dict[str, torch.Tensor] = {}
    applied_awq_scales: dict[str, torch.Tensor] = {}
    awq_applied = 0
    quantized = 0
    policy_kept = 0

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            all_keys = list(f.keys())
            fp8_scale_keys = {k for k in all_keys if k.endswith(".weight_scale") or k.endswith(".input_scale")}
            if fp8_scale_keys:
                logger.info(
                    "INT4 ConvRot dynamic: detected %d FP8 scale tensors; FP8 weights will be dequantized before ConvRot",
                    len(fp8_scale_keys),
                )
            for key in tqdm(all_keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if key in fp8_scale_keys:
                    continue
                renamed = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
                mkey = renamed if renamed is not None else key
                if key_filter is not None and not key_filter(mkey):
                    continue
                value = f.get_tensor(key)
                if value.is_floating_point() and value.dtype.itemsize == 1 and key.endswith(".weight"):
                    scale_key = key.replace(".weight", ".weight_scale")
                    if scale_key not in fp8_scale_keys:
                        raise ValueError(
                            f"INT4 ConvRot dynamic source has FP8 weight without weight_scale: {key}. "
                            "Use a bf16/fp16 checkpoint or a scaled FP8 checkpoint with matching scale tensors."
                        )
                    value = value.to(torch.bfloat16) * f.get_tensor(scale_key).to(value.device)
                is_candidate = (
                    mkey.endswith(".weight")
                    and value.ndim == 2
                    and value.shape[0] >= 8
                    and any(t in mkey for t in target_keys)
                    and not any(e in mkey for e in exclude_keys)
                )
                decision = policy.resolve(mkey) if policy is not None and is_candidate else None
                if decision is not None and not decision.quantize:
                    is_candidate = False
                    policy_kept += 1

                if is_candidate:
                    layer_parameters = resolve_int4_policy_parameters(
                        decision,
                        group_scales=group_scales,
                        group_ratio_q8=group_ratio_q8,
                        scale_refine_steps=scale_refine_steps,
                        name=mkey,
                    )
                    awq_scale = None
                    if awq_calibration:
                        awq_scale = compute_int4_convrot_awq_scale(value, alpha=float(awq_alpha))
                        generated_awq_scales[mkey] = awq_scale
                    elif awq_scales is not None:
                        awq_scale = awq_scales.get(mkey)
                        if awq_scale is None:
                            awq_scale = awq_scales.get(key)
                        if awq_scale is None:
                            raise ValueError(f"INT4 ConvRot AWQ scales missing required key for {mkey}")
                    if awq_scale is not None:
                        value = apply_int4_convrot_awq_scale_to_weight(value, awq_scale)
                        applied_awq_scales[mkey] = awq_scale
                        awq_applied += 1

                    group_size = best_int4_convrot_groupsize(value.shape[1], group_candidates)
                    stabilizer = None
                    if stabilizer_rank > 0:
                        stabilizer = compute_int4_convrot_stabilizer(
                            value,
                            group_size=int(group_size),
                            rank=int(stabilizer_rank),
                            calc_device=calc_device,
                            rotate=True,
                        )
                    quant_kwargs = {
                        "group_size": int(group_size),
                        "calc_device": calc_device,
                        "mse_clip": mse_clip,
                        "collect_quality": collect_quality,
                        "key": mkey,
                        "stabilizer": stabilizer,
                        "scale_refine_steps": layer_parameters.scale_refine_steps,
                    }
                    group_scale_state = None
                    if layer_parameters.group_scales:
                        q, scale, shape, quality, group_scale_state = quantize_int4_convrot_weight_grouped(
                            value,
                            scale_group_size=layer_parameters.group_scales,
                            ratio_q8=layer_parameters.group_ratio_q8,
                            **quant_kwargs,
                        )
                    else:
                        q, scale, shape, quality = quantize_int4_convrot_weight(value, **quant_kwargs)
                    if quality_report:
                        applied_int4_parameters.append(
                            {
                                "key": mkey,
                                "group_scales_requested": layer_parameters.group_scales,
                                "group_scales_resolved": (
                                    int(group_scale_state.group_size.detach().reshape(-1)[0].item())
                                    if group_scale_state is not None
                                    else 0
                                ),
                                "group_ratio_q8": bool(layer_parameters.group_ratio_q8 and group_scale_state is not None),
                                "scale_refine_steps": layer_parameters.scale_refine_steps,
                            }
                        )
                    if comparison_candidates:
                        group_scale_comparisons.append(
                            compare_int4_convrot_group_scales(
                                value,
                                candidates=comparison_candidates,
                                selected_group_scales=layer_parameters.group_scales,
                                selected_quality=quality,
                                selected_group_ratio=group_scale_state.ratio if group_scale_state is not None else None,
                                group_size=group_size,
                                calc_device=calc_device,
                                mse_clip=mse_clip,
                                key=mkey,
                                stabilizer=stabilizer,
                                scale_refine_steps=layer_parameters.scale_refine_steps,
                            )
                        )
                    base = mkey[: -len(".weight")]
                    sd[mkey] = q
                    sd[base + ".scale_weight"] = scale
                    sd[base + ".int4_shape"] = shape
                    sd[base + ".int4_convrot_groupsize"] = torch.tensor(int(group_size), dtype=torch.int32, device=q.device)
                    if group_scale_state is not None:
                        sd[base + INT4_CONVROT_GROUP_SCALE_RATIO_SUFFIX] = group_scale_state.ratio
                        sd[base + INT4_CONVROT_GROUP_SCALE_SIZE_SUFFIX] = group_scale_state.group_size
                    if awq_scale is not None:
                        sd[base + INT4_CONVROT_AWQ_SCALE_SUFFIX] = awq_scale.to(device=q.device, dtype=torch.float32)
                    if stabilizer is not None:
                        sd[base + INT4_CONVROT_STABILIZER_L1_SUFFIX] = stabilizer[0].to(device=q.device)
                        sd[base + INT4_CONVROT_STABILIZER_L2_SUFFIX] = stabilizer[1].to(device=q.device)
                    if quality is not None:
                        quality_layers.append(quality)
                    quantized += 1
                else:
                    if value.is_floating_point() and non_quant_dtype is not None:
                        value = value.to(non_quant_dtype)
                    if calc_device.type == "cuda":
                        value = value.to(calc_device)
                    sd[mkey] = value

    if generated_awq_scales and awq_save_path:
        save_int4_convrot_awq_scales(generated_awq_scales, awq_save_path)
    if awq_applied:
        summary = summarize_int4_convrot_awq_scales(applied_awq_scales)
        logger.info(
            "INT4 ConvRot AWQ: applied scales to %d layers (channels=%s min=%.4g max=%.4g mean=%.4g)",
            awq_applied,
            summary.get("num_channels", 0),
            summary.get("min", 0.0),
            summary.get("max", 0.0),
            summary.get("mean", 0.0),
        )

    logger.info(
        "INT4 ConvRot dynamic: quantized %d Linear weights to packed INT4, kept %d policy-selected weights "
        "in floating point (%d tensors total)",
        quantized,
        policy_kept,
        len(sd),
    )
    if group_scales:
        logger.info("INT4 ConvRot dynamic: per-group weight scales enabled (requested group size %d)", int(group_scales))
        logger.info("INT4 ConvRot dynamic: group-ratio storage is %s", "Q8.8 int16" if group_ratio_q8 else "float32")
    if quality_layers:
        summary = summarize_int4_quality(quality_layers)
        logger.info(
            "INT4 ConvRot quality: min_cosine=%.6f mean_cosine=%.6f weighted_sqnr=%.2f dB max_abs_error=%.6g",
            summary["min_cosine"],
            summary["mean_cosine"],
            summary["weighted_sqnr_db"],
            summary["max_abs_error"],
        )
        if quality_report:
            write_int4_quality_report(
                quality_report,
                source=", ".join(model_files),
                options={
                    "mode": "dynamic",
                    "groupsizes": list(group_candidates),
                    "mse_clip": bool(mse_clip),
                    "target_keys": target_keys,
                    "exclude_keys": exclude_keys,
                    "calc_device": str(calc_device),
                    "storage": "packed_signed_int4",
                    "awq_calibration": bool(awq_calibration),
                    "awq_alpha": float(awq_alpha),
                    "awq_scales": awq_save_path,
                    "scale_refine_steps": int(scale_refine_steps),
                    "group_scales": int(group_scales),
                    "group_ratio_q8": bool(group_ratio_q8),
                    "policy_int4_parameters": bool(policy is not None and policy.has_int4_quantization_parameters()),
                    "compare_group_scales": list(comparison_candidates),
                },
                layers=quality_layers,
                group_scale_comparisons=group_scale_comparisons,
                applied_parameters=applied_int4_parameters,
            )
            logger.info("INT4 ConvRot quality report written to %s", quality_report)
    elif quality_report:
        write_int4_quality_report(
            quality_report,
            source=", ".join(model_files),
            options={"mode": "dynamic", "groupsizes": list(group_candidates), "mse_clip": bool(mse_clip)},
            layers=[],
        )
        logger.warning("INT4 ConvRot quality report requested, but no layers were quantized.")
    return sd


def load_ltx2_model(
    model_path: str,
    device: Union[str, torch.device] = "cpu",
    load_device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    attn_mode: str = "torch",
    audio_video: bool = False,
    audio_only_model: bool = False,
    split_attn_target: Optional[str] = None,
    split_attn_mode: Optional[str] = None,
    split_attn_chunk_size: int = 0,
    ffn_chunk_target: Optional[str] = None,
    ffn_chunk_size: int = 0,
    fp8_scaled: bool = False,
    fp8_w8a8: bool = False,
    w8a8_mode: str = "int8",
    w8a8_backend: str | None = None,
    fp8_upcast: bool = False,
    fp8_upcast_stochastic: bool = False,
    fp8_upcast_seed: int = 0,
    fp8_keep_blocks: Optional[str] = None,
    int8_base: bool = False,
    int8_dynamic: bool = False,
    int8_convrot_base: bool = False,
    int8_convrot_dynamic: bool = False,
    int8_convrot_groupsize: str | int | Iterable[int] | None = None,
    int8_convrot_mse_clip: bool = True,
    int8_convrot_quality_report: Optional[str] = None,
    int4_convrot_base: bool = False,
    int4_convrot_dynamic: bool = False,
    int4_convrot_groupsize: str | int | Iterable[int] | None = None,
    int4_convrot_mse_clip: bool = True,
    int4_convrot_quality_report: Optional[str] = None,
    int4_convrot_scale_refine_steps: int = 0,
    int4_convrot_group_scales: int = 0,
    int4_convrot_group_ratio_q8: bool = False,
    int4_convrot_compare_group_scales: str | Iterable[int] | None = None,
    convrot_policy: Optional[str] = None,
    int4_convrot_awq_calibration: bool = False,
    int4_convrot_awq_alpha: float = 0.25,
    int4_convrot_awq_scales: Optional[str] = None,
    int4_convrot_stabilizer_rank: int = 0,
    nvfp4_training_base: bool = False,
    nvfp4_stabilizer_rank: int = 32,
    nf4_base: bool = False,
    nf4_block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    loftq_init: bool = False,
    loftq_iters: int = 1,
    lora_rank: int = 0,
    quantize_device: Optional[str] = None,
    awq_calibration: bool = False,
    awq_alpha: float = 0.25,
    awq_num_batches: int = 8,
    transformer_block_load_range: tuple[int, int | None] | None = None,
    *,
    low_ram_load: bool = False,
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    offload_block_indices: Optional[set] = None,
    **_: Any,
):
    """Load LTX-2 (video, audio-video, or audio-only) transformer

    Args:
        model_path: Path to safetensors model weights
        device: Target device for model
        load_device: Device to load weights into
        torch_dtype: Data type for model parameters
        attn_mode: Attention implementation (torch, flash, flash3, xformers)
        audio_video: If True, load LTXAV model; if False, load LTXV model
        audio_only_model: If True, load LTX audio-only model (no video modules)
        **_: Additional arguments (ignored)

    Returns:
        Loaded LTX-2 transformer model
    """

    def _cast_non_fp8_params(model: torch.nn.Module, target_dtype: torch.dtype) -> None:
        for module in model.modules():
            is_quantized_linear = isinstance(module, torch.nn.Linear) and (
                hasattr(module, "scale_weight")
                or hasattr(module, "_nvfp4_quantized")
                or hasattr(module, "_nvfp4_training_quantized")
            )
            if is_quantized_linear:
                continue
            for _, param in module.named_parameters(recurse=False):
                if isinstance(param, torch.Tensor) and param.dtype == torch.float32:
                    param.data = param.data.to(dtype=target_dtype)
            for name, buf in module.named_buffers(recurse=False):
                if isinstance(buf, torch.Tensor) and buf.dtype == torch.float32:
                    setattr(module, name, buf.to(dtype=target_dtype))

    target_device = torch.device(device)
    load_device = torch.device(load_device)
    resolved_convrot_policy = load_convrot_policy(convrot_policy)

    # Resolve quantization device: CLI flag > env var > default (cuda)
    _qdev_raw = quantize_device or os.getenv("LTX2_NF4_CALC_DEVICE") or os.getenv("LTX2_FP8_CALC_DEVICE") or "cuda"
    _qdev = _qdev_raw.strip().lower()
    if _qdev in {"1", "true", "yes", "cuda", "gpu"}:
        if target_device.type == "cuda":
            _resolved_quant_device = target_device
        else:
            logger.warning(
                "Quantize device '%s' requested GPU, but target device is %s; falling back to CPU.", _qdev_raw, target_device
            )
            _resolved_quant_device = torch.device("cpu")
    else:
        _resolved_quant_device = torch.device("cpu")

    load_weights_on_cpu = _resolved_quant_device.type != "cuda"
    state_device = torch.device("cpu") if load_weights_on_cpu else load_device

    from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader
    from musubi_tuner.ltx_2.model.transformer.model_configurator import (
        LTXAudioOnlyModelConfigurator,
        LTXModelConfigurator,
        LTXVideoOnlyModelConfigurator,
        LTXV_MODEL_COMFY_RENAMING_MAP,
        amend_forward_with_upcast,
    )
    from musubi_tuner.networks.lora_ltx2 import LTX2Wrapper

    logger.info("Loading LTX-2 transformer via state dict: %s", model_path)
    if load_weights_on_cpu:
        logger.info("LTX-2 load path: load weights on CPU, then move to %s", target_device)
    else:
        logger.info("LTX-2 load path: load weights on %s (quantize_device=%s)", load_device, _qdev_raw)
    loader = SafetensorsModelStateDictLoader()
    _config_path = model_path[0] if isinstance(model_path, list) else model_path
    try:
        config = loader.metadata(_config_path)
    except (KeyError, TypeError):
        # Quantized exports may carry no "config" metadata; rebuild it from weights.
        if not (int8_base or int8_convrot_base or int4_convrot_base):
            raise
        config = infer_ltx2_transformer_config_from_weights(_config_path)
    attn_mode = (attn_mode or "torch").lower()
    attn_type = None
    if attn_mode in {"xformers", "xformers-attn"}:
        attn_type = "xformers"
    elif attn_mode in {"flash3", "flash_attention_3"}:
        attn_type = "flash_attention_3"
    elif attn_mode in {"flash", "flash_attention_2"}:
        attn_type = "flash_attention_2"
    elif attn_mode in {"sageattn", "sage_attention"}:
        attn_type = "sage_attention"
    elif attn_mode in {"cudnn", "pytorch_cudnn"}:
        attn_type = "pytorch_cudnn"
    elif attn_mode in {"torch", "sdpa"}:
        attn_type = "pytorch"
    if attn_type is not None:
        config.setdefault("transformer", {})
        config["transformer"]["attention_type"] = attn_type
    if split_attn_target is not None:
        config.setdefault("transformer", {})
        config["transformer"]["split_attn_target"] = split_attn_target
    if split_attn_mode is not None:
        config.setdefault("transformer", {})
        config["transformer"]["split_attn_mode"] = split_attn_mode
    if split_attn_chunk_size is not None:
        config.setdefault("transformer", {})
        config["transformer"]["split_attn_chunk_size"] = int(split_attn_chunk_size)
    if ffn_chunk_target is not None:
        config.setdefault("transformer", {})
        config["transformer"]["ffn_chunk_target"] = ffn_chunk_target
    if ffn_chunk_size is not None:
        config.setdefault("transformer", {})
        config["transformer"]["ffn_chunk_size"] = int(ffn_chunk_size)
    # Auto-detect gated attention from checkpoint keys
    if not config.get("transformer", {}).get("apply_gated_attention", False):
        from safetensors import safe_open

        _check_path = model_path if isinstance(model_path, str) else model_path[0]
        with safe_open(_check_path, framework="pt") as f:
            if any("to_gate_logits" in k for k in f.keys()):
                config.setdefault("transformer", {})
                config["transformer"]["apply_gated_attention"] = True
                logger.info("Auto-detected gated attention from checkpoint keys")

    if audio_only_model and not audio_video:
        raise ValueError("audio_only_model=True requires audio_video=True")

    if audio_only_model:
        configurator = LTXAudioOnlyModelConfigurator
        model_variant = "audio-only"
    elif audio_video:
        configurator = LTXModelConfigurator
        model_variant = "audio-video"
    else:
        configurator = LTXVideoOnlyModelConfigurator
        model_variant = "video-only"
    logger.info("LTX-2 model variant: %s", model_variant)

    with torch.device("meta"):
        base_model = configurator.from_config(config)
    num_transformer_blocks = len(getattr(base_model, "transformer_blocks", []) or [])
    resolved_block_load_range = _resolve_transformer_block_load_range(
        transformer_block_load_range,
        num_transformer_blocks=num_transformer_blocks,
    )
    state_dict_key_filter: Callable[[str], bool] | None = None
    if resolved_block_load_range is not None:
        keep_start, keep_end = resolved_block_load_range

        def state_dict_key_filter(key: str) -> bool:
            renamed_key = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
            normalized_key = renamed_key if renamed_key is not None else key
            return should_load_ltx2_transformer_block_key(
                normalized_key,
                keep_start=keep_start,
                keep_end=keep_end,
            )

        logger.info("LTX-2 range-aware load enabled for transformer blocks %d:%d", keep_start, keep_end)

    # Streamed placement (opt-in). Each weight is written straight to the device it will
    # be used from, so the whole model is never staged in main RAM. Only the training
    # entry point enables this; every other caller keeps the previous single-device path.
    # The caller supplies the exact offloaded block indices, because classic/aggressive
    # swap offloads a contiguous tail while H2D-only streams an evenly spaced set.
    _placement_fn = None
    if not low_ram_load:
        _swap_idx = set()
    elif offload_block_indices is not None:
        _swap_idx = set(offload_block_indices)
    else:
        from musubi_tuner.modules.custom_offloading_utils import compute_offload_block_indices

        _swap_idx = compute_offload_block_indices(
            num_blocks=num_transformer_blocks,
            blocks_to_swap=int(blocks_to_swap or 0),
            h2d_only=bool(block_swap_h2d_only),
        )
    if _swap_idx:
        _blk_re = re.compile(r"transformer_blocks\.(\d+)\.")
        _cpu_dev = torch.device("cpu")

        def _placement_fn(key: str, default_device: torch.device) -> torch.device:
            # Companion tensors (scale/shape/stabilizer) carry the same block path as the
            # weight they belong to, so they follow it onto the same device.
            match = _blk_re.search(key)
            if match is not None and int(match.group(1)) in _swap_idx:
                return _cpu_dev
            return default_device

        # These components are not part of the transformer, so load_state_dict discards
        # them regardless; skipping them avoids materializing them on the load device.
        # Comfy-style checkpoints may prefix keys, so classify the normalized name.
        _skip_prefixes = ("vae.", "audio_vae.", "vocoder.", "text_embedding_projection.")
        _comfy_prefix = "model.diffusion_model."
        _prev_key_filter = state_dict_key_filter

        def state_dict_key_filter(key: str) -> bool:  # noqa: F811
            normalized = key[len(_comfy_prefix) :] if key.startswith(_comfy_prefix) else key
            if normalized.startswith(_skip_prefixes):
                return False
            return True if _prev_key_filter is None else _prev_key_filter(key)

        logger.info("LTX-2 low-RAM load: streaming placement for %d offloaded blocks", len(_swap_idx))

    fp8_block_exclude_keys = build_fp8_keep_block_exclude_keys(fp8_keep_blocks, num_transformer_blocks) if fp8_scaled else []
    if fp8_block_exclude_keys:
        logger.info(
            "LTX-2 fp8: keeping transformer blocks in high precision: %s",
            ", ".join(str(idx) for idx in parse_fp8_keep_blocks(fp8_keep_blocks)),
        )
    elif fp8_keep_blocks:
        logger.warning("--fp8_keep_blocks is set but --fp8_scaled is disabled; ignoring block FP8 exclusions.")

    _awq_scales = None  # populated if AWQ calibration is used
    _int4_awq_scales = None
    _int4_awq_save_path = None
    if int4_convrot_dynamic and (int4_convrot_awq_calibration or int4_convrot_awq_scales):
        if int4_convrot_awq_calibration:
            _int4_awq_save_path = int4_convrot_awq_scales or default_int4_convrot_awq_scales_path(model_path)
            logger.info(
                "INT4 ConvRot AWQ: computing dataset-independent scales at load time (alpha=%.3f, save=%s)",
                float(int4_convrot_awq_alpha),
                _int4_awq_save_path,
            )
        else:
            if int4_convrot_awq_scales is None:
                raise ValueError("--int4_convrot_awq_scales is required when not computing INT4 ConvRot AWQ calibration")
            _int4_awq_scales = load_int4_convrot_awq_scales(int4_convrot_awq_scales)

    # --- Auto-detect NVFP4 (Lightricks pre-quantized FP4 E2M1) ---
    _is_nvfp4 = False
    if not nf4_base and not fp8_scaled and not int4_convrot_base and not int4_convrot_dynamic and not nvfp4_training_base:
        from musubi_tuner.modules.nvfp4_utils import detect_nvfp4_checkpoint

        _check_path = model_path if isinstance(model_path, str) else model_path[0]
        _is_nvfp4 = detect_nvfp4_checkpoint(_check_path)

    if _is_nvfp4:
        from musubi_tuner.modules.nvfp4_utils import load_nvfp4_state_dict, apply_nvfp4_monkey_patch

        logger.info("Detected NVFP4 (Lightricks FP4 E2M1) checkpoint — loading with on-the-fly dequantization")
        sd = load_nvfp4_state_dict(
            model_files=model_path if isinstance(model_path, list) else [model_path],
            state_dict_key_filter=state_dict_key_filter,
            move_to_device=not load_weights_on_cpu and load_device == target_device,
            target_device=load_device if (not load_weights_on_cpu and load_device == target_device) else None,
        )
    elif nf4_base:
        nf4_calc_device = _resolved_quant_device
        logger.info("LTX-2 nf4: quantization device = %s", nf4_calc_device)
        model_files = model_path if isinstance(model_path, list) else [model_path]
        nf4_target_keys = ["transformer_blocks"]
        nf4_exclude_keys = list(KEEP_FP8_HIGH_PRECISION_TOKENS)

        # AWQ and/or LoftQ both need full-precision weights before quantization
        _needs_full_precision = (loftq_init and lora_rank > 0) or awq_calibration

        # Check for pre-quantized NF4 model (saved by ltx2_quantize_model.py)
        _check_path = model_files[0]
        _pre_quantized = False
        try:
            from safetensors import safe_open as _safe_open

            with _safe_open(_check_path, framework="pt") as _f:
                _meta = _f.metadata()
                _pre_quantized = _meta is not None and _meta.get("nf4_quantized") == "true"
        except Exception:
            pass

        if _pre_quantized:
            if awq_calibration:
                raise ValueError(
                    "Pre-quantized NF4 models are incompatible with --awq_calibration "
                    "(requires full-precision weights). Use the original model instead."
                )
            # Read block_size from pre-quantized metadata
            _saved_bs = int(_meta.get("nf4_block_size", str(nf4_block_size)))
            if _saved_bs != nf4_block_size:
                logger.info(
                    "Using block_size=%d from pre-quantized model (--nf4_block_size=%d ignored)",
                    _saved_bs,
                    nf4_block_size,
                )
                nf4_block_size = _saved_bs
            logger.info("Detected pre-quantized NF4 model (block_size=%d), skipping quantization", nf4_block_size)
            sd = {}
            for model_file in model_files:
                with MemoryEfficientSafeOpen(model_file) as f:
                    for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                        if state_dict_key_filter is not None and not state_dict_key_filter(key):
                            continue
                        sd[key] = f.get_tensor(key)
            # Load pre-computed LoftQ data from companion file if --loftq_init
            if loftq_init and lora_rank > 0:
                from musubi_tuner.ltx2_quantize_model import loftq_path_for_model
                from safetensors.torch import load_file as _load_file

                _loftq_file = loftq_path_for_model(_check_path, lora_rank)
                if os.path.isfile(_loftq_file):
                    logger.info("Loading pre-computed LoftQ data from %s", _loftq_file)
                    _loftq_sd = _load_file(_loftq_file, device="cpu")
                    # Reconstruct {lora_name: (lora_A, lora_B)} dict
                    _loftq_data = {}
                    for k in _loftq_sd:
                        if k.endswith(".lora_A"):
                            lora_name = k[: -len(".lora_A")]
                            _loftq_data[lora_name] = (_loftq_sd[f"{lora_name}.lora_A"], _loftq_sd[f"{lora_name}.lora_B"])
                    load_ltx2_model._loftq_data = _loftq_data
                    logger.info("LoftQ: loaded init data for %d modules (rank=%d)", len(_loftq_data), lora_rank)
                else:
                    raise FileNotFoundError(
                        f"--loftq_init requires pre-computed LoftQ data but file not found: {_loftq_file}\n"
                        f"Re-run ltx2_quantize_model.py with --loftq_init --network_dim {lora_rank} to generate it."
                    )
            _skip_rename = False
        elif _needs_full_precision:
            from musubi_tuner.modules.nf4_optimization_utils import optimize_state_dict_with_nf4

            sd = load_safetensors_with_lora_and_fp8(
                model_files=model_files,
                lora_weights_list=None,
                lora_multipliers=None,
                fp8_optimization=False,
                calc_device=torch.device("cpu"),
                move_to_device=False,
                dit_weight_dtype=None,
                key_filter=state_dict_key_filter,
            )
            # Rename keys (must happen before LoftQ since lora_name is built from key paths)
            renamed_sd: dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                nk = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(k)
                renamed_sd[nk if nk is not None else k] = v
            sd = renamed_sd

            # --- AWQ calibration ---
            if awq_calibration:
                from musubi_tuner.modules.awq_calibration import (
                    get_awq_cache_path,
                    load_awq_scales,
                    save_awq_scales,
                    run_synthetic_calibration,
                    apply_awq_scales_to_state_dict,
                )

                awq_cache_path = get_awq_cache_path(model_files[0])
                if os.path.exists(awq_cache_path):
                    logger.info("AWQ: loading cached scales from %s", awq_cache_path)
                    _awq_scales = load_awq_scales(awq_cache_path)
                else:
                    logger.info("AWQ: no cached scales found, running synthetic calibration...")
                    _awq_scales = run_synthetic_calibration(
                        model=base_model,
                        state_dict=sd,
                        num_batches=awq_num_batches,
                        alpha=awq_alpha,
                        target_layer_keys=nf4_target_keys,
                        exclude_layer_keys=nf4_exclude_keys,
                        device=nf4_calc_device,
                    )
                    if _awq_scales:
                        save_awq_scales(_awq_scales, awq_cache_path)
                    else:
                        logger.warning("AWQ: calibration produced no scales, proceeding without AWQ")

                # Apply AWQ scales to weights before quantization
                if _awq_scales:
                    apply_awq_scales_to_state_dict(sd, _awq_scales)
                    logger.info("AWQ: applied scales to %d weight tensors", len(_awq_scales))

                # Re-create model on meta (calibration may have loaded weights into it)
                with torch.device("meta"):
                    base_model = configurator.from_config(config)

            # --- LoftQ ---
            if loftq_init and lora_rank > 0:
                from musubi_tuner.networks.lora_ltx2 import compute_loftq_from_state_dict

                _loftq_data = compute_loftq_from_state_dict(
                    sd,
                    loftq_config={"num_iterations": loftq_iters, "block_size": nf4_block_size},
                    network_dim=lora_rank,
                    target_layer_keys=nf4_target_keys,
                    exclude_layer_keys=nf4_exclude_keys,
                )
                load_ltx2_model._loftq_data = _loftq_data

            # Quantize in-place
            sd = optimize_state_dict_with_nf4(
                sd,
                calc_device=nf4_calc_device,
                target_layer_keys=nf4_target_keys,
                exclude_layer_keys=nf4_exclude_keys,
                block_size=nf4_block_size,
                move_to_device=not load_weights_on_cpu and load_device == target_device,
            )
            _skip_rename = True
        else:
            sd = load_safetensors_with_nf4_optimization(
                model_files=model_files,
                calc_device=nf4_calc_device,
                target_layer_keys=nf4_target_keys,
                exclude_layer_keys=nf4_exclude_keys,
                block_size=nf4_block_size,
                move_to_device=not load_weights_on_cpu and load_device == target_device,
                key_filter=state_dict_key_filter,
                placement_fn=_placement_fn,
            )
            _skip_rename = False
    elif fp8_scaled:
        fp8_calc_device = _resolved_quant_device
        logger.info("LTX-2 fp8: quantization device = %s", fp8_calc_device)
        fp8_exclude_keys = list(KEEP_FP8_HIGH_PRECISION_TOKENS) + fp8_block_exclude_keys
        sd = load_safetensors_with_lora_and_fp8(
            model_files=model_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=fp8_calc_device,
            move_to_device=not load_weights_on_cpu and load_device == target_device,
            dit_weight_dtype=None,
            target_keys=["transformer_blocks"],
            exclude_keys=fp8_exclude_keys,
            key_filter=state_dict_key_filter,
            placement_fn=_placement_fn,
        )
    elif int8_base:
        logger.info("LTX-2 int8: loading pre-quantized Optimum-Quanto qint8 checkpoint")
        model_files = model_path if isinstance(model_path, list) else [model_path]
        sd = load_quanto_int8_state_dict(
            model_files,
            # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
            non_quant_dtype=torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype,
            key_filter=state_dict_key_filter,
        )
    elif int8_convrot_base:
        logger.info("LTX-2 INT8 ConvRot: loading pre-quantized checkpoint with Comfy-compatible metadata")
        model_files = model_path if isinstance(model_path, list) else [model_path]
        sd = load_comfy_int8_convrot_state_dict(
            model_files,
            # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
            non_quant_dtype=torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype,
            key_filter=state_dict_key_filter,
        )
    elif int8_dynamic:
        logger.info("LTX-2 int8: dynamic per-row int8 quantization of standard checkpoint")
        model_files = model_path if isinstance(model_path, list) else [model_path]
        sd = load_safetensors_dynamic_int8(
            model_files,
            target_keys=["transformer_blocks"],
            exclude_keys=list(KEEP_FP8_HIGH_PRECISION_TOKENS),
            # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
            non_quant_dtype=torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype,
            calc_device=_resolved_quant_device,
            key_filter=state_dict_key_filter,
        )
    elif int8_convrot_dynamic:
        logger.info("LTX-2 INT8 ConvRot: dynamic quantization of standard checkpoint")
        model_files = model_path if isinstance(model_path, list) else [model_path]
        sd = load_safetensors_dynamic_int8_convrot(
            model_files,
            target_keys=["transformer_blocks"],
            exclude_keys=list(KEEP_FP8_HIGH_PRECISION_TOKENS),
            groupsizes=int8_convrot_groupsize,
            mse_clip=bool(int8_convrot_mse_clip),
            quality_report=int8_convrot_quality_report,
            # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
            non_quant_dtype=torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype,
            calc_device=_resolved_quant_device,
            key_filter=state_dict_key_filter,
            policy=resolved_convrot_policy,
        )
    elif int4_convrot_base:
        logger.info("LTX-2 INT4 ConvRot: loading pre-quantized packed INT4 checkpoint")
        model_files = model_path if isinstance(model_path, list) else [model_path]
        sd = load_comfy_int4_convrot_state_dict(
            model_files,
            # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
            non_quant_dtype=torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype,
            key_filter=state_dict_key_filter,
        )
    elif int4_convrot_dynamic:
        logger.info("LTX-2 INT4 ConvRot: dynamic INT4 weight quantization of standard checkpoint")
        model_files = model_path if isinstance(model_path, list) else [model_path]
        sd = load_safetensors_dynamic_int4_convrot(
            model_files,
            target_keys=["transformer_blocks"],
            exclude_keys=list(KEEP_FP8_HIGH_PRECISION_TOKENS),
            groupsizes=int4_convrot_groupsize,
            mse_clip=bool(int4_convrot_mse_clip),
            quality_report=int4_convrot_quality_report,
            awq_calibration=bool(int4_convrot_awq_calibration),
            awq_alpha=float(int4_convrot_awq_alpha),
            awq_scales=_int4_awq_scales,
            awq_save_path=_int4_awq_save_path,
            stabilizer_rank=int(int4_convrot_stabilizer_rank),
            scale_refine_steps=int(int4_convrot_scale_refine_steps),
            group_scales=int(int4_convrot_group_scales),
            group_ratio_q8=bool(int4_convrot_group_ratio_q8),
            compare_group_scales=int4_convrot_compare_group_scales,
            # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
            non_quant_dtype=torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype,
            calc_device=_resolved_quant_device,
            key_filter=state_dict_key_filter,
            policy=resolved_convrot_policy,
        )
    elif nvfp4_training_base:
        from musubi_tuner.modules.nvfp4_training import (
            NVFP4_TARGET_PATTERNS,
            detect_nvfp4_training_checkpoint,
            load_nvfp4_training_state_dict,
            load_safetensors_dynamic_nvfp4_training,
            nvfp4_checkpoint_stabilizer_rank,
            nvfp4_stabilizer_rank_conflict_warning,
        )

        model_files = model_path if isinstance(model_path, list) else [model_path]
        _nvfp4_non_quant_dtype = torch.bfloat16 if torch_dtype in (None, torch.float32) else torch_dtype
        if detect_nvfp4_training_checkpoint(model_files[0]):
            logger.info("LTX-2 NVFP4: detected pre-quantized checkpoint — loading packed NVFP4 directly")
            _rank_warning = nvfp4_stabilizer_rank_conflict_warning(
                nvfp4_stabilizer_rank, nvfp4_checkpoint_stabilizer_rank(model_files[0])
            )
            if _rank_warning:
                logger.warning(_rank_warning)
            sd = load_nvfp4_training_state_dict(
                model_files,
                # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
                non_quant_dtype=_nvfp4_non_quant_dtype,
                key_filter=state_dict_key_filter,
            )
        else:
            logger.info(
                "LTX-2 NVFP4: quantizing bf16/fp16 checkpoint at load (stabilizer_rank=%d, device=%s)",
                int(nvfp4_stabilizer_rank),
                _resolved_quant_device,
            )
            sd = load_safetensors_dynamic_nvfp4_training(
                model_files,
                target_keys=list(NVFP4_TARGET_PATTERNS),
                exclude_keys=list(KEEP_FP8_HIGH_PRECISION_TOKENS),
                stabilizer_rank=int(nvfp4_stabilizer_rank),
                # Frozen base: keep non-quantized weights at bf16; these tensors are not trained.
                non_quant_dtype=_nvfp4_non_quant_dtype,
                calc_device=_resolved_quant_device,
                key_filter=state_dict_key_filter,
            )
    else:
        sd = load_safetensors_with_lora_and_fp8(
            model_files=model_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=False,
            calc_device=state_device,
            move_to_device=not load_weights_on_cpu,
            dit_weight_dtype=torch_dtype,
            target_keys=None,
            exclude_keys=None,
            key_filter=state_dict_key_filter,
            placement_fn=_placement_fn,
        )

    # Dynamic int8/int4 loaders already rename keys during streaming quantization.
    if not (int8_dynamic or int8_convrot_dynamic or int4_convrot_dynamic or (nf4_base and locals().get("_skip_rename", False))):
        renamed_sd: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(k)
            renamed_sd[nk if nk is not None else k] = v
        sd = renamed_sd

    def _trace_vram_ltx2(tag):
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / (1024**3)
            r = torch.cuda.memory_reserved() / (1024**3)
            m = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"[VRAM_TRACE_LTX2] {tag}: alloc={a:.2f}GB res={r:.2f}GB max={m:.2f}GB")

    _trace_vram_ltx2(f"AFTER state dict loading (state_device={state_device}, quantize_device={_resolved_quant_device})")
    if _is_nvfp4:
        from musubi_tuner.modules.nvfp4_utils import apply_nvfp4_monkey_patch

        apply_nvfp4_monkey_patch(base_model, sd)
    elif nf4_base:
        apply_nf4_monkey_patch(base_model, sd, block_size=nf4_block_size, awq_scales=_awq_scales)
    elif fp8_scaled:
        apply_fp8_monkey_patch(base_model, sd, use_scaled_mm=False)
    elif int4_convrot_base or int4_convrot_dynamic:
        register_int4_convrot_buffers(base_model, sd)
    elif nvfp4_training_base:
        from musubi_tuner.modules.nvfp4_training import register_nvfp4_training_buffers

        register_nvfp4_training_buffers(base_model, sd)
    elif int8_base or int8_dynamic or int8_convrot_base or int8_convrot_dynamic:
        register_quanto_int8_scale_buffers(base_model, sd)
    _trace_vram_ltx2("AFTER apply monkey patch")
    base_model.load_state_dict(sd, strict=False, assign=True)
    _trace_vram_ltx2("AFTER load_state_dict (model still on meta/cpu)")
    if resolved_block_load_range is not None:
        keep_start, keep_end = resolved_block_load_range
        _prune_transformer_blocks_to_range(base_model, keep_start=keep_start, keep_end=keep_end)
    if torch_dtype is not None:
        _cast_non_fp8_params(base_model, torch_dtype)
    if fp8_w8a8:
        apply_w8a8_monkey_patch(base_model, w8a8_mode=w8a8_mode, state_dict=sd, w8a8_backend=w8a8_backend)
        _trace_vram_ltx2("AFTER W8A8 monkey patch")
    if int8_base or int8_dynamic or int8_convrot_base or int8_convrot_dynamic:
        apply_quanto_int8_monkey_patch(base_model, w8a8_backend=w8a8_backend, policy=resolved_convrot_policy)
        _trace_vram_ltx2("AFTER quanto int8 monkey patch")
    if int4_convrot_base or int4_convrot_dynamic:
        apply_int4_convrot_monkey_patch(base_model, policy=resolved_convrot_policy)
        _trace_vram_ltx2("AFTER int4 ConvRot monkey patch")
    if nvfp4_training_base:
        from musubi_tuner.modules.nvfp4_training import apply_nvfp4_training_monkey_patch

        apply_nvfp4_training_monkey_patch(base_model)
        _trace_vram_ltx2("AFTER NVFP4 training monkey patch")
    _trace_vram_ltx2(f"AFTER _cast_non_fp8_params, BEFORE base_model.to({load_device})")
    if _placement_fn is not None:
        # Preserve streamed placement: move only the non-block modules, then place each
        # block on the device it was streamed to. A blanket .to() would undo the split.
        _saved_blocks = base_model.transformer_blocks
        base_model.transformer_blocks = torch.nn.ModuleList()
        base_model = base_model.to(load_device)
        base_model.transformer_blocks = _saved_blocks
        for _bi, _blk in enumerate(base_model.transformer_blocks):
            _blk.to(torch.device("cpu") if _bi in _swap_idx else load_device)
    else:
        base_model = base_model.to(load_device)
    _trace_vram_ltx2(f"AFTER base_model.to({load_device})")
    if fp8_upcast or fp8_upcast_stochastic:
        # Upcast FP8 linear weights during forward for stability.
        # This is optional and not enabled by default.
        base_model = amend_forward_with_upcast(
            base_model,
            with_stochastic_rounding=bool(fp8_upcast_stochastic),
            seed=int(fp8_upcast_seed),
        )
        logger.info(
            "Enabled FP8 upcast during linear forward (stochastic=%s, seed=%s).",
            bool(fp8_upcast_stochastic),
            int(fp8_upcast_seed),
        )

    model = LTX2Wrapper(base_model, patch_size=1)
    _trace_vram_ltx2("AFTER LTX2Wrapper creation")

    # Apply FFN chunking and split attention settings
    _apply_memory_optimization_settings(
        base_model,
        ffn_chunk_target=ffn_chunk_target,
        ffn_chunk_size=ffn_chunk_size,
        split_attn_target=split_attn_target,
        split_attn_mode=split_attn_mode,
        split_attn_chunk_size=split_attn_chunk_size,
    )

    if load_device == target_device and _placement_fn is None:
        model = model.to(device=target_device)
        _trace_vram_ltx2(f"AFTER model.to({target_device}) [load_device==target_device]")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(
            "LTX-2 load mem [after_load_ltx2_model]: cuda_allocated=%.2fGB cuda_reserved=%.2fGB max_allocated=%.2fGB",
            allocated,
            reserved,
            max_alloc,
        )
    return model
