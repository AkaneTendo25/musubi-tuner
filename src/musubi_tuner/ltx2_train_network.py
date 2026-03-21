"""LTX-2 LoRA Training Implementation."""

import argparse
import gc
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import wave
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
    should_sample_images,
)
from musubi_tuner.audio_supervision import (
    AudioSupervisionState,
    format_audio_supervision_alert,
    normalize_audio_supervision_mode,
    reset_audio_supervision_state,
    update_and_check_audio_supervision,
)
from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils import model_utils
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen        
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.modules.w8a8_optimization_utils import apply_w8a8_monkey_patch
from musubi_tuner.modules.nf4_optimization_utils import (
    apply_nf4_monkey_patch,
    is_nf4_module,
    load_safetensors_with_nf4_optimization,
    DEFAULT_NF4_BLOCK_SIZE,
)
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.ltx_2.env import apply_ltx2_tweaks
from musubi_tuner.ltx2_text_conditioning import (
    select_audio_text_embeds_for_audio_mode,
    select_video_text_embeds_for_video_mode,
    select_video_text_embeds_for_av_no_audio,
)
from musubi_tuner.ltx2_inference import (
    LTX2Inferencer,
    InferenceConfig,
)
from musubi_tuner.ltx2_lycoris_runtime import (
    apply_lycoris_preset_before_network_creation,
    ensure_adapters_enabled_for_sampling,
    get_adapter_norm_samples,
    is_lycoris_requested,
    process_lycoris_config,
    summarize_active_adapters,
    validate_lycoris_quantized_base_compatibility,
    validate_lycoris_runtime,
)

# LTX-2 latent normalization defaults.
# These are identity stats (mean=0, std=1). We keep them as a safe fallback and
# override them from the loaded VAE if it exposes per-channel statistics.
LTX2_LATENTS_MEAN = [0.0]
LTX2_LATENTS_STD = [1.0]

DEFAULT_SAMPLE_PROMPTS_CACHE = "ltx2_sample_prompts_cache.pt"
DEFAULT_SAMPLE_LATENTS_CACHE = "ltx2_sample_latents_cache.pt"
IC_LORA_STRATEGIES = ("auto", "none", "v2v", "audio_ref_only_ic")


def infer_ic_lora_strategy_from_preset(lora_target_preset: Optional[str]) -> str:
    """Infer IC-LoRA strategy from LoRA target preset for backward-compatible auto mode."""
    preset = str(lora_target_preset or "").lower()
    if preset == "v2v":
        return "v2v"
    if preset == "audio_ref_only_ic":
        return "audio_ref_only_ic"
    return "none"

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


def detect_ltx2_dtype(model_path: str) -> torch.dtype:
    """Detect the data type of LTX-2 model weights"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LTX-2 weights must be a .safetensors file. Got: {model_path}")

    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"Unable to detect LTX-2 dtype; no tensors found in {model_path}")

        floating_dtypes: list[torch.dtype] = []
        fp8_dtype: torch.dtype | None = None

        # Avoid loading tensors: inspect header dtype for each key.
        for key in keys:
            meta = handle.header.get(key)
            if not isinstance(meta, dict) or "dtype" not in meta:
                continue
            dt = handle._get_torch_dtype(meta["dtype"])  # noqa: SLF001
            if not isinstance(dt, torch.dtype):
                continue
            if dt.is_floating_point:
                floating_dtypes.append(dt)
                if dt.itemsize == 1:
                    fp8_dtype = dt
                    break

        dtype = fp8_dtype or (floating_dtypes[0] if floating_dtypes else handle.get_tensor(keys[0]).dtype)

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


def infer_ltx_version_from_checkpoint_config(config: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Infer checkpoint generation (2.0 vs 2.3) from metadata config markers."""
    markers: List[str] = []
    transformer_cfg = config.get("transformer", {})
    vocoder_cfg = config.get("vocoder", {})

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

    detected_version = "2.3" if markers else "2.0"
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
            logger.info("Applied FFN chunking (chunk_size=%d) to %d FeedForward modules (target=%s)",
                       ffn_chunk_size, ffn_count, ffn_chunk_target)

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
            logger.info("Applied split attention (mode=%s, chunk_size=%d) to %d Attention modules (target=%s)",
                       split_attn_mode, split_attn_chunk_size, attn_count, split_attn_target)


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
    fp8_upcast: bool = False,
    fp8_upcast_stochastic: bool = False,
    fp8_upcast_seed: int = 0,
    nf4_base: bool = False,
    nf4_block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    loftq_init: bool = False,
    loftq_iters: int = 1,
    lora_rank: int = 0,
    quantize_device: Optional[str] = None,
    awq_calibration: bool = False,
    awq_alpha: float = 0.25,
    awq_num_batches: int = 8,
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
            is_quantized_linear = isinstance(module, torch.nn.Linear) and hasattr(module, "scale_weight")
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

    # Resolve quantization device: CLI flag > env var > default (cuda)
    _qdev_raw = quantize_device or os.getenv("LTX2_NF4_CALC_DEVICE") or os.getenv("LTX2_FP8_CALC_DEVICE") or "cuda"
    _qdev = _qdev_raw.strip().lower()
    if _qdev in {"1", "true", "yes", "cuda", "gpu"}:
        if target_device.type == "cuda":
            _resolved_quant_device = target_device
        else:
            logger.warning("Quantize device '%s' requested GPU, but target device is %s; falling back to CPU.", _qdev_raw, target_device)
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
    config = loader.metadata(model_path)
    attn_mode = (attn_mode or "torch").lower()
    attn_type = None
    if attn_mode in {"xformers", "xformers-attn"}:
        attn_type = "xformers"
    elif attn_mode in {"flash3", "flash_attention_3"}:
        attn_type = "flash_attention_3"
    elif attn_mode in {"flash", "flash_attention_2"}:
        attn_type = "flash_attention_2"
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

    _awq_scales = None  # populated if AWQ calibration is used

    if nf4_base:
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
                    _saved_bs, nf4_block_size,
                )
                nf4_block_size = _saved_bs
            logger.info("Detected pre-quantized NF4 model (block_size=%d), skipping quantization", nf4_block_size)
            sd = {}
            for model_file in model_files:
                with MemoryEfficientSafeOpen(model_file) as f:
                    for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", unit="key"):
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
            )
            _skip_rename = False
    elif fp8_scaled:
        fp8_calc_device = _resolved_quant_device
        logger.info("LTX-2 fp8: quantization device = %s", fp8_calc_device)
        sd = load_safetensors_with_lora_and_fp8(
            model_files=model_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=fp8_calc_device,
            move_to_device=not load_weights_on_cpu and load_device == target_device,
            dit_weight_dtype=None,
            target_keys=["transformer_blocks"],
            exclude_keys=list(KEEP_FP8_HIGH_PRECISION_TOKENS),
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
        )

    if not (nf4_base and locals().get("_skip_rename", False)):
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

    _trace_vram_ltx2("AFTER state dict loading (sd on CPU)")
    if nf4_base:
        apply_nf4_monkey_patch(base_model, sd, block_size=nf4_block_size, awq_scales=_awq_scales)
    elif fp8_scaled:
        apply_fp8_monkey_patch(base_model, sd, use_scaled_mm=False)
    _trace_vram_ltx2("AFTER apply monkey patch")
    base_model.load_state_dict(sd, strict=False, assign=True)
    _trace_vram_ltx2("AFTER load_state_dict (model still on meta/cpu)")
    if torch_dtype is not None:
        _cast_non_fp8_params(base_model, torch_dtype)
    if fp8_w8a8:
        apply_w8a8_monkey_patch(base_model, w8a8_mode=w8a8_mode)
        _trace_vram_ltx2("AFTER W8A8 monkey patch")
    _trace_vram_ltx2(f"AFTER _cast_non_fp8_params, BEFORE base_model.to({load_device})")
    base_model = base_model.to(load_device)
    _trace_vram_ltx2(f"AFTER base_model.to({load_device})")
    if fp8_upcast or fp8_upcast_stochastic:
        # Upcast FP8 linear weights during forward for stability.
        # This is optional and not enabled by default in upstream configs.
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

    if load_device == target_device:
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


class LTX2NetworkTrainer(NetworkTrainer):
    """Trainer for LTX-2 models with LoRA support"""

    def __init__(self) -> None:
        super().__init__()
        self._text_encoder = None
        self._dit_attn_mode: Optional[str] = None
        self._latent_norm_cache: Dict = {}
        self._warned_missing_audio = False
        self._warned_ignored_ref_latents = False
        self._audio_supervision_state = AudioSupervisionState()

        # Initialize latent normalization
        mean = torch.tensor(LTX2_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LTX2_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = std.clamp_min(1e-6)
        self._latent_norm_base: Tuple[torch.Tensor, torch.Tensor] = (mean, std.reciprocal())

        self._flow_target: str = "noise"  # LTX-2 predicts noise
        self._num_timesteps: int = 1000
        self._audio_video: bool = False
        self._i2v_training: bool = False
        self._ic_lora_strategy: str = "none"
        self._ltx_mode: str = "video"
        self._ltx_version: str = "2.0"
        self._ltx2_audio_only_model: bool = False
        self._logged_audio_only_timestep_shift: bool = False
        self._audio_only_sequence_resolution: int = 64
        self._ltx2_checkpoint_config: Optional[Dict[str, Any]] = None
        self.default_guidance_scale = 3.0
        self._audio_preview_config: Optional[Dict[str, int | float]] = None

        # Preservation / regularization (off by default — zero overhead)
        self._preservation_active: bool = False
        self._preservation_helper = None
        self._last_dit_inputs: Optional[Dict[str, Any]] = None

        # CREPA (off by default)
        self._crepa = None
        # Self-Flow (off by default)
        self._self_flow = None
        self._self_flow_active: bool = False
        self._self_flow_step_context: Optional[Dict[str, Any]] = None

    @staticmethod
    def _apply_caption_dropout(
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        caption_dropout_rate: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        text_embeds = text_embeds.clone()
        if text_mask is not None:
            text_mask = text_mask.clone()

        for i in range(text_embeds.shape[0]):
            if random.random() < caption_dropout_rate:
                text_embeds[i] = 0
                if text_mask is not None:
                    text_mask[i] = False
                    if text_mask.shape[-1] > 0:
                        text_mask[i, 0] = True

        return text_embeds, text_mask

    def _build_audio_ref_transformer_overrides(
        self,
        *,
        args: argparse.Namespace,
        transformer,
        video_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        audio_model_latents: torch.Tensor,
        ref_audio_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Build optional transformer overrides for audio_ref_only_ic training/sampling."""
        overrides: Dict[str, torch.Tensor] = {}

        if ref_audio_seq_len <= 0:
            return overrides

        total_audio_seq_len = int(audio_model_latents.shape[2])
        if total_audio_seq_len <= 0:
            return overrides
        ref_tokens = max(0, min(int(ref_audio_seq_len), total_audio_seq_len))
        bsz = int(audio_model_latents.shape[0])

        mask_dtype = dtype if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32
        neg_inf = torch.finfo(mask_dtype).min

        if bool(getattr(args, "audio_ref_use_negative_positions", False)):
            from musubi_tuner.ltx_2.types import AudioLatentShape

            audio_patchifier = getattr(transformer, "_audio_patchifier", None)
            if audio_patchifier is None and hasattr(transformer, "module"):
                audio_patchifier = getattr(transformer.module, "_audio_patchifier", None)
            if audio_patchifier is None:
                logger.warning("audio_ref_use_negative_positions requested but audio patchifier is unavailable; skipping override")
            else:
                channels = int(audio_model_latents.shape[1])
                mel_bins = int(audio_model_latents.shape[3])
                tgt_tokens = total_audio_seq_len - ref_tokens

                # Generate SEPARATE position arrays for ref and target (matches ID-LoRA reference).
                # Target positions start at 0 (aligned with video time); ref positions are
                # shifted to negative time with a one-step gap for clean positional separation.
                ref_shape = AudioLatentShape(batch=bsz, channels=channels, frames=ref_tokens, mel_bins=mel_bins)
                ref_positions = audio_patchifier.get_patch_grid_bounds(ref_shape, device=device).to(dtype=mask_dtype)

                # Compute time-per-latent for the gap (hop * downsample / sample_rate)
                _hop = getattr(audio_patchifier, "hop_length", 160)
                _ds = getattr(audio_patchifier, "audio_latent_downsample_factor", 4)
                _sr = getattr(audio_patchifier, "sample_rate", 16000)
                time_per_latent = float(_hop) * float(_ds) / float(_sr)

                # Shift ref into negative time: last ref token ends at -gap
                ref_duration = ref_positions[:, :, -1:, 1:2]
                ref_positions = ref_positions - ref_duration - time_per_latent

                tgt_shape = AudioLatentShape(batch=bsz, channels=channels, frames=max(tgt_tokens, 1), mel_bins=mel_bins)
                tgt_positions = audio_patchifier.get_patch_grid_bounds(tgt_shape, device=device).to(dtype=mask_dtype)
                if tgt_tokens <= 0:
                    tgt_positions = tgt_positions[:, :, :0, :]  # empty slice

                audio_positions = torch.cat([ref_positions, tgt_positions], dim=2)
                overrides["audio_positions_override"] = audio_positions.to(
                    device=device,
                    dtype=audio_model_latents.dtype,
                )

        if bool(getattr(args, "audio_ref_mask_cross_attention_to_reference", False)):
            video_seq_len = int(video_latents.shape[2]) * int(video_latents.shape[3]) * int(video_latents.shape[4])
            if video_seq_len > 0:
                a2v_mask = torch.zeros((bsz, video_seq_len, total_audio_seq_len), device=device, dtype=mask_dtype)
                a2v_mask[:, :, :ref_tokens] = neg_inf
                overrides["a2v_cross_attention_mask"] = a2v_mask

        if bool(getattr(args, "audio_ref_mask_reference_from_text_attention", False)):
            text_seq_len = int(text_embeds.shape[1])
            audio_context_mask = torch.zeros(
                (bsz, total_audio_seq_len, text_seq_len),
                device=device,
                dtype=mask_dtype,
            )

            if text_mask is not None:
                if not isinstance(text_mask, torch.Tensor):
                    raise TypeError(f"Expected text_mask to be a torch.Tensor, got: {type(text_mask)}")
                tm = text_mask
                if tm.dim() == 1:
                    tm = tm.unsqueeze(0)
                if tm.dim() != 2:
                    raise ValueError(f"Expected text_mask to be 2D [B, seq_len], got shape: {tuple(tm.shape)}")
                if tm.shape[0] == 1 and bsz != 1:
                    tm = tm.expand(bsz, tm.shape[1])
                if tm.shape[0] != bsz:
                    raise ValueError(f"Batch mismatch for text_mask: got {tm.shape[0]}, expected {bsz}")
                if tm.shape[1] != text_seq_len:
                    if tm.shape[1] > text_seq_len:
                        tm = tm[:, -text_seq_len:]
                    else:
                        tm = F.pad(tm, (text_seq_len - tm.shape[1], 0), value=1)
                tm = tm.to(device=device)
                valid_text = tm.to(torch.bool) if tm.dtype == torch.bool else (tm > 0)
                key_bias = torch.zeros((bsz, text_seq_len), device=device, dtype=mask_dtype)
                key_bias[~valid_text] = neg_inf
                audio_context_mask = key_bias.unsqueeze(1).expand(-1, total_audio_seq_len, -1).clone()

            audio_context_mask[:, :ref_tokens, :] = neg_inf
            overrides["audio_context_mask"] = audio_context_mask

        return overrides

    # ------------------------------------------------------------------
    # Preservation / regularization hooks
    # ------------------------------------------------------------------

    def pre_train_hook(self, args: argparse.Namespace, accelerator: Accelerator,
                       transformer=None, network=None) -> None:
        self._setup_preservation(args, accelerator)
        self._setup_crepa(args, accelerator, transformer)
        self._setup_self_flow(args, accelerator, transformer, network)
        self._apply_network_initialization(args, network)
        validate_lycoris_runtime(args, accelerator, transformer, network, logger)

    def _setup_preservation(self, args: argparse.Namespace, accelerator: Accelerator) -> None:
        """Parse preservation CLI flags and prepare helper.  No-op when no flags are set."""
        blank = getattr(args, "blank_preservation", False)
        dop = getattr(args, "dop", False)
        prior_div = getattr(args, "prior_divergence", False)
        audio_dop = getattr(args, "audio_dop", False)

        if not (blank or dop or prior_div or audio_dop):
            return

        from musubi_tuner.preservation import PreservationConfig, PreservationHelper, parse_preservation_args

        # Validate audio_dop requirements
        if audio_dop:
            if self._ltx_mode != "av":
                raise ValueError("--audio_dop requires --ltx2_mode av (audio-video mode)")
            if getattr(args, "audio_silence_regularizer", False):
                logger.warning(
                    "Both --audio_dop and --audio_silence_regularizer are active. "
                    "The silence regularizer converts non-audio batches to audio batches, "
                    "so audio DOP will never fire. These are mutually exclusive."
                )

        blank_kw = parse_preservation_args(getattr(args, "blank_preservation_args", None))
        dop_kw = parse_preservation_args(getattr(args, "dop_args", None))
        prior_kw = parse_preservation_args(getattr(args, "prior_divergence_args", None))
        audio_dop_kw = parse_preservation_args(getattr(args, "audio_dop_args", None))

        cfg = PreservationConfig(
            blank_preservation=blank,
            blank_multiplier=float(blank_kw.get("multiplier", 1.0)),
            dop=dop,
            dop_multiplier=float(dop_kw.get("multiplier", 1.0)),
            dop_class_prompt=dop_kw.get("class", ""),
            prior_divergence=prior_div,
            prior_divergence_multiplier=float(prior_kw.get("multiplier", 0.1)),
            audio_dop=audio_dop,
            audio_dop_multiplier=float(audio_dop_kw.get("multiplier", 1.0)),
        )

        # Warn about DOP without class prompt (acts identical to blank preservation)
        if dop and not cfg.dop_class_prompt:
            logger.warning(
                "DOP enabled but no class prompt specified (--dop_args class=<prompt>). "
                "This will use an empty prompt, which is identical to blank preservation."
            )

        helper = PreservationHelper(cfg)
        helper.encode_prompts(self, args, accelerator)

        self._preservation_helper = helper
        self._preservation_active = True

        # Log VRAM impact: each technique adds extra transformer forward passes per step
        extra_fwd = 0
        extra_bwd = 0
        if blank:
            extra_fwd += 2  # no-grad OFF + with-grad ON
            extra_bwd += 1
        if dop:
            extra_fwd += 2
            extra_bwd += 1
        if prior_div:
            extra_fwd += 1  # no-grad OFF only
        if audio_dop:
            extra_fwd += 2  # no-grad OFF + with-grad ON (non-audio steps only)
            extra_bwd += 1
        logger.info(
            "Preservation enabled: blank=%s (x%.2f), dop=%s (class=%r, x%.2f), prior_div=%s (x%.3f), audio_dop=%s (x%.2f)",
            cfg.blank_preservation, cfg.blank_multiplier,
            cfg.dop, cfg.dop_class_prompt, cfg.dop_multiplier,
            cfg.prior_divergence, cfg.prior_divergence_multiplier,
            cfg.audio_dop, cfg.audio_dop_multiplier,
        )
        logger.warning(
            "Preservation adds +%d forward passes and +%d backward passes per training step. "
            "This significantly increases VRAM usage and step time.%s",
            extra_fwd, extra_bwd,
            " Audio DOP costs apply only on non-audio steps." if audio_dop else "",
        )

    def _setup_crepa(self, args: argparse.Namespace, accelerator: Accelerator,
                     transformer=None) -> None:
        """Parse CREPA CLI flags and install hooks.  No-op when ``--crepa`` is not set."""
        if not getattr(args, "crepa", False):
            return
        if transformer is None:
            logger.warning("CREPA enabled but transformer not available — skipping setup")
            return

        from musubi_tuner.crepa import CREPAConfig, CREPAModule, parse_crepa_args

        kw = parse_crepa_args(getattr(args, "crepa_args", None))

        # Build config — convert types from string values
        cfg_kwargs: Dict[str, Any] = {}
        _int_keys = {"student_block_idx", "teacher_block_idx", "num_neighbors", "warmup_steps", "max_steps"}
        _float_keys = {"lambda_crepa", "tau"}
        _bool_keys = {"normalize"}
        for k, v in kw.items():
            if k in _int_keys:
                cfg_kwargs[k] = int(v)
            elif k in _float_keys:
                cfg_kwargs[k] = float(v)
            elif k in _bool_keys:
                cfg_kwargs[k] = v.lower() in ("true", "1", "yes")
            else:
                cfg_kwargs[k] = v

        # Auto-fill max_steps for schedule
        if "max_steps" not in cfg_kwargs and hasattr(args, "max_train_steps"):
            cfg_kwargs["max_steps"] = args.max_train_steps

        config = CREPAConfig(**cfg_kwargs)

        unwrapped = accelerator.unwrap_model(transformer)
        module = CREPAModule(config, unwrapped)

        # Determine dtype from model
        first_param = next(iter(unwrapped.parameters()), None)
        dtype = first_param.dtype if first_param is not None else torch.float32

        module.setup(accelerator.device, dtype)

        # Try to load existing projector weights from state directory (for resume)
        if getattr(args, "resume", None):
            proj_path = os.path.join(args.resume, "crepa_projector.safetensors")
            if os.path.exists(proj_path):
                from safetensors.torch import load_file
                sd = load_file(proj_path)
                module.load_state_dict(sd)
                logger.info("CREPA: resumed projector weights from %s", proj_path)

        self._crepa = module

    def _setup_self_flow(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer=None,
        network=None,
    ) -> None:
        """Parse Self-Flow flags and install helper. No-op when ``--self_flow`` is not set."""
        if not getattr(args, "self_flow", False):
            return
        if transformer is None:
            logger.warning("Self-Flow enabled but transformer is unavailable — skipping setup")
            return
        if self._ltx_mode not in {"video", "av"}:
            raise ValueError("--self_flow currently supports --ltx_mode video or av (video branch only in av)")

        from musubi_tuner.self_flow import (
            SelfFlowConfig,
            SelfFlowModule,
            parse_self_flow_args,
        )

        kw = parse_self_flow_args(getattr(args, "self_flow_args", None))

        cfg_kwargs: Dict[str, Any] = {}
        int_keys = {
            "student_block_idx",
            "teacher_block_idx",
            "teacher_update_interval",
            "projector_hidden_multiplier",
            "num_neighbors",
            "patch_spatial_radius",
            "delta_num_steps",
            "temporal_warmup_steps",
            "temporal_max_steps",
            "student_block_stochastic_range",
        }
        float_keys = {
            "student_block_ratio",
            "teacher_block_ratio",
            "lambda_self_flow",
            "lambda_temporal",
            "lambda_delta",
            "temporal_tau",
            "patch_match_temperature",
            "motion_weight_strength",
            "mask_ratio",
            "max_loss",
            "teacher_momentum",
            "projector_lr",
        }
        bool_keys = {
            "dual_timestep",
            "tokenwise_timestep",
            "frame_level_mask",
            "mask_focus_loss",
            "offload_teacher_features",
            "offload_teacher_params",
        }
        for k, v in kw.items():
            if k in int_keys:
                cfg_kwargs[k] = int(v)
            elif k in float_keys:
                cfg_kwargs[k] = float(v)
            elif k in bool_keys:
                cfg_kwargs[k] = v.lower() in ("true", "1", "yes", "on")
            else:
                cfg_kwargs[k] = v

        if "temporal_max_steps" not in cfg_kwargs and hasattr(args, "max_train_steps"):
            cfg_kwargs["temporal_max_steps"] = args.max_train_steps

        config = SelfFlowConfig(**cfg_kwargs)
        if config.mask_ratio < 0.0 or config.mask_ratio > 0.5:
            raise ValueError("Self-Flow mask_ratio must be in [0, 0.5]")
        if config.teacher_momentum < 0.0 or config.teacher_momentum >= 1.0:
            raise ValueError("Self-Flow teacher_momentum must be in [0, 1)")
        if config.student_block_ratio is not None and not (0.0 < config.student_block_ratio < 1.0):
            raise ValueError("Self-Flow student_block_ratio must be in (0, 1)")
        if config.teacher_block_ratio is not None and not (0.0 < config.teacher_block_ratio < 1.0):
            raise ValueError("Self-Flow teacher_block_ratio must be in (0, 1)")
        if config.projector_lr is not None and config.projector_lr <= 0.0:
            raise ValueError("Self-Flow projector_lr must be > 0")
        if config.teacher_mode not in {"base", "ema", "partial_ema"}:
            raise ValueError("Self-Flow teacher_mode must be one of: base, ema, partial_ema")
        if config.student_block_stochastic_range < 0:
            raise ValueError("Self-Flow student_block_stochastic_range must be >= 0")
        if config.max_loss < 0.0:
            raise ValueError("Self-Flow max_loss must be >= 0")
        if config.loss_type not in {"negative_cosine", "one_minus_cosine"}:
            raise ValueError("Self-Flow loss_type must be one of: negative_cosine, one_minus_cosine")
        if config.temporal_mode not in {"off", "frame", "delta", "hybrid"}:
            raise ValueError("Self-Flow temporal_mode must be one of: off, frame, delta, hybrid")
        if config.temporal_granularity not in {"frame", "patch"}:
            raise ValueError("Self-Flow temporal_granularity must be one of: frame, patch")
        if config.patch_spatial_radius < 0:
            raise ValueError("Self-Flow patch_spatial_radius must be >= 0")
        if config.patch_match_mode not in {"hard", "soft"}:
            raise ValueError("Self-Flow patch_match_mode must be one of: hard, soft")
        if config.patch_match_temperature <= 0.0:
            raise ValueError("Self-Flow patch_match_temperature must be > 0")
        if config.delta_num_steps < 1:
            raise ValueError("Self-Flow delta_num_steps must be >= 1")
        if config.motion_weighting not in {"none", "teacher_delta"}:
            raise ValueError("Self-Flow motion_weighting must be one of: none, teacher_delta")
        if config.motion_weight_strength < 0.0:
            raise ValueError("Self-Flow motion_weight_strength must be >= 0")
        if config.lambda_temporal < 0.0:
            raise ValueError("Self-Flow lambda_temporal must be >= 0")
        if config.lambda_delta < 0.0:
            raise ValueError("Self-Flow lambda_delta must be >= 0")
        if config.temporal_tau <= 0.0:
            raise ValueError("Self-Flow temporal_tau must be > 0")
        if config.num_neighbors < 0:
            raise ValueError("Self-Flow num_neighbors must be >= 0")
        if config.temporal_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("Self-Flow temporal_schedule must be one of: constant, linear, cosine")
        if config.temporal_warmup_steps < 0:
            raise ValueError("Self-Flow temporal_warmup_steps must be >= 0")
        if config.temporal_max_steps < 0:
            raise ValueError("Self-Flow temporal_max_steps must be >= 0")

        unwrapped_transformer = accelerator.unwrap_model(transformer)
        if network is not None:
            unwrapped_network = accelerator.unwrap_model(network)
            self_flow_network = unwrapped_network
        else:
            # Full fine-tuning mode: transformer itself is the EMA target.
            # teacher_mode=base is incompatible (requires LoRA multipliers to create the gap).
            if str(config.teacher_mode).lower() == "base":
                raise ValueError(
                    "Self-Flow teacher_mode=base requires a LoRA network — it works by zeroing LoRA multipliers "
                    "to produce a base-model teacher pass. For full fine-tuning use teacher_mode=ema "
                    "(EMA over all transformer weights) or teacher_mode=partial_ema (EMA over teacher block only)."
                )
            self_flow_network = unwrapped_transformer
            logger.info(
                "Self-Flow: no LoRA network detected — using transformer as EMA target (teacher_mode=%s). "
                "teacher_mode=partial_ema is recommended to limit shadow-param memory to one block.",
                config.teacher_mode,
            )
        self._self_flow_network = self_flow_network
        module = SelfFlowModule(config, unwrapped_transformer)

        first_param = next(iter(unwrapped_transformer.parameters()), None)
        dtype = first_param.dtype if first_param is not None else torch.float32
        if isinstance(dtype, torch.dtype) and dtype.itemsize == 1:
            if args.mixed_precision == "fp16":
                dtype = torch.float16
            elif args.mixed_precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        module.setup(accelerator.device, dtype)
        module.init_teacher(self_flow_network)

        if getattr(args, "resume", None):
            proj_path = os.path.join(args.resume, "self_flow_projector.safetensors")
            if os.path.exists(proj_path):
                from safetensors.torch import load_file

                sd = load_file(proj_path)
                module.load_state_dict(sd)
                logger.info("Self-Flow: resumed projector weights from %s", proj_path)
            teacher_path = os.path.join(args.resume, "self_flow_teacher_ema.safetensors")
            if os.path.exists(teacher_path):
                from safetensors.torch import load_file

                teacher_sd = load_file(teacher_path)
                module.load_teacher_state_dict(teacher_sd)
                logger.info("Self-Flow: resumed EMA teacher state from %s", teacher_path)

        self._self_flow = module
        self._self_flow_active = True
        logger.warning(
            "Self-Flow is experimental and adds one extra teacher forward pass per step; expect higher VRAM/time cost."
        )

    def _apply_network_initialization(self, args: argparse.Namespace, network=None) -> None:
        """Apply network initialization customizations.

        Called after network creation to apply special initialization,
        for example LoKR perturbed normal.
        """
        if network is None:
            return

        # Apply special initialization if configured
        if hasattr(args, "_network_init_params"):
            init_params = args._network_init_params

            # LoKR perturbed normal initialization
            if "lokr_norm" in init_params:
                scale = init_params["lokr_norm"]
                logger.info(f"Applying LoKR perturbed normal initialization (scale={scale})")
                try:
                    from musubi_tuner.networks.lycoris_extensions import init_lokr_network_with_perturbed_normal
                    init_lokr_network_with_perturbed_normal(network, scale=scale)
                except Exception as e:
                    logger.warning(f"Failed to apply LoKR initialization: {e}")

    def compute_prior_divergence_addition(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        video_pred: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return ``-MSE(video_pred, prior_pred) * mult`` or None."""
        if not self._preservation_active or self._preservation_helper is None:
            return None
        cfg = self._preservation_helper.config
        if not cfg.prior_divergence:
            return None
        dit_inputs = self._last_dit_inputs
        if dit_inputs is None:
            return None

        prior_pred = self._preservation_helper.compute_prior_divergence(
            self, transformer, network, accelerator, dit_inputs, network_dtype,
        )
        div_loss = -F.mse_loss(video_pred.float(), prior_pred.float()) * cfg.prior_divergence_multiplier
        if not torch.isfinite(div_loss):
            logger.warning("Prior divergence loss is non-finite (%.4g), skipping.", div_loss.item())
            return None
        return div_loss

    def preservation_backward(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        network_dtype: torch.dtype,
    ) -> Dict[str, float]:
        """Run preservation backward passes for blank and DOP.  Returns loss dict for logging."""
        if not self._preservation_active or self._preservation_helper is None:
            return {}
        dit_inputs = self._last_dit_inputs
        self._last_dit_inputs = None  # clear for next step
        if dit_inputs is None:
            return {}

        losses: Dict[str, float] = {}
        helper = self._preservation_helper
        cfg = helper.config

        if cfg.blank_preservation:
            val = helper.compute_preservation_backward(
                "blank", self, transformer, network, accelerator, dit_inputs, network_dtype,
            )
            losses["loss/blank_pres"] = val

        if cfg.dop:
            val = helper.compute_preservation_backward(
                "dop", self, transformer, network, accelerator, dit_inputs, network_dtype,
            )
            losses["loss/dop"] = val

        if cfg.audio_dop and self._ltx_mode == "av":
            is_non_audio_batch = dit_inputs.get("audio_model_timesteps") is None
            if is_non_audio_batch:
                av_inputs = self._build_audio_dop_inputs(args, accelerator, transformer, dit_inputs, network_dtype)
                if av_inputs is not None:
                    val = helper.compute_audio_dop_backward(
                        self, transformer, network, accelerator, av_inputs, network_dtype,
                    )
                    losses["loss/audio_dop"] = val

        return losses

    def compute_self_flow_addition(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        network_dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Compute Self-Flow loss addition and logging values for the current step."""
        if not self._self_flow_active or self._self_flow is None:
            return None, {}
        if not bool(getattr(args, "self_flow", False)):
            return None, {}

        dit_inputs = self._last_dit_inputs
        sf_ctx = self._self_flow_step_context
        if dit_inputs is None or sf_ctx is None:
            self._self_flow.cleanup_step()
            return None, {}
        loss = self._self_flow.compute_loss_from_cached_features(
            num_latent_frames=sf_ctx.get("num_latent_frames"),
            latent_height=sf_ctx.get("latent_height"),
            latent_width=sf_ctx.get("latent_width"),
            token_mask=sf_ctx.get("dual_timestep_mask"),
        )

        metrics: Dict[str, float] = {}
        if loss is not None:
            metrics["loss/self_flow"] = float(loss.detach().item())
        cosine = self._self_flow.last_cosine
        if cosine is not None:
            metrics["self_flow/cosine"] = float(cosine)
        frame_cosine = self._self_flow.last_frame_cosine
        if frame_cosine is not None:
            metrics["self_flow/frame_cosine"] = float(frame_cosine)
        delta_cosine = self._self_flow.last_delta_cosine
        if delta_cosine is not None:
            metrics["self_flow/delta_cosine"] = float(delta_cosine)
        metrics["self_flow/lambda_self_flow"] = float(self._self_flow.current_lambda_self_flow)
        metrics["self_flow/lambda_temporal"] = float(self._self_flow.current_lambda_temporal)
        metrics["self_flow/lambda_delta"] = float(self._self_flow.current_lambda_delta)
        if "masked_token_ratio" in sf_ctx:
            metrics["self_flow/masked_token_ratio"] = float(sf_ctx["masked_token_ratio"])
        if "tau_mean" in sf_ctx:
            metrics["self_flow/tau_mean"] = float(sf_ctx["tau_mean"])
        if "tau_min_mean" in sf_ctx:
            metrics["self_flow/tau_min_mean"] = float(sf_ctx["tau_min_mean"])

        self._self_flow.cleanup_step()
        self._self_flow_step_context = None
        return loss, metrics

    def _get_audio_preview_config(self, args: argparse.Namespace, transformer) -> Dict[str, int | float]:
        if self._audio_preview_config is not None:
            return self._audio_preview_config

        from musubi_tuner.ltx_2.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR

        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required for audio preview config")

        config = self._load_ltx2_checkpoint_config(args)
        audio_vae_cfg = config.get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {}).get("params", {})
        ddconfig = model_cfg.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})

        sample_rate = int(model_cfg.get("sampling_rate", 16000))
        hop_length = int(stft_cfg.get("hop_length", 160))
        channels = int(ddconfig.get("z_channels", 8))
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or 64
        mel_bins = int(mel_bins)

        audio_patchify_proj = getattr(transformer, "audio_patchify_proj", None)
        audio_in_features = getattr(audio_patchify_proj, "in_features", None)
        if isinstance(audio_in_features, int) and channels > 0:
            inferred_mel = audio_in_features // channels
            if inferred_mel > 0 and inferred_mel != mel_bins:
                logger.warning(
                    "Sampling: overriding audio mel_bins from %s to %s to match audio_patchify_proj.in_features=%s",
                    mel_bins,
                    inferred_mel,
                    audio_in_features,
                )
                mel_bins = inferred_mel
            elif audio_in_features % channels != 0:
                logger.warning(
                    "Sampling: audio_patchify_proj.in_features=%s is not divisible by audio channels=%s; audio preview may fail.",
                    audio_in_features,
                    channels,
                )

        self._audio_preview_config = {
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "channels": channels,
            "mel_bins": mel_bins,
            "audio_latent_downsample_factor": int(LATENT_DOWNSAMPLE_FACTOR),
        }
        return self._audio_preview_config

    def _load_ltx2_checkpoint_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        if self._ltx2_checkpoint_config is not None:
            return self._ltx2_checkpoint_config

        from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader

        checkpoint_path = getattr(args, "ltx2_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("--ltx2_checkpoint is required to inspect checkpoint metadata")

        self._ltx2_checkpoint_config = SafetensorsModelStateDictLoader().metadata(str(checkpoint_path))
        return self._ltx2_checkpoint_config

    def _validate_ltx_version_consistency(self, args: argparse.Namespace) -> None:
        check_mode = str(getattr(args, "ltx_version_check_mode", "warn") or "warn").lower()
        if check_mode == "off":
            return
        if check_mode not in {"warn", "error"}:
            raise ValueError(
                f"Invalid ltx_version_check_mode={check_mode!r}. Expected one of: off, warn, error."
            )

        try:
            config = self._load_ltx2_checkpoint_config(args)
            detected_version, markers = infer_ltx_version_from_checkpoint_config(config)
        except Exception as exc:
            message = f"Failed to inspect checkpoint metadata for --ltx_version consistency check: {exc}"
            if check_mode == "error":
                raise ValueError(message) from exc
            logger.warning(message)
            return

        target_version = str(getattr(args, "ltx_version", self._ltx_version))
        if detected_version != target_version:
            marker_text = ", ".join(markers) if markers else "no explicit 2.3 markers"
            message = (
                f"--ltx_version={target_version} does not match checkpoint metadata (detected {detected_version}; "
                f"markers: {marker_text})."
            )
            if check_mode == "error":
                raise ValueError(message)
            logger.warning(message)
            return

        logger.info("LTX version check: --ltx_version=%s matches checkpoint metadata.", target_version)

    def _get_video_temporal_downsample(self) -> int:
        vae = getattr(self, "vae", None)
        return int(getattr(vae, "temporal_downsample_factor", 8))

    def _calculate_expected_audio_latent_length(
        self,
        args: argparse.Namespace,
        transformer,
        latent_frames: int,
        frame_rate: float,
    ) -> int:
        audio_cfg = self._get_audio_preview_config(args, transformer)
        video_temporal_factor = self._get_video_temporal_downsample()
        video_frames = max((latent_frames - 1) * video_temporal_factor + 1, 1)
        duration_s = float(video_frames) / max(float(frame_rate), 1.0)
        latents_per_second = (
            float(audio_cfg["sample_rate"])
            / float(audio_cfg["hop_length"])
            / float(audio_cfg["audio_latent_downsample_factor"])
        )
        return max(int(duration_s * latents_per_second), 1)

    def _adjust_audio_latent_duration(
        self,
        audio_latents: torch.Tensor,
        expected_length: int,
    ) -> torch.Tensor:
        actual_length = int(audio_latents.shape[2])
        if actual_length == expected_length:
            return audio_latents
        if actual_length > expected_length:
            logger.warning(
                "Trimming audio latents from %s to %s frames to match video duration.",
                actual_length,
                expected_length,
            )
            return audio_latents[:, :, :expected_length, :]
        padding_length = expected_length - actual_length
        logger.warning(
            "Padding audio latents from %s to %s frames (+%s) to match video duration.",
            actual_length,
            expected_length,
            padding_length,
        )
        padding = torch.zeros(
            audio_latents.shape[0],
            audio_latents.shape[1],
            padding_length,
            audio_latents.shape[3],
            device=audio_latents.device,
            dtype=audio_latents.dtype,
        )
        return torch.cat([audio_latents, padding], dim=2)

    def _build_empty_audio_latents(
        self,
        args: argparse.Namespace,
        transformer,
        latents: torch.Tensor,
        frame_rate: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        audio_cfg = self._get_audio_preview_config(args, transformer)
        expected_length = self._calculate_expected_audio_latent_length(
            args,
            transformer,
            latent_frames=int(latents.shape[2]),
            frame_rate=frame_rate,
        )
        return torch.zeros(
            (
                latents.shape[0],
                int(audio_cfg["channels"]),
                expected_length,
                int(audio_cfg["mel_bins"]),
            ),
            device=device,
            dtype=dtype,
        )

    def _build_audio_dop_inputs(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> Optional[Dict[str, Any]]:
        """Build AV inputs for audio DOP from a non-audio batch's dit_inputs.

        Takes the current step's noisy video, constructs silence audio latents,
        noises them at the video sigma, duplicates text embeddings to 2×cc,
        and returns a dict ready for the transformer.
        """
        device = accelerator.device

        # Extract video tensor from model_input
        model_input = dit_inputs["model_input"]
        if isinstance(model_input, (list, tuple)):
            video_input = model_input[0]
        else:
            video_input = model_input

        # Get video sigma from timesteps
        model_timesteps = dit_inputs["model_timesteps"]
        sigma = model_timesteps[:, 0] if model_timesteps.dim() > 1 else model_timesteps

        # Get frame rate
        frame_rate = dit_inputs["frame_rate"]
        if isinstance(frame_rate, torch.Tensor):
            fr_float = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()
        else:
            fr_float = float(frame_rate)

        # Build silence audio latents (zeros) with correct shape
        try:
            silence_audio = self._build_empty_audio_latents(
                args=args,
                transformer=transformer,
                latents=video_input,
                frame_rate=fr_float,
                device=device,
                dtype=network_dtype,
            )
        except Exception as e:
            logger.warning("Audio DOP: failed to build silence latents: %s", e)
            return None

        # Noise the silence audio using flow matching with video sigma
        audio_noise = torch.randn_like(silence_audio)
        sigma_audio = sigma.view(-1, 1, 1, 1).to(dtype=silence_audio.dtype)
        noisy_silence = (1.0 - sigma_audio) * silence_audio + sigma_audio * audio_noise
        del silence_audio, audio_noise

        # Build AV model_input: [noisy_video, noisy_silence_audio]
        av_model_input = [video_input, noisy_silence]

        # Duplicate text embeddings to 2×cc for AV forward
        text_embeds = dit_inputs["text_embeds"]
        if isinstance(text_embeds, torch.Tensor):
            # In non-audio batches, text_embeds is video-only (1×cc).
            # Duplicate to 2×cc so the wrapper can split into video + audio connectors.
            av_text_embeds = torch.cat([text_embeds, text_embeds], dim=-1)
        else:
            av_text_embeds = text_embeds

        # Audio timestep = video sigma (coupled timesteps for silence)
        audio_timestep = model_timesteps

        return {
            "model_input": av_model_input,
            "model_timesteps": model_timesteps,
            "audio_timestep": audio_timestep,
            "text_embeds": av_text_embeds,
            "text_mask": dit_inputs["text_mask"],
            "frame_rate": frame_rate,
            "transformer_options": dit_inputs["transformer_options"],
        }

    def _normalize_timesteps_for_model(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Normalize timesteps to the model's expected 0..1 sigma range."""
        if timesteps.numel() == 0:
            return timesteps

        return timesteps / 1000.0

    def _sample_independent_audio_timesteps(
        self,
        args: argparse.Namespace,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample audio timesteps in the same sigma range used by video timesteps."""
        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        min_sigma = (float(min_timestep) / 1000.0) if min_timestep is not None else 0.0
        max_sigma = (float(max_timestep) / 1000.0) if max_timestep is not None else 1.0
        if max_sigma < min_sigma:
            raise ValueError(f"Invalid timestep range: min_sigma={min_sigma} > max_sigma={max_sigma}")
        sigmas = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigmas = sigmas * (max_sigma - min_sigma) + min_sigma
        return sigmas.to(device=device, dtype=dtype).view(batch_size, 1)

    def _ensure_fp8_buffers_on_device(self, model: torch.nn.Module) -> None:
        if not any(True for _ in model.parameters()):
            return
        target_device = next(model.parameters()).device

        # If block swap is enabled, we must NOT call ensure_fp8_modules_on_device on the entire model
        # because it would move all swapped blocks from CPU to GPU, defeating block swapping.
        # Instead, process only non-swapped parts of the model.
        base_model = model.model if hasattr(model, "model") else model
        blocks_to_swap = getattr(base_model, "blocks_to_swap", 0) or 0

        if blocks_to_swap > 0 and hasattr(base_model, "transformer_blocks"):
            # Process non-block components (patchify, adaln, caption_projection, etc.)
            for name, child in base_model.named_children():
                if name == "transformer_blocks":
                    continue  # Skip transformer blocks - they are managed by block swap
                ensure_fp8_modules_on_device(child, target_device)

            # Only process non-swapped blocks (those that should always be on GPU)
            num_blocks = len(base_model.transformer_blocks)
            swap_start = max(0, num_blocks - blocks_to_swap)
            for idx, block in enumerate(base_model.transformer_blocks):
                if idx < swap_start:
                    # This block should be on GPU - ensure FP8 modules are on device
                    ensure_fp8_modules_on_device(block, target_device)
                # Skip swapped blocks - they are managed by the block swap mechanism
        else:
            # No block swap - process entire model as before
            ensure_fp8_modules_on_device(model, target_device)

    def _ensure_nf4_buffers_on_device(self, model: torch.nn.Module) -> None:
        """Move NF4 scale_weight buffers to the same device as the model weights.

        NF4 uint8 packed weights move naturally between CPU/GPU, but the
        scale_weight buffers (float) must be co-located with the weight for
        the dequantize forward to work.  This mirrors _ensure_fp8_buffers_on_device
        but uses the is_nf4_module check instead of FP8 dtype detection.
        """
        if not any(True for _ in model.parameters()):
            return
        target_device = next(model.parameters()).device

        base_model = model.model if hasattr(model, "model") else model
        blocks_to_swap = getattr(base_model, "blocks_to_swap", 0) or 0

        def _sync_nf4_buffers(module: torch.nn.Module, device: torch.device) -> None:
            for submodule in module.modules():
                if is_nf4_module(submodule):
                    sw = getattr(submodule, "scale_weight", None)
                    if isinstance(sw, torch.Tensor) and sw.device != device:
                        submodule.scale_weight = sw.to(device)
                    w = getattr(submodule, "weight", None)
                    if isinstance(w, torch.Tensor) and w.device != device:
                        submodule.weight = w.to(device)

        if blocks_to_swap > 0 and hasattr(base_model, "transformer_blocks"):
            for name, child in base_model.named_children():
                if name == "transformer_blocks":
                    continue
                _sync_nf4_buffers(child, target_device)
            num_blocks = len(base_model.transformer_blocks)
            swap_start = max(0, num_blocks - blocks_to_swap)
            for idx, block in enumerate(base_model.transformer_blocks):
                if idx < swap_start:
                    _sync_nf4_buffers(block, target_device)
        else:
            _sync_nf4_buffers(model, target_device)

    class _DeferredVAE:
        def __init__(self) -> None:
            self._deferred = True
            self.temporal_downsample_factor = 8
            self.spatial_downsample_factor = 32

        def to_device(self, _device) -> None:
            return None

        def to_dtype(self, _dtype) -> None:
            return None

        def eval(self) -> None:
            return None

        def requires_grad_(self, _requires_grad: bool = True):
            return self

    @staticmethod
    def _shifted_logit_normal_shift_for_sequence_length(
        seq_length: int,
        *,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        """Calculate shift value for shifted logit-normal timestep sampling.

        This matches the official LTX-2 trainer implementation where the shift
        is linearly interpolated based on sequence length.
        """
        m = (max_shift - min_shift) / float(max_tokens - min_tokens)
        b = min_shift - m * float(min_tokens)
        return m * float(seq_length) + b

    @staticmethod
    def _shifted_logit_normal_shift_for_sequence_lengths(
        seq_lengths: torch.Tensor,
        *,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> torch.Tensor:
        m = (max_shift - min_shift) / float(max_tokens - min_tokens)
        b = min_shift - m * float(min_tokens)
        return seq_lengths.to(dtype=torch.float32) * float(m) + float(b)

    @staticmethod
    def _sample_shifted_logit_normal_sigmas(
        batch_size: int,
        shifts: torch.Tensor,
        *,
        std: float = 1.0,
        mode: str = "legacy",
        eps: float = 1e-3,
        uniform_prob: float = 0.1,
    ) -> torch.Tensor:
        """Sample sigmas for shifted_logit_normal.

        Modes:
        - legacy: historical behavior, sigma = sigmoid(N(shift, std)).
        - stretched: upstream Mar-2026 behavior with percentile stretch and
          optional uniform fallback.
        """
        if shifts.ndim != 1 or shifts.shape[0] != batch_size:
            raise ValueError(f"shifts must be shape [batch_size], got {tuple(shifts.shape)} for batch_size={batch_size}")

        shifts = shifts.to(dtype=torch.float32)
        std = float(std)
        mode = str(mode).lower()

        normal_samples = torch.randn((batch_size,), device=shifts.device, dtype=torch.float32) * std + shifts
        logitnormal_samples = torch.sigmoid(normal_samples)
        if mode in {"legacy", "classic", "old"}:
            return logitnormal_samples
        if mode not in {"stretched", "v2", "upstream"}:
            raise ValueError(f"Invalid shifted_logit_mode={mode!r}. Expected one of: legacy, stretched.")

        # Upstream constants: 99.9th and 0.5th normal percentiles.
        eps = min(max(float(eps), 0.0), 0.499)
        uniform_prob = min(max(float(uniform_prob), 0.0), 1.0)
        normal_999_percentile = 3.0902 * std
        normal_005_percentile = -2.5758 * std
        percentile_999 = torch.sigmoid(shifts + normal_999_percentile)
        percentile_005 = torch.sigmoid(shifts + normal_005_percentile)
        denom = (percentile_999 - percentile_005).clamp(min=1e-6)

        stretched = (logitnormal_samples - percentile_005) / denom
        stretched = torch.where(stretched >= eps, stretched, 2 * eps - stretched)
        stretched = stretched.clamp(0.0, 1.0)

        if uniform_prob <= 0.0:
            return stretched
        uniform = (1.0 - eps) * torch.rand((batch_size,), device=shifts.device, dtype=torch.float32) + eps
        if uniform_prob >= 1.0:
            return uniform
        prob = torch.rand((batch_size,), device=shifts.device, dtype=torch.float32)
        return torch.where(prob > uniform_prob, stretched, uniform)

    def _resolve_shifted_logit_mode(self, args: argparse.Namespace) -> str:
        explicit_mode = getattr(args, "shifted_logit_mode", None)
        if explicit_mode is not None:
            mode = str(explicit_mode).lower()
            if mode in {"legacy", "stretched"}:
                return mode
            raise ValueError(f"Invalid shifted_logit_mode={explicit_mode!r}. Expected one of: legacy, stretched.")

        # Route defaults by selected LTX version for backward compatibility.
        ltx_version = str(getattr(args, "ltx_version", self._ltx_version))
        return "stretched" if ltx_version == "2.3" else "legacy"

    def _resolve_audio_only_sequence_lengths(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        latents_info = self.get_current_batch_latents_info()
        if not isinstance(latents_info, dict):
            return None

        def _as_batch_int_tensor(value: Any) -> Optional[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.view(1).to(device=device, dtype=torch.int64).expand(batch_size)
                if value.numel() == batch_size:
                    return value.to(device=device, dtype=torch.int64).view(batch_size)
                return None
            if isinstance(value, (int, float)):
                return torch.full((batch_size,), int(value), device=device, dtype=torch.int64)
            return None

        num_frames = _as_batch_int_tensor(latents_info.get("num_frames"))
        if num_frames is None:
            return None

        # Audio-only mode does not optimize video loss; use a minimal virtual spatial
        # area by default to avoid over-scaling shifted_logit_normal with large
        # (irrelevant) video resolutions.
        seq_res = int(getattr(self, "_audio_only_sequence_resolution", 64))
        if seq_res > 0:
            spatial_downsample = int(getattr(getattr(self, "vae", None), "spatial_downsample_factor", 32))
            latent_hw = max(seq_res // max(spatial_downsample, 1), 1)
            return num_frames * latent_hw * latent_hw

        height = _as_batch_int_tensor(latents_info.get("height"))
        width = _as_batch_int_tensor(latents_info.get("width"))
        if height is None or width is None:
            return None
        return num_frames * height * width

    def _resolve_shifted_logit_normal_shift(
        self,
        args: argparse.Namespace,
        seq_len: int,
    ) -> float:
        """Resolve shifted-logit-normal shift for the current mode.

        Audio-only mode requires duration-aware video latents so seq_len
        reflects target token geometry.
        """
        if self._ltx_mode == "audio" and int(seq_len) <= 1:
            raise ValueError(
                "Audio-only training requires sequence-aware video latent geometry (seq_len>1). "
                "Re-cache latents with ltx2_cache_latents.py using --ltx2_mode audio "
                "to generate duration-aware geometry."
            )

        shift = self._shifted_logit_normal_shift_for_sequence_length(seq_len)
        shift = max(0.95, min(2.05, float(shift)))
        if self._ltx_mode == "audio" and not self._logged_audio_only_timestep_shift:
            logger.info(
                "LTX-2 audio-only mode: using shifted_logit_normal shift %.4f from seq_len=%s.",
                shift,
                int(seq_len),
            )
            self._logged_audio_only_timestep_shift = True
        return shift

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # For non-video latents, use parent implementation
        if latents.dim() != 5:
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        if latents.device != device:
            latents = latents.to(device=device)
        if noise.device != device:
            noise = noise.to(device=device)
        self._self_flow_step_context = None

        batch_size = latents.shape[0]
        frames, height, width = latents.shape[2], latents.shape[3], latents.shape[4]
        seq_len = int(frames * height * width)
        audio_seq_lens = None
        if self._ltx_mode == "audio":
            audio_seq_lens = self._resolve_audio_only_sequence_lengths(batch_size, device)
            if audio_seq_lens is not None and torch.any(audio_seq_lens <= 1):
                raise ValueError(
                    "Audio-only training requires sequence-aware video latent geometry (seq_len>1). "
                    "Re-cache latents with ltx2_cache_latents.py using --ltx2_mode audio."
                )

        # Get timestep sampling mode (default to shifted_logit_normal for LTX-2)
        timestep_sampling = getattr(args, "timestep_sampling", "shifted_logit_normal")

        # For LTX-2, treat "sigma" as "shifted_logit_normal" (backward compatibility)
        if timestep_sampling == "sigma":
            timestep_sampling = "shifted_logit_normal"

        if timestep_sampling not in {"shifted_logit_normal", "uniform"}:
            # For other sampling modes, use parent implementation
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        def _sample_sigmas() -> torch.Tensor:
            if timestep_sampling == "shifted_logit_normal":
                if self._ltx_mode == "audio":
                    if audio_seq_lens is not None:
                        shifts = self._shifted_logit_normal_shift_for_sequence_lengths(audio_seq_lens)
                        shifts = shifts.clamp(min=0.95, max=2.05)
                        if not self._logged_audio_only_timestep_shift:
                            logger.info(
                                "LTX-2 audio-only mode: shifted_logit_normal seq_len min=%s max=%s mean=%.2f, "
                                "shift min=%.4f max=%.4f mean=%.4f.",
                                int(audio_seq_lens.min().item()),
                                int(audio_seq_lens.max().item()),
                                float(audio_seq_lens.to(dtype=torch.float32).mean().item()),
                                float(shifts.min().item()),
                                float(shifts.max().item()),
                                float(shifts.mean().item()),
                            )
                            self._logged_audio_only_timestep_shift = True
                    else:
                        shift = self._resolve_shifted_logit_normal_shift(args, seq_len)
                        shifts = torch.full((batch_size,), float(shift), device=device, dtype=torch.float32)
                else:
                    shift = self._shifted_logit_normal_shift_for_sequence_length(seq_len)
                    shifts = torch.full((batch_size,), float(shift), device=device, dtype=torch.float32)
                # Apply manual shift override if set
                shifted_logit_shift_override = getattr(args, "shifted_logit_shift", None)
                if shifted_logit_shift_override is not None:
                    shifts = torch.full((batch_size,), float(shifted_logit_shift_override), device=device, dtype=torch.float32)
                std = getattr(args, "logit_std", 1.0)
                shifted_logit_mode = self._resolve_shifted_logit_mode(args)
                shifted_logit_eps = getattr(args, "shifted_logit_eps", 1e-3)
                shifted_logit_uniform_prob = getattr(args, "shifted_logit_uniform_prob", 0.1)
                sampled = self._sample_shifted_logit_normal_sigmas(
                    batch_size,
                    shifts,
                    std=std,
                    mode=shifted_logit_mode,
                    eps=shifted_logit_eps,
                    uniform_prob=shifted_logit_uniform_prob,
                )
            else:
                sampled = torch.rand((batch_size,), device=device, dtype=torch.float32)

            min_timestep = getattr(args, "min_timestep", None)
            max_timestep = getattr(args, "max_timestep", None)
            if min_timestep is not None or max_timestep is not None:
                min_sigma = (min_timestep / 1000.0) if min_timestep is not None else 0.0
                max_sigma = (max_timestep / 1000.0) if max_timestep is not None else 1.0
                sampled = sampled * (max_sigma - min_sigma) + min_sigma
            return sampled

        sigmas = _sample_sigmas()

        # Optional Self-Flow dual-timestep noising for video and AV modes.
        if (
            self._self_flow_active
            and self._self_flow is not None
            and self._ltx_mode in {"video", "av"}
            and bool(getattr(args, "self_flow", False))
            and bool(getattr(self._self_flow.config, "dual_timestep", True))
        ):
            sigmas_alt = _sample_sigmas()
            t_tokens = sigmas.view(batch_size, 1).expand(batch_size, seq_len)
            s_tokens = sigmas_alt.view(batch_size, 1).expand(batch_size, seq_len)

            mask_ratio = float(getattr(self._self_flow.config, "mask_ratio", 0.10))
            mask_ratio = max(0.0, min(0.5, mask_ratio))
            if bool(getattr(self._self_flow.config, "frame_level_mask", False)):
                # Mask whole frames rather than individual tokens.
                frame_mask = torch.rand((batch_size, frames), device=device, dtype=torch.float32) < mask_ratio
                mask = frame_mask.unsqueeze(-1).expand(batch_size, frames, height * width).reshape(batch_size, seq_len)
            else:
                mask = torch.rand((batch_size, seq_len), device=device, dtype=torch.float32) < mask_ratio

            tau_tokens = torch.where(mask, s_tokens, t_tokens)
            tau_min = torch.minimum(sigmas, sigmas_alt)

            tau_latent = tau_tokens.view(batch_size, frames, height, width).unsqueeze(1)
            tau_min_latent = tau_min.view(batch_size, 1, 1, 1, 1)

            noisy_model_input = (1.0 - tau_latent) * latents.to(dtype=torch.float32) + tau_latent * noise.to(dtype=torch.float32)
            teacher_noisy = (1.0 - tau_min_latent) * latents.to(dtype=torch.float32) + tau_min_latent * noise.to(dtype=torch.float32)

            if bool(getattr(self._self_flow.config, "tokenwise_timestep", True)):
                timesteps_out = tau_tokens.to(device=device, dtype=torch.float32) * 1000.0
            else:
                timesteps_out = tau_tokens.mean(dim=1).to(device=device, dtype=torch.float32) * 1000.0
            teacher_timesteps = tau_min.to(device=device, dtype=torch.float32) * 1000.0

            self._self_flow_step_context = {
                "teacher_noisy_model_input": teacher_noisy.detach(),
                "teacher_model_timesteps": teacher_timesteps.detach(),
                "dual_timestep_mask": mask.detach(),
                "masked_token_ratio": float(mask.float().mean().item()),
                "tau_mean": float(tau_tokens.mean().item()),
                "tau_min_mean": float(tau_min.mean().item()),
                "num_latent_frames": int(frames),
                "latent_height": int(height),
                "latent_width": int(width),
            }
            return noisy_model_input, timesteps_out

        sigmas_expanded = sigmas.view(-1, 1, 1, 1, 1)
        noisy_model_input = (1.0 - sigmas_expanded) * latents.to(dtype=torch.float32) + sigmas_expanded * noise.to(dtype=torch.float32)
        timesteps_out = sigmas.to(device=device, dtype=torch.float32) * 1000.0
        return noisy_model_input, timesteps_out

    # ======== Model-specific properties and configuration ========

    @property
    def architecture(self) -> str:
        """Returns architecture identifier"""
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

        return ARCHITECTURE_LTX2

    @property
    def architecture_full_name(self) -> str:
        """Returns full architecture name with version"""
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2_FULL

        return ARCHITECTURE_LTX2_FULL

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle LTX-2-specific command line arguments"""
        self.dit_dtype = detect_ltx2_dtype(args.ltx2_checkpoint)
        if self.dit_dtype is not None and self.dit_dtype.itemsize == 1:
            if args.mixed_precision == "fp16":
                compute_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float32
            logger.warning(
                "LTX-2 weights are fp8; overriding compute dtype to %s for training stability.",
                compute_dtype,
            )
            self.dit_dtype = compute_dtype
        elif self.dit_dtype == torch.float32 and args.mixed_precision in ["fp16", "bf16"]:
            compute_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
            logger.warning(
                "LTX-2 weights are fp32; casting compute dtype to %s to reduce memory usage.",
                compute_dtype,
            )
            self.dit_dtype = compute_dtype

        if getattr(args, "nf4_base", False) and getattr(args, "fp8_base", False):
            raise ValueError("--nf4_base and --fp8_base are mutually exclusive")
        if getattr(args, "loftq_init", False) and not getattr(args, "nf4_base", False):
            raise ValueError("--loftq_init requires --nf4_base")
        if getattr(args, "awq_calibration", False) and not getattr(args, "nf4_base", False):
            raise ValueError("--awq_calibration requires --nf4_base")

        if getattr(args, "fp8_scaled", False):
            assert getattr(args, "fp8_base", False), "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

        if getattr(args, "fp8_scaled", False) and self.dit_dtype is not None and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

        if getattr(args, "fp8_w8a8", False):
            if not getattr(args, "fp8_scaled", False):
                raise ValueError("--fp8_w8a8 requires --fp8_scaled")
            if not getattr(args, "network_module", None):
                raise ValueError("--fp8_w8a8 requires LoRA training (--network_module)")
            if getattr(args, "fp8_upcast", False):
                raise ValueError("--fp8_w8a8 and --fp8_upcast are mutually exclusive")

        validate_lycoris_quantized_base_compatibility(args, logger, DEFAULT_NF4_BLOCK_SIZE)

        if getattr(args, "save_original_lora", True) and not getattr(args, "convert_to_comfy", True):
            logger.info("--no_convert_to_comfy is set; original LoRA is always saved (--save_original_lora has no extra effect).")

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "LTX-2 weights are fp16; mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "LTX-2 weights are bf16; mixed precision must be bf16 or no"

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        ltx_mode = getattr(args, "ltx_mode", "video")
        if ltx_mode not in {"video", "av", "audio"}:
            raise ValueError(f"Invalid ltx_mode: {ltx_mode}")
        self._ltx_mode = ltx_mode

        ltx_version = str(getattr(args, "ltx_version", "2.0"))
        if ltx_version not in {"2.0", "2.3"}:
            raise ValueError(f"Invalid ltx_version: {ltx_version}. Expected '2.0' or '2.3'.")
        self._ltx_version = ltx_version
        args.ltx_version = ltx_version
        ltx_version_check_mode = str(getattr(args, "ltx_version_check_mode", "warn") or "warn").lower()
        if ltx_version_check_mode not in {"off", "warn", "error"}:
            raise ValueError(
                f"ltx_version_check_mode must be one of ['off', 'warn', 'error']. Got: {ltx_version_check_mode}"
            )
        args.ltx_version_check_mode = ltx_version_check_mode
        self._validate_ltx_version_consistency(args)

        self._audio_video = self._ltx_mode in {"av", "audio"}
        self._ltx2_audio_only_model = bool(getattr(args, "ltx2_audio_only_model", False))
        if self._ltx2_audio_only_model and self._ltx_mode != "audio":
            raise ValueError("--ltx2_audio_only_model requires --ltx2_mode audio")
        self.default_guidance_scale = 1.0
        audio_only_sequence_resolution = int(getattr(args, "audio_only_sequence_resolution", 64))
        if audio_only_sequence_resolution != 0 and audio_only_sequence_resolution < 32:
            raise ValueError(
                "audio_only_sequence_resolution must be 0 (use cached virtual geometry) "
                f"or >= 32, got {audio_only_sequence_resolution}."
            )
        self._audio_only_sequence_resolution = audio_only_sequence_resolution

        args.weighting_scheme = "none"

        audio_balance_mode = str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower()
        if audio_balance_mode not in {"none", "inv_freq", "ema_mag", "uncertainty"}:
            raise ValueError(
                f"audio_loss_balance_mode must be one of ['none', 'inv_freq', 'ema_mag', 'uncertainty']. Got: {audio_balance_mode}"
            )
        args.audio_loss_balance_mode = audio_balance_mode

        audio_balance_beta = float(getattr(args, "audio_loss_balance_beta", 0.01))
        audio_balance_eps = float(getattr(args, "audio_loss_balance_eps", 0.05))
        audio_balance_min = float(getattr(args, "audio_loss_balance_min", 0.05))
        audio_balance_max = float(getattr(args, "audio_loss_balance_max", 4.0))
        audio_balance_ema_init = float(getattr(args, "audio_loss_balance_ema_init", 1.0))
        audio_balance_target_ratio = float(getattr(args, "audio_loss_balance_target_ratio", 0.33))
        audio_balance_ema_decay = float(getattr(args, "audio_loss_balance_ema_decay", 0.99))

        if not (0.0 < audio_balance_beta <= 1.0):
            raise ValueError(f"audio_loss_balance_beta must be in (0, 1]. Got: {audio_balance_beta}")
        if audio_balance_eps <= 0.0:
            raise ValueError(f"audio_loss_balance_eps must be > 0. Got: {audio_balance_eps}")
        if audio_balance_min < 0.0:
            raise ValueError(f"audio_loss_balance_min must be >= 0. Got: {audio_balance_min}")
        if audio_balance_max <= 0.0:
            raise ValueError(f"audio_loss_balance_max must be > 0. Got: {audio_balance_max}")
        if audio_balance_max < audio_balance_min:
            raise ValueError(
                f"audio_loss_balance_max must be >= audio_loss_balance_min. Got: min={audio_balance_min}, max={audio_balance_max}"
            )
        if audio_balance_mode == "inv_freq":
            if not (0.0 < audio_balance_ema_init <= 1.0):
                raise ValueError(f"audio_loss_balance_ema_init must be in (0, 1] for inv_freq. Got: {audio_balance_ema_init}")
        else:
            if audio_balance_ema_init <= 0.0:
                raise ValueError(f"audio_loss_balance_ema_init must be > 0. Got: {audio_balance_ema_init}")
        if audio_balance_target_ratio < 0.0:
            raise ValueError(f"audio_loss_balance_target_ratio must be >= 0. Got: {audio_balance_target_ratio}")
        if not (0.0 < audio_balance_ema_decay < 1.0):
            raise ValueError(f"audio_loss_balance_ema_decay must be in (0, 1). Got: {audio_balance_ema_decay}")

        args.audio_loss_balance_beta = audio_balance_beta
        args.audio_loss_balance_eps = audio_balance_eps
        args.audio_loss_balance_min = audio_balance_min
        args.audio_loss_balance_max = audio_balance_max
        args.audio_loss_balance_ema_init = audio_balance_ema_init
        args.audio_loss_balance_target_ratio = audio_balance_target_ratio
        args.audio_loss_balance_ema_decay = audio_balance_ema_decay

        shifted_logit_mode = getattr(args, "shifted_logit_mode", None)
        if shifted_logit_mode is not None:
            shifted_logit_mode = str(shifted_logit_mode).lower()
            if shifted_logit_mode not in {"legacy", "stretched"}:
                raise ValueError(
                    f"shifted_logit_mode must be one of ['legacy', 'stretched']. Got: {shifted_logit_mode}"
                )
            args.shifted_logit_mode = shifted_logit_mode

        shifted_logit_eps = float(getattr(args, "shifted_logit_eps", 1e-3))
        shifted_logit_uniform_prob = float(getattr(args, "shifted_logit_uniform_prob", 0.1))
        if shifted_logit_eps < 0.0:
            raise ValueError(f"shifted_logit_eps must be >= 0. Got: {shifted_logit_eps}")
        if not (0.0 <= shifted_logit_uniform_prob <= 1.0):
            raise ValueError(
                f"shifted_logit_uniform_prob must be within [0, 1]. Got: {shifted_logit_uniform_prob}"
            )
        args.shifted_logit_eps = shifted_logit_eps
        args.shifted_logit_uniform_prob = shifted_logit_uniform_prob

        args.independent_audio_timestep = bool(getattr(args, "independent_audio_timestep", False))
        args.audio_silence_regularizer = bool(getattr(args, "audio_silence_regularizer", False))
        audio_silence_regularizer_weight = float(getattr(args, "audio_silence_regularizer_weight", 1.0))
        if audio_silence_regularizer_weight < 0.0:
            raise ValueError(
                f"audio_silence_regularizer_weight must be >= 0. Got: {audio_silence_regularizer_weight}"
            )
        args.audio_silence_regularizer_weight = audio_silence_regularizer_weight

        audio_supervision_mode = normalize_audio_supervision_mode(
            getattr(args, "audio_supervision_mode", "off")
        )
        audio_supervision_warmup_steps = int(getattr(args, "audio_supervision_warmup_steps", 50))
        audio_supervision_check_interval = int(getattr(args, "audio_supervision_check_interval", 50))
        audio_supervision_min_ratio = float(getattr(args, "audio_supervision_min_ratio", 0.9))
        if audio_supervision_warmup_steps < 0:
            raise ValueError(
                f"audio_supervision_warmup_steps must be >= 0. Got: {audio_supervision_warmup_steps}"
            )
        if audio_supervision_check_interval <= 0:
            raise ValueError(
                f"audio_supervision_check_interval must be > 0. Got: {audio_supervision_check_interval}"
            )
        if not (0.0 <= audio_supervision_min_ratio <= 1.0):
            raise ValueError(
                f"audio_supervision_min_ratio must be in [0, 1]. Got: {audio_supervision_min_ratio}"
            )
        args.audio_supervision_mode = audio_supervision_mode
        args.audio_supervision_warmup_steps = audio_supervision_warmup_steps
        args.audio_supervision_check_interval = audio_supervision_check_interval
        args.audio_supervision_min_ratio = audio_supervision_min_ratio

        reset_audio_supervision_state(self._audio_supervision_state)

        ic_lora_strategy = str(getattr(args, "ic_lora_strategy", "auto") or "auto").lower()
        if ic_lora_strategy not in IC_LORA_STRATEGIES:
            raise ValueError(
                f"ic_lora_strategy must be one of {list(IC_LORA_STRATEGIES)}. Got: {ic_lora_strategy}"
            )
        if ic_lora_strategy == "auto":
            ic_lora_strategy = infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v"))

        if ic_lora_strategy == "audio_ref_only_ic" and self._ltx_mode not in {"av", "audio"}:
            raise ValueError("--ic_lora_strategy audio_ref_only_ic requires --ltx2_mode av or audio")

        self._ic_lora_strategy = ic_lora_strategy
        args.ic_lora_strategy = ic_lora_strategy
        args.audio_ref_use_negative_positions = bool(getattr(args, "audio_ref_use_negative_positions", False))
        args.audio_ref_mask_cross_attention_to_reference = bool(
            getattr(args, "audio_ref_mask_cross_attention_to_reference", False)
        )
        args.audio_ref_mask_reference_from_text_attention = bool(
            getattr(args, "audio_ref_mask_reference_from_text_attention", False)
        )
        args.audio_ref_identity_guidance_scale = float(getattr(args, "audio_ref_identity_guidance_scale", 0.0) or 0.0)
        if args.audio_ref_identity_guidance_scale < 0.0:
            raise ValueError(
                f"audio_ref_identity_guidance_scale must be >= 0. Got: {args.audio_ref_identity_guidance_scale}"
            )
        if ic_lora_strategy != "audio_ref_only_ic":
            if (
                args.audio_ref_use_negative_positions
                or args.audio_ref_mask_cross_attention_to_reference
                or args.audio_ref_mask_reference_from_text_attention
                or args.audio_ref_identity_guidance_scale > 0.0
            ):
                logger.warning(
                    "audio_ref_* options are set but --ic_lora_strategy is '%s'; options will be ignored.",
                    ic_lora_strategy,
                )

        # IC-LoRA strategies enable I2V-capable sampling flow in trainer.
        self._i2v_training = ic_lora_strategy in {"v2v", "audio_ref_only_ic"}

        apply_ltx2_tweaks(args)

    @property
    def i2v_training(self) -> bool:
        """True when training v2v / IC-LoRA (enables I2V conditioning in sampling)"""
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        """LTX-2 doesn't currently support control conditioning"""
        return False

    def get_checkpoint_metadata(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Return LTX-2-specific metadata for LoRA safetensors (v2v mode info, etc.)."""
        md: Dict[str, Any] = {}
        preset = getattr(args, "lora_target_preset", None)
        if preset:
            md["ss_lora_target_preset"] = preset
        if self._ic_lora_strategy and self._ic_lora_strategy != "none":
            md["ss_ic_lora_strategy"] = self._ic_lora_strategy
        if self._ic_lora_strategy == "v2v":
            md["ss_v2v_training"] = True
        elif self._i2v_training:
            md["ss_i2v_training"] = True
        ref_downscale = max(1, getattr(args, "reference_downscale", 1))
        if ref_downscale != 1:
            md["ss_reference_downscale_factor"] = ref_downscale
        return md
    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Convert saved LoRA to ComfyUI format."""
        if not getattr(args, 'convert_to_comfy', True):
            return

        try:
            from musubi_tuner.ltx_2.convert_lora_to_comfy import convert_lora_to_comfy
            comfy_ckpt_name = ckpt_name.replace('.safetensors', '.comfy.safetensors')
            comfy_ckpt_file = os.path.join(args.output_dir, comfy_ckpt_name)
            convert_lora_to_comfy(ckpt_file, comfy_ckpt_file, verbose=False)
            accelerator.print(f"Saved ComfyUI-compatible LoRA: {comfy_ckpt_file}")

            # Upload ComfyUI version to HuggingFace if enabled
            if args.huggingface_repo_id is not None:
                from musubi_tuner.utils import huggingface_utils
                huggingface_utils.upload(args, comfy_ckpt_file, "/" + comfy_ckpt_name, force_sync_upload=force_sync_upload)

            if not getattr(args, "save_original_lora", True):
                if os.path.exists(ckpt_file):
                    try:
                        os.remove(ckpt_file)  # --no_save_original_lora: keep only ComfyUI LoRA
                        accelerator.print(f"Removed original LoRA checkpoint (--no_save_original_lora): {ckpt_file}")
                    except Exception as e:
                        accelerator.print(f"Warning: Failed to remove original checkpoint '{ckpt_file}': {e}")
        except Exception as e:
            accelerator.print(f"Warning: Failed to convert LoRA to ComfyUI format: {e}")

    # ======== Model loading ========

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        """Load LTX-2 transformer model

        Args:
            accelerator: HF Accelerator instance
            args: Training arguments
            dit_path: Path to LTX-2 weights
            attn_mode: Attention implementation
            split_attn: Whether to split attention (ignored for LTX-2)
            loading_device: Device to load weights to
            dit_weight_dtype: Weight data type

        Returns:
            Loaded LTX-2 transformer model
        """
        # Determine attention mode from args
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.flash3:
            attn_mode = "flash3"
        elif args.xformers:
            attn_mode = "xformers"
        else:
            attn_mode = "torch"

        self._dit_attn_mode = attn_mode

        torch_dtype_to_use = dit_weight_dtype or self.dit_dtype or torch.float32
        if dit_weight_dtype is None:
            logger.info("LTX-2 weight dtype not set; using %s for loading", torch_dtype_to_use)
        transformer = load_ltx2_model(
            model_path=dit_path,
            device=accelerator.device,
            load_device=loading_device,
            torch_dtype=torch_dtype_to_use,
            attn_mode=attn_mode,
            audio_video=self._audio_video,
            audio_only_model=self._ltx2_audio_only_model,
            split_attn_target=getattr(args, "split_attn_target", None),
            split_attn_mode=getattr(args, "split_attn_mode", None),
            split_attn_chunk_size=int(getattr(args, "split_attn_chunk_size", 0) or 0),
            ffn_chunk_target=getattr(args, "ffn_chunk_target", None),
            ffn_chunk_size=int(getattr(args, "ffn_chunk_size", 0) or 0),
            fp8_scaled=bool(getattr(args, "fp8_scaled", False)),
            fp8_w8a8=bool(getattr(args, "fp8_w8a8", False)),
            w8a8_mode=str(getattr(args, "w8a8_mode", "int8")),
            fp8_upcast=bool(getattr(args, "fp8_upcast", False)),
            fp8_upcast_stochastic=bool(getattr(args, "fp8_upcast_stochastic", False)),
            fp8_upcast_seed=int(getattr(args, "fp8_upcast_seed", 0)),
            nf4_base=bool(getattr(args, "nf4_base", False)),
            nf4_block_size=int(getattr(args, "nf4_block_size", DEFAULT_NF4_BLOCK_SIZE)),
            loftq_init=bool(getattr(args, "loftq_init", False)),
            loftq_iters=int(getattr(args, "loftq_iters", 2)),
            lora_rank=int(getattr(args, "network_dim", 0) or 0),
            quantize_device=getattr(args, "quantize_device", None),
            awq_calibration=bool(getattr(args, "awq_calibration", False)),
            awq_alpha=float(getattr(args, "awq_alpha", 0.25)),
            awq_num_batches=int(getattr(args, "awq_num_batches", 8)),
        )

        transformer.eval()
        transformer.requires_grad_(False)

        return transformer

    def compile_transformer(self, args: argparse.Namespace, transformer):
        base_model = transformer.model if hasattr(transformer, "model") else transformer
        target_blocks = []
        if hasattr(base_model, "transformer_blocks"):
            target_blocks.append(base_model.transformer_blocks)
        return model_utils.compile_transformer(args, transformer, target_blocks, disable_linear=self.blocks_to_swap > 0)

    def _load_vae_impl(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load VAE for LTX2"""
        logger.info(f"Loading VAE from {vae_path}")
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER

        class _LTX2VideoVAE(torch.nn.Module):
            def __init__(self, decoder: torch.nn.Module):
                super().__init__()
                self.decoder = decoder

                first_param = next(self.decoder.parameters())
                self.device = first_param.device
                self.dtype = first_param.dtype

                # LTX Video VAE configuration compresses frames by 8 (except the first frame) and spatial dims by 32.
                self.temporal_downsample_factor = 8
                self.spatial_downsample_factor = 32

                stats = getattr(self.decoder, "per_channel_statistics", None)
                self.latents_mean = None
                self.latents_std = None
                if stats is not None:
                    try:
                        self.latents_mean = stats.get_buffer("mean-of-means").detach().cpu()
                        self.latents_std = stats.get_buffer("std-of-means").detach().cpu()
                    except Exception:
                        self.latents_mean = None
                        self.latents_std = None

            def to_device(self, device: torch.device | str) -> None:
                self.device = torch.device(device)
                self.decoder.to(self.device)

            def to_dtype(self, dtype: torch.dtype) -> None:
                self.dtype = dtype
                self.decoder.to(dtype=dtype)

            def eval(self) -> None:
                self.decoder.eval()

            def requires_grad_(self, requires_grad: bool = True):
                self.decoder.requires_grad_(requires_grad)
                return self

            def decode(self, zs):
                outs = []
                for z in zs:
                    if z.dim() == 4:
                        z = z.unsqueeze(0)
                    z = z.to(device=self.device, dtype=self.dtype)
                    video = self.decoder(z)
                    outs.append(video.squeeze(0))
                return outs

            def tiled_decode(self, z, tiling_config=None):
                """Decode latents using tiled processing to reduce VRAM usage.
                
                Args:
                    z: Latent tensor [C, T, H, W] or [B, C, T, H, W]
                    tiling_config: TilingConfig object for spatial/temporal tiling
                    
                Returns:
                    Decoded video tensor [B, C, T, H, W]
                """
                if z.dim() == 4:
                    z = z.unsqueeze(0)
                z = z.to(device=self.device, dtype=self.dtype)
                
                # Collect all chunks from tiled decode generator
                chunks = []
                for frame_chunk in self.decoder.tiled_decode(z, tiling_config):
                    chunks.append(frame_chunk)
                
                # Concatenate along temporal dimension
                video = torch.cat(chunks, dim=2)  # [B, C, T, H, W]
                return video

        decoder = SingleGPUModelBuilder(
            model_path=str(vae_path),
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=torch.device("cpu"), dtype=vae_dtype)
        decoder.eval()
        decoder.requires_grad_(False)

        vae = _LTX2VideoVAE(decoder)
        self._update_latent_norm_base_from_vae(vae)
        return vae

    def _load_audio_components(
        self,
        args: argparse.Namespace,
        audio_dtype: torch.dtype,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ):
        device = device or torch.device("cpu")
        logger.info("Loading LTX-2 audio decoder/vocoder from %s (device=%s)", checkpoint_path, device)
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
            AudioDecoderConfigurator,
            VocoderConfigurator,
            AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            VOCODER_COMFY_KEYS_FILTER,
        )

        audio_decoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=audio_dtype)
        vocoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=audio_dtype)

        audio_decoder.eval()
        vocoder.eval()
        return audio_decoder, vocoder

    @staticmethod
    def _save_audio_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
        audio = audio.detach().cpu().float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        if audio.shape[0] > 2:
            audio = audio[:2, :]
        audio_int16 = (audio.clamp(-1, 1) * 32767.0).to(torch.int16)
        interleaved = audio_int16.t().contiguous().numpy().tobytes()
        with wave.open(path, "wb") as wav:
            wav.setnchannels(audio_int16.shape[0])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(interleaved)

    def _decode_audio_preview_subprocess(
        self,
        *,
        audio_latents: torch.Tensor,
        output_path: str,
        checkpoint_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix="_ltx2_audio_latents.pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save({"latents": audio_latents.detach().cpu()}, tmp_path)
            cmd = [
                sys.executable,
                "-m",
                "musubi_tuner.ltx2_audio_preview",
                "--checkpoint",
                checkpoint_path,
                "--input",
                tmp_path,
                "--output",
                output_path,
                "--device",
                "auto",
                "--dtype",
                "fp32",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(
                    "Audio preview subprocess failed (code=%s): %s",
                    result.returncode,
                    (result.stderr or result.stdout).strip(),
                )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _cleanup_cuda(device: torch.device) -> None:
        clean_memory_on_device(device)
        if device.type == "cuda":
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()

    @staticmethod
    def _mux_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return
        try:
            import av
            import numpy as np
        except Exception as exc:
            logger.warning("Sampling: unable to mux audio/video (PyAV missing?): %s", exc)
            return

        with wave.open(audio_path, "rb") as wav_in:
            sample_rate = wav_in.getframerate()
            channels = wav_in.getnchannels()
            frames = wav_in.readframes(wav_in.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        else:
            audio = audio.reshape(-1, 1)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        container_in = av.open(video_path)
        video_stream_in = next((s for s in container_in.streams if s.type == "video"), None)
        if video_stream_in is None:
            container_in.close()
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        container_out = av.open(output_path, mode="w")
        video_stream_out = container_out.add_stream(
            "libx264",
            rate=video_stream_in.average_rate or video_stream_in.base_rate or 24,
        )
        video_stream_out.width = video_stream_in.width
        video_stream_out.height = video_stream_in.height
        video_stream_out.pix_fmt = "yuv420p"

        audio_stream = container_out.add_stream("aac", rate=sample_rate)
        audio_stream.codec_context.sample_rate = sample_rate
        audio_stream.codec_context.layout = "stereo"
        audio_stream.codec_context.time_base = Fraction(1, sample_rate)

        for frame in container_in.decode(video_stream_in):
            for packet in video_stream_out.encode(frame):
                container_out.mux(packet)
        for packet in video_stream_out.encode():
            container_out.mux(packet)

        frame_in = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="stereo")
        frame_in.sample_rate = sample_rate
        target_format = audio_stream.codec_context.format or "fltp"
        target_layout = audio_stream.codec_context.layout or "stereo"
        target_rate = audio_stream.codec_context.sample_rate or sample_rate
        audio_resampler = av.audio.resampler.AudioResampler(
            format=target_format,
            layout=target_layout,
            rate=target_rate,
        )
        audio_next_pts = 0
        for rframe in audio_resampler.resample(frame_in):
            if rframe.pts is None:
                rframe.pts = audio_next_pts
            audio_next_pts += rframe.samples
            rframe.sample_rate = sample_rate
            for packet in audio_stream.encode(rframe):
                container_out.mux(packet)
        for packet in audio_stream.encode():
            container_out.mux(packet)

        container_out.close()
        container_in.close()

    @staticmethod
    def _override_attention_function(transformer, attention_function):
        from musubi_tuner.ltx_2.model.transformer.attention import Attention

        overrides = []
        for module in transformer.modules():
            if isinstance(module, Attention):
                overrides.append((module, module.attention_function))
                module.attention_function = attention_function
        return overrides

    @staticmethod
    def _restore_attention_function(overrides) -> None:
        for module, attention_function in overrides:
            module.attention_function = attention_function

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        use_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )
        if getattr(args, "sample_prompts", None) or use_precached:
            logger.info("LTX-2 sampling: deferring VAE load until sampling")
            return self._DeferredVAE()
        return self._load_vae_impl(args, vae_dtype, vae_path)

    def _update_latent_norm_base_from_vae(self, vae) -> None:
        """Update latent normalization statistics from VAE config"""
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)

        if latents_mean is None or latents_std is None:
            # Some VAE wrappers expose mean/std instead of latents_mean/latents_std
            latents_mean = getattr(vae, "mean", None)
            latents_std = getattr(vae, "std", None)

        if latents_mean is None or latents_std is None:
            config = getattr(vae, "config", None)
            if config is None:
                return
            latents_mean = getattr(config, "latents_mean", None)
            latents_std = getattr(config, "latents_std", None)

        if latents_mean is None or latents_std is None:
            return

        if isinstance(latents_mean, torch.Tensor):
            mean = latents_mean.to(dtype=torch.float32).view(1, -1, 1, 1, 1)
        else:
            mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)

        if isinstance(latents_std, torch.Tensor):
            std = latents_std.to(dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        else:
            std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        self._latent_norm_base = (mean, std.reciprocal())
        self._latent_norm_cache.clear()

    # ======== Training loop methods ========

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Tuple[object, torch.Tensor]:
        """Forward pass through LTX-2 (video or audio-video) model

        Args:
            args: Training arguments
            accelerator: HF Accelerator
            transformer: LTX-2 model
            latents: Video latents [B, 128, T, H, W]
            batch: Batch data including text embeddings
            noise: Noise tensor (same shape as latents)
            noisy_model_input: Noisy latents [B, 128, T, H, W]
            timesteps: Diffusion timesteps (normalized 0-1 for flow matching)
            network_dtype: Network precision

        Returns:
            Tuple of (model_prediction, target) for loss computation
        """
        diag_enabled = os.getenv("LTX2_NAN_DIAG", "0") == "1"
        skip_nonfinite = bool(getattr(args, "skip_nonfinite_steps", False))
        nonfinite_flag = {"hit": False, "tag": None}

        def _check_finite(tag: str, tensor: Optional[torch.Tensor]) -> None:
            if not skip_nonfinite or tensor is None:
                return
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).sum().item()
                logger.error("%s has non-finite values (count=%s).", tag, bad)
                nonfinite_flag["hit"] = True
                nonfinite_flag["tag"] = tag
                return

        def _log_stats(tag: str, tensor: Optional[torch.Tensor]) -> None:
            if not diag_enabled or tensor is None:
                return
            with torch.no_grad():
                t = tensor.detach().float()
                logger.info(
                    "DIAG %s: shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
                    tag,
                    tuple(t.shape),
                    float(t.min().item()),
                    float(t.max().item()),
                    float(t.mean().item()),
                    float(t.std().item()),
                )

        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got: {type(batch)}")

        def _resolve_loss_weight(batch_key: str, arg_key: str, default: float = 1.0) -> float:
            batch_value = batch.get(batch_key)
            if batch_value is not None:
                try:
                    return float(batch_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{batch_key} must be a float-compatible scalar, got {batch_value!r}") from exc
            return float(getattr(args, arg_key, default))

        if latents is None or not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected latents to be a torch.Tensor, got: {type(latents)}")
        if latents.dim() != 5:
            raise ValueError(f"Expected latents to be 5D [B, C, F, H, W], got shape: {tuple(latents.shape)}")
        in_channels = getattr(transformer, "in_channels", None)
        if in_channels is None and hasattr(transformer, "patchify_proj"):
            in_channels = getattr(getattr(transformer, "patchify_proj", None), "in_features", None)
        if in_channels is not None and latents.shape[1] != int(in_channels):
            raise ValueError(
                f"Latents channel mismatch: got {latents.shape[1]}, expected {int(in_channels)} (transformer.in_channels)"
            )
        if not torch.isfinite(latents).all():
            raise ValueError("Non-finite (NaN/Inf) detected in latents")
        _log_stats("latents", latents)

        if timesteps is None or not isinstance(timesteps, torch.Tensor):
            raise TypeError(f"Expected timesteps to be a torch.Tensor, got: {type(timesteps)}")

        conditions = batch.get("conditions")
        if conditions is not None:
            if not isinstance(conditions, dict):
                raise TypeError(f"Expected batch['conditions'] to be a dict, got: {type(conditions)}")
            if self._ltx_mode == "audio":
                text_embeds = conditions.get("audio_prompt_embeds")
                if text_embeds is None:
                    text_embeds = conditions.get("prompt_embeds")
                if text_embeds is None:
                    text_embeds = conditions.get("video_prompt_embeds")
            elif self._audio_video:
                video_prompt_embeds = conditions.get("video_prompt_embeds")
                audio_prompt_embeds = conditions.get("audio_prompt_embeds")
                if video_prompt_embeds is not None and audio_prompt_embeds is not None:
                    if not isinstance(video_prompt_embeds, torch.Tensor) or video_prompt_embeds.dim() != 3:
                        raise ValueError(
                            f"conditions['video_prompt_embeds'] must be a 3D tensor [B, seq_len, dim], "
                            f"got {type(video_prompt_embeds).__name__} "
                            f"{tuple(video_prompt_embeds.shape) if isinstance(video_prompt_embeds, torch.Tensor) else ''}"
                        )
                    if not isinstance(audio_prompt_embeds, torch.Tensor) or audio_prompt_embeds.dim() != 3:
                        raise ValueError(
                            f"conditions['audio_prompt_embeds'] must be a 3D tensor [B, seq_len, dim], "
                            f"got {type(audio_prompt_embeds).__name__} "
                            f"{tuple(audio_prompt_embeds.shape) if isinstance(audio_prompt_embeds, torch.Tensor) else ''}"
                        )
                    if video_prompt_embeds.shape[:2] != audio_prompt_embeds.shape[:2]:
                        raise ValueError(
                            f"video_prompt_embeds {tuple(video_prompt_embeds.shape)} and audio_prompt_embeds "
                            f"{tuple(audio_prompt_embeds.shape)} must have the same batch and seq_len dimensions. "
                            "Caches may have been created with different sequence length settings or different checkpoints."
                        )
                    text_embeds = torch.cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)
                else:
                    text_embeds = conditions.get("prompt_embeds")
            else:
                text_embeds = conditions.get("video_prompt_embeds")

            text_mask = conditions.get("prompt_attention_mask")
        else:
            text_embeds = batch.get("text")
            text_mask = batch.get("text_mask")

        if text_embeds is None:
            raise ValueError(
                "Cached text embeddings missing from batch. Expected either batch['conditions'] (official format) "
                "or 'text'/'text_mask' (legacy musubi format)."
            )

        base_model = transformer.model if hasattr(transformer, "model") else transformer
        expected_video_dim = int(getattr(base_model, "cross_attention_dim", 0) or 0)
        expected_audio_dim = int(getattr(base_model, "audio_cross_attention_dim", 0) or 0)

        if self._ltx_mode == "video" and isinstance(text_embeds, torch.Tensor):
            video_source = text_embeds
            if conditions is not None:
                prompt_embeds = conditions.get("prompt_embeds")
                if isinstance(prompt_embeds, torch.Tensor):
                    video_source = prompt_embeds
            text_embeds = select_video_text_embeds_for_video_mode(
                video_source,
                expected_video_dim=expected_video_dim,
                expected_audio_dim=expected_audio_dim,
            )

        if self._ltx_mode == "audio" and isinstance(text_embeds, torch.Tensor):
            text_embeds = select_audio_text_embeds_for_audio_mode(
                text_embeds,
                conditions,
                expected_audio_dim=expected_audio_dim,
                expected_video_dim=expected_video_dim,
            )

        # LTX-2.3 (caption_proj_before_connector=True) expects already-projected context
        # dimensions for each modality. In audio mode this must be audio_prompt_embeds
        # (audio_cross_attention_dim), not generic/video prompt embeds.
        if self._ltx_mode == "audio" and bool(getattr(base_model, "caption_proj_before_connector", False)):
            if expected_audio_dim > 0 and text_embeds.shape[-1] != expected_audio_dim:
                raise ValueError(
                    "Audio mode received text embeddings with incompatible hidden size for this checkpoint. "
                    f"Expected audio_prompt_embeds dim={expected_audio_dim}, got dim={text_embeds.shape[-1]}. "
                    "This usually means text encoder cache was created without audio embeddings. "
                    "Re-run ltx2_cache_text_encoder_outputs.py with --ltx2_mode audio (or av) using the same "
                    "--ltx2_checkpoint, then train again."
                )

        if not isinstance(text_embeds, torch.Tensor):
            raise TypeError(f"Expected text embeddings to be a torch.Tensor, got: {type(text_embeds)}")
        if text_embeds.dim() != 3:
            raise ValueError(f"Expected text embeddings to be 3D [B, seq_len, hidden_dim], got shape: {tuple(text_embeds.shape)}")
        if text_embeds.shape[0] != latents.shape[0]:
            raise ValueError(f"Batch size mismatch: latents batch={latents.shape[0]} vs text batch={text_embeds.shape[0]}")

        text_embeds = text_embeds.to(device=accelerator.device, dtype=network_dtype)
        _log_stats("text_embeds", text_embeds)

        # Check for NaN values
        if torch.isnan(text_embeds).any():
            raise ValueError("NaN detected in cached text embeddings!")

        if text_mask is not None:
            if not isinstance(text_mask, torch.Tensor):
                raise TypeError(f"Expected text_mask to be a torch.Tensor, got: {type(text_mask)}")
            if text_mask.dim() != 2:
                raise ValueError(f"Expected text_mask to be 2D [B, seq_len], got shape: {tuple(text_mask.shape)}")
            if text_mask.shape[0] != latents.shape[0]:
                raise ValueError(f"Batch size mismatch: latents batch={latents.shape[0]} vs text_mask batch={text_mask.shape[0]}")
            if text_mask.shape[1] != text_embeds.shape[1]:
                raise ValueError(
                    f"text_mask seq_len ({text_mask.shape[1]}) does not match text_embeds seq_len ({text_embeds.shape[1]}). "
                    "This usually means the attention mask and text embedding caches were created with different "
                    "sequence length settings. Re-run the text encoder caching step."
                )
            text_mask = text_mask.to(device=accelerator.device)
            if args.gradient_checkpointing:
                text_mask = text_mask.to(torch.bool)

        # Caption dropout: zero out text conditioning with probability p (for CFG training)
        caption_dropout_rate = getattr(args, "caption_dropout_rate", 0.0)
        if caption_dropout_rate > 0.0 and getattr(self, "training", False):
            text_embeds, text_mask = self._apply_caption_dropout(text_embeds, text_mask, caption_dropout_rate)

        # Move latents to device
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)

        # Check for NaN in latents
        if torch.isnan(latents).any():
            raise ValueError("NaN detected in latents!")

        # Get frame rate from batch or use default
        frame_rate = batch.get("frame_rate", None)
        if frame_rate is None:
            latents_info = batch.get("latents")
            if isinstance(latents_info, dict):
                frame_rate = latents_info.get("fps", None)
        if frame_rate is None:
            frame_rate = 25
        if isinstance(frame_rate, torch.Tensor):
            frame_rate = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()

        model_timesteps = timesteps.to(device=accelerator.device, dtype=network_dtype)

        model_timesteps = self._normalize_timesteps_for_model(model_timesteps)

        if model_timesteps.dim() == 0:
            model_timesteps = model_timesteps.unsqueeze(0)
        if model_timesteps.dim() == 1:
            model_timesteps = model_timesteps.unsqueeze(1)

        sigma = model_timesteps[:, 0]
        audio_model_timesteps = model_timesteps
        if self._ltx_mode in {"av", "audio"} and model_timesteps.dim() == 2 and model_timesteps.shape[1] > 1:
            # Self-Flow token-wise video timesteps can have a different token length than audio.
            # Keep audio timesteps per-sample unless explicitly overridden below.
            audio_model_timesteps = model_timesteps[:, :1]
        if self._ltx_mode in {"av", "audio"} and bool(getattr(args, "independent_audio_timestep", False)):
            audio_model_timesteps = self._sample_independent_audio_timesteps(
                args,
                batch_size=model_timesteps.shape[0],
                device=accelerator.device,
                dtype=network_dtype,
            )
        audio_sigma = audio_model_timesteps[:, 0]
        ic_lora_strategy = str(
            getattr(
                args,
                "ic_lora_strategy",
                self._ic_lora_strategy
                or infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v")),
            )
            or "none"
        ).lower()
        audio_ref_only_ic_enabled = ic_lora_strategy == "audio_ref_only_ic"

        ref_latents = batch.get("ref_latents")
        if isinstance(ref_latents, dict):
            ref_latents = ref_latents.get("latents")

        if ref_latents is not None:
            if ic_lora_strategy != "v2v":
                if not self._warned_ignored_ref_latents:
                    logger.warning(
                        "ref_latents were provided but --ic_lora_strategy is '%s'; ignoring reference-video conditioning.",
                        ic_lora_strategy,
                    )
                    self._warned_ignored_ref_latents = True
                ref_latents = None
            else:
                if self._audio_video or self._ltx_mode != "video":
                    raise ValueError("Reference latent conditioning is only supported for video-only LTX-2 training")
                if not isinstance(ref_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_latents to be a torch.Tensor, got: {type(ref_latents)}")
                if ref_latents.dim() != 5:
                    raise ValueError(f"Expected ref_latents to be 5D [B, C, F, H, W], got shape: {tuple(ref_latents.shape)}")
                if ref_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_latents batch={ref_latents.shape[0]}"
                    )
                if ref_latents.shape[1] != latents.shape[1]:
                    raise ValueError(f"Channel mismatch: latents C={latents.shape[1]} vs ref_latents C={ref_latents.shape[1]}")
                ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
                tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
                if ref_h == tgt_h and ref_w == tgt_w:
                    reference_downscale_factor = 1
                else:
                    h_ratio = tgt_h / ref_h
                    w_ratio = tgt_w / ref_w
                    if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                        raise ValueError(
                            f"Spatial mismatch: latents HxW={tgt_h}x{tgt_w} vs ref_latents HxW={ref_h}x{ref_w}. "
                            f"Ratios h={h_ratio:.2f} w={w_ratio:.2f} are not consistent integer downscale factors."
                        )
                    reference_downscale_factor = round(h_ratio)

        if self._ltx_mode == "audio":
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")
            if audio_latents is None:
                raise ValueError("audio_latents are required for --ltx_mode audio")
            if not isinstance(audio_latents, torch.Tensor):
                raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
            if audio_latents.dim() != 4:
                raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got shape: {tuple(audio_latents.shape)}")
            if audio_latents.shape[0] != latents.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: latents batch={latents.shape[0]} vs audio_latents batch={audio_latents.shape[0]}"
                )

            audio_latents = audio_latents.to(device=accelerator.device, dtype=network_dtype)
            audio_noise = torch.randn_like(audio_latents)
            sigma_audio = audio_sigma.view(-1, 1, 1, 1)
            noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

            # Compute target and loss mask BEFORE IC block so they can be concatenated with ref tokens.
            audio_target = audio_noise - audio_latents
            audio_seq_len = int(audio_latents.shape[2])
            audio_loss_mask = torch.ones(
                (audio_latents.shape[0], audio_seq_len),
                device=accelerator.device,
                dtype=torch.bool,
            )

            # Audio-only mode always masks padding to prevent loss on zero-padded
            # positions that arise from batching variable-length audio clips.
            audio_lengths = batch.get("audio_lengths")
            if isinstance(audio_lengths, dict):
                audio_lengths = audio_lengths.get("lengths")
            if isinstance(audio_lengths, torch.Tensor):
                if audio_lengths.dim() == 0:
                    audio_lengths = audio_lengths.view(1)
                if audio_lengths.dim() != 1:
                    raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                if audio_lengths.numel() == 1 and audio_latents.shape[0] != 1:
                    audio_lengths = audio_lengths.expand(audio_latents.shape[0])
                if audio_lengths.shape[0] != audio_latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: audio_latents batch={audio_latents.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                    )

                audio_lengths = audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                audio_lengths = audio_lengths.clamp(min=0, max=audio_seq_len)
                t = torch.arange(audio_seq_len, device=accelerator.device).view(1, -1)
                audio_loss_mask = t < audio_lengths.view(-1, 1)

            video_latents = torch.zeros(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=accelerator.device,
                dtype=network_dtype,
            )

            audio_timestep_local = audio_model_timesteps
            resolved_transformer_options: Dict[str, Any] = {"patches_replace": {}}
            ref_audio_seq_len = 0

            if audio_ref_only_ic_enabled:
                ref_audio_latents = batch.get("ref_audio_latents")
                if isinstance(ref_audio_latents, dict):
                    ref_audio_latents = ref_audio_latents.get("latents")
                if ref_audio_latents is None:
                    raise ValueError(
                        "--ic_lora_strategy audio_ref_only_ic requires ref_audio_latents. "
                        "Set reference_audio_directory/reference_audio_cache_directory and cache reference audio latents."
                    )
                if not isinstance(ref_audio_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(ref_audio_latents)}")
                if ref_audio_latents.dim() != 4:
                    raise ValueError(
                        f"Expected ref_audio_latents to be 4D [B, C, T, F], got shape: {tuple(ref_audio_latents.shape)}"
                    )
                if ref_audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                    )
                if ref_audio_latents.shape[1] != audio_latents.shape[1] or ref_audio_latents.shape[3] != audio_latents.shape[3]:
                    raise ValueError(
                        "ref_audio_latents channel/mel dimensions must match audio_latents. "
                        f"Got ref={tuple(ref_audio_latents.shape)} target={tuple(audio_latents.shape)}"
                    )

                ref_audio_latents = ref_audio_latents.to(device=accelerator.device, dtype=network_dtype)

                ref_audio_lengths = batch.get("ref_audio_lengths")
                if isinstance(ref_audio_lengths, dict):
                    ref_audio_lengths = ref_audio_lengths.get("lengths")
                if isinstance(ref_audio_lengths, torch.Tensor):
                    if ref_audio_lengths.dim() == 0:
                        ref_audio_lengths = ref_audio_lengths.view(1)
                    if ref_audio_lengths.numel() == 1 and ref_audio_latents.shape[0] != 1:
                        ref_audio_lengths = ref_audio_lengths.expand(ref_audio_latents.shape[0])
                    if ref_audio_lengths.shape[0] != ref_audio_latents.shape[0]:
                        raise ValueError(
                            "Batch size mismatch: ref_audio_lengths batch="
                            f"{ref_audio_lengths.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                        )
                    ref_audio_lengths = ref_audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                    if (ref_audio_lengths <= 0).any():
                        raise ValueError(
                            "ref_audio_lengths contains zeros; missing reference-audio caches in batch. "
                            "Ensure every training sample has cached reference audio."
                        )

                ref_audio_seq_len = int(ref_audio_latents.shape[2])
                tgt_seq_len = int(audio_latents.shape[2])
                noisy_audio = torch.cat([ref_audio_latents, noisy_audio], dim=2)

                target_audio_timestep = (
                    audio_model_timesteps
                    if audio_model_timesteps.shape[1] == tgt_seq_len
                    else audio_model_timesteps[:, :1].expand(audio_model_timesteps.shape[0], tgt_seq_len)
                )
                ref_audio_timestep = torch.zeros(
                    (audio_model_timesteps.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=network_dtype,
                )
                audio_timestep_local = torch.cat([ref_audio_timestep, target_audio_timestep], dim=1)

                zero_ref_target = torch.zeros_like(ref_audio_latents)
                audio_target = torch.cat([zero_ref_target, audio_target], dim=2)

                ref_audio_loss_mask = torch.zeros(
                    (audio_latents.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=torch.bool,
                )
                audio_loss_mask = torch.cat([ref_audio_loss_mask, audio_loss_mask], dim=1)

                resolved_transformer_options = dict(resolved_transformer_options)
                resolved_transformer_options.update(
                    self._build_audio_ref_transformer_overrides(
                        args=args,
                        transformer=transformer,
                        video_latents=video_latents,
                        text_embeds=text_embeds,
                        text_mask=text_mask,
                        audio_model_latents=noisy_audio,
                        ref_audio_seq_len=ref_audio_seq_len,
                        device=accelerator.device,
                        dtype=network_dtype,
                    )
                )

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(transformer)
            elif getattr(args, "nf4_base", False):
                self._ensure_nf4_buffers_on_device(transformer)
            with accelerator.autocast():
                model_pred = transformer(
                    [video_latents, noisy_audio],
                    timestep=model_timesteps,
                    audio_timestep=audio_timestep_local,
                    context=text_embeds,
                    attention_mask=text_mask,
                    frame_rate=frame_rate,
                    transformer_options=resolved_transformer_options,
                    audio_only=True,
                )

            video_pred = model_pred
            audio_pred = None
            if isinstance(model_pred, (list, tuple)):
                if len(model_pred) != 2:
                    raise ValueError(f"Expected audio-only model to return [video_pred, audio_pred], got {len(model_pred)} outputs")
                video_pred, audio_pred = model_pred
            if audio_pred is None:
                raise ValueError("Audio-only mode expected an audio prediction but got None")

            video_target = torch.zeros_like(video_pred)
            out_audio: Dict[str, Any] = {
                "video_pred": video_pred,
                "video_target": video_target,
                "video_loss_weight": 0.0,
            }

            out_audio.update(
                {
                    "audio_pred": audio_pred,
                    "audio_target": audio_target,
                    "audio_loss_mask": audio_loss_mask,
                    "audio_loss_weight": _resolve_loss_weight("audio_loss_weight", "audio_loss_weight"),
                }
            )
            if out_audio["audio_loss_weight"] < 0.0:
                raise ValueError(f"audio_loss_weight must be >= 0. Got: {out_audio['audio_loss_weight']}")

            self._last_dit_inputs = None  # audio-only path — skip preservation
            return out_audio, torch.tensor(0.0, device=accelerator.device)

        first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
        if not (0.0 <= first_frame_p <= 1.0):
            raise ValueError(f"ltx2_first_frame_conditioning_p must be in [0,1]. Got: {first_frame_p}")

        video_conditioning_enabled = None
        # Skip first-frame conditioning for single-frame samples (images)
        # since there are no subsequent frames to generate from frame 0
        num_frames = latents.shape[2]
        if first_frame_p > 0.0 and num_frames > 1:
            enable_conditioning = bool(torch.rand((), device=accelerator.device) < first_frame_p)
            if enable_conditioning:
                video_conditioning_enabled = torch.ones((latents.shape[0],), device=accelerator.device, dtype=torch.bool)

        model_noisy_video = noisy_model_input
        if video_conditioning_enabled is not None and model_noisy_video.shape[2] > 0:
            model_noisy_video = model_noisy_video.clone()
            model_noisy_video[video_conditioning_enabled, :, 0:1, :, :] = latents[video_conditioning_enabled, :, 0:1, :, :]

        if ref_latents is not None:
            from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
            from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
            from musubi_tuner.ltx_2.model.transformer.modality import Modality
            from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape

            patchifier = VideoLatentPatchifier(patch_size=1)

            ref_latents = ref_latents.to(device=accelerator.device, dtype=network_dtype)
            ref_tokens = patchifier.patchify(ref_latents)
            target_tokens = patchifier.patchify(model_noisy_video)
            combined_tokens = torch.cat([ref_tokens, target_tokens], dim=1)

            bsz = combined_tokens.shape[0]
            ref_seq_len = ref_tokens.shape[1]
            target_seq_len = target_tokens.shape[1]

            ref_height = int(ref_latents.shape[3])
            ref_width = int(ref_latents.shape[4])
            tgt_height = int(latents.shape[3])
            tgt_width = int(latents.shape[4])

            ref_conditioning_mask = torch.ones((bsz, ref_seq_len), device=accelerator.device, dtype=torch.bool)

            target_conditioning_mask = torch.zeros((bsz, target_seq_len), device=accelerator.device, dtype=torch.bool)
            if video_conditioning_enabled is not None:
                first_frame_tokens = tgt_height * tgt_width
                if first_frame_tokens > 0:
                    target_conditioning_mask[video_conditioning_enabled, :first_frame_tokens] = True
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

            combined_timesteps = sigma.view(bsz, 1).expand(bsz, ref_seq_len + target_seq_len)
            combined_timesteps = torch.where(conditioning_mask, torch.zeros_like(combined_timesteps), combined_timesteps)

            frame_rate_v2v = frame_rate
            if frame_rate_v2v is None:
                frame_rate_v2v = 25

            ref_frames = int(ref_latents.shape[2])
            tgt_frames = int(latents.shape[2])

            ref_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(ref_latents.shape[1]),
                    frames=ref_frames,
                    height=ref_height,
                    width=ref_width,
                ),
                device=accelerator.device,
            )
            ref_positions = get_pixel_coords(
                latent_coords=ref_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / float(frame_rate_v2v)
            if reference_downscale_factor != 1:
                ref_positions = ref_positions.clone()
                ref_positions[:, 1, ...] *= reference_downscale_factor
                ref_positions[:, 2, ...] *= reference_downscale_factor

            tgt_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(latents.shape[1]),
                    frames=tgt_frames,
                    height=tgt_height,
                    width=tgt_width,
                ),
                device=accelerator.device,
            )
            tgt_positions = get_pixel_coords(
                latent_coords=tgt_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            tgt_positions[:, 0, ...] = tgt_positions[:, 0, ...] / float(frame_rate_v2v)

            combined_positions = torch.cat([ref_positions, tgt_positions], dim=2)

            video_modality = Modality(
                enabled=True,
                latent=combined_tokens,
                timesteps=combined_timesteps,
                positions=combined_positions,
                context=text_embeds,
                sigma=sigma,
                context_mask=text_mask,
            )

            perturbations = BatchedPerturbationConfig.empty(bsz)
            unwrapped_transformer = accelerator.unwrap_model(transformer)

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(unwrapped_transformer)
            elif getattr(args, "nf4_base", False):
                self._ensure_nf4_buffers_on_device(unwrapped_transformer)
            with accelerator.autocast():
                if hasattr(unwrapped_transformer, "forward_modalities"):
                    pred_tokens, _ = unwrapped_transformer.forward_modalities(video_modality, None, perturbations)
                else:
                    base_model = (
                        unwrapped_transformer.model if hasattr(unwrapped_transformer, "model") else unwrapped_transformer
                    )
                    pred_tokens, _ = base_model(video_modality, None, perturbations)

            target_pred_tokens = pred_tokens[:, ref_seq_len:, :]
            target_velocity = patchifier.patchify(noise - latents)
            target_loss_mask = ~target_conditioning_mask

            out_v2v: Dict[str, Any] = {
                "video_pred": target_pred_tokens,
                "video_target": target_velocity,
                "video_loss_mask": target_loss_mask,
                "video_loss_weight": _resolve_loss_weight("video_loss_weight", "video_loss_weight"),
            }
            if out_v2v["video_loss_weight"] < 0.0:
                raise ValueError(f"video_loss_weight must be >= 0. Got: {out_v2v['video_loss_weight']}")

            self._last_dit_inputs = None  # reference-latent path — skip preservation
            return out_v2v, torch.tensor(0.0, device=accelerator.device)

        audio_latents = None
        audio_noise = None
        noisy_audio = None
        audio_enabled_for_batch = False
        audio_regularizer_active = False
        audio_expected_for_batch = self._ltx_mode == "av"
        audio_loss_mask = None
        audio_target = None
        audio_timestep_for_model = None
        ref_audio_latents = None
        ref_audio_seq_len = 0
        if self._ltx_mode == "av":
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")

            if audio_latents is None:
                if bool(getattr(args, "audio_silence_regularizer", False)):
                    audio_latents = self._build_empty_audio_latents(
                        args=args,
                        transformer=transformer,
                        latents=latents,
                        frame_rate=float(frame_rate),
                        device=accelerator.device,
                        dtype=network_dtype,
                    )
                    audio_regularizer_active = True
                    if not self._warned_missing_audio:
                        logger.warning(
                            "LTXAV mode: missing audio latents in this batch; using silence regularizer fallback."
                        )
                        self._warned_missing_audio = True
                else:
                    if not self._warned_missing_audio:
                        logger.warning(
                            "LTXAV mode: missing audio latents in this batch; skipping audio branch. "
                            "Provide cached audio latents to train audio generation."
                        )
                        self._warned_missing_audio = True
            if audio_latents is not None:
                if not isinstance(audio_latents, torch.Tensor):
                    raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
                if audio_latents.dim() != 4:
                    raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got shape: {tuple(audio_latents.shape)}")
                if audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs audio_latents batch={audio_latents.shape[0]}"
                    )
                audio_latents = audio_latents.to(device=accelerator.device, dtype=network_dtype)
                _check_finite("audio_latents", audio_latents)
                _log_stats("audio_latents", audio_latents)
                if getattr(args, "align_audio_latents_train", False):
                    expected_length = self._calculate_expected_audio_latent_length(
                        args,
                        transformer,
                        latent_frames=int(latents.shape[2]),
                        frame_rate=float(frame_rate),
                    )
                    audio_latents = self._adjust_audio_latent_duration(audio_latents, expected_length)
                audio_loss_mask = torch.ones(
                    (audio_latents.shape[0], audio_latents.shape[2]),
                    device=accelerator.device,
                    dtype=torch.bool,
                )

                audio_enabled_for_batch = True
                audio_noise = torch.randn_like(audio_latents)
                sigma_audio = audio_sigma.view(-1, 1, 1, 1)
                noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise
                _check_finite("noisy_audio", noisy_audio)
                _log_stats("noisy_audio", noisy_audio)
                audio_target = audio_noise - audio_latents
                audio_timestep_for_model = audio_model_timesteps

            if audio_ref_only_ic_enabled:
                ref_audio_latents = batch.get("ref_audio_latents")
                if isinstance(ref_audio_latents, dict):
                    ref_audio_latents = ref_audio_latents.get("latents")

                if not audio_enabled_for_batch or audio_latents is None or noisy_audio is None:
                    raise ValueError(
                        "--ic_lora_strategy audio_ref_only_ic requires target audio_latents in every AV batch"
                    )
                if ref_audio_latents is None:
                    raise ValueError(
                        "--ic_lora_strategy audio_ref_only_ic requires ref_audio_latents. "
                        "Set reference_audio_directory/reference_audio_cache_directory and cache reference audio latents."
                    )
                if not isinstance(ref_audio_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(ref_audio_latents)}")
                if ref_audio_latents.dim() != 4:
                    raise ValueError(
                        f"Expected ref_audio_latents to be 4D [B, C, T, F], got shape: {tuple(ref_audio_latents.shape)}"
                    )
                if ref_audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                    )
                if ref_audio_latents.shape[1] != audio_latents.shape[1] or ref_audio_latents.shape[3] != audio_latents.shape[3]:
                    raise ValueError(
                        "ref_audio_latents channel/mel dimensions must match audio_latents. "
                        f"Got ref={tuple(ref_audio_latents.shape)} target={tuple(audio_latents.shape)}"
                    )

                ref_audio_latents = ref_audio_latents.to(device=accelerator.device, dtype=network_dtype)
                _check_finite("ref_audio_latents", ref_audio_latents)
                _log_stats("ref_audio_latents", ref_audio_latents)

                ref_audio_lengths = batch.get("ref_audio_lengths")
                if isinstance(ref_audio_lengths, dict):
                    ref_audio_lengths = ref_audio_lengths.get("lengths")
                if isinstance(ref_audio_lengths, torch.Tensor):
                    if ref_audio_lengths.dim() == 0:
                        ref_audio_lengths = ref_audio_lengths.view(1)
                    if ref_audio_lengths.numel() == 1 and ref_audio_latents.shape[0] != 1:
                        ref_audio_lengths = ref_audio_lengths.expand(ref_audio_latents.shape[0])
                    if ref_audio_lengths.shape[0] != ref_audio_latents.shape[0]:
                        raise ValueError(
                            "Batch size mismatch: ref_audio_lengths batch="
                            f"{ref_audio_lengths.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                        )
                    ref_audio_lengths = ref_audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                    if (ref_audio_lengths <= 0).any():
                        raise ValueError(
                            "ref_audio_lengths contains zeros; missing reference-audio caches in batch. "
                            "Ensure every training sample has cached reference audio."
                        )

                ref_audio_seq_len = int(ref_audio_latents.shape[2])
                tgt_seq_len = int(audio_latents.shape[2])
                noisy_audio = torch.cat([ref_audio_latents, noisy_audio], dim=2)
                _check_finite("noisy_audio_with_reference", noisy_audio)

                target_audio_timestep = (
                    audio_model_timesteps
                    if audio_model_timesteps.shape[1] == tgt_seq_len
                    else audio_model_timesteps[:, :1].expand(audio_model_timesteps.shape[0], tgt_seq_len)
                )
                ref_audio_timestep = torch.zeros(
                    (audio_model_timesteps.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=network_dtype,
                )
                audio_timestep_for_model = torch.cat([ref_audio_timestep, target_audio_timestep], dim=1)

                if audio_target is None:
                    raise ValueError("Internal error: audio_target must be initialized before audio_ref_only_ic composition")
                zero_ref_target = torch.zeros_like(ref_audio_latents)
                audio_target = torch.cat([zero_ref_target, audio_target], dim=2)

                target_audio_loss_mask = audio_loss_mask
                if target_audio_loss_mask is None:
                    target_audio_loss_mask = torch.ones(
                        (audio_latents.shape[0], tgt_seq_len),
                        device=accelerator.device,
                        dtype=torch.bool,
                    )
                if getattr(args, "use_audio_length_mask", False):
                    audio_lengths = batch.get("audio_lengths")
                    if isinstance(audio_lengths, dict):
                        audio_lengths = audio_lengths.get("lengths")
                    if isinstance(audio_lengths, torch.Tensor):
                        if audio_lengths.dim() == 0:
                            audio_lengths = audio_lengths.view(1)
                        if audio_lengths.numel() == 1 and audio_latents.shape[0] != 1:
                            audio_lengths = audio_lengths.expand(audio_latents.shape[0])
                        if audio_lengths.shape[0] != audio_latents.shape[0]:
                            raise ValueError(
                                f"Batch size mismatch: audio_latents batch={audio_latents.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                            )
                        audio_lengths = audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                        audio_lengths = audio_lengths.clamp(min=0, max=tgt_seq_len)
                        t = torch.arange(tgt_seq_len, device=accelerator.device).view(1, -1)
                        target_audio_loss_mask = t < audio_lengths.view(-1, 1)
                ref_audio_loss_mask = torch.zeros(
                    (audio_latents.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=torch.bool,
                )
                audio_loss_mask = torch.cat([ref_audio_loss_mask, target_audio_loss_mask], dim=1)

        if self._ltx_mode == "av" and not audio_enabled_for_batch:
            text_embeds = select_video_text_embeds_for_av_no_audio(
                text_embeds,
                conditions,
                expected_video_dim=expected_video_dim,
                expected_audio_dim=expected_audio_dim,
            )

        if bool(getattr(transformer, "training", False)) and self._ltx_mode == "av":
            supervision_alert = update_and_check_audio_supervision(
                self._audio_supervision_state,
                mode=str(getattr(args, "audio_supervision_mode", "off")),
                warmup_steps=int(getattr(args, "audio_supervision_warmup_steps", 50)),
                check_interval=int(getattr(args, "audio_supervision_check_interval", 50)),
                min_ratio=float(getattr(args, "audio_supervision_min_ratio", 0.9)),
                audio_expected_for_batch=audio_expected_for_batch,
                audio_supervised_for_batch=audio_enabled_for_batch and not audio_regularizer_active,
            )
            if supervision_alert is not None:
                message = format_audio_supervision_alert(supervision_alert)
                if str(getattr(args, "audio_supervision_mode", "off")) == "error":
                    raise ValueError(message)
                logger.warning("%s Running in warning mode; training will continue.", message)

        if skip_nonfinite and nonfinite_flag["hit"]:
            return {"_skip_step": True, "skip_reason": nonfinite_flag["tag"]}, torch.tensor(
                0.0, device=accelerator.device
            )

        caption_channels = getattr(transformer, "caption_channels", None)
        if caption_channels is None:
            base_model = transformer.model if hasattr(transformer, "model") else transformer
            _caption_proj = getattr(base_model, "caption_projection", None)
            if _caption_proj is not None:
                caption_channels = getattr(getattr(_caption_proj, "linear_1", None), "in_features", None)
        if caption_channels is not None:
            expected_last_dim = int(caption_channels) * (2 if audio_enabled_for_batch else 1)
            if text_embeds.shape[-1] != expected_last_dim:
                if (
                    self._ltx_mode == "av"
                    and audio_enabled_for_batch
                    and audio_regularizer_active
                    and text_embeds.shape[-1] * 2 == expected_last_dim
                ):
                    text_embeds = torch.cat([text_embeds, torch.zeros_like(text_embeds)], dim=-1)
                    expected_last_dim = text_embeds.shape[-1]
                else:
                    raise ValueError(
                        f"Text embedding dim mismatch for {'LTXAV' if self._audio_video else 'LTXV'}: "
                        f"got {text_embeds.shape[-1]}, expected {expected_last_dim}. "
                        f"(caption_channels={caption_channels})"
                    )
            if text_embeds.shape[-1] != expected_last_dim:
                raise ValueError(
                    f"Text embedding dim mismatch for {'LTXAV' if self._audio_video else 'LTXV'}: "
                    f"got {text_embeds.shape[-1]}, expected {expected_last_dim}. "
                    f"(caption_channels={caption_channels})"
                )

        if self._ltx_mode == "av" and bool(getattr(base_model, "caption_proj_before_connector", False)):
            expected_ctx_dim = expected_video_dim + expected_audio_dim if audio_enabled_for_batch else expected_video_dim
            if expected_ctx_dim > 0 and int(text_embeds.shape[-1]) != expected_ctx_dim:
                mode_name = "AV (video+audio)" if audio_enabled_for_batch else "AV-no-audio (video-only)"
                raise ValueError(
                    f"{mode_name} received text embeddings with incompatible hidden size for this checkpoint. "
                    f"Expected dim={expected_ctx_dim}, got dim={text_embeds.shape[-1]}. "
                    "Ensure caches contain modality-specific embeddings generated with the same --ltx2_checkpoint."
                )

        if self._ltx_mode == "video" and bool(getattr(base_model, "caption_proj_before_connector", False)):
            if expected_video_dim > 0 and int(text_embeds.shape[-1]) != expected_video_dim:
                raise ValueError(
                    f"Video mode received text embeddings with incompatible hidden size for this checkpoint. "
                    f"Expected dim={expected_video_dim}, got dim={text_embeds.shape[-1]}. "
                    "Ensure text encoder caches were generated with the same --ltx2_checkpoint."
                )

        model_input = model_noisy_video
        if self._ltx_mode == "av" and audio_enabled_for_batch:
            model_input = [model_noisy_video, noisy_audio]
        _log_stats("noisy_video", model_noisy_video)
        _log_stats("timesteps", timesteps)

        video_conditioning_mask_tokens = None
        video_loss_mask = None
        if video_conditioning_enabled is not None:
            bsz, _c, frames, height, width = latents.shape
            seq_len = frames * height * width
            first_frame_tokens = height * width
            video_conditioning_mask_tokens = torch.zeros((bsz, seq_len), device=accelerator.device, dtype=torch.bool)
            if first_frame_tokens > 0:
                video_conditioning_mask_tokens[video_conditioning_enabled, :first_frame_tokens] = True
            transformer_options = {"patches_replace": {}, "video_conditioning_mask": video_conditioning_mask_tokens}

            if getattr(args, "video_loss_mask_5d", False):
                video_loss_mask = torch.ones((bsz, 1, frames, 1, 1), device=accelerator.device, dtype=torch.bool)
                if frames > 0:
                    video_loss_mask[video_conditioning_enabled, :, 0:1, :, :] = False
            else:
                video_loss_mask = torch.ones((bsz, frames), device=accelerator.device, dtype=torch.bool)
                if frames > 0:
                    video_loss_mask[video_conditioning_enabled, 0] = False

        resolved_transformer_options = transformer_options if video_conditioning_mask_tokens is not None else {"patches_replace": {}}
        if (
            self._ltx_mode == "av"
            and audio_ref_only_ic_enabled
            and audio_enabled_for_batch
            and noisy_audio is not None
            and ref_audio_seq_len > 0
        ):
            resolved_transformer_options = dict(resolved_transformer_options)
            resolved_transformer_options.update(
                self._build_audio_ref_transformer_overrides(
                    args=args,
                    transformer=transformer,
                    video_latents=model_noisy_video,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    audio_model_latents=noisy_audio,
                    ref_audio_seq_len=ref_audio_seq_len,
                    device=accelerator.device,
                    dtype=network_dtype,
                )
            )

        # Store inputs for preservation / Self-Flow techniques (no-op when both are off)
        if self._preservation_active or self._self_flow_active:
            self._last_dit_inputs = {
                "model_input": model_input,
                "model_timesteps": model_timesteps,
                "audio_model_timesteps": audio_timestep_for_model if audio_enabled_for_batch else None,
                "text_embeds": text_embeds,
                "text_mask": text_mask,
                "frame_rate": frame_rate,
                "transformer_options": resolved_transformer_options,
            }

        if self._self_flow_active and self._self_flow is not None:
            self._self_flow.cleanup_step()
            network_for_self_flow = getattr(self, "_self_flow_network", None)
            is_train_step = bool(getattr(network_for_self_flow, "training", False)) if network_for_self_flow is not None else bool(
                getattr(transformer, "training", False)
            )
            if is_train_step and bool(getattr(args, "self_flow", False)):
                sf_ctx = self._self_flow_step_context
                if sf_ctx is not None:
                    teacher_noisy = sf_ctx.get("teacher_noisy_model_input")
                    teacher_timesteps = sf_ctx.get("teacher_model_timesteps")
                    if isinstance(teacher_noisy, torch.Tensor) and isinstance(teacher_timesteps, torch.Tensor):
                        teacher_noisy_input = teacher_noisy
                        if video_conditioning_enabled is not None and teacher_noisy_input.shape[2] > 0:
                            teacher_noisy_input = teacher_noisy_input.clone()
                            teacher_noisy_input[video_conditioning_enabled, :, 0:1, :, :] = latents[
                                video_conditioning_enabled, :, 0:1, :, :
                            ]

                        teacher_model_input_for_self_flow: Any = teacher_noisy_input.to(
                            device=accelerator.device, dtype=network_dtype
                        )
                        teacher_audio_timestep = None
                        if (
                            isinstance(model_input, (list, tuple))
                            and len(model_input) >= 2
                            and isinstance(model_input[1], torch.Tensor)
                        ):
                            teacher_audio_input = model_input[1].to(device=accelerator.device, dtype=network_dtype)
                            teacher_model_input_for_self_flow = [teacher_model_input_for_self_flow, teacher_audio_input]
                            if isinstance(audio_timestep_for_model, torch.Tensor):
                                teacher_audio_timestep = audio_timestep_for_model.to(
                                    device=accelerator.device, dtype=network_dtype
                                )

                        teacher_timesteps_model = self._normalize_timesteps_for_model(
                            teacher_timesteps.to(device=accelerator.device, dtype=network_dtype)
                        )
                        if teacher_timesteps_model.dim() == 0:
                            teacher_timesteps_model = teacher_timesteps_model.unsqueeze(0)
                        if teacher_timesteps_model.dim() == 1:
                            teacher_timesteps_model = teacher_timesteps_model.unsqueeze(1)

                        if network_for_self_flow is not None:
                            self._self_flow.prepare_teacher_features(
                                accelerator=accelerator,
                                transformer=transformer,
                                network=network_for_self_flow,
                                teacher_model_input=teacher_model_input_for_self_flow,
                                teacher_timesteps=teacher_timesteps_model,
                                audio_timestep=teacher_audio_timestep,
                                text_embeds=text_embeds,
                                text_mask=text_mask,
                                frame_rate=frame_rate,
                                transformer_options=resolved_transformer_options,
                            )
                self._self_flow.mark_student_forward()

        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            self._ensure_fp8_buffers_on_device(transformer)
        with accelerator.autocast():
            model_pred = transformer(
                model_input,
                timestep=model_timesteps,
                audio_timestep=audio_timestep_for_model if audio_enabled_for_batch else None,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options=resolved_transformer_options,
            )

        video_pred = model_pred
        audio_pred = None
        if isinstance(model_pred, (list, tuple)):
            if len(model_pred) != 2:
                raise ValueError(f"Expected AV model to return [video_pred, audio_pred], got {len(model_pred)} outputs")
            video_pred, audio_pred = model_pred
        _check_finite("video_pred", video_pred)
        _check_finite("audio_pred", audio_pred)
        _log_stats("video_pred", video_pred)
        _log_stats("audio_pred", audio_pred)

        if skip_nonfinite and nonfinite_flag["hit"]:
            return {"_skip_step": True, "skip_reason": nonfinite_flag["tag"]}, torch.tensor(
                0.0, device=accelerator.device
            )

        video_target = noise - latents
        _check_finite("video_target", video_target)
        _log_stats("video_target", video_target)

        out: Dict[str, Any] = {
            "video_pred": video_pred,
            "video_target": video_target,
            "video_loss_mask": video_loss_mask,
            "video_loss_weight": _resolve_loss_weight("video_loss_weight", "video_loss_weight"),
        }

        if out["video_loss_weight"] < 0.0:
            raise ValueError(f"video_loss_weight must be >= 0. Got: {out['video_loss_weight']}")

        if self._ltx_mode == "av" and audio_enabled_for_batch:
            if audio_pred is None:
                raise ValueError("AV mode expected an audio prediction but got None")
            if audio_target is None:
                raise ValueError("Internal error: audio_target is missing in AV path")
            _check_finite("audio_target", audio_target)
            _log_stats("audio_target", audio_target)

            audio_seq_len = int(audio_target.shape[2])
            if audio_loss_mask is None:
                audio_loss_mask = torch.ones(
                    (audio_target.shape[0], audio_seq_len),
                    device=accelerator.device,
                    dtype=torch.bool,
                )

            if getattr(args, "use_audio_length_mask", False) and not audio_ref_only_ic_enabled:
                audio_lengths = batch.get("audio_lengths")
                if isinstance(audio_lengths, dict):
                    audio_lengths = audio_lengths.get("lengths")
                if isinstance(audio_lengths, torch.Tensor):
                    if audio_lengths.dim() == 0:
                        audio_lengths = audio_lengths.view(1)
                    if audio_lengths.dim() != 1:
                        raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                    if audio_lengths.numel() == 1 and audio_target.shape[0] != 1:
                        audio_lengths = audio_lengths.expand(audio_target.shape[0])
                    if audio_lengths.shape[0] != audio_target.shape[0]:
                        raise ValueError(
                            f"Batch size mismatch: audio_target batch={audio_target.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                        )

                    audio_lengths = audio_lengths.to(device=accelerator.device)
                    if audio_lengths.dtype.is_floating_point:
                        audio_lengths = audio_lengths.to(dtype=torch.int64)
                    else:
                        audio_lengths = audio_lengths.to(dtype=torch.int64)

                    audio_lengths = audio_lengths.clamp(min=0, max=audio_seq_len)
                    t = torch.arange(audio_seq_len, device=accelerator.device).view(1, -1)
                    audio_loss_mask = t < audio_lengths.view(-1, 1)
            out.update(
                {
                    "audio_pred": audio_pred,
                    "audio_target": audio_target,
                    "audio_loss_mask": audio_loss_mask,
                    "audio_loss_weight": _resolve_loss_weight("audio_loss_weight", "audio_loss_weight")
                    * (
                        float(getattr(args, "audio_silence_regularizer_weight", 1.0))
                        if audio_regularizer_active
                        else 1.0
                    ),
                }
            )
            if out["audio_loss_weight"] < 0.0:
                raise ValueError(f"audio_loss_weight must be >= 0. Got: {out['audio_loss_weight']}")

        if diag_enabled:
            item_keys = batch.get("item_keys")
            if isinstance(item_keys, list) and item_keys:
                logger.info("DIAG item_keys: %s", item_keys[:5])
            latent_paths = batch.get("latent_cache_paths")
            if isinstance(latent_paths, list) and latent_paths:
                logger.info("DIAG latent_cache_paths: %s", latent_paths[:3])

        return out, torch.tensor(0.0, device=accelerator.device)

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale and shift latents for training (optional normalization)"""
        # LTX-2 typically doesn't require normalization, but can be enabled if needed
        return latents

    def _apply_sample_defaults(self, args: argparse.Namespace, prompts: List[Dict]) -> List[Dict]:
        default_height = int(getattr(args, "height", 512))
        default_width = int(getattr(args, "width", 768))
        default_frame_count = int(getattr(args, "sample_num_frames", 45))
        default_guidance_scale = float(getattr(args, "guidance_scale", self.default_guidance_scale))
        default_discrete_flow_shift = getattr(args, "discrete_flow_shift", None)

        sample_parameters = []
        for prompt_data in prompts:
            prompt_text = prompt_data.get("prompt", "")
            param = prompt_data.copy()
            param.setdefault("prompt", prompt_text)
            param.setdefault("negative_prompt", prompt_data.get("negative_prompt", ""))
            if "frame_count" not in param and "num_frames" in param:
                param["frame_count"] = param["num_frames"]
            param.setdefault("height", prompt_data.get("height", default_height))
            param.setdefault("width", prompt_data.get("width", default_width))
            param.setdefault("frame_count", prompt_data.get("frame_count", default_frame_count))
            param.setdefault("sample_steps", prompt_data.get("sample_steps", 20))
            param.setdefault("guidance_scale", prompt_data.get("guidance_scale", default_guidance_scale))
            if default_discrete_flow_shift is not None:
                param.setdefault("discrete_flow_shift", prompt_data.get("discrete_flow_shift", default_discrete_flow_shift))
            param.setdefault("seed", prompt_data.get("seed", 0))
            sample_parameters.append(param)

        return sample_parameters

    def _load_precached_sample_prompts(self, args: argparse.Namespace) -> List[Dict]:
        cache_path = getattr(args, "sample_prompts_cache", None) or self._resolve_default_sample_prompts_cache(args)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Precached sample prompt embeddings not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid sample prompt cache format: {cache_path}")
        cached_params = payload.get("prompt_cache") or payload.get("sample_parameters")
        if not isinstance(cached_params, list) or not cached_params:
            raise ValueError(f"No sample prompts found in cache: {cache_path}")

        if args.sample_prompts is None:
            raise ValueError("--sample_prompts is required when --use_precached_sample_prompts is set")
        prompts = load_prompts(args.sample_prompts)
        if not prompts:
            raise ValueError(f"No prompts found in {args.sample_prompts}")

        sample_params = self._apply_sample_defaults(args, prompts)
        if len(sample_params) != len(cached_params):
            raise ValueError(
                "Sample prompt count does not match precached embeddings "
                f"(prompts={len(sample_params)} cache={len(cached_params)})."
            )

        def _normalize_text(value: Optional[str]) -> str:
            if value is None:
                return ""
            return " ".join(str(value).split())

        for idx, param in enumerate(sample_params):
            cache_entry = cached_params[idx]
            if not isinstance(cache_entry, dict):
                raise ValueError(f"Invalid cache entry at {idx} ({cache_path})")

            cfg_scale = param.get("cfg_scale", None)
            guidance_scale = param.get("guidance_scale", self.default_guidance_scale)
            effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
            try:
                requires_negative_embed = float(effective_cfg_scale) != 1.0
            except (TypeError, ValueError):
                requires_negative_embed = False

            expected_prompt = _normalize_text(param.get("prompt", ""))
            cached_prompt = _normalize_text(cache_entry.get("prompt", ""))
            if expected_prompt != cached_prompt:
                raise ValueError(
                    "Prompt text mismatch with precached embeddings at index "
                    f"{idx} ({cache_path}). Rebuild sample prompt cache or disable "
                    "--use_precached_sample_prompts.\n"
                    f"Current: {param.get('prompt', '')}\n"
                    f"Cached : {cache_entry.get('prompt', '')}"
                )

            expected_negative = _normalize_text(param.get("negative_prompt", ""))
            cached_negative = _normalize_text(cache_entry.get("negative_prompt", ""))
            if expected_negative != cached_negative:
                raise ValueError(
                    "Negative prompt mismatch with precached embeddings at index "
                    f"{idx} ({cache_path}). Rebuild sample prompt cache or disable "
                    "--use_precached_sample_prompts.\n"
                    f"Current: {param.get('negative_prompt', '')}\n"
                    f"Cached : {cache_entry.get('negative_prompt', '')}"
                )

            if cache_entry.get("prompt_embeds") is None or cache_entry.get("prompt_attention_mask") is None:
                raise ValueError(f"Missing prompt embeddings in cache entry {idx} ({cache_path})")
            param["prompt_embeds"] = cache_entry["prompt_embeds"]
            param["prompt_attention_mask"] = cache_entry["prompt_attention_mask"]
            if requires_negative_embed or param.get("negative_prompt"):
                if cache_entry.get("negative_prompt_embeds") is None or cache_entry.get(
                    "negative_prompt_attention_mask"
                ) is None:
                    raise ValueError(
                        "Missing negative prompt embeddings in cache entry "
                        f"{idx} ({cache_path}); this prompt needs CFG (guidance/cfg != 1), "
                        "so negative embeddings must be precached."
                    )
                param["negative_prompt_embeds"] = cache_entry["negative_prompt_embeds"]
                param["negative_prompt_attention_mask"] = cache_entry["negative_prompt_attention_mask"]

        return sample_params

    def _load_precached_sample_latents(self, args: argparse.Namespace, sample_params: List[Dict]) -> None:
        """Load precached I2V / V2V / reference-audio latents into sample_params (in-place)."""
        cache_path = getattr(args, "sample_latents_cache", None) or self._resolve_default_sample_latents_cache(args)
        if not os.path.exists(cache_path):
            logger.warning("Precached latents not found: %s — skipping (samples will run without conditioning)", cache_path)
            return

        logger.info(f"Loading precached conditioning latents from {cache_path}")
        try:
            latent_payload = torch.load(cache_path, map_location="cpu")
            latent_cache = latent_payload.get("latent_cache", [])

            # Match latents with prompts by index
            i2v_count = 0
            v2v_count = 0
            ref_audio_count = 0
            for entry in latent_cache:
                prompt_idx = entry.get("prompt_index")
                if prompt_idx is not None and 0 <= prompt_idx < len(sample_params):
                    if "conditioning_latent" in entry:
                        sample_params[prompt_idx]["conditioning_latent"] = entry["conditioning_latent"]
                        i2v_count += 1
                    if "v2v_ref_latent" in entry:
                        sample_params[prompt_idx]["v2v_ref_latent"] = entry["v2v_ref_latent"]
                        v2v_count += 1
                    ref_audio_latent_entry = entry.get("ref_audio_latent")
                    if ref_audio_latent_entry is None and "reference_audio_latent" in entry:
                        ref_audio_latent_entry = entry["reference_audio_latent"]
                    if ref_audio_latent_entry is not None:
                        sample_params[prompt_idx]["ref_audio_latent"] = ref_audio_latent_entry
                        ref_audio_count += 1

                    ref_audio_path_entry = entry.get("ref_audio_path")
                    if ref_audio_path_entry is None and "reference_audio_path" in entry:
                        ref_audio_path_entry = entry["reference_audio_path"]
                    if ref_audio_path_entry is not None and "ref_audio_path" not in sample_params[prompt_idx]:
                        sample_params[prompt_idx]["ref_audio_path"] = ref_audio_path_entry

            logger.info(
                "Loaded precached latents: %d I2V, %d V2V references, %d reference-audio",
                i2v_count,
                v2v_count,
                ref_audio_count,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load latents cache: {e}")

    def _resolve_first_dataset_cache_directory(self, args: argparse.Namespace) -> str:
        from musubi_tuner.dataset import config_utils
        from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

        if getattr(args, "dataset_manifest", None):
            dataset_manifest = config_utils.load_dataset_manifest(args.dataset_manifest)
            manifest_architecture = dataset_manifest.get("architecture")
            if manifest_architecture is not None and manifest_architecture != ARCHITECTURE_LTX2:
                raise ValueError(
                    f"dataset manifest architecture mismatch: expected '{ARCHITECTURE_LTX2}', got '{manifest_architecture}'"
                )
            datasets = dataset_manifest.get("datasets", [])
            if not datasets:
                raise ValueError("No datasets available in dataset manifest to resolve sample cache directory")
            cache_dir = datasets[0].get("params", {}).get("cache_directory")
            if not cache_dir:
                raise ValueError("First manifest dataset has no cache_directory")
            return str(cache_dir)

        if getattr(args, "dataset_config", None):
            user_config = config_utils.load_user_config(args.dataset_config)
            blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_LTX2)
            dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
            datasets = dataset_group.datasets
            if not datasets:
                raise ValueError("No datasets available to resolve sample cache directory")
            cache_dir = getattr(datasets[0], "cache_directory", None)
            if not cache_dir:
                raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
            return cache_dir

        raise ValueError("--dataset_config or --dataset_manifest is required to resolve sample cache directory")

    def _resolve_default_sample_prompts_cache(self, args: argparse.Namespace) -> str:
        cache_dir = self._resolve_first_dataset_cache_directory(args)
        return os.path.join(cache_dir, DEFAULT_SAMPLE_PROMPTS_CACHE)

    def _resolve_default_sample_latents_cache(self, args: argparse.Namespace) -> str:
        """Resolve default path for sample latents cache (same directory as prompts cache)."""
        cache_dir = self._resolve_first_dataset_cache_directory(args)
        return os.path.join(cache_dir, DEFAULT_SAMPLE_LATENTS_CACHE)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ) -> Optional[List[Dict]]:
        """Process sample prompts for inference preview during training"""
        use_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )
        if use_precached:
            logger.info("LTX-2 sampling: using precached Gemma embeddings for sample prompts")
            sample_params = self._load_precached_sample_prompts(args)
        else:
            logger.info("LTX-2 sampling: deferring Gemma encoding until sampling")
            prompts = load_prompts(sample_prompts)
            if not prompts:
                return None
            sample_params = self._apply_sample_defaults(args, prompts)

        # Load precached I2V latents if requested (independent of text embedding caching)
        use_precached_latents = bool(getattr(args, "use_precached_sample_latents", False))
        if use_precached_latents:
            logger.info("LTX-2 sampling: using precached I2V conditioning latents")
            self._load_precached_sample_latents(args, sample_params)

        return sample_params

    def _build_text_encoder(self, args: argparse.Namespace, accelerator: Accelerator) -> torch.dtype:
        logger.info("Loading Gemma text encoder for LTX-2 sampling")
        gemma_safetensors = getattr(args, "gemma_safetensors", None)
        if getattr(args, "gemma_root", None) is None and not gemma_safetensors:
            raise ValueError("--gemma_root or --gemma_safetensors is required for LTX-2 sample prompts")
        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required for LTX-2 sample prompts")
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.av_encoder import (
            AVGemmaTextEncoderModelConfigurator,
            AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        )
        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.video_only_encoder import (
            VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
            VideoGemmaTextEncoderModelConfigurator,
        )

        configurator = AVGemmaTextEncoderModelConfigurator if self._audio_video else VideoGemmaTextEncoderModelConfigurator
        key_ops = AV_GEMMA_TEXT_ENCODER_KEY_OPS if self._audio_video else VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS

        mixed_precision = getattr(accelerator, "mixed_precision", "no")
        if mixed_precision == "bf16":
            text_encoder_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            text_encoder_dtype = torch.float16
        else:
            text_encoder_dtype = torch.float32

        if getattr(args, "gemma_load_in_8bit", False) or getattr(args, "gemma_load_in_4bit", False):
            if accelerator.device.type != "cuda":
                raise ValueError("Gemma 8-bit/4-bit loading requires CUDA")

        build_device = accelerator.device
        is_quantized_load = getattr(args, "gemma_load_in_8bit", False) or getattr(args, "gemma_load_in_4bit", False)

        self._text_encoder = SingleGPUModelBuilder(
            model_path=str(args.ltx2_checkpoint),
            model_class_configurator=configurator,
            model_sd_ops=key_ops,
            module_ops=module_ops_from_gemma_root(
                args.gemma_root,
                gemma_safetensors=gemma_safetensors,
                torch_dtype=text_encoder_dtype,
                load_in_8bit=bool(getattr(args, "gemma_load_in_8bit", False)),
                load_in_4bit=bool(getattr(args, "gemma_load_in_4bit", False)),
                bnb_4bit_quant_type=str(getattr(args, "gemma_bnb_4bit_quant_type", "nf4")),
                bnb_4bit_use_double_quant=not bool(getattr(args, "gemma_bnb_4bit_disable_double_quant", False)),
                bnb_4bit_compute_dtype=text_encoder_dtype,
                device=build_device,
            ),
        ).build(device=build_device, dtype=text_encoder_dtype)
        text_model = getattr(self._text_encoder, "model", None)
        is_quantized = False
        if text_model is not None:
            is_quantized = bool(getattr(text_model, "is_loaded_in_8bit", False)) or bool(
                getattr(text_model, "is_loaded_in_4bit", False)
            )
        is_fp8 = bool(getattr(self._text_encoder, "_has_fp8_model", False))
        if not is_quantized and not is_fp8 and accelerator.device.type != "cpu":
            self._text_encoder.to(accelerator.device)
        text_model = getattr(self._text_encoder, "model", None)
        if text_model is not None:
            try:
                first_param = next(text_model.parameters())
                logger.info(
                    "Gemma text encoder device: %s dtype: %s",
                    first_param.device,
                    first_param.dtype,
                )
            except StopIteration:
                pass
        self._text_encoder.eval()
        return text_encoder_dtype

    def _encode_prompt_text(
        self,
        accelerator: Accelerator,
        prompt_text: str,
        text_encoder_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with accelerator.autocast(), torch.no_grad():
            out = self._text_encoder(prompt_text, padding_side="left")
            if self._ltx_mode == "audio":
                embed = out.audio_encoding if hasattr(out, "audio_encoding") else out.video_encoding
            elif self._audio_video:
                embed = torch.cat([out.video_encoding, out.audio_encoding], dim=-1)
            else:
                embed = out.video_encoding
            mask = out.attention_mask
        return embed.squeeze(0).detach().cpu(), mask.squeeze(0).detach().cpu()

    def _cleanup_text_encoder(self, accelerator: Accelerator) -> None:
        if self._text_encoder is None:
            return
        if hasattr(self._text_encoder, "model"):
            self._text_encoder.model = None
        if hasattr(self._text_encoder, "tokenizer"):
            self._text_encoder.tokenizer = None
        if hasattr(self._text_encoder, "feature_extractor_linear"):
            self._text_encoder.feature_extractor_linear = None
        self._text_encoder = None
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()

    def sample_images(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        epoch,
        steps,
        vae,
        transformer,
        sample_parameters,
        dit_dtype,
    ):
        """LTX-2 sampling with optional DiT offloading between prompts."""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
        if sample_parameters is None:
            if getattr(args, "use_precached_sample_prompts", False) or getattr(args, "precache_sample_prompts", False):
                logger.error("No precached sample prompt embeddings found. Check --sample_prompts_cache.")
            else:
                logger.error(f"No prompt file / ???????????????: {args.sample_prompts}")
            return

        distributed_state = PartialState()  # for multi gpu distributed inference

        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()
        original_device = next(transformer.parameters()).device
        offload = bool(getattr(args, "sample_with_offloading", False))
        transformer_offloaded = offload and accelerator.device.type == "cuda"
        if transformer_offloaded:
            transformer.to("cpu")
            logger.info("Sampling offload: moved transformer to CPU before prompt loop")
            clean_memory_on_device(accelerator.device)
        if getattr(transformer, "blocks_to_swap", 0) and original_device.type == "cpu" and not transformer_offloaded:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            else:
                transformer.to(accelerator.device)
            clean_memory_on_device(accelerator.device)
            original_device = accelerator.device

        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        def ensure_transformer_on_device() -> None:
            if transformer_offloaded:
                logger.info("Sampling offload: moving transformer to GPU for denoise")
                if hasattr(transformer, "move_to_device_except_swap_blocks"):
                    transformer.move_to_device_except_swap_blocks(accelerator.device)
                else:
                    transformer.to(accelerator.device)
                clean_memory_on_device(accelerator.device)

        def offload_transformer_if_needed() -> None:
            if transformer_offloaded:
                logger.info("Sampling offload: moving transformer back to CPU")
                transformer.to("cpu")
                clean_memory_on_device(accelerator.device)

        def cleanup_embeddings(sample_parameter: Dict) -> None:
            sample_parameter.pop("prompt_embeds", None)
            sample_parameter.pop("prompt_attention_mask", None)
            sample_parameter.pop("negative_prompt_embeds", None)
            sample_parameter.pop("negative_prompt_attention_mask", None)

        def prepare_all_embeddings_batch(sample_params_list: List[Dict]) -> None:
            """Load text encoder once and encode ALL prompts before unloading."""
            def _requires_negative_embeddings(sample_parameter: Dict) -> bool:
                cfg_scale = sample_parameter.get("cfg_scale", None)
                guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
                effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
                try:
                    return float(effective_cfg_scale) != 1.0
                except (TypeError, ValueError):
                    return False

            missing_indices = []
            for idx, sample_parameter in enumerate(sample_params_list):
                needs_prompt = sample_parameter.get("prompt_embeds") is None
                needs_negative = _requires_negative_embeddings(sample_parameter) and sample_parameter.get(
                    "negative_prompt_embeds"
                ) is None
                if needs_prompt or needs_negative:
                    missing_indices.append(idx)

            if not missing_indices:
                return

            strict_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
                getattr(args, "precache_sample_prompts", False)
            )
            if strict_precached:
                preview = ",".join(str(i) for i in missing_indices[:10])
                if len(missing_indices) > 10:
                    preview += ",..."
                raise ValueError(
                    "Precached sample prompt embeddings are incomplete; refusing to load Gemma during training. "
                    f"Missing prompt/negative embeddings for sample indices [{preview}]. "
                    "Rebuild sample prompt cache with ltx2_cache_text_encoder_outputs.py."
                )

            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            logger.info("Sampling batch: loaded text encoder for %d prompts", len(sample_params_list))

            for sample_parameter in sample_params_list:
                if sample_parameter.get("prompt_embeds") is None:
                    prompt_text = sample_parameter.get("prompt", "")
                    prompt_embeds, prompt_mask = self._encode_prompt_text(accelerator, prompt_text, text_encoder_dtype)
                    sample_parameter["prompt_embeds"] = prompt_embeds
                    sample_parameter["prompt_attention_mask"] = prompt_mask

                if _requires_negative_embeddings(sample_parameter) and sample_parameter.get("negative_prompt_embeds") is None:
                    negative_prompt = sample_parameter.get("negative_prompt")
                    if negative_prompt is None:
                        negative_prompt = ""
                        sample_parameter["negative_prompt"] = negative_prompt
                    neg_embeds, neg_mask = self._encode_prompt_text(accelerator, negative_prompt, text_encoder_dtype)
                    sample_parameter["negative_prompt_embeds"] = neg_embeds
                    sample_parameter["negative_prompt_attention_mask"] = neg_mask

            self._cleanup_text_encoder(accelerator)
            logger.info("Sampling batch: unloaded text encoder after encoding all prompts")
            self._cleanup_cuda(accelerator.device)

        # Check if using precached prompts (don't cleanup precached embeddings - they're reused)
        use_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )

        # Pre-load audio components only in NON-offloading mode without subprocess (high VRAM)
        # With subprocess (default): audio decoded in separate process, no in-process loading needed
        # With offloading: audio will be decoded via subprocess during decode phase
        audio_decoder = None
        vocoder = None
        use_audio_subprocess = bool(getattr(args, "sample_audio_subprocess", True))
        disable_audio_preview = bool(getattr(args, "sample_disable_audio", False))
        audio_only_preview = bool(getattr(args, "sample_audio_only", False))
        if self._ltx_mode == "audio":
            audio_only_preview = True
        enable_audio_preview = (self._audio_video or audio_only_preview) and not disable_audio_preview
        if not transformer_offloaded and not use_audio_subprocess and enable_audio_preview and getattr(args, "ltx_mode", "video") in {"av", "audio"}:
            # High VRAM mode without subprocess: pre-load audio to GPU
            audio_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
            try:
                audio_decoder, vocoder = self._load_audio_components(
                    args,
                    audio_dtype=audio_dtype,
                    checkpoint_path=args.ltx2_checkpoint,
                    device=accelerator.device,
                )
                logger.info("Sampling: pre-loaded audio decoder/vocoder to GPU (high VRAM mode)")
            except Exception as exc:
                logger.warning("Sampling audio decoder load failed; continuing without audio preview: %s", exc)
                audio_decoder, vocoder = None, None

        if distributed_state.num_processes <= 1:
            # Batch encode all prompts upfront when offloading is enabled
            if transformer_offloaded:
                offload_transformer_if_needed()
                prepare_all_embeddings_batch(sample_parameters)

            # Load VAE once before the prompt loop to avoid repeated disk reads from the
            # (potentially huge) safetensors checkpoint.  Keep it on CPU between prompts.
            vae_for_sampling = None
            if transformer_offloaded:
                vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                logger.info("Sampling offload: loading VAE for sampling (once)")
                vae_for_sampling = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)

            with torch.no_grad(), accelerator.autocast():
                for sample_parameter in sample_parameters:
                    try:
                        if transformer_offloaded:
                            ensure_transformer_on_device()
                            self.sample_image_inference(
                                accelerator, args, transformer, dit_dtype, vae_for_sampling, save_dir, sample_parameter, epoch, steps,
                                audio_decoder=audio_decoder, vocoder=vocoder,
                            )
                            offload_transformer_if_needed()
                            vae_for_sampling.to_device("cpu")
                            self._cleanup_cuda(accelerator.device)
                        else:
                            self.sample_image_inference(
                                accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps,
                                audio_decoder=audio_decoder, vocoder=vocoder,
                            )
                    except Exception as exc:
                        logger.error("Sampling failed for prompt, skipping: %s", exc, exc_info=True)
                    clean_memory_on_device(accelerator.device)
                    self._cleanup_cuda(accelerator.device)

            if vae_for_sampling is not None:
                del vae_for_sampling
                self._cleanup_cuda(accelerator.device)

            # Cleanup embeddings after all samples are done (but NOT if precached - they're reused)
            if transformer_offloaded and not use_precached:
                for sample_parameter in sample_parameters:
                    cleanup_embeddings(sample_parameter)
        else:
            per_process_params = []
            for i in range(distributed_state.num_processes):
                per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                    my_sample_params = sample_parameter_lists[0]

                    # Batch encode all prompts for this process upfront
                    if transformer_offloaded:
                        offload_transformer_if_needed()
                        prepare_all_embeddings_batch(my_sample_params)

                    # Load VAE once before the prompt loop
                    vae_for_sampling = None
                    if transformer_offloaded:
                        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                        logger.info("Sampling offload: loading VAE for sampling (once)")
                        vae_for_sampling = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)

                    for sample_parameter in my_sample_params:
                        try:
                            if transformer_offloaded:
                                ensure_transformer_on_device()
                                self.sample_image_inference(
                                    accelerator,
                                    args,
                                    transformer,
                                    dit_dtype,
                                    vae_for_sampling,
                                    save_dir,
                                    sample_parameter,
                                    epoch,
                                    steps,
                                    audio_decoder=audio_decoder,
                                    vocoder=vocoder,
                                )
                                offload_transformer_if_needed()
                                vae_for_sampling.to_device("cpu")
                                self._cleanup_cuda(accelerator.device)
                            else:
                                self.sample_image_inference(
                                    accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps,
                                    audio_decoder=audio_decoder, vocoder=vocoder,
                                )
                        except Exception as exc:
                            logger.error("Sampling failed for prompt, skipping: %s", exc, exc_info=True)
                        self._cleanup_cuda(accelerator.device)

                    if vae_for_sampling is not None:
                        del vae_for_sampling
                        self._cleanup_cuda(accelerator.device)

                    # Cleanup embeddings after all samples for this process (but NOT if precached)
                    if transformer_offloaded and not use_precached:
                        for sample_parameter in my_sample_params:
                            cleanup_embeddings(sample_parameter)

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        if transformer_offloaded and next(transformer.parameters()).device != accelerator.device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            else:
                transformer.to(accelerator.device)
            logger.info("Sampling offload: restored transformer to training device")
            clean_memory_on_device(accelerator.device)

        transformer.switch_block_swap_for_training()
        # Ensure block-swap layout is re-applied after sampling to avoid VRAM creep.
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(accelerator.device)
        self._cleanup_cuda(accelerator.device)

    @staticmethod
    def _load_reference_for_output(
        ref_path: str,
        target_height: int,
        target_width: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Load reference image/video as [1, C, T, H, W] in [0,1] for side-by-side output."""
        from PIL import Image
        import torchvision.transforms.functional as TF
        from musubi_tuner.dataset.image_video_dataset import VIDEO_EXTENSIONS

        ext = os.path.splitext(ref_path)[1].lower()
        is_video = ext in [e.lower() for e in VIDEO_EXTENSIONS]

        def _cover_center_crop_out(pil_img, tw, th):
            cw, ch = pil_img.size
            if ch == th and cw == tw:
                return pil_img
            ar = cw / ch
            tar = tw / th
            if ar > tar:
                rh = th
                rw = max(tw, int(round(th * ar)))
            else:
                rw = tw
                rh = max(th, int(round(tw / ar)))
            pil_img = pil_img.resize((rw, rh), Image.LANCZOS)
            left = max((rw - tw) // 2, 0)
            top = max((rh - th) // 2, 0)
            return pil_img.crop((left, top, left + tw, top + th))

        frames = []
        if is_video:
            try:
                import av
                container = av.open(ref_path)
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= num_frames:
                        break
                    pil_frame = _cover_center_crop_out(frame.to_image().convert("RGB"), target_width, target_height)
                    frames.append(TF.to_tensor(pil_frame))
                container.close()
            except Exception as e:
                logger.warning(f"Failed to load reference video for output: {e}")
        if not frames:
            image = _cover_center_crop_out(Image.open(ref_path).convert("RGB"), target_width, target_height)
            frames = [TF.to_tensor(image)]

        while len(frames) < num_frames:
            frames.append(frames[-1])
        frames = frames[:num_frames]

        video = torch.stack(frames, dim=1).unsqueeze(0)
        return video.clamp(0, 1).to(torch.float32)

    def _load_and_encode_v2v_reference(
        self,
        ref_path: str,
        target_height: int,
        target_width: int,
        vae_checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
        max_frames: int = 1,
    ) -> torch.Tensor:
        """Load image or video from disk and encode through VAE for V2V reference conditioning.

        Returns:
            Encoded latent tensor [1, C, F, H_latent, W_latent]
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"V2V reference not found: {ref_path}")

        from musubi_tuner.dataset.image_video_dataset import VIDEO_EXTENSIONS

        ext = os.path.splitext(ref_path)[1].lower()
        is_video = ext in {e.lower() for e in VIDEO_EXTENSIONS}

        def _cover_center_crop(pil_img, tw, th):
            cw, ch = pil_img.size
            if ch == th and cw == tw:
                return pil_img
            ar = cw / ch
            tar = tw / th
            if ar > tar:
                rh = th
                rw = max(tw, int(round(th * ar)))
            else:
                rw = tw
                rh = max(th, int(round(tw / ar)))
            pil_img = pil_img.resize((rw, rh), Image.LANCZOS)
            left = max((rw - tw) // 2, 0)
            top = max((rh - th) // 2, 0)
            return pil_img.crop((left, top, left + tw, top + th))

        frames = []
        if is_video:
            import av
            container = av.open(ref_path)
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_frames:
                    break
                pil_frame = _cover_center_crop(frame.to_image().convert("RGB"), target_width, target_height)
                frames.append(TF.to_tensor(pil_frame))
            container.close()
            if not frames:
                raise ValueError(f"No frames decoded from V2V reference video: {ref_path}")
        else:
            image = _cover_center_crop(Image.open(ref_path).convert("RGB"), target_width, target_height)
            frames.append(TF.to_tensor(image))

        # [F, 3, H, W] → [1, 3, F, H, W], normalize to [-1, 1]
        video_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # [1, F, 3, H, W]
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous()  # [1, 3, F, H, W]
        video_tensor = (video_tensor * 2.0 - 1.0).to(device=device, dtype=dtype)

        # Pad frames to VAE alignment (LTX-2 VAE needs (F-1) % 8 == 0)
        num_frames = video_tensor.shape[2]
        remainder = (num_frames - 1) % 8
        if remainder != 0:
            pad = 8 - remainder
            last = video_tensor[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
            video_tensor = torch.cat([video_tensor, last], dim=2)

        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

        logger.info("Loading VAE encoder for V2V reference")
        vae_encoder = SingleGPUModelBuilder(
            model_path=str(vae_checkpoint_path),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        vae_encoder.eval()

        with torch.no_grad():
            latent = vae_encoder(video_tensor)  # [1, C, F_latent, H_latent, W_latent]

        logger.info(f"V2V reference encoded: {ref_path} → {latent.shape}")

        del vae_encoder
        clean_memory_on_device(device)

        return latent

    def _load_and_encode_conditioning_image(
        self,
        image_path: str,
        target_height: int,
        target_width: int,
        vae_checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load image from disk and encode through VAE for I2V conditioning.

        Args:
            image_path: Path to conditioning image (absolute or relative to working directory)
            target_height: Target video height (image will be resized)
            target_width: Target video width (image will be resized)
            vae_checkpoint_path: Path to VAE checkpoint
            device: Target device
            dtype: Target dtype

        Returns:
            Encoded image latent tensor [1, C, 1, H_latent, W_latent]
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"I2V conditioning image not found: {image_path}")

        logger.info(f"Loading I2V conditioning image: {image_path}")

        # Load and resize image with official-style "cover + center crop" behavior.
        # This preserves aspect ratio (unlike direct resize) and matches LTX-2 validation sampler.
        image = Image.open(image_path).convert("RGB")
        current_width, current_height = image.size
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                resize_height = target_height
                resize_width = max(target_width, int(round(target_height * aspect_ratio)))
            else:
                resize_width = target_width
                resize_height = max(target_height, int(round(target_width / aspect_ratio)))

            image = image.resize((resize_width, resize_height), Image.LANCZOS)
            left = max((resize_width - target_width) // 2, 0)
            top = max((resize_height - target_height) // 2, 0)
            image = image.crop((left, top, left + target_width, top + target_height))

        # Convert to tensor and normalize to [-1, 1]
        image_tensor = TF.to_tensor(image).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = (image_tensor * 2.0 - 1.0).to(device=device, dtype=dtype)

        # Add temporal dimension for VAE encoder: [B, C, T, H, W]
        image_tensor = image_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

        # Load VAE encoder (training only loads decoder, we need encoder for I2V)
        # Same approach as ltx2_cache_latents.py
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

        logger.info("Loading VAE encoder for I2V conditioning")
        vae_encoder = SingleGPUModelBuilder(
            model_path=str(vae_checkpoint_path),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        vae_encoder.eval()

        # Encode through VAE encoder
        with torch.no_grad():
            latent = vae_encoder(image_tensor)  # [1, C, 1, H_latent, W_latent]

        logger.info(f"Encoded I2V conditioning image to latent shape: {latent.shape}")

        # Clean up encoder to free VRAM
        del vae_encoder
        clean_memory_on_device(device)

        return latent

    def _load_and_encode_reference_audio_latent(
        self,
        audio_path: str,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load reference audio and encode it to LTX-2 audio latents [1, C, T, F]."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        try:
            import torchaudio
        except Exception as e:
            raise RuntimeError("torchaudio is required for reference-audio sampling") from e

        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            AudioEncoderConfigurator,
        )
        from musubi_tuner.ltx_2.model.audio_vae.ops import AudioProcessor

        logger.info("Loading reference audio: %s", audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape from {audio_path}: {tuple(waveform.shape)}")

        channels = int(waveform.shape[0])
        if channels == 1:
            waveform = waveform.repeat(2, 1)
        elif channels == 2:
            pass
        elif channels > 2:
            mono = waveform.float().mean(dim=0, keepdim=True)
            waveform = mono.repeat(2, 1)

        encoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        encoder.eval()

        processor = AudioProcessor(
            sample_rate=int(getattr(encoder, "sample_rate", 16000)),
            mel_bins=int(getattr(encoder, "mel_bins", 64)),
            mel_hop_length=int(getattr(encoder, "mel_hop_length", 160)),
            n_fft=int(getattr(encoder, "n_fft", 1024)),
        ).to(device=device, dtype=torch.float32)
        processor.eval()

        try:
            waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
            encoder_dtype = next(encoder.parameters()).dtype
            with torch.no_grad():
                mel = processor.waveform_to_mel(waveform, int(sample_rate)).to(device=device, dtype=encoder_dtype)
                latents = encoder(mel)
            latents = latents[0].detach().to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()
            logger.info("Encoded reference audio latent shape: %s", tuple(latents.shape))
            return latents
        finally:
            del encoder
            del processor
            clean_memory_on_device(device)

    def sample_image_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer,
        dit_dtype: torch.dtype,
        vae,
        save_dir: str,
        sample_parameter: Dict,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        """LTX-2-specific sampling with proper frame/size rounding."""

        # ===== PHASE 1: I2V Image Encoding (if needed) =====
        # Do this FIRST, before loading any other models, to respect --sample_with_offloading
        # This ensures only VAE encoder is in VRAM during encoding, then it's cleaned up completely
        conditioning_latent = None
        image_path = sample_parameter.get("image_path", None)

        # Check if we have a precached conditioning latent first
        if "conditioning_latent" in sample_parameter:
            conditioning_latent = sample_parameter["conditioning_latent"]
            if conditioning_latent is not None:
                device = accelerator.device
                conditioning_latent = conditioning_latent.to(device=device, dtype=dit_dtype)
                logger.info(f"I2V: Using precached conditioning latent (shape: {conditioning_latent.shape})")
                image_path = None  # Skip encoding since we have precached latent

        if image_path:
            logger.info("I2V: encoding conditioning image")
            try:
                vae_checkpoint = getattr(args, "vae", None) or getattr(args, "ltx2_checkpoint", None)
                if not vae_checkpoint:
                    raise ValueError("VAE checkpoint path required for I2V conditioning (--vae or --ltx2_checkpoint)")

                device = accelerator.device
                spatial_factor = 32
                temporal_factor = 8
                width = sample_parameter.get("width", 768)
                height = sample_parameter.get("height", 512)
                width = (width // spatial_factor) * spatial_factor
                height = (height // spatial_factor) * spatial_factor

                conditioning_latent = self._load_and_encode_conditioning_image(
                    image_path=image_path,
                    target_height=height,
                    target_width=width,
                    vae_checkpoint_path=vae_checkpoint,
                    device=device,
                    dtype=dit_dtype,
                )
                logger.info("I2V: conditioning image encoded")
            except Exception as e:
                logger.error(f"I2V: failed to load conditioning image '{image_path}': {e}")
                conditioning_latent = None

        v2v_ref_latent = None
        v2v_ref_path = sample_parameter.get("v2v_ref_path", None)

        if "v2v_ref_latent" in sample_parameter:
            v2v_ref_latent = sample_parameter["v2v_ref_latent"]
            if v2v_ref_latent is not None:
                device = accelerator.device
                v2v_ref_latent = v2v_ref_latent.to(device=device, dtype=dit_dtype)
                logger.info("V2V: using precached reference latent %s", v2v_ref_latent.shape)
                v2v_ref_path = None

        if v2v_ref_path:
            logger.info("V2V: encoding reference")
            try:
                vae_checkpoint = getattr(args, "vae", None) or getattr(args, "ltx2_checkpoint", None)
                if not vae_checkpoint:
                    raise ValueError("VAE checkpoint path required for V2V reference (--vae or --ltx2_checkpoint)")

                device = accelerator.device
                spatial_factor = 32
                width = sample_parameter.get("width", 768)
                height = sample_parameter.get("height", 512)
                width = (width // spatial_factor) * spatial_factor
                height = (height // spatial_factor) * spatial_factor

                ref_downscale = max(1, getattr(args, "reference_downscale", 1))
                if ref_downscale > 1:
                    ref_w = max((width // ref_downscale // spatial_factor) * spatial_factor, spatial_factor)
                    ref_h = max((height // ref_downscale // spatial_factor) * spatial_factor, spatial_factor)
                else:
                    ref_w, ref_h = width, height

                ref_frames = max(1, getattr(args, "reference_frames", 1))
                v2v_ref_latent = self._load_and_encode_v2v_reference(
                    ref_path=v2v_ref_path,
                    target_height=ref_h,
                    target_width=ref_w,
                    vae_checkpoint_path=vae_checkpoint,
                    device=device,
                    dtype=dit_dtype,
                    max_frames=ref_frames,
                )
            except Exception as e:
                logger.error(f"V2V: failed to load reference '{v2v_ref_path}': {e}")
                v2v_ref_latent = None

        ref_audio_latent = None
        ref_audio_path = sample_parameter.get("ref_audio_path") or sample_parameter.get("reference_audio_path")
        if "ref_audio_latent" in sample_parameter:
            ref_audio_latent = sample_parameter["ref_audio_latent"]
        elif "reference_audio_latent" in sample_parameter:
            ref_audio_latent = sample_parameter["reference_audio_latent"]

        if ref_audio_latent is not None:
            device = accelerator.device
            if isinstance(ref_audio_latent, torch.Tensor):
                if ref_audio_latent.dim() == 3:
                    ref_audio_latent = ref_audio_latent.unsqueeze(0)
                ref_audio_latent = ref_audio_latent.to(device=device, dtype=torch.float32)
                logger.info("Audio-ref: using precached reference audio latent %s", tuple(ref_audio_latent.shape))
                ref_audio_path = None
            else:
                logger.warning("Audio-ref: ignoring non-tensor ref_audio_latent of type %s", type(ref_audio_latent))
                ref_audio_latent = None

        if ref_audio_path and ref_audio_latent is None:
            logger.info("Audio-ref: encoding reference audio")
            try:
                checkpoint_path = getattr(args, "ltx2_checkpoint", None)
                if not checkpoint_path:
                    raise ValueError("--ltx2_checkpoint is required for reference-audio encoding")
                ref_audio_latent = self._load_and_encode_reference_audio_latent(
                    audio_path=ref_audio_path,
                    checkpoint_path=checkpoint_path,
                    device=accelerator.device,
                    dtype=dit_dtype,
                )
            except Exception as e:
                logger.error("Audio-ref: failed to load reference audio '%s': %s", ref_audio_path, e)
                ref_audio_latent = None

        lora_count = ensure_adapters_enabled_for_sampling(transformer)
        adapter_summary = summarize_active_adapters(transformer)
        if lora_count:
            logger.info("Sampling: LoRA modules active in transformer: %s", lora_count)
            if adapter_summary["lycoris"] > 0:
                logger.info(
                    "Sampling LyCORIS summary: active=%d blocks=%d attn1=%d attn2=%d ff=%d audio=%d quantized_origins=%d",
                    adapter_summary["lycoris"],
                    adapter_summary["block_count"],
                    adapter_summary["attn1"],
                    adapter_summary["attn2"],
                    adapter_summary["ff"],
                    adapter_summary["audio"],
                    adapter_summary["lycoris_quantized_origin"],
                )
            lora_stats = get_adapter_norm_samples(transformer)
            for stat in lora_stats:
                logger.info("Sampling LoRA norm: %s", stat)
        else:
            logger.warning("Sampling: no LoRA modules detected on transformer")

        loaded_vae = False
        if vae is None or getattr(vae, "_deferred", False):
            vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
            vae = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)
            loaded_vae = True

        # Use pre-loaded audio components if provided, otherwise load here (fallback for non-offload mode)
        loaded_audio = False
        disable_audio_preview = bool(getattr(args, "sample_disable_audio", False))
        use_audio_subprocess = bool(getattr(args, "sample_audio_subprocess", True))
        audio_only_preview = bool(getattr(args, "sample_audio_only", False))
        # When training mode is audio-only, inference must also use audio_only=True
        # to avoid context embedding split corruption and incorrect video modality.
        if self._ltx_mode == "audio":
            audio_only_preview = True
        if audio_only_preview and getattr(args, "ltx_mode", "video") not in {"av", "audio"}:
            raise ValueError("--sample_audio_only requires --ltx2_mode av or audio")
        enable_audio_preview = (self._audio_video or audio_only_preview) and not disable_audio_preview
        resolved_ic_strategy = str(
            getattr(
                args,
                "ic_lora_strategy",
                self._ic_lora_strategy
                or infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v")),
            )
            or "none"
        ).lower()
        audio_ref_only_sampling = (
            resolved_ic_strategy == "audio_ref_only_ic"
            and self._ltx_mode in {"av", "audio"}
            and isinstance(ref_audio_latent, torch.Tensor)
        )
        if isinstance(ref_audio_latent, torch.Tensor) and resolved_ic_strategy != "audio_ref_only_ic":
            logger.warning(
                "Audio-ref latent provided but --ic_lora_strategy is '%s'; sample will ignore reference audio.",
                resolved_ic_strategy,
            )
            ref_audio_latent = None
            audio_ref_only_sampling = False
        force_audio_conditioning = audio_ref_only_sampling

        # Only load audio components here if NOT in offloading mode and not pre-loaded
        # In offloading mode with subprocess enabled (default), audio is decoded in a subprocess.
        # With --no-sample_audio_subprocess, audio is loaded lazily in-process during decode phase.
        sample_with_offloading = bool(getattr(args, "sample_with_offloading", False))
        if (
            audio_decoder is None
            and vocoder is None
            and enable_audio_preview
            and getattr(args, "ltx_mode", "video") in {"av", "audio"}
        ):
            if not sample_with_offloading and not use_audio_subprocess:
                # High VRAM mode without subprocess: load audio to GPU now (everything fits)
                audio_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                try:
                    audio_decoder, vocoder = self._load_audio_components(
                        args,
                        audio_dtype=audio_dtype,
                        checkpoint_path=args.ltx2_checkpoint,
                        device=accelerator.device,
                    )
                    loaded_audio = True
                except Exception as exc:
                    logger.warning("Sampling audio decoder load failed; continuing without audio preview: %s", exc)
                    audio_decoder, vocoder = None, None
                    loaded_audio = False
            # else: subprocess mode or offloading mode - audio will be decoded later

        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get("width", 768)
        height = sample_parameter.get("height", 512)
        frame_count = sample_parameter.get("frame_count", 45)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 5.0)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)
        negative_prompt = sample_parameter.get("negative_prompt", None)
        effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
        do_classifier_free_guidance = float(effective_cfg_scale) != 1.0
        if do_classifier_free_guidance and negative_prompt is None:
            # Official CFG path still uses unconditional embedding (empty prompt).
            negative_prompt = ""
            sample_parameter["negative_prompt"] = negative_prompt

        spatial_factor = int(getattr(vae, "spatial_downsample_factor", 32))
        temporal_factor = int(getattr(vae, "temporal_downsample_factor", 8))
        width = (width // spatial_factor) * spatial_factor
        height = (height // spatial_factor) * spatial_factor
        frame_count = (frame_count - 1) // temporal_factor * temporal_factor + 1

        loaded_text_encoder = False
        strict_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )
        missing_prompt_embeds = sample_parameter.get("prompt_embeds") is None
        missing_negative_embeds = do_classifier_free_guidance and sample_parameter.get("negative_prompt_embeds") is None
        if strict_precached and (missing_prompt_embeds or missing_negative_embeds):
            missing_parts = []
            if missing_prompt_embeds:
                missing_parts.append("prompt")
            if missing_negative_embeds:
                missing_parts.append("negative")
            missing_desc = "/".join(missing_parts)
            raise ValueError(
                "Precached sample prompt embeddings are incomplete; refusing to load Gemma during training. "
                f"Missing {missing_desc} embeddings for sample index {sample_parameter.get('enum', 0)}. "
                "Rebuild sample prompt cache with ltx2_cache_text_encoder_outputs.py."
            )

        if sample_parameter.get("prompt_embeds") is None:
            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            prompt_embeds, prompt_mask = self._encode_prompt_text(accelerator, prompt, text_encoder_dtype)
            sample_parameter["prompt_embeds"] = prompt_embeds
            sample_parameter["prompt_attention_mask"] = prompt_mask
            if do_classifier_free_guidance and sample_parameter.get("negative_prompt_embeds") is None:
                neg_embeds, neg_mask = self._encode_prompt_text(
                    accelerator, negative_prompt, text_encoder_dtype
                )
                sample_parameter["negative_prompt_embeds"] = neg_embeds
                sample_parameter["negative_prompt_attention_mask"] = neg_mask
            loaded_text_encoder = True
        elif do_classifier_free_guidance and sample_parameter.get("negative_prompt_embeds") is None:
            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            neg_embeds, neg_mask = self._encode_prompt_text(
                accelerator, negative_prompt, text_encoder_dtype
            )
            sample_parameter["negative_prompt_embeds"] = neg_embeds
            sample_parameter["negative_prompt_attention_mask"] = neg_mask
            loaded_text_encoder = True

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"prompt: {prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"frame count: {frame_count}")
        logger.info(f"sample steps: {sample_steps}")
        logger.info(f"guidance scale: {guidance_scale}")
        logger.info(f"discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        # (I2V encoding now happens at the start of the method, before any model loading)

        do_classifier_free_guidance = float(effective_cfg_scale) != 1.0
        if do_classifier_free_guidance:
            logger.info(f"negative prompt: {negative_prompt}")
            logger.info(f"cfg scale: {cfg_scale}")

        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_train = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )
        wav_path = os.path.join(save_dir, save_path) + ".wav"

        # Check if two-stage inference is enabled
        use_two_stage = bool(getattr(args, "sample_two_stage", False))
        spatial_upsampler_path = getattr(args, "spatial_upsampler_path", None)
        distilled_lora_path = getattr(args, "distilled_lora_path", None)
        enable_audio_conditioning = bool(enable_audio_preview) or bool(force_audio_conditioning)

        if use_two_stage:
            if not spatial_upsampler_path:
                logger.warning("Two-stage inference requested but --spatial_upsampler_path not set; falling back to single-stage")
                use_two_stage = False
            elif force_audio_conditioning:
                logger.warning(
                    "Reference-audio conditioning is not supported with two-stage inference; falling back to single-stage"
                )
                use_two_stage = False

        if use_two_stage:
            if v2v_ref_latent is not None:
                logger.warning("V2V reference conditioning is not supported with two-stage inference; ignoring V2V reference")
                v2v_ref_latent = None
            video, audio_waveform = self.do_inference_two_stage(
                accelerator=accelerator,
                args=args,
                sample_parameter=sample_parameter,
                vae=vae,
                dit_dtype=dit_dtype,
                transformer=transformer,
                width=width,
                height=height,
                frame_count=frame_count,
                sample_steps=sample_steps,
                guidance_scale=guidance_scale,
                cfg_scale=cfg_scale,
                seed=seed,
                generator=generator,
                spatial_upsampler_path=spatial_upsampler_path,
                conditioning_latent=conditioning_latent,
                distilled_lora_path=distilled_lora_path,
                stage2_steps=int(getattr(args, "sample_stage2_steps", 3)),
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                enable_audio_preview=enable_audio_preview,
                decode_video=not audio_only_preview,
                audio_only=audio_only_preview,
            )
        else:
            video, audio_waveform = self.do_inference(
                accelerator,
                args,
                sample_parameter,
                vae,
                dit_dtype,
                transformer,
                discrete_flow_shift,
                sample_steps,
                width,
                height,
                frame_count,
                generator,
                do_classifier_free_guidance,
                guidance_scale,
                cfg_scale,
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                offload_transformer_for_decode=bool(getattr(args, "sample_with_offloading", False)),
                transformer_offload_device=torch.device("cpu"),
                restore_transformer_device=not (getattr(args, "sample_with_offloading", False) and accelerator.device.type == "cuda"),
                audio_output_path=wav_path if enable_audio_preview else None,
                use_audio_subprocess=use_audio_subprocess,
                enable_audio_preview=enable_audio_conditioning,
                decode_video=not audio_only_preview,
                audio_only=audio_only_preview,
                conditioning_latent=conditioning_latent,
                v2v_ref_latents=v2v_ref_latent,
                ref_audio_latents=ref_audio_latent,
            )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        if video is None and not audio_only_preview:
            logger.error("No video generated / 生成された動画がありません")
            return

        if getattr(args, "sample_include_reference", False) and video is not None:
            ref_path = sample_parameter.get("v2v_ref_path")
            if ref_path and os.path.exists(ref_path):
                try:
                    ref_video = self._load_reference_for_output(
                        ref_path, video.shape[3], video.shape[4], video.shape[2]
                    )
                    video = torch.cat([ref_video.to(video.device), video], dim=4)
                except Exception as e:
                    logger.warning(f"Failed to prepend reference to output: {e}")

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
        except:
            wandb = None

        video_path = None
        if video is not None:
            if video.shape[2] == 1:
                image_paths = save_images_grid(video, save_dir, save_path, create_subdir=False)
                if wandb_tracker is not None and wandb is not None:
                    for image_path in image_paths:
                        wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
            else:
                video_path = os.path.join(save_dir, save_path) + ".mp4"
                save_videos_grid(video, video_path)
                if wandb_tracker is not None and wandb is not None:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)
        if audio_waveform is not None:
            wav_path = os.path.join(save_dir, save_path) + ".wav"
            sample_rate = int(getattr(vocoder, "output_sample_rate", 24000)) if vocoder is not None else 24000
            self._save_audio_wav(wav_path, audio_waveform, sample_rate)
            if getattr(args, "sample_merge_audio", False) and video_path is not None:
                merged_path = os.path.join(save_dir, save_path) + "_av.mp4"
                self._mux_video_audio(video_path, wav_path, merged_path)
        elif getattr(args, "sample_merge_audio", False) and video_path is not None:
            wav_path = os.path.join(save_dir, save_path) + ".wav"
            if os.path.exists(wav_path):
                merged_path = os.path.join(save_dir, save_path) + "_av.mp4"
                self._mux_video_audio(video_path, wav_path, merged_path)

        if loaded_text_encoder:
            sample_parameter.pop("prompt_embeds", None)
            sample_parameter.pop("prompt_attention_mask", None)
            sample_parameter.pop("negative_prompt_embeds", None)
            sample_parameter.pop("negative_prompt_attention_mask", None)
            self._cleanup_text_encoder(accelerator)
        if loaded_vae:
            vae.to_device("cpu")
            clean_memory_on_device(device)
        if loaded_audio:
            audio_decoder.to("cpu")
            vocoder.to("cpu")
            clean_memory_on_device(device)

    def do_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: Dict,
        vae,
        dit_dtype: torch.dtype,
        transformer,
        discrete_flow_shift: float,
        sample_steps: int,
        width: int,
        height: int,
        frame_count: int,
        generator: torch.Generator,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        image_path: Optional[str] = None,
        control_video_path: Optional[str] = None,
        audio_decoder: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None,
        offload_transformer_for_decode: bool = False,
        transformer_offload_device: Optional[torch.device] = None,
        restore_transformer_device: bool = True,
        audio_output_path: Optional[str] = None,
        use_audio_subprocess: bool = False,
        enable_audio_preview: bool = False,
        decode_video: bool = True,
        audio_only: bool = False,
        conditioning_latent: Optional[torch.Tensor] = None,
        v2v_ref_latents: Optional[torch.Tensor] = None,
        ref_audio_latents: Optional[torch.Tensor] = None,
    ):
        """Generate sample video during training using LTX-2 denoising loop"""
        from musubi_tuner.ltx_2.types import AudioLatentShape, VideoPixelShape

        transformer_device = next(transformer.parameters()).device
        transformer_offload_device = transformer_offload_device or torch.device("cpu")
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)
        # Keep VAE off GPU during denoise when offloading is enabled.
        if not offload_transformer_for_decode:
            vae.to_device(transformer_device)
        vae.to_dtype(original_vae_dtype)

        # Get text embeddings
        prompt_embeds = sample_parameter.get("prompt_embeds")
        if prompt_embeds is None:
            raise ValueError("Sample parameter missing prompt embeddings")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=dit_dtype)

        prompt_mask = sample_parameter.get("prompt_attention_mask")
        def _normalize_prompt_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if mask.dim() == 1:
                return mask.unsqueeze(0)
            if mask.dim() > 2:
                return mask.view(mask.shape[0], -1)
            return mask

        if do_classifier_free_guidance:
            negative_prompt_embeds = sample_parameter.get("negative_prompt_embeds")
            negative_prompt_mask = sample_parameter.get("negative_prompt_attention_mask")
            if negative_prompt_embeds is not None:
                if negative_prompt_embeds.dim() == 2:
                    negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=transformer_device, dtype=dit_dtype
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_mask = _normalize_prompt_mask(prompt_mask)
                negative_prompt_mask = _normalize_prompt_mask(negative_prompt_mask)
                if prompt_mask is not None and negative_prompt_mask is not None:
                    prompt_mask = torch.cat([negative_prompt_mask, prompt_mask], dim=0)
                elif prompt_mask is not None:
                    logger.warning(
                        "Sampling: negative prompt mask missing; duplicating prompt mask."
                    )
                    prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
            else:
                logger.warning(
                    "Sampling: negative prompt embeddings missing; duplicating prompt embeds."
                )
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                prompt_mask = _normalize_prompt_mask(prompt_mask)
                if prompt_mask is not None:
                    prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
        if prompt_mask is not None:
            prompt_mask = _normalize_prompt_mask(prompt_mask)
        if prompt_mask is not None:
            mask_len = prompt_mask.shape[-1]
            embed_len = prompt_embeds.shape[1]
            if mask_len != embed_len:
                logger.warning(
                    "Sample prompt mask length %s != embeds length %s; aligning mask for sampling.",
                    mask_len,
                    embed_len,
                )
                if mask_len > embed_len:
                    # padding_side="left" in the Gemma encoder, keep rightmost tokens.
                    prompt_mask = prompt_mask[:, -embed_len:]
                else:
                    pad = embed_len - mask_len
                    prompt_mask = F.pad(prompt_mask, (pad, 0), value=1)
            if prompt_mask.shape[-1] != prompt_embeds.shape[1]:
                logger.warning(
                    "Sample prompt mask still mismatched after alignment (mask=%s, embeds=%s); disabling mask for sampling.",
                    prompt_mask.shape[-1],
                    prompt_embeds.shape[1],
                )
                prompt_mask = None
        prompt_mask = prompt_mask.to(device=transformer_device, dtype=torch.int64) if prompt_mask is not None else None

        resolved_ic_strategy = str(
            getattr(
                args,
                "ic_lora_strategy",
                self._ic_lora_strategy
                or infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v")),
            )
            or "none"
        ).lower()

        if ref_audio_latents is not None:
            if not isinstance(ref_audio_latents, torch.Tensor):
                raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(ref_audio_latents)}")
            if ref_audio_latents.dim() == 3:
                ref_audio_latents = ref_audio_latents.unsqueeze(0)
            if ref_audio_latents.dim() != 4:
                raise ValueError(
                    f"Expected ref_audio_latents to be 4D [B, C, T, F], got shape: {tuple(ref_audio_latents.shape)}"
                )

        if ref_audio_latents is not None and resolved_ic_strategy != "audio_ref_only_ic":
            logger.warning(
                "Sampling: reference-audio latents provided but --ic_lora_strategy is '%s'; ignoring ref-audio conditioning.",
                resolved_ic_strategy,
            )
            ref_audio_latents = None

        audio_ref_only_ic_sampling = (
            resolved_ic_strategy == "audio_ref_only_ic"
            and self._ltx_mode in {"av", "audio"}
            and ref_audio_latents is not None
        )

        attention_overrides = []
        if getattr(args, "sample_disable_flash_attn", True):
            from musubi_tuner.ltx_2.model.transformer.attention import AttentionFunction

            logger.info("Sampling: disabling FlashAttention for preview")
            attention_overrides = self._override_attention_function(
                transformer, AttentionFunction.PYTORCH
            )
            if prompt_mask is not None:
                logger.info("Sampling: disabling prompt attention mask for preview")
                prompt_mask = None

        enable_audio_preview = bool(enable_audio_preview)
        if not enable_audio_preview and not audio_ref_only_ic_sampling:
            expected_embed_dim = None
            try:
                caption_proj = getattr(transformer, "caption_projection", None)
                if caption_proj is not None and hasattr(caption_proj, "linear_1"):
                    expected_embed_dim = int(caption_proj.linear_1.in_features)
            except Exception:
                expected_embed_dim = None

            current_dim = int(prompt_embeds.shape[-1])
            if expected_embed_dim is not None and current_dim == expected_embed_dim * 2:
                logger.warning(
                    "Sampling: audio preview disabled; using video-only prompt embeddings (half of dim=%s).",
                    current_dim,
                )
                prompt_embeds = prompt_embeds[..., : expected_embed_dim]

        # Setup LTX-2 specific stepper
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper
        from musubi_tuner.ltx_2.components.schedulers import LTX2Scheduler

        stepper = EulerDiffusionStep()

        # Calculate latent dimensions
        vae_scale_factor_temporal = getattr(vae, "temporal_downsample_factor", 4)
        vae_scale_factor_spatial = getattr(vae, "spatial_downsample_factor", 8)
        latent_frames = (frame_count - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        in_channels = getattr(transformer, "in_channels", 128)

        # Initialize latents
        latents = torch.randn(
            (1, int(in_channels), latent_frames, latent_height, latent_width),
            dtype=torch.float32,
            device=transformer_device,
            generator=generator,
        )

        # ===== V2V / IC-LoRA sampling path =====
        # Mirrors the training forward pass: patchify ref+target, build Modality with
        # per-token timesteps (ref=0, target=sigma), call base_model directly.
        if v2v_ref_latents is not None:
            video, audio_waveform = self._do_v2v_denoising(
                latents=latents,
                v2v_ref_latents=v2v_ref_latents,
                transformer=transformer,
                dit_dtype=dit_dtype,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                sample_parameter=sample_parameter,
                sample_steps=sample_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                cfg_scale=cfg_scale,
                vae=vae,
                args=args,
                offload_transformer_for_decode=offload_transformer_for_decode,
                transformer_offload_device=transformer_offload_device,
                restore_transformer_device=restore_transformer_device,
                decode_video=decode_video,
                attention_overrides=attention_overrides,
            )
            return video, audio_waveform

        # Setup I2V conditioning mask if provided
        denoise_mask = None
        clean_latent = None
        i2v_conditioning_mask_tokens = None
        use_i2v_token_timestep_mask = bool(getattr(args, "sample_i2v_token_timestep_mask", True))
        if conditioning_latent is not None:
            # Validate conditioning_latent shape
            if conditioning_latent.dim() != 5:
                logger.warning(f"I2V: conditioning_latent has wrong dimensions {conditioning_latent.shape}, expected [B,C,T,H,W]. Skipping I2V conditioning.")
            elif conditioning_latent.shape[2] != 1:
                logger.warning(f"I2V: conditioning_latent has {conditioning_latent.shape[2]} frames, expected 1. Skipping I2V conditioning.")
            elif latents.shape[2] < 1:
                logger.warning("I2V: Video latents have no temporal frames. Skipping I2V conditioning.")
            elif conditioning_latent.shape[1] != latents.shape[1]:
                logger.warning(f"I2V: Channel dimension mismatch - conditioning {conditioning_latent.shape[1]} vs latents {latents.shape[1]}. Skipping I2V conditioning.")
            elif conditioning_latent.shape[-2:] != latents.shape[-2:]:
                logger.warning(f"I2V: Spatial dimension mismatch - conditioning {conditioning_latent.shape[-2:]} vs latents {latents.shape[-2:]}. Skipping I2V conditioning.")
            else:
                try:
                    cond_on_device = conditioning_latent.to(device=latents.device, dtype=latents.dtype)

                    # CRITICAL: Initialize first frame of latents with conditioning image
                    # This ensures the first frame starts as the conditioning, not random noise
                    latents[:, :, 0:1, :, :] = cond_on_device

                    # Create denoise_mask: 0.0 for first frame (locked), 1.0 for others (denoised)
                    denoise_mask = torch.ones_like(latents)
                    denoise_mask[:, :, 0:1, :, :] = 0.0

                    # Store clean conditioning latent (will be blended back at each step)
                    clean_latent = torch.zeros_like(latents)
                    clean_latent[:, :, 0:1, :, :] = cond_on_device

                    if use_i2v_token_timestep_mask:
                        bsz, _c, frames, h_lat, w_lat = latents.shape
                        seq_len = frames * h_lat * w_lat
                        first_frame_tokens = h_lat * w_lat
                        i2v_conditioning_mask_tokens = torch.zeros(
                            (bsz, seq_len),
                            device=latents.device,
                            dtype=torch.bool,
                        )
                        if first_frame_tokens > 0:
                            i2v_conditioning_mask_tokens[:, :first_frame_tokens] = True
                        logger.info("I2V: enabled token timestep mask for conditioned first-frame tokens")

                    logger.info(f"I2V: Initialized first frame conditioning (shape: {conditioning_latent.shape})")
                except Exception as e:
                    logger.error(f"I2V: Failed to setup conditioning: {e}", exc_info=True)
                    denoise_mask = None
                    clean_latent = None
                    i2v_conditioning_mask_tokens = None

        # Setup scheduler - official pipeline does NOT pass latent, uses default MAX_SHIFT_ANCHOR=4096
        ltx2_scheduler = LTX2Scheduler()
        sigmas = ltx2_scheduler.execute(steps=sample_steps).to(device=transformer_device, dtype=torch.float32)

        audio_latents = None
        ref_audio_latents_device = None
        ref_audio_seq_len = 0
        if enable_audio_preview or audio_ref_only_ic_sampling:
            frame_rate = sample_parameter.get("frame_rate", 25)
            video_shape = VideoPixelShape(
                batch=1,
                frames=int(frame_count),
                height=int(height),
                width=int(width),
                fps=float(frame_rate),
            )
            audio_cfg = self._get_audio_preview_config(args, transformer)
            channels = int(audio_cfg["channels"])
            mel_bins = int(audio_cfg["mel_bins"])
            sample_rate = int(audio_cfg["sample_rate"])
            hop_length = int(audio_cfg["hop_length"])
            audio_downsample = int(audio_cfg["audio_latent_downsample_factor"])
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                video_shape,
                channels=channels,
                mel_bins=mel_bins,
                sample_rate=sample_rate,
                hop_length=hop_length,
                audio_latent_downsample_factor=audio_downsample,
            )
            audio_frames = max(int(audio_shape.frames), 1)
            audio_latents = torch.randn(
                (1, channels, audio_frames, mel_bins),
                dtype=torch.float32,
                device=transformer_device,
                generator=generator,
            )

            if audio_ref_only_ic_sampling:
                if ref_audio_latents is None:
                    raise ValueError("audio_ref_only_ic sampling requires reference-audio latents")
                ref_audio_latents_device = ref_audio_latents.to(device=transformer_device, dtype=torch.float32)
                if int(ref_audio_latents_device.shape[0]) != int(audio_latents.shape[0]):
                    raise ValueError(
                        f"Batch mismatch for reference-audio: ref={tuple(ref_audio_latents_device.shape)} target={tuple(audio_latents.shape)}"
                    )
                if int(ref_audio_latents_device.shape[1]) != channels or int(ref_audio_latents_device.shape[3]) != mel_bins:
                    raise ValueError(
                        "Reference-audio latent channel/mel mismatch. "
                        f"Got ref={tuple(ref_audio_latents_device.shape)} target={tuple(audio_latents.shape)}"
                    )
                ref_audio_seq_len = int(ref_audio_latents_device.shape[2])
                if ref_audio_seq_len <= 0:
                    raise ValueError("Reference-audio latent sequence length must be > 0")

        # Denoising loop using LTX-2 scheduler with sigmas
        with torch.no_grad():
            for step_idx in tqdm(range(len(sigmas) - 1), desc="LTX-2 preview", leave=False):
                sigma = sigmas[step_idx]
                
                # Expand for CFG if needed
                latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype=dit_dtype)

                audio_model_input = None
                audio_timestep_for_model = None
                if audio_latents is not None:
                    if audio_ref_only_ic_sampling and ref_audio_latents_device is not None and ref_audio_seq_len > 0:
                        combined_audio = torch.cat([ref_audio_latents_device, audio_latents], dim=2)
                        audio_model_input = (
                            torch.cat([combined_audio, combined_audio], dim=0)
                            if do_classifier_free_guidance
                            else combined_audio
                        )
                        audio_model_input = audio_model_input.to(dtype=dit_dtype)

                        tgt_seq_len = int(audio_latents.shape[2])
                        target_audio_timestep = sigma.expand(tgt_seq_len).view(1, -1).to(
                            device=transformer_device,
                            dtype=dit_dtype,
                        )
                        ref_audio_timestep = torch.zeros(
                            (1, ref_audio_seq_len),
                            device=transformer_device,
                            dtype=dit_dtype,
                        )
                        audio_timestep_for_model = torch.cat([ref_audio_timestep, target_audio_timestep], dim=1)
                        if do_classifier_free_guidance:
                            audio_timestep_for_model = audio_timestep_for_model.repeat(2, 1)
                    else:
                        audio_model_input = (
                            torch.cat([audio_latents, audio_latents], dim=0)
                            if do_classifier_free_guidance
                            else audio_latents
                        )
                        audio_model_input = audio_model_input.to(dtype=dit_dtype)
                        audio_timestep_for_model = sigma.expand(audio_model_input.shape[0]).to(
                            device=transformer_device,
                            dtype=dit_dtype,
                        ).unsqueeze(1)

                # Prepare timestep (sigma in [0, 1])
                timestep_for_model = sigma.expand(latent_model_input.shape[0]).to(device=transformer_device, dtype=dit_dtype)

                resolved_transformer_options = {"patches_replace": {}}
                if i2v_conditioning_mask_tokens is not None:
                    video_conditioning_mask_tokens = i2v_conditioning_mask_tokens
                    if do_classifier_free_guidance:
                        video_conditioning_mask_tokens = torch.cat(
                            [video_conditioning_mask_tokens, video_conditioning_mask_tokens],
                            dim=0,
                        )
                    resolved_transformer_options["video_conditioning_mask"] = video_conditioning_mask_tokens

                if (
                    audio_ref_only_ic_sampling
                    and audio_model_input is not None
                    and ref_audio_seq_len > 0
                ):
                    resolved_transformer_options.update(
                        self._build_audio_ref_transformer_overrides(
                            args=args,
                            transformer=transformer,
                            video_latents=latent_model_input,
                            text_embeds=prompt_embeds,
                            text_mask=prompt_mask,
                            audio_model_latents=audio_model_input,
                            ref_audio_seq_len=ref_audio_seq_len,
                            device=transformer_device,
                            dtype=dit_dtype,
                        )
                    )

                # Model prediction
                if self._audio_video and audio_model_input is not None:
                    model_input = [latent_model_input, audio_model_input]
                else:
                    model_input = latent_model_input

                model_pred = transformer(
                    model_input,
                    timestep=timestep_for_model.unsqueeze(1),  # [B, 1] for per-token timesteps
                    audio_timestep=audio_timestep_for_model,
                    context=prompt_embeds,
                    attention_mask=prompt_mask,
                    frame_rate=sample_parameter.get("frame_rate", 25),
                    transformer_options=resolved_transformer_options,
                    audio_only=audio_only,
                )

                audio_pred = None
                if isinstance(model_pred, (list, tuple)):
                    video_pred, audio_pred = model_pred
                else:
                    video_pred = model_pred

                if audio_ref_only_ic_sampling and audio_pred is not None and ref_audio_seq_len > 0:
                    if int(audio_pred.shape[2]) <= ref_audio_seq_len:
                        raise ValueError(
                            f"audio_pred length {audio_pred.shape[2]} is too short for ref_audio_seq_len={ref_audio_seq_len}"
                        )
                    audio_pred = audio_pred[:, :, ref_audio_seq_len:, :]

                # IMPORTANT: Convert velocity to x0 FIRST, then apply CFG to x0
                # This matches the official LTX-2 pipeline where X0Model wraps velocity model
                # and CFG is applied to denoised (x0) outputs, not velocity predictions
                video_pred = video_pred.to(dtype=latents.dtype)

                sigma_for_video = denoise_mask * sigma if denoise_mask is not None else sigma

                if do_classifier_free_guidance:
                    effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
                    # Split velocity predictions for CFG
                    vel_uncond, vel_cond = video_pred.chunk(2)
                    # Convert each to x0
                    x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, vel_uncond, sigma_for_video)
                    x0_cond = X0PredictionWrapper.velocity_to_x0(latents, vel_cond, sigma_for_video)
                    # Apply CFG to x0 (official formula)
                    video_x0 = x0_uncond + effective_cfg_scale * (x0_cond - x0_uncond)
                else:
                    video_x0 = X0PredictionWrapper.velocity_to_x0(latents, video_pred, sigma_for_video)

                if denoise_mask is not None and clean_latent is not None:
                    # Official LTX-2 ordering: blend denoised x0 before Euler step.
                    video_x0 = video_x0 * denoise_mask + clean_latent * (1.0 - denoise_mask)

                # Euler step to next latent
                latents = stepper.step(latents, video_x0, sigmas, step_idx)

                # CRITICAL: Hard-lock conditioned frames after Euler step
                # The Euler step performs gradual correction, but I2V requires absolute locking
                if denoise_mask is not None and clean_latent is not None:
                    # Restore locked frames: where denoise_mask == 0.0, force latents = clean_latent
                    latents = latents * denoise_mask + clean_latent * (1.0 - denoise_mask)

                if audio_pred is not None and audio_latents is not None:
                    audio_pred = audio_pred.to(dtype=audio_latents.dtype)
                    audio_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
                    if audio_ref_only_ic_sampling:
                        identity_guidance_scale = float(getattr(args, "audio_ref_identity_guidance_scale", 0.0) or 0.0)
                        if identity_guidance_scale > 0.0:
                            audio_cfg_scale = identity_guidance_scale
                    if do_classifier_free_guidance:
                        aud_vel_uncond, aud_vel_cond = audio_pred.chunk(2)
                        aud_x0_uncond = X0PredictionWrapper.velocity_to_x0(audio_latents, aud_vel_uncond, sigma.item())
                        aud_x0_cond = X0PredictionWrapper.velocity_to_x0(audio_latents, aud_vel_cond, sigma.item())
                        audio_x0 = aud_x0_uncond + audio_cfg_scale * (aud_x0_cond - aud_x0_uncond)
                    else:
                        audio_x0 = X0PredictionWrapper.velocity_to_x0(audio_latents, audio_pred, sigma.item())
                    audio_latents = stepper.step(audio_latents, audio_x0, sigmas, step_idx)

        # Free I2V conditioning tensors to reclaim memory before VAE decode
        if denoise_mask is not None or clean_latent is not None:
            del denoise_mask, clean_latent
            if transformer_device.type == "cuda":
                torch.cuda.empty_cache()

        if offload_transformer_for_decode and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_offload_device)
            else:
                transformer.to(transformer_offload_device)
            logger.info("Sampling offload: moved transformer to CPU for VAE decode")
            self._cleanup_cuda(transformer_device)

        # Decode latents
        if not decode_video:
            video = None
        else:
            if offload_transformer_for_decode:
                logger.info("Sampling offload: moving VAE to GPU for decode")
                vae.to_device(transformer_device)
            with torch.no_grad():
                use_tiled_vae = getattr(args, "sample_tiled_vae", False)
                if use_tiled_vae:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
                    tile_size = getattr(args, "sample_vae_tile_size", 512)
                    tile_overlap = getattr(args, "sample_vae_tile_overlap", 64)
                    temporal_tile_size = getattr(args, "sample_vae_temporal_tile_size", 0)
                    temporal_tile_overlap = getattr(args, "sample_vae_temporal_tile_overlap", 8)
                    
                    # Use configured temporal tiling, or 9999 frames (all at once) if disabled
                    effective_temporal_size = temporal_tile_size if temporal_tile_size > 0 else 9999
                    effective_temporal_overlap = temporal_tile_overlap if temporal_tile_size > 0 else 0
                    
                    tiling_config = TilingConfig(
                        spatial_config=SpatialTilingConfig(
                            tile_size_in_pixels=tile_size,
                            tile_overlap_in_pixels=tile_overlap,
                        ),
                        temporal_config=TemporalTilingConfig(
                            tile_size_in_frames=effective_temporal_size,
                            tile_overlap_in_frames=effective_temporal_overlap,
                        ),
                    )
                    if temporal_tile_size > 0:
                        logger.info("Using tiled VAE decode (spatial=%dx%d, temporal=%d/%d)", 
                                   tile_size, tile_overlap, temporal_tile_size, temporal_tile_overlap)
                    else:
                        logger.info("Using tiled VAE decode (spatial=%dx%d, no temporal tiling)", 
                                   tile_size, tile_overlap)
                    video = vae.tiled_decode(latents.squeeze(0), tiling_config)
                    if video.dim() == 4:  # [C, T, H, W]
                        video = video.unsqueeze(0)  # [1, C, T, H, W]
                else:
                    video = vae.decode([latents.squeeze(0)])
                    if isinstance(video, list) and video:
                        video = video[0]
                        if video.dim() == 4:  # [C, T, H, W]
                            video = video.unsqueeze(0)  # [1, C, T, H, W]

        audio_waveform = None
        loaded_audio_lazily = False
        if audio_latents is not None:
            # When no audio decoder/vocoder is loaded (subprocess mode or offloading),
            # decode audio in a separate process to avoid native crashes / OOM segfaults.
            if audio_decoder is None and vocoder is None:
                if audio_output_path and enable_audio_preview:
                    logger.info("Sampling: decoding audio via subprocess")
                    if offload_transformer_for_decode:
                        vae.to_device(original_vae_device)
                        clean_memory_on_device(transformer_device)
                    self._decode_audio_preview_subprocess(
                        audio_latents=audio_latents,
                        output_path=audio_output_path,
                        checkpoint_path=args.ltx2_checkpoint,
                    )
                    # audio_waveform stays None — the .wav was written by the subprocess
                else:
                    logger.info("Sampling: skipping audio decode (no output path or audio preview disabled)")

            elif audio_decoder is not None and vocoder is not None:
                if offload_transformer_for_decode:
                    vae.to_device(original_vae_device)
                    clean_memory_on_device(transformer_device)

                decode_device = transformer_device
                if decode_device.type == "cpu":
                    logger.info("Sampling offload: decoding audio on CPU")
                try:
                    audio_decoder.to(decode_device)
                    vocoder.to(decode_device)
                    with torch.no_grad():
                        decode_dtype = torch.bfloat16
                        audio_latents = audio_latents.to(device=decode_device, dtype=decode_dtype)
                        decoded_audio = audio_decoder(audio_latents)
                        audio_waveform = vocoder(decoded_audio).squeeze(0).float().cpu()
                except Exception as exc:
                    logger.warning("Sampling: audio decode failed; skipping audio output: %s", exc)
                    audio_waveform = None
                finally:
                    audio_decoder.to("cpu")
                    vocoder.to("cpu")
            else:
                logger.warning("Sampling: audio preview requested but no decoder/vocoder available; skipping audio decode.")

        if attention_overrides:
            self._restore_attention_function(attention_overrides)
        if offload_transformer_for_decode and restore_transformer_device and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_device)
            else:
                transformer.to(transformer_device)
            logger.info("Sampling offload: restored transformer to GPU after decode")
            clean_memory_on_device(transformer_device)

        # Normalize to [0, 1]
        if video is not None:
            video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        # Restore VAE state
        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video, audio_waveform

    def _do_v2v_denoising(
        self,
        latents: torch.Tensor,
        v2v_ref_latents: torch.Tensor,
        transformer,
        dit_dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor],
        sample_parameter: Dict,
        sample_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        vae,
        args: argparse.Namespace,
        offload_transformer_for_decode: bool = False,
        transformer_offload_device: Optional[torch.device] = None,
        restore_transformer_device: bool = True,
        decode_video: bool = True,
        attention_overrides=None,
    ):
        """V2V / IC-LoRA denoising: concatenate reference + target tokens with per-token timesteps.

        Mirrors the training forward pass exactly — patchify ref & target, build a ``Modality``
        with ref timesteps=0 / target timesteps=sigma, and call the base ``LTXModel`` directly
        (bypassing the LTX2Wrapper).
        """
        from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
        from musubi_tuner.ltx_2.components.schedulers import LTX2Scheduler
        from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper
        from musubi_tuner.ltx_2.model.transformer.modality import Modality
        from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape

        transformer_device = latents.device
        transformer_offload_device = transformer_offload_device or torch.device("cpu")
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)

        patchifier = VideoLatentPatchifier(patch_size=1)
        stepper = EulerDiffusionStep()

        # Prepare reference latents
        v2v_ref_latents = v2v_ref_latents.to(device=transformer_device, dtype=dit_dtype)
        bsz = latents.shape[0]
        ref_frames = int(v2v_ref_latents.shape[2])
        tgt_frames = int(latents.shape[2])
        ref_height = int(v2v_ref_latents.shape[3])
        ref_width = int(v2v_ref_latents.shape[4])
        tgt_height = int(latents.shape[3])
        tgt_width = int(latents.shape[4])

        if ref_height == tgt_height and ref_width == tgt_width:
            reference_downscale_factor = 1
        else:
            h_ratio = tgt_height / ref_height
            w_ratio = tgt_width / ref_width
            if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                raise ValueError(
                    f"V2V spatial mismatch: target HxW={tgt_height}x{tgt_width} vs ref HxW={ref_height}x{ref_width}. "
                    f"Ratios h={h_ratio:.2f} w={w_ratio:.2f} are not consistent integer downscale factors."
                )
            reference_downscale_factor = round(h_ratio)

        # Patchify reference tokens (constant across denoising steps)
        ref_tokens = patchifier.patchify(v2v_ref_latents)  # [B, ref_seq, D]
        ref_seq_len = ref_tokens.shape[1]

        # Conditioning mask: ref=True (conditioned, t=0), target=False (denoised, t=sigma)
        ref_conditioning_mask = torch.ones((bsz, ref_seq_len), device=transformer_device, dtype=torch.bool)

        # Compute position embeddings (constant across steps)
        ref_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=int(v2v_ref_latents.shape[1]),
                frames=ref_frames,
                height=ref_height,
                width=ref_width,
            ),
            device=transformer_device,
        )
        frame_rate_v2v = float(sample_parameter.get("frame_rate", 25))
        ref_positions = get_pixel_coords(
            latent_coords=ref_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype=dit_dtype)
        ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / frame_rate_v2v
        if reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= reference_downscale_factor
            ref_positions[:, 2, ...] *= reference_downscale_factor

        tgt_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=int(latents.shape[1]),
                frames=tgt_frames,
                height=tgt_height,
                width=tgt_width,
            ),
            device=transformer_device,
        )
        tgt_positions = get_pixel_coords(
            latent_coords=tgt_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype=dit_dtype)
        tgt_positions[:, 0, ...] = tgt_positions[:, 0, ...] / frame_rate_v2v

        combined_positions = torch.cat([ref_positions, tgt_positions], dim=2)

        # Get base model (bypass LTX2Wrapper)
        base_model = transformer.model if hasattr(transformer, "model") else transformer

        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            self._ensure_fp8_buffers_on_device(base_model)
        elif getattr(args, "nf4_base", False):
            self._ensure_nf4_buffers_on_device(base_model)

        # Scheduler
        ltx2_scheduler = LTX2Scheduler()
        sigmas = ltx2_scheduler.execute(steps=sample_steps).to(device=transformer_device, dtype=torch.float32)

        # V2V denoising loop
        logger.info("V2V sampling: %d steps, ref_frames=%d, target_frames=%d", sample_steps, ref_frames, tgt_frames)
        with torch.no_grad():
            for step_idx in tqdm(range(len(sigmas) - 1), desc="V2V preview", leave=False):
                sigma = sigmas[step_idx]

                # Patchify current noisy target
                target_tokens = patchifier.patchify(latents.to(dtype=dit_dtype))
                target_seq_len = target_tokens.shape[1]

                # Concatenate ref + target
                combined_tokens = torch.cat([ref_tokens, target_tokens], dim=1)

                # Target conditioning mask (all False = all denoised)
                target_conditioning_mask = torch.zeros(
                    (bsz, target_seq_len), device=transformer_device, dtype=torch.bool
                )
                conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

                # Per-token timesteps: ref=0, target=sigma
                combined_timesteps = sigma.view(1, 1).expand(bsz, ref_seq_len + target_seq_len)
                combined_timesteps = torch.where(
                    conditioning_mask, torch.zeros_like(combined_timesteps), combined_timesteps
                )

                perturbations = BatchedPerturbationConfig.empty(bsz)

                if do_classifier_free_guidance:
                    # Duplicate everything for CFG (unconditional + conditional)
                    cfg_tokens = combined_tokens.repeat(2, 1, 1)
                    cfg_timesteps = combined_timesteps.repeat(2, 1)
                    cfg_positions = combined_positions.repeat(2, 1, 1)
                    cfg_perturbations = BatchedPerturbationConfig.empty(bsz * 2)

                    video_modality = Modality(
                        enabled=True,
                        latent=cfg_tokens,
                        timesteps=cfg_timesteps,
                        positions=cfg_positions,
                        context=prompt_embeds,  # already [neg+pos, seq, dim] from CFG setup
                        sigma=sigma,
                        context_mask=prompt_mask,
                    )
                    pred_tokens, _ = base_model(video_modality, None, cfg_perturbations)

                    # Split and extract target predictions only
                    pred_tokens = pred_tokens[:, ref_seq_len:, :]
                    vel_uncond, vel_cond = pred_tokens.chunk(2)

                    # Unpatchify to 5D for x0 conversion
                    vel_uncond_5d = patchifier.unpatchify(
                        vel_uncond,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_height, width=tgt_width,
                        ),
                    ).to(dtype=latents.dtype)
                    vel_cond_5d = patchifier.unpatchify(
                        vel_cond,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_height, width=tgt_width,
                        ),
                    ).to(dtype=latents.dtype)

                    x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, vel_uncond_5d, sigma)
                    x0_cond = X0PredictionWrapper.velocity_to_x0(latents, vel_cond_5d, sigma)

                    effective_cfg = cfg_scale if cfg_scale is not None else guidance_scale
                    video_x0 = x0_uncond + effective_cfg * (x0_cond - x0_uncond)
                else:
                    video_modality = Modality(
                        enabled=True,
                        latent=combined_tokens,
                        timesteps=combined_timesteps,
                        positions=combined_positions,
                        context=prompt_embeds,
                        sigma=sigma,
                        context_mask=prompt_mask,
                    )
                    pred_tokens, _ = base_model(video_modality, None, perturbations)

                    # Extract target predictions only
                    target_pred = pred_tokens[:, ref_seq_len:, :]
                    target_pred_5d = patchifier.unpatchify(
                        target_pred,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_height, width=tgt_width,
                        ),
                    ).to(dtype=latents.dtype)

                    video_x0 = X0PredictionWrapper.velocity_to_x0(latents, target_pred_5d, sigma)

                # Euler step
                latents = stepper.step(latents, video_x0, sigmas, step_idx)

        # Offload transformer for VAE decode
        if offload_transformer_for_decode and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_offload_device)
            else:
                transformer.to(transformer_offload_device)
            logger.info("V2V sampling offload: moved transformer to CPU for VAE decode")
            self._cleanup_cuda(transformer_device)

        # Decode latents
        if not decode_video:
            video = None
        else:
            if offload_transformer_for_decode:
                vae.to_device(transformer_device)
            with torch.no_grad():
                use_tiled_vae = getattr(args, "sample_tiled_vae", False)
                if use_tiled_vae:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
                    tile_size = getattr(args, "sample_vae_tile_size", 512)
                    tile_overlap = getattr(args, "sample_vae_tile_overlap", 64)
                    temporal_tile_size = getattr(args, "sample_vae_temporal_tile_size", 0)
                    temporal_tile_overlap = getattr(args, "sample_vae_temporal_tile_overlap", 8)
                    effective_temporal_size = temporal_tile_size if temporal_tile_size > 0 else 9999
                    effective_temporal_overlap = temporal_tile_overlap if temporal_tile_size > 0 else 0
                    tiling_config = TilingConfig(
                        spatial_config=SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=tile_overlap),
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=effective_temporal_size, tile_overlap_in_frames=effective_temporal_overlap),
                    )
                    video = vae.tiled_decode(latents.squeeze(0), tiling_config)
                    if video.dim() == 4:
                        video = video.unsqueeze(0)
                else:
                    video = vae.decode([latents.squeeze(0)])
                    if isinstance(video, list) and video:
                        video = video[0]
                        if video.dim() == 4:
                            video = video.unsqueeze(0)

        if attention_overrides:
            self._restore_attention_function(attention_overrides)
        if offload_transformer_for_decode and restore_transformer_device and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_device)
            else:
                transformer.to(transformer_device)
            logger.info("V2V sampling offload: restored transformer to GPU after decode")
            self._cleanup_cuda(transformer_device)

        if video is not None:
            video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video, None  # no audio for v2v sampling

    def do_inference_two_stage(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: Dict,
        vae,
        dit_dtype: torch.dtype,
        transformer,
        width: int,
        height: int,
        frame_count: int,
        sample_steps: int,
        guidance_scale: float,
        cfg_scale: Optional[float],
        seed: Optional[int],
        generator: torch.Generator,
        spatial_upsampler_path: str,
        distilled_lora_path: Optional[str] = None,
        stage2_steps: int = 4,
        audio_decoder: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None,
        enable_audio_preview: bool = False,
        decode_video: bool = True,
        audio_only: bool = False,
        conditioning_latent: Optional[torch.Tensor] = None,
    ):
        """Generate sample video using two-stage inference (half-res + upsample + refine)."""
        device = accelerator.device

        # Create inferencer
        inferencer = LTX2Inferencer(
            transformer=transformer,
            vae=vae,
            device=device,
            dit_dtype=dit_dtype,
            audio_video_mode=self._audio_video,
        )

        # Load upsampler
        inferencer.load_spatial_upsampler(spatial_upsampler_path, device=torch.device("cpu"))

        # Load distilled LoRA if provided
        if distilled_lora_path:
            inferencer.load_distilled_lora(distilled_lora_path)

        # Get prompt embeddings from sample_parameter
        prompt_embeds = sample_parameter.get("prompt_embeds")
        prompt_mask = sample_parameter.get("prompt_attention_mask")
        negative_embeds = sample_parameter.get("negative_prompt_embeds")
        negative_mask = sample_parameter.get("negative_prompt_attention_mask")

        # Build audio config if needed
        audio_config = None
        if enable_audio_preview and self._audio_video:
            audio_config = self._get_audio_preview_config(args, transformer)

        # Prepare tiled VAE config
        tiled_vae_config = None
        if getattr(args, "sample_tiled_vae", False):
            tiled_vae_config = {
                "tile_size": getattr(args, "sample_vae_tile_size", 512),
                "tile_overlap": getattr(args, "sample_vae_tile_overlap", 64),
                "temporal_tile_size": getattr(args, "sample_vae_temporal_tile_size", 0) or 9999,
                "temporal_tile_overlap": getattr(args, "sample_vae_temporal_tile_overlap", 8),
            }

        # Build inference config
        config = InferenceConfig(
            prompt=sample_parameter.get("prompt", ""),
            negative_prompt=sample_parameter.get("negative_prompt"),
            width=width,
            height=height,
            frame_count=frame_count,
            frame_rate=sample_parameter.get("frame_rate", 25.0),
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            cfg_scale=cfg_scale,
            seed=seed,
            two_stage=True,
            spatial_upsampler_path=spatial_upsampler_path,
            distilled_lora_path=distilled_lora_path,
            stage2_steps=stage2_steps,
            enable_audio=enable_audio_preview,
            audio_only=audio_only,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_attention_mask=negative_mask,
            conditioning_latent=conditioning_latent,
            use_i2v_token_timestep_mask=bool(getattr(args, "sample_i2v_token_timestep_mask", True)),
            offload_between_stages=bool(getattr(args, "sample_with_offloading", False)),
            extra={"audio_config": audio_config} if audio_config else {},
        )

        # Disable flash attention for sampling if requested
        attention_overrides = []
        if getattr(args, "sample_disable_flash_attn", True):
            from musubi_tuner.ltx_2.model.transformer.attention import AttentionFunction
            logger.info("Two-stage sampling: disabling FlashAttention for preview")
            attention_overrides = self._override_attention_function(
                transformer, AttentionFunction.PYTORCH
            )

        try:
            # Run two-stage inference
            video, audio_waveform = inferencer.generate(
                config=config,
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                decode_video=decode_video,
                use_tiled_vae=bool(tiled_vae_config),
                tiled_vae_config=tiled_vae_config,
            )
        finally:
            # Restore attention settings
            if attention_overrides:
                self._restore_attention_function(attention_overrides)

        return video, audio_waveform


# ======== Argument parser setup ========


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add LTX-2-specific arguments to parser"""

    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        required=True,
        help="Path to LTX-2 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--gemma_root",
        type=str,
        default=None,
        help="Local directory containing Gemma weights/tokenizer (used for sample prompts)",
    )
    parser.add_argument(
        "--gemma_safetensors",
        type=str,
        default=None,
        help="Path to a single Gemma safetensors file (e.g. fp8 from ComfyUI). Loads weights, config, and tokenizer from one file. No --gemma_root needed.",
    )
    parser.add_argument(
        "--gemma_load_in_8bit",
        action="store_true",
        help="Load Gemma LLM in 8-bit (bitsandbytes). CUDA only.",
    )
    parser.add_argument(
        "--gemma_load_in_4bit",
        action="store_true",
        help="Load Gemma LLM in 4-bit (bitsandbytes). CUDA only.",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="bitsandbytes 4-bit quant type (nf4 or fp4)",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_disable_double_quant",
        action="store_true",
        help="Disable bitsandbytes double quant for 4-bit loading.",
    )

    parser.add_argument(
        "--ltx2_mode", "--ltx_mode",
        dest="ltx_mode",
        type=str,
        default="v",
        choices=["video", "av", "audio", "v", "a", "va"],
        help="Training modality.",
    )
    parser.add_argument(
        "--ltx_version",
        type=str,
        default="2.0",
        choices=["2.0", "2.3"],
        help=(
            "Target LTX major trainer behavior. "
            "2.0 keeps legacy defaults; 2.3 enables 2.3-oriented defaults when mode is not explicitly overridden."
        ),
    )
    parser.add_argument(
        "--ltx_version_check_mode",
        type=str,
        default="warn",
        choices=["off", "warn", "error"],
        help=(
            "How strictly to enforce --ltx_version vs checkpoint metadata consistency. "
            "'warn' logs mismatches, 'error' stops startup, 'off' disables checks."
        ),
    )
    parser.add_argument(
        "--ltx2_audio_only_model",
        action="store_true",
        help="Load physically audio-only LTX-2 transformer (omit video modules). Requires --ltx2_mode audio.",
    )
    parser.add_argument(
        "--split_attn_target",
        type=str,
        default=None,
        choices=["none", "all", "self", "cross", "text_cross", "av_cross", "video", "audio"],
        help=(
            "Enable split attention for selected modules. "
            "Targets: none/all/self/cross/text_cross/av_cross/video/audio."
        ),
    )
    parser.add_argument(
        "--split_attn_mode",
        type=str,
        default=None,
        choices=["batch", "query"],
        help="Split attention mode: batch (split by batch) or query (split by query length).",
    )
    parser.add_argument(
        "--split_attn_chunk_size",
        type=int,
        default=0,
        help="Chunk size for split_attn_mode=query. 0 uses the internal default (1024).",
    )
    parser.add_argument(
        "--ffn_chunk_target",
        type=str,
        default=None,
        choices=["none", "all", "video", "audio"],
        help="Enable FFN chunking for selected modules. Targets: none/all/video/audio.",
    )
    parser.add_argument(
        "--ffn_chunk_size",
        type=int,
        default=0,
        help="Chunk size for FFN chunking. 0 disables chunking.",
    )
    parser.add_argument(
        "--lora_target_preset",
        type=str,
        default="t2v",
        choices=["t2v", "v2v", "audio", "audio_ref_only_ic", "full"],
        help=(
            "LoRA target preset: "
            "'t2v' = text-to-video (attention only, official default), "
            "'v2v' = video-to-video/IC-LoRA (attention + feed-forward), "
            "'audio' = audio-only (audio attn/ffn + audio-side cross-modal), "
            "'audio_ref_only_ic' = ID-LoRA-style AV preset "
            "(audio attn/ffn + audio/video cross-modal both directions), "
            "'full' = all linear layers. "
            "Can be overridden by --network_args include_patterns=..."
        ),
    )
    parser.add_argument(
        "--ic_lora_strategy",
        type=str,
        default="auto",
        choices=list(IC_LORA_STRATEGIES),
        help=(
            "IC-LoRA conditioning strategy. "
            "'auto' keeps backward-compatible behavior "
            "(uses 'v2v' when --lora_target_preset=v2v, "
            "'audio_ref_only_ic' when --lora_target_preset=audio_ref_only_ic, else 'none'). "
            "'v2v' uses reference-video conditioning. "
            "'audio_ref_only_ic' uses reference-audio conditioning (ID-LoRA-style) in AV or audio-only mode."
        ),
    )
    parser.add_argument(
        "--audio_ref_use_negative_positions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For --ic_lora_strategy audio_ref_only_ic: place reference-audio token positions in negative time.",
    )
    parser.add_argument(
        "--audio_ref_mask_cross_attention_to_reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --ic_lora_strategy audio_ref_only_ic: mask A2V cross-attention so video attends only to target audio, "
            "not reference-audio tokens."
        ),
    )
    parser.add_argument(
        "--audio_ref_mask_reference_from_text_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --ic_lora_strategy audio_ref_only_ic: block reference-audio tokens from attending to text tokens "
            "(target-audio tokens still attend to text)."
        ),
    )
    parser.add_argument(
        "--audio_ref_identity_guidance_scale",
        type=float,
        default=0.0,
        help=(
            "For --ic_lora_strategy audio_ref_only_ic sampling: optional CFG scale override for target-audio branch. "
            "0.0 keeps standard cfg/guidance scale."
        ),
    )
    parser.add_argument(
        "--separate_audio_buckets",
        action="store_true",
        default=None,
        help="Split LTX-2 buckets by audio presence to avoid mixed audio/non-audio batches.",
    )
    parser.add_argument(
        "--audio_bucket_strategy",
        type=str,
        default=None,
        choices=["pad", "truncate"],
        help=(
            "Audio duration bucketing strategy. "
            "'pad' (default): round-to-nearest bucket boundary, pad shorter clips and mask loss. "
            "'truncate': floor to bucket boundary, truncate all clips to bucket length (no padding/masking needed)."
        ),
    )
    parser.add_argument(
        "--audio_bucket_interval",
        type=float,
        default=None,
        help="Audio bucket step size in seconds (default: 2.0). Controls how finely audio clips are grouped by duration.",
    )
    parser.add_argument(
        "--video_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the video diffusion loss.",
    )
    parser.add_argument(
        "--audio_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the audio diffusion loss.",
    )
    parser.add_argument(
        "--audio_loss_balance_mode",
        type=str,
        default="none",
        choices=["none", "inv_freq", "ema_mag", "uncertainty"],
        help=(
            "Optional dynamic balancing for audio loss. "
            "'none' keeps static --audio_loss_weight; "
            "'inv_freq' scales audio weight by inverse EMA of audio-batch frequency; "
            "'ema_mag' matches audio loss magnitude to a target fraction of video loss; "
            "'uncertainty' uses learnable log-variance scalars per modality "
            "(Kendall et al., CVPR 2018), no hyperparameters required."
        ),
    )
    parser.add_argument(
        "--audio_loss_balance_beta",
        type=float,
        default=0.01,
        help="EMA update factor for audio-batch frequency when --audio_loss_balance_mode=inv_freq.",
    )
    parser.add_argument(
        "--audio_loss_balance_eps",
        type=float,
        default=0.05,
        help="Minimum denominator for inverse-frequency audio weighting (prevents extreme weights).",
    )
    parser.add_argument(
        "--audio_loss_balance_min",
        type=float,
        default=0.05,
        help="Minimum clamp for effective audio loss weight after inverse-frequency scaling.",
    )
    parser.add_argument(
        "--audio_loss_balance_max",
        type=float,
        default=4.0,
        help="Maximum clamp for effective audio loss weight after inverse-frequency scaling.",
    )
    parser.add_argument(
        "--audio_loss_balance_ema_init",
        type=float,
        default=1.0,
        help="Initial EMA value used by audio loss balancing modes.",
    )
    parser.add_argument(
        "--audio_loss_balance_target_ratio",
        type=float,
        default=0.33,
        help="Target audio/video loss magnitude ratio when --audio_loss_balance_mode=ema_mag.",
    )
    parser.add_argument(
        "--audio_loss_balance_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for loss magnitude tracking when --audio_loss_balance_mode=ema_mag.",
    )
    parser.add_argument(
        "--uncertainty_lr",
        type=float,
        default=None,
        help="Learning rate for uncertainty weighting log-variance parameters. "
             "Defaults to --learning_rate. Only used with --audio_loss_balance_mode=uncertainty.",
    )
    parser.add_argument(
        "--independent_audio_timestep",
        action="store_true",
        help="Sample independent timesteps for audio noising/conditioning in AV and audio modes.",
    )
    parser.add_argument(
        "--audio_only_sequence_resolution",
        type=int,
        default=64,
        help=(
            "Virtual pixel resolution used to derive sequence length for shifted_logit_normal "
            "in --ltx_mode audio. Set 0 to use cached virtual geometry."
        ),
    )
    parser.add_argument(
        "--shifted_logit_mode",
        type=str,
        default=None,
        choices=["legacy", "stretched"],
        help=(
            "Shifted logit-normal sigma sampler mode. "
            "'legacy' keeps historical behavior; 'stretched' enables upstream Mar-2026 sampling. "
            "If unset, defaults by --ltx_version (2.0->legacy, 2.3->stretched)."
        ),
    )
    parser.add_argument(
        "--shifted_logit_eps",
        type=float,
        default=1e-3,
        help="Numerical epsilon used by --shifted_logit_mode stretched (reflection floor and uniform lower bound).",
    )
    parser.add_argument(
        "--shifted_logit_uniform_prob",
        type=float,
        default=0.1,
        help="Uniform fallback probability used by --shifted_logit_mode stretched.",
    )
    parser.add_argument(
        "--shifted_logit_shift",
        type=float,
        default=None,
        help=(
            "Override the auto-calculated logit-normal shift value. "
            "Lower values bias toward low noise / fine details, higher values toward high noise / global structure. "
            "If unset, shift is computed from sequence length (range [0.95, 2.05])."
        ),
    )
    parser.add_argument(
        "--audio_silence_regularizer",
        action="store_true",
        help="Use synthetic silence audio latents for AV batches that are missing audio latents.",
    )
    parser.add_argument(
        "--audio_silence_regularizer_weight",
        type=float,
        default=1.0,
        help="Multiplier applied to audio loss on synthetic-silence fallback batches.",
    )
    parser.add_argument(
        "--audio_supervision_mode",
        type=str,
        default="off",
        choices=["off", "warn", "error"],
        help=(
            "Monitor AV audio supervision quality. "
            "'warn' logs periodic warnings when supervised-audio ratio is too low; "
            "'error' stops training; 'off' disables checks."
        ),
    )
    parser.add_argument(
        "--audio_supervision_warmup_steps",
        type=int,
        default=50,
        help="Number of expected AV batches to observe before audio supervision checks begin.",
    )
    parser.add_argument(
        "--audio_supervision_check_interval",
        type=int,
        default=50,
        help="Run audio supervision checks every N expected AV batches.",
    )
    parser.add_argument(
        "--audio_supervision_min_ratio",
        type=float,
        default=0.9,
        help="Minimum required supervised/expected audio ratio for AV training.",
    )
    parser.add_argument(
        "--min_audio_batches_per_accum",
        type=int,
        default=0,
        help=(
            "Minimum number of audio-bearing microbatches per gradient accumulation window. "
            "0 disables quota sampling and preserves existing random sampling behavior."
        ),
    )
    parser.add_argument(
        "--audio_batch_probability",
        type=float,
        default=None,
        help=(
            "Probability of selecting an audio-bearing batch when both audio/non-audio batches remain. "
            "Mutually exclusive with --min_audio_batches_per_accum. "
            "Unset keeps existing random sampling behavior."
        ),
    )

    parser.add_argument(
        "--lycoris_config",
        type=str,
        default=None,
        help=(
            "Path to LyCORIS TOML configuration file. "
            "Use this for module-level algorithm settings without bundled example files."
        ),
    )
    parser.add_argument(
        "--init_lokr_norm",
        type=float,
        default=None,
        help=(
            "Initialize LoKR network with perturbed normal distribution (e.g., 1e-3). "
            "Helps training stability. Only applies when using LoKR algorithm."
        ),
    )
    parser.add_argument(
        "--lycoris_quantized_base_check_mode",
        type=str,
        default="warn",
        choices=["off", "warn", "error"],
        help=(
            "LyCORIS-only compatibility check when base-model quantization flags are enabled. "
            "'warn' logs a warning, 'error' stops startup, 'off' disables checks."
        ),
    )
    parser.add_argument(
        "--ltx2_first_frame_conditioning_p",
        type=float,
        default=0.1,
        help="Probability of first-frame conditioning during training (keep frame 0 clean and set its timestep to 0).",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う",
    )
    parser.add_argument(
        "--fp8_w8a8",
        action="store_true",
        help="Enable W8A8 activation quantization (saves VRAM by not storing dequantized weights "
        "in autograd graph). Requires --fp8_scaled and LoRA training.",
    )
    parser.add_argument(
        "--w8a8_mode",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
        help="W8A8 quantization format: int8 (Turing+, default) or fp8 (Ada Lovelace+).",
    )
    parser.add_argument(
        "--nf4_base",
        action="store_true",
        help="use NF4 4-bit quantization for base DiT model (reduces VRAM ~75%%)",
    )
    parser.add_argument(
        "--nf4_block_size",
        type=int,
        default=32,
        help="block size for NF4 quantization (default 32)",
    )
    parser.add_argument(
        "--quantize_device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "gpu"],
        help="Device for NF4/FP8 quantization math (default: cuda). Overrides LTX2_NF4_CALC_DEVICE / LTX2_FP8_CALC_DEVICE env vars.",
    )
    parser.add_argument(
        "--loftq_init",
        action="store_true",
        help="use LoftQ initialization for LoRA (compensates NF4 quantization error, requires --nf4_base)",
    )
    parser.add_argument(
        "--loftq_iters",
        type=int,
        default=2,
        help="number of LoftQ alternating iterations (default 2)",
    )
    parser.add_argument(
        "--awq_calibration",
        action="store_true",
        help="experimental: use AWQ-style activation-aware calibration for NF4 (requires --nf4_base)",
    )
    parser.add_argument(
        "--awq_alpha",
        type=float,
        default=0.25,
        help="AWQ scaling strength (0=no effect, 1=full activation-aware, default 0.25)",
    )
    parser.add_argument(
        "--awq_num_batches",
        type=int,
        default=8,
        help="number of synthetic calibration batches for AWQ (default 8)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Default sample height for LTX-2 preview generation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Default sample width for LTX-2 preview generation.",
    )
    parser.add_argument(
        "--sample_num_frames",
        type=int,
        default=45,
        help="Default frame count for LTX-2 preview generation.",
    )
    parser.add_argument(
        "--sample_with_offloading",
        action="store_true",
        help="Offload LTX-2 DiT to CPU between sampling prompts to save VRAM.",
    )
    parser.add_argument(
        "--precache_sample_prompts",
        action="store_true",
        help="Use precached Gemma embeddings for sample prompts (no Gemma load during training).",
    )
    parser.add_argument(
        "--use_precached_sample_prompts",
        action="store_true",
        help="Use precached Gemma embeddings for sample prompts (no Gemma load during training).",
    )
    parser.add_argument(
        "--sample_prompts_cache",
        type=str,
        default=None,
        help=(
            "Path to precached sample prompt embeddings (.pt). Defaults to "
            "the first dataset's cache_directory/ltx2_sample_prompts_cache.pt"
        ),
    )
    parser.add_argument(
        "--use_precached_sample_latents",
        action="store_true",
        help="Use precached I2V conditioning latents for sample prompts (no VAE encoder load during training).",
    )
    parser.add_argument(
        "--sample_latents_cache",
        type=str,
        default=None,
        help="Path to precached I2V conditioning latents (.pt) for sample prompts.",
    )
    parser.add_argument(
        "--sample_disable_audio",
        action="store_true",
        help="Disable audio decoding during LTX-2 preview sampling (AV mode).",
    )
    parser.add_argument(
        "--sample_audio_only",
        action="store_true",
        help="Generate audio-only previews during sampling (skip video decode/save).",
    )
    parser.add_argument(
        "--sample_disable_flash_attn",
        action="store_true",
        help="Disable FlashAttention during LTX-2 preview sampling (use SDPA).",
    )
    parser.add_argument(
        "--sample_i2v_token_timestep_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use official-style I2V token timestep masking during sampling "
            "(conditioned first-frame tokens use timestep=0 via video_conditioning_mask)."
        ),
    )
    parser.add_argument(
        "--sample_audio_subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode audio previews in a separate subprocess (default: enabled). "
            "This prevents native crashes / OOM segfaults when loading the audio "
            "decoder on low-VRAM GPUs. Use --no-sample_audio_subprocess to decode "
            "audio in-process (requires enough GPU memory for audio decoder + vocoder)."
        ),
    )
    parser.add_argument(
        "--sample_merge_audio",
        action="store_true",
        help="Mux sample audio into the sample video (outputs *_av.mp4).",
    )
    parser.add_argument(
        "--sample_include_reference",
        action="store_true",
        help="Show V2V reference side-by-side with generated output in sample videos.",
    )
    parser.add_argument(
        "--reference_downscale",
        type=int,
        default=1,
        help="Spatial downscale factor for V2V references (1=same res, 2=half). Must be >= 1.",
    )
    parser.add_argument(
        "--reference_frames",
        type=int,
        default=1,
        help="Number of reference frames to use for V2V sampling. Images are repeated to fill this count.",
    )

    # Two-stage inference arguments
    parser.add_argument(
        "--sample_two_stage",
        action="store_true",
        help="Enable two-stage inference: generate at half resolution, then upsample and refine.",
    )
    parser.add_argument(
        "--spatial_upsampler_path",
        type=str,
        default=None,
        help="Path to spatial upsampler model (ltx-2-spatial-upscaler-x2-1.0.safetensors) for two-stage inference.",
    )
    parser.add_argument(
        "--distilled_lora_path",
        type=str,
        default=None,
        help="Path to distilled LoRA (ltx-2-19b-distilled-lora-384.safetensors) for two-stage refinement.",
    )
    parser.add_argument(
        "--sample_stage2_steps",
        type=int,
        default=3,
        help="Number of denoising steps for stage 2 refinement (default: 3, official uses 3 steps with 4 sigma values).",
    )

    parser.add_argument(
        "--sample_tiled_vae",
        action="store_true",
        help="Enable tiled VAE decoding during sampling to reduce VRAM usage.",
    )
    parser.add_argument(
        "--sample_vae_tile_size",
        type=int,
        default=512,
        help="Spatial tile size in pixels for tiled VAE decode (default: 512).",
    )
    parser.add_argument(
        "--sample_vae_tile_overlap",
        type=int,
        default=64,
        help="Spatial tile overlap in pixels for tiled VAE decode (default: 64).",
    )
    parser.add_argument(
        "--sample_vae_temporal_tile_size",
        type=int,
        default=0,
        help="Temporal tile size in frames for tiled VAE decode. 0=no temporal tiling (default: 0).",
    )
    parser.add_argument(
        "--sample_vae_temporal_tile_overlap",
        type=int,
        default=8,
        help="Temporal tile overlap in frames for tiled VAE decode (default: 8).",
    )
    parser.add_argument(
        "--blockwise_checkpointing",
        action="store_true",
        help="Enable block-wise weight offloading during backward (ultra-low VRAM).",
    )
    parser.add_argument(
        "--blocks_to_checkpoint",
        type=int,
        default=-1,
        help="Number of blocks to checkpoint. -1 = all (default), 0 = none, N = last N blocks. "
             "Use with --blockwise_checkpointing to trade VRAM for speed on 12-16GB cards.",
    )
    parser.add_argument(
        "--no_convert_to_comfy",
        action="store_false",
        dest="convert_to_comfy",
        default=True,
        help="Disable automatic conversion of saved LoRA to ComfyUI format. "
             "By default, both original and ComfyUI checkpoints are saved.",
    )
    parser.add_argument(
        "--save_original_lora",
        action="store_true",
        default=True,
        help="(Default: True) Keep the original non-Comfy LoRA alongside the ComfyUI-converted checkpoint. "
             "Use --no_save_original_lora to disable.",
    )
    parser.add_argument(
        "--no_save_original_lora",
        action="store_false",
        dest="save_original_lora",
        help="Delete the original LoRA after ComfyUI conversion, keeping only *.comfy.safetensors.",
    )

    # -- Preservation / regularization flags --
    parser.add_argument(
        "--blank_preservation",
        action="store_true",
        help="Regularize LoRA to not change blank-prompt output (MSE between LoRA ON/OFF with empty prompt).",
    )
    parser.add_argument(
        "--blank_preservation_args",
        type=str,
        nargs="*",
        help="Key=value args for blank preservation, e.g. multiplier=0.5",
    )
    parser.add_argument(
        "--dop",
        action="store_true",
        help="Differential Output Preservation: regularize LoRA to not change class-prompt output.",
    )
    parser.add_argument(
        "--dop_args",
        type=str,
        nargs="*",
        help="Key=value args for DOP, e.g. class=woman multiplier=1.0",
    )
    parser.add_argument(
        "--prior_divergence",
        action="store_true",
        help="Encourage LoRA output to diverge from base model on training prompts.",
    )
    parser.add_argument(
        "--prior_divergence_args",
        type=str,
        nargs="*",
        help="Key=value args for prior divergence, e.g. multiplier=0.1",
    )
    parser.add_argument(
        "--use_precached_preservation",
        action="store_true",
        help="Load preservation embeddings from precached .pt file instead of loading Gemma. "
             "Run ltx2_cache_text_encoder_outputs.py with --precache_preservation_prompts first.",
    )
    parser.add_argument(
        "--preservation_prompts_cache",
        type=str,
        default=None,
        help="Path to precached preservation prompt embeddings (.pt). "
             "Defaults to <cache_directory>/ltx2_preservation_cache.pt. Requires --use_precached_preservation.",
    )
    parser.add_argument(
        "--audio_dop",
        action="store_true",
        help="Audio DOP: preserve base model audio predictions on non-audio batches. "
             "Only active in AV mode (--ltx2_mode av). Adds +2 forwards and +1 backward on non-audio steps.",
    )
    parser.add_argument(
        "--audio_dop_args",
        type=str,
        nargs="*",
        help="Key=value args for audio DOP, e.g. multiplier=0.5",
    )

    # -- CREPA (Cross-frame Representation Alignment) --
    parser.add_argument(
        "--crepa",
        action="store_true",
        help="Enable CREPA temporal consistency regularization (arxiv 2506.09229). "
             "Aligns DiT hidden states across video frames via a small projector MLP.",
    )
    parser.add_argument(
        "--crepa_args",
        type=str,
        nargs="*",
        help="Key=value args for CREPA, e.g. student_block_idx=16 teacher_block_idx=32 "
             "lambda_crepa=0.1 tau=1.0 num_neighbors=2 schedule=constant normalize=true",
    )
    parser.add_argument(
        "--self_flow",
        action="store_true",
        help="Enable Self-Flow regularization (dual-timestep noising + EMA-teacher feature alignment). "
             "Supported for --ltx_mode video and --ltx_mode av (video branch only in av). "
             "Single-frame image-like samples are supported via --ltx_mode video.",
    )
    parser.add_argument(
        "--self_flow_args",
        type=str,
        nargs="*",
        help="Key=value args for Self-Flow, e.g. student_block_idx=16 teacher_block_idx=32 "
        "lambda_self_flow=0.1 temporal_mode=hybrid lambda_temporal=0.1 lambda_delta=0.05 "
        "temporal_tau=1.0 num_neighbors=2 temporal_granularity=patch patch_spatial_radius=1 "
        "patch_match_mode=soft patch_match_temperature=0.2 delta_num_steps=2 "
        "motion_weighting=teacher_delta motion_weight_strength=0.5 "
        "temporal_schedule=linear temporal_warmup_steps=200 temporal_max_steps=2000 mask_ratio=0.1 "
        "frame_level_mask=false teacher_mode=base mask_focus_loss=false max_loss=0.0 "
        "student_block_stochastic_range=2 teacher_momentum=0.999 "
        "dual_timestep=true student_block_ratio=0.3 teacher_block_ratio=0.7 projector_lr=5e-5",
    )

    # -- Per-module learning rate groups --
    parser.add_argument(
        "--audio_lr",
        type=float,
        default=None,
        help="Learning rate for audio LoRA modules (audio_attn, audio_ff, cross-modal). "
             "Overridden by more specific --lr_args patterns. Defaults to --learning_rate.",
    )
    parser.add_argument(
        "--lr_args",
        type=str,
        nargs="*",
        default=None,
        help="Per-module learning rate overrides (pattern=lr). Patterns are matched via regex "
             "against LoRA module names. Example: --lr_args audio_attn=1e-6 audio_ff=1e-6 "
             "video_to_audio=1e-5",
    )

    # -- Per-module rank (dim) overrides --
    parser.add_argument(
        "--audio_dim",
        type=int,
        default=None,
        help="LoRA rank (dim) for audio modules (names containing 'audio_'). "
             "Defaults to --network_dim. Allows lower rank for audio to reduce overfitting.",
    )
    parser.add_argument(
        "--audio_alpha",
        type=float,
        default=None,
        help="LoRA alpha for audio modules. Defaults to --network_alpha. "
             "Typically set equal to --audio_dim for consistent scaling.",
    )

    # -- Caption dropout --
    parser.add_argument(
        "--caption_dropout_rate",
        type=float,
        default=0.0,
        help="Probability of dropping the caption for each sample (0.0 = disabled). "
             "Zeros out text embeddings and mask to train unconditional generation for CFG.",
    )

    return parser


# ======== Main training entry point ========


def main() -> None:
    """Main training entry point"""
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    if hasattr(args, "ltx_mode"):
        short_map = {"v": "video", "a": "audio", "va": "av"}
        if args.ltx_mode in short_map:
            args.ltx_mode = short_map[args.ltx_mode]
    apply_ltx2_tweaks(args)
    if getattr(args, "auto_blocks_to_checkpoint", False):
        if getattr(args, "blockwise_checkpointing", False) and int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            if int(getattr(args, "blocks_to_checkpoint", -1)) == -1:
                args.blocks_to_checkpoint = int(getattr(args, "blocks_to_swap", 0) or 0)
            logger.warning(
                "Using blockwise checkpointing with block swap enabled (slower but lower VRAM). "
                "blocks_to_checkpoint=%s blocks_to_swap=%s",
                args.blocks_to_checkpoint,
                args.blocks_to_swap,
            )

    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    has_bw_checkpointing = getattr(args, "blockwise_checkpointing", False)
    # Auto-enable LTX2_SWAP_TRAIN_FULL for proper block swapping during training
    if blocks_to_swap > 0:
        current_val = os.environ.get("LTX2_SWAP_TRAIN_FULL")
        # Always set to "1" for training with block swap (override any previous value)
        os.environ["LTX2_SWAP_TRAIN_FULL"] = "1"
        if current_val is None:
            logger.info("Auto-enabled LTX2_SWAP_TRAIN_FULL=1 (blocks_to_swap=%d)", blocks_to_swap)
        elif current_val != "1":
            logger.info("Overriding LTX2_SWAP_TRAIN_FULL from '%s' to '1' (blocks_to_swap=%d)", current_val, blocks_to_swap)
        else:
            logger.info("LTX2_SWAP_TRAIN_FULL=1 already set (blocks_to_swap=%d)", blocks_to_swap)

    explicit_lora_preset = any(
        arg == "--lora_target_preset" or arg.startswith("--lora_target_preset=") for arg in sys.argv
    )
    explicit_ic_strategy = any(
        arg == "--ic_lora_strategy" or arg.startswith("--ic_lora_strategy=") for arg in sys.argv
    )

    if getattr(args, "dit", None) is not None and args.dit != args.ltx2_checkpoint:
        logger.warning("Ignoring --dit for LTX-2; using --ltx2_checkpoint instead")
    args.dit = args.ltx2_checkpoint

    if getattr(args, "vae", None) is not None and args.vae != args.ltx2_checkpoint:
        logger.warning("Ignoring --vae for LTX-2; using --ltx2_checkpoint instead")
    args.vae = args.ltx2_checkpoint

    if getattr(args, "weighting_scheme", None) not in {None, "none"}:
        logger.warning("Ignoring --weighting_scheme for LTX-2; forcing weighting_scheme=none")
    args.weighting_scheme = "none"

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    uses_lycoris_module = is_lycoris_requested(args)
    requested_ic_strategy = str(getattr(args, "ic_lora_strategy", "auto") or "auto").lower()

    # Inject lora_target_preset into network_args (LTX-2 specific, non-LyCORIS only)
    if getattr(args, "ltx_mode", "video") == "audio" and not explicit_lora_preset and not uses_lycoris_module:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "audio"
    elif (
        requested_ic_strategy == "audio_ref_only_ic"
        and not explicit_lora_preset
        and not uses_lycoris_module
    ):
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "audio_ref_only_ic"
            logger.info("Using lora_target_preset=audio_ref_only_ic for --ic_lora_strategy audio_ref_only_ic")

    if explicit_ic_strategy and requested_ic_strategy == "audio_ref_only_ic" and getattr(args, "ltx_mode", "video") not in {"av", "audio"}:
        logger.warning("--ic_lora_strategy audio_ref_only_ic works in --ltx2_mode av or audio; current mode is %s", args.ltx_mode)

    if (
        explicit_lora_preset
        and getattr(args, "lora_target_preset", None) == "audio_ref_only_ic"
        and getattr(args, "ltx_mode", "video") == "audio"
    ):
        logger.warning(
            "--lora_target_preset audio_ref_only_ic in --ltx2_mode audio trains cross-modal layers that only "
            "affect the (dummy) video branch; consider --lora_target_preset audio instead."
        )

    lora_target_preset = getattr(args, "lora_target_preset", None)
    if uses_lycoris_module:
        if args.network_args is not None:
            filtered_args = [arg for arg in args.network_args if not arg.startswith("lora_target_preset=")]
            if len(filtered_args) != len(args.network_args):
                args.network_args = filtered_args
                logger.info("Removed lora_target_preset from --network_args for LyCORIS module compatibility")
        if lora_target_preset is not None:
            logger.info("Skipping lora_target_preset injection for LyCORIS network module")
    elif lora_target_preset is not None:
        if args.network_args is None:
            args.network_args = []
        # Only add if not already specified in network_args
        if not any(arg.startswith("lora_target_preset=") for arg in args.network_args):
            args.network_args.append(f"lora_target_preset={lora_target_preset}")
            logger.info(f"Using LoRA target preset: {lora_target_preset}")

    # Inject audio_dim/audio_alpha into network_args (regular LoRA only, not LyCORIS)
    if not uses_lycoris_module:
        audio_dim = getattr(args, "audio_dim", None)
        audio_alpha = getattr(args, "audio_alpha", None)
        if audio_dim is not None or audio_alpha is not None:
            if args.network_args is None:
                args.network_args = []
            if audio_dim is not None and not any(arg.startswith("audio_dim=") for arg in args.network_args):
                args.network_args.append(f"audio_dim={audio_dim}")
            if audio_alpha is not None and not any(arg.startswith("audio_alpha=") for arg in args.network_args):
                args.network_args.append(f"audio_alpha={audio_alpha}")
            logger.info(f"Per-modality LoRA rank: audio_dim={audio_dim}, audio_alpha={audio_alpha}")

    process_lycoris_config(args, logger)
    apply_lycoris_preset_before_network_creation(args, logger)

    trainer = LTX2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()

