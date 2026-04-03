"""LTX-2 model loading and configuration detection utilities."""

import os
import re
import logging

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
from musubi_tuner.modules.nf4_optimization_utils import apply_nf4_monkey_patch, load_safetensors_with_nf4_optimization, DEFAULT_NF4_BLOCK_SIZE
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.modules.w8a8_optimization_utils import apply_w8a8_monkey_patch

logger = logging.getLogger(__name__)

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
