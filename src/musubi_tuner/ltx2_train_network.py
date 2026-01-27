"""LTX-2 LoRA Training Implementation."""

import argparse
import gc
import glob
import os
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
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen        
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch  
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8    
from musubi_tuner.ltx_2.env import apply_ltx2_tweaks
from musubi_tuner.ltx_2 import inference as ltx2_inference


def _resolve_auto_ltx_mode(args: argparse.Namespace) -> str:
    if not getattr(args, "dataset_config", None):
        raise ValueError("--dataset_config is required for --ltx2_mode auto")

    from musubi_tuner.dataset import config_utils
    from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2, ImageDataset, VideoDataset
    from musubi_tuner.dataset.audio_dataset import AudioDataset

    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, args, architecture=ARCHITECTURE_LTX2
    )
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = list(dataset_group.datasets)

    has_video = False
    has_audio = False

    for dataset in datasets:
        if isinstance(dataset, AudioDataset):
            has_audio = True
            continue
        if isinstance(dataset, (ImageDataset, VideoDataset)):
            has_video = True
        if isinstance(dataset, VideoDataset):
            cache_dir = getattr(dataset, "cache_directory", None)
            if cache_dir:
                audio_caches = glob.glob(os.path.join(cache_dir, "*_ltx2_audio.safetensors"))
                if audio_caches:
                    has_audio = True

    if has_video and has_audio:
        return "av"
    if has_audio:
        return "a"
    return "v"

# LTX-2 latent normalization defaults.
# These are identity stats (mean=0, std=1). We keep them as a safe fallback and
# override them from the loaded VAE if it exposes per-channel statistics.
LTX2_LATENTS_MEAN = [0.0]
LTX2_LATENTS_STD = [1.0]

DEFAULT_SAMPLE_PROMPTS_CACHE = "ltx2_sample_prompts_cache.pt"

# Modules to keep in high precision for FP8 quantization
KEEP_FP8_HIGH_PRECISION_TOKENS = (
    "norm",
    "bias",
    "scale_shift_table",
    "patchify_proj",
    "proj_out",
    "adaln_single",
    "caption_projection",
    "layer_norm",
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


def load_ltx2_model(
    model_path: str,
    device: Union[str, torch.device] = "cpu",
    load_device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    attn_mode: str = "torch",
    ltx_mode: str = "v",
    split_attn_target: Optional[str] = None,
    split_attn_mode: Optional[str] = None,
    split_attn_chunk_size: int = 0,
    ffn_chunk_target: Optional[str] = None,
    ffn_chunk_size: int = 0,
    fp8_base: bool = False,
    fp8_scaled: bool = False,
    fp8_upcast: bool = False,
    fp8_upcast_stochastic: bool = False,
    fp8_upcast_seed: int = 0,
    load_weights_on_cpu: bool = False,
    blocks_to_swap: int = 0,
    **_: Any,
):
    """Load LTX-2 transformer for the requested modality

    Args:
        model_path: Path to safetensors model weights
        device: Target device for model
        load_device: Device to load weights into
        torch_dtype: Data type for model parameters
        attn_mode: Attention implementation (torch, flash, flash3, xformers)
        ltx_mode: "v", "av", or "a" to select video-only, audio-video, or audio-only model
        **_: Additional arguments (ignored)

    Returns:
        Loaded LTX-2 transformer model
    """
    def _cast_non_fp8_params(model: torch.nn.Module, target_dtype: torch.dtype) -> None:
        for module in model.modules():
            is_fp8_linear = isinstance(module, torch.nn.Linear) and hasattr(module, "scale_weight")
            if is_fp8_linear:
                continue
            for _, param in module.named_parameters(recurse=False):
                if isinstance(param, torch.Tensor) and param.dtype != target_dtype:
                    param.data = param.data.to(dtype=target_dtype)
            for name, buf in module.named_buffers(recurse=False):
                if isinstance(buf, torch.Tensor) and buf.dtype != target_dtype:
                    setattr(module, name, buf.to(dtype=target_dtype))

    def _detect_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor) and tensor.dtype in (
                torch.float8_e4m3fn,
                getattr(torch, "float8_e5m2", torch.float8_e4m3fn),
            ):
                return True
        return False

    target_device = torch.device(device)
    load_device = torch.device(load_device)
    from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader
    from musubi_tuner.ltx_2.model.transformer.model_configurator import (
        LTXAudioOnlyModelConfigurator,
        LTXModelConfigurator,
        LTXVideoOnlyModelConfigurator,
        LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    from musubi_tuner.networks.lora_ltx2 import LTX2Wrapper

    logger.info("Loading LTX-2 transformer via state dict: %s", model_path)
    if load_weights_on_cpu:
        logger.info("LTX-2 load path: load weights on CPU, then move to %s", target_device)
    else:
        logger.info("LTX-2 load path: load weights on %s", load_device)
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
    if ltx_mode == "av":
        configurator = LTXModelConfigurator
    elif ltx_mode == "a":
        configurator = LTXAudioOnlyModelConfigurator
    elif ltx_mode == "v":
        configurator = LTXVideoOnlyModelConfigurator
    else:
        raise ValueError(f"Invalid ltx_mode: {ltx_mode}")

    with torch.device("meta"):
        base_model = configurator.from_config(config)

    use_fp8_scaled = bool(fp8_scaled)
    use_fp8_base = bool(fp8_base)
    use_fp8 = use_fp8_scaled or use_fp8_base
    if use_fp8_scaled:
        load_dtype = None
    elif use_fp8_base:
        load_dtype = torch.float8_e4m3fn
    else:
        load_dtype = torch_dtype

    sd = load_safetensors_with_lora_and_fp8(
        model_files=model_path,
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=use_fp8_scaled,
        calc_device=target_device,
        move_to_device=(load_device == target_device),
        dit_weight_dtype=load_dtype,
        target_keys=["transformer_blocks"] if use_fp8_scaled else None,
        exclude_keys=list(KEEP_FP8_HIGH_PRECISION_TOKENS) if use_fp8_scaled else None,
    )

    renamed_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(k)
        renamed_sd[nk if nk is not None else k] = v
    sd = renamed_sd

    is_fp8_ckpt = _detect_fp8_state_dict(sd)
    if is_fp8_ckpt:
        logger.info("LTX-2 checkpoint already contains fp8 weights; using as-is.")
        apply_fp8_monkey_patch(base_model, sd, use_scaled_mm=False)
    elif use_fp8_scaled:
        apply_fp8_monkey_patch(base_model, sd, use_scaled_mm=False)

    base_model.load_state_dict(sd, strict=False, assign=True)
    if torch_dtype is not None:
        _cast_non_fp8_params(base_model, torch_dtype)
    base_model = base_model.to(load_device)
    if fp8_upcast or fp8_upcast_stochastic:
        logger.warning("LTX-2 fp8 upcast flags ignored to match Wan/Kandinsky fp8 flow.")

    model = LTX2Wrapper(base_model, patch_size=1)
    if load_device == target_device:
        model = model.to(device=target_device)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            "LTX-2 load mem [after_load_ltx2_model]: cuda_allocated=%.2fGB cuda_reserved=%.2fGB",
            allocated,
            reserved,
        )
    return model


class LTX2NetworkTrainer(NetworkTrainer):
    supports_split_av_passes: bool = True
    """Trainer for LTX-2 models with LoRA support"""

    def __init__(self) -> None:
        super().__init__()
        self._text_encoder = None
        self._dit_attn_mode: Optional[str] = None
        self._latent_norm_cache: Dict = {}
        self._warned_missing_audio = False

        # Initialize latent normalization
        mean = torch.tensor(LTX2_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LTX2_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = std.clamp_min(1e-6)
        self._latent_norm_base: Tuple[torch.Tensor, torch.Tensor] = (mean, std.reciprocal())

        self._flow_target: str = "noise"  # LTX-2 predicts noise
        self._num_timesteps: int = 1000
        self._audio_video: bool = False
        self._ltx_mode: str = "v"
        self.default_guidance_scale = 3.0
        self._audio_preview_config: Optional[Dict[str, int | float]] = None

    def _get_audio_preview_config(self, args: argparse.Namespace, transformer) -> Dict[str, int | float]:
        if self._audio_preview_config is not None:
            return self._audio_preview_config

        from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader
        from musubi_tuner.ltx_2.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR

        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required for audio preview config")

        config = SafetensorsModelStateDictLoader().metadata(str(args.ltx2_checkpoint))
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

    def _normalize_timesteps_for_model(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Normalize timesteps to the model's expected 0..1 sigma range."""
        if timesteps.numel() == 0:
            return timesteps

        return timesteps / 1000.0

    def _ensure_fp8_buffers_on_device(self, model: torch.nn.Module) -> None:
        if not any(True for _ in model.parameters()):
            return
        from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
        try:
            ensure_fp8_modules_on_device(model, next(model.parameters()).device)
        except StopIteration:
            return

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

        batch_size = latents.shape[0]
        frames, height, width = latents.shape[2], latents.shape[3], latents.shape[4]
        seq_len = int(frames * height * width)

        # Get timestep sampling mode (default to shifted_logit_normal for LTX-2)
        timestep_sampling = getattr(args, "timestep_sampling", "shifted_logit_normal")

        # For LTX-2, treat "sigma" as "shifted_logit_normal" (backward compatibility)
        if timestep_sampling == "sigma":
            timestep_sampling = "shifted_logit_normal"

        if timestep_sampling == "shifted_logit_normal":
            # Official LTX-2 implementation: shifted logit-normal distribution
            # Shift is computed based on sequence length
            shift = self._shifted_logit_normal_shift_for_sequence_length(seq_len)
            std = getattr(args, "logit_std", 1.0)
            normal_samples = torch.randn((batch_size,), device=device, dtype=torch.float32) * std + float(shift)
            sigmas = torch.sigmoid(normal_samples)
        elif timestep_sampling == "uniform":
            # Uniform sampling from [0, 1]
            sigmas = torch.rand((batch_size,), device=device, dtype=torch.float32)
        else:
            # For other sampling modes, use parent implementation
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        # Apply min/max timestep constraints if specified
        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        if min_timestep is not None or max_timestep is not None:
            min_sigma = (min_timestep / 1000.0) if min_timestep is not None else 0.0
            max_sigma = (max_timestep / 1000.0) if max_timestep is not None else 1.0
            sigmas = sigmas * (max_sigma - min_sigma) + min_sigma

        sigmas_expanded = sigmas.view(-1, 1, 1, 1, 1)
        noisy_model_input = (1.0 - sigmas_expanded) * latents.to(dtype=torch.float32) + sigmas_expanded * noise.to(
            dtype=torch.float32
        )

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

        if getattr(args, "fp8_scaled", False):
            assert getattr(args, "fp8_base", False), "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

        if getattr(args, "fp8_scaled", False) and self.dit_dtype is not None and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "LTX-2 weights are fp16; mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "LTX-2 weights are bf16; mixed precision must be bf16 or no"

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        ltx_mode = getattr(args, "ltx_mode", "auto")
        if ltx_mode == "auto":
            ltx_mode = _resolve_auto_ltx_mode(args)
            args.ltx_mode = ltx_mode
        if ltx_mode not in {"v", "av", "a"}:
            raise ValueError(f"Invalid ltx_mode: {ltx_mode}")
        self._ltx_mode = ltx_mode
        self._audio_video = self._ltx_mode in {"av", "a"}
        self.default_guidance_scale = 1.0

        args.weighting_scheme = "none"

        apply_ltx2_tweaks(args)

        if getattr(args, "separate_audio_buckets", None) is None:
            args.separate_audio_buckets = self._audio_video

    @property
    def i2v_training(self) -> bool:
        """LTX-2 doesn't currently support I2V conditioning"""
        return False

    @property
    def control_training(self) -> bool:
        """LTX-2 doesn't currently support control conditioning"""
        return False

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Convert saved LoRA to ComfyUI format"""
        if not getattr(args, 'convert_to_comfy', True):
            return

        try:
            from musubi_tuner.ltx_2.convert_lora_to_comfy import convert_lora_to_comfy
            comfy_ckpt_name = ckpt_name.replace('.safetensors', '_comfy.safetensors')
            comfy_ckpt_file = os.path.join(args.output_dir, comfy_ckpt_name)
            convert_lora_to_comfy(ckpt_file, comfy_ckpt_file, verbose=False)
            accelerator.print(f"Saved ComfyUI-compatible LoRA: {comfy_ckpt_file}")

            # Upload ComfyUI version to HuggingFace if enabled
            if args.huggingface_repo_id is not None:
                from musubi_tuner.utils import huggingface_utils
                huggingface_utils.upload(args, comfy_ckpt_file, "/" + comfy_ckpt_name, force_sync_upload=force_sync_upload)
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
            ltx_mode=self._ltx_mode,
            split_attn_target=getattr(args, "split_attn_target", None),
            split_attn_mode=getattr(args, "split_attn_mode", None),
            split_attn_chunk_size=int(getattr(args, "split_attn_chunk_size", 0) or 0),
            ffn_chunk_target=getattr(args, "ffn_chunk_target", None),
            ffn_chunk_size=int(getattr(args, "ffn_chunk_size", 0) or 0),
            fp8_base=bool(getattr(args, "fp8_base", False)),
            fp8_scaled=bool(getattr(args, "fp8_scaled", False)),
            fp8_upcast=bool(getattr(args, "fp8_upcast", False)),
            fp8_upcast_stochastic=bool(getattr(args, "fp8_upcast_stochastic", False)),
            fp8_upcast_seed=int(getattr(args, "fp8_upcast_seed", 0)),
            load_weights_on_cpu=True,
            blocks_to_swap=int(getattr(args, "blocks_to_swap", 0) or 0),
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
        cpu_device = torch.device("cpu")
        logger.info(
            "Loading LTX-2 audio decoder/vocoder from %s (load_device=%s, target_device=%s)",
            checkpoint_path,
            cpu_device,
            device,
        )
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
        ).build(device=cpu_device, dtype=audio_dtype)
        vocoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
        ).build(device=cpu_device, dtype=audio_dtype)

        if device.type != "cpu":
            audio_decoder.to(device)
            vocoder.to(device)

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
                "cpu",
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
    def _ensure_lora_enabled_for_sampling(transformer) -> int:
        try:
            from musubi_tuner.networks.lora import LoRAModule
        except Exception:
            return 0

        lora_count = 0
        for module in transformer.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            bound = getattr(module.forward, "__self__", None)
            if bound is None or not isinstance(bound, LoRAModule):
                continue
            bound.enabled = True
            lora_count += 1
        return lora_count

    @staticmethod
    def _get_lora_norm_samples(transformer, limit: int = 5) -> list[str]:
        try:
            from musubi_tuner.networks.lora import LoRAModule
        except Exception:
            return []

        stats = []
        for name, module in transformer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            bound = getattr(module.forward, "__self__", None)
            if bound is None or not isinstance(bound, LoRAModule):
                continue
            try:
                up = bound.lora_up
                down = bound.lora_down
                if isinstance(up, torch.nn.ModuleList):
                    up_norm = sum(u.weight.norm().item() for u in up)
                else:
                    up_norm = up.weight.norm().item()
                if isinstance(down, torch.nn.ModuleList):
                    down_norm = sum(d.weight.norm().item() for d in down)
                else:
                    down_norm = down.weight.norm().item()
                stats.append(f"{name}: up_norm={up_norm:.6f}, down_norm={down_norm:.6f}")
            except Exception:
                continue
            if len(stats) >= limit:
                break
        return stats

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
        *,
        force_video: Optional[bool] = None,
        force_audio: Optional[bool] = None,
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

        audio_only_override = bool(force_audio) and force_video is False
        video_only_override = bool(force_video) and force_audio is False
        if audio_only_override and video_only_override:
            raise ValueError("force_video and force_audio cannot both be True")

        audio_only_mode = self._ltx_mode == "a" or audio_only_override

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
            if audio_only_mode:
                text_embeds = conditions.get("audio_prompt_embeds")
                if text_embeds is None:
                    text_embeds = conditions.get("prompt_embeds")
                if text_embeds is None:
                    text_embeds = conditions.get("video_prompt_embeds")
            elif self._audio_video:
                video_prompt_embeds = conditions.get("video_prompt_embeds")
                audio_prompt_embeds = conditions.get("audio_prompt_embeds")
                if video_prompt_embeds is not None and audio_prompt_embeds is not None:
                    text_embeds = torch.cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)
                else:
                    text_embeds = conditions.get("prompt_embeds")
            else:
                text_embeds = conditions.get("video_prompt_embeds")

            text_mask = conditions.get("prompt_attention_mask")
        else:
            text_embeds = batch.get("text")
            text_mask = batch.get("text_mask")
            if audio_only_mode and isinstance(text_embeds, torch.Tensor) and text_embeds.shape[-1] % 2 == 0:
                text_embeds = text_embeds[..., text_embeds.shape[-1] // 2 :]

        if text_embeds is None:
            raise ValueError(
                "Cached text embeddings missing from batch. Expected either batch['conditions'] (official format) "
                "or 'text'/'text_mask' (legacy musubi format)."
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
            text_mask = text_mask.to(device=accelerator.device)
            if args.gradient_checkpointing:
                text_mask = text_mask.to(torch.bool)

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
            frame_rate = 24
        if isinstance(frame_rate, torch.Tensor):
            frame_rate = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()

        model_timesteps = timesteps.to(device=accelerator.device, dtype=network_dtype)

        model_timesteps = self._normalize_timesteps_for_model(model_timesteps)

        if model_timesteps.dim() == 0:
            model_timesteps = model_timesteps.unsqueeze(0)
        if model_timesteps.dim() == 1:
            model_timesteps = model_timesteps.unsqueeze(1)

        sigma = model_timesteps[:, 0]

        ref_latents = batch.get("ref_latents")
        if ref_latents is None:
            ref_latents = batch.get("reference_latents")
        if isinstance(ref_latents, dict):
            ref_latents = ref_latents.get("latents")

        if ref_latents is not None:
            if self._audio_video or self._ltx_mode != "v":
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
            if ref_latents.shape[3] != latents.shape[3] or ref_latents.shape[4] != latents.shape[4]:
                raise ValueError(
                    f"Spatial mismatch: latents HxW={latents.shape[3]}x{latents.shape[4]} vs ref_latents HxW={ref_latents.shape[3]}x{ref_latents.shape[4]}"     
                )

        if audio_only_mode:
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")
            if audio_latents is None:
                if self._ltx_mode == "a":
                    raise ValueError("audio_latents are required for --ltx2_mode a")
                dummy_video = torch.zeros(
                    (latents.shape[0], latents.shape[1], 1, 1, 1),
                    device=accelerator.device,
                    dtype=network_dtype,
                )
                out_audio: Dict[str, Any] = {
                    "video_pred": dummy_video,
                    "video_target": dummy_video,
                    "video_loss_weight": 0.0,
                    "audio_loss_weight": 0.0,
                    "audio_enabled_for_batch": False,
                }
                return out_audio, torch.tensor(0.0, device=accelerator.device)
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
            sigma_audio = sigma.view(-1, 1, 1, 1)
            noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

            dummy_video = torch.zeros(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=accelerator.device,
                dtype=network_dtype,
            )

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(transformer)
            with accelerator.autocast():
                model_pred = transformer(
                    [None, noisy_audio],
                    timestep=model_timesteps,
                    context=text_embeds,
                    attention_mask=text_mask,
                    frame_rate=frame_rate,
                    transformer_options={"patches_replace": {}},
                    audio_only=True,
                    video_enabled=False,
                    audio_enabled=True,
                )

            video_pred = model_pred
            audio_pred = None
            if isinstance(model_pred, (list, tuple)):
                if len(model_pred) != 2:
                    raise ValueError(f"Expected audio-only model to return [video_pred, audio_pred], got {len(model_pred)} outputs")
                video_pred, audio_pred = model_pred
            if audio_pred is None:
                raise ValueError("Audio-only mode expected an audio prediction but got None")

            if video_pred is None:
                video_pred = torch.zeros_like(dummy_video)
            video_target = torch.zeros_like(video_pred)
            out_audio: Dict[str, Any] = {
                "video_pred": video_pred,
                "video_target": video_target,
                "video_loss_weight": 0.0,
                "audio_enabled_for_batch": True,
            }

            audio_target = audio_noise - audio_latents
            audio_seq_len = int(audio_latents.shape[2])
            audio_loss_mask = torch.ones(
                (audio_latents.shape[0], audio_seq_len),
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
                    if audio_lengths.dim() != 1:
                        raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                    if audio_lengths.numel() == 1 and audio_latents.shape[0] != 1:
                        audio_lengths = audio_lengths.expand(audio_latents.shape[0])
                    if audio_lengths.shape[0] != audio_latents.shape[0]:
                        raise ValueError(
                            f"Batch size mismatch: audio_latents batch={audio_latents.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                        )

                    audio_lengths = audio_lengths.to(device=accelerator.device)
                    if audio_lengths.dtype.is_floating_point:
                        audio_lengths = audio_lengths.to(dtype=torch.int64)
                    else:
                        audio_lengths = audio_lengths.to(dtype=torch.int64)

                    audio_lengths = audio_lengths.clamp(min=0, max=audio_seq_len)
                    t = torch.arange(audio_seq_len, device=accelerator.device).view(1, -1)
                    audio_loss_mask = t < audio_lengths.view(-1, 1)

            out_audio.update(
                {
                    "audio_pred": audio_pred,
                    "audio_target": audio_target,
                    "audio_loss_mask": audio_loss_mask,
                    "audio_loss_weight": float(getattr(args, "audio_loss_weight", 1.0)),
                }
            )
            if out_audio["audio_loss_weight"] < 0.0:
                raise ValueError(f"audio_loss_weight must be >= 0. Got: {out_audio['audio_loss_weight']}")

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

            height = int(ref_latents.shape[3])
            width = int(ref_latents.shape[4])

            ref_conditioning_mask = torch.ones((bsz, ref_seq_len), device=accelerator.device, dtype=torch.bool)

            target_conditioning_mask = torch.zeros((bsz, target_seq_len), device=accelerator.device, dtype=torch.bool)
            if video_conditioning_enabled is not None:
                first_frame_tokens = height * width
                if first_frame_tokens > 0:
                    target_conditioning_mask[video_conditioning_enabled, :first_frame_tokens] = True
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

            combined_timesteps = sigma.view(bsz, 1).expand(bsz, ref_seq_len + target_seq_len)
            combined_timesteps = torch.where(conditioning_mask, torch.zeros_like(combined_timesteps), combined_timesteps)

            frame_rate_v2v = frame_rate
            if frame_rate_v2v is None:
                frame_rate_v2v = 24

            ref_frames = int(ref_latents.shape[2])
            tgt_frames = int(latents.shape[2])

            ref_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(ref_latents.shape[1]),
                    frames=ref_frames,
                    height=height,
                    width=width,
                ),
                device=accelerator.device,
            )
            ref_positions = get_pixel_coords(
                latent_coords=ref_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / float(frame_rate_v2v)

            tgt_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(latents.shape[1]),
                    frames=tgt_frames,
                    height=height,
                    width=width,
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
                context_mask=text_mask,
            )

            perturbations = BatchedPerturbationConfig.empty(bsz)
            base_model = transformer.model if hasattr(transformer, "model") else transformer

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(base_model)
            with accelerator.autocast():
                pred_tokens, _ = base_model(video_modality, None, perturbations)

            target_pred_tokens = pred_tokens[:, ref_seq_len:, :]
            target_velocity = patchifier.patchify(noise - latents)
            target_loss_mask = ~target_conditioning_mask

            out_v2v: Dict[str, Any] = {
                "video_pred": target_pred_tokens,
                "video_target": target_velocity,
                "video_loss_mask": target_loss_mask,
                "video_loss_weight": float(getattr(args, "video_loss_weight", 1.0)),
                "audio_enabled_for_batch": False,
            }
            if out_v2v["video_loss_weight"] < 0.0:
                raise ValueError(f"video_loss_weight must be >= 0. Got: {out_v2v['video_loss_weight']}")

            return out_v2v, torch.tensor(0.0, device=accelerator.device)

        audio_latents = None
        audio_noise = None
        noisy_audio = None
        audio_enabled_for_batch = False
        audio_loss_mask = None
        if self._ltx_mode == "av":
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")
            if video_only_override:
                audio_latents = None

            if audio_latents is None:
                if not self._warned_missing_audio:
                    logger.warning(
                        "LTXAV mode: missing audio latents in this batch; skipping audio branch. "
                        "Provide cached audio latents to train audio generation."
                    )
                    self._warned_missing_audio = True
            elif not isinstance(audio_latents, torch.Tensor):
                raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
            else:
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
                sigma_audio = sigma.view(-1, 1, 1, 1)
                noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise
                _check_finite("noisy_audio", noisy_audio)
                _log_stats("noisy_audio", noisy_audio)

        if self._ltx_mode == "av" and not audio_enabled_for_batch:
            if getattr(args, "av_use_video_prompt_embeds", False) and conditions is not None:
                video_prompt_embeds = conditions.get("video_prompt_embeds")
                if isinstance(video_prompt_embeds, torch.Tensor):
                    text_embeds = video_prompt_embeds
            elif isinstance(text_embeds, torch.Tensor) and text_embeds.shape[-1] % 2 == 0:
                half = text_embeds.shape[-1] // 2
                text_embeds = text_embeds[..., :half]

        if skip_nonfinite and nonfinite_flag["hit"]:
            return {"_skip_step": True, "skip_reason": nonfinite_flag["tag"]}, torch.tensor(
                0.0, device=accelerator.device
            )

        caption_channels = getattr(transformer, "caption_channels", None)
        if caption_channels is None:
            base_model = transformer.model if hasattr(transformer, "model") else transformer
            if hasattr(base_model, "caption_projection"):
                caption_channels = getattr(getattr(base_model, "caption_projection", None), "in_features", None)
        if caption_channels is not None:
            expected_last_dim = int(caption_channels) * (2 if audio_enabled_for_batch else 1)
            if text_embeds.shape[-1] != expected_last_dim:
                raise ValueError(
                    f"Text embedding dim mismatch for {'LTXAV' if self._audio_video else 'LTXV'}: "
                    f"got {text_embeds.shape[-1]}, expected {expected_last_dim}. "
                    f"(caption_channels={caption_channels})"
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

        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            self._ensure_fp8_buffers_on_device(transformer)
        with accelerator.autocast():
            model_pred = transformer(
                model_input,
                timestep=model_timesteps,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options=transformer_options if video_conditioning_mask_tokens is not None else {"patches_replace": {}},
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
            "video_loss_weight": float(getattr(args, "video_loss_weight", 1.0)),
            "audio_enabled_for_batch": bool(audio_enabled_for_batch),
        }

        if out["video_loss_weight"] < 0.0:
            raise ValueError(f"video_loss_weight must be >= 0. Got: {out['video_loss_weight']}")

        if self._ltx_mode == "av" and audio_enabled_for_batch:
            if audio_pred is None:
                raise ValueError("AV mode expected an audio prediction but got None")
            audio_target = audio_noise - audio_latents
            _check_finite("audio_target", audio_target)
            _log_stats("audio_target", audio_target)

            audio_seq_len = int(audio_latents.shape[2])
            if audio_loss_mask is None:
                audio_loss_mask = torch.ones(
                    (audio_latents.shape[0], audio_seq_len),
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
                    if audio_lengths.dim() != 1:
                        raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                    if audio_lengths.numel() == 1 and audio_latents.shape[0] != 1:
                        audio_lengths = audio_lengths.expand(audio_latents.shape[0])
                    if audio_lengths.shape[0] != audio_latents.shape[0]:
                        raise ValueError(
                            f"Batch size mismatch: audio_latents batch={audio_latents.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
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
                    "audio_loss_weight": float(getattr(args, "audio_loss_weight", 1.0)),
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
        for idx, param in enumerate(sample_params):
            cache_entry = cached_params[idx]
            if not isinstance(cache_entry, dict):
                raise ValueError(f"Invalid cache entry at {idx} ({cache_path})")
            if cache_entry.get("prompt_embeds") is None or cache_entry.get("prompt_attention_mask") is None:
                raise ValueError(f"Missing prompt embeddings in cache entry {idx} ({cache_path})")
            param["prompt_embeds"] = cache_entry["prompt_embeds"]
            param["prompt_attention_mask"] = cache_entry["prompt_attention_mask"]
            if param.get("negative_prompt"):
                if cache_entry.get("negative_prompt_embeds") is None or cache_entry.get(
                    "negative_prompt_attention_mask"
                ) is None:
                    raise ValueError(f"Missing negative prompt embeddings in cache entry {idx} ({cache_path})")
                param["negative_prompt_embeds"] = cache_entry["negative_prompt_embeds"]
                param["negative_prompt_attention_mask"] = cache_entry["negative_prompt_attention_mask"]

        return sample_params

    def _resolve_default_sample_prompts_cache(self, args: argparse.Namespace) -> str:
        from musubi_tuner.dataset import config_utils
        from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

        if not getattr(args, "dataset_config", None):
            raise ValueError("--dataset_config is required to resolve the sample prompt cache directory")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_LTX2)
        dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        datasets = dataset_group.datasets
        if not datasets:
            raise ValueError("No datasets available to resolve sample prompt cache directory")
        cache_dir = getattr(datasets[0], "cache_directory", None)
        if not cache_dir:
            raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
        return os.path.join(cache_dir, DEFAULT_SAMPLE_PROMPTS_CACHE)

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
            return self._load_precached_sample_prompts(args)

        logger.info("LTX-2 sampling: deferring Gemma encoding until sampling")
        prompts = load_prompts(sample_prompts)
        if not prompts:
            return None

        return self._apply_sample_defaults(args, prompts)

    def _build_text_encoder(self, args: argparse.Namespace, accelerator: Accelerator) -> torch.dtype:
        logger.info("Loading Gemma text encoder for LTX-2 sampling")
        if getattr(args, "gemma_root", None) is None:
            raise ValueError("--gemma_root is required for LTX-2 sample prompts")
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
                gemma_weights_path=getattr(args, "gemma_safetensors", None),
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
        if not is_quantized and accelerator.device.type != "cpu":
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
            if self._ltx_mode == "a":
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
        saved_swap_state = None
        if (getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False)) and getattr(transformer, "blocks_to_swap", 0):
            saved_swap_state = {
                "blocks_to_swap": int(getattr(transformer, "blocks_to_swap", 0) or 0),
                "use_pinned": bool(getattr(args, "use_pinned_memory_for_block_swap", False)),
            }
            logger.info("Sampling: disabling block swap temporarily to avoid FP8 CPU/GPU mismatch")
            base_model = transformer.model if hasattr(transformer, "model") else transformer
            transformer.blocks_to_swap = 0
            transformer.offloader = None
            base_model.blocks_to_swap = 0
            base_model.offloader = None
            if hasattr(transformer, "_ltx2_block_swap"):
                transformer._ltx2_block_swap = None
            if hasattr(base_model, "_ltx2_block_swap"):
                base_model._ltx2_block_swap = None
            base_model.to(accelerator.device)
            transformer.to(accelerator.device)
            try:
                from musubi_tuner.modules.custom_offloading_utils import weighs_to_device as _weights_to_device
                from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import move_fp8_scale_weights as _move_fp8_scales
                blocks = getattr(base_model, "transformer_blocks", None)
                if blocks is not None:
                    for block in blocks:
                        _weights_to_device(block, accelerator.device)
                        _move_fp8_scales(block, accelerator.device)
            except Exception:
                pass
            clean_memory_on_device(accelerator.device)
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
                base_model = transformer.model if hasattr(transformer, "model") else transformer
                base_model.to(accelerator.device)
                transformer.to(accelerator.device)
                clean_memory_on_device(accelerator.device)
            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                try:
                    for param in transformer.parameters():
                        if isinstance(param, torch.Tensor) and param.device != accelerator.device:
                            param.data = param.data.to(device=accelerator.device)
                    for buf in transformer.buffers():
                        if isinstance(buf, torch.Tensor) and buf.device != accelerator.device:
                            buf.data = buf.data.to(device=accelerator.device)
                except Exception:
                    pass
                self._ensure_fp8_buffers_on_device(transformer)
                try:
                    from musubi_tuner.modules.custom_offloading_utils import weighs_to_device as _weights_to_device
                    from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import move_fp8_scale_weights as _move_fp8_scales
                    base_model = transformer.model if hasattr(transformer, "model") else transformer
                    blocks = getattr(base_model, "transformer_blocks", None)
                    if blocks is not None:
                        for block in blocks:
                            _weights_to_device(block, accelerator.device)
                            _move_fp8_scales(block, accelerator.device)
                except Exception:
                    pass

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
            # Check if any prompt needs embeddings
            needs_encoding = any(p.get("prompt_embeds") is None for p in sample_params_list)
            if not needs_encoding:
                return

            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            logger.info("Sampling batch: loaded text encoder for %d prompts", len(sample_params_list))

            for sample_parameter in sample_params_list:
                if sample_parameter.get("prompt_embeds") is not None:
                    continue  # Already has embeddings

                prompt_text = sample_parameter.get("prompt", "")
                prompt_embeds, prompt_mask = self._encode_prompt_text(accelerator, prompt_text, text_encoder_dtype)
                sample_parameter["prompt_embeds"] = prompt_embeds
                sample_parameter["prompt_attention_mask"] = prompt_mask

                negative_prompt = sample_parameter.get("negative_prompt", None)
                if negative_prompt:
                    neg_embeds, neg_mask = self._encode_prompt_text(accelerator, negative_prompt, text_encoder_dtype)
                    sample_parameter["negative_prompt_embeds"] = neg_embeds
                    sample_parameter["negative_prompt_attention_mask"] = neg_mask

            self._cleanup_text_encoder(accelerator)
            logger.info("Sampling batch: unloaded text encoder after encoding all prompts")
            self._cleanup_cuda(accelerator.device)

        try:
            if distributed_state.num_processes <= 1:
                # Batch encode all prompts upfront when offloading is enabled
                if transformer_offloaded:
                    offload_transformer_if_needed()
                    prepare_all_embeddings_batch(sample_parameters)

                with torch.no_grad(), accelerator.autocast():
                    for sample_parameter in sample_parameters:
                        if transformer_offloaded:
                            offload_transformer_if_needed()
                            vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                            logger.info("Sampling offload: loading VAE for sampling")
                            vae_for_sampling = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)
                            ensure_transformer_on_device()
                            self.sample_image_inference(
                                accelerator, args, transformer, dit_dtype, vae_for_sampling, save_dir, sample_parameter, epoch, steps
                            )
                            offload_transformer_if_needed()
                            vae_for_sampling.to_device("cpu")
                            logger.info("Sampling offload: moved VAE back to CPU after sampling")
                            self._cleanup_cuda(accelerator.device)
                        else:
                            self.sample_image_inference(
                                accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                            )
                        clean_memory_on_device(accelerator.device)
                        self._cleanup_cuda(accelerator.device)

                # Cleanup embeddings after all samples are done
                if transformer_offloaded:
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

                        for sample_parameter in my_sample_params:
                            if transformer_offloaded:
                                offload_transformer_if_needed()
                                vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                                logger.info("Sampling offload: loading VAE for sampling")
                                vae_for_sampling = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)
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
                                )
                                offload_transformer_if_needed()
                                vae_for_sampling.to_device("cpu")
                                logger.info("Sampling offload: moved VAE back to CPU after sampling")
                                self._cleanup_cuda(accelerator.device)
                            else:
                                self.sample_image_inference(
                                    accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                                )
                            self._cleanup_cuda(accelerator.device)

                        # Cleanup embeddings after all samples for this process
                        if transformer_offloaded:
                            for sample_parameter in my_sample_params:
                                cleanup_embeddings(sample_parameter)
        finally:
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

            if saved_swap_state is not None:
                transformer.enable_block_swap(
                    saved_swap_state["blocks_to_swap"],
                    accelerator.device,
                    supports_backward=True,
                    use_pinned_memory=saved_swap_state["use_pinned"],
                )

            transformer.switch_block_swap_for_training()
            # Ensure block-swap layout is re-applied after sampling to avoid VRAM creep.
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            self._cleanup_cuda(accelerator.device)

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
    ):
        return ltx2_inference.sample_image_inference(
            self,
            accelerator,
            args,
            transformer,
            dit_dtype,
            vae,
            save_dir,
            sample_parameter,
            epoch,
            steps,
        )

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
    ):
        return ltx2_inference.do_inference(
            self,
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
            image_path=image_path,
            control_video_path=control_video_path,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            offload_transformer_for_decode=offload_transformer_for_decode,
            transformer_offload_device=transformer_offload_device,
            restore_transformer_device=restore_transformer_device,
            audio_output_path=audio_output_path,
            use_audio_subprocess=use_audio_subprocess,
            enable_audio_preview=enable_audio_preview,
            decode_video=decode_video,
            audio_only=audio_only,
        )

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
        help="Optional Gemma weights .safetensors file for sampling (tokenizer/config still from --gemma_root).",
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
        "--ltx2_mode",
        type=str,
        default="auto",
        choices=["v", "av", "a", "auto"],
        help="Training modality: v=video, a=audio-only, av=audio+video, auto=detect from datasets.",
    )
    parser.add_argument(
        "--split_av_passes",
        action="store_true",
        help="When using --ltx2_mode av, run separate video-only and audio-only passes per batch (lower peak activations, slower).",
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
        choices=["t2v", "v2v", "audio", "full"],
        help=(
            "LoRA target preset: "
            "'t2v' = text-to-video (attention only, official default), "
            "'v2v' = video-to-video/IC-LoRA (attention + feed-forward), "
            "'audio' = audio-only (audio attn/ffn + audio-side cross-modal), "
            "'full' = all linear layers. "
            "Can be overridden by --network_args include_patterns=..."
        ),
    )
    parser.add_argument(
        "--separate_audio_buckets",
        action="store_true",
        default=None,
        help="Split LTX-2 buckets by audio presence to avoid mixed audio/non-audio batches.",
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
        "--sample_two_stage",
        action="store_true",
        help="Enable two-stage sampling (half-res pass + upsample + refinement). Requires --sample_upsampler.",
    )
    parser.add_argument(
        "--sample_upsampler",
        type=str,
        default=None,
        help="Path to LTX-2 spatial upsampler checkpoint for two-stage sampling.",
    )
    parser.add_argument(
        "--sample_distilled_lora",
        type=str,
        default=None,
        help="Optional distilled LoRA for stage-2 refinement in two-stage sampling.",
    )
    parser.add_argument(
        "--sample_distilled_lora_strength",
        type=float,
        default=1.0,
        help="Strength for --sample_distilled_lora (default: 1.0).",
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
        "--sample_merge_audio",
        action="store_true",
        help="Mux sample audio into the sample video (outputs *_av.mp4).",
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
             "By default, a *_comfy.safetensors file is created alongside the original.",
    )

    return parser


# ======== Main training entry point ========


def main() -> None:
    """Main training entry point"""
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)
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

    explicit_lora_preset = any(
        arg == "--lora_target_preset" or arg.startswith("--lora_target_preset=") for arg in sys.argv
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

    # Inject lora_target_preset into network_args (LTX-2 specific)
    if getattr(args, "ltx_mode", "v") == "a" and not explicit_lora_preset:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "audio"

    lora_target_preset = getattr(args, "lora_target_preset", None)
    if lora_target_preset is not None:
        if args.network_args is None:
            args.network_args = []
        # Only add if not already specified in network_args
        if not any(arg.startswith("lora_target_preset=") for arg in args.network_args):
            args.network_args.append(f"lora_target_preset={lora_target_preset}")
            logger.info(f"Using LoRA target preset: {lora_target_preset}")

    trainer = LTX2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()

