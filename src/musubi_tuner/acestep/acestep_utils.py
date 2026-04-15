"""
ACE-Step model loading utilities.

Functions for loading ACE-Step models, VAE, and text encoder.
"""

import json
import os
from typing import Optional, Tuple, Dict, Any, List, Mapping, Sequence

import glob

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .acestep_config import (
    TURBO_SHIFT3_TIMESTEPS,
    TEXT_ENCODER_MAX_LENGTH,
    SFT_GEN_PROMPT,
    ACESTEP_SAMPLE_RATE,
    DEFAULT_DIT_INSTRUCTION,
    ACESTEP_FP8_TARGET_KEYS,
    ACESTEP_FP8_EXCLUDE_KEYS,
    ACESTEP_TEXT_CACHE_SCHEMA_VERSION,
)

import logging

logger = logging.getLogger(__name__)

ACESTEP_MODEL_SIGNATURE_KEYS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "encoder_hidden_size",
    "encoder_num_hidden_layers",
    "encoder_num_attention_heads",
    "encoder_intermediate_size",
    "text_hidden_dim",
)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def build_acestep_model_arch_signature(config: Any) -> str:
    """Build a stable architecture signature for ACE-Step family checkpoints."""
    signature = {}
    for key in ACESTEP_MODEL_SIGNATURE_KEYS:
        value = _get_config_value(config, key, None)
        if value is not None:
            signature[key] = value
    if "model_type" not in signature:
        signature["model_type"] = type(config).__name__
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def build_acestep_text_cache_metadata(
    cache_kind: str,
    text_encoder_path: Optional[str] = None,
    model_config: Optional[Any] = None,
    model_path: Optional[str] = None,
) -> Dict[str, str]:
    """Build metadata stored in ACE-Step text caches."""
    metadata = {
        "acestep_text_cache_schema": ACESTEP_TEXT_CACHE_SCHEMA_VERSION,
        "acestep_cache_kind": cache_kind,
        "acestep_text_encoder_source": text_encoder_path or "",
    }
    if model_config is not None:
        metadata["acestep_dit_arch_signature"] = build_acestep_model_arch_signature(model_config)
    if model_path is not None:
        metadata["acestep_dit_source"] = model_path
    return metadata


def validate_acestep_text_cache(
    metadata: Optional[Mapping[str, str]],
    tensor_keys: Sequence[str],
    expected_dit_arch_signature: Optional[str],
    allow_legacy: bool = False,
) -> None:
    """Validate ACE-Step text cache metadata against the current training setup."""
    metadata = dict(metadata or {})
    key_set = set(tensor_keys)

    schema = metadata.get("acestep_text_cache_schema")
    if schema is None:
        if allow_legacy:
            return
        raise ValueError(
            "legacy ACE-Step text cache detected (missing acestep_text_cache_schema). "
            "Rebuild text caches with the updated acestep_cache_text_encoder_outputs.py "
            "or pass --acestep_allow_legacy_text_cache."
        )
    if schema != ACESTEP_TEXT_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ACE-Step text cache schema '{schema}' "
            f"(expected {ACESTEP_TEXT_CACHE_SCHEMA_VERSION})"
        )

    cache_kind = metadata.get("acestep_cache_kind")
    if cache_kind not in ("raw", "conditioned"):
        raise ValueError(f"unknown ACE-Step cache kind '{cache_kind}'")

    if cache_kind == "raw":
        has_lyric_branch = any(key.startswith("lyric_hidden_states_") for key in key_set) and "lyric_attention_mask" in key_set
        if not has_lyric_branch:
            raise ValueError(
                "raw ACE-Step text cache is missing lyric branch tensors. "
                "Rebuild text caches with the updated acestep_cache_text_encoder_outputs.py."
            )
        return

    cache_signature = metadata.get("acestep_dit_arch_signature")
    if cache_signature is None:
        raise ValueError("conditioned ACE-Step text cache is missing acestep_dit_arch_signature")
    if expected_dit_arch_signature is not None and cache_signature != expected_dit_arch_signature:
        raise ValueError(
            "conditioned ACE-Step text cache was created for a different model architecture. "
            "Rebuild text caches with the same --dit checkpoint family used for training."
        )


def _find_safetensors_file(model_path: str) -> str:
    """Find the safetensors file in model directory.

    Returns a single file path. For split weights (model-00001-of-00002.safetensors),
    returns the first file; load_safetensors_with_lora_and_fp8 handles split detection.
    """
    # Check for single model.safetensors
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return single
    # Check for split or other safetensors files
    pattern = os.path.join(model_path, "*.safetensors")
    files = sorted(glob.glob(pattern))
    if files:
        return files[0]
    raise FileNotFoundError(f"No safetensors files found in {model_path}")


def load_acestep_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "sdpa",
    fp8_scaled: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load ACE-Step model from checkpoint.

    Uses split architecture/weights loading to support FP8 quantization during loading,
    avoiding bf16 VRAM peak. Follows the same pattern as HunyuanVideo 1.5 model loading.

    Args:
        model_path: Path to ACE-Step checkpoint directory
        device: Device to load model on
        dtype: Model dtype (bfloat16 recommended)
        attn_mode: Attention implementation ("sdpa", "flash_attention_2", "eager")
        fp8_scaled: Whether to use scaled FP8 quantization (per-block with scales)
        dit_weight_dtype: Weight dtype override. float8_e4m3fn for fp8_base, None for fp8_scaled, bf16 for normal.

    Returns:
        Tuple of (model, config_dict)
        config_dict includes 'silence_latent' if found in checkpoint
    """
    # Map attn_mode to valid transformers attn_implementation values
    attn_map = {
        "flash": "flash_attention_2",
        "flash_attn": "flash_attention_2",
        "flash_attention_2": "flash_attention_2",
        "sdpa": "sdpa",
        "torch": "sdpa",  # torch -> sdpa
        "xformers": "sdpa",  # xformers -> sdpa
        "eager": "eager",
        None: "sdpa",
    }
    attn_impl = attn_map.get(attn_mode, "sdpa")

    logger.info(f"Loading ACE-Step model from {model_path}")

    # Step 1: Load config only (no weights)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Step 2: Create empty model on meta device (zero memory)
    with init_empty_weights():
        model = AutoModel.from_config(
            config,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

    # Step 3: Find safetensors file (split detection handled by loader)
    model_file = _find_safetensors_file(model_path)

    # Step 4: Load weights (with optional FP8 quantization during loading)
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    sd = load_safetensors_with_lora_and_fp8(
        model_files=model_file,
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=fp8_scaled,
        calc_device=torch.device(device),
        move_to_device=True,
        dit_weight_dtype=None if fp8_scaled else (dit_weight_dtype or dtype),
        target_keys=ACESTEP_FP8_TARGET_KEYS if fp8_scaled else None,
        exclude_keys=ACESTEP_FP8_EXCLUDE_KEYS if fp8_scaled else None,
    )

    # Step 5: Apply FP8 monkey patch (scaled mode only)
    if fp8_scaled:
        from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        # Keep strict loading while tolerating FP8 scales for non-Linear layers.
        # ACE-Step has conv weights like decoder.proj_in.1/proj_out.1 that can be
        # targeted by broad patterns, but only monkey-patched Linear layers
        # should carry scale_weight buffers.
        expected_keys = set(model.state_dict().keys())
        unexpected_scale_keys = [k for k in sd.keys() if k.endswith(".scale_weight") and k not in expected_keys]
        if unexpected_scale_keys:
            logger.warning(
                f"Dropping {len(unexpected_scale_keys)} unexpected FP8 scale_weight key(s), "
                f"example: {unexpected_scale_keys[:2]}"
            )
            for key in unexpected_scale_keys:
                weight_key = key.replace(".scale_weight", ".weight")
                # If this weight got quantized but the module was not monkey-patched,
                # dequantize back to its original numeric range before loading.
                if weight_key in sd and isinstance(sd[weight_key], torch.Tensor):
                    weight = sd[weight_key]
                    scale = sd[key]
                    if weight.ndim == 2:
                        # handled by monkey-patched Linear path; skip defensive dequant
                        pass
                    elif scale.ndim < 3:
                        sd[weight_key] = (weight.to(torch.float32) * scale.to(torch.float32)).to(scale.dtype)
                    else:
                        out_features, num_blocks, _ = scale.shape
                        deq = weight.to(torch.float32).contiguous().view(out_features, num_blocks, -1)
                        deq = (deq * scale.to(torch.float32)).view(weight.shape)
                        sd[weight_key] = deq.to(scale.dtype)
                sd.pop(key, None)

    # Step 6: Load weights into model
    model.load_state_dict(sd, strict=True, assign=True)
    model.to(device)
    model.eval()

    # Extract config info
    config_dict = {
        "is_turbo": getattr(config, "is_turbo", True),
        "hidden_size": getattr(config, "hidden_size", 1024),
        "encoder_hidden_size": getattr(config, "encoder_hidden_size", getattr(config, "hidden_size", 1024)),
        "timestep_mu": getattr(config, "timestep_mu", -0.4),
        "timestep_sigma": getattr(config, "timestep_sigma", 1.0),
        "arch_signature": build_acestep_model_arch_signature(config),
    }

    # Load silence_latent from separate .pt file (following official ACE-Step handler)
    # silence_latent is NOT a model attribute - it's stored separately
    silence_latent_path = os.path.join(model_path, "silence_latent.pt")
    if os.path.exists(silence_latent_path):
        logger.info(f"Loading silence_latent from {silence_latent_path}")
        # Official handler loads and transposes: torch.load(path).transpose(1, 2)
        # Original shape: [1, 64, T] -> transposed: [1, T, 64]
        silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
        silence_latent = silence_latent.to(device=device, dtype=dtype)
        config_dict["silence_latent"] = silence_latent
        logger.info(f"Loaded silence_latent with shape {silence_latent.shape}")
    else:
        logger.warning(f"silence_latent.pt not found at {silence_latent_path}, will use zeros as fallback")
        config_dict["silence_latent"] = None

    logger.info(
        "ACE-Step model loaded: "
        f"is_turbo={config_dict['is_turbo']}, "
        f"encoder_hidden_size={config_dict['encoder_hidden_size']}, "
        f"hidden_size={config_dict['hidden_size']}"
    )

    return model, config_dict


def load_acestep_vae(
    vae_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load AutoencoderOobleck VAE for audio latent encoding.

    Args:
        vae_path: Path to VAE checkpoint or HuggingFace model ID
        device: Device to load on
        dtype: VAE dtype

    Returns:
        AutoencoderOobleck instance
    """
    from diffusers.models import AutoencoderOobleck

    logger.info(f"Loading ACE-Step VAE from {vae_path}")
    vae = AutoencoderOobleck.from_pretrained(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()

    return vae


def load_text_encoder(
    text_encoder_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, nn.Module]:
    """Load an ACE-Step-compatible text encoder.

    Args:
        text_encoder_path: Path to text encoder checkpoint
        device: Device
        dtype: Model dtype

    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading text encoder from {text_encoder_path}")

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        text_encoder_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    return tokenizer, model


def format_text_for_acestep(
    caption: str,
    lyrics: str = "",
    instruction: str = DEFAULT_DIT_INSTRUCTION,
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    time_signature: Optional[int] = None,
    duration: Optional[float] = None,
) -> str:
    """Format text using ACE-Step's SFT format.

    Must match the exact format used by the original ACE-Step trainer for proper conditioning.

    Args:
        caption: Music description/prompt
        lyrics: Song lyrics (optional)
        instruction: Task instruction (default matches original ACE-Step)
        bpm: Beats per minute (optional)
        key: Musical key (optional)
        time_signature: Time signature (optional)
        duration: Duration in seconds (optional)

    Returns:
        Formatted text string
    """
    # Build metas string - MUST include all fields like original trainer
    # Original uses "N/A" for missing values, always includes all fields
    bpm_str = str(bpm) if bpm is not None else "N/A"
    timesig_str = str(time_signature) if time_signature is not None else "N/A"
    key_str = key if key is not None else "N/A"
    duration_str = f"{duration:.1f} seconds" if duration is not None else "N/A"

    metas_str = (
        f"- bpm: {bpm_str}\n"
        f"- timesignature: {timesig_str}\n"
        f"- keyscale: {key_str}\n"
        f"- duration: {duration_str}"
    )

    # Format using SFT template
    formatted = SFT_GEN_PROMPT.format(instruction, caption, metas_str)

    # Add lyrics if present
    if lyrics and lyrics.strip() and lyrics.strip().lower() != "[instrumental]":
        formatted += f"\n\n# Lyrics\n{lyrics}"
    else:
        formatted += "\n\n# Lyrics\n[Instrumental]"

    return formatted


def format_lyrics_for_acestep(
    lyrics: str,
    language: str = "unknown",
) -> str:
    """Format lyrics using ACE-Step's format for lyric branch encoding.

    The lyrics are encoded separately through embed_tokens (not the full text encoder),
    and must be in this specific format with language header and endoftext token.

    Args:
        lyrics: Song lyrics (or "[Instrumental]" for instrumental tracks)
        language: Language code (e.g., "en", "zh", "ja", "unknown")

    Returns:
        Formatted lyrics string for tokenization
    """
    if not lyrics or not lyrics.strip() or lyrics.strip().lower() == "[instrumental]":
        lyrics = "[Instrumental]"
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def encode_text_for_acestep(
    tokenizer,
    text_encoder: nn.Module,
    caption: str,
    lyrics: str = "",
    max_length: int = TEXT_ENCODER_MAX_LENGTH,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    **format_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text using ACE-Step's format.

    Args:
        tokenizer: ACE-Step-compatible tokenizer
        text_encoder: ACE-Step-compatible text encoder
        caption: Music description/prompt
        lyrics: Song lyrics (optional)
        max_length: Maximum token length
        device: Device for outputs
        dtype: Output dtype
        **format_kwargs: Additional arguments for format_text_for_acestep

    Returns:
        Tuple of (encoder_hidden_states, encoder_attention_mask)
    """
    # Format text
    formatted = format_text_for_acestep(caption, lyrics, **format_kwargs)

    # Tokenize
    inputs = tokenizer(
        formatted,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Encode
    with torch.no_grad():
        outputs = text_encoder(input_ids)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

    hidden_states = hidden_states.to(dtype)

    return hidden_states, attention_mask


def sample_logit_normal_timestep(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample timesteps from logit-normal distribution (original ACE-Step training).

    Matches the official ACE-Step sample_t_r() with use_meanflow=False exactly:
        t1 = sigmoid(randn * sigma + mu)
        t2 = sigmoid(randn * sigma + mu)
        t = max(t1, t2)   # order statistic shifts distribution toward higher t
        r = t              # use_meanflow=False forces r = t

    Args:
        batch_size: Number of samples
        device: Device for output
        dtype: Output dtype
        timestep_mu: Mean for logit-normal sampling (default: -0.4 from ACE-Step config)
        timestep_sigma: Std for logit-normal sampling (default: 1.0 from ACE-Step config)

    Returns:
        Tuple of (t, r) timestep tensors (both same value, matching use_meanflow=False)
    """
    t1 = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu)
    t2 = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu)
    t = torch.maximum(t1, t2)
    r = t
    return t, r


def sample_discrete_timestep(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    timestep_schedule: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample timesteps from a discrete schedule.

    Randomly samples from the provided discrete timestep schedule,
    defaulting to TURBO_SHIFT3_TIMESTEPS if none is provided.

    Args:
        batch_size: Number of samples
        device: Device for output
        dtype: Output dtype
        timestep_schedule: List of discrete timestep values to sample from.
            If None, uses TURBO_SHIFT3_TIMESTEPS (shift=3.0, 8 steps).

    Returns:
        Tuple of (t, r) timestep tensors (both same value)
    """
    if timestep_schedule is None:
        timestep_schedule = TURBO_SHIFT3_TIMESTEPS

    # Randomly select indices for each sample in batch
    indices = torch.randint(0, len(timestep_schedule), (batch_size,), device=device)

    # Convert to tensor and index
    timesteps_tensor = torch.tensor(timestep_schedule, device=device, dtype=dtype)
    t = timesteps_tensor[indices]

    # r = t
    r = t

    return t, r


def get_silence_latent(
    silence_latent: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    latent_length: int = 750,
) -> torch.Tensor:
    """Get or create silence latent tensor.

    Args:
        silence_latent: Pre-loaded silence latent from config_dict, or None
        device: Target device
        dtype: Target dtype
        latent_length: Required length in time dimension (default 750 = 30 seconds at 25Hz)

    Returns:
        Silence latent tensor of shape [1, latent_length, 64]
    """
    if silence_latent is not None:
        silence = silence_latent.to(device=device, dtype=dtype)
        # Pad or truncate to required length
        if silence.shape[1] < latent_length:
            pad_len = latent_length - silence.shape[1]
            silence = torch.cat([
                silence,
                torch.zeros(1, pad_len, 64, device=device, dtype=dtype)
            ], dim=1)
        silence = silence[:, :latent_length, :]
        return silence
    else:
        # Return zeros as fallback
        logger.warning("silence_latent not provided, using zeros")
        return torch.zeros(1, latent_length, 64, device=device, dtype=dtype)


def compute_flow_matching_loss(
    model_pred: torch.Tensor,
    noise: torch.Tensor,
    latents: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute flow matching loss.

    Flow matching predicts the velocity field v = x1 - x0 (noise - data).

    Args:
        model_pred: Model prediction
        noise: Sampled noise (x1)
        latents: Target latents (x0)
        attention_mask: Optional mask for valid positions

    Returns:
        Loss tensor
    """
    # Flow matching target: v = x1 - x0
    target = noise - latents

    # MSE loss
    if attention_mask is not None:
        # Mask out invalid positions
        mask = attention_mask.unsqueeze(-1).float()
        loss = ((model_pred - target) ** 2 * mask).sum() / mask.sum()
    else:
        loss = torch.nn.functional.mse_loss(model_pred, target)

    return loss
