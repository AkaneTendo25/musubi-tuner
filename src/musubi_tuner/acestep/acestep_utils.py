"""
ACE-Step 1.5 model loading utilities.

Functions for loading ACE-Step models, VAE, and text encoder.
"""

import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .acestep_config import (
    TURBO_SHIFT3_TIMESTEPS,
    TEXT_ENCODER_MAX_LENGTH,
    SFT_GEN_PROMPT,
    ACESTEP_SAMPLE_RATE,
)

import logging

logger = logging.getLogger(__name__)


def load_acestep_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "sdpa",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load ACE-Step model from checkpoint.

    Args:
        model_path: Path to ACE-Step checkpoint directory
        device: Device to load model on
        dtype: Model dtype (bfloat16 recommended)
        attn_mode: Attention implementation ("sdpa", "flash_attention_2", "eager")

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
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )

    model = model.to(device)
    model.eval()

    # Extract config info
    config_dict = {
        "is_turbo": getattr(model.config, "is_turbo", True),
        "hidden_size": getattr(model.config, "hidden_size", 1024),
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

    logger.info(f"ACE-Step model loaded: is_turbo={config_dict['is_turbo']}")

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
    """Load Qwen3-Embedding text encoder.

    Args:
        text_encoder_path: Path to Qwen3 checkpoint
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
    instruction: str = "Generate music based on the description.",
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    time_signature: Optional[int] = None,
    duration: Optional[float] = None,
) -> str:
    """Format text using ACE-Step's SFT format.

    Args:
        caption: Music description/prompt
        lyrics: Song lyrics (optional)
        instruction: Task instruction
        bpm: Beats per minute (optional)
        key: Musical key (optional)
        time_signature: Time signature (optional)
        duration: Duration in seconds (optional)

    Returns:
        Formatted text string
    """
    # Build metas string
    metas_parts = []
    if bpm is not None:
        metas_parts.append(f"- bpm: {bpm}")
    if time_signature is not None:
        metas_parts.append(f"- timesignature: {time_signature}")
    if key is not None:
        metas_parts.append(f"- keyscale: {key}")
    if duration is not None:
        metas_parts.append(f"- duration: {duration:.1f} seconds")

    metas_str = "\n".join(metas_parts) if metas_parts else "- No specific metadata"

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
        tokenizer: Qwen3 tokenizer
        text_encoder: Qwen3 text encoder
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


def sample_discrete_timestep(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample timesteps from discrete turbo shift=3 schedule.

    For turbo model training, randomly samples from 8 discrete timesteps.

    Args:
        batch_size: Number of samples
        device: Device for output
        dtype: Output dtype

    Returns:
        Tuple of (t, r) timestep tensors (both same value for turbo)
    """
    # Randomly select indices for each sample in batch
    indices = torch.randint(0, len(TURBO_SHIFT3_TIMESTEPS), (batch_size,), device=device)

    # Convert to tensor and index
    timesteps_tensor = torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=dtype)
    t = timesteps_tensor[indices]

    # r = t for turbo model
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
