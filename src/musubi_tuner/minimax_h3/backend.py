from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from musubi_tuner.minimax_h3.request import H3GenerationRequest
from musubi_tuner.minimax_h3.training import H3ModelPrediction, H3TrainingMode


class H3LatentEncoder(Protocol):
    def encode_latents(self, batch: list[Any]) -> Any:
        """Encode image or joint AV caches; image caches contain no audio tensors."""
        ...


class H3ConditioningEncoder(Protocol):
    conditioning_requires_content: bool

    def encode_conditioning(self, batch: list[Any], *, include_empty: bool = False) -> Any: ...


class H3Generator(Protocol):
    def generate(self, request: H3GenerationRequest) -> None: ...


class H3TrainingBackend(Protocol):
    def get_training_transformer(self) -> torch.nn.Module: ...

    def predict_training(
        self,
        transformer: torch.nn.Module,
        batch: dict[str, Any],
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        *,
        conditioning: Literal["prompt", "empty"] = "prompt",
    ) -> H3ModelPrediction: ...


class H3BackendUnavailableError(RuntimeError):
    pass


_SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}
_SUPPORTED_ATTENTION_MODES = {"torch"}


def _validate_dtype(dtype: str) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported H3 compute dtype: {dtype}")


def create_latent_encoder(
    *,
    video_vae: Path,
    audio_vae: Path | None,
    device: str | None,
    dtype: str,
) -> H3LatentEncoder:
    """Load the video VAE and, for video datasets, the audio VAE used by latent caching."""
    _validate_dtype(dtype)
    from musubi_tuner.minimax_h3.integration import create_latent_encoder as create_integrated_latent_encoder

    return create_integrated_latent_encoder(
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
    )


def create_conditioning_encoder(
    *,
    text_encoder: Path,
    tokenizer: Path,
    task: str,
    device: str | None,
    dtype: str,
    quantization: Literal["none", "int8", "nf4", "nvfp4_awq"] = "none",
) -> H3ConditioningEncoder:
    """Load only the understanding encoder required for conditioning caches."""
    _validate_dtype(dtype)
    from musubi_tuner.minimax_h3.integration import create_conditioning_encoder as create_integrated_conditioning_encoder

    return create_integrated_conditioning_encoder(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        task=task,
        device=device,
        dtype=dtype,
        quantization=quantization,
    )


def create_generator(
    *,
    model: Path,
    text_encoder: Path,
    tokenizer: Path,
    video_vae: Path,
    audio_vae: Path,
    device: str | None,
    dtype: str,
    request: H3GenerationRequest,
    num_inference_steps: int = 20,
    height: int | None = None,
    width: int | None = None,
    fp8_scaled: bool = False,
    int8_convrot: bool = False,
    text_encoder_quantization: Literal["none", "int8", "nf4", "nvfp4_awq"] = "none",
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    block_swap_ring_size: int = 2,
    block_swap_granularity: Literal["block", "layer"] = "block",
    use_pinned_memory_for_block_swap: bool = False,
    lora_weights: tuple[Path, ...] = (),
    lora_multipliers: tuple[float, ...] = (),
) -> H3Generator:
    """Load only the inference variant and components required by the request."""
    _validate_dtype(dtype)
    from musubi_tuner.minimax_h3.integration import create_generator as create_integrated_generator

    return create_integrated_generator(
        model=model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
        request=request,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        fp8_scaled=fp8_scaled,
        int8_convrot=int8_convrot,
        text_encoder_quantization=text_encoder_quantization,
        blocks_to_swap=blocks_to_swap,
        block_swap_h2d_only=block_swap_h2d_only,
        block_swap_ring_size=block_swap_ring_size,
        block_swap_granularity=block_swap_granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
        lora_weights=lora_weights,
        lora_multipliers=lora_multipliers,
    )


def create_training_backend(
    *,
    model: Path,
    device: str | None,
    dtype: str,
    mode: H3TrainingMode,
    attention_mode: str,
    split_attention: bool,
    fp8_scaled: bool = False,
    quantization_device: str | None = None,
    int8_convrot: bool = False,
) -> H3TrainingBackend:
    """Load only the transformer required for cache-backed LoRA training.

    The native implementation is adapted through the Musubi-owned
    ``integration`` module. Checkpoints never select Python code.
    """
    _validate_dtype(dtype)
    if attention_mode not in _SUPPORTED_ATTENTION_MODES:
        raise ValueError("MiniMax H3 supports only Musubi's --sdpa attention path")
    if split_attention:
        raise ValueError("MiniMax H3 does not support split attention")
    from musubi_tuner.minimax_h3.integration import create_training_backend as create_integrated_training_backend

    return create_integrated_training_backend(
        model=model,
        device=device,
        dtype=dtype,
        mode=mode,
        attention_mode=attention_mode,
        split_attention=split_attention,
        fp8_scaled=fp8_scaled,
        quantization_device=quantization_device,
        int8_convrot=int8_convrot,
    )
