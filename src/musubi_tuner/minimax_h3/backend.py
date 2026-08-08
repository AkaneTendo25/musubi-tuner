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
        # Extension context. Only forwarded when non-zero, so a backend that
        # does not support it may omit these parameters entirely.
        extension_video_frames: int = 0,
        extension_audio_latents: int = 0,
    ) -> H3ModelPrediction: ...


class H3BackendUnavailableError(RuntimeError):
    pass


_SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}
_SUPPORTED_ATTENTION_MODES = {"torch", "flash", "flash3"}


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
    adaln_rank: int | None = None,
    text_encoder_quantization: Literal["none", "int8", "nf4", "nvfp4_awq"] = "none",
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    block_swap_ring_size: int = 2,
    block_swap_granularity: Literal["block", "layer"] = "block",
    use_pinned_memory_for_block_swap: bool = False,
    lora_weights: tuple[Path, ...] = (),
    lora_multipliers: tuple[float, ...] = (),
    compile_model: bool = False,
    compile_backend: str = "inductor",
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_dynamic: str | None = None,
    compile_fullgraph: bool = False,
    compile_cache_size_limit: int | None = None,
    compile_auto_cache_size_limit: bool = False,
    compile_fallback_to_eager: bool = False,
    inductor_config: tuple[str, ...] = (),
    fused_qk_norm_rope: bool = False,
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
        adaln_rank=adaln_rank,
        text_encoder_quantization=text_encoder_quantization,
        blocks_to_swap=blocks_to_swap,
        block_swap_h2d_only=block_swap_h2d_only,
        block_swap_ring_size=block_swap_ring_size,
        block_swap_granularity=block_swap_granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
        lora_weights=lora_weights,
        lora_multipliers=lora_multipliers,
        compile_model=compile_model,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_fullgraph=compile_fullgraph,
        compile_cache_size_limit=compile_cache_size_limit,
        compile_auto_cache_size_limit=compile_auto_cache_size_limit,
        compile_fallback_to_eager=compile_fallback_to_eager,
        inductor_config=inductor_config,
        fused_qk_norm_rope=fused_qk_norm_rope,
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
    adaln_rank: int | None = None,
    fp8_quantization_mode: str = "block",
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    convrot_int8_fwd: str = "int8",
    target_device: str | None = None,
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    low_ram_load: bool = True,
    base_lora_weights: list[dict[str, torch.Tensor]] | None = None,
    base_lora_multipliers: list[float] | None = None,
) -> H3TrainingBackend:
    """Load only the transformer required for cache-backed LoRA training.

    The native implementation is adapted through the Musubi-owned
    ``integration`` module. Checkpoints never select Python code.
    """
    _validate_dtype(dtype)
    if attention_mode not in _SUPPORTED_ATTENTION_MODES:
        raise ValueError("MiniMax H3 supports only --sdpa, --flash_attn, or --flash3")
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
        adaln_rank=adaln_rank,
        # Forwarded only when they differ from the released defaults, so the
        # wrapper still routes exactly what it was given.
        **({} if fp8_quantization_mode == "block" else {"fp8_quantization_mode": fp8_quantization_mode}),
        **(
            {}
            if not convrot_int8
            else {"convrot_int8": True, "convrot_int8_bwd": convrot_int8_bwd, "convrot_int8_fwd": convrot_int8_fwd}
        ),
        **({} if target_device is None else {"target_device": target_device}),
        **({} if blocks_to_swap == 0 else {"blocks_to_swap": blocks_to_swap}),
        **({} if not block_swap_h2d_only else {"block_swap_h2d_only": True}),
        **({} if low_ram_load else {"low_ram_load": False}),
        **(
            {}
            if not base_lora_weights
            else {"base_lora_weights": base_lora_weights, "base_lora_multipliers": base_lora_multipliers}
        ),
    )
