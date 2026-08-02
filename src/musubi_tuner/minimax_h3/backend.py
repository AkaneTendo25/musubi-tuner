from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from musubi_tuner.minimax_h3.request import H3GenerationRequest
from musubi_tuner.minimax_h3.training import H3ModelPrediction, H3TrainingMode


class H3LatentEncoder(Protocol):
    def encode_latents(self, batch: list[Any]) -> Any: ...


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


def create_latent_encoder(*, model: Path, device: str | None, dtype: str) -> H3LatentEncoder:
    """Load only the video and audio VAEs required for latent caching."""
    _validate_dtype(dtype)
    from musubi_tuner.minimax_h3.integration import create_latent_encoder as create_integrated_latent_encoder

    return create_integrated_latent_encoder(model=model, device=device, dtype=dtype)


def create_conditioning_encoder(*, model: Path, device: str | None, dtype: str) -> H3ConditioningEncoder:
    """Load only the understanding encoder required for conditioning caches."""
    _validate_dtype(dtype)
    from musubi_tuner.minimax_h3.integration import create_conditioning_encoder as create_integrated_conditioning_encoder

    return create_integrated_conditioning_encoder(model=model, device=device, dtype=dtype)


def create_generator(*, model: Path, device: str | None, dtype: str, request: H3GenerationRequest) -> H3Generator:
    """Load only the inference variant and components required by the request."""
    _validate_dtype(dtype)
    from musubi_tuner.minimax_h3.integration import create_generator as create_integrated_generator

    return create_integrated_generator(model=model, device=device, dtype=dtype, request=request)


def create_training_backend(
    *,
    model: Path,
    device: str | None,
    dtype: str,
    mode: H3TrainingMode,
    attention_mode: str,
    split_attention: bool,
) -> H3TrainingBackend:
    """Load only the transformer required for cache-backed LoRA training.

    Upstream source is copied under ``vendor/official`` and adapted by the
    Musubi-owned ``integration`` module. Checkpoints never select Python code.
    """
    _validate_dtype(dtype)
    if attention_mode not in _SUPPORTED_ATTENTION_MODES:
        raise ValueError("MiniMax H3 currently supports only Musubi's --sdpa attention path")
    if split_attention:
        raise ValueError("MiniMax H3 split attention is not supported until the released transformer forward is validated")
    from musubi_tuner.minimax_h3.integration import create_training_backend as create_integrated_training_backend

    return create_integrated_training_backend(
        model=model,
        device=device,
        dtype=dtype,
        mode=mode,
        attention_mode=attention_mode,
        split_attention=split_attention,
    )
