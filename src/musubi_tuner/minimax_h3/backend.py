from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from musubi_tuner.minimax_h3.load_options import H3LoadOptions
from musubi_tuner.minimax_h3.request import H3GenerationRequest
from musubi_tuner.minimax_h3.training import H3ModelPrediction, H3TrainingMode


class H3Backend(Protocol):
    def encode_latents(self, batch: list[Any]) -> Any: ...

    def encode_conditioning(self, batch: list[Any], *, include_empty: bool = False) -> Any: ...

    def generate(self, request: H3GenerationRequest) -> None: ...

    def get_training_transformer(self, mode: H3TrainingMode = "t2va") -> torch.nn.Module: ...

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


def create_backend(
    *,
    model: Path,
    device: str | None,
    dtype: str | None = None,
    load_options: H3LoadOptions | None = None,
) -> H3Backend:
    """Create the architecture-owned backend used by Musubi's model scripts.

    Upstream source is copied under ``vendor/official`` and adapted by the
    Musubi-owned ``integration`` module. Checkpoints never select Python code.
    """
    if load_options is None:
        load_options = H3LoadOptions(dtype=dtype or "float16")
    elif dtype is not None and dtype != load_options.dtype:
        raise ValueError("H3 dtype and load_options.dtype disagree")

    from musubi_tuner.minimax_h3.integration import create_backend as create_integrated_backend

    return create_integrated_backend(model=model, device=device, options=load_options)
