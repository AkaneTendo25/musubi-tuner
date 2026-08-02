from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from musubi_tuner.minimax_h3.request import H3GenerationRequest
from musubi_tuner.minimax_h3.training import H3TrainingMode


def _raise_unavailable(component: str) -> NoReturn:
    from musubi_tuner.minimax_h3.backend import H3BackendUnavailableError

    raise H3BackendUnavailableError(
        f"MiniMax H3 {component} support is unavailable; install the upstream source under "
        "musubi_tuner/minimax_h3/vendor/official and implement the component loader in minimax_h3/integration.py"
    )


def create_latent_encoder(*, model: Path, device: str | None, dtype: str):
    """Load the released video and audio VAEs and adapt their encoders to Musubi."""
    del model, device, dtype
    _raise_unavailable("latent encoder")


def create_conditioning_encoder(*, model: Path, device: str | None, dtype: str):
    """Load the released understanding encoder and adapt its hidden-state output to Musubi."""
    del model, device, dtype
    _raise_unavailable("conditioning encoder")


def create_generator(*, model: Path, device: str | None, dtype: str, request: H3GenerationRequest):
    """Load the released inference components and adapt generation to Musubi."""
    del model, device, dtype, request
    _raise_unavailable("generation")


def create_training_backend(
    *,
    model: Path,
    device: str | None,
    dtype: str,
    mode: H3TrainingMode,
    attention_mode: str,
    split_attention: bool,
):
    """Load the released transformer and adapt its training forward to Musubi."""
    del model, device, dtype, mode, attention_mode, split_attention
    _raise_unavailable("training")
