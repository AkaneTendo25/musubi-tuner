"""Shared default local model paths for LTX-2 workflows."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_DIR_NAME = "models"
DEFAULT_LTX2_CHECKPOINT_NAME = "ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT_NAME = "gemma-3-12b-it-qat-q4_0-unquantized"


def default_model_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir:
        return Path(base_dir)
    return Path(DEFAULT_MODEL_DIR_NAME)


def default_ltx2_checkpoint_path(base_dir: str | Path | None = None) -> str:
    return str(default_model_dir(base_dir) / DEFAULT_LTX2_CHECKPOINT_NAME)


def default_gemma_root_path(base_dir: str | Path | None = None) -> str:
    return str(default_model_dir(base_dir) / DEFAULT_GEMMA_ROOT_NAME)
