"""Shared scheduling and argument validation for trainer validation hooks."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    """Add the common validation CLI surface exactly once."""
    if "--validation_dataset_config" in parser._option_string_actions:
        return
    group = parser.add_argument_group("validation")
    group.add_argument("--validation_dataset_config", type=Path, default=None)
    group.add_argument("--validate_every_n_steps", type=int, default=None)
    group.add_argument("--validate_every_n_epochs", type=int, default=None)
    group.add_argument("--validate_at_start", action="store_true")
    group.add_argument("--validation_seed", type=int, default=None)
    group.add_argument("--validation_timestep_bins", type=int, default=4)
    group.add_argument("--validation_min_timestep", type=int, default=0)
    group.add_argument("--validation_max_timestep", type=int, default=1000)
    group.add_argument("--max_validation_items", type=int, default=None)


def validation_requested(args: argparse.Namespace) -> bool:
    """Return whether any non-default validation option was supplied."""
    return bool(
        getattr(args, "validation_dataset_config", None) is not None
        or getattr(args, "validate_at_start", False)
        or getattr(args, "validate_every_n_steps", None) is not None
        or getattr(args, "validate_every_n_epochs", None) is not None
        or getattr(args, "validation_seed", None) is not None
        or getattr(args, "validation_timestep_bins", 4) != 4
        or getattr(args, "validation_min_timestep", 0) != 0
        or getattr(args, "validation_max_timestep", 1000) != 1000
        or getattr(args, "max_validation_items", None) is not None
    )


def validate_validation_args(args: argparse.Namespace, *, supported: bool) -> None:
    """Fail early for incomplete, unsupported, or invalid validation settings."""
    requested = validation_requested(args)
    if requested and not supported:
        raise ValueError("This trainer does not support validation loss metrics")

    dataset_config = getattr(args, "validation_dataset_config", None)
    has_gate = bool(
        getattr(args, "validate_at_start", False)
        or getattr(args, "validate_every_n_steps", None) is not None
        or getattr(args, "validate_every_n_epochs", None) is not None
    )
    if dataset_config is None and requested:
        raise ValueError("Validation options require --validation_dataset_config")
    if dataset_config is not None and not has_gate:
        raise ValueError("--validation_dataset_config requires a validation schedule")

    steps = getattr(args, "validate_every_n_steps", None)
    epochs = getattr(args, "validate_every_n_epochs", None)
    if steps is not None and steps < 1:
        raise ValueError("--validate_every_n_steps must be at least 1")
    if epochs is not None and epochs < 1:
        raise ValueError("--validate_every_n_epochs must be at least 1")
    if getattr(args, "validation_timestep_bins", 4) < 1:
        raise ValueError("--validation_timestep_bins must be at least 1")

    minimum = getattr(args, "validation_min_timestep", 0)
    maximum = getattr(args, "validation_max_timestep", 1000)
    if not 0 <= minimum < maximum <= 1000:
        raise ValueError("Validation timestep bounds must satisfy 0 <= min < max <= 1000")

    max_items = getattr(args, "max_validation_items", None)
    if max_items is not None and max_items < 1:
        raise ValueError("--max_validation_items must be at least 1")


def should_validate(args: argparse.Namespace, global_step: int, epoch: int | None, *, at_start: bool = False) -> bool:
    """Return whether validation is due at this lifecycle point.

    Epoch scheduling takes precedence over step scheduling. Start validation is
    a separate, explicit event.
    """
    if getattr(args, "validation_dataset_config", None) is None:
        return False
    if at_start:
        return bool(getattr(args, "validate_at_start", False))

    every_epoch = getattr(args, "validate_every_n_epochs", None)
    if every_epoch is not None:
        return epoch is not None and epoch > 0 and epoch % every_epoch == 0

    every_step = getattr(args, "validate_every_n_steps", None)
    return every_step is not None and epoch is None and global_step > 0 and global_step % every_step == 0


@dataclass
class ValidationEventDeduplicator:
    """Suppress repeated delivery of the same lifecycle validation event."""

    last_event: tuple[str, int] | None = None

    def claim(self, global_step: int, epoch: int | None, *, at_start: bool = False) -> bool:
        event = ("start", global_step) if at_start else ("epoch", epoch) if epoch is not None else ("step", global_step)
        if event == self.last_event:
            return False
        self.last_event = event
        return True


def derive_validation_seed(
    base_seed: int,
    *,
    bin_index: int,
    item_key: str | None = None,
    dataset_index: int | None = None,
    stream: str = "noise",
) -> int:
    """Derive one portable seed for an explicit item identity and stream."""
    if (item_key is None) == (dataset_index is None):
        raise ValueError("provide exactly one validation item_key or dataset_index")
    if item_key == "":
        raise ValueError("validation item_key must not be empty")
    if dataset_index is not None and dataset_index < 0:
        raise ValueError("validation dataset_index must be non-negative")
    if bin_index < 0:
        raise ValueError("validation bin_index must be non-negative")
    if not stream:
        raise ValueError("validation random stream must not be empty")

    identity_kind = "item" if item_key is not None else "index"
    identity_value = item_key if item_key is not None else str(dataset_index)
    payload = f"validation-v1\0{base_seed}\0{identity_kind}\0{identity_value}\0{bin_index}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
