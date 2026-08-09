"""Helpers for recovering Musubi training position from Accelerate states."""

import logging
import os
import pickle
import random
import re
from dataclasses import dataclass

import numpy as np
import torch

from musubi_tuner.utils import train_utils

logger = logging.getLogger(__name__)


@dataclass
class TrainingRNGState:
    """Process-local RNG state restored by Accelerate at checkpoint load time."""

    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def capture_training_rng_state() -> TrainingRNGState:
    """Capture RNG streams that dataset setup or resume bookkeeping may disturb."""
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return TrainingRNGState(random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_state)


def restore_training_rng_state(state: TrainingRNGState) -> None:
    """Restore the checkpoint RNG streams immediately before resumed iteration."""
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def validate_resume_state_dir(state_dir: str, process_index: int = 0, num_processes: int | None = None) -> None:
    """Reject weight files and incomplete Accelerate states before loading them."""
    if not os.path.isdir(state_dir):
        if os.path.isfile(state_dir):
            raise ValueError(
                f"--resume requires an Accelerate state directory, not a model file: {state_dir}. "
                "Use --network_weights for a LoRA .safetensors file; that only loads weights and starts at step 0."
            )
        raise ValueError(f"--resume state directory does not exist: {state_dir}")

    if not train_utils.is_complete_state_dir(state_dir):
        raise ValueError(
            f"--resume state directory is incomplete: {state_dir}. A resumable state needs model, optimizer, and position files."
        )

    metadata_path = os.path.join(state_dir, train_utils.RESUME_METADATA_NAME)
    scheduler_path = os.path.join(state_dir, "scheduler.bin")
    if not os.path.isfile(metadata_path) and not os.path.isfile(scheduler_path):
        raise ValueError(
            f"--resume requires a saved training-state directory, but {state_dir} contains neither "
            f"{train_utils.RESUME_METADATA_NAME} nor scheduler.bin. Enable --save_state when saving checkpoints."
        )

    process_indices = range(num_processes) if num_processes is not None else (process_index,)
    for expected_process in process_indices:
        random_state_path = os.path.join(state_dir, f"random_states_{expected_process}.pkl")
        if not os.path.isfile(random_state_path):
            raise ValueError(
                f"Resume state is incomplete for process {expected_process}: missing {random_state_path}. "
                "Each distributed process must save its own RNG state; resuming this folder would not be exact."
            )


def state_dir_matches_output_name(entry: str, output_name: str | None) -> bool:
    """Return whether a state directory belongs to the requested output name."""
    if not output_name:
        return True
    escaped = re.escape(str(output_name))
    return re.match(rf"^{escaped}(?:-state|-\d{{6}}-state|-step\d+-state)$", entry) is not None


def find_latest_state_dir(args, num_processes: int = 1) -> str | None:
    """Find the highest-step complete state in output_dir for --autoresume."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir or not os.path.isdir(output_dir):
        return None

    best_key = (-1, -1.0)
    best_path = None
    for entry in sorted(os.listdir(output_dir)):
        full_path = os.path.join(output_dir, entry)
        if not entry.endswith("-state") or not state_dir_matches_output_name(entry, getattr(args, "output_name", None)):
            continue
        if not train_utils.is_complete_state_dir(full_path):
            continue
        if any(not os.path.isfile(os.path.join(full_path, f"random_states_{index}.pkl")) for index in range(num_processes)):
            continue
        try:
            step = recover_global_step(full_path)
            modified = os.path.getmtime(full_path)
        except (OSError, TypeError, ValueError):
            continue
        if step > 0 and (step, modified) > best_key:
            best_key = (step, modified)
            best_path = full_path
    return best_path


def _restore_dataset_seed(dataset, data_seed: int) -> None:
    """Restore Musubi dataset seeds before resumed workers are started."""
    child_datasets = getattr(dataset, "datasets", None)
    if child_datasets is not None:
        for child_dataset in child_datasets:
            _restore_dataset_seed(child_dataset, data_seed)
        return

    set_seed = getattr(dataset, "set_seed", None)
    shared_epoch = getattr(dataset, "shared_epoch", None)
    if set_seed is not None and shared_epoch is not None:
        set_seed(data_seed, shared_epoch)


def configure_dataloader_for_epoch(dataloader, epoch: int, data_seed: int | None = None) -> None:
    """Select a reproducible epoch order and optionally restore its saved data seed."""
    if data_seed is not None:
        data_seed = int(data_seed)
        get_sampler = getattr(dataloader, "get_sampler", None)
        sampler = get_sampler() if get_sampler is not None else getattr(dataloader, "sampler", None)
        if hasattr(sampler, "initial_seed"):
            sampler.initial_seed = data_seed
        generator = getattr(sampler, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(data_seed + epoch)
        _restore_dataset_seed(dataloader.dataset, data_seed)

    set_epoch = getattr(dataloader, "set_epoch", None)
    if set_epoch is not None:
        set_epoch(epoch)


def recover_global_step(state_dir: str, *, strict: bool = False) -> int:
    """Recover Musubi's optimization step from new or legacy state directories."""
    metadata = train_utils.load_resume_metadata(state_dir)
    if metadata is not None:
        try:
            global_step = int(metadata.get("global_step", 0))
            if global_step > 0:
                logger.info("Recovered global_step=%d from %s", global_step, train_utils.RESUME_METADATA_NAME)
                return global_step
        except (TypeError, ValueError) as e:
            logger.warning("Invalid global_step in %s: %s", train_utils.RESUME_METADATA_NAME, e)

    scheduler_path = os.path.join(state_dir, "scheduler.bin")
    try:
        scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
        global_step = int(scheduler_state["last_epoch"])
        if global_step <= 0:
            raise ValueError(f"scheduler last_epoch must be positive, got {global_step}")
        logger.info("Recovered global_step=%d from %s", global_step, scheduler_path)
        return global_step
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as e:
        message = f"Could not recover a positive global step from {state_dir}: {e}"
        if strict:
            raise ValueError(
                message + ". Refusing to silently restart at step 0; use a *-state directory created with --save_state."
            ) from e
        logger.warning("%s; training bookkeeping will start from step 0", message)
        return 0
