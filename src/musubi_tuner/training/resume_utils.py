"""Helpers for recovering Musubi training position from Accelerate states."""

import logging
import os
import pickle

import torch

from musubi_tuner.utils import train_utils

logger = logging.getLogger(__name__)


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


def recover_global_step(state_dir: str) -> int:
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
        logger.info("Recovered global_step=%d from %s", global_step, scheduler_path)
        return global_step
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as e:
        logger.warning(
            "Could not recover global step from %s: %s; training bookkeeping will start from step 0",
            scheduler_path,
            e,
        )
        return 0
