"""Helpers for recovering Musubi training position from Accelerate states."""

import logging
import os
import pickle

import torch

from musubi_tuner.utils import train_utils

logger = logging.getLogger(__name__)


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
