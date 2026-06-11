"""Resume and autoresume helpers for NetworkTrainer."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from typing import Optional

import huggingface_hub
import torch
from accelerate import Accelerator

from musubi_tuner.utils import huggingface_utils, train_utils

logger = logging.getLogger(__name__)


def resume_from_local_or_hf_if_specified(self, accelerator: Accelerator, args: argparse.Namespace) -> int:
    """Resume training state. Returns the recovered global_step, or 0 when not resuming."""
    if not args.resume:
        self._resume_state_dir = None
        return 0

    if not args.resume_from_huggingface:
        if not train_utils.is_complete_state_dir(args.resume):
            if getattr(args, "_autoresume_selected", False):
                logger.warning(
                    "autoresume: selected state directory is missing or incomplete, starting from scratch: %s",
                    args.resume,
                )
                args.resume = None
                self._resume_state_dir = None
                return 0
            raise FileNotFoundError(f"resume state directory is missing or incomplete: {args.resume}")

        self._register_optimizer_resume_safe_globals(args)
        logger.info("resume training from local state: %s", args.resume)
        try:
            accelerator.load_state(args.resume)
        except Exception:
            if getattr(args, "_autoresume_selected", False) and not train_utils.is_complete_state_dir(args.resume):
                logger.warning(
                    "autoresume: selected state disappeared or became incomplete before loading, starting from scratch: %s",
                    args.resume,
                )
                args.resume = None
                self._resume_state_dir = None
                return 0
            raise
        self._resume_state_dir = args.resume
        return self._recover_global_step(args.resume)

    logger.info("resume training from huggingface state: %s", args.resume)
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info("Downloading state from huggingface: %s/%s@%s", repo_id, path_in_repo, revision)

    list_files = huggingface_utils.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
    if len(results) == 0:
        raise ValueError("No files found in the specified repo id/path/revision")
    dirname = os.path.dirname(results[0])
    self._register_optimizer_resume_safe_globals(args)
    accelerator.load_state(dirname)
    self._resume_state_dir = dirname

    return self._recover_global_step(dirname)


def recover_global_step(state_dir: str) -> int:
    """Read global_step from resume metadata or LR scheduler state."""
    metadata = train_utils.load_resume_metadata(state_dir)
    if metadata is not None and metadata.get("global_step", 0) > 0:
        global_step = int(metadata["global_step"])
        logger.info("recovered global_step=%s from resume_metadata.json", global_step)
        return global_step

    scheduler_path = os.path.join(state_dir, "scheduler.bin")
    try:
        scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
        global_step = int(scheduler_state["last_epoch"])
        logger.info("recovered global_step=%s from %s", global_step, scheduler_path)
        return global_step
    except Exception as e:
        logger.warning("could not recover global_step from %s: %s  (starting from step 0)", scheduler_path, e)
        return 0


def state_dir_matches_output_name(entry: str, output_name: Optional[str]) -> bool:
    """Return True when entry is a state directory generated for output_name."""
    if not output_name:
        return True

    escaped = re.escape(str(output_name))
    pattern = (
        rf"^{escaped}(?:"
        r"-state"
        r"|-\d{6}-state"
        r"|-step\d+-state"
        r"|-interrupt-step\d+(?:-\d+)?-state"
        r")$"
    )
    return re.match(pattern, entry) is not None


def find_latest_state_dir(args: argparse.Namespace) -> Optional[str]:
    """Find the latest complete training state directory in output_dir for --autoresume."""
    if not args.output_dir or not os.path.isdir(args.output_dir):
        return None

    best_step = -1
    best_path = None
    output_name = getattr(args, "output_name", None)

    for entry in os.listdir(args.output_dir):
        full_path = os.path.join(args.output_dir, entry)
        if not os.path.isdir(full_path) or not entry.endswith("-state"):
            continue

        if not state_dir_matches_output_name(entry, output_name):
            continue

        if not train_utils.is_complete_state_dir(full_path):
            continue

        scheduler_path = os.path.join(full_path, "scheduler.bin")
        if not os.path.exists(scheduler_path):
            scheduler_path = None

        metadata = train_utils.load_resume_metadata(full_path)
        if metadata is not None and metadata.get("global_step", 0) > 0:
            step = int(metadata["global_step"])
        else:
            step_match = re.search(r"-step(\d+)-state$", entry)
            if step_match:
                step = int(step_match.group(1))
            else:
                if scheduler_path is None:
                    continue
                try:
                    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
                    step = int(scheduler_state["last_epoch"])
                except Exception:
                    continue

        if step > best_step:
            best_step = step
            best_path = full_path

    return best_path
