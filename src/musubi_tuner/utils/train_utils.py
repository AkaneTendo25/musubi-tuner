from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable

import accelerate
import torch

from musubi_tuner.utils import huggingface_utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def is_dashboard_stop_requested() -> bool:
    """Return whether the dashboard requested a graceful training stop."""
    stop_file = os.environ.get("MUSUBI_DASHBOARD_STOP_FILE")
    if not stop_file:
        return False
    try:
        return Path(stop_file).is_file()
    except OSError:
        return False


def poll_checkpoint_request_files(
    save_request_file: str | None,
    save_and_stop_request_file: str | None,
    accelerator: accelerate.Accelerator,
) -> tuple[bool, bool]:
    """Return synchronized save/stop requests observed by global rank zero."""
    if not save_request_file and not save_and_stop_request_file:
        return False, False
    flags = torch.zeros(2, dtype=torch.uint8, device=accelerator.device)
    if accelerator.is_main_process:
        try:
            flags[0] = bool(save_request_file and Path(save_request_file).is_file())
            flags[1] = bool(save_and_stop_request_file and Path(save_and_stop_request_file).is_file())
        except OSError as error:
            logger.warning("Failed to inspect checkpoint request files: %s", error)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(flags, src=0)
    return bool(flags[0].item()), bool(flags[1].item())


def consume_checkpoint_request_file(path: str | None) -> None:
    """Remove a fulfilled request while tolerating a concurrent user deletion."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as error:
        logger.warning("Checkpoint was saved, but request file %s could not be removed: %s", path, error)


# checkpointファイル名
STATE_MANIFEST_NAME = "state_manifest.json"
RESUME_METADATA_NAME = "resume_metadata.json"
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"


def save_resume_metadata(
    state_dir: str,
    global_step: int,
    step_in_epoch: int,
    epoch: int,
    data_seed: int | None = None,
) -> None:
    """Atomically save Musubi's training position beside an Accelerate state."""
    metadata = {
        "global_step": int(global_step),
        "step_in_epoch": int(step_in_epoch),
        "epoch": int(epoch),
    }
    if data_seed is not None:
        metadata["data_seed"] = int(data_seed)
    path = os.path.join(state_dir, RESUME_METADATA_NAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    os.replace(tmp_path, path)


def load_resume_metadata(state_dir: str) -> dict | None:
    """Load Musubi training-position metadata, if this state has it."""
    path = os.path.join(state_dir, RESUME_METADATA_NAME)
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            logger.warning("Resume metadata in %s must be a JSON object", path)
            return None
        return metadata
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as e:
        logger.warning("Could not read resume metadata from %s: %s", path, e)
        return None


def save_state_manifest(
    state_dir: str,
    *,
    state_type: str,
    global_step: int,
    step_in_epoch: int,
    epoch: int,
) -> None:
    """Write an atomic completion marker after every process finishes saving."""
    files = sorted(
        name for name in os.listdir(state_dir) if os.path.isfile(os.path.join(state_dir, name)) and name != STATE_MANIFEST_NAME
    )
    manifest = {
        "format_version": 1,
        "complete": True,
        "state_type": state_type,
        "global_step": int(global_step),
        "step_in_epoch": int(step_in_epoch),
        "epoch": int(epoch),
        "files": files,
        "saved_at": time.time(),
    }
    path = os.path.join(state_dir, STATE_MANIFEST_NAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def load_state_manifest(state_dir: str) -> dict | None:
    path = os.path.join(state_dir, STATE_MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(manifest, dict) or not manifest.get("complete"):
        return None
    return manifest


def is_complete_state_dir(state_dir: str, process_index: int | None = None) -> bool:
    """Return whether a directory contains a complete, resumable Accelerate state."""
    if not os.path.isdir(state_dir):
        return False

    try:
        names = os.listdir(state_dir)
    except OSError:
        return False

    manifest = load_state_manifest(state_dir)
    if manifest is not None:
        files = manifest.get("files")
        if not isinstance(files, list) or not all(
            isinstance(name, str) and os.path.isfile(os.path.join(state_dir, name)) for name in files
        ):
            return False

    has_model = any(
        name in {"model.safetensors", "pytorch_model.bin"} or re.match(r"model_\d+\.(safetensors|bin)$", name) for name in names
    )
    has_optimizer = any(name == "optimizer.bin" or re.match(r"optimizer_\d+\.bin$", name) for name in names)
    has_position = RESUME_METADATA_NAME in names or "scheduler.bin" in names
    if not (has_model and has_optimizer and has_position):
        return False
    if process_index is not None and f"random_states_{process_index}.pkl" not in names:
        return False
    return True


def get_resume_position(
    initial_global_step: int,
    num_update_steps_per_epoch: int,
    metadata: dict | None,
    num_batches_per_epoch: int | None = None,
) -> tuple[int, int]:
    """Return the zero-based epoch and number of already-consumed batches to skip."""
    if initial_global_step <= 0:
        return 0, 0

    try:
        metadata_global_step = int(metadata.get("global_step", 0)) if metadata is not None else 0
    except (TypeError, ValueError):
        metadata_global_step = 0

    if metadata_global_step > 0:
        try:
            saved_epoch = int(metadata.get("epoch", 1))
            step_in_epoch = int(metadata.get("step_in_epoch", 0))
        except (TypeError, ValueError):
            return initial_global_step // max(num_update_steps_per_epoch, 1), 0
        if num_batches_per_epoch is not None and step_in_epoch >= num_batches_per_epoch:
            return max(saved_epoch, 0), 0
        if step_in_epoch > 0:
            return max(saved_epoch - 1, 0), step_in_epoch
        return max(saved_epoch, 0), 0

    return initial_global_step // max(num_update_steps_per_epoch, 1), 0


def normalize_step_in_epoch(step_in_epoch: int, num_batches_per_epoch: int) -> int:
    """Represent an epoch-boundary checkpoint as the start of the following epoch."""
    if step_in_epoch >= num_batches_per_epoch:
        return 0
    return max(step_in_epoch, 0)


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args
    # if wandb is not enabled, the log is not exposed to the public, but it is fine to filter out sensitive values to be safe

    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "dit",
        "vae",
        "text_encoder1",
        "text_encoder2",
        "image_encoder",
        "base_weights",
        "network_weights",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if v is None or isinstance(v, bool) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


class LossRecorder:
    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)


def get_epoch_ckpt_name(model_name, epoch_no: int):
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name, step_no: int):
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name):
    return model_name + ".safetensors"


def resolve_save_retention(args: argparse.Namespace, save_kind: str, *, for_state: bool = False) -> int | None:
    """Resolve how many checkpoints to keep on disk.

    ``--save_last_n_checkpoints`` is the single new option and applies to both model
    checkpoints and states. Legacy options retain their original scope so an old
    state-only command cannot unexpectedly delete model checkpoints. Non-positive
    values disable retention instead of deleting the checkpoint that was just saved.
    """
    if save_kind not in {"steps", "epochs"}:
        raise ValueError(f"unsupported checkpoint save kind: {save_kind}")

    value = getattr(args, "save_last_n_checkpoints", None)
    if value is None and for_state:
        value = getattr(args, f"save_last_n_{save_kind}_state", None)
    if value is None:
        value = getattr(args, f"save_last_n_{save_kind}", None)
    return value if value is not None and value > 0 else None


def get_remove_ckpt_no(
    args: argparse.Namespace,
    current_no: int,
    save_every_n: int | None,
    save_kind: str,
    *,
    for_state: bool = False,
) -> int | None:
    """Return the cadence slot to delete so only the last N checkpoints remain.

    Single retention core for both cadences: at each save, the checkpoint written
    ``N * save_every_n`` units ago (epochs or steps) falls out of the keep window.
    ``N`` comes from :func:`resolve_save_retention` — a checkpoint count, so
    ``save_last_n=8`` keeps 8 checkpoints even when they are 400 steps apart.
    Returns ``None`` when retention is disabled or nothing is old enough to remove.
    """
    keep_n = resolve_save_retention(args, save_kind, for_state=for_state)
    if keep_n is None or not save_every_n:
        return None

    # e.g. if save_every_n_steps=400, save_last_n_checkpoints=3, at step 1600 remove step 400 (keeps 800, 1200, 1600)
    remove_no = current_no - save_every_n * keep_n
    if remove_no < 0:
        return None
    return remove_no


def save_and_remove_state_on_epoch_end(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
    epoch_no: int,
    global_step: int = 0,
    step_in_epoch: int = 0,
):
    model_name = args.output_name

    is_main_process = accelerator.is_main_process
    if is_main_process:
        logger.info("")
        logger.info(f"saving state at epoch {epoch_no}")
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if is_main_process:
        save_resume_metadata(state_dir, global_step, step_in_epoch, epoch_no, data_seed=getattr(args, "seed", None))
        save_state_manifest(
            state_dir,
            state_type="epoch",
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            epoch=epoch_no,
        )
        logger.info("state saved to %s; resume with --resume %s", state_dir, state_dir)
        if args.save_state_to_huggingface:
            logger.info("uploading state to huggingface.")
            huggingface_utils.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

        remove_epoch_no = get_remove_ckpt_no(args, epoch_no, args.save_every_n_epochs, "epochs", for_state=True)
        if remove_epoch_no is not None:
            state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)
    accelerator.wait_for_everyone()


def save_and_remove_state_stepwise(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
    step_no: int,
    epoch: int = 0,
    step_in_epoch: int = 0,
    apply_retention: bool = True,
):
    model_name = args.output_name

    is_main_process = accelerator.is_main_process
    if is_main_process:
        logger.info("")
        logger.info(f"saving state at step {step_no}")
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if is_main_process:
        save_resume_metadata(state_dir, step_no, step_in_epoch, epoch, data_seed=getattr(args, "seed", None))
        save_state_manifest(state_dir, state_type="step", global_step=step_no, step_in_epoch=step_in_epoch, epoch=epoch)
        logger.info("state saved to %s; resume with --resume %s", state_dir, state_dir)
        if args.save_state_to_huggingface:
            logger.info("uploading state to huggingface.")
            huggingface_utils.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

        # The unified option keeps states aligned with model files. Legacy state
        # overrides retain their historical, state-only behavior.
        remove_step_no = (
            get_remove_ckpt_no(args, step_no, args.save_every_n_steps, "steps", for_state=True) if apply_retention else None
        )
        if remove_step_no is not None:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)
    accelerator.wait_for_everyone()


def save_state_on_train_end(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
    global_step: int = 0,
    epoch: int = 0,
    step_in_epoch: int = 0,
):
    model_name = args.output_name

    is_main_process = accelerator.is_main_process
    if is_main_process:
        logger.info("")
        logger.info("saving last state.")
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if is_main_process:
        save_resume_metadata(state_dir, global_step, step_in_epoch, epoch, data_seed=getattr(args, "seed", None))
        save_state_manifest(
            state_dir,
            state_type="final",
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            epoch=epoch,
        )
        logger.info("state saved to %s; resume with --resume %s", state_dir, state_dir)

        if args.save_state_to_huggingface:
            logger.info("uploading last state to huggingface.")
            huggingface_utils.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))
    accelerator.wait_for_everyone()


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def resolve_save_dtype(save_precision: str | None, full_fp16: bool = False, full_bf16: bool = False) -> torch.dtype:
    """Resolve the dtype for saving network weights.

    Explicit --save_precision wins; otherwise follow full_fp16/full_bf16 so the
    saved weights match the training precision; otherwise fp32, the precision
    the network weights are actually trained in.
    """
    if save_precision is not None:
        return str_to_dtype(save_precision)
    if full_fp16:
        return torch.float16
    if full_bf16:
        return torch.bfloat16
    return torch.float32
