import argparse
import json
import logging
import os
import shutil
from typing import Callable

import accelerate

from musubi_tuner.utils import huggingface_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# checkpointファイル名
RESUME_METADATA_NAME = "resume_metadata.json"
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"


def save_resume_metadata(state_dir: str, global_step: int, step_in_epoch: int, epoch: int):
    """Save resume metadata alongside an accelerator state checkpoint."""
    metadata = {"global_step": global_step, "step_in_epoch": step_in_epoch, "epoch": epoch}
    path = os.path.join(state_dir, RESUME_METADATA_NAME)
    with open(path, "w") as f:
        json.dump(metadata, f)


def load_resume_metadata(state_dir: str) -> dict | None:
    """Load resume metadata from a state directory, or None if not present (old checkpoint)."""
    path = os.path.join(state_dir, RESUME_METADATA_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def update_resume_metadata(state_dir: str, extra: dict):
    """Update an existing resume_metadata.json with additional fields (e.g. loss info)."""
    path = os.path.join(state_dir, RESUME_METADATA_NAME)
    metadata = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                metadata = json.load(f)
        except Exception:
            pass
    metadata.update(extra)
    with open(path, "w") as f:
        json.dump(metadata, f)


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

    def prefill(self, avg: float, count: int) -> None:
        """Pre-fill with a known average from a resumed checkpoint so the
        moving average starts close to where training left off."""
        if count > 0 and avg > 0:
            self.loss_list = [avg] * count
            self.loss_total = avg * count

    @property
    def moving_average(self) -> float:
        if not self.loss_list:
            return 0.0
        return self.loss_total / len(self.loss_list)


def save_checkpoint_metadata(ckpt_file: str, metadata: dict) -> None:
    """Save a JSON sidecar file alongside a checkpoint."""
    json_path = os.path.splitext(ckpt_file)[0] + ".json"
    clean = {k: v for k, v in metadata.items() if v is not None}
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)


def remove_checkpoint_metadata(ckpt_file: str) -> None:
    """Remove JSON sidecar if it exists."""
    json_path = os.path.splitext(ckpt_file)[0] + ".json"
    if os.path.exists(json_path):
        os.remove(json_path)


def get_epoch_ckpt_name(model_name, epoch_no: int):
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name, step_no: int):
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name):
    return model_name + ".safetensors"


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # calculate the step number to remove from the last_n_steps and save_every_n_steps
    # e.g. if save_every_n_steps=10, save_last_n_steps=30, at step 50, keep 30 steps and remove step 10
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_and_remove_state_on_epoch_end(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
    epoch_no: int,
    global_step: int = 0,
    step_in_epoch: int = 0,
):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    save_resume_metadata(state_dir, global_step, step_in_epoch, epoch_no)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
    step_no: int,
    epoch: int = 0,
    step_in_epoch: int = 0,
):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    save_resume_metadata(state_dir, step_no, step_in_epoch, epoch)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
    global_step: int = 0,
    epoch: int = 0,
    step_in_epoch: int = 0,
):
    model_name = args.output_name

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)
    save_resume_metadata(state_dir, global_step, step_in_epoch, epoch)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b
