import argparse
import logging
import os
import re
import shutil
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


_STEP_STATE_DIR_RE = re.compile(r"-step(?P<step>\d+)-state$")


def get_resume_step(
    resume_path: str | None,
    lr_scheduler=None,
    optimizer=None,
    num_processes: int = 1,
) -> int:
    """Recover the completed optimization step from a loaded training state."""
    if not resume_path:
        return 0

    candidates: list[int] = []
    match = _STEP_STATE_DIR_RE.search(Path(resume_path).name)
    if match:
        candidates.append(int(match.group("step")))

    scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)
    scheduler_step = getattr(scheduler, "last_epoch", None)
    if scheduler_step is not None:
        try:
            candidates.append(max(0, int(scheduler_step) // max(1, int(num_processes))))
        except (TypeError, ValueError):
            pass

    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    optimizer_steps = []
    for state in getattr(raw_optimizer, "state", {}).values():
        step = state.get("step") if isinstance(state, dict) else None
        if step is None:
            continue
        try:
            optimizer_steps.append(int(step.item() if hasattr(step, "item") else step))
        except (TypeError, ValueError):
            continue
    if optimizer_steps:
        candidates.append(max(optimizer_steps))

    return max(candidates, default=0)


def get_resume_position(
    completed_steps: int,
    update_steps_per_epoch: int,
    gradient_accumulation_steps: int,
    batches_per_epoch: int,
) -> tuple[int, int]:
    """Return the zero-based epoch and batch count already consumed there."""
    if update_steps_per_epoch <= 0:
        raise ValueError("update_steps_per_epoch must be positive")
    epoch = completed_steps // update_steps_per_epoch
    steps_into_epoch = completed_steps % update_steps_per_epoch
    batches_to_skip = min(batches_per_epoch, steps_into_epoch * gradient_accumulation_steps)
    return epoch, batches_to_skip


# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"


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


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
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


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
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


def save_state_on_train_end(args: argparse.Namespace, accelerator: accelerate.Accelerator):
    model_name = args.output_name

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def is_complete_state_dir(state_dir: str) -> bool:
    """Return whether an Accelerate state directory has the files needed to resume."""
    if not os.path.isdir(state_dir):
        return False

    def has_file(*names: str) -> bool:
        return any(os.path.isfile(os.path.join(state_dir, name)) for name in names)

    has_model = has_file("model.safetensors", "pytorch_model.bin")
    return has_model and has_file("optimizer.bin") and has_file("scheduler.bin")


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
