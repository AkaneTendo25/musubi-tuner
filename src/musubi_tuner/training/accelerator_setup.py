"""Accelerator/device setup utilities shared across training scripts."""

from datetime import timedelta
import argparse
import gc
import logging
import os
import time

import torch
from packaging.version import Version
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import TorchDynamoPlugin, DynamoBackend

logger = logging.getLogger(__name__)


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


# for collate_fn: epoch and step is multiprocessing.Value
class collator_class:
    def __init__(self, epoch, dataset):
        self.current_epoch = epoch
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch for validation
        dataset.set_current_epoch(self.current_epoch.value)
        return examples[0]  # batch size is always 1, so we unwrap it here


def dataloader_extra_kwargs(args: argparse.Namespace, n_workers: int) -> dict:
    """Opt-in DataLoader kwargs derived from CLI args (default off -> baseline behavior).

    With --dataloader_pin_memory unset this returns {"pin_memory": False}, which equals the
    torch DataLoader default, so the constructed loader is behaviorally identical to the
    pre-feature call. --dataloader_prefetch_factor is only forwarded when explicitly set AND
    n_workers > 0 (torch raises if prefetch_factor is passed with num_workers == 0).
    """
    extra = {"pin_memory": bool(getattr(args, "dataloader_pin_memory", False))}
    prefetch_factor = getattr(args, "dataloader_prefetch_factor", None)
    if prefetch_factor is not None and n_workers > 0:
        extra["prefetch_factor"] = prefetch_factor
    return extra


def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    """
    DeepSpeed is not supported in this script currently.
    """
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
                static_graph=args.ddp_static_graph,
                find_unused_parameters=bool(getattr(args, "ddp_find_unused_parameters", False)),
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph or bool(getattr(args, "ddp_find_unused_parameters", False))
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    dynamo_plugin = None
    if args.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(args.dynamo_backend.upper()),
            mode=args.dynamo_mode,
            fullgraph=args.dynamo_fullgraph,
            dynamic=args.dynamo_dynamic,
        )

    accelerator_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision else None,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    if getattr(args, "dataloader_pin_memory", False):
        # Opt-in: enable non_blocking H2D for Accelerate's prepared DataLoader (pairs with
        # base-loader pin_memory). Off by default -> kwargs unchanged from the prior call.
        accelerator_kwargs["dataloader_config"] = DataLoaderConfiguration(non_blocking=True)
    # Opt-in FSDP1/ZeRO sharding (--ltx2_fsdp).
    from musubi_tuner.ltx2_fsdp import build_ltx2_fsdp_plugin

    fsdp_plugin = build_ltx2_fsdp_plugin(args)
    if fsdp_plugin is not None:
        accelerator_kwargs["fsdp_plugin"] = fsdp_plugin
    accelerator = Accelerator(**accelerator_kwargs)
    print("accelerator device:", accelerator.device)
    if (
        args.log_cuda_memory_every_n_steps is not None
        and args.log_cuda_memory_every_n_steps > 0
        and accelerator.device.type == "cuda"
    ):
        props = torch.cuda.get_device_properties(accelerator.device)
        total_mb = props.total_memory / (1024**2)
        logger.info("CUDA device: %s (%s) total=%.0fMB", accelerator.device, props.name, total_mb)
    return accelerator
