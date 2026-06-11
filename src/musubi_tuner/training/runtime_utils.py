from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
from typing import Optional

import torch

from musubi_tuner.training.accelerator_setup import clean_memory_on_device

logger = logging.getLogger(__name__)

_global_peak_alloc_mb: float = 0.0
_global_peak_reserved_mb: float = 0.0


def configure_console_output_for_help() -> None:
    """Avoid argparse --help crashes on Windows consoles with narrow encodings."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="replace")
        except Exception:
            pass


def update_global_peak() -> tuple[float, float]:
    """Update and return global peak memory stats since app launch."""
    global _global_peak_alloc_mb, _global_peak_reserved_mb
    if not torch.cuda.is_available():
        return _global_peak_alloc_mb, _global_peak_reserved_mb
    torch.cuda.synchronize()
    current_max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    current_max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    _global_peak_alloc_mb = max(_global_peak_alloc_mb, current_max_alloc)
    _global_peak_reserved_mb = max(_global_peak_reserved_mb, current_max_reserved)
    return _global_peak_alloc_mb, _global_peak_reserved_mb


def log_vram(tag: str, logger=None):
    """Log VRAM usage at a specific point for debugging spikes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        global_peak_alloc, global_peak_reserved = update_global_peak()
        msg = f"[VRAM_TRACE] {tag}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB max_allocated={max_alloc:.2f}GB PEAK_SINCE_START={global_peak_reserved / 1024:.2f}GB"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def log_cuda_memory_stats(tag: str, *, latents_shape: Optional[tuple] = None) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    global_peak_alloc, global_peak_reserved = update_global_peak()
    free_mb = None
    total_mb = None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_mb = free_b / (1024**2)
        total_mb = total_b / (1024**2)
    except Exception:
        pass
    if latents_shape is not None:
        if free_mb is not None and total_mb is not None:
            logger.info(
                "CUDA mem [%s] alloc=%.0fMB reserved=%.0fMB max_alloc=%.0fMB max_reserved=%.0fMB "
                "PEAK_SINCE_START=%.0fMB free=%.0fMB total=%.0fMB latents=%s",
                tag,
                alloc,
                reserved,
                max_alloc,
                max_reserved,
                global_peak_reserved,
                free_mb,
                total_mb,
                latents_shape,
            )
        else:
            logger.info(
                "CUDA mem [%s] alloc=%.0fMB reserved=%.0fMB max_alloc=%.0fMB max_reserved=%.0fMB "
                "PEAK_SINCE_START=%.0fMB latents=%s",
                tag,
                alloc,
                reserved,
                max_alloc,
                max_reserved,
                global_peak_reserved,
                latents_shape,
            )
    else:
        if free_mb is not None and total_mb is not None:
            logger.info(
                "CUDA mem [%s] alloc=%.0fMB reserved=%.0fMB max_alloc=%.0fMB max_reserved=%.0fMB "
                "PEAK_SINCE_START=%.0fMB free=%.0fMB total=%.0fMB",
                tag,
                alloc,
                reserved,
                max_alloc,
                max_reserved,
                global_peak_reserved,
                free_mb,
                total_mb,
            )
        else:
            logger.info(
                "CUDA mem [%s] alloc=%.0fMB reserved=%.0fMB max_alloc=%.0fMB max_reserved=%.0fMB PEAK_SINCE_START=%.0fMB",
                tag,
                alloc,
                reserved,
                max_alloc,
                max_reserved,
                global_peak_reserved,
            )


@contextmanager
def offload_optimizer_state_during_validation(
    optimizer,
    accelerator,
    enabled: bool,
    *,
    logger=None,
):
    """Temporarily move CUDA optimizer state to CPU for validation/sampling."""
    offloaded: list[tuple[dict, str, torch.device]] = []
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    distributed_type_name = getattr(getattr(accelerator, "distributed_type", None), "name", "")

    if enabled and distributed_type_name != "FSDP" and hasattr(base_optimizer, "state"):
        offloaded_bytes = 0
        for state in base_optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    offloaded.append((state, key, value.device))
                    offloaded_bytes += value.numel() * value.element_size()
                    state[key] = value.cpu()

        if offloaded:
            if logger is not None:
                logger.info("Offloaded optimizer state to CPU for validation: %.2f GB", offloaded_bytes / (1024**3))
            clean_memory_on_device(accelerator.device)

    try:
        yield
    finally:
        for state, key, device in offloaded:
            state[key] = state[key].to(device)


_update_global_peak = update_global_peak
_log_vram = log_vram
_log_cuda_memory_stats = log_cuda_memory_stats
