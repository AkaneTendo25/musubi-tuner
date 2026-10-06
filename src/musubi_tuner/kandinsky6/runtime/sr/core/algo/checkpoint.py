"""Local checkpoint loading helpers used by SR inference."""

from __future__ import annotations

from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from pydantic import BaseModel


def join_path(base: str, *parts: str) -> str:
    """Join local path segments."""
    validate_local_path(base, "base path")
    return str(Path(base).joinpath(*parts))


def validate_local_path(path: str, field_name: str) -> None:
    """Reject URI-style paths because SR inference reads local files only."""
    if "://" in path:
        msg = f"{field_name} must be a local filesystem path, got {path!r}"
        raise ValueError(msg)


class CheckpointConfig(BaseModel):
    """Options for loading a local sharded checkpoint."""

    num_threads: int = 0


def _load_distributed_checkpoint(
    path: str,
    config: CheckpointConfig | None = None,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Load all ``.pt`` shard files from a local checkpoint directory."""
    validate_local_path(path, "checkpoint path")
    cfg = config or CheckpointConfig()
    state_dict_shard_paths = [str(p) for p in Path(path).iterdir() if p.suffix == ".pt"]

    def distributed_load(thread_num: int) -> list[tuple[int, dict[str, torch.Tensor]]]:
        thread_shards = []
        for shard_path in state_dict_shard_paths[thread_num :: max(1, cfg.num_threads)]:
            shard_number = int(Path(shard_path).stem)
            shard = load(shard_path, map_location=map_location, weights_only=weights_only)
            thread_shards.append((shard_number, shard))
        return thread_shards

    state_dict_shards: list[tuple[int, dict[str, torch.Tensor]]] = []
    if cfg.num_threads == 0:
        state_dict_shards += distributed_load(0)
    else:
        with ThreadPool(processes=cfg.num_threads) as pool:
            for thread_shards in pool.map(distributed_load, list(range(cfg.num_threads))):
                state_dict_shards += thread_shards

    state_dict_shards.sort()
    return [shard for _, shard in state_dict_shards]


def load(
    path: str,
    map_location: str | torch.device | None = None,
    *,
    weights_only: bool = False,
) -> Any:
    """Load a PyTorch object from a local path."""
    validate_local_path(path, "checkpoint path")
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def path_exists(path: str) -> bool:
    """Check whether a local file exists."""
    validate_local_path(path, "checkpoint path")
    return Path(path).exists()


def gather_distributed_model_state_dict(
    path: str,
    config: CheckpointConfig | None = None,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Merge local sharded checkpoint files into one state dict."""
    state_dict_shards = _load_distributed_checkpoint(path, config, map_location=map_location, weights_only=weights_only)

    shard_lists: dict[str, list[torch.Tensor]] = {}
    for shard in state_dict_shards:
        for key, value in shard.items():
            shard_lists.setdefault(key, []).append(value)

    logger.info("Merged {} local checkpoint shards from {}", len(state_dict_shards), path)
    return {key: torch.cat(tensors) for key, tensors in shard_lists.items()}
