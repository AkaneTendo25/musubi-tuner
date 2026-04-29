from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Optional

import torch

from musubi_tuner.dataset.audio_quota_sampler import sync_dataset_group_epoch_without_loading


class AccumulationGroupIndexSampler(torch.utils.data.Sampler[int]):
    """Order batch indices so each gradient accumulation window uses one group key."""

    def __init__(
        self,
        *,
        dataset_group: Any,
        group_by: str,
        accumulation_steps: int,
        num_processes: int,
        remainder: str,
        seed: int,
        shared_epoch: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        if group_by not in {"frames", "bucket", "dataset"}:
            raise ValueError(f"group_by must be one of frames, bucket, dataset; got {group_by!r}")
        if remainder not in {"drop", "pad", "allow_mixed"}:
            raise ValueError(f"remainder must be one of drop, pad, allow_mixed; got {remainder!r}")
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        if num_processes < 1:
            raise ValueError(f"num_processes must be >= 1, got {num_processes}")

        self.dataset_group = dataset_group
        self.group_by = group_by
        self.accumulation_steps = int(accumulation_steps)
        self.num_processes = int(num_processes)
        self.window_size = self.accumulation_steps * self.num_processes
        self.remainder = remainder
        self.seed = int(seed)
        self.shared_epoch = shared_epoch
        self.logger = logger
        self.epoch = 0

        self._initial_groups = split_concat_indices_by_accumulation_key(dataset_group, group_by)
        self._len = _planned_length(self._initial_groups, self.window_size, remainder)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        target_epoch = None
        if self.shared_epoch is not None:
            target_epoch = int(getattr(self.shared_epoch, "value", 0))
            sync_dataset_group_epoch_without_loading(self.dataset_group, target_epoch, logger=self.logger)

        rng_epoch = self.epoch if target_epoch is None else target_epoch
        rng = random.Random(self.seed + int(rng_epoch))
        if target_epoch is None:
            self.epoch += 1

        groups = split_concat_indices_by_accumulation_key(self.dataset_group, self.group_by)
        chunks: List[List[int]] = []
        mixed_remainders: List[int] = []

        for indices in groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            full_count = len(shuffled) // self.window_size
            for chunk_idx in range(full_count):
                start = chunk_idx * self.window_size
                chunks.append(shuffled[start : start + self.window_size])

            remainder = shuffled[full_count * self.window_size :]
            if not remainder:
                continue
            if self.remainder == "drop":
                continue
            if self.remainder == "pad":
                padded = list(remainder)
                while len(padded) < self.window_size:
                    padded.append(rng.choice(shuffled))
                chunks.append(padded)
            else:
                mixed_remainders.extend(remainder)

        if mixed_remainders:
            rng.shuffle(mixed_remainders)
            for start in range(0, len(mixed_remainders), self.window_size):
                chunks.append(mixed_remainders[start : start + self.window_size])

        rng.shuffle(chunks)
        ordered: List[int] = []
        for chunk in chunks:
            ordered.extend(chunk)
        return iter(ordered)


def _bucket_key_for_batch_manager(batch_manager: Any, local_idx: int) -> tuple[Any, ...]:
    bucket_reso, _batch_idx = batch_manager.bucket_batch_indices[local_idx]
    if isinstance(bucket_reso, tuple):
        return bucket_reso
    return (bucket_reso,)


def _frame_key_from_bucket_key(bucket_key: Iterable[Any]) -> Hashable:
    bucket_tuple = tuple(bucket_key)
    if len(bucket_tuple) >= 3 and type(bucket_tuple[2]) is int:
        return bucket_tuple[2]
    return "no_frame_count"


def split_concat_indices_by_accumulation_key(dataset_group: Any, group_by: str) -> Dict[Hashable, List[int]]:
    datasets = getattr(dataset_group, "datasets", None)
    groups: Dict[Hashable, List[int]] = defaultdict(list)

    if datasets is None:
        for idx in range(len(dataset_group)):
            groups["dataset:0"].append(idx)
        return dict(groups)

    global_offset = 0
    for dataset_idx, dataset in enumerate(datasets):
        dataset_len = len(dataset)
        batch_manager = getattr(dataset, "batch_manager", None)

        for local_idx in range(dataset_len):
            global_idx = global_offset + local_idx
            if group_by == "dataset" or batch_manager is None:
                key: Hashable = ("dataset", dataset_idx)
            else:
                bucket_key = _bucket_key_for_batch_manager(batch_manager, local_idx)
                if group_by == "frames":
                    key = ("frames", _frame_key_from_bucket_key(bucket_key))
                else:
                    key = ("bucket", bucket_key)
            groups[key].append(global_idx)

        global_offset += dataset_len

    return dict(groups)


def _planned_length(groups: Dict[Hashable, List[int]], window_size: int, remainder: str) -> int:
    if remainder == "allow_mixed":
        return sum(len(indices) for indices in groups.values())

    total = 0
    for indices in groups.values():
        n = len(indices)
        if remainder == "drop":
            total += (n // window_size) * window_size
        else:
            total += int(math.ceil(n / window_size)) * window_size if n > 0 else 0
    return total


def build_accumulation_group_sampler(
    *,
    dataset_group: Any,
    group_by: str,
    gradient_accumulation_steps: int,
    num_processes: int,
    remainder: str,
    seed: int,
    shared_epoch: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> tuple[Optional[torch.utils.data.Sampler[int]], Dict[str, Any]]:
    group_by = str(group_by or "none").lower()
    remainder = str(remainder or "drop").lower()
    stats: Dict[str, Any] = {}

    if group_by == "none":
        groups = split_concat_indices_by_accumulation_key(dataset_group, "bucket")
        stats["bucket_groups"] = len(groups)
        stats["batches"] = sum(len(indices) for indices in groups.values())
        return None, stats

    sampler = AccumulationGroupIndexSampler(
        dataset_group=dataset_group,
        group_by=group_by,
        accumulation_steps=int(gradient_accumulation_steps),
        num_processes=int(num_processes),
        remainder=remainder,
        seed=int(seed),
        shared_epoch=shared_epoch,
        logger=logger,
    )
    groups = sampler._initial_groups
    original_batches = sum(len(indices) for indices in groups.values())
    planned_batches = len(sampler)
    if planned_batches <= 0 and original_batches > 0:
        raise ValueError(
            "--accumulation_group_by with --accumulation_group_remainder drop would drop every batch. "
            "Use --accumulation_group_remainder pad or allow_mixed, reduce gradient_accumulation_steps, "
            "or choose a coarser grouping mode."
        )
    stats.update(
        {
            "group_by": group_by,
            "remainder": remainder,
            "groups": len(groups),
            "original_batches": original_batches,
            "planned_batches": planned_batches,
            "dropped_or_added_batches": planned_batches - original_batches,
            "accumulation_steps": int(gradient_accumulation_steps),
            "num_processes": int(num_processes),
            "window_size": int(gradient_accumulation_steps) * int(num_processes),
        }
    )
    return sampler, stats
