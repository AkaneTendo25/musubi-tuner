from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import torch


class AudioQuotaIndexSampler(torch.utils.data.Sampler[int]):
    """Opt-in sampler that enforces a minimum audio-batch quota per accumulation window."""

    def __init__(
        self,
        *,
        audio_indices: List[int],
        non_audio_indices: List[int],
        accumulation_steps: int,
        min_audio_batches_per_accum: int,
        seed: int,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        if min_audio_batches_per_accum < 0:
            raise ValueError(
                f"min_audio_batches_per_accum must be >= 0, got {min_audio_batches_per_accum}"
            )
        if min_audio_batches_per_accum > accumulation_steps:
            raise ValueError(
                "min_audio_batches_per_accum must be <= gradient_accumulation_steps "
                f"(got {min_audio_batches_per_accum} > {accumulation_steps})"
            )

        self.audio_indices = list(audio_indices)
        self.non_audio_indices = list(non_audio_indices)
        self.accumulation_steps = int(accumulation_steps)
        self.min_audio_batches_per_accum = int(min_audio_batches_per_accum)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def update_groups(self, audio_indices: List[int], non_audio_indices: List[int]) -> None:
        self.audio_indices = list(audio_indices)
        self.non_audio_indices = list(non_audio_indices)

    def __len__(self) -> int:
        return len(self.audio_indices) + len(self.non_audio_indices)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        audio = self.audio_indices.copy()
        non_audio = self.non_audio_indices.copy()
        rng.shuffle(audio)
        rng.shuffle(non_audio)

        ai = 0
        ni = 0
        total = len(audio) + len(non_audio)
        ordered: List[int] = []

        while (ai + ni) < total:
            slots = min(self.accumulation_steps, total - (ai + ni))
            audio_now = min(self.min_audio_batches_per_accum, slots, len(audio) - ai)

            required_non_audio = slots - audio_now
            available_non_audio = len(non_audio) - ni
            if available_non_audio < required_non_audio:
                extra_audio_needed = required_non_audio - available_non_audio
                audio_now = min(slots, audio_now + extra_audio_needed, len(audio) - ai)

            non_audio_now = slots - audio_now
            window = audio[ai : ai + audio_now] + non_audio[ni : ni + non_audio_now]
            ai += audio_now
            ni += non_audio_now

            missing = slots - len(window)
            if missing > 0 and ai < len(audio):
                take = min(missing, len(audio) - ai)
                window.extend(audio[ai : ai + take])
                ai += take
                missing -= take
            if missing > 0 and ni < len(non_audio):
                take = min(missing, len(non_audio) - ni)
                window.extend(non_audio[ni : ni + take])
                ni += take

            rng.shuffle(window)
            ordered.extend(window)

        return iter(ordered)


class AudioProbabilityIndexSampler(torch.utils.data.Sampler[int]):
    """Opt-in sampler that draws from audio/non-audio pools using a target probability."""

    def __init__(
        self,
        *,
        audio_indices: List[int],
        non_audio_indices: List[int],
        audio_batch_probability: float,
        seed: int,
    ) -> None:
        if not (0.0 <= audio_batch_probability <= 1.0):
            raise ValueError(
                f"audio_batch_probability must be in [0, 1], got {audio_batch_probability}"
            )

        self.audio_indices = list(audio_indices)
        self.non_audio_indices = list(non_audio_indices)
        self.audio_batch_probability = float(audio_batch_probability)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def update_groups(self, audio_indices: List[int], non_audio_indices: List[int]) -> None:
        self.audio_indices = list(audio_indices)
        self.non_audio_indices = list(non_audio_indices)

    def __len__(self) -> int:
        return len(self.audio_indices) + len(self.non_audio_indices)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        audio = self.audio_indices.copy()
        non_audio = self.non_audio_indices.copy()
        rng.shuffle(audio)
        rng.shuffle(non_audio)

        ai = 0
        ni = 0
        total = len(audio) + len(non_audio)
        ordered: List[int] = []

        while len(ordered) < total:
            audio_remaining = ai < len(audio)
            non_audio_remaining = ni < len(non_audio)

            if audio_remaining and non_audio_remaining:
                choose_audio = rng.random() < self.audio_batch_probability
                if choose_audio:
                    ordered.append(audio[ai])
                    ai += 1
                else:
                    ordered.append(non_audio[ni])
                    ni += 1
            elif audio_remaining:
                ordered.append(audio[ai])
                ai += 1
            elif non_audio_remaining:
                ordered.append(non_audio[ni])
                ni += 1
            else:
                break

        return iter(ordered)


def sync_dataset_group_epoch_without_loading(dataset_group, target_epoch: int, logger=None) -> None:
    datasets = getattr(dataset_group, "datasets", None)
    if datasets is None:
        return

    for dataset in datasets:
        current = getattr(dataset, "current_epoch", None)
        shuffle_fn = getattr(dataset, "shuffle_buckets", None)
        if current is None or not callable(shuffle_fn):
            continue

        if target_epoch > current:
            for _ in range(target_epoch - current):
                dataset.current_epoch += 1
                shuffle_fn()
        elif target_epoch < current:
            if logger is not None:
                logger.warning(
                    "epoch is not incremented. current_epoch: %s, epoch: %s",
                    current,
                    target_epoch,
                )
            dataset.current_epoch = target_epoch


def _batch_has_audio_from_batch_manager(
    batch_manager: Any,
    local_idx: int,
    exists_cache: Dict[str, bool],
) -> bool:
    bucket_reso, batch_idx = batch_manager.bucket_batch_indices[local_idx]
    bucket = batch_manager.buckets[bucket_reso]
    start = batch_idx * batch_manager.batch_size
    end = min(start + batch_manager.batch_size, len(bucket))

    for item_info in bucket[start:end]:
        audio_cache_path = getattr(item_info, "audio_latent_cache_path", None)
        if not audio_cache_path:
            continue

        exists = exists_cache.get(audio_cache_path)
        if exists is None:
            exists = os.path.exists(audio_cache_path)
            exists_cache[audio_cache_path] = exists
        if exists:
            return True

    return False


def split_concat_indices_by_audio(dataset_group) -> tuple[List[int], List[int]]:
    datasets = getattr(dataset_group, "datasets", None)
    if datasets is None:
        total = len(dataset_group)
        return [], list(range(total))

    audio_indices: List[int] = []
    non_audio_indices: List[int] = []
    global_offset = 0
    exists_cache: Dict[str, bool] = {}

    for dataset in datasets:
        dataset_len = len(dataset)
        batch_manager = getattr(dataset, "batch_manager", None)

        if batch_manager is None:
            non_audio_indices.extend(range(global_offset, global_offset + dataset_len))
            global_offset += dataset_len
            continue

        for local_idx in range(dataset_len):
            global_idx = global_offset + local_idx
            if _batch_has_audio_from_batch_manager(batch_manager, local_idx, exists_cache):
                audio_indices.append(global_idx)
            else:
                non_audio_indices.append(global_idx)

        global_offset += dataset_len

    return audio_indices, non_audio_indices


def build_audio_sampler(
    *,
    dataset_group,
    gradient_accumulation_steps: int,
    min_audio_batches_per_accum: int = 0,
    audio_batch_probability: Optional[float] = None,
    seed: int = 0,
) -> tuple[Optional[torch.utils.data.Sampler[int]], Optional[str], Dict[str, Any]]:
    """Build an opt-in audio-aware sampler.

    Returns:
        (sampler, mode, stats)
        - sampler: None when both controls are disabled
        - mode: "quota", "probability", or None
        - stats: includes audio/non-audio counts and effective control values
    """
    stats: Dict[str, Any] = {}
    min_audio_batches_per_accum = int(min_audio_batches_per_accum or 0)

    if audio_batch_probability is not None:
        audio_batch_probability = float(audio_batch_probability)
        if not (0.0 <= audio_batch_probability <= 1.0):
            raise ValueError(f"audio_batch_probability must be in [0, 1], got {audio_batch_probability}")

    if min_audio_batches_per_accum > 0 and audio_batch_probability is not None:
        raise ValueError(
            "--min_audio_batches_per_accum and --audio_batch_probability are mutually exclusive. "
            "Set only one of them."
        )

    if min_audio_batches_per_accum <= 0 and audio_batch_probability is None:
        return None, None, stats

    if min_audio_batches_per_accum > 0 and min_audio_batches_per_accum > int(gradient_accumulation_steps):
        raise ValueError(
            "min_audio_batches_per_accum must be <= gradient_accumulation_steps "
            f"(got {min_audio_batches_per_accum} > {gradient_accumulation_steps})"
        )

    audio_indices, non_audio_indices = split_concat_indices_by_audio(dataset_group)
    if len(audio_indices) == 0:
        if min_audio_batches_per_accum > 0:
            raise ValueError(
                "--min_audio_batches_per_accum is set, but no audio-bearing batches were found in the training dataset."
            )
        raise ValueError("--audio_batch_probability is set, but no audio-bearing batches were found in the training dataset.")

    stats["audio_batches"] = len(audio_indices)
    stats["non_audio_batches"] = len(non_audio_indices)

    if min_audio_batches_per_accum > 0:
        sampler = AudioQuotaIndexSampler(
            audio_indices=audio_indices,
            non_audio_indices=non_audio_indices,
            accumulation_steps=int(gradient_accumulation_steps),
            min_audio_batches_per_accum=min_audio_batches_per_accum,
            seed=int(seed),
        )
        stats["min_audio_batches_per_accum"] = min_audio_batches_per_accum
        stats["accumulation_steps"] = int(gradient_accumulation_steps)
        return sampler, "quota", stats

    sampler = AudioProbabilityIndexSampler(
        audio_indices=audio_indices,
        non_audio_indices=non_audio_indices,
        audio_batch_probability=float(audio_batch_probability),
        seed=int(seed),
    )
    stats["audio_batch_probability"] = float(audio_batch_probability)
    return sampler, "probability", stats
