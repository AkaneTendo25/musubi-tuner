"""How the YuE2 cache scripts see a dataset group.

The shared cache drivers hand the encode callbacks one batch of ``ItemInfo`` at a time; the per-record fields
(style, lyrics, ABC and its flavour, song identity, excerpt bounds, audio file) live in the audio datasource.
``plan_yue2_datasets`` snapshots them per dataset, and an item finds its record through ``ItemInfo.dataset_index``
(the dataset's position in its ``DatasetGroup``) and ``ItemInfo.datasource_index`` (the record's position in the
datasource), as ``minimax_h3/cache_plan.py`` does for MiniMax-H3.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Mapping, Sequence

from safetensors import safe_open

from musubi_tuner.dataset.audio_dataset import AudioDataset
from musubi_tuner.dataset.datasources import AudioRecord
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo

logger = logging.getLogger(__name__)

# a record as the cache scripts see it: audio_path, item_key, style, lyrics, abc, abc_mode, song_id, start_s, end_s, source
YuE2Record = AudioRecord


@dataclass(frozen=True)
class YuE2DatasetPlan:
    dataset_index: int
    # aligned with the datasource indices (ItemInfo.datasource_index)
    records: list[AudioRecord]


def plan_yue2_datasets(datasets: Sequence[BaseDataset]) -> list[YuE2DatasetPlan]:
    """One plan per dataset of a DatasetGroup, in group order."""
    plans = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, AudioDataset):
            raise ValueError(
                f"YuE2 caching accepts only audio datasets (audio_directory / audio_jsonl_file); dataset {index} is not"
            )
        if dataset.dataset_index != index:
            raise ValueError(f"YuE2 dataset {index} is not at its DatasetGroup position (dataset_index={dataset.dataset_index})")
        plans.append(YuE2DatasetPlan(dataset_index=index, records=list(dataset.datasource.records)))
    return plans


def item_record(plans: Sequence[YuE2DatasetPlan], item: ItemInfo) -> AudioRecord:
    """The record an item (a latent segment or a text record) came from."""
    if item.dataset_index is None or not 0 <= item.dataset_index < len(plans):
        raise ValueError(f"YuE2 cache item is missing its dataset provenance: {item.item_key}")
    records = plans[item.dataset_index].records
    if item.datasource_index is None or not 0 <= item.datasource_index < len(records):
        raise ValueError(f"YuE2 cache item is missing its datasource provenance: {item.item_key}")
    record = records[item.datasource_index]
    if record.item_key != item.item_key:
        raise ValueError(f"YuE2 cache item {item.item_key} does not match its record {record.item_key}")
    return record


def cache_metadata(path: str | Path) -> dict[str, str]:
    """The metadata of a cache file (header only); empty when it cannot be read."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception as error:
        logger.warning("Unable to read YuE2 cache metadata from %s: %s", path, error)
        return {}


def latent_metadata_matches(path: str | Path, expected: Mapping[str, str]) -> bool:
    """Whether a cache file carries every expected metadata value (the --skip_existing staleness check)."""
    if not Path(path).is_file():
        return False
    actual = cache_metadata(path)
    return all(actual.get(key) == value for key, value in expected.items())
