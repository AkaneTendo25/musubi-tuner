"""Replay of base-model ("minted") songs during YuE2 AR training.

Each record is a codec token stream that the frozen base model sampled on its own, stored together with the style
and lyrics prompt that produced it. Mixing these songs into training batches anchors the AR LoRA to the token
distribution the base model already has. Three input layouts are understood:

* kit ``.pt``: a list of dicts with ``codec`` (integer array ``[T]``), ``src`` (``"minted"`` or ``"minted_val"``),
  ``style`` and ``lyrics``. It is read with ``torch.load(weights_only=True)``; the only extra globals allowed are the
  NumPy pieces needed to rebuild arrays.
* manifest ``.json``: ``{"songs": [...]}`` where every song names a ``tokens`` file (``.npy`` or ``.safetensors``),
  sets ``true_tokens`` and may carry ``split``, ``style``, ``lyrics`` and ``song`` or ``name``.
* ``.jsonl``: one record per line with ``codes`` (``codec`` and ``tokens`` also work) given inline as ints or as a
  path to a ``.npy``/``.safetensors`` file, ``style`` or ``caption``, ``lyrics`` and optional ``song_id`` and
  ``split``/``src``.

``tokenize`` encodes the prompts once with the run's tokenizer, after which ``sample`` hands out finished AR items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from musubi_tuner.yue2.yue2_protocol import (
    CODEC_SIZE,
    COT_MODES,
    DEFAULT_INSTRUMENTAL_LYRICS,
    normalize_prompt_fields,
    text_ids,
)

logger = logging.getLogger(__name__)

VALIDATION_SPLITS = ("minted_val", "validation", "val")


@dataclass(frozen=True)
class MintedRecord:
    codes: torch.Tensor  # int64 [T], codec indices in [0, CODEC_SIZE)
    style: str
    lyrics: str
    song_id: str
    is_validation: Optional[bool]  # None when the source names no split


@dataclass(frozen=True)
class MintedItem:
    codes: torch.Tensor  # int64 [T]
    text_ids: dict[str, list[int]] = field(default_factory=dict)  # cot -> [EOD] + prompt ids
    song_id: str = ""
    style: str = ""
    lyrics: str = ""


def _as_codes(value: Any, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label}: minted codes must be a nonempty 1-D integer array, got {array.dtype} {array.shape}")
    if int(array.min()) < 0 or int(array.max()) >= CODEC_SIZE:
        raise ValueError(f"{label}: minted codes must be in [0, {CODEC_SIZE})")
    return torch.from_numpy(array.astype(np.int64, copy=True))


def _load_codes_file(path: str, label: str) -> torch.Tensor:
    if path.lower().endswith(".npy"):
        return _as_codes(np.load(path, allow_pickle=False), label)
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        tensors = load_file(path)
        for key in ("codes", "codec", "tokens"):
            if key in tensors:
                return _as_codes(tensors[key], label)
        if len(tensors) == 1:
            return _as_codes(next(iter(tensors.values())), label)
        raise ValueError(f"{label}: {path} holds several tensors and none is named codes/codec/tokens")
    raise ValueError(f"{label}: unsupported codes file {path} (use .npy or .safetensors)")


def _split_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return str(value) in VALIDATION_SPLITS


def _numpy_safe_globals() -> list:
    """Allowlist for ``weights_only`` loading of kits holding int32/int64 NumPy arrays; nothing else is admitted."""
    allowed: list = [np.ndarray, np.dtype, type(np.dtype(np.int32)), type(np.dtype(np.int64))]
    # the array reconstructor lives under a different module name in NumPy 1.x and 2.x
    for module_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        allowed.append((module._reconstruct, f"{module_name}._reconstruct"))
    return allowed


def _read_kit_pack(path: str) -> list[MintedRecord]:
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(f"{path}: expected a nonempty minted record list")

    name = os.path.basename(path)
    records = []
    for index, entry in enumerate(payload):
        label = f"{name} record {index}"
        if not (isinstance(entry, dict) and isinstance(entry.get("style"), str) and isinstance(entry.get("lyrics"), str)):
            raise ValueError(f"{label}: invalid minted record")
        src = entry.get("src")
        if src not in ("minted", "minted_val"):
            raise ValueError(f"{label}: src must be minted or minted_val")
        song_id = next((entry[k] for k in ("song_id", "song", "name") if entry.get(k)), None)
        song_id = str(song_id) if song_id is not None else f"minted-{index:06d}"
        codes = _as_codes(entry["codec"], label)
        records.append(MintedRecord(codes, entry["style"], entry["lyrics"], song_id, src == "minted_val"))
    return records


def _read_manifest(path: str) -> list[MintedRecord]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    songs = manifest.get("songs") if isinstance(manifest, dict) else None
    if not songs:
        raise ValueError(f"{path}: expected a manifest with a nonempty songs list")

    base = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    records = []
    for index, song in enumerate(songs):
        label = f"{name} song {index}"
        if not song.get("true_tokens"):
            raise ValueError(
                f"{label}: true_tokens is not set; replay songs must be codec streams the base model sampled itself, "
                "not teacher-forced predictions"
            )
        tokens = song.get("tokens")
        if not isinstance(tokens, str) or not tokens:
            raise ValueError(f"{label}: missing tokens file")
        codes_path = tokens if os.path.isabs(tokens) else os.path.join(base, tokens)
        codes = _load_codes_file(codes_path, label)
        song_id = next((song[k] for k in ("song", "name") if song.get(k)), None)
        song_id = str(song_id) if song_id is not None else f"minted-{index:06d}"
        style = str(song.get("style") or "")
        lyrics = str(song.get("lyrics") or "")
        records.append(MintedRecord(codes, style, lyrics, song_id, _split_flag(song.get("split"))))
    return records


def _read_jsonl(path: str) -> list[MintedRecord]:
    base = os.path.dirname(os.path.abspath(path))
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            label = f"{os.path.basename(path)} line {line_no}"
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{label}: every line must be a JSON object")
            codes = next((data[k] for k in ("codes", "codec", "tokens") if data.get(k) is not None), None)
            if codes is None:
                raise ValueError(f"{label}: codes are required")
            if isinstance(codes, str):
                codes = _load_codes_file(codes if os.path.isabs(codes) else os.path.join(base, codes), label)
            else:
                codes = _as_codes(codes, label)
            style = data.get("style", data.get("caption"))
            lyrics = data.get("lyrics")
            if not isinstance(style, str) or (lyrics is not None and not isinstance(lyrics, str)):
                raise ValueError(f"{label}: style (or caption) must be a string, lyrics a string or null")
            song_id = str(data.get("song_id") or f"minted-{line_no:06d}")
            split = data.get("split", data.get("src"))
            records.append(MintedRecord(codes, style, lyrics or "", song_id, _split_flag(split)))
    if not records:
        raise ValueError(f"{path}: no minted records")
    return records


def read_minted_records(path: str) -> list[MintedRecord]:
    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        return _read_kit_pack(path)
    if suffix == ".json":
        return _read_manifest(path)
    if suffix == ".jsonl":
        return _read_jsonl(path)
    raise ValueError(f"unsupported minted pack {path}: use a .pt kit, a .json manifest, or a .jsonl")


class YuE2MintedPack:
    """Minted AR replay pool. ``tokenizer_fp`` (optional) pins the tokenizer the texts must be tokenized with."""

    def __init__(
        self, path: Optional[str], tokenizer_fp: Optional[str] = None, *, records: Optional[Sequence[MintedRecord]] = None
    ):
        if records is None:
            if path is None:
                raise ValueError("a minted pack needs a path or records")
            records = read_minted_records(path)
        self.path = path
        self.tokenizer_fp = tokenizer_fp
        self.records: list[MintedRecord] = list(records)
        self.text_ids: Optional[list[dict[str, list[int]]]] = None
        self.instrumental_lyrics: Optional[str] = None
        if path is not None:
            logger.info(f"minted pack {path}: {len(self.records)} songs, {sum(r.codes.numel() for r in self.records)} frames")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def fingerprint(self) -> str:
        """sha256 over the codes and texts (recorded in the training metadata)."""
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(record.codes.numpy().tobytes())
            digest.update(json.dumps([record.style, record.lyrics, record.song_id], ensure_ascii=False).encode("utf-8"))
        return digest.hexdigest()

    def tokenize(self, tokenizer, instrumental_lyrics: str = DEFAULT_INSTRUMENTAL_LYRICS) -> None:
        """Tokenizes every record's prompt for every cot mode (``[EOD] + prompt ids``), with the text-cache normalisation."""
        tokenizer_fp = getattr(tokenizer, "fingerprint", None)
        if self.tokenizer_fp is not None and tokenizer_fp is not None and tokenizer_fp != self.tokenizer_fp:
            raise ValueError(f"minted pack expects tokenizer {self.tokenizer_fp}, got {tokenizer_fp}")
        table = []
        for record in self.records:
            style, lyrics = normalize_prompt_fields(record.style, record.lyrics, instrumental_lyrics)
            table.append({cot: text_ids(tokenizer, style, lyrics, cot) for cot in COT_MODES})
        self.text_ids = table
        self.instrumental_lyrics = instrumental_lyrics
        if tokenizer_fp is not None:
            self.tokenizer_fp = tokenizer_fp

    def item(self, index: int) -> MintedItem:
        record = self.records[index]
        texts = self.text_ids[index] if self.text_ids is not None else {}
        return MintedItem(codes=record.codes, text_ids=texts, song_id=record.song_id, style=record.style, lyrics=record.lyrics)

    def sample(self, rng) -> MintedItem:
        """A uniformly drawn item; ``rng`` is a ``random.Random`` (anything with ``randrange``)."""
        if not self.records:
            raise ValueError("minted pack is empty")
        if self.text_ids is None:
            raise ValueError("minted pack texts are not tokenized; call tokenize(tokenizer) first")
        return self.item(rng.randrange(len(self.records)))

    def _subset(self, indices: Sequence[int]) -> "YuE2MintedPack":
        pack = YuE2MintedPack(None, self.tokenizer_fp, records=[self.records[i] for i in indices])
        pack.path = self.path
        if self.text_ids is not None:
            pack.text_ids = [self.text_ids[i] for i in indices]
            pack.instrumental_lyrics = self.instrumental_lyrics
        return pack

    def split(self, validation_fraction: float, seed: int = 0) -> tuple["YuE2MintedPack", "YuE2MintedPack"]:
        """(train, validation). A source that names splits (``src: minted_val``) is split as named; otherwise songs are
        held out by song id with the audio dataset's rule (``validation_song_split``)."""
        if any(r.is_validation is not None for r in self.records):
            held_indices = [i for i, r in enumerate(self.records) if r.is_validation]
        else:
            from musubi_tuner.dataset.audio_dataset import validation_song_split

            held = validation_song_split((r.song_id for r in self.records), validation_fraction, seed)
            held_indices = [i for i, r in enumerate(self.records) if r.song_id in held]
        held_set = set(held_indices)
        train_indices = [i for i in range(len(self.records)) if i not in held_set]
        return self._subset(train_indices), self._subset(held_indices)
