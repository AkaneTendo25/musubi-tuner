from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import toml
import torch
from safetensors.torch import load_file

from musubi_tuner.minimax_h3.cache import H3_AUDIO_LATENTS_KEY, H3_VIDEO_GEOMETRY_KEY
from musubi_tuner.utils.model_utils import remove_dtype_suffix

H3SliderMode = Literal["text", "reference", "ref2va"]
H3SliderTargetModality = Literal["video", "audio", "av"]


@dataclass(frozen=True)
class H3SliderTarget:
    positive: str
    negative: str
    target_class: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class H3SliderAnchor:
    prompt: str


@dataclass(frozen=True)
class H3SliderConfig:
    mode: H3SliderMode = "text"
    target_modality: H3SliderTargetModality = "av"
    guidance_strength: float = 3.0
    targets: tuple[H3SliderTarget, ...] = ()
    anchors: tuple[H3SliderAnchor, ...] = ()
    anchor_strength: float = 1.0
    anchor_cap_mult: float = 5.0
    batch_all_targets: bool = False
    sample_slider_range: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    positive_cache_dir: str | None = None
    negative_cache_dir: str | None = None
    conditioning_cache_dir: str | None = None
    latent_frames: int = 1
    latent_height: int = 24
    latent_width: int = 40
    audio_latent_frames: int = 1


def load_h3_slider_config(path: str | os.PathLike[str]) -> H3SliderConfig:
    raw = toml.load(path)
    mode = str(raw.get("mode", "text"))
    target_modality = str(raw.get("target_modality", "av"))
    if mode not in {"text", "reference", "ref2va"}:
        raise ValueError("H3 slider mode must be text, reference, or ref2va")
    if target_modality not in {"video", "audio", "av"}:
        raise ValueError("H3 slider target_modality must be video, audio, or av")

    targets = tuple(
        H3SliderTarget(
            positive=str(value["positive"]),
            negative=str(value["negative"]),
            target_class=str(value.get("target_class", "")),
            weight=float(value.get("weight", 1.0)),
        )
        for value in raw.get("targets", ())
    )
    anchors = tuple(H3SliderAnchor(prompt=str(value["prompt"])) for value in raw.get("anchors", ()))
    slider_range = raw.get("sample_slider_range", (-2.0, -1.0, 0.0, 1.0, 2.0))
    if isinstance(slider_range, (int, float)):
        slider_range = (float(slider_range),)
    elif isinstance(slider_range, str):
        slider_range = tuple(float(value.strip()) for value in slider_range.split(",") if value.strip())
    else:
        slider_range = tuple(float(value) for value in slider_range)

    config = H3SliderConfig(
        mode=mode,
        target_modality=target_modality,
        guidance_strength=float(raw.get("guidance_strength", 3.0)),
        targets=targets,
        anchors=anchors,
        anchor_strength=float(raw.get("anchor_strength", 1.0)),
        anchor_cap_mult=float(raw.get("anchor_cap_mult", 5.0)),
        batch_all_targets=bool(raw.get("batch_all_targets", False)),
        sample_slider_range=slider_range,
        positive_cache_dir=raw.get("positive_cache_dir"),
        negative_cache_dir=raw.get("negative_cache_dir"),
        conditioning_cache_dir=raw.get("conditioning_cache_dir"),
        latent_frames=int(raw.get("latent_frames", 1)),
        latent_height=int(raw.get("latent_height", 24)),
        latent_width=int(raw.get("latent_width", 40)),
        audio_latent_frames=int(raw.get("audio_latent_frames", 1)),
    )
    validate_h3_slider_config(config)
    return config


def validate_h3_slider_config(config: H3SliderConfig) -> None:
    if config.guidance_strength <= 0:
        raise ValueError("H3 slider guidance_strength must be positive")
    if config.anchor_strength < 0:
        raise ValueError("H3 slider anchor_strength must be non-negative")
    if config.anchor_cap_mult < 0:
        raise ValueError("H3 slider anchor_cap_mult must be non-negative")
    if not config.sample_slider_range:
        raise ValueError("H3 slider sample_slider_range cannot be empty")
    if config.mode == "text":
        if not config.targets:
            raise ValueError("text H3 sliders require at least one [[targets]] entry")
        if any(not target.target_class.strip() for target in config.targets):
            raise ValueError("text H3 slider targets require a non-empty target_class for neutral conditioning")
        if min(config.latent_frames, config.latent_height, config.latent_width, config.audio_latent_frames) <= 0:
            raise ValueError("text H3 slider latent dimensions must be positive")
        if config.latent_height % 2 or config.latent_width % 2:
            raise ValueError("text H3 slider latent height and width must be divisible by 2")
    else:
        if not config.positive_cache_dir or not config.negative_cache_dir:
            raise ValueError("paired H3 sliders require positive_cache_dir and negative_cache_dir")
        if config.anchors:
            raise ValueError("H3 slider anchors apply only to text mode")
        if config.mode == "ref2va" and not config.conditioning_cache_dir:
            raise ValueError("ref2va H3 sliders require conditioning_cache_dir with the shared reference presentation")


def normalize_slider_direction(source: torch.Tensor, neutral: torch.Tensor) -> torch.Tensor:
    """Match a directional target's mean/std to the neutral prediction."""
    source_float = source.float()
    neutral_float = neutral.float()
    normalized = (source_float - source_float.mean()) / source_float.std(correction=0).clamp_min(1e-8)
    return (normalized * neutral_float.std(correction=0) + neutral_float.mean()).to(source.dtype)


def slider_direction_targets(
    positive: torch.Tensor,
    neutral: torch.Tensor,
    negative: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    direction = positive.float() - negative.float()
    enhance = normalize_slider_direction(neutral.float() + strength * direction, neutral).detach()
    erase = normalize_slider_direction(neutral.float() - strength * direction, neutral).detach()
    return enhance, erase


def _normalized_cache(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    raw = load_file(str(path))
    result: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        variable = key.startswith("varlen_")
        logical = key.removeprefix("varlen_")
        if not logical.endswith("_mask"):
            logical = remove_dtype_suffix(logical)
            if logical.startswith("latents_"):
                logical = logical.rsplit("_", 1)[0]
        result[logical] = value if variable else value.unsqueeze(0)
    return result


_LATENT_NAME = re.compile(r"^(?P<stem>.+)_\d{4}x\d{4}_mmh3\.safetensors$")
_TARGET_KEYS = {"latents", H3_AUDIO_LATENTS_KEY, "video_loss_mask", "audio_loss_mask"}


def _latent_files(cache_dir: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for value in glob.glob(os.path.join(cache_dir, "*_mmh3.safetensors")):
        name = os.path.basename(value)
        match = _LATENT_NAME.match(name)
        if match is None:
            continue
        stem = match.group("stem")
        if stem in files:
            raise ValueError(f"multiple H3 latent caches resolve to slider item {stem!r} in {cache_dir}")
        files[stem] = Path(value)
    return files


def _conditioning_paths(cache_dir: str, stem: str) -> tuple[Path, Path]:
    latent_matches = sorted(Path(cache_dir).glob(f"{stem}_*x*_mmh3.safetensors"))
    if len(latent_matches) != 1:
        raise ValueError(f"expected one conditioning latent cache for {stem!r} in {cache_dir}, found {len(latent_matches)}")
    text_path = Path(cache_dir) / f"{stem}_mmh3_te.safetensors"
    if not text_path.is_file():
        raise FileNotFoundError(f"H3 slider text cache not found: {text_path}")
    return latent_matches[0], text_path


def _require_modalities(batch: dict[str, torch.Tensor], modality: H3SliderTargetModality, label: str) -> None:
    if modality in {"video", "av"} and "latents" not in batch:
        raise ValueError(f"{label} H3 slider cache has no video latents")
    if modality in {"audio", "av"} and H3_AUDIO_LATENTS_KEY not in batch:
        raise ValueError(f"{label} H3 slider cache has no audio latents")


class H3SliderDataset(torch.utils.data.Dataset):
    """Minimal dataset-group contract used by the shared Musubi trainer loop."""

    batch_size = 1

    def __init__(self, config: H3SliderConfig) -> None:
        self.config = config
        # The shared trainer expects a dataset-group-style ``datasets`` list for
        # reporting. Its resume walker recursively descends that list, so the
        # leaf must not be the group object itself.
        self.datasets = [_H3SliderDatasetMetadata(self)]
        self._epoch = 0
        self._max_steps = 0
        if config.mode == "text":
            self.pairs: list[tuple[str, Path, Path]] = []
            self.num_train_items = max(1, len(config.targets))
            return

        positive = _latent_files(str(config.positive_cache_dir))
        negative = _latent_files(str(config.negative_cache_dir))
        common = sorted(set(positive) & set(negative))
        if not common:
            raise ValueError("H3 slider cache directories contain no filename-matched latent pairs")
        missing_positive = sorted(set(negative) - set(positive))
        missing_negative = sorted(set(positive) - set(negative))
        if missing_positive or missing_negative:
            details = []
            if missing_positive:
                details.append("missing positive: " + ", ".join(missing_positive[:5]))
            if missing_negative:
                details.append("missing negative: " + ", ".join(missing_negative[:5]))
            raise ValueError("unmatched H3 slider cache items; " + "; ".join(details))
        self.pairs = [(stem, positive[stem], negative[stem]) for stem in common]
        self.num_train_items = len(self.pairs)

    def __len__(self) -> int:
        return self.num_train_items

    def set_current_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def set_max_train_steps(self, max_train_steps: int) -> None:
        self._max_steps = int(max_train_steps)

    def get_metadata(self) -> dict[str, object]:
        return {
            "architecture": "minimax_h3",
            "slider_mode": self.config.mode,
            "target_modality": self.config.target_modality,
            "num_items": self.num_train_items,
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.config.mode == "text":
            video = None
            audio = None
            if self.config.target_modality in {"video", "av"}:
                video = torch.zeros(
                    1,
                    24,
                    self.config.latent_frames,
                    self.config.latent_height,
                    self.config.latent_width,
                    dtype=torch.float32,
                )
            if self.config.target_modality in {"audio", "av"}:
                audio = torch.zeros(1, 2, 32, self.config.audio_latent_frames, dtype=torch.float32)
            return {"slider_mode": "text", "latents": video, H3_AUDIO_LATENTS_KEY: audio, "timesteps": None}

        stem, positive_path, negative_path = self.pairs[index]
        positive_targets = _normalized_cache(positive_path)
        negative_targets = _normalized_cache(negative_path)
        conditioning_dir = self.config.conditioning_cache_dir or self.config.positive_cache_dir
        conditioning_latent_path, conditioning_text_path = _conditioning_paths(str(conditioning_dir), stem)
        conditioning = {**_normalized_cache(conditioning_latent_path), **_normalized_cache(conditioning_text_path)}
        conditioning = {key: value for key, value in conditioning.items() if key not in _TARGET_KEYS}
        positive_batch = dict(conditioning)
        negative_batch = dict(conditioning)
        selected = {"video_loss_mask", "audio_loss_mask"}
        if self.config.target_modality in {"video", "av"}:
            selected.add("latents")
        if self.config.target_modality in {"audio", "av"}:
            selected.add(H3_AUDIO_LATENTS_KEY)
        positive_batch.update({key: value for key, value in positive_targets.items() if key in selected})
        negative_batch.update({key: value for key, value in negative_targets.items() if key in selected})
        if self.config.target_modality == "audio" and H3_VIDEO_GEOMETRY_KEY not in positive_batch:
            positive_video = positive_targets.get("latents")
            negative_video = negative_targets.get("latents")
            if positive_video is None or negative_video is None:
                raise ValueError(f"audio-only H3 slider cache for {stem} needs video latents or {H3_VIDEO_GEOMETRY_KEY}")
            geometry = torch.tensor(positive_video.shape[-2:], dtype=torch.long).unsqueeze(0)
            positive_batch[H3_VIDEO_GEOMETRY_KEY] = geometry
            negative_batch[H3_VIDEO_GEOMETRY_KEY] = geometry
        positive_batch["timesteps"] = None
        negative_batch["timesteps"] = None
        _require_modalities(positive_batch, self.config.target_modality, "positive")
        _require_modalities(negative_batch, self.config.target_modality, "negative")
        for key in ("latents", H3_AUDIO_LATENTS_KEY):
            if key in positive_batch and key in negative_batch and positive_batch[key].shape != negative_batch[key].shape:
                raise ValueError(
                    f"paired H3 slider {key} shape mismatch for {stem}: "
                    f"{tuple(positive_batch[key].shape)} vs {tuple(negative_batch[key].shape)}"
                )
        return {"slider_mode": self.config.mode, "positive": positive_batch, "negative": negative_batch, "stem": stem}


def slider_collator(examples: list[dict[str, object]]) -> dict[str, object]:
    if len(examples) != 1:
        raise ValueError("H3 slider training requires DataLoader batch size 1")
    return examples[0]


class _H3SliderDatasetMetadata:
    batch_size = 1

    def __init__(self, owner: H3SliderDataset) -> None:
        self._owner = owner

    def get_metadata(self) -> dict[str, object]:
        return self._owner.get_metadata()
