"""Training-only guidance reconstruction.

The released model supplies both frozen fields; this approximates a teacher and
does not recover MiniMax's unavailable original undistilled model.
"""

import hashlib
import math
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import torch
from safetensors import safe_open

from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, logical_cache_key
from musubi_tuner.minimax_h3.training import H3JointLoss, H3ModelPrediction


def parse_scale_schedule(text: str) -> tuple[float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except (ValueError, AttributeError) as error:
        raise ValueError("--h3_repair_scale must be constant or intercept,slope") from error
    if len(values) not in (1, 2) or not all(math.isfinite(value) for value in values):
        raise ValueError("--h3_repair_scale must contain one or two finite numbers")
    if (len(values) == 1 or values[1] == 0) and values[0] < 1:
        raise ValueError("constant --h3_repair_scale must be at least 1")
    return values[0], values[1] if len(values) == 2 else 0.0


def scale_at(schedule: tuple[float, float], sigma: torch.Tensor) -> torch.Tensor:
    return (schedule[0] + schedule[1] * sigma.float()).clamp_min(1.0)


def interpolate_schedule(values, sigma: torch.Tensor) -> torch.Tensor:
    """Linear at .40/.95 or piecewise linear at .40/.80/.95, clamped outside."""
    value = sigma.float()
    if len(values) == 2:
        return values[0] + (values[1] - values[0]) * ((value - 0.40) / 0.55).clamp(0, 1)
    low = values[0] + (values[1] - values[0]) * ((value - 0.40) / 0.40).clamp(0, 1)
    high = values[1] + (values[2] - values[1]) * ((value - 0.80) / 0.15).clamp(0, 1)
    return torch.where(value <= 0.80, low, high)


def align_anchor_hidden(anchor: torch.Tensor, empty_hidden: torch.Tensor) -> torch.Tensor:
    if anchor.ndim != 2 or empty_hidden.ndim != 2:
        raise ValueError(
            f"anchor and empty hidden states must be [rows, dim], got {tuple(anchor.shape)} and {tuple(empty_hidden.shape)}"
        )
    if anchor.shape[-1] != empty_hidden.shape[-1]:
        raise ValueError(f"anchor width {anchor.shape[-1]} does not match the empty presentation {empty_hidden.shape[-1]}")
    rows = empty_hidden.shape[0]
    anchor = anchor.to(device=empty_hidden.device, dtype=empty_hidden.dtype)
    if anchor.shape[0] >= rows:
        return anchor[:rows].clone()
    return torch.cat((anchor, empty_hidden[anchor.shape[0] :]), dim=0)


def anchor_hidden_states(anchors: Iterable[torch.Tensor], empty_hidden: torch.Tensor) -> list[torch.Tensor]:
    return [align_anchor_hidden(anchor, empty_hidden) for anchor in anchors]


def caption_marginal_reference(predictions: Sequence[H3ModelPrediction]) -> H3ModelPrediction:
    if not predictions:
        raise ValueError("the caption-marginal reference needs at least one anchor prediction")
    first = predictions[0]
    for other in predictions[1:]:
        if (other.video is None) != (first.video is None) or (other.audio is None) != (first.audio is None):
            raise ValueError("every caption anchor must predict the same modalities")

    def mean(name: str) -> torch.Tensor | None:
        head = getattr(first, name)
        if head is None:
            return None
        total = head.detach().float()
        for other in predictions[1:]:
            value = getattr(other, name)
            if value.shape != head.shape:
                raise ValueError(f"caption anchor {name} predictions must have matching shapes")
            total = total + value.detach().float()
        return (total / len(predictions)).to(head.dtype)

    return H3ModelPrediction(video=mean("video"), audio=mean("audio"))


def unbiased_caption_repair_loss(empty: H3JointLoss, caption: H3JointLoss, probability: float) -> H3JointLoss:
    """Return the active-step Horvitz-Thompson correction around the empty baseline."""
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("caption repair probability must be finite and lie in (0, 1]")
    return replace(
        empty,
        loss=empty.loss + (caption.loss - empty.loss) / probability,
        video_loss=empty.video_loss + (caption.video_loss - empty.video_loss) / probability,
        audio_loss=empty.audio_loss + (caption.audio_loss - empty.audio_loss) / probability,
    )


class H3CaptionAnchorPool:
    def __init__(self, paths: Sequence[str], *, cache_size: int = 64) -> None:
        if not paths:
            raise ValueError("the caption-marginal reference found no cached text-encoder outputs to draw anchors from")
        if cache_size < 1:
            raise ValueError(f"anchor cache size must be positive, got {cache_size}")
        canonical: dict[str, str] = {}
        for value in paths:
            resolved = str(os.path.realpath(os.path.abspath(os.path.expanduser(value))))
            canonical.setdefault(os.path.normcase(resolved), resolved)
        self._paths = sorted(canonical.values(), key=os.path.normcase)
        self._cache_size = cache_size
        self._cache: dict[str, torch.Tensor] = {}
        self._identity: str | None = None

    @classmethod
    def from_dataset_group(cls, dataset_group: Any, *, cache_size: int = 64):
        paths: list[str] = []
        seen: set[str] = set()
        for dataset in getattr(dataset_group, "datasets", []):
            lister = getattr(dataset, "get_all_text_encoder_output_cache_files", None)
            if not callable(lister):
                continue
            for path in lister():
                resolved = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
                identity = os.path.normcase(resolved)
                if identity not in seen:
                    seen.add(identity)
                    paths.append(resolved)
        return cls(sorted(paths), cache_size=cache_size)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def identity(self) -> str:
        return self.compute_identity()

    def compute_identity(self, *, refresh: bool = False) -> str:
        if self._identity is None or refresh:
            digest = hashlib.sha256(b"h3-caption-anchor-pool-v1\0")
            for path in self._paths:
                digest.update(path.encode("utf-8"))
                digest.update(b"\0")
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
            self._identity = digest.hexdigest()
        return self._identity

    def _read(self, path: str) -> torch.Tensor:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        with safe_open(path, framework="pt", device="cpu") as handle:
            matches = [key for key in list(handle.keys()) if logical_cache_key(key) == H3_TEXT_HIDDEN_KEY]
            if len(matches) != 1:
                raise ValueError(f"H3 text-encoder cache {path} must carry exactly one {H3_TEXT_HIDDEN_KEY} tensor")
            # Own the storage so a bounded cache entry does not keep the
            # safetensors mapping open and prevent mutation detection on Windows.
            hidden = handle.get_tensor(matches[0]).clone()
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = hidden
        return hidden

    def draw(self, count: int, *, generator: random.Random, exclude: torch.Tensor | None = None) -> list[torch.Tensor]:
        if count < 1:
            raise ValueError(f"the caption-marginal reference needs at least one anchor, got {count}")
        paths = list(self._paths)
        generator.shuffle(paths)
        chosen = []
        for path in paths:
            hidden = self._read(path)
            if (
                exclude is not None
                and hidden.shape == exclude.shape
                and torch.equal(hidden, exclude.detach().to(device=hidden.device, dtype=hidden.dtype))
            ):
                continue
            chosen.append(hidden)
            if len(chosen) == count:
                return chosen
        raise ValueError(f"only {len(chosen)} of {count} caption anchors were usable; the corpus is too small")


def substituted_caption_batch(batch: dict[str, Any], empty_hidden_key: str, replacement: torch.Tensor) -> dict[str, Any]:
    clone = dict(batch)
    clone[empty_hidden_key] = [replacement]
    return clone


def guidance_recovered_target(prompted, reference, scale, multiplier=None):
    if prompted.shape != reference.shape:
        raise ValueError("repair prompted and reference predictions must have matching shapes")
    scale = torch.as_tensor(scale, device=prompted.device, dtype=torch.float32)
    if not bool(torch.isfinite(scale).all()) or bool((scale < 1).any()):
        raise ValueError("repair scale must be finite and at least 1")
    if scale.ndim and scale.numel() != prompted.shape[0]:
        raise ValueError("repair scale must have one value per sample")
    fraction = 1 - scale.reciprocal()
    if multiplier is not None:
        fraction = fraction * multiplier.to(device=prompted.device, dtype=torch.float32)
        if not bool(torch.isfinite(fraction).all()) or bool((fraction > 1 + 1e-7).any()):
            raise ValueError("--h3_repair_target_multiplier extrapolates past the reference branch")
    fraction = fraction.reshape(-1, *([1] * (prompted.ndim - 1)))
    return (prompted.detach().float() + fraction * (reference.detach().float() - prompted.detach().float())).to(prompted.dtype)
