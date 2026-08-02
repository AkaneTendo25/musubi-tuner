from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from musubi_tuner.minimax_h3.architecture import H3TemporalShape, VIDEO_FPS, temporal_shape


class ReferenceKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ReferenceRole(str, Enum):
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    REFERENCE = "reference"


SUPPORTED_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")


@dataclass(frozen=True)
class H3Reference:
    path: Path
    kind: ReferenceKind
    role: ReferenceRole = ReferenceRole.REFERENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.role in (ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME) and self.kind is not ReferenceKind.IMAGE:
            raise ValueError(f"{self.role.value} must be an image")


@dataclass(frozen=True)
class H3GenerationRequest:
    prompt: str
    output: Path
    duration: int = 5
    ratio: str = "16:9"
    seed: int = 42
    references: tuple[H3Reference, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "references", tuple(self.references))
        self.validate()

    def validate(self, *, check_files: bool = False) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(self.prompt) > 7000:
            raise ValueError("prompt must contain at most 7000 characters")
        if not isinstance(self.duration, int) or isinstance(self.duration, bool) or not 5 <= self.duration <= 15:
            raise ValueError("duration must be an integer from 5 through 15 seconds")
        if self.ratio not in SUPPORTED_RATIOS:
            raise ValueError(f"ratio must be one of: {', '.join(SUPPORTED_RATIOS)}")

        first_frames = self._with_role(ReferenceRole.FIRST_FRAME)
        last_frames = self._with_role(ReferenceRole.LAST_FRAME)
        if len(first_frames) > 1 or len(last_frames) > 1:
            raise ValueError("at most one first-frame and one last-frame image are supported")

        ordinary = self._with_role(ReferenceRole.REFERENCE)
        counts = {kind: sum(ref.kind is kind for ref in ordinary) for kind in ReferenceKind}
        if counts[ReferenceKind.IMAGE] > 9:
            raise ValueError("at most 9 reference images are supported")
        if counts[ReferenceKind.VIDEO] > 3:
            raise ValueError("at most 3 reference videos are supported")
        if counts[ReferenceKind.AUDIO] > 3:
            raise ValueError("at most 3 reference audio clips are supported")
        if len(ordinary) > 12:
            raise ValueError("at most 12 ordinary references are supported in total")
        if counts[ReferenceKind.AUDIO] and not (counts[ReferenceKind.IMAGE] or counts[ReferenceKind.VIDEO]):
            raise ValueError("reference audio must be paired with at least one reference image or video")
        if (first_frames or last_frames) and ordinary:
            raise ValueError("first/last-frame conditioning and ordinary multimodal references are separate H3 modes")

        if check_files:
            missing = [str(ref.path) for ref in self.references if not ref.path.is_file()]
            if missing:
                raise FileNotFoundError(f"reference file(s) not found: {', '.join(missing)}")

    def _with_role(self, role: ReferenceRole) -> tuple[H3Reference, ...]:
        return tuple(ref for ref in self.references if ref.role is role)

    @property
    def mode(self) -> str:
        roles = {ref.role for ref in self.references}
        if roles & {ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME}:
            return "first_last_frame"
        if self.references:
            return "reference"
        return "text_to_video"

    @property
    def temporal_shape(self) -> H3TemporalShape:
        return temporal_shape(round(self.duration * VIDEO_FPS), align=True)


def make_references(
    *,
    first_frame: str | None = None,
    last_frame: str | None = None,
    images: Iterable[str] = (),
    videos: Iterable[str] = (),
    audio: Iterable[str] = (),
) -> tuple[H3Reference, ...]:
    references: list[H3Reference] = []
    if first_frame:
        references.append(H3Reference(Path(first_frame), ReferenceKind.IMAGE, ReferenceRole.FIRST_FRAME))
    if last_frame:
        references.append(H3Reference(Path(last_frame), ReferenceKind.IMAGE, ReferenceRole.LAST_FRAME))
    references.extend(H3Reference(Path(path), ReferenceKind.IMAGE) for path in images)
    references.extend(H3Reference(Path(path), ReferenceKind.VIDEO) for path in videos)
    references.extend(H3Reference(Path(path), ReferenceKind.AUDIO) for path in audio)
    return tuple(references)
