from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from musubi_tuner.minimax_h3.architecture import VIDEO_FPS, H3TemporalShape, temporal_shape


class ReferenceKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ReferenceRole(str, Enum):
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    KEYFRAME = "keyframe"
    REFERENCE = "reference"


SUPPORTED_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")


@dataclass(frozen=True)
class H3Reference:
    path: Path
    kind: ReferenceKind
    role: ReferenceRole = ReferenceRole.REFERENCE
    #: Latent frame the image conditions, for ``KEYFRAME`` only. The transformer
    #: places a conditioning row at its temporal position, and nothing in the
    #: packing distinguishes an interior position from the first or the last.
    latent_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.role in (ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME, ReferenceRole.KEYFRAME):
            if self.kind is not ReferenceKind.IMAGE:
                raise ValueError(f"{self.role.value} must be an image")
        if self.role is ReferenceRole.KEYFRAME:
            if self.latent_index is None:
                raise ValueError("keyframe conditioning requires a latent index")
            if isinstance(self.latent_index, bool) or not isinstance(self.latent_index, int):
                raise ValueError("keyframe latent index must be an integer")
        elif self.latent_index is not None:
            raise ValueError("latent_index applies only to keyframe conditioning")


@dataclass(frozen=True)
class H3Guide:
    """A video and/or audio guide beginning at a decoded pixel frame."""

    frame_index: int
    image: Path | None = None
    video: Path | None = None
    audio: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ValueError("guide frame index must be an integer")
        if self.image is None and self.video is None and self.audio is None:
            raise ValueError("a guide requires image, video, or audio media")
        if self.image is not None and self.video is not None:
            raise ValueError("a guide pixel frame accepts an image or video, not both")
        for name in ("image", "video", "audio"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


@dataclass(frozen=True)
class H3GenerationRequest:
    prompt: str
    output: Path
    duration: int = 5
    ratio: str = "16:9"
    seed: int = 42
    references: tuple[H3Reference, ...] = field(default_factory=tuple)
    frame_count_override: int | None = None
    selected_frame: int = 0
    guides: tuple[H3Guide, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "guides", tuple(self.guides))
        self.validate()

    def canvas_reference(self) -> Path | None:
        """Image whose proportions govern the canvas under ``adaptive``.

        Precedence follows what the viewer sees first and longest: the opening
        frame, then the closing one, then the ordinary reference images in the
        order they were given. Video and audio references are ignored -- a video
        reference is resampled to its own geometry and never dictates the
        output canvas.

        Returns ``None`` for a text-only request, which has nothing to adapt to.
        """
        for role in (ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME, ReferenceRole.KEYFRAME, ReferenceRole.REFERENCE):
            for reference in self._with_role(role):
                if reference.kind is ReferenceKind.IMAGE:
                    return reference.path
        return None

    def validate(self, *, check_files: bool = False) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.frame_count_override is None and (
            not isinstance(self.duration, int) or isinstance(self.duration, bool) or not 5 <= self.duration <= 15
        ):
            raise ValueError("duration must be an integer from 5 through 15 seconds")
        if self.frame_count_override is not None and (
            isinstance(self.frame_count_override, bool)
            or not isinstance(self.frame_count_override, int)
            or self.frame_count_override < 5
            or (self.frame_count_override - 5) % 17
        ):
            raise ValueError("frame_count_override must satisfy frame_count % 17 == 5")
        if isinstance(self.selected_frame, bool) or not isinstance(self.selected_frame, int) or self.selected_frame < 0:
            raise ValueError("selected_frame must be a non-negative integer")
        if self.ratio not in SUPPORTED_RATIOS:
            raise ValueError(f"ratio must be one of: {', '.join(SUPPORTED_RATIOS)}")

        first_frames = self._with_role(ReferenceRole.FIRST_FRAME)
        last_frames = self._with_role(ReferenceRole.LAST_FRAME)
        keyframes = self._with_role(ReferenceRole.KEYFRAME)
        if len(first_frames) > 1 or len(last_frames) > 1:
            raise ValueError("at most one first-frame and one last-frame image are supported")
        latent_frames = self.temporal_shape.video_latent_frames
        indices = [self.resolve_keyframe_index(ref.latent_index) for ref in keyframes]
        if len(set(indices)) != len(indices):
            raise ValueError("each keyframe latent index may be conditioned only once")
        if any(not 0 <= index < latent_frames for index in indices):
            raise ValueError(f"keyframe latent index must resolve inside the {latent_frames}-frame latent timeline")
        if first_frames and 0 in indices:
            raise ValueError("the first frame may be conditioned only once")
        guide_indices = [
            guide.frame_index if guide.frame_index >= 0 else self.temporal_shape.frame_count + guide.frame_index
            for guide in self.guides
        ]
        if any(not 0 <= index < self.temporal_shape.frame_count for index in guide_indices):
            raise ValueError(f"guide frame index must resolve inside the {self.temporal_shape.frame_count}-frame pixel timeline")
        if len(set(guide_indices)) != len(guide_indices):
            raise ValueError("each pixel frame may begin only one guide")

        ordinary = self._with_role(ReferenceRole.REFERENCE)
        counts = {kind: sum(ref.kind is kind for ref in ordinary) for kind in ReferenceKind}
        if counts[ReferenceKind.IMAGE] > 9:
            raise ValueError("at most 9 reference images are supported")
        if counts[ReferenceKind.VIDEO] > 3:
            raise ValueError("at most 3 reference videos are supported")
        if counts[ReferenceKind.AUDIO] > 3:
            raise ValueError("at most 3 reference audio clips are supported")
        if len(ordinary) > 12:
            raise ValueError("at most 12 references are supported")
        if ordinary and all(ref.kind is ReferenceKind.AUDIO for ref in ordinary):
            raise ValueError("reference audio requires at least one reference image or video")
        if check_files:
            paths = [ref.path for ref in self.references]
            paths.extend(path for guide in self.guides for path in (guide.image, guide.video, guide.audio) if path is not None)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"reference file(s) not found: {', '.join(missing)}")

    def _with_role(self, role: ReferenceRole) -> tuple[H3Reference, ...]:
        return tuple(ref for ref in self.references if ref.role is role)

    def resolve_keyframe_index(self, index: int | None) -> int:
        """Resolve a possibly negative keyframe index against the latent timeline."""
        if index is None:
            raise ValueError("keyframe conditioning requires a latent index")
        return index if index >= 0 else self.temporal_shape.video_latent_frames + index

    @property
    def mode(self) -> str:
        if self.guides:
            return "reference"
        roles = {ref.role for ref in self.references}
        if ReferenceRole.REFERENCE in roles:
            return "reference"
        if roles & {ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME, ReferenceRole.KEYFRAME}:
            return "first_last_frame"
        return "text_to_video"

    @property
    def temporal_shape(self) -> H3TemporalShape:
        return temporal_shape(
            self.frame_count_override if self.frame_count_override is not None else round(self.duration * VIDEO_FPS),
            align=self.frame_count_override is None,
        )


def make_references(
    *,
    first_frame: str | None = None,
    last_frame: str | None = None,
    keyframes: Iterable[tuple[int, str]] = (),
    images: Iterable[str] = (),
    videos: Iterable[str] = (),
    audio: Iterable[str] = (),
) -> tuple[H3Reference, ...]:
    references: list[H3Reference] = []
    if first_frame:
        references.append(H3Reference(Path(first_frame), ReferenceKind.IMAGE, ReferenceRole.FIRST_FRAME))
    if last_frame:
        references.append(H3Reference(Path(last_frame), ReferenceKind.IMAGE, ReferenceRole.LAST_FRAME))
    references.extend(
        H3Reference(Path(path), ReferenceKind.IMAGE, ReferenceRole.KEYFRAME, latent_index=int(index)) for index, path in keyframes
    )
    references.extend(H3Reference(Path(path), ReferenceKind.IMAGE) for path in images)
    references.extend(H3Reference(Path(path), ReferenceKind.VIDEO) for path in videos)
    references.extend(H3Reference(Path(path), ReferenceKind.AUDIO) for path in audio)
    return tuple(references)
