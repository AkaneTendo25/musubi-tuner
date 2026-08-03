from __future__ import annotations

from dataclasses import dataclass

VIDEO_FPS = 24
VIDEO_LATENT_CHANNELS = 24
VIDEO_SPATIAL_COMPRESSION = 16
VIDEO_DIT_PATCH_SIZE = (1, 2, 2)
CANVAS_MULTIPLE = VIDEO_SPATIAL_COMPRESSION * VIDEO_DIT_PATCH_SIZE[-1]

AUDIO_SAMPLE_RATE = 32_000
AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
AUDIO_HOP_LENGTH = 800
AUDIO_LATENT_FPS = AUDIO_SAMPLE_RATE // AUDIO_HOP_LENGTH

FRAME_GRID_PERIOD = 17
FRAME_GRID_REMAINDER = 5

VIDEO_FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0
TEXT_DIM = 5120


@dataclass(frozen=True)
class H3TemporalShape:
    frame_count: int
    video_latent_frames: int
    audio_latent_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / VIDEO_FPS

    @property
    def audio_samples(self) -> int:
        return self.audio_latent_frames * AUDIO_HOP_LENGTH


def is_valid_frame_count(frame_count: int) -> bool:
    return (
        isinstance(frame_count, int)
        and not isinstance(frame_count, bool)
        and frame_count >= FRAME_GRID_REMAINDER
        and frame_count % FRAME_GRID_PERIOD == FRAME_GRID_REMAINDER
    )


def align_frame_count(frame_count: int) -> int:
    frame_count = max(frame_count, FRAME_GRID_REMAINDER)
    remainder = frame_count % FRAME_GRID_PERIOD
    return frame_count + (FRAME_GRID_REMAINDER - remainder) % FRAME_GRID_PERIOD


def temporal_shape(frame_count: int, *, align: bool = False) -> H3TemporalShape:
    if not isinstance(frame_count, int) or isinstance(frame_count, bool):
        raise TypeError("H3 frame_count must be an integer")
    if align:
        frame_count = align_frame_count(frame_count)
    elif not is_valid_frame_count(frame_count):
        raise ValueError("H3 frame_count must be at least 5 and satisfy frame_count % 17 == 5")

    video_latent_frames = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    # Exact nearest-grid form of frame_count / 24 * 40 for the valid 17n+5
    # frame grid. Integer arithmetic keeps cache identities platform-stable.
    audio_latent_frames = (10 * frame_count + 3) // 6
    return H3TemporalShape(frame_count, video_latent_frames, audio_latent_frames)
