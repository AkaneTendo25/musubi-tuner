"""TARP (Temporally Aligned RoPE and Partitioning) + DCR (Dynamic Context Routing).

Training-only techniques from "Improving Joint Audio-Video Generation with
Cross-Modal Context Learning" (arXiv:2603.18600v1).

- TARP: Windowed cross-attention mask restricting each video frame to nearby audio tokens.
- DCR: Per-sample gradient detachment for mixed audio/video batches.
"""

from __future__ import annotations

import torch


def compute_tarp_a2v_mask(
    video_frames: int,
    video_spatial_tokens: int,
    audio_seq_len: int,
    window_multiplier: int = 3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Compute windowed A2V cross-attention mask (TARP partitioning).

    Each video frame's spatial tokens attend only to a local window of audio
    tokens centred on the temporally corresponding position.

    The mask uses additive format: ``0.0`` = attend, ``-inf`` = block.
    This matches the mask convention used by the LTX-2 attention code.

    Args:
        video_frames: Number of latent video frames.
        video_spatial_tokens: Spatial tokens per frame (after patching).
        audio_seq_len: Total audio token count.
        window_multiplier: Window size = multiplier * (audio_tokens_per_frame).
        device: Target device for the mask tensor.
        dtype: Mask dtype (must be floating-point).

    Returns:
        Additive mask [1, video_seq_len, audio_seq_len] or None when
        windowing is not applicable.
    """
    if video_frames <= 0 or audio_seq_len <= 0 or video_spatial_tokens <= 0:
        return None

    c = audio_seq_len // video_frames  # audio tokens per video frame
    if c <= 0:
        return None

    s = window_multiplier * c  # window size
    video_seq_len = video_frames * video_spatial_tokens
    neg_inf = torch.finfo(dtype).min

    # Start with all blocked, open windows
    mask = torch.full(
        (video_seq_len, audio_seq_len), neg_inf, dtype=dtype, device=device
    )

    for i in range(video_frames):
        m_i = c // 2 + c * i  # window centre
        win_start = max(0, m_i - s // 2)
        win_end = min(audio_seq_len, m_i + (s + 1) // 2)

        vid_start = i * video_spatial_tokens
        vid_end = vid_start + video_spatial_tokens
        mask[vid_start:vid_end, win_start:win_end] = 0.0

    return mask.unsqueeze(0)  # [1, video_seq_len, audio_seq_len]


def compute_tarp_v2a_mask(
    video_frames: int,
    video_spatial_tokens: int,
    audio_seq_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Compute V2A cross-attention mask (TARP nearest-neighbour partitioning).

    Each audio token attends only to the spatially-complete nearest video frame
    (window size s=1 in the paper's terminology). This restricts temporal scope
    while preserving full spatial context per frame.

    The mask uses additive format: ``0.0`` = attend, ``-inf`` = block.

    Args:
        video_frames: Number of latent video frames.
        video_spatial_tokens: Spatial tokens per frame (after patching).
        audio_seq_len: Total audio token count.
        device: Target device for the mask tensor.
        dtype: Mask dtype (must be floating-point).

    Returns:
        Additive mask [1, audio_seq_len, video_seq_len] or None when not
        applicable.
    """
    if video_frames <= 0 or audio_seq_len <= 0 or video_spatial_tokens <= 0:
        return None

    video_seq_len = video_frames * video_spatial_tokens
    neg_inf = torch.finfo(dtype).min

    mask = torch.full(
        (audio_seq_len, video_seq_len), neg_inf, dtype=dtype, device=device
    )

    # Nearest-neighbour: audio token j maps to video frame round(j * t_v / t_a)
    for j in range(audio_seq_len):
        frame_idx = min((j * video_frames) // audio_seq_len, video_frames - 1)
        vid_start = frame_idx * video_spatial_tokens
        vid_end = vid_start + video_spatial_tokens
        mask[j, vid_start:vid_end] = 0.0

    return mask.unsqueeze(0)  # [1, audio_seq_len, video_seq_len]
