"""
VACE control signal preparation for LTX-2.

Supports all VACE task types through a unified control signal format:
  - T2V: empty control, all-ones mask (generate everything)
  - R2V (Reference): reference images prepended, mask=0 for reference frames
  - V2V (Control): depth/pose/scribble/flow/layout maps, mask=1 everywhere
  - V2V (General): source video as control, mask=1 everywhere
  - MV2V (Inpaint): original video with masked region zeroed, mask=1 for inpaint region
  - MV2V (Outpaint): original video padded, mask=1 for extended region
  - MV2V (Extension): known frames + zeros, mask=1 for frames to generate

All tasks produce the same output format: (B, vace_in_dim, F', H', W')
where vace_in_dim = 2 * latent_channels + mask_channels.

Reference: VACE paper (arXiv:2503.07598), Figure 2
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# LTX-2 Video VAE spatial compression ratio (pixel -> latent)
LTX2_VAE_SPATIAL_COMPRESSION = 32


class VaceTaskType(Enum):
    """VACE task types corresponding to the paper's taxonomy."""
    # T2V: Text-to-Video (no control, generate freely)
    T2V = "t2v"
    # R2V: Reference-to-Video (subject face/object, or frame reference)
    REFERENCE = "reference"
    # V2V Control: structured control signals (depth, pose, scribble, flow, layout, gray)
    CONTROL = "control"
    # V2V General: general video-to-video with source video as control
    GENERAL_V2V = "general_v2v"
    # MV2V Inpaint: inpaint a masked region of a video
    INPAINT = "inpaint"
    # MV2V Outpaint: extend video spatially
    OUTPAINT = "outpaint"
    # MV2V Extension: extend video temporally (first/last clip)
    EXTENSION = "extension"


def prepare_vace_context(
    control_video: torch.Tensor,
    mask: torch.Tensor,
    vae_encode_fn,
    reference_images: Optional[torch.Tensor] = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Prepare VACE context tensor from control video, mask, and optional references.

    This is the unified entry point for all VACE task types. The task type
    is implicit in how control_video and mask are prepared by the caller.

    Args:
        control_video: Control video tensor (B, C, F, H, W) in pixel space [-1, 1].
                       For inpaint: original with masked regions zeroed.
                       For control: depth/pose/etc. map.
                       For reference: can be zeros (references go via reference_images).
        mask: Binary mask (B, 1, F, H, W).
              1 = reactive (generate/transform this region).
              0 = inactive (keep as-is / reference).
        vae_encode_fn: Callable: (tensor) -> latent_tensor
        reference_images: Optional reference images (B, C, N_ref, H, W) to prepend
                          temporally. Used for R2V tasks (face/object/frame reference).
        device: Target device.
        dtype: Target dtype.

    Returns:
        VACE context tensor (B, vace_in_dim, F', H', W').
    """
    if device is None:
        device = control_video.device
    if dtype is None:
        dtype = control_video.dtype

    # Split into inactive and reactive regions
    inactive = control_video * (1 - mask)
    reactive = control_video * mask

    # VAE encode both
    inactive_latents = vae_encode_fn(inactive)  # (B, C_latent, F', H', W')
    reactive_latents = vae_encode_fn(reactive)

    # Concatenate latents along channel dimension
    vace_video_latents = torch.cat([inactive_latents, reactive_latents], dim=1)

    # Handle reference images: prepend to temporal dimension
    if reference_images is not None:
        ref_inactive = reference_images  # reference is fully "inactive" (keep as-is)
        ref_reactive = torch.zeros_like(reference_images)  # nothing reactive in reference

        ref_inactive_latents = vae_encode_fn(ref_inactive)
        ref_reactive_latents = vae_encode_fn(ref_reactive)
        ref_video_latents = torch.cat([ref_inactive_latents, ref_reactive_latents], dim=1)

        # Prepend reference latents temporally
        vace_video_latents = torch.cat([ref_video_latents, vace_video_latents], dim=2)

    # Rearrange mask to match latent spatial resolution
    P = LTX2_VAE_SPATIAL_COMPRESSION
    vace_mask_latents = rearrange(
        mask[:, 0],  # (B, F, H, W)
        "b f (h p1) (w p2) -> b (p1 p2) f h w",
        p1=P,
        p2=P,
    )  # (B, P*P, F, H', W')

    # Handle reference frames in mask: prepend zeros (reference = inactive)
    if reference_images is not None:
        ref_f = ref_inactive_latents.shape[2]  # temporal dim of reference latents
        ref_mask = torch.zeros(
            vace_mask_latents.shape[0], vace_mask_latents.shape[1],
            ref_f, vace_mask_latents.shape[3], vace_mask_latents.shape[4],
            device=vace_mask_latents.device, dtype=vace_mask_latents.dtype,
        )
        vace_mask_latents = torch.cat([ref_mask, vace_mask_latents], dim=2)

    # Temporal dimension: match latent temporal dimension
    latent_f = vace_video_latents.shape[2]
    if vace_mask_latents.shape[2] != latent_f:
        # Interpolate mask temporal dimension to match video latents
        B_m, C_m, F_m, H_m, W_m = vace_mask_latents.shape
        vace_mask_latents = F.interpolate(
            vace_mask_latents.reshape(B_m * C_m, 1, F_m, H_m, W_m),
            size=(latent_f, H_m, W_m),
            mode="nearest-exact",
        ).reshape(B_m, C_m, latent_f, H_m, W_m)

    # Concatenate: video latents + mask -> final VACE context
    vace_context = torch.cat([vace_video_latents, vace_mask_latents], dim=1)

    return vace_context.to(device=device, dtype=dtype)


# =====================================================================
# Task-specific helpers for preparing control_video and mask
# =====================================================================

def prepare_inpaint_inputs(
    source_video: torch.Tensor,
    inpaint_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare control inputs for inpainting.

    Args:
        source_video: Original video (B, C, F, H, W) in [-1, 1].
        inpaint_mask: Binary mask (B, 1, F, H, W). 1 = region to inpaint.

    Returns:
        (control_video, mask) tuple ready for prepare_vace_context().
    """
    # Control = original video with inpaint region zeroed out
    control_video = source_video * (1 - inpaint_mask)
    # Mask = inpaint region (1 = generate here)
    return control_video, inpaint_mask


def prepare_outpaint_inputs(
    source_video: torch.Tensor,
    target_h: int,
    target_w: int,
    pad_mode: str = "center",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare control inputs for outpainting (spatial extension).

    Args:
        source_video: Original video (B, C, F, H, W) in [-1, 1].
        target_h: Target height (must be >= source height).
        target_w: Target width (must be >= source width).
        pad_mode: Where to place the source ("center", "top_left", "custom").

    Returns:
        (control_video, mask) tuple ready for prepare_vace_context().
    """
    B, C, F, H, W = source_video.shape
    pad_h = target_h - H
    pad_w = target_w - W

    if pad_mode == "center":
        top = pad_h // 2
        left = pad_w // 2
    else:
        top = 0
        left = 0

    bottom = pad_h - top
    right = pad_w - left

    # Pad source video with zeros
    control_video = F.pad(source_video, (left, right, top, bottom), value=0)

    # Mask: 0 for original region, 1 for padded region
    mask = torch.ones(B, 1, F, target_h, target_w, device=source_video.device, dtype=source_video.dtype)
    mask[:, :, :, top:top + H, left:left + W] = 0

    return control_video, mask


def prepare_extension_inputs(
    source_video: torch.Tensor,
    target_frames: int,
    extend_mode: str = "last",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare control inputs for temporal extension.

    Args:
        source_video: Known video (B, C, F, H, W) in [-1, 1].
        target_frames: Total number of target frames (must be > F).
        extend_mode: "first" (extend before), "last" (extend after), "both".

    Returns:
        (control_video, mask) tuple ready for prepare_vace_context().
    """
    B, C, F_src, H, W = source_video.shape
    F_pad = target_frames - F_src

    if extend_mode == "last":
        # Source at beginning, zeros at end
        padding = torch.zeros(B, C, F_pad, H, W, device=source_video.device, dtype=source_video.dtype)
        control_video = torch.cat([source_video, padding], dim=2)
        mask = torch.ones(B, 1, target_frames, H, W, device=source_video.device, dtype=source_video.dtype)
        mask[:, :, :F_src, :, :] = 0  # Keep known frames
    elif extend_mode == "first":
        # Zeros at beginning, source at end
        padding = torch.zeros(B, C, F_pad, H, W, device=source_video.device, dtype=source_video.dtype)
        control_video = torch.cat([padding, source_video], dim=2)
        mask = torch.ones(B, 1, target_frames, H, W, device=source_video.device, dtype=source_video.dtype)
        mask[:, :, F_pad:, :, :] = 0  # Keep known frames
    else:  # "both"
        front_pad = F_pad // 2
        back_pad = F_pad - front_pad
        front = torch.zeros(B, C, front_pad, H, W, device=source_video.device, dtype=source_video.dtype)
        back = torch.zeros(B, C, back_pad, H, W, device=source_video.device, dtype=source_video.dtype)
        control_video = torch.cat([front, source_video, back], dim=2)
        mask = torch.ones(B, 1, target_frames, H, W, device=source_video.device, dtype=source_video.dtype)
        mask[:, :, front_pad:front_pad + F_src, :, :] = 0

    return control_video, mask


def prepare_control_inputs(
    control_signal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare control inputs for structured control (depth, pose, scribble, etc.).

    Args:
        control_signal: Control signal video (B, C, F, H, W) in [-1, 1].
                        This is the depth map, pose skeleton, scribble, etc.

    Returns:
        (control_video, mask) tuple ready for prepare_vace_context().
    """
    # All reactive: the entire control signal is "active"
    mask = torch.ones(
        control_signal.shape[0], 1, *control_signal.shape[2:],
        device=control_signal.device, dtype=control_signal.dtype,
    )
    return control_signal, mask


def prepare_reference_inputs(
    target_frames: int,
    target_h: int,
    target_w: int,
    reference_images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare control inputs for reference-based generation (face, object, frame).

    Args:
        target_frames: Number of target video frames to generate.
        target_h: Target height.
        target_w: Target width.
        reference_images: Reference images (B, C, N_ref, H, W) in [-1, 1].
        device: Target device.
        dtype: Target dtype.

    Returns:
        (control_video, mask, reference_images) tuple.
        control_video is zeros, mask is all-ones (everything to generate),
        reference_images are passed separately to prepare_vace_context().
    """
    B = reference_images.shape[0]
    C = reference_images.shape[1]

    # Control video is zeros (no explicit control, just reference)
    control_video = torch.zeros(B, C, target_frames, target_h, target_w, device=device, dtype=dtype)

    # Mask is all-ones (generate everything)
    mask = torch.ones(B, 1, target_frames, target_h, target_w, device=device, dtype=dtype)

    return control_video, mask, reference_images


def prepare_t2v_inputs(
    batch_size: int,
    channels: int,
    target_frames: int,
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare control inputs for text-to-video (no explicit control).

    Returns:
        (control_video, mask) tuple. Both are zeros/ones respectively,
        meaning "generate everything from text only."
    """
    control_video = torch.zeros(batch_size, channels, target_frames, target_h, target_w, device=device, dtype=dtype)
    mask = torch.ones(batch_size, 1, target_frames, target_h, target_w, device=device, dtype=dtype)
    return control_video, mask


# =====================================================================
# Latent-space utilities (for cached data during training)
# =====================================================================

# =====================================================================
# Audio VACE control signal preparation
# =====================================================================

def prepare_audio_vace_context(
    control_audio_tokens: torch.Tensor,
    temporal_mask: torch.Tensor,
) -> torch.Tensor:
    """Prepare audio VACE context in token space (post-patchification).

    Unlike video VACE which works in 4D latent space, audio VACE operates on
    already-patchified tokens. This avoids dependence on the raw C/F split of
    audio latents and keeps ``vace_in_dim = 2 * audio_in_channels + 1`` constant.

    Args:
        control_audio_tokens: Patchified audio latent tokens (B, T, audio_in_channels).
            Should be produced by AudioPatchifier: ``rearrange("b c t f -> b t (c f)")``.
        temporal_mask: Temporal binary mask (B, T, 1). 1 = reactive (generate),
            0 = inactive (keep). Broadcast across all channel dims.

    Returns:
        Audio VACE context tokens (B, T, 2 * audio_in_channels + 1).
    """
    inactive = control_audio_tokens * (1 - temporal_mask)
    reactive = control_audio_tokens * temporal_mask

    # Concatenate: inactive_tokens | reactive_tokens | mask_scalar
    return torch.cat([inactive, reactive, temporal_mask], dim=-1)


def patchify_audio_latents_for_vace(
    audio_latents: torch.Tensor,
) -> torch.Tensor:
    """Patchify raw 4D audio latents matching AudioPatchifier convention.

    Args:
        audio_latents: Raw audio latents (B, C, T, F) from audio VAE cache.

    Returns:
        (B, T, C*F) token sequence matching AudioPatchifier output.
    """
    B, C, T, F_dim = audio_latents.shape
    # Match AudioPatchifier: rearrange "b c t f -> b t (c f)"
    return audio_latents.permute(0, 2, 1, 3).reshape(B, T, C * F_dim)


def prepare_audio_inpaint_mask(
    num_frames: int,
    inpaint_start: int,
    inpaint_end: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create a temporal inpainting mask for audio VACE.

    Args:
        num_frames: Total number of temporal steps.
        inpaint_start: First frame to inpaint (inclusive).
        inpaint_end: Last frame to inpaint (exclusive).

    Returns:
        Temporal mask (B, T, 1). 1 = generate, 0 = keep.
    """
    mask = torch.zeros(batch_size, num_frames, 1, device=device, dtype=dtype)
    mask[:, inpaint_start:inpaint_end, :] = 1
    return mask


def prepare_audio_extension_mask(
    source_frames: int,
    target_frames: int,
    extend_mode: str = "last",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create a temporal extension mask for audio VACE.

    Args:
        source_frames: Number of known frames.
        target_frames: Total target frames (must be > source_frames).
        extend_mode: "first" (extend before), "last" (extend after), "both".

    Returns:
        Temporal mask (B, T, 1). 1 = generate, 0 = keep.
    """
    mask = torch.ones(batch_size, target_frames, 1, device=device, dtype=dtype)
    pad = target_frames - source_frames
    if extend_mode == "last":
        mask[:, :source_frames, :] = 0
    elif extend_mode == "first":
        mask[:, pad:, :] = 0
    else:  # "both"
        front = pad // 2
        mask[:, front:front + source_frames, :] = 0
    return mask


def prepare_audio_reference_context(
    reference_audio_tokens: torch.Tensor,
    target_frames: int,
) -> torch.Tensor:
    """Prepare audio VACE context with reference prefix and reactive target region.

    The reference tokens are placed as inactive context (mask=0); the target
    region is marked reactive (mask=1). The model sees the reference during
    denoising, but whether it uses it for any particular purpose depends on
    training.

    To use as a prefix-conditioning setup, set target_frames to the desired
    output length. The reference tokens are prepended as inactive context;
    the text prompt conditions the reactive region.

    Args:
        reference_audio_tokens: Patchified reference audio tokens (B, T_ref, audio_in_channels).
        target_frames: Number of target frames to generate after the reference.

    Returns:
        Audio VACE context tokens (B, T_ref + target_frames, 2 * audio_in_channels + 1).
    """
    B, T_ref, C = reference_audio_tokens.shape
    device = reference_audio_tokens.device
    dtype = reference_audio_tokens.dtype

    # Reference region: inactive (keep), mask=0
    ref_mask = torch.zeros(B, T_ref, 1, device=device, dtype=dtype)

    # Target region: reactive (generate), mask=1
    target_tokens = torch.zeros(B, target_frames, C, device=device, dtype=dtype)
    target_mask = torch.ones(B, target_frames, 1, device=device, dtype=dtype)

    # Concatenate temporally: [reference | target]
    all_tokens = torch.cat([reference_audio_tokens, target_tokens], dim=1)
    all_mask = torch.cat([ref_mask, target_mask], dim=1)

    return prepare_audio_vace_context(all_tokens, all_mask)


def prepare_audio_style_transfer_context(
    reference_audio_tokens: torch.Tensor,
    content_audio_tokens: torch.Tensor,
) -> torch.Tensor:
    """Prepare audio VACE context with reference prefix and reactive content region.

    The reference audio is prepended as inactive context (mask=0), and the
    content audio region is marked fully reactive (mask=1). Analogous to
    video VACE R2V mode, where a reference is provided alongside the
    generation target.

    Args:
        reference_audio_tokens: Patchified reference audio tokens (B, T_ref, audio_in_channels).
            Placed in the inactive prefix region.
        content_audio_tokens: Patchified content audio tokens (B, T_content, audio_in_channels).
            Placed in the reactive region after the reference prefix.

    Returns:
        Audio VACE context tokens (B, T_ref + T_content, 2 * audio_in_channels + 1).
    """
    B = reference_audio_tokens.shape[0]
    T_ref = reference_audio_tokens.shape[1]
    T_content = content_audio_tokens.shape[1]
    device = reference_audio_tokens.device
    dtype = reference_audio_tokens.dtype

    # Reference: inactive (prefix context), mask=0
    ref_mask = torch.zeros(B, T_ref, 1, device=device, dtype=dtype)

    # Content: reactive (target for generation), mask=1
    content_mask = torch.ones(B, T_content, 1, device=device, dtype=dtype)

    # Concatenate temporally: [reference | content]
    all_tokens = torch.cat([reference_audio_tokens, content_audio_tokens], dim=1)
    all_mask = torch.cat([ref_mask, content_mask], dim=1)

    return prepare_audio_vace_context(all_tokens, all_mask)


# =====================================================================
# Joint audio-video VACE mask utilities
# =====================================================================

def create_joint_av_inpaint_masks(
    video_latent_frames: int,
    audio_latent_steps: int,
    inpaint_start_frame: int,
    inpaint_end_frame: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create temporally aligned video and audio inpainting masks.

    Maps video frame indices to audio timestep indices proportionally,
    so both modalities have matching reactive/inactive regions.

    Args:
        video_latent_frames: Number of video latent frames (F').
        audio_latent_steps: Number of audio latent timesteps (T).
        inpaint_start_frame: First video frame to inpaint (inclusive).
        inpaint_end_frame: Last video frame to inpaint (exclusive).
        device: Target device.
        dtype: Target dtype.
        batch_size: Batch size.

    Returns:
        (video_mask, audio_mask):
            video_mask: (B, 1, F', 1, 1) — 1=generate, 0=keep. Broadcasts spatially.
            audio_mask: (B, T, 1) — 1=generate, 0=keep. For token-space audio VACE.
    """
    # Video mask: frame-level, broadcast to spatial dims
    video_mask = torch.zeros(batch_size, 1, video_latent_frames, 1, 1, device=device, dtype=dtype)
    video_mask[:, :, inpaint_start_frame:inpaint_end_frame, :, :] = 1

    # Audio mask: proportional mapping from video frames to audio timesteps
    if video_latent_frames > 0:
        audio_start = int(inpaint_start_frame * audio_latent_steps / video_latent_frames)
        audio_end = int(inpaint_end_frame * audio_latent_steps / video_latent_frames)
    else:
        audio_start, audio_end = 0, audio_latent_steps
    audio_start = max(0, min(audio_start, audio_latent_steps))
    audio_end = max(audio_start, min(audio_end, audio_latent_steps))

    audio_mask = torch.zeros(batch_size, audio_latent_steps, 1, device=device, dtype=dtype)
    audio_mask[:, audio_start:audio_end, :] = 1

    return video_mask, audio_mask


def create_joint_av_extension_masks(
    video_latent_frames: int,
    audio_latent_steps: int,
    source_video_frames: int,
    extend_mode: str = "last",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create temporally aligned video and audio extension masks.

    Args:
        video_latent_frames: Total target video latent frames.
        audio_latent_steps: Total target audio latent timesteps.
        source_video_frames: Number of known source video frames.
        extend_mode: "last" (extend after), "first" (extend before), "both".

    Returns:
        (video_mask, audio_mask) — both with 1=generate, 0=keep.
    """
    video_mask = torch.ones(batch_size, 1, video_latent_frames, 1, 1, device=device, dtype=dtype)
    pad = video_latent_frames - source_video_frames

    if extend_mode == "last":
        video_mask[:, :, :source_video_frames, :, :] = 0
    elif extend_mode == "first":
        video_mask[:, :, pad:, :, :] = 0
    else:  # "both"
        front = pad // 2
        video_mask[:, :, front:front + source_video_frames, :, :] = 0

    # Map to audio: proportional
    source_audio = int(source_video_frames * audio_latent_steps / video_latent_frames) if video_latent_frames > 0 else 0
    audio_mask = torch.ones(batch_size, audio_latent_steps, 1, device=device, dtype=dtype)
    audio_pad = audio_latent_steps - source_audio

    if extend_mode == "last":
        audio_mask[:, :source_audio, :] = 0
    elif extend_mode == "first":
        audio_mask[:, audio_pad:, :] = 0
    else:
        front = audio_pad // 2
        audio_mask[:, front:front + source_audio, :] = 0

    return video_mask, audio_mask


# =====================================================================
# Video VACE latent-space utilities (for cached data during training)
# =====================================================================

def prepare_vace_context_from_latents(
    control_latents: torch.Tensor,
    mask_latents: torch.Tensor,
) -> torch.Tensor:
    """Prepare VACE context from pre-cached latents and mask.

    Used during training when latents are pre-cached.

    Args:
        control_latents: Pre-encoded control video latents (B, 2*C_latent, F', H', W').
                         Already concatenation of inactive + reactive latents.
        mask_latents: Mask in latent resolution (B, P*P, F', H', W').

    Returns:
        VACE context tensor (B, 2*C_latent + P*P, F', H', W').
    """
    return torch.cat([control_latents, mask_latents], dim=1)


def patchify_vace_context(
    vace_context: torch.Tensor,
    patchifier=None,
) -> torch.Tensor:
    """Convert VACE context from spatial to sequence format.

    Args:
        vace_context: (B, C, F, H, W) VACE context tensor.
        patchifier: Optional VideoLatentPatchifier. If None, uses simple flatten.

    Returns:
        (B, seq_len, C) token sequence.
    """
    if patchifier is not None:
        return patchifier.patchify(vace_context)

    # Simple flatten: (B, C, F, H, W) -> (B, F*H*W, C)
    B, C, Fr, H, W = vace_context.shape
    return vace_context.reshape(B, C, Fr * H * W).permute(0, 2, 1)
