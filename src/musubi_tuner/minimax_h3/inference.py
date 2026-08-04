# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from musubi_tuner.minimax_h3.architecture import (
    AUDIO_FLOW_SHIFT,
    AUDIO_SAMPLE_RATE,
    CANVAS_MULTIPLE,
    VIDEO_FLOW_SHIFT,
    VIDEO_FPS,
    VIDEO_SPATIAL_COMPRESSION,
    temporal_shape,
)
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.component_loader import load_video_vae_encoder
from musubi_tuner.minimax_h3.packing import (
    build_row_timesteps,
    build_t2va_packed_sequence,
    pack_audio_latents,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from musubi_tuner.utils.device_utils import clean_memory_on_device

_SHORT_EDGE = 768
_MAX_PIXELS = 768 * 1344


@dataclass(frozen=True)
class H3GeneratedMedia:
    video: torch.Tensor
    audio: torch.Tensor
    fps: int = VIDEO_FPS
    sample_rate: int = AUDIO_SAMPLE_RATE


def resolve_canvas_size(ratio: str) -> tuple[int, int]:
    """Resolve an H3 ratio to its released 768-short-edge canvas."""
    if ratio == "adaptive":
        ratio = "16:9"
    try:
        aspect_width, aspect_height = (float(value) for value in ratio.split(":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid MiniMax H3 aspect ratio: {ratio!r}") from error
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError("MiniMax H3 aspect-ratio values must be positive")
    value = aspect_width / aspect_height
    if not 0.25 <= value <= 4.0:
        raise ValueError("MiniMax H3 supports aspect ratios from 1:4 through 4:1")
    if value >= 1.0:
        width, height = _SHORT_EDGE * value, float(_SHORT_EDGE)
    else:
        width, height = float(_SHORT_EDGE), _SHORT_EDGE / value
    if width * height > _MAX_PIXELS:
        scale = (_MAX_PIXELS / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    return (
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def shifted_flow_schedule(num_inference_steps: int, shift: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return H3's shifted sigmas and model timesteps, including terminal sigma zero."""
    if num_inference_steps < 2:
        raise ValueError("MiniMax H3 inference requires at least two sigma points")
    if shift <= 0:
        raise ValueError("MiniMax H3 flow shift must be positive")
    base = torch.linspace(1.0, 0.0, num_inference_steps, dtype=torch.float32)
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = torch.unique_consecutive(sigmas).to(device)
    return sigmas, 1.0 - sigmas[:-1]


def data_ward_euler_step(
    sample: torch.Tensor,
    prediction: torch.Tensor,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """One deterministic H3 rectified-flow step for a data-ward velocity."""
    sigma_from_timestep = 1.0 - timestep.to(device=sample.device, dtype=sample.dtype)
    denoised = sample + sigma_from_timestep * prediction.to(sample.dtype)
    ratio = sigma_next.to(device=sample.device, dtype=torch.float32) / sigma.to(device=sample.device, dtype=torch.float32)
    result = ratio * sample.float() + (1.0 - ratio) * denoised.float()
    return result.to(sample.dtype)


def _validate_geometry(height: int, width: int, frame_count: int) -> None:
    if height <= 0 or width <= 0 or height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError(f"MiniMax H3 height and width must be positive multiples of {CANVAS_MULTIPLE}")
    temporal_shape(frame_count)


def prepare_keyframe_image(image: Image.Image, height: int, width: int, *, stretch: bool) -> Image.Image:
    """Place an FL2VA keyframe on the target canvas using the released resize policy."""
    image = image.convert("RGB")
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.width, height / image.height)
    resized_width = max(width, round(image.width * scale))
    resized_height = max(height, round(image.height * scale))
    image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _augment_keyframe_rows(rows: torch.Tensor, rows_per_anchor: int, seed: int) -> torch.Tensor:
    """Apply the released fixed-timestep condition noise, restarting the RNG for each anchor."""
    augmented = []
    for anchor_rows in rows.split(rows_per_anchor):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(anchor_rows.shape, generator=generator, dtype=torch.float32)
        augmented.append(0.999 * anchor_rows.float() + 0.001 * noise.to(anchor_rows.device))
    return torch.cat(augmented)


@torch.no_grad()
def encode_keyframe_images(
    video_vae: Path,
    images: list[Image.Image],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Sequentially load the video encoder and return patchified FL2VA rows for each image."""
    if not images:
        return ()
    video_encoder = load_video_vae_encoder(video_vae, device)
    rows = []
    for image in images:
        content = np.array(image, dtype=np.uint8, copy=True, order="C")
        pixels = torch.from_numpy(content).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        weight_dtype = next(video_encoder.parameters()).dtype
        pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        latent = video_encoder.encode_reference(((pixels - mean) / std).to(weight_dtype), image=True)[0]
        rows.append(patchify_video_latents(latent[None], (1, 2, 2))[0].cpu())
    del video_encoder
    clean_memory_on_device(device)
    return tuple(rows)


@torch.no_grad()
def denoise_fl2va(
    transformer: torch.nn.Module,
    conditioning: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    frame_count: int,
    num_inference_steps: int,
    generator: torch.Generator,
    device: torch.device,
    keyframe_rows: torch.Tensor | None = None,
    keyframe_anchors: tuple[str, ...] = (),
    condition_seed: int = 0,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate FL2VA joint latents with optional first/last keyframe conditioning."""
    _validate_geometry(height, width, frame_count)
    shape = temporal_shape(frame_count)
    config = transformer.config
    latent_height = height // VIDEO_SPATIAL_COMPRESSION
    latent_width = width // VIDEO_SPATIAL_COMPRESSION
    video = torch.randn(
        (1, config.in_channels, shape.video_latent_frames, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    audio = torch.randn(
        (1, 2, config.audio_in_channels, shape.audio_latent_frames),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    text_hidden = conditioning[H3_TEXT_HIDDEN_KEY]
    text_tags = conditioning[H3_TEXT_TOKEN_TAGS_KEY]
    if text_hidden.ndim != 2 or text_hidden.shape[-1] != config.text_dim:
        raise ValueError(f"MiniMax H3 conditioning must have shape [tokens, {config.text_dim}]")
    if text_tags.shape != (text_hidden.shape[0],) or text_tags.dtype != torch.long:
        raise ValueError("MiniMax H3 conditioning tags must be int64 with one entry per text row")
    rows_per_anchor = (latent_height // config.patch_size[1]) * (latent_width // config.patch_size[2])
    expected_keyframe_rows = len(keyframe_anchors) * rows_per_anchor
    keyframe_width = config.in_channels * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    if expected_keyframe_rows:
        if keyframe_rows is None or keyframe_rows.shape != (expected_keyframe_rows, keyframe_width):
            actual = None if keyframe_rows is None else tuple(keyframe_rows.shape)
            raise ValueError(f"MiniMax H3 keyframe rows have shape {actual}, expected {(expected_keyframe_rows, keyframe_width)}")
        keyframe_rows = _augment_keyframe_rows(keyframe_rows, rows_per_anchor, condition_seed).to(device)
    elif keyframe_rows is not None:
        raise ValueError("MiniMax H3 keyframe rows require first/last anchors")
    layout = build_t2va_packed_sequence(
        text_tags,
        num_latent_frames=shape.video_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=shape.audio_latent_frames,
        patch_size=tuple(config.patch_size),
        keyframe_anchors=keyframe_anchors,
    )
    text_hidden = text_hidden[None].to(device)
    token_tags = layout.token_tags.to(device)
    position_ids = layout.position_ids.to(device)
    video_indices = layout.video_indices.to(device)
    audio_indices = layout.audio_indices.to(device)
    text_indices = layout.text_indices.to(device)
    video_sigmas, video_timesteps = shifted_flow_schedule(num_inference_steps, VIDEO_FLOW_SHIFT, device)
    audio_sigmas, audio_timesteps = shifted_flow_schedule(num_inference_steps, AUDIO_FLOW_SHIFT, device)
    if video_timesteps.shape != audio_timesteps.shape:
        raise RuntimeError("MiniMax H3 video and audio schedules produced different step counts")

    iterator = zip(video_timesteps, audio_timesteps)
    if show_progress:
        iterator = tqdm(iterator, total=video_timesteps.numel(), desc="MiniMax H3 denoising")
    autocast_enabled = device.type == "cuda"
    for index, (video_timestep, audio_timestep) in enumerate(iterator):
        video_rows = patchify_video_latents(video, tuple(config.patch_size))
        if keyframe_rows is not None:
            video_rows = torch.cat((keyframe_rows[None], video_rows), dim=1)
        audio_rows = pack_audio_latents(audio)
        condition_timestep = (
            torch.maximum(video_timestep.reshape(1), torch.tensor([0.999], device=device)) if keyframe_rows is not None else None
        )
        timestep, timestep_indices = build_row_timesteps(
            layout,
            video_timestep,
            audio_timestep,
            condition_timestep,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = transformer(
                video_hidden_states=video_rows,
                audio_hidden_states=audio_rows,
                encoder_hidden_states=text_hidden,
                timestep=timestep.to(device),
                timestep_indices=timestep_indices.to(device),
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
            )
        video_prediction = unpatchify_video_tokens(
            output.video[:, layout.num_condition_video_rows :],
            latent_shape=(config.in_channels, shape.video_latent_frames, latent_height, latent_width),
            patch_size=tuple(config.patch_size),
        )
        audio_prediction = unpack_audio_tokens(output.audio, num_audio_latents=shape.audio_latent_frames)
        video = data_ward_euler_step(
            video,
            video_prediction,
            video_timestep,
            video_sigmas[index],
            video_sigmas[index + 1],
        )
        audio = data_ward_euler_step(
            audio,
            audio_prediction,
            audio_timestep,
            audio_sigmas[index],
            audio_sigmas[index + 1],
        )
    return video, audio


@torch.no_grad()
def decode_latents_sequentially(
    video_decoder: torch.nn.Module,
    audio_decoder: torch.nn.Module,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    device: torch.device,
) -> H3GeneratedMedia:
    """Decode video and audio one module at a time to minimize peak residency."""
    video_decoder.to(device).eval()
    video = video_decoder.decode(video_latents.to(device)).cpu()
    video_decoder.to("cpu")
    clean_memory_on_device(device)
    audio_decoder.to(device).eval()
    audio = audio_decoder.decode(audio_latents.to(device)).cpu()
    audio_decoder.to("cpu")
    clean_memory_on_device(device)
    planned_audio_samples = round(video.shape[2] * AUDIO_SAMPLE_RATE / VIDEO_FPS)
    common_audio_samples = min(audio.shape[-1], planned_audio_samples)
    common_video_frames = min(video.shape[2], max(1, round(common_audio_samples * VIDEO_FPS / AUDIO_SAMPLE_RATE)))
    video = video[:, :, :common_video_frames]
    audio = audio[:, :, :common_audio_samples]
    return H3GeneratedMedia(video=video, audio=audio)


def save_av_mp4(media: H3GeneratedMedia, output: Path, metadata: dict | None = None) -> None:
    """Mux an RGB video tensor and synchronized stereo waveform into an MP4 file."""
    if media.video.ndim != 5 or media.video.shape[:2] != (1, 3):
        raise ValueError(f"video must have shape [1, 3, F, H, W], got {tuple(media.video.shape)}")
    if media.audio.ndim != 3 or media.audio.shape[:2] != (1, 2):
        raise ValueError(f"audio must have shape [1, 2, samples], got {tuple(media.audio.shape)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = (media.video[0].permute(1, 2, 3, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    audio = media.audio[0].clamp(-1, 1).numpy().astype(np.float32, copy=False)
    with av.open(str(output), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=media.fps)
        video_stream.width = frames.shape[2]
        video_stream.height = frames.shape[1]
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=media.sample_rate)
        audio_stream.layout = "stereo"
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        samples_per_frame = 1024
        pts = 0
        for start in range(0, audio.shape[1], samples_per_frame):
            chunk = np.ascontiguousarray(audio[:, start : start + samples_per_frame])
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
            frame.sample_rate = media.sample_rate
            frame.time_base = Fraction(1, media.sample_rate)
            frame.pts = pts
            pts += chunk.shape[1]
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
    if metadata is not None:
        output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
