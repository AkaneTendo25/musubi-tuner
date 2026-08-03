from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
import torch

from musubi_tuner.minimax_h3.architecture import temporal_shape
from musubi_tuner.minimax_h3.audio import (
    audio_valid_mask_to_latent_mask,
    load_audio_asset,
    target_audio_processing_spec,
)
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.component_loader import load_audio_vae_encoder, load_video_vae_encoder
from musubi_tuner.minimax_h3.media import MediaModality
from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag
from musubi_tuner.minimax_h3.packing import (
    build_row_timesteps,
    build_t2va_packed_sequence,
    pack_audio_latents,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from musubi_tuner.minimax_h3.request import H3GenerationRequest
from musubi_tuner.minimax_h3.training import H3ModelPrediction, H3TrainingMode
from musubi_tuner.utils.model_utils import dtype_to_str, str_to_dtype


def _raise_unavailable(component: str) -> NoReturn:
    from musubi_tuner.minimax_h3.backend import H3BackendUnavailableError

    raise H3BackendUnavailableError(f"MiniMax H3 {component} is unsupported")


def create_latent_encoder(
    *,
    video_vae: Path,
    audio_vae: Path,
    device: str | None,
    dtype: str,
):
    """Load the released video and audio VAEs and adapt their encoders to Musubi."""
    target_device = torch.device(device or "cpu")
    output_dtype = str_to_dtype(dtype)
    video_encoder = load_video_vae_encoder(video_vae, target_device)
    audio_encoder = load_audio_vae_encoder(audio_vae, target_device)
    return _NativeLatentEncoder(video_encoder, audio_encoder, output_dtype)


def create_conditioning_encoder(
    *,
    text_encoder: Path,
    tokenizer: Path,
    task: Literal["t2va", "fl2va"],
    device: str | None,
    dtype: str,
    quantization: Literal["none", "int8", "nf4"] = "none",
):
    """Load the released understanding encoder and adapt its hidden-state output to Musubi."""
    from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder, load_text_conditioner

    output_dtype = str_to_dtype(dtype)
    processor, model = load_text_conditioner(
        text_encoder,
        tokenizer,
        device=device or "cpu",
        dtype=output_dtype,
        quantization=quantization,
    )
    return MiniMaxH3ConditioningEncoder(processor, model, output_dtype, task)


def create_generator(*, model: Path, device: str | None, dtype: str, request: H3GenerationRequest):
    """Load the released inference components and adapt generation to Musubi."""
    del model, device, dtype, request
    _raise_unavailable("generation")


def create_training_backend(
    *,
    model: Path,
    device: str | None,
    dtype: str,
    mode: H3TrainingMode,
    attention_mode: str,
    split_attention: bool,
    fp8_scaled: bool = False,
    quantization_device: str | None = None,
):
    """Load the released BF16 transformer and adapt its training forward to Musubi."""
    if mode != "fl2va":
        _raise_unavailable("reference-conditioned training")
    if dtype != "bfloat16":
        raise ValueError("MiniMax H3 full-checkpoint training requires bfloat16 compute")
    if attention_mode != "torch" or split_attention:
        raise ValueError("the native MiniMax H3 backend supports only unsplit PyTorch SDPA")
    from musubi_tuner.minimax_h3.model_loader import load_transformer

    transformer = load_transformer(
        model,
        mode=mode,
        loading_device=device or "cpu",
        fp8_scaled=fp8_scaled,
        quantization_device=quantization_device,
    )
    return _NativeTrainingBackend(transformer)


class _NativeTrainingBackend:
    def __init__(self, transformer: torch.nn.Module):
        self.transformer = transformer

    def get_training_transformer(self) -> torch.nn.Module:
        return self.transformer

    def predict_training(
        self,
        transformer: torch.nn.Module,
        batch: dict,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        *,
        conditioning: Literal["prompt", "empty"] = "prompt",
    ) -> H3ModelPrediction:
        if video_hidden_states.shape[0] != 1 or audio_hidden_states.shape[0] != 1:
            raise ValueError("MiniMax H3 training requires batch size 1")
        config = getattr(transformer, "config", getattr(self.transformer, "config", None))
        if config is None:
            raise TypeError("MiniMax H3 transformer must expose its released config")
        if video_hidden_states.ndim != 5 or video_hidden_states.shape[1] != config.in_channels:
            raise ValueError(
                f"H3 video input must have shape [1, {config.in_channels}, T, H, W], got {tuple(video_hidden_states.shape)}"
            )
        if audio_hidden_states.ndim != 4 or audio_hidden_states.shape[1:3] != (2, config.audio_in_channels):
            raise ValueError(
                f"H3 audio input must have shape [1, 2, {config.audio_in_channels}, T], got {tuple(audio_hidden_states.shape)}"
            )

        hidden_key, tags_key = (
            (H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY)
            if conditioning == "prompt"
            else (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY)
        )
        text_hidden = self._one_conditioning_item(batch, hidden_key, expected_ndim=2)
        text_tags = self._one_conditioning_item(batch, tags_key, expected_ndim=1)
        if text_hidden.ndim != 2 or text_hidden.shape[-1] != config.text_dim:
            raise ValueError(f"H3 {hidden_key} must have shape [tokens, {config.text_dim}]")
        if text_tags.dtype != torch.long or text_tags.shape != (text_hidden.shape[0],):
            raise ValueError(f"H3 {tags_key} must be int64 with one tag per text token")
        if not bool((text_tags == int(MiniMaxH3TokenTag.TEXT)).all()):
            raise ValueError("MiniMax H3 training accepts only text-only T2VA conditioning; re-cache with --task t2va")

        patch_size = tuple(config.patch_size)
        video_rows = patchify_video_latents(video_hidden_states, patch_size)
        audio_rows = pack_audio_latents(audio_hidden_states)
        _, _, latent_frames, latent_height, latent_width = video_hidden_states.shape
        num_audio_latents = audio_hidden_states.shape[-1]
        layout = build_t2va_packed_sequence(
            text_tags,
            num_latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=patch_size,
        )
        model_device = video_hidden_states.device
        timestep, timestep_indices = build_row_timesteps(layout, video_timestep, audio_timestep)
        output = transformer(
            video_hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_hidden[None].to(model_device),
            timestep=timestep.to(model_device),
            timestep_indices=timestep_indices.to(model_device),
            token_tags=layout.token_tags.to(model_device),
            position_ids=layout.position_ids.to(model_device),
            video_indices=layout.video_indices.to(model_device),
            audio_indices=layout.audio_indices.to(model_device),
            text_indices=layout.text_indices.to(model_device),
        )
        video = unpatchify_video_tokens(
            output.video,
            latent_shape=(config.in_channels, latent_frames, latent_height, latent_width),
            patch_size=patch_size,
        )
        audio = unpack_audio_tokens(output.audio, num_audio_latents=num_audio_latents)
        return H3ModelPrediction(video=video, audio=audio)

    @staticmethod
    def _one_conditioning_item(batch: dict, key: str, *, expected_ndim: int) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"MiniMax H3 training cache is missing {key}")
        value = batch[key]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"H3 {key} must contain exactly one batch item")
            value = value[0]
        elif isinstance(value, torch.Tensor) and value.ndim == expected_ndim + 1 and value.shape[0] == 1:
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"H3 {key} must be a tensor or one-element tensor sequence")
        if value.ndim != expected_ndim:
            raise ValueError(f"H3 {key} must have {expected_ndim} dimensions after selecting one batch item")
        return value


class _NativeLatentEncoder:
    def __init__(
        self,
        video_encoder: torch.nn.Module,
        audio_encoder: torch.nn.Module,
        output_dtype: torch.dtype,
    ) -> None:
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.output_dtype = output_dtype

    @staticmethod
    def _target_asset(item: Any):
        assets = getattr(item, "h3_media_assets", ())
        targets = [asset for asset in assets if asset.role == "target" and asset.modality is MediaModality.VIDEO]
        if len(targets) != 1:
            raise ValueError(f"H3 item {item.item_key!r} must have one attached target video")
        return targets[0]

    def _encode_video(self, content: np.ndarray) -> torch.Tensor:
        if not isinstance(content, np.ndarray):
            raise TypeError("MiniMax H3 latent caching requires a numpy video array")
        if content.ndim == 3:
            content = content[None]
        if content.ndim != 4 or content.shape[-1] != 3:
            raise ValueError(f"H3 video content must have shape [F, H, W, 3], got {content.shape}")
        pixels = torch.from_numpy(np.ascontiguousarray(content)).permute(3, 0, 1, 2).unsqueeze(0)
        device = next(self.video_encoder.parameters()).device
        weight_dtype = next(self.video_encoder.parameters()).dtype
        pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        pixels = ((pixels - pixel_mean) / pixel_std).to(weight_dtype)
        with torch.no_grad():
            return self.video_encoder.encode(pixels)[0].to(self.output_dtype)

    def _encode_audio(self, item: Any) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._target_asset(item)
        clip = load_audio_asset(target, target_audio_processing_spec(target))
        if clip is None:
            raise RuntimeError("H3 target audio policy unexpectedly dropped the target")
        device = next(self.audio_encoder.parameters()).device
        waveform = clip.waveform.to(device=device, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            latents = self.audio_encoder.encode(waveform).to(self.output_dtype)
        mask = audio_valid_mask_to_latent_mask(clip.valid_mask)
        if mask.shape != (latents.shape[-1],):
            raise ValueError(f"H3 audio cache length mismatch: encoder produced {latents.shape[-1]} rows, mask has {mask.shape[0]}")
        return latents, mask

    def encode_latents(self, batch: list[Any]) -> tuple[dict[str, torch.Tensor], ...]:
        dtype_name = dtype_to_str(self.output_dtype)
        results = []
        for item in batch:
            video = self._encode_video(item.content)
            expected = temporal_shape(video_frame_count := int(item.content.shape[0]))
            if video.shape[1] != expected.video_latent_frames:
                raise ValueError(
                    f"H3 video VAE produced {video.shape[1]} latent frames for {video_frame_count} pixels; "
                    f"expected {expected.video_latent_frames}"
                )
            audio, audio_mask = self._encode_audio(item)
            if audio.shape[-1] != expected.audio_latent_frames:
                raise ValueError(f"H3 audio VAE produced {audio.shape[-1]} rows; expected {expected.audio_latent_frames}")
            frame_shape = "x".join(str(value) for value in video.shape[-3:])
            results.append(
                {
                    f"latents_{frame_shape}_{dtype_name}": video,
                    f"{H3_AUDIO_LATENTS_KEY}_2x32x{audio.shape[-1]}_{dtype_name}": audio,
                    H3_AUDIO_LOSS_MASK_KEY: audio_mask,
                }
            )
        return tuple(results)
