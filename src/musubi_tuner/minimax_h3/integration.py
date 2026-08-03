from __future__ import annotations

import gc
import logging
import time
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
from musubi_tuner.minimax_h3.component_loader import (
    load_audio_vae_decoder,
    load_audio_vae_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
)
from musubi_tuner.minimax_h3.inference import decode_latents_sequentially, denoise_t2va, resolve_canvas_size, save_av_mp4
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
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.model_utils import dtype_to_str, str_to_dtype

logger = logging.getLogger(__name__)


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


def create_generator(
    *,
    model: Path,
    text_encoder: Path,
    tokenizer: Path,
    video_vae: Path,
    audio_vae: Path,
    device: str | None,
    dtype: str,
    request: H3GenerationRequest,
    num_inference_steps: int = 20,
    height: int | None = None,
    width: int | None = None,
    fp8_scaled: bool = False,
    text_encoder_quantization: Literal["none", "int8", "nf4"] = "none",
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    block_swap_ring_size: int = 2,
    block_swap_granularity: Literal["block", "layer"] = "block",
    use_pinned_memory_for_block_swap: bool = False,
    lora_weights: tuple[Path, ...] = (),
    lora_multipliers: tuple[float, ...] = (),
):
    """Create a sequentially-loaded native T2VA generator."""
    if request.mode != "text_to_video":
        _raise_unavailable("reference-conditioned generation")
    if dtype != "bfloat16":
        raise ValueError("MiniMax H3 native generation requires bfloat16 transformer compute")
    return _NativeGenerator(
        model=model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        fp8_scaled=fp8_scaled,
        text_encoder_quantization=text_encoder_quantization,
        blocks_to_swap=blocks_to_swap,
        block_swap_h2d_only=block_swap_h2d_only,
        block_swap_ring_size=block_swap_ring_size,
        block_swap_granularity=block_swap_granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
        lora_weights=lora_weights,
        lora_multipliers=lora_multipliers,
    )


class _NativeGenerator:
    def __init__(
        self,
        *,
        model: Path,
        text_encoder: Path,
        tokenizer: Path,
        video_vae: Path,
        audio_vae: Path,
        device: torch.device,
        num_inference_steps: int,
        height: int | None,
        width: int | None,
        fp8_scaled: bool,
        text_encoder_quantization: Literal["none", "int8", "nf4"],
        blocks_to_swap: int,
        block_swap_h2d_only: bool,
        block_swap_ring_size: int,
        block_swap_granularity: Literal["block", "layer"],
        use_pinned_memory_for_block_swap: bool,
        lora_weights: tuple[Path, ...],
        lora_multipliers: tuple[float, ...],
    ) -> None:
        self.model = Path(model)
        self.text_encoder = Path(text_encoder)
        self.tokenizer = Path(tokenizer)
        self.video_vae = Path(video_vae)
        self.audio_vae = Path(audio_vae)
        self.device = device
        self.num_inference_steps = num_inference_steps
        if (height is None) != (width is None):
            raise ValueError("MiniMax H3 height and width must be provided together")
        self.height = height
        self.width = width
        self.fp8_scaled = fp8_scaled
        self.text_encoder_quantization = text_encoder_quantization
        self.blocks_to_swap = blocks_to_swap
        self.block_swap_h2d_only = block_swap_h2d_only
        self.block_swap_ring_size = block_swap_ring_size
        self.block_swap_granularity = block_swap_granularity
        self.use_pinned_memory_for_block_swap = use_pinned_memory_for_block_swap
        self.lora_weights = tuple(Path(path) for path in lora_weights)
        self.lora_multipliers = lora_multipliers

    def _measure(self, name: str, operation, metrics: dict[str, dict]):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = operation()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        values = {"seconds": time.perf_counter() - started}
        if self.device.type == "cuda":
            values.update(
                allocated_gib=torch.cuda.memory_allocated(self.device) / 2**30,
                reserved_gib=torch.cuda.memory_reserved(self.device) / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
            )
        metrics[name] = values
        logger.info("MiniMax H3 inference stage %s: %s", name, values)
        return result

    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        encoder = create_conditioning_encoder(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            task="t2va",
            device=str(self.device),
            dtype="bfloat16",
            quantization=self.text_encoder_quantization,
        )
        conditioning = encoder.encode_prompt(prompt)
        del encoder
        gc.collect()
        clean_memory_on_device(self.device)
        return conditioning

    def _load_transformer(self):
        from safetensors.torch import load_file

        from musubi_tuner.minimax_h3.model_loader import load_transformer
        from musubi_tuner.networks import lora_minimax_h3

        loading_device = torch.device("cpu") if self.blocks_to_swap else self.device
        transformer = load_transformer(
            self.model,
            mode="fl2va",
            loading_device=loading_device,
            fp8_scaled=self.fp8_scaled,
            quantization_device=self.device if self.fp8_scaled else None,
        )
        transformer.requires_grad_(False).eval()
        if self.blocks_to_swap:
            swap_config = BlockSwapConfig(
                device=self.device,
                supports_backward=False,
                use_pinned_memory=self.use_pinned_memory_for_block_swap,
                h2d_only=self.block_swap_h2d_only,
                ring_size=self.block_swap_ring_size,
                granularity=self.block_swap_granularity,
            )
            transformer.enable_block_swap(self.blocks_to_swap, swap_config)

        networks = []
        for index, weights_path in enumerate(self.lora_weights):
            multiplier = self.lora_multipliers[index] if index < len(self.lora_multipliers) else 1.0
            weights = load_file(weights_path)
            network = lora_minimax_h3.create_arch_network_from_weights(
                multiplier,
                weights,
                unet=transformer,
                for_inference=True,
            )
            network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights, strict=True)
            if info.missing_keys or info.unexpected_keys:
                raise RuntimeError(f"strict H3 LoRA load failed: {info}")
            network.to(self.device).eval()
            networks.append(network)
        if self.blocks_to_swap:
            transformer.move_to_device_except_swap_blocks(self.device)
            transformer.switch_block_swap_for_inference()
        else:
            transformer.to(self.device)
        return transformer, networks

    @torch.no_grad()
    def generate(self, request: H3GenerationRequest) -> None:
        if request.mode != "text_to_video":
            _raise_unavailable("reference-conditioned generation")
        metrics: dict[str, dict] = {}
        total_started = time.perf_counter()
        conditioning = self._measure("text_conditioning", lambda: self._encode_prompt(request.prompt), metrics)
        loaded_transformer = self._measure("transformer_load", self._load_transformer, metrics)
        transformer, networks = loaded_transformer
        del loaded_transformer
        height, width = (self.height, self.width) if self.height is not None else resolve_canvas_size(request.ratio)
        shape = request.temporal_shape
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        video_latents, audio_latents = self._measure(
            "joint_denoising",
            lambda: denoise_t2va(
                transformer,
                conditioning,
                height=height,
                width=width,
                frame_count=shape.frame_count,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                device=self.device,
            ),
            metrics,
        )
        video_latents = video_latents.cpu()
        audio_latents = audio_latents.cpu()
        del networks
        transformer = None
        gc.collect()
        clean_memory_on_device(self.device)

        video_decoder, audio_decoder = self._measure(
            "decoder_load",
            lambda: (
                load_video_vae_decoder(self.video_vae, "cpu"),
                load_audio_vae_decoder(self.audio_vae, "cpu"),
            ),
            metrics,
        )
        media = self._measure(
            "sequential_av_decode",
            lambda: decode_latents_sequentially(
                video_decoder,
                audio_decoder,
                video_latents,
                audio_latents,
                self.device,
            ),
            metrics,
        )
        metrics["total"] = {"seconds": time.perf_counter() - total_started}
        save_av_mp4(
            media,
            request.output,
            {
                "prompt": request.prompt,
                "seed": request.seed,
                "height": height,
                "width": width,
                "frames": shape.frame_count,
                "fps": media.fps,
                "sample_rate": media.sample_rate,
                "sigma_points": self.num_inference_steps,
                "model_evaluations": self.num_inference_steps - 1,
                "lora_weights": [path.name for path in self.lora_weights],
                "metrics": metrics,
            },
        )


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
