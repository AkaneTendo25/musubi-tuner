from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from musubi_tuner.minimax_h3.architecture import IMAGE_FRAME_COUNT, temporal_shape
from musubi_tuner.minimax_h3.audio import (
    audio_valid_mask_to_latent_mask,
    load_audio_asset,
    target_audio_processing_spec,
)
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_KEYFRAME_VIDEO_ROWS_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_VIDEO_GEOMETRY_KEY,
)
from musubi_tuner.minimax_h3.component_loader import (
    load_audio_vae_decoder,
    load_audio_vae_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
)
from musubi_tuner.minimax_h3.inference import (
    decode_latents_sequentially,
    denoise_fl2va,
    denoise_ref2va,
    encode_keyframe_images,
    encode_reference_media,
    prepare_keyframe_image,
    resolve_canvas_size,
    save_av_mp4,
)
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality
from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag
from musubi_tuner.minimax_h3.packing import (
    AUDIO_CHANNELS,
    MiniMaxH3ReferenceGeometry,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    build_t2va_packed_sequence,
    pack_audio_latents,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from musubi_tuner.minimax_h3.references import H3ReferenceKind, prepare_references, trim_reference_frames
from musubi_tuner.minimax_h3.request import H3GenerationRequest, ReferenceKind, ReferenceRole
from musubi_tuner.minimax_h3.training import H3ModelPrediction, H3TrainingMode
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.model_utils import dtype_to_str, str_to_dtype

logger = logging.getLogger(__name__)


def create_latent_encoder(
    *,
    video_vae: Path | None,
    audio_vae: Path | None,
    device: str | None,
    dtype: str,
):
    """Load the released video VAE and the optional target/reference audio VAE."""
    target_device = torch.device(device or "cpu")
    output_dtype = str_to_dtype(dtype)
    video_encoder = load_video_vae_encoder(video_vae, target_device) if video_vae is not None else None
    audio_encoder = load_audio_vae_encoder(audio_vae, target_device) if audio_vae is not None else None
    return _NativeLatentEncoder(video_encoder, audio_encoder, output_dtype)


def create_conditioning_encoder(
    *,
    text_encoder: Path,
    tokenizer: Path,
    task: Literal["t2va", "i2va", "fl2va", "ref2va"],
    device: str | None,
    dtype: str,
    quantization: Literal["none", "int8", "nf4", "nvfp4_awq"] = "none",
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
    int8_convrot: bool = False,
    text_encoder_quantization: Literal["none", "int8", "nf4", "nvfp4_awq"] = "none",
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    block_swap_ring_size: int = 2,
    block_swap_granularity: Literal["block", "layer"] = "block",
    use_pinned_memory_for_block_swap: bool = False,
    lora_weights: tuple[Path, ...] = (),
    lora_multipliers: tuple[float, ...] = (),
):
    """Create a sequentially-loaded native FL2VA or Ref2VA generator."""
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
        int8_convrot=int8_convrot,
        text_encoder_quantization=text_encoder_quantization,
        blocks_to_swap=blocks_to_swap,
        block_swap_h2d_only=block_swap_h2d_only,
        block_swap_ring_size=block_swap_ring_size,
        block_swap_granularity=block_swap_granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
        lora_weights=lora_weights,
        lora_multipliers=lora_multipliers,
        mode="ref2va" if request.mode == "reference" else "fl2va",
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
        int8_convrot: bool,
        text_encoder_quantization: Literal["none", "int8", "nf4", "nvfp4_awq"],
        blocks_to_swap: int,
        block_swap_h2d_only: bool,
        block_swap_ring_size: int,
        block_swap_granularity: Literal["block", "layer"],
        use_pinned_memory_for_block_swap: bool,
        lora_weights: tuple[Path, ...],
        lora_multipliers: tuple[float, ...],
        mode: H3TrainingMode,
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
        self.int8_convrot = int8_convrot
        self.text_encoder_quantization = text_encoder_quantization
        self.blocks_to_swap = blocks_to_swap
        self.block_swap_h2d_only = block_swap_h2d_only
        self.block_swap_ring_size = block_swap_ring_size
        self.block_swap_granularity = block_swap_granularity
        self.use_pinned_memory_for_block_swap = use_pinned_memory_for_block_swap
        self.lora_weights = tuple(Path(path) for path in lora_weights)
        self.lora_multipliers = lora_multipliers
        self.mode = mode

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

    def _encode_prompt(self, prompt: str, images=(), references=()) -> dict[str, torch.Tensor]:
        task = "ref2va" if references else "t2va" if not images else "i2va" if len(images) == 1 else "fl2va"
        encoder = create_conditioning_encoder(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            task=task,
            device=str(self.device),
            dtype="bfloat16",
            quantization=self.text_encoder_quantization,
        )
        conditioning = encoder.encode_reference_prompt(prompt, references) if references else encoder.encode_prompt(prompt, images)
        del encoder
        gc.collect()
        clean_memory_on_device(self.device)
        return conditioning

    def _prepare_keyframes(
        self,
        request: H3GenerationRequest,
        height: int,
        width: int,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        images = []
        anchors = []
        for reference in request.references:
            if reference.role is ReferenceRole.FIRST_FRAME:
                with Image.open(reference.path) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=True))
                anchors.append("first")
            elif reference.role is ReferenceRole.LAST_FRAME:
                with Image.open(reference.path) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=False))
                anchors.append("last")
        return images, tuple(anchors)

    @staticmethod
    def _prepare_references(request: H3GenerationRequest):
        modality_by_kind = {
            ReferenceKind.IMAGE: MediaModality.IMAGE,
            ReferenceKind.VIDEO: MediaModality.VIDEO,
            ReferenceKind.AUDIO: MediaModality.AUDIO,
        }
        assets = tuple(
            MediaAsset(reference.path, modality_by_kind[reference.kind], "reference")
            for reference in request.references
            if reference.role is ReferenceRole.REFERENCE
        )
        return prepare_references(SimpleNamespace(h3_media_assets=assets, frame_count=request.temporal_shape.frame_count))

    def _load_transformer(self):
        from safetensors.torch import load_file

        from musubi_tuner.minimax_h3.model_loader import load_transformer
        from musubi_tuner.networks import lora_minimax_h3

        loading_device = torch.device("cpu") if self.blocks_to_swap else self.device
        transformer = load_transformer(
            self.model,
            mode=self.mode,
            loading_device=loading_device,
            fp8_scaled=self.fp8_scaled,
            quantization_device=self.device if self.fp8_scaled else None,
            int8_convrot=self.int8_convrot,
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
        metrics: dict[str, dict] = {}
        total_started = time.perf_counter()
        height, width = (self.height, self.width) if self.height is not None else resolve_canvas_size(request.ratio)
        shape = request.temporal_shape
        references = None
        prepared_references = ()
        if request.mode == "reference":
            prepared_references = self._prepare_references(request)
            conditioning = self._measure(
                "text_conditioning",
                lambda: self._encode_prompt(request.prompt, references=prepared_references),
                metrics,
            )
            references = self._measure(
                "reference_encoding",
                lambda: encode_reference_media(
                    self.video_vae,
                    self.audio_vae,
                    prepared_references,
                    self.device,
                ),
                metrics,
            )
            images, anchors, keyframe_rows = [], (), None
            reference_kinds = [reference.kind.name.lower() for reference in prepared_references]
            prepared_references = ()
            gc.collect()
        else:
            images, anchors = self._prepare_keyframes(request, height, width)
            conditioning = self._measure("text_conditioning", lambda: self._encode_prompt(request.prompt, images), metrics)
            keyframe_rows = (
                self._measure(
                    "keyframe_encoding",
                    lambda: torch.cat(encode_keyframe_images(self.video_vae, images, self.device)),
                    metrics,
                )
                if images
                else None
            )
            reference_kinds = []
        loaded_transformer = self._measure("transformer_load", self._load_transformer, metrics)
        transformer, networks = loaded_transformer
        del loaded_transformer
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        if references is not None:
            denoise = lambda: denoise_ref2va(
                transformer,
                conditioning,
                references,
                height=height,
                width=width,
                frame_count=shape.frame_count,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                device=self.device,
                condition_seed=request.seed,
            )
        else:
            denoise = lambda: denoise_fl2va(
                transformer,
                conditioning,
                height=height,
                width=width,
                frame_count=shape.frame_count,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                device=self.device,
                keyframe_rows=keyframe_rows,
                keyframe_anchors=anchors,
                condition_seed=request.seed,
            )
        video_latents, audio_latents = self._measure(
            "joint_denoising",
            denoise,
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
                "keyframe_anchors": list(anchors),
                "reference_kinds": reference_kinds,
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
    int8_convrot: bool = False,
    adaln_rank: int | None = None,
):
    """Load the selected released transformer and adapt its training forward to Musubi."""
    if dtype != "bfloat16":
        raise ValueError("MiniMax H3 full-checkpoint training requires bfloat16 compute")
    if attention_mode not in {"torch", "flash", "flash3"} or split_attention:
        raise ValueError("the native MiniMax H3 backend supports only unsplit SDPA, FlashAttention 2, or FlashAttention 3")
    from musubi_tuner.minimax_h3.model_loader import load_transformer

    transformer = load_transformer(
        model,
        mode=mode,
        loading_device=device or "cpu",
        fp8_scaled=fp8_scaled,
        quantization_device=quantization_device,
        int8_convrot=int8_convrot,
        adaln_rank=adaln_rank,
        attention_mode=attention_mode,
    )
    return _NativeTrainingBackend(transformer, mode)


class _NativeTrainingBackend:
    def __init__(self, transformer: torch.nn.Module, mode: H3TrainingMode = "fl2va"):
        self.transformer = transformer
        self.mode = mode

    def get_training_transformer(self) -> torch.nn.Module:
        return self.transformer

    def predict_training(
        self,
        transformer: torch.nn.Module,
        batch: dict,
        video_hidden_states: torch.Tensor | None,
        audio_hidden_states: torch.Tensor | None,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        *,
        conditioning: Literal["prompt", "empty"] = "prompt",
        extension_video_frames: int = 0,
        extension_audio_latents: int = 0,
    ) -> H3ModelPrediction:
        present = video_hidden_states if video_hidden_states is not None else audio_hidden_states
        if present is None:
            raise ValueError("MiniMax H3 training requires at least one target modality")
        if present.shape[0] != 1 or (
            video_hidden_states is not None and audio_hidden_states is not None and audio_hidden_states.shape[0] != 1
        ):
            raise ValueError("MiniMax H3 training requires batch size 1")
        config = getattr(transformer, "config", getattr(self.transformer, "config", None))
        if config is None:
            raise TypeError("MiniMax H3 transformer must expose its released config")
        if video_hidden_states is not None and (
            video_hidden_states.ndim != 5 or video_hidden_states.shape[1] != config.in_channels
        ):
            raise ValueError(
                f"H3 video input must have shape [1, {config.in_channels}, T, H, W], got {tuple(video_hidden_states.shape)}"
            )
        if audio_hidden_states is not None and (
            audio_hidden_states.ndim != 4 or audio_hidden_states.shape[1:3] != (2, config.audio_in_channels)
        ):
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
        conditioning_task = self._one_conditioning_item(batch, H3_CONDITIONING_TASK_KEY, expected_ndim=0)
        if text_hidden.ndim != 2 or text_hidden.shape[-1] != config.text_dim:
            raise ValueError(f"H3 {hidden_key} must have shape [tokens, {config.text_dim}]")
        if text_tags.dtype != torch.long or text_tags.shape != (text_hidden.shape[0],):
            raise ValueError(f"H3 {tags_key} must be int64 with one tag per text token")
        if bool(((text_tags < int(MiniMaxH3TokenTag.VIDEO)) | (text_tags > int(MiniMaxH3TokenTag.TEXT))).any()):
            raise ValueError("MiniMax H3 text cache contains invalid Ref2VA modality tags")
        if conditioning_task.dtype != torch.long:
            raise ValueError(f"H3 {H3_CONDITIONING_TASK_KEY} must be int64")
        task_id = int(conditioning_task.detach().cpu())
        task_by_id = {value: key for key, value in H3_CONDITIONING_TASK_IDS.items()}
        task = task_by_id.get(task_id)
        if self.mode == "ref2va":
            accepted_tasks = {"ref2va"}
        elif self.mode == "ref2va_omni":
            accepted_tasks = {"ref2va_omni"}
        else:
            accepted_tasks = {"t2va", "i2va", "fl2va"}
        if task not in accepted_tasks:
            expected = ", ".join(f"--task {name}" for name in sorted(accepted_tasks))
            raise ValueError(f"MiniMax H3 {self.mode} training requires {expected} conditioning; re-cache text outputs")
        has_vision = bool((text_tags == int(MiniMaxH3TokenTag.VIDEO)).any())
        if task == "t2va" and has_vision:
            raise ValueError("MiniMax H3 T2VA training requires text-only conditioning; re-cache with --task t2va")
        if task in ("i2va", "fl2va") and not has_vision:
            raise ValueError(f"MiniMax H3 {task.upper()} training requires keyframe vision rows; re-cache with --task {task}")
        if task == "ref2va" and not has_vision:
            raise ValueError("MiniMax H3 Ref2VA training requires a reference presentation; re-cache with --task ref2va")

        patch_size = tuple(config.patch_size)
        model_device = present.device
        if video_hidden_states is None:
            geometry = self._one_conditioning_item(batch, H3_VIDEO_GEOMETRY_KEY, expected_ndim=1)
            if geometry.shape != (2,):
                raise ValueError(f"H3 {H3_VIDEO_GEOMETRY_KEY} must contain latent height and width")
            latent_frames = 0
            latent_height, latent_width = (int(value) for value in geometry.detach().cpu())
            video_width = config.in_channels * int(np.prod(patch_size))
            video_rows = torch.empty((1, 0, video_width), device=model_device, dtype=present.dtype)
        else:
            video_rows = patchify_video_latents(video_hidden_states, patch_size)
            _, _, latent_frames, latent_height, latent_width = video_hidden_states.shape
        if audio_hidden_states is None:
            audio_rows = torch.empty((1, 0, config.audio_in_channels), device=model_device, dtype=present.dtype)
            num_audio_latents = 0
        else:
            audio_rows = pack_audio_latents(audio_hidden_states)
            num_audio_latents = int(audio_hidden_states.shape[-1])
        is_image_target = video_hidden_states is not None and latent_frames == 1 and audio_hidden_states is None
        if is_image_target and (self.mode != "fl2va" or task != "t2va"):
            raise ValueError("MiniMax H3 image training currently requires the FL2VA transformer with --task t2va caches")
        if self.mode in ("ref2va", "ref2va_omni"):
            references, reference_video, reference_audio = self._reference_cache(
                batch,
                patch_size=patch_size,
                video_width=video_rows.shape[-1],
                audio_width=audio_rows.shape[-1],
                device=model_device,
                dtype=video_rows.dtype,
            )
            layout = build_ref2va_packed_sequence(
                text_tags,
                references,
                num_latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
            )
            condition_video_timestep = torch.maximum(
                video_timestep.reshape(1).to(model_device, torch.float32),
                torch.tensor([0.999], device=model_device),
            )
            if reference_video.numel():
                reference_video = 0.999 * reference_video + 0.001 * torch.randn_like(reference_video)
                video_rows = torch.cat((reference_video[None], video_rows), dim=1)
            if reference_audio.numel():
                audio_rows = torch.cat((reference_audio[None], audio_rows), dim=1)
            timestep, timestep_indices = build_row_timesteps(
                layout,
                video_timestep,
                audio_timestep,
                condition_video_timestep,
                torch.ones(1, device=model_device),
            )
        elif task in ("i2va", "fl2va"):
            anchors = ("first",) if task == "i2va" else ("first", "last")
            layout = build_t2va_packed_sequence(
                text_tags,
                num_latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
                keyframe_anchors=anchors,
            )
            keyframe_rows = self._keyframe_cache(
                batch,
                num_anchors=len(anchors),
                rows_per_anchor=layout.num_condition_video_rows // len(anchors),
                row_width=video_rows.shape[-1],
                device=model_device,
                dtype=video_rows.dtype,
            )
            keyframe_rows = 0.999 * keyframe_rows + 0.001 * torch.randn_like(keyframe_rows)
            video_rows = torch.cat((keyframe_rows[None], video_rows), dim=1)
            condition_video_timestep = torch.maximum(
                video_timestep.reshape(1).to(model_device, torch.float32),
                torch.tensor([0.999], device=model_device),
            )
            timestep, timestep_indices = build_row_timesteps(
                layout,
                video_timestep,
                audio_timestep,
                condition_video_timestep,
            )
        else:
            # Extension observes a leading run of the target itself, so the
            # condition rows are sliced from the already-packed target rows
            # rather than cached separately. The target keeps its full length;
            # the observed span is removed from the loss by the caller.
            if extension_video_frames and extension_video_frames >= latent_frames:
                raise ValueError(
                    f"H3 video extension needs a shorter context than the target: "
                    f"{extension_video_frames} of {latent_frames} latent frames"
                )
            if extension_audio_latents and extension_audio_latents >= num_audio_latents:
                raise ValueError(
                    f"H3 audio extension needs a shorter context than the target: "
                    f"{extension_audio_latents} of {num_audio_latents} audio latents"
                )
            layout = build_t2va_packed_sequence(
                text_tags,
                num_latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
                keyframe_anchors=tuple(range(extension_video_frames)),
                num_condition_audio_latents=extension_audio_latents,
            )
            condition_video_timestep = None
            condition_audio_timestep = None
            if extension_video_frames:
                context_rows = video_rows[:, : layout.num_condition_video_rows]
                context_rows = 0.999 * context_rows + 0.001 * torch.randn_like(context_rows)
                video_rows = torch.cat((context_rows, video_rows), dim=1)
                condition_video_timestep = torch.maximum(
                    video_timestep.reshape(1).to(model_device, torch.float32),
                    torch.tensor([0.999], device=model_device),
                )
            if extension_audio_latents:
                # Audio rows are channel-major, so the observed prefix is the
                # opening latents of each channel rather than a flat slice.
                per_channel = audio_rows.shape[1] // AUDIO_CHANNELS
                context_audio = torch.cat(
                    [
                        audio_rows[:, channel * per_channel : channel * per_channel + extension_audio_latents]
                        for channel in range(AUDIO_CHANNELS)
                    ],
                    dim=1,
                )
                audio_rows = torch.cat((context_audio, audio_rows), dim=1)
                condition_audio_timestep = torch.ones(1, device=model_device)
            timestep, timestep_indices = build_row_timesteps(
                layout,
                video_timestep,
                audio_timestep,
                condition_video_timestep,
                condition_audio_timestep,
            )
        crepa = getattr(transformer, "_h3_crepa_controller", None)
        if crepa is not None and video_hidden_states is not None:
            patch_h, patch_w = patch_size[-2:]
            target_video_indices = layout.video_indices[layout.num_condition_video_rows :]
            crepa.set_layout(
                target_video_indices,
                latent_frames,
                (latent_height // patch_h) * (latent_width // patch_w),
            )
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
        target_video = output.video[:, layout.num_condition_video_rows :]
        target_audio = output.audio[:, layout.num_condition_audio_rows :]
        video = (
            unpatchify_video_tokens(
                target_video,
                latent_shape=(config.in_channels, latent_frames, latent_height, latent_width),
                patch_size=patch_size,
            )
            if video_hidden_states is not None
            else None
        )
        audio = unpack_audio_tokens(target_audio, num_audio_latents=num_audio_latents) if audio_hidden_states is not None else None
        return H3ModelPrediction(video=video, audio=audio)

    def _keyframe_cache(
        self,
        batch: dict,
        *,
        num_anchors: int,
        rows_per_anchor: int,
        row_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = self._one_conditioning_item(batch, H3_KEYFRAME_VIDEO_ROWS_KEY, expected_ndim=2)
        expected_all_rows = 2 * rows_per_anchor
        if rows.shape != (expected_all_rows, row_width):
            raise ValueError(f"H3 keyframe cache has shape {tuple(rows.shape)}, expected {(expected_all_rows, row_width)}")
        selected_rows = rows[: num_anchors * rows_per_anchor]
        return selected_rows.to(device=device, dtype=dtype)

    def _reference_cache(
        self,
        batch: dict,
        *,
        patch_size: tuple[int, int, int],
        video_width: int,
        audio_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[tuple[MiniMaxH3ReferenceGeometry, ...], torch.Tensor, torch.Tensor]:
        reference_keys = {
            H3_REFERENCE_KINDS_KEY,
            H3_REFERENCE_VIDEO_SHAPES_KEY,
            H3_REFERENCE_AUDIO_LENGTHS_KEY,
            H3_REFERENCE_VIDEO_ROWS_KEY,
            H3_REFERENCE_AUDIO_ROWS_KEY,
        }
        present_keys = reference_keys.intersection(batch)
        if not present_keys:
            if self.mode != "ref2va_omni":
                raise KeyError(f"MiniMax H3 Ref2VA training cache is missing {H3_REFERENCE_KINDS_KEY}")
            return (
                (),
                torch.empty((0, video_width), device=device, dtype=dtype),
                torch.empty((0, audio_width), device=device, dtype=dtype),
            )
        if present_keys != reference_keys:
            missing = ", ".join(sorted(reference_keys - present_keys))
            raise KeyError(f"H3 Ref2VA cache has a partial reference bundle; missing {missing}")
        kinds = self._one_conditioning_item(batch, H3_REFERENCE_KINDS_KEY, expected_ndim=1).to(torch.long)
        video_shapes = self._one_conditioning_item(batch, H3_REFERENCE_VIDEO_SHAPES_KEY, expected_ndim=2).to(torch.long)
        audio_lengths = self._one_conditioning_item(batch, H3_REFERENCE_AUDIO_LENGTHS_KEY, expected_ndim=1).to(torch.long)
        video_rows = self._one_conditioning_item(batch, H3_REFERENCE_VIDEO_ROWS_KEY, expected_ndim=2)
        audio_rows = self._one_conditioning_item(batch, H3_REFERENCE_AUDIO_ROWS_KEY, expected_ndim=2)
        if video_shapes.shape != (kinds.numel(), 3) or audio_lengths.shape != kinds.shape:
            raise ValueError("H3 Ref2VA cache has inconsistent reference metadata")
        kind_values = kinds.detach().cpu().tolist()
        shape_values = video_shapes.detach().cpu().tolist()
        audio_length_values = audio_lengths.detach().cpu().tolist()
        references = tuple(
            MiniMaxH3ReferenceGeometry(
                kind=int(kind),
                num_latent_frames=int(shape[0]),
                latent_height=int(shape[1]),
                latent_width=int(shape[2]),
                num_audio_latents=int(audio_length),
            )
            for kind, shape, audio_length in zip(kind_values, shape_values, audio_length_values)
        )
        expected_video_rows = sum(reference.num_video_rows(patch_size) for reference in references)
        expected_audio_rows = sum(reference.num_audio_rows for reference in references)
        if video_rows.shape != (expected_video_rows, video_width):
            raise ValueError(
                f"H3 Ref2VA video cache has shape {tuple(video_rows.shape)}, expected {(expected_video_rows, video_width)}"
            )
        if audio_rows.shape != (expected_audio_rows, audio_width):
            raise ValueError(
                f"H3 Ref2VA audio cache has shape {tuple(audio_rows.shape)}, expected {(expected_audio_rows, audio_width)}"
            )
        return references, video_rows.to(device=device, dtype=dtype), audio_rows.to(device=device, dtype=dtype)

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
        video_encoder: torch.nn.Module | None,
        audio_encoder: torch.nn.Module | None,
        output_dtype: torch.dtype,
    ) -> None:
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.output_dtype = output_dtype

    @staticmethod
    def _target_asset(item: Any):
        assets = getattr(item, "h3_media_assets", ())
        targets = [
            asset
            for asset in assets
            if asset.role == "target" and asset.modality in {MediaModality.IMAGE, MediaModality.VIDEO, MediaModality.AUDIO}
        ]
        if len(targets) != 1:
            raise ValueError(f"H3 item {item.item_key!r} must have one attached target image, video, or audio clip")
        return targets[0]

    def _encode_video(self, content: np.ndarray, *, is_image: bool) -> torch.Tensor:
        if self.video_encoder is None:
            raise ValueError("MiniMax H3 visual latent caching requires --vae")
        if not isinstance(content, np.ndarray):
            raise TypeError("MiniMax H3 latent caching requires a numpy image or video array")
        if content.ndim == 3:
            content = content[None]
        if content.ndim != 4 or content.shape[-1] != 3:
            raise ValueError(f"H3 video content must have shape [F, H, W, 3], got {content.shape}")
        pixels = torch.from_numpy(np.array(content, copy=True, order="C")).permute(3, 0, 1, 2).unsqueeze(0)
        device = next(self.video_encoder.parameters()).device
        weight_dtype = next(self.video_encoder.parameters()).dtype
        pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        pixels = ((pixels - pixel_mean) / pixel_std).to(weight_dtype)
        with torch.no_grad():
            encode = self.video_encoder.encode_image if is_image else self.video_encoder.encode
            return encode(pixels)[0].to(self.output_dtype)

    def _encode_reference_video(self, content: np.ndarray, *, image: bool) -> torch.Tensor:
        if self.video_encoder is None:
            raise ValueError("MiniMax H3 visual references require --vae during latent caching")
        if content.ndim == 3:
            content = content[None]
        pixels = torch.from_numpy(np.array(content, copy=True, order="C")).permute(3, 0, 1, 2).unsqueeze(0)
        device = next(self.video_encoder.parameters()).device
        weight_dtype = next(self.video_encoder.parameters()).dtype
        pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        pixels = ((pixels - pixel_mean) / pixel_std).to(weight_dtype)
        with torch.no_grad():
            return self.video_encoder.encode_reference(pixels, image=image)[0].to(self.output_dtype)

    def _encode_reference_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.audio_encoder is None:
            raise ValueError("MiniMax H3 audio references require --audio_vae during latent caching")
        device = next(self.audio_encoder.parameters()).device
        with torch.no_grad():
            latents = self.audio_encoder.encode(waveform.to(device=device, dtype=torch.float32).unsqueeze(1))
        return latents.to(self.output_dtype)

    def _encode_references(self, item: Any) -> dict[str, torch.Tensor]:
        references = prepare_references(item)
        if not references:
            return {}
        video_rows: list[torch.Tensor] = []
        audio_rows: list[torch.Tensor] = []
        video_shapes: list[tuple[int, int, int]] = []
        audio_lengths: list[int] = []
        kinds: list[int] = []
        for reference in references:
            kinds.append(int(reference.kind))
            if reference.kind is H3ReferenceKind.IMAGE:
                if reference.image is None:
                    raise ValueError("H3 prepared image reference has no image")
                latent = self._encode_reference_video(np.asarray(reference.image), image=True)
            elif reference.kind is H3ReferenceKind.VIDEO:
                if reference.frames is None:
                    raise ValueError("H3 prepared video reference has no frames")
                frames = reference.frames[: trim_reference_frames(reference.frames.shape[0])]
                latent = self._encode_reference_video(frames, image=False)
            else:
                latent = None
            if latent is None:
                video_shapes.append((0, 0, 0))
            else:
                video_shapes.append(tuple(int(value) for value in latent.shape[-3:]))
                video_rows.append(patchify_video_latents(latent[None], (1, 2, 2))[0])

            if reference.waveform is None:
                audio_lengths.append(0)
            else:
                audio = self._encode_reference_audio(reference.waveform)
                audio_lengths.append(int(audio.shape[-1]))
                audio_rows.append(pack_audio_latents(audio[None])[0])

        dtype_name = dtype_to_str(self.output_dtype)
        return {
            f"varlen_{H3_REFERENCE_KINDS_KEY}_int64": torch.tensor(kinds, dtype=torch.long),
            f"varlen_{H3_REFERENCE_VIDEO_SHAPES_KEY}_int64": torch.tensor(video_shapes, dtype=torch.long),
            f"varlen_{H3_REFERENCE_AUDIO_LENGTHS_KEY}_int64": torch.tensor(audio_lengths, dtype=torch.long),
            f"varlen_{H3_REFERENCE_VIDEO_ROWS_KEY}_{dtype_name}": (
                torch.cat(video_rows) if video_rows else torch.empty((0, 96), dtype=self.output_dtype)
            ),
            f"varlen_{H3_REFERENCE_AUDIO_ROWS_KEY}_{dtype_name}": (
                torch.cat(audio_rows) if audio_rows else torch.empty((0, 32), dtype=self.output_dtype)
            ),
        }

    def _encode_audio(self, item: Any) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._target_asset(item)
        if target.modality not in {MediaModality.VIDEO, MediaModality.AUDIO}:
            raise ValueError("MiniMax H3 target audio is defined only for video or audio targets")
        if self.audio_encoder is None:
            raise ValueError("MiniMax H3 video latent caching requires --audio_vae")
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
            target = self._target_asset(item)
            target_mode = getattr(item, "h3_target_mode", "av")
            if target_mode == "audio":
                expected = temporal_shape(int(target.metadata["frame_count"]))
                audio, audio_mask = self._encode_audio(item)
                if audio.shape[-1] != expected.audio_latent_frames:
                    raise ValueError(f"H3 audio VAE produced {audio.shape[-1]} rows; expected {expected.audio_latent_frames}")
                latent_height = int(item.original_size[1]) // 16
                latent_width = int(item.original_size[0]) // 16
                results.append(
                    {
                        f"{H3_AUDIO_LATENTS_KEY}_2x32x{audio.shape[-1]}_{dtype_name}": audio,
                        H3_AUDIO_LOSS_MASK_KEY: audio_mask,
                        f"{H3_VIDEO_GEOMETRY_KEY}_int64": torch.tensor([latent_height, latent_width], dtype=torch.long),
                    }
                )
                continue
            is_image = target.modality is MediaModality.IMAGE
            video = self._encode_video(item.content, is_image=is_image)
            video_frame_count = IMAGE_FRAME_COUNT if is_image else int(item.content.shape[0])
            expected_video_frames = IMAGE_FRAME_COUNT if is_image else temporal_shape(video_frame_count).video_latent_frames
            if video.shape[1] != expected_video_frames:
                raise ValueError(
                    f"H3 video VAE produced {video.shape[1]} latent frames for {video_frame_count} pixels; "
                    f"expected {expected_video_frames}"
                )
            frame_shape = "x".join(str(value) for value in video.shape[-3:])
            tensors = {f"latents_{frame_shape}_{dtype_name}": video}
            if not is_image and target_mode != "video":
                expected = temporal_shape(video_frame_count)
                audio, audio_mask = self._encode_audio(item)
                if audio.shape[-1] != expected.audio_latent_frames:
                    raise ValueError(f"H3 audio VAE produced {audio.shape[-1]} rows; expected {expected.audio_latent_frames}")
                tensors.update(
                    {
                        f"{H3_AUDIO_LATENTS_KEY}_2x32x{audio.shape[-1]}_{dtype_name}": audio,
                        H3_AUDIO_LOSS_MASK_KEY: audio_mask,
                    }
                )
            if not is_image:
                first = self._encode_reference_video(item.content[0], image=True)
                last = self._encode_reference_video(item.content[-1], image=True)
                keyframe_rows = torch.cat(
                    (
                        patchify_video_latents(first[None], (1, 2, 2))[0],
                        patchify_video_latents(last[None], (1, 2, 2))[0],
                    )
                )
                tensors[f"varlen_{H3_KEYFRAME_VIDEO_ROWS_KEY}_{dtype_name}"] = keyframe_rows
            tensors.update(self._encode_references(item))
            results.append(tensors)
        return tuple(results)
