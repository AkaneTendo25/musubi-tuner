from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from musubi_tuner.minimax_h3.architecture import (
    AUDIO_LATENT_CHANNELS,
    IMAGE_FRAME_COUNT,
    VIDEO_DIT_PATCH_SIZE,
    VIDEO_LATENT_CHANNELS,
    temporal_shape,
)
from musubi_tuner.minimax_h3.audio import (
    audio_valid_mask_to_latent_mask,
    load_audio_asset,
    target_audio_processing_spec,
)
from musubi_tuner.minimax_h3.backend import H3PairedConditioningUnsupportedError
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_KEYFRAME_VIDEO_ROWS_KEY,
    H3_KEYFRAME_VISUAL_LAST,
    H3_KEYFRAME_VISUALS_KEY,
    H3_MAX_CAPTION_TOKENS_KEY,
    H3_QWEN_CONTROL_VISUALS_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_IMAGE_MAX_PIXELS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_REFERENCE_IMAGE_SIZE_MODE_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_VERSION,
    H3_REFERENCE_VIDEO_FPS_KEY,
    H3_REFERENCE_VIDEO_MAX_PIXELS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_REFERENCE_VIDEO_SHORT_EDGE_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_VISUAL_MAX_PIXELS_KEY,
    H3_VIDEO_GEOMETRY_KEY,
    format_keyframe_visuals,
    qwen_control_dropout_key,
    reference_key_suffix,
    reference_variant_key,
)
from musubi_tuner.minimax_h3.component_loader import (
    load_audio_vae_decoder,
    load_audio_vae_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
)
from musubi_tuner.minimax_h3.inference import (
    CANVAS_MULTIPLE,
    VIDEO_SPATIAL_COMPRESSION,
    H3EncodedReferences,
    decode_latents_sequentially,
    denoise_fl2va,
    denoise_ref2va,
    encode_guide_media,
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
    MiniMaxH3GuideGeometry,
    MiniMaxH3ReferenceGeometry,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    build_t2va_packed_sequence,
    pack_audio_latents,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from musubi_tuner.minimax_h3.references import (
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODE,
    REFERENCE_VIDEO_FPS,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SHORT_EDGE,
    H3PreparedReference,
    H3ReferenceKind,
    prepare_references,
    trim_reference_frames,
)
from musubi_tuner.minimax_h3.request import H3GenerationRequest, ReferenceKind, ReferenceRole
from musubi_tuner.minimax_h3.training import H3ModelPrediction, H3TrainingMode
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.model_utils import dtype_to_str, str_to_dtype

logger = logging.getLogger(__name__)


def _round_to(value: float, multiple: int) -> int:
    """Nearest size that is a whole number of ``multiple``, never below one unit."""
    return max(multiple, int(round(value / multiple)) * multiple)


def _validate_inference_lora_metadata(
    path: Path,
    mode: str,
    reference_image_short_edge: int,
    reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    reference_image_max_pixels: int = 0,
    reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    text_visual_max_pixels: int = 0,
) -> None:
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    trained_mode = metadata.get("ss_h3_training_mode")
    saved_reference_size = metadata.get("ss_h3_reference_image_short_edge")
    saved_size_mode = metadata.get("ss_h3_reference_image_size_mode", REFERENCE_IMAGE_SIZE_MODE)
    try:
        saved_max_pixels = int(metadata.get("ss_h3_reference_image_max_pixels", "0"))
    except ValueError as exc:
        raise ValueError(f"invalid ss_h3_reference_image_max_pixels metadata in {path}") from exc
    try:
        saved_text_visual_max_pixels = int(metadata.get("ss_h3_text_visual_max_pixels", "0"))
    except ValueError as exc:
        raise ValueError(f"invalid ss_h3_text_visual_max_pixels metadata in {path}") from exc
    try:
        saved_video_short_edge = int(metadata.get("ss_h3_reference_video_short_edge", str(REFERENCE_VIDEO_SHORT_EDGE)))
        saved_video_max_pixels = int(metadata.get("ss_h3_reference_video_max_pixels", str(REFERENCE_VIDEO_MAX_PIXELS)))
    except ValueError as exc:
        raise ValueError(f"invalid H3 reference-video sizing metadata in {path}") from exc
    if trained_mode in {"ref2va", "ref2va_omni"} and saved_reference_size is not None:
        try:
            saved_reference_size_int = int(saved_reference_size)
        except ValueError as exc:
            raise ValueError(f"invalid ss_h3_reference_image_short_edge metadata in {path}") from exc
        if mode in {"ref2va", "ref2va_omni"} and saved_reference_size_int != reference_image_short_edge:
            raise ValueError(
                f"H3 LoRA {path} was trained with reference_image_short_edge={saved_reference_size_int}, "
                f"but inference requested {reference_image_short_edge}"
            )
    if trained_mode in {"ref2va", "ref2va_omni"} and mode in {"ref2va", "ref2va_omni"}:
        if saved_size_mode != reference_image_size_mode or saved_max_pixels != reference_image_max_pixels:
            raise ValueError(
                f"H3 LoRA {path} was trained with reference sizing {saved_size_mode}/{saved_max_pixels}, "
                f"but inference requested {reference_image_size_mode}/{reference_image_max_pixels}"
            )
        if saved_text_visual_max_pixels != text_visual_max_pixels:
            raise ValueError(
                f"H3 LoRA {path} was trained with h3_text_visual_max_pixels={saved_text_visual_max_pixels}, "
                f"but inference requested {text_visual_max_pixels}"
            )
        if saved_video_short_edge != reference_video_short_edge or saved_video_max_pixels != reference_video_max_pixels:
            raise ValueError(
                f"H3 LoRA {path} was trained with reference-video sizing "
                f"{saved_video_short_edge}/{saved_video_max_pixels}, but inference requested "
                f"{reference_video_short_edge}/{reference_video_max_pixels}"
            )
    adaln_rank = metadata.get("ss_h3_adaln_rank")
    if adaln_rank not in {None, "full"}:
        logger.warning(
            "H3 LoRA %s was trained against a rank-%s frozen AdaLN approximation; inference uses the selected "
            "checkpoint's AdaLN projections, so verify output fidelity against the training setup",
            path,
            adaln_rank,
        )


def create_latent_encoder(
    *,
    video_vae: Path | None,
    audio_vae: Path | None,
    device: str | None,
    dtype: str,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    reference_image_max_pixels: int = 0,
    reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    reference_video_fps: float = REFERENCE_VIDEO_FPS,
):
    """Load the released video VAE and the optional target/reference audio VAE."""
    target_device = torch.device(device or "cpu")
    output_dtype = str_to_dtype(dtype)
    video_encoder = load_video_vae_encoder(video_vae, target_device) if video_vae is not None else None
    audio_encoder = load_audio_vae_encoder(audio_vae, target_device) if audio_vae is not None else None
    return _NativeLatentEncoder(
        video_encoder,
        audio_encoder,
        output_dtype,
        reference_image_short_edge,
        reference_image_size_mode,
        reference_image_max_pixels,
        reference_video_short_edge,
        reference_video_max_pixels,
        reference_video_fps,
    )


def create_conditioning_encoder(
    *,
    text_encoder: Path,
    tokenizer: Path,
    task: Literal["t2va", "i2va", "fl2va", "l2va", "ref2va", "ref2va_omni"],
    device: str | None,
    dtype: str,
    quantization: Literal["none", "int8", "nf4", "nvfp4", "nvfp4_awq"] = "none",
    blocks_to_stream: int = 0,
    nvfp4_scaled_mm: bool = False,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    reference_image_max_pixels: int = 0,
    reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    reference_video_fps: float = REFERENCE_VIDEO_FPS,
    text_visual_max_pixels: int = 0,
    max_caption_tokens: int = 0,
    keyframe_visuals: tuple[int, ...] = (),
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
        blocks_to_stream=blocks_to_stream,
        nvfp4_scaled_mm=nvfp4_scaled_mm,
    )
    return MiniMaxH3ConditioningEncoder(
        processor,
        model,
        output_dtype,
        task,
        reference_image_short_edge=reference_image_short_edge,
        text_visual_max_pixels=text_visual_max_pixels,
        reference_image_size_mode=reference_image_size_mode,
        reference_image_max_pixels=reference_image_max_pixels,
        reference_video_short_edge=reference_video_short_edge,
        reference_video_max_pixels=reference_video_max_pixels,
        reference_video_fps=reference_video_fps,
        max_caption_tokens=max_caption_tokens,
        keyframe_visuals=keyframe_visuals,
    )


def create_generator(
    *,
    model: Path,
    text_encoder: Path,
    tokenizer: Path,
    video_vae: Path,
    audio_vae: Path | None,
    device: str | None,
    dtype: str,
    request: H3GenerationRequest,
    num_inference_steps: int = 20,
    height: int | None = None,
    width: int | None = None,
    fp8_scaled: bool = False,
    int8_convrot: bool = False,
    adaln_rank: int | None = None,
    text_encoder_quantization: Literal["none", "int8", "nf4", "nvfp4", "nvfp4_awq"] = "none",
    text_encoder_blocks_to_stream: int = 0,
    text_encoder_nvfp4_scaled_mm: bool = False,
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    block_swap_ring_size: int = 2,
    block_swap_granularity: Literal["block", "layer"] = "block",
    use_pinned_memory_for_block_swap: bool = False,
    lora_weights: tuple[Path, ...] = (),
    lora_multipliers: tuple[float, ...] = (),
    learned_contexts: tuple[Path, ...] = (),
    learned_context_multipliers: tuple[float, ...] = (),
    learned_context_composition: str = "prepend",
    first_pass_scale: float = 0.0,
    first_pass_steps: int = 0,
    second_pass_strength: float = 0.5,
    first_pass_lora: bool = False,
    latent_upscaler: Path | None = None,
    latent_upscale_scale: float = 1.0,
    compile_model: bool = False,
    compile_backend: str = "inductor",
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_dynamic: str | None = None,
    compile_fullgraph: bool = False,
    compile_cache_size_limit: int | None = None,
    compile_auto_cache_size_limit: bool = False,
    compile_fallback_to_eager: bool = False,
    inductor_config: tuple[str, ...] = (),
    fused_qk_norm_rope: bool = False,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    reference_image_max_pixels: int = 0,
    reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    text_visual_max_pixels: int = 0,
):
    """Create a sequentially-loaded native FL2VA or Ref2VA generator."""
    del adaln_rank  # AdaLN pruning is a training-time approximation; inference loads the checkpoint as stored.
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
        text_encoder_blocks_to_stream=text_encoder_blocks_to_stream,
        text_encoder_nvfp4_scaled_mm=text_encoder_nvfp4_scaled_mm,
        blocks_to_swap=blocks_to_swap,
        block_swap_h2d_only=block_swap_h2d_only,
        block_swap_ring_size=block_swap_ring_size,
        block_swap_granularity=block_swap_granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
        lora_weights=lora_weights,
        lora_multipliers=lora_multipliers,
        learned_contexts=learned_contexts,
        learned_context_multipliers=learned_context_multipliers,
        learned_context_composition=learned_context_composition,
        first_pass_scale=first_pass_scale,
        first_pass_steps=first_pass_steps,
        second_pass_strength=second_pass_strength,
        first_pass_lora=first_pass_lora,
        latent_upscaler=latent_upscaler,
        latent_upscale_scale=latent_upscale_scale,
        compile_model=compile_model,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_fullgraph=compile_fullgraph,
        compile_cache_size_limit=compile_cache_size_limit,
        compile_auto_cache_size_limit=compile_auto_cache_size_limit,
        compile_fallback_to_eager=compile_fallback_to_eager,
        inductor_config=inductor_config,
        fused_qk_norm_rope=fused_qk_norm_rope,
        mode="ref2va" if request.mode == "reference" else "fl2va",
        reference_image_short_edge=reference_image_short_edge,
        reference_image_size_mode=reference_image_size_mode,
        reference_image_max_pixels=reference_image_max_pixels,
        reference_video_short_edge=reference_video_short_edge,
        reference_video_max_pixels=reference_video_max_pixels,
        text_visual_max_pixels=text_visual_max_pixels,
    )


class _NativeGenerator:
    def __init__(
        self,
        *,
        model: Path,
        text_encoder: Path,
        tokenizer: Path,
        video_vae: Path,
        audio_vae: Path | None,
        device: torch.device,
        num_inference_steps: int,
        height: int | None,
        width: int | None,
        fp8_scaled: bool,
        int8_convrot: bool,
        text_encoder_quantization: Literal["none", "int8", "nf4", "nvfp4", "nvfp4_awq"],
        blocks_to_swap: int,
        block_swap_h2d_only: bool,
        block_swap_ring_size: int,
        block_swap_granularity: Literal["block", "layer"],
        use_pinned_memory_for_block_swap: bool,
        lora_weights: tuple[Path, ...],
        lora_multipliers: tuple[float, ...],
        compile_model: bool,
        compile_backend: str,
        compile_mode: str,
        compile_dynamic: str | None,
        compile_fullgraph: bool,
        compile_cache_size_limit: int | None,
        compile_auto_cache_size_limit: bool,
        compile_fallback_to_eager: bool,
        inductor_config: tuple[str, ...],
        fused_qk_norm_rope: bool,
        mode: H3TrainingMode,
        learned_contexts: tuple[Path, ...] = (),
        learned_context_multipliers: tuple[float, ...] = (),
        learned_context_composition: str = "prepend",
        text_encoder_blocks_to_stream: int = 0,
        text_encoder_nvfp4_scaled_mm: bool = False,
        reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
        reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
        reference_image_max_pixels: int = 0,
        reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
        reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
        text_visual_max_pixels: int = 0,
        first_pass_scale: float = 0.0,
        first_pass_steps: int = 0,
        second_pass_strength: float = 0.5,
        first_pass_lora: bool = False,
        latent_upscaler: Path | None = None,
        latent_upscale_scale: float = 1.0,
    ) -> None:
        self.model = Path(model)
        self.text_encoder = Path(text_encoder)
        self.tokenizer = Path(tokenizer)
        self.video_vae = Path(video_vae)
        self.audio_vae = Path(audio_vae) if audio_vae is not None else None
        self.device = device
        self.num_inference_steps = num_inference_steps
        # A second pass costs a whole canvas of denoising, so it stays off unless
        # a scale is given. Speed adapters are the reason it exists: the first
        # pass settles motion without one, the second sharpens with one.
        self.first_pass_scale = first_pass_scale
        self.first_pass_steps = first_pass_steps
        self.second_pass_strength = second_pass_strength
        self.first_pass_lora = first_pass_lora
        self.latent_upscaler = Path(latent_upscaler) if latent_upscaler else None
        self.latent_upscale_scale = latent_upscale_scale
        if (height is None) != (width is None):
            raise ValueError("MiniMax H3 height and width must be provided together")
        self.height = height
        self.width = width
        self.fp8_scaled = fp8_scaled
        self.int8_convrot = int8_convrot
        self.text_encoder_quantization = text_encoder_quantization
        self.text_encoder_blocks_to_stream = text_encoder_blocks_to_stream
        self.text_encoder_nvfp4_scaled_mm = text_encoder_nvfp4_scaled_mm
        self.blocks_to_swap = blocks_to_swap
        self.block_swap_h2d_only = block_swap_h2d_only
        self.block_swap_ring_size = block_swap_ring_size
        self.block_swap_granularity = block_swap_granularity
        self.use_pinned_memory_for_block_swap = use_pinned_memory_for_block_swap
        self.lora_weights = tuple(Path(path) for path in lora_weights)
        self.lora_multipliers = lora_multipliers
        from musubi_tuner.minimax_h3.learned_context import load_learned_context_sequence

        self.learned_context = load_learned_context_sequence(learned_contexts, learned_context_multipliers)
        if learned_context_composition not in {"prepend", "replace"}:
            raise ValueError("H3 learned-context composition must be prepend or replace")
        self.learned_context_composition = learned_context_composition
        self.compile_model = compile_model
        self.compile_options = SimpleNamespace(
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
            compile_fullgraph=compile_fullgraph,
            compile_cache_size_limit=compile_cache_size_limit,
            compile_auto_cache_size_limit=compile_auto_cache_size_limit,
            compile_fallback_to_eager=compile_fallback_to_eager,
            inductor_config=inductor_config,
        )
        self.fused_qk_norm_rope = fused_qk_norm_rope
        self.reference_image_short_edge = reference_image_short_edge
        self.reference_image_size_mode = reference_image_size_mode
        self.reference_image_max_pixels = reference_image_max_pixels
        self.reference_video_short_edge = reference_video_short_edge
        self.reference_video_max_pixels = reference_video_max_pixels
        self.text_visual_max_pixels = text_visual_max_pixels
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

    @staticmethod
    def _conditioning_task(images=(), references=(), keyframe_anchors=()) -> str:
        if references:
            return "ref2va"
        if not images:
            return "t2va"
        if tuple(keyframe_anchors) == ("last",):
            return "l2va"
        if tuple(keyframe_anchors) == ("first",):
            return "i2va"
        return "fl2va"

    def _encode_prompt(self, prompt: str, images=(), references=(), keyframe_anchors=()) -> dict[str, torch.Tensor]:
        task = self._conditioning_task(images, references, keyframe_anchors)
        encoder = create_conditioning_encoder(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            task=task,
            device=str(self.device),
            dtype="bfloat16",
            quantization=self.text_encoder_quantization,
            blocks_to_stream=self.text_encoder_blocks_to_stream,
            nvfp4_scaled_mm=self.text_encoder_nvfp4_scaled_mm,
            reference_image_short_edge=self.reference_image_short_edge,
            reference_image_size_mode=self.reference_image_size_mode,
            reference_image_max_pixels=self.reference_image_max_pixels,
            reference_video_short_edge=self.reference_video_short_edge,
            reference_video_max_pixels=self.reference_video_max_pixels,
            text_visual_max_pixels=self.text_visual_max_pixels,
        )
        try:
            conditioning = (
                encoder.encode_reference_prompt(prompt, references) if references else encoder.encode_prompt(prompt, images)
            )
        finally:
            encoder.close()
            del encoder
        gc.collect()
        clean_memory_on_device(self.device)
        from musubi_tuner.minimax_h3.learned_context import apply_learned_context

        return apply_learned_context(
            conditioning,
            self.learned_context,
            self.learned_context_composition,
        )

    def _prepare_keyframes(
        self,
        request: H3GenerationRequest,
        height: int,
        width: int,
    ) -> tuple[list[Image.Image], tuple[str | int, ...]]:
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
            elif reference.role is ReferenceRole.KEYFRAME:
                # An anchor is a clean row carrying a temporal position, and the
                # packing treats an interior index exactly as it treats the ends.
                with Image.open(reference.path) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=False))
                anchors.append(request.resolve_keyframe_index(reference.latent_index))
        return images, tuple(anchors)

    def _prepare_references(self, request: H3GenerationRequest, height: int, width: int):
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
        return prepare_references(
            SimpleNamespace(
                h3_media_assets=assets,
                frame_count=request.temporal_shape.frame_count,
                bucket_size=(width, height),
            ),
            self.reference_image_short_edge,
            self.reference_image_size_mode,
            self.reference_image_max_pixels,
            self.reference_video_short_edge,
            self.reference_video_max_pixels,
        )

    def _prepare_guides(self, request: H3GenerationRequest, height: int, width: int):
        """Decode request guides and place visual spans on the target canvas."""
        prepared = []
        frame_count = request.temporal_shape.frame_count
        for guide in request.guides:
            frame_index = guide.frame_index if guide.frame_index >= 0 else frame_count + guide.frame_index
            video_guide = None
            audio_guide = None
            if guide.image is not None:
                with Image.open(guide.image) as image:
                    image = prepare_keyframe_image(image, height, width, stretch=False)
                    video_guide = H3PreparedReference(H3ReferenceKind.IMAGE, image=image.copy())
            elif guide.video is not None:
                values = prepare_references(
                    SimpleNamespace(
                        h3_media_assets=(MediaAsset(guide.video, MediaModality.VIDEO, "reference"),),
                        frame_count=frame_count,
                        bucket_size=(width, height),
                    ),
                    self.reference_image_short_edge,
                    self.reference_image_size_mode,
                    self.reference_image_max_pixels,
                    self.reference_video_short_edge,
                    self.reference_video_max_pixels,
                )
                source = values[0]
                if source.frames is None or not source.frames.shape[0]:
                    raise ValueError(f"H3 guide video {guide.video} contains no frames")
                remaining = frame_count - frame_index
                frames = source.frames[:remaining]
                if frames.shape[0] < 5:
                    image = prepare_keyframe_image(Image.fromarray(frames[0]), height, width, stretch=False)
                    video_guide = H3PreparedReference(H3ReferenceKind.IMAGE, image=image)
                else:
                    valid = trim_reference_frames(frames.shape[0])
                    frames = np.stack(
                        [
                            np.asarray(prepare_keyframe_image(Image.fromarray(frame), height, width, stretch=False))
                            for frame in frames[:valid]
                        ]
                    )
                    video_guide = H3PreparedReference(H3ReferenceKind.VIDEO, frames=frames)
            if guide.audio is not None:
                values = prepare_references(
                    SimpleNamespace(
                        h3_media_assets=(MediaAsset(guide.audio, MediaModality.AUDIO, "reference"),),
                        frame_count=frame_count,
                        bucket_size=(width, height),
                    ),
                    self.reference_image_short_edge,
                    self.reference_image_size_mode,
                    self.reference_image_max_pixels,
                    self.reference_video_short_edge,
                    self.reference_video_max_pixels,
                )
                audio_guide = values[0]
            prepared.append((frame_index, video_guide, audio_guide))
        return tuple(prepared)

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
            target_device=self.device,
            blocks_to_swap=self.blocks_to_swap,
            block_swap_h2d_only=self.block_swap_h2d_only,
        )
        transformer.requires_grad_(False).eval()
        if self.fused_qk_norm_rope:
            transformer.enable_fused_qk_norm_rope()
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
            _validate_inference_lora_metadata(
                weights_path,
                self.mode,
                self.reference_image_short_edge,
                self.reference_image_size_mode,
                self.reference_image_max_pixels,
                self.reference_video_short_edge,
                self.reference_video_max_pixels,
                self.text_visual_max_pixels,
            )
            weights = load_file(weights_path)
            if lora_minimax_h3.is_foreign_lora(weights):
                # Community speed adapters ship ComfyUI/PEFT key names.
                weights = lora_minimax_h3.convert_foreign_lora(weights)
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
        if self.compile_model:
            from musubi_tuner.utils import model_utils

            targets = model_utils.resolve_compile_block_lists(transformer, ("blocks", "token_refiner.blocks"))
            transformer = model_utils.compile_transformer(
                self.compile_options,
                transformer,
                targets,
                disable_linear=self.blocks_to_swap > 0,
            )
        return transformer, networks

    @staticmethod
    def _set_lora_multiplier(networks, multiplier: float | None) -> None:
        """Scale every adapter module, or restore each module's own multiplier."""
        for network in networks:
            for module in getattr(network, "unet_loras", ()):
                if not hasattr(module, "_h3_base_multiplier"):
                    module._h3_base_multiplier = module.multiplier
                module.multiplier = module._h3_base_multiplier if multiplier is None else multiplier

    def _upscale_video_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resample video latents to the second-pass canvas.

        Spatial only: the frame count and the audio track are the same in both
        passes, and it is the canvas that changes.
        """
        frames = latents.shape[2]
        flat = latents.transpose(1, 2).flatten(0, 1)
        target = (height // VIDEO_SPATIAL_COMPRESSION, width // VIDEO_SPATIAL_COMPRESSION)
        flat = torch.nn.functional.interpolate(flat, size=target, mode="bicubic", align_corners=False)
        return flat.unflatten(0, (latents.shape[0], frames)).transpose(1, 2).contiguous()

    def _denoise_two_pass(self, denoise, networks, height: int, width: int, metrics: dict, first_pass_kwargs: dict) -> tuple:
        """Settle motion on a small canvas, then refine on the full one.

        A speed adapter trades motion for steps, so it is kept out of the pass
        that decides the motion and applied to the pass that only sharpens what
        is already decided.
        """
        # The canvas multiple, not the latent cell: a canvas the model refuses is
        # the one error this path cannot recover from mid-run.
        small_height = _round_to(height * self.first_pass_scale, CANVAS_MULTIPLE)
        small_width = _round_to(width * self.first_pass_scale, CANVAS_MULTIPLE)
        first_steps = self.first_pass_steps or self.num_inference_steps
        if not self.first_pass_lora:
            self._set_lora_multiplier(networks, 0.0)
        try:
            video, audio = self._measure(
                "first_pass",
                lambda: denoise(
                    height=small_height,
                    width=small_width,
                    num_inference_steps=first_steps,
                    **first_pass_kwargs,
                ),
                metrics,
            )
        finally:
            self._set_lora_multiplier(networks, None)
        video = self._upscale_video_latents(video, height, width)
        return self._measure(
            "second_pass",
            lambda: denoise(
                height=height,
                width=width,
                num_inference_steps=self.num_inference_steps,
                init_video=video,
                init_audio=audio,
                denoise_strength=self.second_pass_strength,
            ),
            metrics,
        )

    def _upscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Enlarge the finished video latent with the trained upscaler.

        H3 is trained at one megapixel and adds no detail above it on its own,
        which is why the released pipeline reaches 1440p through a second model
        that was never published. Enlarging in latent space rather than out in
        pixels is what keeps speech and lip movement intact.
        """
        from musubi_tuner.minimax_h3.component_loader import load_video_vae_encoder
        from musubi_tuner.minimax_h3.latent_upscaler import load_latent_upscaler

        # The statistics come from this run's VAE rather than a copy kept beside
        # the upscaler, so a checkpoint with different ones cannot disagree.
        statistics = load_video_vae_encoder(self.video_vae, "cpu")
        upscaler = load_latent_upscaler(
            self.latent_upscaler,
            self.device,
            torch.bfloat16,
            latents_mean=statistics.latents_mean,
            latents_std=statistics.latents_std,
        )
        del statistics
        try:
            enlarged = upscaler(latents.to(self.device, torch.bfloat16), scale=self.latent_upscale_scale)
        finally:
            del upscaler
            clean_memory_on_device(self.device)
        return enlarged.float()

    @torch.no_grad()
    def generate(self, request: H3GenerationRequest) -> None:
        metrics: dict[str, dict] = {}
        total_started = time.perf_counter()
        height, width = (
            (self.height, self.width) if self.height is not None else resolve_canvas_size(request.ratio, request.canvas_reference())
        )
        shape = request.temporal_shape
        # Keyframes and references are encoded against a canvas, so a first pass
        # on a smaller one needs its own copies: rows encoded for the full canvas
        # do not fit the smaller layout.
        first_pass_size = (
            (_round_to(height * self.first_pass_scale, CANVAS_MULTIPLE), _round_to(width * self.first_pass_scale, CANVAS_MULTIPLE))
            if self.first_pass_scale
            else None
        )
        first_pass_kwargs: dict = {}
        references = None
        prepared_references = ()
        if request.mode == "reference":
            prepared_references = self._prepare_references(request, height, width)
            prepared_guides = self._prepare_guides(request, height, width)
            images, anchors = self._prepare_keyframes(request, height, width)
            conditioning = self._measure(
                "text_conditioning",
                lambda: self._encode_prompt(request.prompt, references=prepared_references),
                metrics,
            )
            references = (
                self._measure(
                    "reference_encoding",
                    lambda: encode_reference_media(
                        self.video_vae,
                        self.audio_vae,
                        prepared_references,
                        self.device,
                    ),
                    metrics,
                )
                if prepared_references
                else H3EncodedReferences((), torch.empty(0, 96), torch.empty(0, 32))
            )
            guides = (
                self._measure(
                    "guide_encoding",
                    lambda: encode_guide_media(
                        self.video_vae,
                        self.audio_vae,
                        prepared_guides,
                        shape.audio_latent_frames,
                        self.device,
                    ),
                    metrics,
                )
                if prepared_guides
                else None
            )
            keyframe_rows = (
                self._measure(
                    "keyframe_encoding",
                    lambda: torch.cat(encode_keyframe_images(self.video_vae, images, self.device)),
                    metrics,
                )
                if images
                else None
            )
            if first_pass_size is not None:
                small = self._prepare_references(request, *first_pass_size)
                first_pass_kwargs["references"] = (
                    self._measure(
                        "first_pass_reference_encoding",
                        lambda: encode_reference_media(self.video_vae, self.audio_vae, small, self.device),
                        metrics,
                    )
                    if small
                    else H3EncodedReferences((), torch.empty(0, 96), torch.empty(0, 32))
                )
                small_guides = self._prepare_guides(request, *first_pass_size)
                if small_guides:
                    encoded_small_guides = self._measure(
                        "first_pass_guide_encoding",
                        lambda: encode_guide_media(
                            self.video_vae,
                            self.audio_vae,
                            small_guides,
                            shape.audio_latent_frames,
                            self.device,
                        ),
                        metrics,
                    )
                    first_pass_kwargs.update(
                        guide_geometries=encoded_small_guides.geometries,
                        guide_video_rows=encoded_small_guides.video_rows,
                        guide_audio_rows=encoded_small_guides.audio_rows,
                    )
                if images:
                    small_images, small_anchors = self._prepare_keyframes(request, *first_pass_size)
                    first_pass_kwargs["keyframe_rows"] = self._measure(
                        "first_pass_keyframe_encoding",
                        lambda: torch.cat(encode_keyframe_images(self.video_vae, small_images, self.device)),
                        metrics,
                    )
                    first_pass_kwargs["keyframe_anchors"] = small_anchors
                    del small_images
                del small
            reference_kinds = [reference.kind.name.lower() for reference in prepared_references]
            prepared_references = ()
            gc.collect()
        else:
            images, anchors = self._prepare_keyframes(request, height, width)
            conditioning = self._measure(
                "text_conditioning",
                lambda: self._encode_prompt(request.prompt, images, keyframe_anchors=anchors),
                metrics,
            )
            keyframe_rows = (
                self._measure(
                    "keyframe_encoding",
                    lambda: torch.cat(encode_keyframe_images(self.video_vae, images, self.device)),
                    metrics,
                )
                if images
                else None
            )
            if first_pass_size is not None and images:
                small_images, small_anchors = self._prepare_keyframes(request, *first_pass_size)
                first_pass_kwargs["keyframe_rows"] = self._measure(
                    "first_pass_keyframe_encoding",
                    lambda: torch.cat(encode_keyframe_images(self.video_vae, small_images, self.device)),
                    metrics,
                )
                first_pass_kwargs["keyframe_anchors"] = small_anchors
                del small_images
            reference_kinds = []
        loaded_transformer = self._measure("transformer_load", self._load_transformer, metrics)
        transformer, networks = loaded_transformer
        del loaded_transformer
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        base_kwargs = dict(
            height=height,
            width=width,
            frame_count=shape.frame_count,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            device=self.device,
            condition_seed=request.seed,
        )
        if references is not None:
            base_kwargs.update(references=references, keyframe_rows=keyframe_rows, keyframe_anchors=anchors)
            if guides is not None:
                base_kwargs.update(
                    guide_geometries=guides.geometries,
                    guide_video_rows=guides.video_rows,
                    guide_audio_rows=guides.audio_rows,
                )

            # Overrides let a pass reuse this call with another canvas, another
            # step count, its own conditioning, and the latents of the pass before.
            def denoise(**overrides):
                merged = {**base_kwargs, **overrides}
                return denoise_ref2va(transformer, conditioning, merged.pop("references"), **merged)

        else:
            base_kwargs.update(keyframe_rows=keyframe_rows, keyframe_anchors=anchors)

            def denoise(**overrides):
                return denoise_fl2va(transformer, conditioning, **{**base_kwargs, **overrides})

        if self.first_pass_scale:
            video_latents, audio_latents = self._measure(
                "two_pass_denoising",
                # Bound by value: the name is deleted further down, and a lambda
                # holding it by reference would depend on call order to survive.
                lambda nets=networks: self._denoise_two_pass(denoise, nets, height, width, metrics, first_pass_kwargs),
                metrics,
            )
        else:
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

        # After the transformer is gone, not before: it holds tens of gigabytes,
        # and enlarging the latent allocates several more for its activations.
        if self.latent_upscaler is not None and self.latent_upscale_scale > 1.0:
            video_latents = self._measure("latent_upscale", lambda: self._upscale_latents(video_latents), metrics).cpu()

        if request.output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            video_decoder = self._measure("decoder_load", lambda: load_video_vae_decoder(self.video_vae, "cpu"), metrics)
            video_decoder.to(self.device).eval()
            video = self._measure("video_decode", lambda: video_decoder.decode(video_latents.to(self.device)).cpu(), metrics)
            video_decoder.to("cpu")
            clean_memory_on_device(self.device)
            frame_index = min(max(int(getattr(request, "selected_frame", 0)), 0), int(video.shape[2]) - 1)
            frame = video[0, :, frame_index].permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            request.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(request.output)
            return

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
    fp8_quantization_mode: str = "block",
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    convrot_int8_fwd: str = "int8",
    target_device: str | None = None,
    blocks_to_swap: int = 0,
    block_swap_h2d_only: bool = False,
    low_ram_load: bool = True,
    base_lora_weights: list[dict[str, torch.Tensor]] | None = None,
    base_lora_multipliers: list[float] | None = None,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    reference_image_max_pixels: int = 0,
    reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    reference_video_fps: float = REFERENCE_VIDEO_FPS,
    text_visual_max_pixels: int = 0,
    max_caption_tokens: int = 0,
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
        fp8_quantization_mode=fp8_quantization_mode,
        convrot_int8=convrot_int8,
        convrot_int8_bwd=convrot_int8_bwd,
        convrot_int8_fwd=convrot_int8_fwd,
        target_device=target_device,
        blocks_to_swap=blocks_to_swap,
        block_swap_h2d_only=block_swap_h2d_only,
        low_ram_load=low_ram_load,
        base_lora_weights=base_lora_weights,
        base_lora_multipliers=base_lora_multipliers,
    )
    return _NativeTrainingBackend(
        transformer,
        mode,
        reference_image_short_edge,
        reference_image_size_mode,
        reference_image_max_pixels,
        reference_video_short_edge,
        reference_video_max_pixels,
        reference_video_fps,
        text_visual_max_pixels,
        max_caption_tokens,
    )


def _observe_leading_video_rows(
    video_rows: torch.Tensor,
    context_rows: torch.Tensor,
    *,
    condition_rows: int,
    base_timestep: torch.Tensor,
    observed_timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap the leading *target* video rows for clean context and pin their sigma.

    ``condition_rows`` is the number of packed rows that precede the target
    block -- keyframes under FL2VA, references under Ref2VA -- so the observed
    span is addressed relative to the target, never to the packed sequence.
    """
    observed = int(context_rows.shape[1])
    target_rows = int(video_rows.shape[1]) - condition_rows
    rows = torch.cat(
        (video_rows[:, :condition_rows], context_rows, video_rows[:, condition_rows + observed :]),
        dim=1,
    )
    schedule = base_timestep.reshape(1).to(dtype=torch.float32).expand(target_rows).clone()
    schedule[:observed] = observed_timestep.reshape(1)[0]
    return rows, schedule


def _observe_leading_audio_rows(
    audio_rows: torch.Tensor,
    context_rows: torch.Tensor,
    *,
    condition_rows: int,
    num_audio_latents: int,
    context_latents: int,
    base_timestep: torch.Tensor,
    observed_timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap each channel's leading target audio latents for clean context.

    Packed audio is channel-major, so the observed prefix appears once per
    channel and cannot be expressed as one contiguous slice.
    """
    kept = [audio_rows[:, :condition_rows]]
    for channel in range(AUDIO_CHANNELS):
        start = condition_rows + channel * num_audio_latents
        kept.append(context_rows[:, channel * context_latents : (channel + 1) * context_latents])
        kept.append(audio_rows[:, start + context_latents : start + num_audio_latents])
    rows = torch.cat(kept, dim=1)
    schedule = base_timestep.reshape(1).to(dtype=torch.float32).expand(AUDIO_CHANNELS * num_audio_latents).clone()
    for channel in range(AUDIO_CHANNELS):
        start = channel * num_audio_latents
        schedule[start : start + context_latents] = observed_timestep.reshape(1)[0]
    return rows, schedule


def _pin_observed_rows(
    rows: torch.Tensor,
    clean_rows: torch.Tensor,
    observed: torch.Tensor,
    *,
    condition_rows: int,
    base_timestep: torch.Tensor,
    observed_timestep: torch.Tensor,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Present an arbitrary subset of the target rows as clean, pinned context.

    The mask is drawn at target-latent granularity, so it indexes the target
    block alone. Conditioning rows -- keyframes or references -- keep their own
    timestep and are never scored, which is why the mask must be offset past
    them rather than compared against the whole packed sequence.
    """
    target = rows[:, condition_rows:]
    observed = observed.to(rows.device)
    if observed.shape != (target.shape[1],):
        raise ValueError(f"H3 observed {modality} mask has {tuple(observed.shape)} rows for {target.shape[1]} target rows")
    target = torch.where(observed[None, :, None], clean_rows, target)
    rows = torch.cat((rows[:, :condition_rows], target), dim=1)
    schedule = base_timestep.reshape(1).to(dtype=torch.float32).expand(target.shape[1]).clone()
    schedule[observed] = observed_timestep.reshape(1)[0]
    return rows, schedule


class _NativeTrainingBackend:
    supports_paired_conditioning = True

    def __init__(
        self,
        transformer: torch.nn.Module,
        mode: H3TrainingMode = "fl2va",
        reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
        reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
        reference_image_max_pixels: int = 0,
        reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
        reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
        reference_video_fps: float = REFERENCE_VIDEO_FPS,
        text_visual_max_pixels: int = 0,
        max_caption_tokens: int = 0,
    ):
        self.transformer = transformer
        self.mode = mode
        self.reference_image_short_edge = reference_image_short_edge
        self.max_caption_tokens = max_caption_tokens
        self.reference_image_size_mode = reference_image_size_mode
        self.reference_image_max_pixels = reference_image_max_pixels
        self.reference_video_short_edge = reference_video_short_edge
        self.reference_video_max_pixels = reference_video_max_pixels
        self.reference_video_fps = reference_video_fps
        self.text_visual_max_pixels = text_visual_max_pixels
        # One warning per run: keyframe anchors and cached keyframe visuals are
        # independent, so a divergence is reported once and never repeated.
        self._keyframe_visual_mismatch_warned = False

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
        conditioning: Literal["prompt", "empty"] | tuple[Literal["prompt", "empty"], ...] = "prompt",
        reference_modality: Literal["av", "video", "audio"] = "av",
        qwen_control_dropout: bool = False,
        extension_video_frames: int = 0,
        extension_audio_latents: int = 0,
        condition_video_anchors: tuple[str | int, ...] = (),
        guide_geometries: tuple[MiniMaxH3GuideGeometry, ...] = (),
        guide_video_latents: tuple[torch.Tensor, ...] = (),
        guide_audio_latents: tuple[torch.Tensor, ...] = (),
        extension_video_context: torch.Tensor | None = None,
        extension_audio_context: torch.Tensor | None = None,
        extension_route: Literal["condition_rows", "per_row_sigma"] = "condition_rows",
        video_row_schedule: torch.Tensor | None = None,
        spatial_density_scale: float = 1.0,
        observed_video_rows: torch.Tensor | None = None,
        observed_audio_rows: torch.Tensor | None = None,
        clean_video_latents: torch.Tensor | None = None,
        clean_audio_latents: torch.Tensor | None = None,
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

        conditionings = (conditioning,) if isinstance(conditioning, str) else tuple(conditioning)
        if not conditionings or any(value not in ("prompt", "empty") for value in conditionings):
            raise ValueError("H3 conditioning must contain only 'prompt' or 'empty'")
        if len(conditionings) > 2:
            raise ValueError("H3 paired conditioning supports at most two presentations")

        text_presentations: list[tuple[torch.Tensor, torch.Tensor, str, str]] = []
        for presentation in conditionings:
            hidden_key, tags_key = (
                (H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY)
                if presentation == "prompt"
                else (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY)
            )
            if reference_modality != "av":
                if self.mode not in ("ref2va", "ref2va_omni"):
                    raise ValueError("reference modality selection is only valid for Ref2VA training")
                hidden_key = reference_variant_key(hidden_key, reference_modality)
                tags_key = reference_variant_key(tags_key, reference_modality)
            if qwen_control_dropout:
                # The control-free twin of whichever presentation the branches above
                # selected: dropout composes with the empty branch and with every
                # reference-modality variant rather than replacing them.
                hidden_key = qwen_control_dropout_key(hidden_key)
                tags_key = qwen_control_dropout_key(tags_key)
                missing = [key for key in (hidden_key, tags_key) if key not in batch]
                if missing:
                    raise KeyError(
                        "--h3_qwen_control_dropout_rate requires a text cache written with --h3_qwen_control_dropout; missing "
                        + ", ".join(missing)
                    )
            text_presentations.append(
                (
                    self._one_conditioning_item(batch, hidden_key, expected_ndim=2),
                    self._one_conditioning_item(batch, tags_key, expected_ndim=1),
                    hidden_key,
                    tags_key,
                )
            )

        text_hidden, text_tags, hidden_key, tags_key = text_presentations[0]
        if len(text_presentations) > 1:
            for paired_hidden, paired_tags, paired_hidden_key, paired_tags_key in text_presentations[1:]:
                if paired_hidden.shape != text_hidden.shape or not torch.equal(paired_tags, text_tags):
                    raise H3PairedConditioningUnsupportedError(
                        "H3 prompt and empty conditioning do not share one packed text layout; using sequential teachers"
                    )
                if paired_hidden.dtype != text_hidden.dtype:
                    raise H3PairedConditioningUnsupportedError(
                        f"H3 {hidden_key} and {paired_hidden_key} use different dtypes; using sequential teachers"
                    )
                if paired_tags.dtype != text_tags.dtype:
                    raise H3PairedConditioningUnsupportedError(
                        f"H3 {tags_key} and {paired_tags_key} use different dtypes; using sequential teachers"
                    )
            text_hidden = torch.stack([value[0] for value in text_presentations])
        else:
            text_hidden = text_hidden[None]
        conditioning_task = self._one_conditioning_item(batch, H3_CONDITIONING_TASK_KEY, expected_ndim=0)
        cached_caption_cap = batch.get(H3_MAX_CAPTION_TOKENS_KEY)
        if cached_caption_cap is None:
            if self.max_caption_tokens:
                raise ValueError("legacy H3 text cache lacks caption-token cap identity; re-cache conditioning")
        else:
            cached_caption_cap = self._one_conditioning_item(batch, H3_MAX_CAPTION_TOKENS_KEY, expected_ndim=0)
            if cached_caption_cap.dtype != torch.long or int(cached_caption_cap) != self.max_caption_tokens:
                raise ValueError("H3 text cache uses a different h3_max_caption_tokens value; re-cache conditioning")
        cached_text_visual_max_pixels = batch.get(H3_TEXT_VISUAL_MAX_PIXELS_KEY)
        if cached_text_visual_max_pixels is None:
            if self.text_visual_max_pixels:
                raise ValueError("legacy H3 text cache lacks Qwen visual-pixel identity; re-cache conditioning")
        else:
            cached_text_visual_max_pixels = self._one_conditioning_item(batch, H3_TEXT_VISUAL_MAX_PIXELS_KEY, expected_ndim=0)
            if (
                cached_text_visual_max_pixels.dtype != torch.long
                or int(cached_text_visual_max_pixels) != self.text_visual_max_pixels
            ):
                raise ValueError("H3 text cache uses a different h3_text_visual_max_pixels value; re-cache conditioning")
        if self.mode in ("ref2va", "ref2va_omni"):
            cached_reference_size = batch.get(H3_REFERENCE_IMAGE_SHORT_EDGE_KEY)
            if cached_reference_size is None:
                if self.reference_image_short_edge != REFERENCE_IMAGE_SHORT_EDGE:
                    raise ValueError(
                        "Legacy H3 Ref2VA text cache has no reference-image size identity; "
                        "re-cache text conditioning for a non-default reference_image_short_edge"
                    )
            else:
                cached_reference_size = self._one_conditioning_item(batch, H3_REFERENCE_IMAGE_SHORT_EDGE_KEY, expected_ndim=0)
                if cached_reference_size.dtype != torch.long or int(cached_reference_size) != self.reference_image_short_edge:
                    raise ValueError(
                        f"H3 Ref2VA text cache uses reference short edge {int(cached_reference_size)}, "
                        f"but training requested {self.reference_image_short_edge}; re-cache text conditioning"
                    )
            cached_size_mode = batch.get(H3_REFERENCE_IMAGE_SIZE_MODE_KEY)
            cached_max_pixels = batch.get(H3_REFERENCE_IMAGE_MAX_PIXELS_KEY)
            expected_size_mode = 0 if self.reference_image_size_mode == "short_edge" else 1
            if cached_size_mode is None or cached_max_pixels is None:
                if self.reference_image_size_mode != REFERENCE_IMAGE_SIZE_MODE or self.reference_image_max_pixels:
                    raise ValueError("legacy H3 Ref2VA text cache lacks reference sizing identity; re-cache conditioning")
            else:
                cached_size_mode = self._one_conditioning_item(batch, H3_REFERENCE_IMAGE_SIZE_MODE_KEY, expected_ndim=0)
                cached_max_pixels = self._one_conditioning_item(batch, H3_REFERENCE_IMAGE_MAX_PIXELS_KEY, expected_ndim=0)
                if int(cached_size_mode) != expected_size_mode or int(cached_max_pixels) != self.reference_image_max_pixels:
                    raise ValueError("H3 Ref2VA text cache uses a different reference sizing strategy; re-cache conditioning")
            cached_video_short_edge = batch.get(H3_REFERENCE_VIDEO_SHORT_EDGE_KEY)
            cached_video_max_pixels = batch.get(H3_REFERENCE_VIDEO_MAX_PIXELS_KEY)
            if cached_video_short_edge is None or cached_video_max_pixels is None:
                if (
                    self.reference_video_short_edge != REFERENCE_VIDEO_SHORT_EDGE
                    or self.reference_video_max_pixels != REFERENCE_VIDEO_MAX_PIXELS
                ):
                    raise ValueError("legacy H3 Ref2VA text cache lacks reference-video sizing identity; re-cache conditioning")
            else:
                cached_video_short_edge = self._one_conditioning_item(batch, H3_REFERENCE_VIDEO_SHORT_EDGE_KEY, expected_ndim=0)
                cached_video_max_pixels = self._one_conditioning_item(batch, H3_REFERENCE_VIDEO_MAX_PIXELS_KEY, expected_ndim=0)
                if (
                    int(cached_video_short_edge) != self.reference_video_short_edge
                    or int(cached_video_max_pixels) != self.reference_video_max_pixels
                ):
                    raise ValueError("H3 Ref2VA text cache uses different reference-video sizing; re-cache conditioning")
            cached_video_fps = batch.get(H3_REFERENCE_VIDEO_FPS_KEY)
            if cached_video_fps is None:
                if self.reference_video_fps != REFERENCE_VIDEO_FPS:
                    raise ValueError("legacy H3 Ref2VA text cache lacks reference-video fps identity; re-cache conditioning")
            else:
                cached_video_fps = self._one_conditioning_item(batch, H3_REFERENCE_VIDEO_FPS_KEY, expected_ndim=0)
                if float(cached_video_fps) != float(self.reference_video_fps):
                    raise ValueError(
                        f"H3 Ref2VA text cache uses reference_video_fps={float(cached_video_fps)}, "
                        f"but training requested {float(self.reference_video_fps)}; re-cache conditioning"
                    )
        if text_hidden.ndim != 3 or text_hidden.shape[-1] != config.text_dim:
            raise ValueError(f"H3 {hidden_key} must have shape [batch, tokens, {config.text_dim}]")
        if text_tags.dtype != torch.long or text_tags.shape != (text_hidden.shape[1],):
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
            accepted_tasks = {"t2va", "i2va", "fl2va", "l2va"}
        if task not in accepted_tasks:
            expected = ", ".join(f"--task {name}" for name in sorted(accepted_tasks))
            raise ValueError(f"MiniMax H3 {self.mode} training requires {expected} conditioning; re-cache text outputs")
        if condition_video_anchors and task not in ("t2va", "ref2va", "ref2va_omni"):
            raise ValueError("H3 custom keyframe anchors require T2VA or Ref2VA conditioning caches")
        if task in ("ref2va", "ref2va_omni"):
            # Masked and per-row-sigma conditioning only pin rows inside the
            # target block, which the Ref2VA layout carries unchanged behind its
            # reference prefix. Arbitrary keyframe guides have their own
            # reference-aware rows below; generic condition-row extension does not.
            if (extension_video_frames or extension_audio_latents) and extension_route != "per_row_sigma":
                raise ValueError(
                    "H3 Ref2VA extension is only supported on the per_row_sigma route; "
                    "pass --h3_extension_route per_row_sigma, or use FL2VA training with --task t2va caches"
                )
        else:
            t2va_only_conditioning = {
                "observed video rows": observed_video_rows,
                "observed audio rows": observed_audio_rows,
                "clean video latents": clean_video_latents,
                "clean audio latents": clean_audio_latents,
                "video extension context": extension_video_context,
                "audio extension context": extension_audio_context,
            }
            requested_t2va_only = [name for name, value in t2va_only_conditioning.items() if value is not None]
            if task != "t2va" and requested_t2va_only:
                raise ValueError(
                    f"H3 {task.upper()} caches cannot represent {', '.join(requested_t2va_only)}; "
                    "disable that conditioning option, or use FL2VA training with --task t2va caches"
                )
        has_vision = bool((text_tags == int(MiniMaxH3TokenTag.VIDEO)).any())
        # EXPERIMENTAL Qwen control visuals produce ordinary VIDEO-tagged text
        # rows, indistinguishable from keyframe or reference spans by tag alone.
        # The cache therefore carries an explicit marker, and it is the only thing
        # that tells a T2VA presentation with control spans apart from a stale
        # keyframe cache. No DiT rows are involved either way.
        has_qwen_controls = H3_QWEN_CONTROL_VISUALS_KEY in batch and int(
            self._one_conditioning_item(batch, H3_QWEN_CONTROL_VISUALS_KEY, expected_ndim=0)
        )
        # EXPERIMENTAL keyframe visuals are the second marked population: target
        # frames shown to Qwen so custom anchors are visible to the conditioner.
        # They too produce VIDEO-tagged text rows and no DiT rows, so the T2VA
        # acceptance check ORs the two markers.
        cached_keyframe_visuals = ()
        if H3_KEYFRAME_VISUALS_KEY in batch:
            cached_keyframe_visuals = tuple(
                int(value) for value in self._one_conditioning_item(batch, H3_KEYFRAME_VISUALS_KEY, expected_ndim=1)
            )
        if task == "t2va" and has_vision and not (has_qwen_controls or cached_keyframe_visuals):
            raise ValueError("MiniMax H3 T2VA training requires text-only conditioning; re-cache with --task t2va")
        if condition_video_anchors and cached_keyframe_visuals and not self._keyframe_visual_mismatch_warned:
            # The pinned rows and the presented frames are independently valid --
            # one lives in latent space, the other in decoded pixel frames -- so a
            # divergence is worth naming once and never worth failing on.
            pinned = tuple(
                0 if anchor == "first" else H3_KEYFRAME_VISUAL_LAST if anchor == "last" else int(anchor)
                for anchor in condition_video_anchors
            )
            if set(pinned) != set(cached_keyframe_visuals):
                self._keyframe_visual_mismatch_warned = True
                logger.warning(
                    "H3 keyframe anchors pin %s while the text cache presents %s to the conditioner; "
                    "re-cache with --h3_keyframe_visuals %s to show the anchored frames",
                    format_keyframe_visuals(pinned),
                    format_keyframe_visuals(cached_keyframe_visuals),
                    format_keyframe_visuals(pinned),
                )
        if task in ("i2va", "fl2va", "l2va") and not has_vision:
            raise ValueError(f"MiniMax H3 {task.upper()} training requires keyframe vision rows; re-cache with --task {task}")
        if task == "ref2va" and not has_vision:
            # An audio-only reference set is presented to Qwen as text alone --
            # reference audio never reaches the vision tower -- so a text-only
            # presentation is legitimate exactly when the cached reference bundle
            # is kind=2 throughout. Anything else is a stale or mismatched cache.
            cached_kinds = self._cached_reference_kinds(batch)
            if cached_kinds is None or bool((cached_kinds != int(H3ReferenceKind.AUDIO)).any()):
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
        if self.mode in ("ref2va", "ref2va_omni"):
            references, reference_video, reference_audio = self._reference_cache(
                batch,
                patch_size=patch_size,
                video_width=video_rows.shape[-1],
                audio_width=audio_rows.shape[-1],
                device=model_device,
                dtype=video_rows.dtype,
                reference_modality=reference_modality,
            )
            layout = build_ref2va_packed_sequence(
                text_tags,
                references,
                num_latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
                keyframe_anchors=condition_video_anchors,
                guides=guide_geometries,
                spatial_density_scale=spatial_density_scale,
            )
            condition_video_timestep = torch.maximum(
                video_timestep.reshape(1).to(model_device, torch.float32),
                torch.tensor([0.999], device=model_device),
            )
            if reference_video.numel():
                reference_video = 0.999 * reference_video + 0.001 * torch.randn_like(reference_video)
                video_rows = torch.cat((reference_video[None], video_rows), dim=1)
            if condition_video_anchors:
                if extension_video_context is None:
                    raise ValueError("H3 keyframe conditioning requires the clean guide latents")
                guide_video = patchify_video_latents(extension_video_context, patch_size)
                expected_guide_rows = layout.num_condition_video_rows - int(reference_video.shape[0])
                if guide_video.shape[1] != expected_guide_rows:
                    raise ValueError(
                        f"H3 keyframe conditioning produced {guide_video.shape[1]} rows, expected {expected_guide_rows}"
                    )
                guide_video = 0.999 * guide_video + 0.001 * torch.randn_like(guide_video)
                reference_rows = int(reference_video.shape[0])
                video_rows = torch.cat((video_rows[:, :reference_rows], guide_video, video_rows[:, reference_rows:]), dim=1)
            explicit_video = [patchify_video_latents(value, patch_size) for value in guide_video_latents]
            explicit_audio = [pack_audio_latents(value) for value in guide_audio_latents]
            expected_video_streams = sum(geometry.num_video_latents > 0 for geometry in guide_geometries)
            expected_audio_streams = sum(geometry.num_audio_latents > 0 for geometry in guide_geometries)
            if len(explicit_video) != expected_video_streams or len(explicit_audio) != expected_audio_streams:
                raise ValueError("H3 guide geometry and supplied guide streams do not match")
            if explicit_video:
                guide_video = torch.cat(explicit_video, dim=1)
                rows_per_frame = (latent_height // patch_size[1]) * (latent_width // patch_size[2])
                expected_rows = sum(geometry.num_video_rows(rows_per_frame) for geometry in guide_geometries)
                if guide_video.shape[1] != expected_rows:
                    raise ValueError(f"H3 guide video produced {guide_video.shape[1]} rows, expected {expected_rows}")
                guide_video = 0.999 * guide_video + 0.001 * torch.randn_like(guide_video)
                reference_rows = int(reference_video.shape[0])
                legacy_rows = layout.num_condition_video_rows - reference_rows - expected_rows
                insert_at = reference_rows + legacy_rows
                video_rows = torch.cat((video_rows[:, :insert_at], guide_video, video_rows[:, insert_at:]), dim=1)
            guide_audio_rows = torch.cat(explicit_audio, dim=1) if explicit_audio else None
            if guide_audio_rows is not None:
                expected_rows = sum(geometry.num_audio_rows for geometry in guide_geometries)
                if guide_audio_rows.shape[1] != expected_rows:
                    raise ValueError(f"H3 guide audio produced {guide_audio_rows.shape[1]} rows, expected {expected_rows}")
            audio_parts = []
            if reference_audio.numel():
                audio_parts.append(reference_audio[None])
            if guide_audio_rows is not None:
                audio_parts.append(guide_audio_rows)
            audio_parts.append(audio_rows)
            audio_rows = torch.cat(audio_parts, dim=1)
            # Every offset below is read from this step's layout: a per-step
            # reference-modality redraw changes how many conditioning rows the
            # sequence carries, and a zero-reference omni sample carries none at
            # all, in which case the target block starts at row zero exactly as
            # it does under T2VA.
            condition_video_rows = layout.num_condition_video_rows
            condition_audio_rows = layout.num_condition_audio_rows
            observed_video_timestep = condition_video_timestep
            observed_audio_timestep = torch.ones(1, device=model_device)
            row_video_timestep = video_timestep
            row_audio_timestep = audio_timestep
            per_row = False

            if extension_video_frames:
                if extension_video_frames >= latent_frames:
                    raise ValueError(
                        f"H3 video extension needs a shorter context than the target: "
                        f"{extension_video_frames} of {latent_frames} latent frames"
                    )
                if extension_video_context is None:
                    raise ValueError("H3 conditioning anchors require the clean context latents")
                context_video_rows = patchify_video_latents(extension_video_context, patch_size)
                context_video_rows = 0.999 * context_video_rows + 0.001 * torch.randn_like(context_video_rows)
                video_rows, row_video_timestep = _observe_leading_video_rows(
                    video_rows,
                    context_video_rows,
                    condition_rows=condition_video_rows,
                    base_timestep=video_timestep,
                    observed_timestep=observed_video_timestep,
                )
                per_row = True
            if extension_audio_latents:
                if extension_audio_latents >= num_audio_latents:
                    raise ValueError(
                        f"H3 audio extension needs a shorter context than the target: "
                        f"{extension_audio_latents} of {num_audio_latents} audio latents"
                    )
                if extension_audio_context is None:
                    raise ValueError("H3 audio extension requires the clean context latents")
                audio_rows, row_audio_timestep = _observe_leading_audio_rows(
                    audio_rows,
                    pack_audio_latents(extension_audio_context),
                    condition_rows=condition_audio_rows,
                    num_audio_latents=num_audio_latents,
                    context_latents=extension_audio_latents,
                    base_timestep=audio_timestep,
                    observed_timestep=observed_audio_timestep,
                )
                per_row = True

            if video_row_schedule is not None:
                target_rows = video_rows.shape[1] - condition_video_rows
                if video_row_schedule.numel() != target_rows:
                    raise ValueError(
                        f"H3 per-row video timesteps have {video_row_schedule.numel()} entries for {target_rows} target rows"
                    )
                row_video_timestep = video_row_schedule.to(device=model_device, dtype=torch.float32)
                per_row = True

            if observed_video_rows is not None:
                if clean_video_latents is None:
                    raise ValueError("H3 masked conditioning requires the clean video latents")
                clean_rows = patchify_video_latents(clean_video_latents, patch_size)
                clean_rows = 0.999 * clean_rows + 0.001 * torch.randn_like(clean_rows)
                video_rows, row_video_timestep = _pin_observed_rows(
                    video_rows,
                    clean_rows,
                    observed_video_rows,
                    condition_rows=condition_video_rows,
                    base_timestep=video_timestep,
                    observed_timestep=observed_video_timestep,
                    modality="video",
                )
                per_row = True
            if observed_audio_rows is not None:
                if clean_audio_latents is None:
                    raise ValueError("H3 masked conditioning requires the clean audio latents")
                audio_rows, row_audio_timestep = _pin_observed_rows(
                    audio_rows,
                    pack_audio_latents(clean_audio_latents),
                    observed_audio_rows,
                    condition_rows=condition_audio_rows,
                    base_timestep=audio_timestep,
                    observed_timestep=observed_audio_timestep,
                    modality="audio",
                )
                per_row = True

            timestep, timestep_indices = build_row_timesteps(
                layout,
                row_video_timestep,
                row_audio_timestep,
                condition_video_timestep,
                torch.ones(1, device=model_device),
                per_row_timesteps=per_row,
            )
        elif task in ("i2va", "fl2va", "l2va"):
            anchors = {"i2va": ("first",), "fl2va": ("first", "last"), "l2va": ("last",)}[task]
            layout = build_t2va_packed_sequence(
                text_tags,
                num_latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
                keyframe_anchors=anchors,
                spatial_density_scale=spatial_density_scale,
            )
            keyframe_rows = self._keyframe_cache(
                batch,
                anchors=anchors,
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
            row_video_timestep = video_timestep
            if video_row_schedule is not None:
                target_rows = video_rows.shape[1] - layout.num_condition_video_rows
                if video_row_schedule.numel() != target_rows:
                    raise ValueError(
                        f"H3 per-row video timesteps have {video_row_schedule.numel()} entries for {target_rows} target rows"
                    )
                row_video_timestep = video_row_schedule.to(device=model_device, dtype=torch.float32)
            timestep, timestep_indices = build_row_timesteps(
                layout,
                row_video_timestep,
                audio_timestep,
                condition_video_timestep,
                per_row_timesteps=video_row_schedule is not None,
            )
        else:
            # Extension observes a leading run of the target. The observed rows
            # must be the *clean* latents: the packed target rows are already
            # noised, so slicing them would present noise as context. The caller
            # supplies the clean span separately.
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
            # Extension observes a leading run; keyframe conditioning observes an
            # arbitrary set. Both present the same clean rows at the coordinates
            # of the frames they duplicate, so they share one path.
            anchors = condition_video_anchors or tuple(range(extension_video_frames))
            if anchors and extension_video_context is None:
                raise ValueError("H3 conditioning anchors require the clean context latents")
            # Anchors may be the released "first"/"last" tokens as well as latent
            # frame indices, so only the indices carry a range to check; the
            # packer validates the tokens themselves.
            if any(isinstance(anchor, int) and not 0 <= anchor < latent_frames for anchor in anchors):
                raise ValueError(f"H3 conditioning anchors must lie inside the {latent_frames} target latent frames")
            if len(set(anchors)) != len(anchors):
                raise ValueError("H3 conditioning anchors must be unique")
            if extension_audio_latents and extension_audio_context is None:
                raise ValueError("H3 audio extension requires the clean context latents")

            # condition_rows duplicates the observed span as extra clean rows,
            # generalizing the released keyframe contract. per_row_sigma instead
            # pins the observed rows inside the target block, which costs no
            # extra tokens but is not something the released weights have seen.
            duplicate_context = extension_route == "condition_rows" or bool(condition_video_anchors)
            context_video_rows = None
            context_audio_rows = None
            if anchors:
                context_video_rows = patchify_video_latents(extension_video_context, patch_size)
                context_video_rows = 0.999 * context_video_rows + 0.001 * torch.randn_like(context_video_rows)
            if extension_audio_latents:
                context_audio_rows = pack_audio_latents(extension_audio_context)

            layout = build_t2va_packed_sequence(
                text_tags,
                num_latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
                keyframe_anchors=anchors if duplicate_context else (),
                num_condition_audio_latents=extension_audio_latents if duplicate_context else 0,
                spatial_density_scale=spatial_density_scale,
            )
            condition_video_timestep = None
            condition_audio_timestep = None
            row_video_timestep = video_timestep
            row_audio_timestep = audio_timestep
            per_row = False
            observed_video_timestep = torch.maximum(
                video_timestep.reshape(1).to(model_device, torch.float32),
                torch.tensor([0.999], device=model_device),
            )
            observed_audio_timestep = torch.ones(1, device=model_device)

            if duplicate_context:
                if context_video_rows is not None:
                    video_rows = torch.cat((context_video_rows, video_rows), dim=1)
                    condition_video_timestep = observed_video_timestep
                if context_audio_rows is not None:
                    audio_rows = torch.cat((context_audio_rows, audio_rows), dim=1)
                    condition_audio_timestep = observed_audio_timestep
            else:
                if context_video_rows is not None:
                    video_rows, row_video_timestep = _observe_leading_video_rows(
                        video_rows,
                        context_video_rows,
                        condition_rows=layout.num_condition_video_rows,
                        base_timestep=video_timestep,
                        observed_timestep=observed_video_timestep,
                    )
                    per_row = True
                if context_audio_rows is not None:
                    audio_rows, row_audio_timestep = _observe_leading_audio_rows(
                        audio_rows,
                        context_audio_rows,
                        condition_rows=layout.num_condition_audio_rows,
                        num_audio_latents=num_audio_latents,
                        context_latents=extension_audio_latents,
                        base_timestep=audio_timestep,
                        observed_timestep=observed_audio_timestep,
                    )
                    per_row = True

            # A caller-supplied schedule gives every target video row its own
            # noise level, which the transformer already supports because it
            # selects modulation through timestep_indices. It covers the target
            # block alone, matching what build_row_timesteps assigns.
            if video_row_schedule is not None:
                target_rows = video_rows.shape[1] - layout.num_condition_video_rows
                if video_row_schedule.numel() != target_rows:
                    raise ValueError(
                        f"H3 per-row video timesteps have {video_row_schedule.numel()} entries for {target_rows} target rows"
                    )
                row_video_timestep = video_row_schedule.to(device=model_device, dtype=torch.float32)
                per_row = True

            # A conditioning mask observes an arbitrary subset rather than a
            # leading run, so it can only be expressed by pinning rows in place.
            if observed_video_rows is not None:
                if clean_video_latents is None:
                    raise ValueError("H3 masked conditioning requires the clean video latents")
                clean_rows = patchify_video_latents(clean_video_latents, patch_size)
                clean_rows = 0.999 * clean_rows + 0.001 * torch.randn_like(clean_rows)
                video_rows, row_video_timestep = _pin_observed_rows(
                    video_rows,
                    clean_rows,
                    observed_video_rows,
                    condition_rows=layout.num_condition_video_rows,
                    base_timestep=video_timestep,
                    observed_timestep=observed_video_timestep,
                    modality="video",
                )
                per_row = True
            if observed_audio_rows is not None:
                if clean_audio_latents is None:
                    raise ValueError("H3 masked conditioning requires the clean audio latents")
                audio_rows, row_audio_timestep = _pin_observed_rows(
                    audio_rows,
                    pack_audio_latents(clean_audio_latents),
                    observed_audio_rows,
                    condition_rows=layout.num_condition_audio_rows,
                    base_timestep=audio_timestep,
                    observed_timestep=observed_audio_timestep,
                    modality="audio",
                )
                per_row = True

            timestep, timestep_indices = build_row_timesteps(
                layout,
                row_video_timestep,
                row_audio_timestep,
                condition_video_timestep,
                condition_audio_timestep,
                per_row_timesteps=per_row,
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
        presentation_batch = text_hidden.shape[0]
        if presentation_batch > 1:
            # Both teachers see exactly the same stochastic reference/context
            # rows. Expanding after those rows are constructed preserves the
            # sequential path's fork_rng replay instead of drawing new noise for
            # the second presentation.
            video_rows = video_rows.expand(presentation_batch, -1, -1)
            audio_rows = audio_rows.expand(presentation_batch, -1, -1)
        output = transformer(
            video_hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_hidden.to(model_device),
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
        anchors: tuple[str, ...],
        rows_per_anchor: int,
        row_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = self._one_conditioning_item(batch, H3_KEYFRAME_VIDEO_ROWS_KEY, expected_ndim=2)
        expected_all_rows = 2 * rows_per_anchor
        if rows.shape != (expected_all_rows, row_width):
            raise ValueError(f"H3 keyframe cache has shape {tuple(rows.shape)}, expected {(expected_all_rows, row_width)}")
        chunks = rows.split(rows_per_anchor)
        selected = torch.cat(tuple(chunks[0 if anchor == "first" else 1] for anchor in anchors))
        return selected.to(device=device, dtype=dtype)

    def _cached_reference_kinds(self, batch: dict) -> torch.Tensor | None:
        """Peek at the cached reference kinds without unpacking the whole bundle."""
        suffix = reference_key_suffix(
            self.reference_image_short_edge,
            self.reference_image_size_mode,
            self.reference_image_max_pixels,
            self.reference_video_short_edge,
            self.reference_video_max_pixels,
            self.reference_video_fps,
        )
        kinds_key = f"{H3_REFERENCE_KINDS_KEY}{suffix}"
        if kinds_key not in batch:
            return None
        kinds = self._one_conditioning_item(batch, kinds_key, expected_ndim=1).to(torch.long)
        return kinds if kinds.numel() else None

    def _reference_cache(
        self,
        batch: dict,
        *,
        patch_size: tuple[int, int, int],
        video_width: int,
        audio_width: int,
        device: torch.device,
        dtype: torch.dtype,
        reference_modality: Literal["av", "video", "audio"] = "av",
    ) -> tuple[tuple[MiniMaxH3ReferenceGeometry, ...], torch.Tensor, torch.Tensor]:
        suffix = reference_key_suffix(
            self.reference_image_short_edge,
            self.reference_image_size_mode,
            self.reference_image_max_pixels,
            self.reference_video_short_edge,
            self.reference_video_max_pixels,
            self.reference_video_fps,
        )
        kinds_key = f"{H3_REFERENCE_KINDS_KEY}{suffix}"
        video_shapes_key = f"{H3_REFERENCE_VIDEO_SHAPES_KEY}{suffix}"
        audio_lengths_key = f"{H3_REFERENCE_AUDIO_LENGTHS_KEY}{suffix}"
        video_rows_key = f"{H3_REFERENCE_VIDEO_ROWS_KEY}{suffix}"
        audio_rows_key = f"{H3_REFERENCE_AUDIO_ROWS_KEY}{suffix}"
        reference_keys = {kinds_key, video_shapes_key, audio_lengths_key, video_rows_key, audio_rows_key}
        present_keys = reference_keys.intersection(batch)
        if not present_keys:
            if self.mode != "ref2va_omni":
                raise KeyError(f"MiniMax H3 Ref2VA training cache is missing {kinds_key}")
            return (
                (),
                torch.empty((0, video_width), device=device, dtype=dtype),
                torch.empty((0, audio_width), device=device, dtype=dtype),
            )
        if present_keys != reference_keys:
            missing = ", ".join(sorted(reference_keys - present_keys))
            raise KeyError(f"H3 Ref2VA cache has a partial reference bundle; missing {missing}")
        kinds = self._one_conditioning_item(batch, kinds_key, expected_ndim=1).to(torch.long)
        video_shapes = self._one_conditioning_item(batch, video_shapes_key, expected_ndim=2).to(torch.long)
        audio_lengths = self._one_conditioning_item(batch, audio_lengths_key, expected_ndim=1).to(torch.long)
        video_rows = self._one_conditioning_item(batch, video_rows_key, expected_ndim=2)
        audio_rows = self._one_conditioning_item(batch, audio_rows_key, expected_ndim=2)
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
        if reference_modality != "av":
            selected_references = []
            selected_video_rows = []
            selected_audio_rows = []
            video_offset = audio_offset = 0
            for reference in references:
                video_count = reference.num_video_rows(patch_size)
                audio_count = reference.num_audio_rows
                video_chunk = video_rows[video_offset : video_offset + video_count]
                audio_chunk = audio_rows[audio_offset : audio_offset + audio_count]
                video_offset += video_count
                audio_offset += audio_count
                if reference_modality == "video":
                    if reference.kind != int(H3ReferenceKind.AUDIO):
                        selected_references.append(
                            MiniMaxH3ReferenceGeometry(
                                kind=reference.kind,
                                num_latent_frames=reference.num_latent_frames,
                                latent_height=reference.latent_height,
                                latent_width=reference.latent_width,
                                num_audio_latents=0,
                            )
                        )
                        selected_video_rows.append(video_chunk)
                elif reference.kind == int(H3ReferenceKind.IMAGE):
                    selected_references.append(reference)
                    selected_video_rows.append(video_chunk)
                elif audio_count:
                    selected_references.append(
                        MiniMaxH3ReferenceGeometry(
                            kind=int(H3ReferenceKind.AUDIO),
                            num_latent_frames=0,
                            latent_height=0,
                            latent_width=0,
                            num_audio_latents=reference.num_audio_latents,
                        )
                    )
                    selected_audio_rows.append(audio_chunk)
            references = tuple(selected_references)
            # Mirrors ``reference_modality_variant``: an audio-only survivor set
            # is legal, an empty one is not -- it would silently degrade the
            # sample to T2VA while the cached variant text still says Ref2VA.
            if not references:
                raise ValueError(f"H3 stochastic {reference_modality}-reference variant keeps no reference")
            video_rows = torch.cat(selected_video_rows) if selected_video_rows else video_rows.new_empty((0, video_width))
            audio_rows = torch.cat(selected_audio_rows) if selected_audio_rows else audio_rows.new_empty((0, audio_width))
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
        reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
        reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
        reference_image_max_pixels: int = 0,
        reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
        reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
        reference_video_fps: float = REFERENCE_VIDEO_FPS,
    ) -> None:
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.output_dtype = output_dtype
        self.reference_image_short_edge = reference_image_short_edge
        self.reference_image_size_mode = reference_image_size_mode
        self.reference_image_max_pixels = reference_image_max_pixels
        self.reference_video_short_edge = reference_video_short_edge
        self.reference_video_max_pixels = reference_video_max_pixels
        self.reference_video_fps = reference_video_fps

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

    @staticmethod
    def _video_loss_mask(item: Any, latent_shape: tuple[int, int, int]) -> torch.Tensor | None:
        content = getattr(item, "loss_mask_content", None)
        if content is None:
            return None
        mask = torch.as_tensor(np.asarray(content), dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(f"H3 loss mask must have pixel shape [frames, height, width], got {tuple(mask.shape)}")
        if mask.numel() and float(mask.max()) > 1.0:
            mask = mask / 255.0
        pixel_frames = int(mask.shape[0])
        latent_frames, latent_height, latent_width = latent_shape
        if latent_frames == 1:
            chunks = ((0, pixel_frames, 1),)
        else:
            if pixel_frames < 5 or (pixel_frames - 5) % 17:
                raise ValueError(f"H3 loss mask has {pixel_frames} frames, which is outside the 17n+5 video grid")
            temporal_blocks = (pixel_frames - 5) // 17
            expected_latents = 2 + 5 * temporal_blocks
            if latent_frames != expected_latents:
                raise ValueError(
                    f"H3 loss mask has {pixel_frames} frames for {latent_frames} video latents; "
                    f"expected {expected_latents} latent frames"
                )
            chunks = ((0, 5, 2),) + tuple((5 + 17 * index, 5 + 17 * (index + 1), 5) for index in range(temporal_blocks))
        pooled = []
        for start, end, output_frames in chunks:
            if end <= start:
                raise ValueError("H3 loss mask contains no frames")
            spatial = F.adaptive_max_pool2d(mask[start:end, None], (latent_height, latent_width))
            temporal = F.adaptive_max_pool1d(
                spatial[:, 0].permute(1, 2, 0).reshape(1, latent_height * latent_width, end - start),
                output_frames,
            )
            pooled.append(temporal.reshape(latent_height, latent_width, output_frames).permute(2, 0, 1))
        return torch.cat(pooled).to(dtype=torch.bool)

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
        references = prepare_references(
            item,
            self.reference_image_short_edge,
            self.reference_image_size_mode,
            self.reference_image_max_pixels,
            self.reference_video_short_edge,
            self.reference_video_max_pixels,
            self.reference_video_fps,
        )
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
                video_rows.append(patchify_video_latents(latent[None], VIDEO_DIT_PATCH_SIZE)[0])

            if reference.waveform is None:
                audio_lengths.append(0)
            else:
                audio = self._encode_reference_audio(reference.waveform)
                audio_lengths.append(int(audio.shape[-1]))
                audio_rows.append(pack_audio_latents(audio[None])[0])

        dtype_name = dtype_to_str(self.output_dtype)
        suffix = reference_key_suffix(
            self.reference_image_short_edge,
            self.reference_image_size_mode,
            self.reference_image_max_pixels,
            self.reference_video_short_edge,
            self.reference_video_max_pixels,
            self.reference_video_fps,
        )
        tensors = {
            f"varlen_{H3_REFERENCE_KINDS_KEY}{suffix}_int64": torch.tensor(kinds, dtype=torch.long),
            f"varlen_{H3_REFERENCE_VIDEO_SHAPES_KEY}{suffix}_int64": torch.tensor(video_shapes, dtype=torch.long),
            f"varlen_{H3_REFERENCE_AUDIO_LENGTHS_KEY}{suffix}_int64": torch.tensor(audio_lengths, dtype=torch.long),
            f"varlen_{H3_REFERENCE_VIDEO_ROWS_KEY}{suffix}_{dtype_name}": (
                torch.cat(video_rows)
                if video_rows
                else torch.empty((0, VIDEO_LATENT_CHANNELS * int(np.prod(VIDEO_DIT_PATCH_SIZE))), dtype=self.output_dtype)
            ),
            f"varlen_{H3_REFERENCE_AUDIO_ROWS_KEY}{suffix}_{dtype_name}": (
                torch.cat(audio_rows) if audio_rows else torch.empty((0, AUDIO_LATENT_CHANNELS), dtype=self.output_dtype)
            ),
        }
        if any(reference.kind is H3ReferenceKind.VIDEO for reference in references):
            tensors[H3_REFERENCE_TEMPORAL_CONTRACT_KEY] = torch.tensor(H3_REFERENCE_TEMPORAL_CONTRACT_VERSION, dtype=torch.long)
        return tensors

    def _encode_audio(self, item: Any) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._target_asset(item)
        audio_path = target.metadata.get("audio_path")
        if audio_path:
            frame_count = int(target.metadata.get("audio_frame_count", target.metadata.get("frame_count", 1)))
            target = MediaAsset(
                Path(audio_path),
                MediaModality.AUDIO,
                "target",
                start_seconds=target.start_seconds,
                duration_seconds=target.duration_seconds,
                metadata={"frame_count": frame_count, "fps": target.metadata.get("fps")},
            )
        if target.modality not in {MediaModality.VIDEO, MediaModality.AUDIO}:
            raise ValueError("MiniMax H3 target audio is defined only for video or audio targets")
        if self.audio_encoder is None:
            raise ValueError("MiniMax H3 audio latent caching requires --audio_vae")
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
                tensors = {
                    f"{H3_AUDIO_LATENTS_KEY}_2x32x{audio.shape[-1]}_{dtype_name}": audio,
                    H3_AUDIO_LOSS_MASK_KEY: audio_mask,
                    f"{H3_VIDEO_GEOMETRY_KEY}_int64": torch.tensor([latent_height, latent_width], dtype=torch.long),
                }
                # An audio target has no video rows of its own, but Ref2VA references are
                # cached exactly as they are for a video target: the packed sequence keeps
                # the reference prefix and only the target video block is empty.
                tensors.update(self._encode_references(item))
                results.append(tensors)
                continue
            conditioned_image = getattr(item, "h3_image_mode", "none") != "none"
            is_image = target.modality is MediaModality.IMAGE and not conditioned_image
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
            video_loss_mask = self._video_loss_mask(item, tuple(int(value) for value in video.shape[-3:]))
            if video_loss_mask is not None:
                tensors["video_loss_mask"] = video_loss_mask
            if target_mode != "video" and (not is_image or target.metadata.get("audio_path")):
                audio_frame_count = int(target.metadata.get("audio_frame_count", video_frame_count))
                expected = temporal_shape(audio_frame_count)
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
                if conditioned_image:
                    from musubi_tuner.minimax_h3.image_training import condition_images

                    height, width = int(item.content.shape[1]), int(item.content.shape[2])
                    prepared = tuple(
                        np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)).copy()
                        for image in condition_images(item.h3_image_mode, item.h3_condition_paths)
                    )
                    first_content, last_content = prepared
                else:
                    first_content, last_content = item.content[0], item.content[-1]
                first = self._encode_reference_video(first_content, image=True)
                last = self._encode_reference_video(last_content, image=True)
                keyframe_rows = torch.cat(
                    (
                        patchify_video_latents(first[None], VIDEO_DIT_PATCH_SIZE)[0],
                        patchify_video_latents(last[None], VIDEO_DIT_PATCH_SIZE)[0],
                    )
                )
                tensors[f"varlen_{H3_KEYFRAME_VIDEO_ROWS_KEY}_{dtype_name}"] = keyframe_rows
            tensors.update(self._encode_references(item))
            results.append(tensors)
        return tuple(results)
