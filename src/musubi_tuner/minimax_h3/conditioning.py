from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from accelerate import init_empty_weights
from PIL import Image, ImageOps
from safetensors import safe_open
from torch import nn
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLConfig, Qwen3VLModel

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_REFERENCE_VIDEO_FPS_KEY,
    H3_REFERENCE_VIDEO_MAX_PIXELS_KEY,
    H3_REFERENCE_VIDEO_SHORT_EDGE_KEY,
    H3_REFERENCE_IMAGE_MAX_PIXELS_KEY,
    H3_REFERENCE_IMAGE_SIZE_MODE_KEY,
    H3_REFERENCE_MODALITY_PROBABILITIES_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_VERSION,
    H3_MAX_CAPTION_TOKENS_KEY,
    H3_QWEN_CONTROL_VISUALS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_VISUAL_MAX_PIXELS_KEY,
    qwen_control_assets,
    qwen_control_dropout_key,
    reference_variant_key,
)
from musubi_tuner.minimax_h3.comfy_quant import (
    has_comfy_quantized_layers,
    load_comfy_quantized_state_dict,
    nvfp4_scaled_mm_available,
)
from musubi_tuner.minimax_h3.component_loader import resolve_nvfp4_awq_text_encoder_checkpoint, text_encoder_metadata
from musubi_tuner.minimax_h3.media import MediaModality
from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag
from musubi_tuner.minimax_h3.references import (
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODE,
    REFERENCE_VIDEO_FPS,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SAMPLE_FPS,
    REFERENCE_VIDEO_SHORT_EDGE,
    H3PreparedReference,
    H3ReferenceKind,
    _decode_video,
    prepare_references,
    reference_modality_variant,
    sample_reference_video_frames,
    subsample_reference_frames,
)
from musubi_tuner.utils.model_utils import dtype_to_str

logger = logging.getLogger(__name__)

H3TextEncoderQuantization = Literal["none", "int8", "nf4", "nvfp4_awq"]


def _cap_image_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    size = (max(32, int(image.width * scale) // 32 * 32), max(32, int(image.height * scale) // 32 * 32))
    return image.resize(size, Image.Resampling.LANCZOS)


def _cap_reference_visuals(references: tuple[H3PreparedReference, ...], max_pixels: int) -> tuple[H3PreparedReference, ...]:
    """Create the Qwen-only visual presentation without changing VAE references."""
    if max_pixels <= 0:
        return references
    capped: list[H3PreparedReference] = []
    for reference in references:
        if reference.kind is H3ReferenceKind.IMAGE and reference.image is not None:
            capped.append(replace(reference, image=_cap_image_pixels(reference.image, max_pixels)))
        elif reference.kind is H3ReferenceKind.VIDEO and reference.frames is not None:
            height, width = reference.frames.shape[1:3]
            if height * width <= max_pixels:
                capped.append(reference)
            else:
                capped_size = _cap_image_pixels(Image.fromarray(reference.frames[0]), max_pixels).size
                frames = np.stack(
                    [np.asarray(Image.fromarray(frame).resize(capped_size, Image.Resampling.LANCZOS)) for frame in reference.frames]
                )
                capped.append(replace(reference, frames=frames))
        else:
            capped.append(reference)
    return tuple(capped)


def prepare_qwen_controls(item: Any, video_sample_fps: float = REFERENCE_VIDEO_FPS) -> tuple[H3PreparedReference, ...]:
    """Decode the EXPERIMENTAL Qwen-only control visuals attached to one item.

    Nothing here touches the VAE: the frames and images exist solely to be handed
    to the Qwen3-VL processor, so no reference short-edge sizing, no 17n+5 frame
    landing and no soundtrack preparation apply. A control video is subsampled
    straight onto the presentation grid -- ``reference_video_fps`` when the run
    opted into temporal subsampling, otherwise Qwen's own 2 fps presentation rate
    -- because the released 24 fps resample only exists to feed the video VAE.
    The Qwen visual pixel cap is applied later, with the other visuals.
    """
    prepared: list[H3PreparedReference] = []
    for asset in qwen_control_assets(item):
        if asset.modality is MediaModality.IMAGE:
            with Image.open(asset.path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").copy()
            prepared.append(H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=image))
        else:
            frames, source_fps = _decode_video(asset.path)
            sample_fps = float(video_sample_fps) or float(REFERENCE_VIDEO_SAMPLE_FPS)
            prepared.append(
                H3PreparedReference(
                    kind=H3ReferenceKind.VIDEO,
                    frames=subsample_reference_frames(frames, source_fps, sample_fps),
                    sample_fps=sample_fps,
                )
            )
    return tuple(prepared)


def _text_encoder_key(source_prefix: str) -> str:
    if source_prefix.startswith("visual"):
        return source_prefix
    if source_prefix == "model":
        return "language_model"
    if source_prefix.startswith("model."):
        return f"language_model.{source_prefix.removeprefix('model.')}"
    return source_prefix


def _mapped_text_encoder_state_dict(
    checkpoint_path: Path,
    expected: set[str],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device=str(device)) as handle:
        for source_key in handle.keys():  # noqa: SIM118 - safetensors.safe_open is not iterable
            if source_key.startswith(("visual.", "model.")):
                source_prefix, _, suffix = source_key.rpartition(".")
                target_key = f"{_text_encoder_key(source_prefix)}.{suffix}"
            else:
                raise ValueError(f"{checkpoint_path.name} contains unexpected key {source_key!r}")
            if target_key not in expected:
                raise ValueError(f"{checkpoint_path.name} contains unexpected conditioner key {source_key!r}")
            tensor = handle.get_tensor(source_key)
            if tensor.dtype is not torch.bfloat16:
                raise ValueError(f"{source_key}: expected torch.bfloat16, got {tensor.dtype}")
            state_dict[target_key] = tensor
    missing = sorted(expected - set(state_dict))
    if missing:
        raise ValueError(f"{checkpoint_path.name} is missing {len(missing)} text tensor(s), examples: {missing[:5]}")
    return state_dict


def _load_bnb_text_conditioner(
    full_config: Qwen3VLConfig,
    state_dict: dict[str, torch.Tensor],
    target_device: torch.device,
    quantization: Literal["int8", "nf4"],
) -> Qwen3VLModel:
    if target_device.type != "cuda":
        raise ValueError("MiniMax H3 bitsandbytes text-encoder quantization requires a CUDA device")
    if quantization == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Qwen3VLModel constructs its final RMSNorm even though H3 consumes the raw
    # layer-50 state. A disposable value preserves strict load accounting; the
    # H3 adapter then exposes the pre-norm language model.
    state_dict["language_model.norm.weight"] = torch.ones(
        full_config.text_config.hidden_size,
        dtype=torch.bfloat16,
    )

    model, loading_info = Qwen3VLModel.from_pretrained(
        None,
        config=full_config,
        state_dict=state_dict,
        quantization_config=quantization_config,
        device_map={"": target_device},
        dtype=torch.bfloat16,
        local_files_only=True,
        output_loading_info=True,
    )
    # H3 deliberately consumes the unnormalized layer-50 hidden state, so the
    # final Qwen RMSNorm is absent from the released conditioner checkpoint.
    missing = set(loading_info["missing_keys"])
    if missing or loading_info["unexpected_keys"] or loading_info["mismatched_keys"] or loading_info["error_msgs"]:
        raise RuntimeError(
            "strict quantized Qwen3-VL load failed: "
            f"missing={sorted(missing)[:5]}, unexpected={loading_info['unexpected_keys'][:5]}, "
            f"mismatched={loading_info['mismatched_keys'][:5]}, errors={loading_info['error_msgs'][:5]}"
        )
    model.language_model.norm = nn.Identity()
    return model


def load_text_conditioner(
    checkpoint: Path,
    tokenizer: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    quantization: H3TextEncoderQuantization = "none",
    blocks_to_stream: int = 0,
    stream_ring_size: int = 2,
    stream_pinned_memory: bool = False,
    nvfp4_scaled_mm: bool = False,
) -> tuple[Any, Qwen3VLModel]:
    if dtype is not torch.bfloat16:
        raise ValueError("MiniMax H3 Qwen3-VL conditioning requires bfloat16")
    if quantization not in ("none", "int8", "nf4", "nvfp4_awq"):
        raise ValueError(f"unsupported MiniMax H3 text-encoder quantization: {quantization}")
    if not 0 <= blocks_to_stream <= 50:
        raise ValueError("MiniMax H3 text-encoder blocks_to_stream must be between 0 and 50")
    if blocks_to_stream and torch.device(device).type != "cuda":
        raise ValueError("MiniMax H3 text-encoder layer streaming requires CUDA")
    if blocks_to_stream and quantization in {"int8", "nf4"}:
        raise ValueError("MiniMax H3 text-encoder layer streaming does not support bitsandbytes INT8/NF4")
    if nvfp4_scaled_mm and quantization != "nvfp4_awq":
        raise ValueError("NVFP4 scaled_mm requires --text_encoder_quantization nvfp4_awq")
    target_device = torch.device(device)
    if nvfp4_scaled_mm and not nvfp4_scaled_mm_available(target_device):
        raise ValueError("NVFP4 scaled_mm requires PyTorch 2.10+ and a Blackwell CUDA GPU")
    checkpoint_source = Path(checkpoint)
    if quantization == "nvfp4_awq":
        checkpoint_path = resolve_nvfp4_awq_text_encoder_checkpoint(checkpoint_source)
        if not has_comfy_quantized_layers(checkpoint_path):
            raise ValueError("NVFP4/AWQ text-encoder mode requires the Comfy-Org quantized Qwen3-VL checkpoint")
    else:
        checkpoint_path, _ = text_encoder_metadata(checkpoint_source)
    full_config = Qwen3VLConfig.from_pretrained(tokenizer, local_files_only=True)
    full_config.text_config.num_hidden_layers = 50
    full_config.text_config.use_cache = False
    with init_empty_weights(include_buffers=True):
        model = Qwen3VLModel(full_config)
        model.language_model.norm = nn.Identity()

    if quantization == "nvfp4_awq":
        load_comfy_quantized_state_dict(
            model,
            checkpoint_path,
            key_map=_text_encoder_key,
            output_dtype=dtype,
            nvfp4_scaled_mm=nvfp4_scaled_mm,
        )
    else:
        expected = set(model.state_dict())
        checkpoint_device = target_device if quantization == "none" and not blocks_to_stream else torch.device("cpu")
        state_dict = _mapped_text_encoder_state_dict(checkpoint_path, expected, device=checkpoint_device)
        if quantization == "none":
            info = model.load_state_dict(state_dict, strict=True, assign=True)
            if info.missing_keys or info.unexpected_keys:
                raise RuntimeError(f"strict Qwen3-VL load failed: {info}")
        else:
            del model
            model = _load_bnb_text_conditioner(full_config, state_dict, target_device, quantization)
        del state_dict
    vision_config = full_config.vision_config
    vision_head_dim = vision_config.hidden_size // vision_config.num_heads
    model.visual.rotary_pos_emb = type(model.visual.rotary_pos_emb)(vision_head_dim // 2).to(target_device)
    model.language_model.rotary_emb = type(model.language_model.rotary_emb)(
        full_config.text_config,
        device=target_device,
    )
    if quantization == "nvfp4_awq" and not blocks_to_stream:
        model.to(target_device)
    if blocks_to_stream:
        from musubi_tuner.modules.custom_offloading_utils import ForwardOnlyBlockStreamer

        model.requires_grad_(False)
        layers = model.language_model.layers
        model.language_model.layers = nn.ModuleList()
        try:
            model.to(target_device)
        finally:
            model.language_model.layers = layers
        streamer = ForwardOnlyBlockStreamer(
            "h3_text_encoder",
            list(layers),
            blocks_to_stream,
            target_device,
            ring_size=stream_ring_size,
            use_pinned_memory=stream_pinned_memory,
        )
        streamer.prepare()
        model._h3_layer_streamer = streamer
    model.requires_grad_(False).eval()
    processor = AutoProcessor.from_pretrained(tokenizer, local_files_only=True, use_fast=True)
    logger.info(
        "Loaded raw-layer-50 Qwen3-VL conditioner using %s weights on %s%s",
        quantization,
        target_device,
        f" with {blocks_to_stream} streamed layers" if blocks_to_stream else "",
    )
    return processor, model


class MiniMaxH3ConditioningEncoder:
    def __init__(
        self,
        processor: Any,
        model: Qwen3VLModel,
        output_dtype: torch.dtype,
        task: Literal["t2va", "i2va", "fl2va", "l2va", "ref2va", "ref2va_omni"],
        reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
        text_visual_max_pixels: int = 0,
        reference_image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
        reference_image_max_pixels: int = 0,
        reference_video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
        reference_video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
        reference_video_fps: float = REFERENCE_VIDEO_FPS,
        max_caption_tokens: int = 0,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model = model
        self.output_dtype = output_dtype
        self.task = task
        self.reference_image_short_edge = reference_image_short_edge
        self.max_caption_tokens = max_caption_tokens
        self.text_visual_max_pixels = text_visual_max_pixels
        self.reference_image_size_mode = reference_image_size_mode
        self.reference_image_max_pixels = reference_image_max_pixels
        self.reference_video_short_edge = reference_video_short_edge
        self.reference_video_max_pixels = reference_video_max_pixels
        self.reference_video_fps = reference_video_fps
        # Even T2VA enumerates decoded video crops so its cache filename shares
        # the same crop identity as FL2VA and the corresponding latent cache.
        self.conditioning_requires_content = True

    def close(self) -> None:
        streamer = getattr(self.model, "_h3_layer_streamer", None)
        if streamer is not None:
            streamer.close()
            delattr(self.model, "_h3_layer_streamer")

    def _encode_prompt(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        references: tuple[H3PreparedReference, ...] | None = None,
        null_instruction: bool = False,
        qwen_controls: tuple[H3PreparedReference, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images and references:
            raise ValueError("H3 conditioning accepts keyframes or Ref2VA references, not both")
        if references:
            references = _cap_reference_visuals(references, self.text_visual_max_pixels)
        # Qwen control visuals are a third population: they compose with either of
        # the two above and are never subject to the keyframe/reference XOR.
        qwen_controls = _cap_reference_visuals(qwen_controls, self.text_visual_max_pixels) if qwen_controls else ()
        token_ids: list[int] = []
        token_tags: list[int] = []
        pixel_values = None
        image_grid_thw = None
        pixel_values_videos = None
        video_grid_thw = None
        merge_size = self.processor.image_processor.merge_size**2
        vision_start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        image_pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        prepared_images = images or []
        if references:
            prepared_images = []
            for reference in references:
                if reference.kind is H3ReferenceKind.IMAGE:
                    if reference.image is None:
                        raise ValueError("H3 prepared image reference has no image")
                    prepared_images.append(reference.image)
        control_images = []
        for control in qwen_controls:
            if control.kind is H3ReferenceKind.IMAGE:
                if control.image is None:
                    raise ValueError("H3 prepared Qwen control image has no image")
                control_images.append(control.image)
        image_token_counts: list[int] = []
        control_image_token_counts: list[int] = []
        if prepared_images or control_images:
            # One processor call keeps the flattened patch tensor in the same
            # order the vision spans are emitted in: primaries first, controls
            # after them.
            vision = self.processor.image_processor(images=[*prepared_images, *control_images], return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]
            image_token_counts = counts[: len(prepared_images)]
            control_image_token_counts = counts[len(prepared_images) :]

        video_token_counts: list[int] = []
        control_video_token_counts: list[int] = []
        videos = [reference for reference in references or () if reference.kind is H3ReferenceKind.VIDEO]
        control_videos = [control for control in qwen_controls if control.kind is H3ReferenceKind.VIDEO]
        if videos or control_videos:
            if any(reference.frames is None for reference in (*videos, *control_videos)):
                raise ValueError("H3 prepared video reference has no frames")
            sampled = [
                sample_reference_video_frames(reference.frames, reference.sample_fps) for reference in (*videos, *control_videos)
            ]
            for reference, (_, timestamps) in zip((*videos, *control_videos), sampled):
                reference.block_timestamps = timestamps
            vision = self.processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled],
                do_sample_frames=False,
                return_tensors="pt",
            )
            pixel_values_videos = vision["pixel_values_videos"]
            video_grid_thw = vision["video_grid_thw"]
            counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]
            video_token_counts = counts[: len(videos)]
            control_video_token_counts = counts[len(videos) :]
            for reference, grid in zip((*videos, *control_videos), video_grid_thw):
                if int(grid[0]) != len(reference.block_timestamps):
                    raise ValueError("H3 reference video timestamps do not match Qwen3-VL vision blocks")

        def emit_text(value: str) -> None:
            ids = self.tokenizer(value, add_special_tokens=False)["input_ids"]
            token_ids.extend(ids)
            token_tags.extend([int(MiniMaxH3TokenTag.TEXT)] * len(ids))

        def emit_vision(pad_token: int, count: int) -> None:
            ids = [vision_start, *([pad_token] * count), vision_end]
            token_ids.extend(ids)
            token_tags.extend([int(MiniMaxH3TokenTag.VIDEO)] * len(ids))

        picture_index = 0
        video_index = 0
        if references:
            counts = {H3ReferenceKind.IMAGE: 0, H3ReferenceKind.VIDEO: 0, H3ReferenceKind.AUDIO: 0}
            for reference in references:
                if reference.has_audio:
                    counts[H3ReferenceKind.AUDIO] += 1
                    emit_text(f"<Audio {counts[H3ReferenceKind.AUDIO]}>: ")
                if reference.kind is H3ReferenceKind.IMAGE:
                    counts[H3ReferenceKind.IMAGE] += 1
                    index = counts[H3ReferenceKind.IMAGE] - 1
                    emit_text(f"<Picture {index + 1}>: ")
                    emit_vision(image_pad, image_token_counts[index])
                elif reference.kind is H3ReferenceKind.VIDEO:
                    counts[H3ReferenceKind.VIDEO] += 1
                    index = counts[H3ReferenceKind.VIDEO] - 1
                    emit_text(f"<Video {index + 1}>: ")
                    for timestamp in reference.block_timestamps:
                        emit_text(f"<{timestamp:.1f} seconds>")
                        emit_vision(video_pad, video_token_counts[index])
            picture_index = counts[H3ReferenceKind.IMAGE]
            video_index = counts[H3ReferenceKind.VIDEO]
        elif images:
            for index, image_tokens in enumerate(image_token_counts):
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = [vision_start, *([image_pad] * image_tokens), vision_end]
                token_ids.extend(label_ids)
                token_ids.extend(vision_ids)
                token_tags.extend([int(MiniMaxH3TokenTag.TEXT)] * len(label_ids))
                token_tags.extend([int(MiniMaxH3TokenTag.VIDEO)] * len(vision_ids))
            picture_index = len(image_token_counts)
        if qwen_controls:
            # Placement contract: the released presentation puts every vision span
            # before the instruction, so the controls close the visual prefix --
            # after any keyframes or Ref2VA references, immediately before the
            # caption. Existing spans keep both their position and their
            # <Picture N>/<Video N> numbers; the controls continue those counters.
            control_image_index = 0
            control_video_index = 0
            for control in qwen_controls:
                if control.kind is H3ReferenceKind.IMAGE:
                    picture_index += 1
                    emit_text(f"<Picture {picture_index}>: ")
                    emit_vision(image_pad, control_image_token_counts[control_image_index])
                    control_image_index += 1
                else:
                    video_index += 1
                    emit_text(f"<Video {video_index}>: ")
                    for timestamp in control.block_timestamps:
                        emit_text(f"<{timestamp:.1f} seconds>")
                        emit_vision(video_pad, control_video_token_counts[control_video_index])
                    control_video_index += 1
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if self.max_caption_tokens > 0:
            prompt_ids = prompt_ids[: self.max_caption_tokens]
        if null_instruction:
            # The unconditional branch must drop the instruction without dropping
            # its rows. H3's media rotary clock originates at the number of text
            # rows (packing.py, num_text_rows), so a null branch encoded from ""
            # is shorter and silently relocates every audio and video coordinate
            # -- and for T2VA, where the instruction is the entire presentation,
            # it collapses to zero tokens. Substituting padding in place keeps the
            # row count, the tags and every media coordinate identical between
            # branches while carrying no instruction content.
            prompt_ids = [self._null_token_id()] * len(prompt_ids)
        token_ids.extend(prompt_ids)
        token_tags.extend([int(MiniMaxH3TokenTag.TEXT)] * len(prompt_ids))
        if not token_ids:
            hidden = torch.empty((0, self.model.config.text_config.hidden_size), dtype=self.output_dtype)
            tags = torch.empty((0,), dtype=torch.long)
            return hidden, tags
        if len(token_ids) > 32_768:
            raise ValueError(f"MiniMax H3 Qwen3-VL presentation has {len(token_ids)} tokens; maximum is 32768")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        mm_token_type_ids[input_ids == image_pad_id] = 1
        mm_token_type_ids[input_ids == video_pad_id] = 2
        with torch.no_grad():
            bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
            previous_bnb_level = bnb_logger.level
            if getattr(self.model, "is_loaded_in_8bit", False):
                bnb_logger.setLevel(logging.ERROR)
            try:
                hidden = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    mm_token_type_ids=mm_token_type_ids,
                    pixel_values=None if pixel_values is None else pixel_values.to(self.model.device, dtype=self.model.dtype),
                    image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(self.model.device),
                    pixel_values_videos=(
                        None if pixel_values_videos is None else pixel_values_videos.to(self.model.device, dtype=self.model.dtype)
                    ),
                    video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(self.model.device),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[0]
            finally:
                bnb_logger.setLevel(previous_bnb_level)
        hidden = hidden.to(dtype=self.output_dtype, device="cpu")
        tags = torch.tensor(token_tags, dtype=torch.long)
        return hidden, tags

    def _null_token_id(self) -> int:
        """The filler token that stands in for a removed instruction."""
        for candidate in (getattr(self.tokenizer, "pad_token_id", None), getattr(self.tokenizer, "eos_token_id", None)):
            if candidate is not None:
                return int(candidate)
        raise ValueError("H3 null conditioning needs a pad or eos token; this tokenizer defines neither")

    def _images_for_item(self, item: Any) -> list[Image.Image] | None:
        if self.task in ("t2va", "ref2va", "ref2va_omni"):
            return None
        if getattr(item, "h3_image_mode", "none") != "none":
            if self.task != "fl2va":
                raise ValueError("MiniMax H3 conditioned-image training requires --task fl2va")
            from musubi_tuner.minimax_h3.image_training import read_text_visual

            images = [read_text_visual(path, self.text_visual_max_pixels) for path in item.h3_condition_paths]
            return [images[0], images[0]] if item.h3_image_mode == "first" else images
        content = item.content
        minimum_frames = 2 if self.task == "fl2va" else 1
        if not isinstance(content, np.ndarray) or content.ndim != 4 or content.shape[0] < minimum_frames:
            raise ValueError(
                f"MiniMax H3 {self.task.upper()} conditioning requires a decoded target video with at least "
                f"{minimum_frames} frame(s)"
            )
        if self.task == "i2va":
            return [Image.fromarray(content[0].astype(np.uint8))]
        if self.task == "l2va":
            return [Image.fromarray(content[-1].astype(np.uint8))]
        return [
            Image.fromarray(content[0].astype(np.uint8)),
            Image.fromarray(content[-1].astype(np.uint8)),
        ]

    def encode_prompt(self, prompt: str, images: list[Image.Image] | None = None) -> dict[str, torch.Tensor]:
        """Encode one FL2VA-family prompt with optional prepared endpoint keyframes."""
        if images and self.text_visual_max_pixels > 0:
            resized = []
            for image in images:
                if image.width * image.height <= self.text_visual_max_pixels:
                    resized.append(image)
                    continue
                scale = (self.text_visual_max_pixels / (image.width * image.height)) ** 0.5
                size = (max(32, int(image.width * scale) // 32 * 32), max(32, int(image.height * scale) // 32 * 32))
                resized.append(image.resize(size, Image.Resampling.LANCZOS))
            images = resized
        hidden, tags = self._encode_prompt(prompt, images)
        return {H3_TEXT_HIDDEN_KEY: hidden, H3_TEXT_TOKEN_TAGS_KEY: tags}

    def encode_reference_prompt(
        self,
        prompt: str,
        references: tuple[H3PreparedReference, ...],
    ) -> dict[str, torch.Tensor]:
        """Encode one Ref2VA prompt and its optional ordered reference presentation."""
        if not references and self.task != "ref2va_omni":
            raise ValueError("MiniMax H3 Ref2VA conditioning requires at least one reference")
        hidden, tags = self._encode_prompt(prompt, references=references)
        return {H3_TEXT_HIDDEN_KEY: hidden, H3_TEXT_TOKEN_TAGS_KEY: tags}

    def encode_conditioning(
        self,
        batch: list[Any],
        *,
        include_empty: bool = False,
        include_qwen_control_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        dtype_name = dtype_to_str(self.output_dtype)
        results = []
        for item in batch:
            references = (
                prepare_references(
                    item,
                    self.reference_image_short_edge,
                    self.reference_image_size_mode,
                    self.reference_image_max_pixels,
                    self.reference_video_short_edge,
                    self.reference_video_max_pixels,
                    self.reference_video_fps,
                )
                if self.task in ("ref2va", "ref2va_omni")
                else None
            )
            if self.task == "ref2va" and not references:
                raise ValueError("MiniMax H3 Ref2VA conditioning requires at least one reference")
            qwen_controls = prepare_qwen_controls(item, self.reference_video_fps)
            # EXPERIMENTAL per-step control dropout is a cache-side dual
            # presentation, mirroring the reference-modality variants: every
            # presentation this item caches gains a control-free twin, and the
            # trainer draws between the two per step without re-encoding.
            control_dropout = bool(include_qwen_control_dropout and qwen_controls)
            images = self._images_for_item(item)
            hidden, tags = self._encode_prompt(item.caption, images, references, qwen_controls=qwen_controls)
            tensors = {
                f"varlen_{H3_TEXT_HIDDEN_KEY}_{dtype_name}": hidden,
                f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": tags,
                H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS[self.task], dtype=torch.long),
            }
            if self.max_caption_tokens:
                tensors[H3_MAX_CAPTION_TOKENS_KEY] = torch.tensor(self.max_caption_tokens, dtype=torch.long)
            if self.text_visual_max_pixels:
                tensors[H3_TEXT_VISUAL_MAX_PIXELS_KEY] = torch.tensor(self.text_visual_max_pixels, dtype=torch.long)
            if qwen_controls:
                # The marker is what lets T2VA training accept a presentation that
                # carries vision rows: the tags themselves cannot tell a control
                # span apart from a keyframe or reference span.
                tensors[H3_QWEN_CONTROL_VISUALS_KEY] = torch.tensor(len(qwen_controls), dtype=torch.long)
            if control_dropout:
                dropped_hidden, dropped_tags = self._encode_prompt(item.caption, images, references)
                tensors[f"varlen_{qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY)}_{dtype_name}"] = dropped_hidden
                tensors[f"varlen_{qwen_control_dropout_key(H3_TEXT_TOKEN_TAGS_KEY)}_int64"] = dropped_tags
            probabilities = getattr(item, "h3_reference_modality_probabilities", None)
            if probabilities is not None:
                if references is None:
                    raise ValueError("control_modality_probabilities is only valid for Ref2VA conditioning")
                tensors[f"{H3_REFERENCE_MODALITY_PROBABILITIES_KEY}_float32"] = torch.tensor(probabilities, dtype=torch.float32)
                for modality, probability in zip(("av", "video", "audio"), probabilities):
                    if modality == "av" or probability <= 0:
                        continue
                    variant = reference_modality_variant(references, modality)
                    variant_hidden, variant_tags = self._encode_prompt(
                        item.caption, references=variant, qwen_controls=qwen_controls
                    )
                    tensors[f"varlen_{reference_variant_key(H3_TEXT_HIDDEN_KEY, modality)}_{dtype_name}"] = variant_hidden
                    tensors[f"varlen_{reference_variant_key(H3_TEXT_TOKEN_TAGS_KEY, modality)}_int64"] = variant_tags
                    if control_dropout:
                        dropped_variant_hidden, dropped_variant_tags = self._encode_prompt(item.caption, references=variant)
                        hidden_key = qwen_control_dropout_key(reference_variant_key(H3_TEXT_HIDDEN_KEY, modality))
                        tags_key = qwen_control_dropout_key(reference_variant_key(H3_TEXT_TOKEN_TAGS_KEY, modality))
                        tensors[f"varlen_{hidden_key}_{dtype_name}"] = dropped_variant_hidden
                        tensors[f"varlen_{tags_key}_int64"] = dropped_variant_tags
                    if include_empty:
                        empty_hidden, empty_tags = self._encode_prompt(
                            item.caption, references=variant, null_instruction=True, qwen_controls=qwen_controls
                        )
                        tensors[f"varlen_{reference_variant_key(H3_EMPTY_TEXT_HIDDEN_KEY, modality)}_{dtype_name}"] = empty_hidden
                        tensors[f"varlen_{reference_variant_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY, modality)}_int64"] = empty_tags
                        if control_dropout:
                            dropped_empty_hidden, dropped_empty_tags = self._encode_prompt(
                                item.caption, references=variant, null_instruction=True
                            )
                            hidden_key = qwen_control_dropout_key(reference_variant_key(H3_EMPTY_TEXT_HIDDEN_KEY, modality))
                            tags_key = qwen_control_dropout_key(reference_variant_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY, modality))
                            tensors[f"varlen_{hidden_key}_{dtype_name}"] = dropped_empty_hidden
                            tensors[f"varlen_{tags_key}_int64"] = dropped_empty_tags
            if self.task in ("ref2va", "ref2va_omni"):
                tensors[H3_REFERENCE_IMAGE_SHORT_EDGE_KEY] = torch.tensor(self.reference_image_short_edge, dtype=torch.long)
                tensors[H3_REFERENCE_IMAGE_SIZE_MODE_KEY] = torch.tensor(
                    0 if self.reference_image_size_mode == "short_edge" else 1, dtype=torch.long
                )
                tensors[H3_REFERENCE_IMAGE_MAX_PIXELS_KEY] = torch.tensor(self.reference_image_max_pixels, dtype=torch.long)
                tensors[H3_REFERENCE_VIDEO_SHORT_EDGE_KEY] = torch.tensor(self.reference_video_short_edge, dtype=torch.long)
                tensors[H3_REFERENCE_VIDEO_MAX_PIXELS_KEY] = torch.tensor(self.reference_video_max_pixels, dtype=torch.long)
                tensors[H3_REFERENCE_VIDEO_FPS_KEY] = torch.tensor(float(self.reference_video_fps), dtype=torch.float64)
                if references and any(reference.kind is H3ReferenceKind.VIDEO for reference in references):
                    tensors[H3_REFERENCE_TEMPORAL_CONTRACT_KEY] = torch.tensor(
                        H3_REFERENCE_TEMPORAL_CONTRACT_VERSION, dtype=torch.long
                    )
            if include_empty:
                empty_hidden, empty_tags = self._encode_prompt(
                    item.caption, images, references, null_instruction=True, qwen_controls=qwen_controls
                )
                tensors[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_{dtype_name}"] = empty_hidden
                tensors[f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64"] = empty_tags
                if control_dropout:
                    dropped_empty_hidden, dropped_empty_tags = self._encode_prompt(
                        item.caption, images, references, null_instruction=True
                    )
                    tensors[f"varlen_{qwen_control_dropout_key(H3_EMPTY_TEXT_HIDDEN_KEY)}_{dtype_name}"] = dropped_empty_hidden
                    tensors[f"varlen_{qwen_control_dropout_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY)}_int64"] = dropped_empty_tags
            results.append(tensors)
        return tuple(results)
