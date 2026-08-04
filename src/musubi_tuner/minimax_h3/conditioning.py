from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open
from torch import nn
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLConfig, Qwen3VLModel

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.component_loader import text_encoder_metadata
from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag
from musubi_tuner.minimax_h3.references import (
    H3PreparedReference,
    H3ReferenceKind,
    prepare_references,
    sample_reference_video_frames,
)
from musubi_tuner.utils.model_utils import dtype_to_str

logger = logging.getLogger(__name__)

H3TextEncoderQuantization = Literal["none", "int8", "nf4"]


def _mapped_text_encoder_state_dict(
    checkpoint_path: Path,
    expected: set[str],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device=str(device)) as handle:
        for source_key in handle.keys():  # noqa: SIM118 - safetensors.safe_open is not iterable
            if source_key.startswith("visual."):
                target_key = source_key
            elif source_key.startswith("model."):
                target_key = f"language_model.{source_key.removeprefix('model.')}"
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
) -> tuple[Any, Qwen3VLModel]:
    if dtype is not torch.bfloat16:
        raise ValueError("MiniMax H3 Qwen3-VL conditioning requires bfloat16")
    if quantization not in ("none", "int8", "nf4"):
        raise ValueError(f"unsupported MiniMax H3 text-encoder quantization: {quantization}")
    checkpoint_path, _ = text_encoder_metadata(checkpoint)
    full_config = Qwen3VLConfig.from_pretrained(tokenizer, local_files_only=True)
    full_config.text_config.num_hidden_layers = 50
    full_config.text_config.use_cache = False
    with init_empty_weights(include_buffers=True):
        model = Qwen3VLModel(full_config)
        model.language_model.norm = nn.Identity()

    expected = set(model.state_dict())
    target_device = torch.device(device)
    checkpoint_device = target_device if quantization == "none" else torch.device("cpu")
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
    model.requires_grad_(False).eval()
    processor = AutoProcessor.from_pretrained(tokenizer, local_files_only=True, use_fast=True)
    logger.info(
        "Loaded raw-layer-50 Qwen3-VL conditioner using %s weights on %s",
        quantization,
        target_device,
    )
    return processor, model


class MiniMaxH3ConditioningEncoder:
    def __init__(
        self,
        processor: Any,
        model: Qwen3VLModel,
        output_dtype: torch.dtype,
        task: Literal["t2va", "i2va", "fl2va", "ref2va"],
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model = model
        self.output_dtype = output_dtype
        self.task = task
        # Even T2VA enumerates decoded video crops so its cache filename shares
        # the same crop identity as FL2VA and the corresponding latent cache.
        self.conditioning_requires_content = True

    def _encode_prompt(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        references: tuple[H3PreparedReference, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images and references:
            raise ValueError("H3 conditioning accepts keyframes or Ref2VA references, not both")
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
        image_token_counts: list[int] = []
        if prepared_images:
            vision = self.processor.image_processor(images=prepared_images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            image_token_counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]

        video_token_counts: list[int] = []
        if references:
            videos = [reference for reference in references if reference.kind is H3ReferenceKind.VIDEO]
            if videos:
                if any(reference.frames is None for reference in videos):
                    raise ValueError("H3 prepared video reference has no frames")
                sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
                for reference, (_, timestamps) in zip(videos, sampled):
                    reference.block_timestamps = timestamps
                vision = self.processor.video_processor(
                    videos=[np.stack(frames) for frames, _ in sampled],
                    do_sample_frames=False,
                    return_tensors="pt",
                )
                pixel_values_videos = vision["pixel_values_videos"]
                video_grid_thw = vision["video_grid_thw"]
                video_token_counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]
                for reference, grid in zip(videos, video_grid_thw):
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
        elif images:
            for index, image_tokens in enumerate(image_token_counts):
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = [vision_start, *([image_pad] * image_tokens), vision_end]
                token_ids.extend(label_ids)
                token_ids.extend(vision_ids)
                token_tags.extend([int(MiniMaxH3TokenTag.TEXT)] * len(label_ids))
                token_tags.extend([int(MiniMaxH3TokenTag.VIDEO)] * len(vision_ids))
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
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

    def _images_for_item(self, item: Any) -> list[Image.Image] | None:
        if self.task in ("t2va", "ref2va"):
            return None
        content = item.content
        minimum_frames = 1 if self.task == "i2va" else 2
        if not isinstance(content, np.ndarray) or content.ndim != 4 or content.shape[0] < minimum_frames:
            raise ValueError(
                f"MiniMax H3 {self.task.upper()} conditioning requires a decoded target video with at least "
                f"{minimum_frames} frame(s)"
            )
        images = [Image.fromarray(content[0].astype(np.uint8))]
        if self.task == "fl2va":
            images.append(Image.fromarray(content[-1].astype(np.uint8)))
        return images

    def encode_prompt(self, prompt: str, images: list[Image.Image] | None = None) -> dict[str, torch.Tensor]:
        """Encode one FL2VA prompt with optional prepared first/last keyframes."""
        hidden, tags = self._encode_prompt(prompt, images)
        return {H3_TEXT_HIDDEN_KEY: hidden, H3_TEXT_TOKEN_TAGS_KEY: tags}

    def encode_reference_prompt(
        self,
        prompt: str,
        references: tuple[H3PreparedReference, ...],
    ) -> dict[str, torch.Tensor]:
        """Encode one Ref2VA prompt and its ordered multimodal presentation."""
        if not references:
            raise ValueError("MiniMax H3 Ref2VA conditioning requires at least one reference")
        hidden, tags = self._encode_prompt(prompt, references=references)
        return {H3_TEXT_HIDDEN_KEY: hidden, H3_TEXT_TOKEN_TAGS_KEY: tags}

    def encode_conditioning(self, batch: list[Any], *, include_empty: bool = False) -> tuple[dict[str, torch.Tensor], ...]:
        dtype_name = dtype_to_str(self.output_dtype)
        results = []
        for item in batch:
            references = prepare_references(item) if self.task == "ref2va" else None
            if self.task == "ref2va" and not references:
                raise ValueError("MiniMax H3 Ref2VA conditioning requires at least one reference")
            hidden, tags = self._encode_prompt(item.caption, self._images_for_item(item), references)
            tensors = {
                f"varlen_{H3_TEXT_HIDDEN_KEY}_{dtype_name}": hidden,
                f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": tags,
                H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS[self.task], dtype=torch.long),
            }
            if include_empty:
                empty_hidden, empty_tags = self._encode_prompt("", self._images_for_item(item), references)
                tensors[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_{dtype_name}"] = empty_hidden
                tensors[f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64"] = empty_tags
            results.append(tensors)
        return tuple(results)
