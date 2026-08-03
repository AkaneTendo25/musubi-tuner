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
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLModel

from musubi_tuner.minimax_h3.cache import (
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.component_loader import text_encoder_metadata
from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag
from musubi_tuner.utils.model_utils import dtype_to_str

logger = logging.getLogger(__name__)


def load_text_conditioner(
    checkpoint: Path,
    tokenizer: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Qwen3VLModel]:
    if dtype is not torch.bfloat16:
        raise ValueError("MiniMax H3 Qwen3-VL conditioning requires bfloat16")
    checkpoint_path, _ = text_encoder_metadata(checkpoint)
    full_config = Qwen3VLConfig.from_pretrained(tokenizer, local_files_only=True)
    full_config.text_config.num_hidden_layers = 50
    full_config.text_config.use_cache = False
    with init_empty_weights(include_buffers=True):
        model = Qwen3VLModel(full_config)
        model.language_model.norm = nn.Identity()

    expected = set(model.state_dict())
    state_dict: dict[str, torch.Tensor] = {}
    target_device = torch.device(device)
    with safe_open(checkpoint_path, framework="pt", device=str(target_device)) as handle:
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
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"strict Qwen3-VL load failed: {info}")
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
        "Loaded raw-layer-50 Qwen3-VL conditioner: %d tensors on %s",
        len(state_dict),
        target_device,
    )
    return processor, model


class MiniMaxH3ConditioningEncoder:
    def __init__(
        self,
        processor: Any,
        model: Qwen3VLModel,
        output_dtype: torch.dtype,
        task: Literal["t2va", "fl2va"],
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model = model
        self.output_dtype = output_dtype
        self.task = task
        # Even T2VA enumerates decoded video crops so its cache filename shares
        # the same crop identity as FL2VA and the corresponding latent cache.
        self.conditioning_requires_content = True

    def _encode_prompt(self, prompt: str, images: list[Image.Image] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids: list[int] = []
        token_tags: list[int] = []
        pixel_values = None
        image_grid_thw = None
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            merge_size = self.processor.image_processor.merge_size**2
            vision_start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            image_pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            vision_end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            for index, grid in enumerate(image_grid_thw):
                image_tokens = int(grid.prod()) // merge_size
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
            hidden = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=None if pixel_values is None else pixel_values.to(self.model.device, dtype=self.model.dtype),
                image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(self.model.device),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0]
        hidden = hidden.to(dtype=self.output_dtype, device="cpu")
        tags = torch.tensor(token_tags, dtype=torch.long)
        return hidden, tags

    def _images_for_item(self, item: Any) -> list[Image.Image] | None:
        if self.task == "t2va":
            return None
        content = item.content
        if not isinstance(content, np.ndarray) or content.ndim != 4 or content.shape[0] < 2:
            raise ValueError("MiniMax H3 FL2VA conditioning requires a decoded target video with at least two frames")
        return [Image.fromarray(content[0].astype(np.uint8)), Image.fromarray(content[-1].astype(np.uint8))]

    def encode_conditioning(self, batch: list[Any], *, include_empty: bool = False) -> tuple[dict[str, torch.Tensor], ...]:
        dtype_name = dtype_to_str(self.output_dtype)
        empty = self._encode_prompt("") if include_empty else None
        results = []
        for item in batch:
            hidden, tags = self._encode_prompt(item.caption, self._images_for_item(item))
            tensors = {
                f"varlen_{H3_TEXT_HIDDEN_KEY}_{dtype_name}": hidden,
                f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": tags,
            }
            if empty is not None:
                empty_hidden, empty_tags = empty
                tensors[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_{dtype_name}"] = empty_hidden
                tensors[f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64"] = empty_tags
            results.append(tensors)
        return tuple(results)
