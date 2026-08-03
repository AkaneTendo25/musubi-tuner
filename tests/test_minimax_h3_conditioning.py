from types import SimpleNamespace

import numpy as np
import torch

from musubi_tuner.cache_text_encoder_outputs import process_text_encoder_batches
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.cache import (
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder


class _Tokenizer:
    def __call__(self, prompt, **kwargs):
        del kwargs
        length = len(prompt.split())
        return {"input_ids": list(range(length))}

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 100, "<|image_pad|>": 101, "<|vision_end|>": 102}[token]


class _ImageProcessor:
    merge_size = 2

    def __call__(self, images, **kwargs):
        del kwargs
        return {
            "pixel_values": torch.zeros(len(images), 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class _Processor:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()


class _TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=5120))

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return torch.bfloat16

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        shape = (input_ids.shape[0], input_ids.shape[1], self.config.text_config.hidden_size)
        return SimpleNamespace(last_hidden_state=torch.ones(shape, dtype=torch.bfloat16))


def test_conditioning_cache_is_raw_text_rows_with_text_tags():
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), _TextModel(), torch.bfloat16, "t2va")
    item = SimpleNamespace(caption="two tokens")
    result = encoder.encode_conditioning([item])[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    tags = result[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert hidden.shape == (2, 5120)
    assert hidden.dtype is torch.bfloat16
    assert torch.equal(tags, torch.ones(2, dtype=torch.long))


def test_empty_conditioning_is_a_zero_row_sequence():
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), _TextModel(), torch.bfloat16, "t2va")
    result = encoder.encode_conditioning([SimpleNamespace(caption="prompt")], include_empty=True)[0]

    empty_hidden = result[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_bfloat16"]
    empty_tags = result[f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert empty_hidden.shape == (0, 5120)
    assert empty_tags.shape == (0,)


def test_fl2va_conditioning_includes_first_last_vision_rows():
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), _TextModel(), torch.bfloat16, "fl2va")
    content = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    result = encoder.encode_conditioning([SimpleNamespace(caption="prompt", content=content)])[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    tags = result[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert hidden.shape == (11, 5120)
    assert torch.equal(tags, torch.tensor([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]))
    assert encoder.conditioning_requires_content


def test_content_conditioning_populates_video_text_cache_path(tmp_path):
    item = ItemInfo("sample.mp4", "prompt", (4, 4), (4, 4), content=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    class _Dataset:
        def retrieve_latent_cache_batches(self, num_workers):
            assert num_workers == 1
            return [((4, 4, 2), [item])]

        def get_text_encoder_output_cache_path(self, item_info):
            assert item_info is item
            return str(tmp_path / "sample_mmh3_te.safetensors")

    encoded = []
    process_text_encoder_batches(
        1,
        False,
        1,
        [_Dataset()],
        [set()],
        [set()],
        lambda batch: encoded.extend(batch),
        requires_content=True,
    )

    assert encoded == [item]
    assert item.text_encoder_output_cache_path == str(tmp_path / "sample_mmh3_te.safetensors")
