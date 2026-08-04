from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from musubi_tuner.cache_text_encoder_outputs import process_text_encoder_batches
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.cache import (
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder
from musubi_tuner.minimax_h3.references import H3PreparedReference, H3ReferenceKind


class _Tokenizer:
    pad_token_id = 151643

    def __call__(self, prompt, **kwargs):
        del kwargs
        length = len(prompt.split())
        return {"input_ids": list(range(length))}

    def convert_tokens_to_ids(self, token):
        return {
            "<|vision_start|>": 100,
            "<|image_pad|>": 101,
            "<|vision_end|>": 102,
            "<|video_pad|>": 103,
        }[token]


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


class _RecordingTokenizer(_Tokenizer):
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append(prompt)
        return super().__call__(prompt, **kwargs)


class _VideoProcessor:
    def __call__(self, videos, **kwargs):
        del kwargs
        assert len(videos) == 1
        return {
            "pixel_values_videos": torch.zeros(3, 3, 2, 2),
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
        }


class _RefProcessor:
    def __init__(self):
        self.tokenizer = _RecordingTokenizer()
        self.image_processor = _ImageProcessor()
        self.video_processor = _VideoProcessor()


class _TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=5120))
        self.last_mm_token_type_ids = None
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return torch.bfloat16

    def forward(self, input_ids, attention_mask, mm_token_type_ids, **kwargs):
        del attention_mask, kwargs
        self.last_mm_token_type_ids = mm_token_type_ids.detach().cpu()
        self.calls.append((input_ids.detach().cpu(), self.last_mm_token_type_ids))
        shape = (input_ids.shape[0], input_ids.shape[1], self.config.text_config.hidden_size)
        return SimpleNamespace(last_hidden_state=torch.ones(shape, dtype=torch.bfloat16))


def test_conditioning_cache_is_raw_text_rows_with_text_tags():
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "t2va")
    item = SimpleNamespace(caption="two tokens")
    result = encoder.encode_conditioning([item])[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    tags = result[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert hidden.shape == (2, 5120)
    assert hidden.dtype is torch.bfloat16
    assert torch.equal(tags, torch.ones(2, dtype=torch.long))
    assert torch.equal(model.last_mm_token_type_ids, torch.zeros(1, 2, dtype=torch.long))


def test_empty_conditioning_preserves_the_prompt_row_count():
    """The null branch drops the instruction without dropping its rows.

    H3's media rotary clock starts at the number of text rows, so a shorter null
    branch moves every audio and video coordinate. For T2VA, where the prompt is
    the whole presentation, encoding "" collapsed it to nothing at all.
    """
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "t2va")
    result = encoder.encode_conditioning([SimpleNamespace(caption="three tokens here")], include_empty=True)[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    empty_hidden = result[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_bfloat16"]
    empty_tags = result[f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert empty_hidden.shape == hidden.shape
    assert torch.equal(empty_tags, result[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"])
    # Instruction rows carry the filler token, not the caption.
    positive_ids, null_ids = model.calls[0][0], model.calls[1][0]
    assert not torch.equal(positive_ids, null_ids)
    assert torch.equal(null_ids, torch.full((1, 3), _Tokenizer.pad_token_id, dtype=torch.long))


def test_null_conditioning_keeps_the_vision_prefix_intact():
    """Only the instruction is replaced; media labels and vision rows are untouched."""
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "fl2va")
    content = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    result = encoder.encode_conditioning(
        [SimpleNamespace(caption="two tokens", content=content)], include_empty=True
    )[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    empty_hidden = result[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_bfloat16"]
    assert empty_hidden.shape == hidden.shape
    # The trailing two instruction rows became filler; everything before them,
    # including both vision blocks, is byte-identical to the positive branch.
    prefix = hidden.shape[0] - 2
    (positive_ids, positive_types), (null_ids, null_types) = model.calls[0], model.calls[1]
    assert torch.equal(null_ids[0, prefix:], torch.full((2,), _Tokenizer.pad_token_id, dtype=torch.long))
    assert torch.equal(null_ids[0, :prefix], positive_ids[0, :prefix])
    assert torch.equal(null_types, positive_types)


def test_fl2va_conditioning_includes_first_last_vision_rows():
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "fl2va")
    content = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    result = encoder.encode_conditioning([SimpleNamespace(caption="prompt", content=content)])[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    tags = result[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert hidden.shape == (11, 5120)
    assert torch.equal(tags, torch.tensor([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]))
    assert torch.equal(model.last_mm_token_type_ids, torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]]))
    assert encoder.conditioning_requires_content


def test_i2va_conditioning_includes_only_first_frame_vision_rows():
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "i2va")
    item = SimpleNamespace(content=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    hidden, tags = encoder._encode_prompt("prompt", encoder._images_for_item(item))

    assert hidden.shape == (6, 5120)
    assert tags.tolist() == [1, 1, 0, 0, 0, 1]
    assert int((model.last_mm_token_type_ids == 1).sum()) == 1
    assert encoder.conditioning_requires_content


def test_ref2va_conditioning_matches_released_ordered_presentation():
    processor = _RefProcessor()
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(processor, model, torch.bfloat16, "ref2va")
    references = (
        H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (4, 4))),
        H3PreparedReference(
            kind=H3ReferenceKind.VIDEO,
            frames=np.zeros((25, 4, 4, 3), dtype=np.uint8),
            waveform=torch.zeros(2, 32),
        ),
        H3PreparedReference(kind=H3ReferenceKind.AUDIO, waveform=torch.zeros(2, 32)),
    )

    hidden, tags = encoder._encode_prompt("final prompt", references=references)

    assert hidden.shape == (tags.shape[0], 5120)
    assert processor.tokenizer.calls == [
        "<Picture 1>: ",
        "<Audio 1>: ",
        "<Video 1>: ",
        "<0.2 seconds>",
        "<1.0 seconds>",
        "<Audio 2>: ",
        "final prompt",
    ]
    assert references[1].block_timestamps == (0.25, 1.0)
    assert int((model.last_mm_token_type_ids == 1).sum()) == 1
    assert int((model.last_mm_token_type_ids == 2).sum()) == 2
    assert set(tags.tolist()) == {0, 1}


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
