from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from musubi_tuner.cache_text_encoder_outputs import process_text_encoder_batches
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3 import conditioning as h3_conditioning
from musubi_tuner.minimax_h3.cache import (
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_KEYFRAME_VISUALS_KEY,
    H3_MAX_CAPTION_TOKENS_KEY,
    H3_QWEN_CONTROL_VISUALS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_VISUAL_MAX_PIXELS_KEY,
    QWEN_CONTROL_ROLE,
    qwen_control_dropout_key,
    reference_variant_key,
)
from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality
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


def test_caption_token_cap_is_opt_in_and_preserves_structural_vision_rows():
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "fl2va", max_caption_tokens=1)
    content = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    result = encoder.encode_conditioning([SimpleNamespace(caption="three caption tokens", content=content)])[0]

    hidden = result[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"]
    tags = result[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert hidden.shape == (11, 5120)
    assert tags.tolist() == [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]
    assert int(result[H3_MAX_CAPTION_TOKENS_KEY]) == 1


def test_null_conditioning_keeps_the_vision_prefix_intact():
    """Only the instruction is replaced; media labels and vision rows are untouched."""
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "fl2va")
    content = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    result = encoder.encode_conditioning([SimpleNamespace(caption="two tokens", content=content)], include_empty=True)[0]

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


def test_l2va_conditioning_includes_only_last_frame_vision_rows():
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(_Processor(), model, torch.bfloat16, "l2va")
    content = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    content[-1] = 255
    images = encoder._images_for_item(SimpleNamespace(content=content))

    assert len(images) == 1
    assert np.asarray(images[0]).min() == 255
    hidden, tags = encoder._encode_prompt("prompt", images)

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


def test_ref2va_text_visual_cap_changes_only_qwen_reference_pixels():
    class RecordingImageProcessor(_ImageProcessor):
        def __init__(self):
            self.sizes = []

        def __call__(self, images, **kwargs):
            self.sizes.append([image.size for image in images])
            return super().__call__(images, **kwargs)

    class RecordingVideoProcessor(_VideoProcessor):
        def __init__(self):
            self.shapes = []

        def __call__(self, videos, **kwargs):
            self.shapes.append([video.shape for video in videos])
            return super().__call__(videos, **kwargs)

    processor = _RefProcessor()
    processor.image_processor = RecordingImageProcessor()
    processor.video_processor = RecordingVideoProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "ref2va", text_visual_max_pixels=65_536)
    image = Image.new("RGB", (640, 320))
    frames = np.zeros((25, 320, 640, 3), dtype=np.uint8)
    references = (
        H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=image),
        H3PreparedReference(kind=H3ReferenceKind.VIDEO, frames=frames),
    )

    encoder._encode_prompt("prompt", references=references)

    qwen_image_size = processor.image_processor.sizes[0][0]
    qwen_video_shape = processor.video_processor.shapes[0][0]
    assert qwen_image_size[0] * qwen_image_size[1] <= 65_536
    assert qwen_video_shape[1] * qwen_video_shape[2] <= 65_536
    assert references[0].image is image and references[0].image.size == (640, 320)
    assert references[1].frames is frames and references[1].frames.shape == (25, 320, 640, 3)


def test_ref2va_text_visual_cap_is_recorded_in_conditioning_cache(monkeypatch):
    encoder = MiniMaxH3ConditioningEncoder(_RefProcessor(), _TextModel(), torch.bfloat16, "ref2va", text_visual_max_pixels=65_536)
    references = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (64, 64))),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_references", lambda *_args, **_kwargs: references)
    item = SimpleNamespace(caption="reference", content=np.zeros((5, 4, 4, 3), dtype=np.uint8))

    cached = encoder.encode_conditioning([item])[0]

    assert int(cached[H3_TEXT_VISUAL_MAX_PIXELS_KEY]) == 65_536


def test_ref2va_omni_conditioning_accepts_text_only_presentation():
    processor = _RefProcessor()
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(processor, model, torch.bfloat16, "ref2va_omni")

    result = encoder.encode_reference_prompt("text only", ())

    assert result[H3_TEXT_HIDDEN_KEY].shape == (2, 5120)
    assert result[H3_TEXT_TOKEN_TAGS_KEY].tolist() == [1, 1]
    assert processor.tokenizer.calls == ["text only"]

    item = SimpleNamespace(
        caption="text only",
        content=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        h3_media_assets=(),
    )
    cached = encoder.encode_conditioning([item])[0]
    assert cached["varlen_mmh3_token_tags_int64"].tolist() == [1, 1]
    assert int(cached[H3_REFERENCE_IMAGE_SHORT_EDGE_KEY]) == 2048


def test_ref2va_video_conditioning_records_reference_temporal_contract(monkeypatch):
    processor = _RefProcessor()
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(processor, model, torch.bfloat16, "ref2va")
    references = (H3PreparedReference(kind=H3ReferenceKind.VIDEO, frames=np.zeros((5, 4, 4, 3), dtype=np.uint8)),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_references", lambda *_args, **_kwargs: references)
    monkeypatch.setattr(
        encoder,
        "_encode_prompt",
        lambda *_args, **_kwargs: (torch.zeros(2, 5120, dtype=torch.bfloat16), torch.ones(2, dtype=torch.long)),
    )
    item = SimpleNamespace(caption="video reference", content=np.zeros((5, 4, 4, 3), dtype=np.uint8))

    cached = encoder.encode_conditioning([item])[0]

    assert int(cached["mmh3_reference_temporal_contract"]) == 1


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


def _qwen_control_item(caption, controls, **extra):
    return SimpleNamespace(caption=caption, h3_media_assets=(), _qwen_controls=controls, **extra)


def test_t2va_qwen_controls_add_vision_spans_and_a_cache_marker(monkeypatch):
    """EXPERIMENTAL: control visuals ride the text channel, so T2VA gains vision rows."""
    processor = _RefProcessor()
    model = _TextModel()
    encoder = MiniMaxH3ConditioningEncoder(processor, model, torch.bfloat16, "t2va")
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: controls)

    cached = encoder.encode_conditioning([SimpleNamespace(caption="two tokens")])[0]

    tags = cached[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    # <Picture 1>: label, vision span, then the caption -- the controls close the
    # visual prefix and the instruction stays last.
    assert tags.tolist() == [1, 1, 0, 0, 0, 1, 1]
    assert processor.tokenizer.calls == ["<Picture 1>: ", "two tokens"]
    assert int(cached[H3_QWEN_CONTROL_VISUALS_KEY]) == 1


def test_ref2va_audio_only_references_produce_a_text_only_presentation(monkeypatch):
    # Reference audio never reaches the vision tower, so an audio-only reference
    # set presents to Qwen as labelled text alone -- no vision spans at all.
    processor = _RefProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "ref2va")
    references = (H3PreparedReference(kind=H3ReferenceKind.AUDIO, waveform=torch.zeros(2, 100)),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_references", lambda *_a, **_k: references)
    item = SimpleNamespace(caption="one", content=np.zeros((5, 4, 4, 3), dtype=np.uint8))

    cached = encoder.encode_conditioning([item])[0]

    assert processor.tokenizer.calls == ["<Audio 1>: ", "one"]
    tags = cached[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert tags.tolist() == [1, 1, 1]
    assert cached[f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16"].shape == (3, 5120)


def test_qwen_controls_compose_with_ref2va_references_and_continue_the_numbering(monkeypatch):
    processor = _RefProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "ref2va")
    references = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    controls = (
        H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),
        H3PreparedReference(kind=H3ReferenceKind.VIDEO, frames=np.zeros((4, 4, 4, 3), dtype=np.uint8), sample_fps=2.0),
    )
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_references", lambda *_a, **_k: references)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: controls)
    item = SimpleNamespace(caption="prompt", content=np.zeros((5, 4, 4, 3), dtype=np.uint8))

    cached = encoder.encode_conditioning([item])[0]

    # The XOR guard governs keyframes vs references only; controls compose with both.
    assert processor.tokenizer.calls == [
        "<Picture 1>: ",
        "<Picture 2>: ",
        "<Video 1>: ",
        "<0.2 seconds>",
        "<1.2 seconds>",
        "prompt",
    ]
    assert int(cached[H3_QWEN_CONTROL_VISUALS_KEY]) == 2


def test_qwen_controls_honour_the_text_visual_pixel_cap_without_mutating_the_source():
    class RecordingImageProcessor(_ImageProcessor):
        def __init__(self):
            self.sizes = []

        def __call__(self, images, **kwargs):
            self.sizes.append([image.size for image in images])
            return super().__call__(images, **kwargs)

    processor = _RefProcessor()
    processor.image_processor = RecordingImageProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "t2va", text_visual_max_pixels=65_536)
    image = Image.new("RGB", (640, 320))
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=image),)

    encoder._encode_prompt("prompt", qwen_controls=controls)

    presented = processor.image_processor.sizes[0][0]
    assert presented[0] * presented[1] <= 65_536
    assert controls[0].image is image and image.size == (640, 320)


def test_keyframe_reference_xor_still_fires_and_ignores_qwen_controls():
    encoder = MiniMaxH3ConditioningEncoder(_RefProcessor(), _TextModel(), torch.bfloat16, "fl2va")
    references = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)

    with pytest.raises(ValueError, match="keyframes or Ref2VA references"):
        encoder._encode_prompt("prompt", [Image.new("RGB", (8, 8))], references)

    # The same call with controls in place of references is accepted.
    hidden, tags = encoder._encode_prompt("prompt", [Image.new("RGB", (8, 8))], qwen_controls=controls)
    assert hidden.shape[0] == tags.shape[0] == 11


def test_prepare_qwen_controls_presents_video_frames_without_vae_preparation(monkeypatch, tmp_path):
    control = tmp_path / "pose.mp4"
    control.write_bytes(b"video")
    monkeypatch.setattr(
        "musubi_tuner.minimax_h3.conditioning._decode_video",
        lambda path, target_frames=None: (np.zeros((24, 4, 4, 3), dtype=np.uint8), 24.0),
    )
    item = SimpleNamespace(
        h3_media_assets=(MediaAsset(control, MediaModality.VIDEO, QWEN_CONTROL_ROLE),),
    )

    prepared = h3_conditioning.prepare_qwen_controls(item)

    # No 17n+5 landing, no soundtrack: 24 source frames at Qwen's 2 fps presentation rate.
    assert prepared[0].frames.shape == (2, 4, 4, 3)
    assert prepared[0].sample_fps == 2.0
    assert prepared[0].waveform is None


def test_qwen_control_dropout_caches_the_control_free_twin(monkeypatch):
    """EXPERIMENTAL: both presentations live in one cache, as the modality variants do."""
    processor = _RefProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "t2va")
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: controls)

    cached = encoder.encode_conditioning(
        [SimpleNamespace(caption="two tokens")],
        include_empty=True,
        include_qwen_control_dropout=True,
    )[0]

    # The control presentation keeps its vision spans; the twin is the caption alone.
    assert cached[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"].tolist() == [1, 1, 0, 0, 0, 1, 1]
    dropped_tags = cached[f"varlen_{qwen_control_dropout_key(H3_TEXT_TOKEN_TAGS_KEY)}_int64"]
    assert dropped_tags.tolist() == [1, 1]
    assert cached[f"varlen_{qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY)}_bfloat16"].shape == (2, 5120)
    # The empty branch gets a twin too, so a dropped step stays consistent across branches.
    assert cached[f"varlen_{qwen_control_dropout_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY)}_int64"].tolist() == [1, 1]
    assert int(cached[H3_QWEN_CONTROL_VISUALS_KEY]) == 1


def test_qwen_control_dropout_is_absent_without_the_flag_or_without_controls(monkeypatch):
    processor = _RefProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "t2va")
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: controls)

    without_flag = encoder.encode_conditioning([SimpleNamespace(caption="two tokens")])[0]

    assert not any(qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY) in key for key in without_flag)

    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: ())
    without_controls = encoder.encode_conditioning([SimpleNamespace(caption="two tokens")], include_qwen_control_dropout=True)[0]

    # Nothing to drop, so the item's cache is unchanged.
    assert not any(qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY) in key for key in without_controls)
    assert H3_QWEN_CONTROL_VISUALS_KEY not in without_controls


def test_qwen_control_dropout_covers_every_reference_modality_variant(monkeypatch):
    encoder = MiniMaxH3ConditioningEncoder(_RefProcessor(), _TextModel(), torch.bfloat16, "ref2va")
    references = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_references", lambda *_a, **_k: references)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: controls)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.reference_modality_variant", lambda refs, modality: refs)
    item = SimpleNamespace(
        caption="prompt",
        content=np.zeros((5, 4, 4, 3), dtype=np.uint8),
        h3_reference_modality_probabilities=(0.5, 0.5, 0.0),
    )

    cached = encoder.encode_conditioning([item], include_empty=True, include_qwen_control_dropout=True)[0]

    # Dropout composes with the variants rather than replacing them: every
    # cached presentation has a control-free twin.
    for base in (H3_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_HIDDEN_KEY):
        variant = reference_variant_key(base, "video")
        assert f"varlen_{variant}_bfloat16" in cached
        assert f"varlen_{qwen_control_dropout_key(variant)}_bfloat16" in cached
    assert f"varlen_{qwen_control_dropout_key(reference_variant_key(H3_TEXT_TOKEN_TAGS_KEY, 'video'))}_int64" in cached
    assert f"varlen_{qwen_control_dropout_key(reference_variant_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY, 'video'))}_int64" in cached


def test_t2va_keyframe_visuals_present_target_frames_and_mark_the_cache(monkeypatch):
    """EXPERIMENTAL: custom anchors regain the visibility fl2va keyframes have."""
    processor = _RefProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "t2va", keyframe_visuals=(0, 5, -1))
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: ())
    content = np.zeros((9, 4, 4, 3), dtype=np.uint8)
    item = SimpleNamespace(caption="two tokens", content=content)

    cached = encoder.encode_conditioning([item])[0]

    # Three picture spans in the listed order, then the instruction -- the same
    # placement the released FL2VA endpoint keyframes use.
    assert processor.tokenizer.calls == ["<Picture 1>: ", "<Picture 2>: ", "<Picture 3>: ", "two tokens"]
    tags = cached[f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64"]
    assert tags.tolist() == [1, 1, 0, 0, 0] * 3 + [1, 1]
    assert cached[f"{H3_KEYFRAME_VISUALS_KEY}_int64"].tolist() == [0, 5, -1]


def test_keyframe_visuals_compose_with_qwen_controls_and_lead_the_numbering(monkeypatch):
    processor = _RefProcessor()
    encoder = MiniMaxH3ConditioningEncoder(processor, _TextModel(), torch.bfloat16, "t2va", keyframe_visuals=(0, -1))
    controls = (H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8))),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: controls)
    item = SimpleNamespace(caption="prompt", content=np.zeros((4, 4, 4, 3), dtype=np.uint8))

    cached = encoder.encode_conditioning([item])[0]

    # Keyframes are target-derived and lead, exactly as they do on the FL2VA
    # route; the controls continue the <Picture N> counter and close the prefix.
    assert processor.tokenizer.calls == ["<Picture 1>: ", "<Picture 2>: ", "<Picture 3>: ", "prompt"]
    assert cached[f"{H3_KEYFRAME_VISUALS_KEY}_int64"].tolist() == [0, -1]
    assert int(cached[H3_QWEN_CONTROL_VISUALS_KEY]) == 1


def test_keyframe_visuals_validate_the_task_and_the_decoded_frame_count(monkeypatch):
    with pytest.raises(ValueError, match="--task t2va"):
        MiniMaxH3ConditioningEncoder(_RefProcessor(), _TextModel(), torch.bfloat16, "fl2va", keyframe_visuals=(0,))

    encoder = MiniMaxH3ConditioningEncoder(_RefProcessor(), _TextModel(), torch.bfloat16, "t2va", keyframe_visuals=(11,))
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_qwen_controls", lambda *_a, **_k: ())
    item = SimpleNamespace(caption="prompt", content=np.zeros((4, 4, 4, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="outside the 4 target frames"):
        encoder.encode_conditioning([item])

    with pytest.raises(ValueError, match="decoded target video"):
        encoder.encode_conditioning([SimpleNamespace(caption="prompt", content=None)])
