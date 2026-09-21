from types import SimpleNamespace

import numpy as np
import pytest
import torch

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_KEY,
    H3_TEACHER_CONDITION_IDS,
    H3_TEACHER_CONDITIONS_KEY,
    H3_TEACHER_FINGERPRINT_CACHE_KEY,
    H3_TEACHER_HIDDEN_KEY,
    H3_TEACHER_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder
from musubi_tuner.minimax_h3.references import H3PreparedReference, H3ReferenceKind
from musubi_tuner.minimax_h3.teacher_presentations import ref_teacher_caption, subject_reference_caption


class _Item:
    item_key = "test-item"


def _payload():
    return {
        f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16": torch.zeros((2, 5120), dtype=torch.bfloat16),
        f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.zeros(2, dtype=torch.long),
        H3_CONDITIONING_TASK_KEY: torch.tensor(0, dtype=torch.long),
        f"varlen_{H3_TEACHER_HIDDEN_KEY}_bfloat16": torch.zeros((3, 5120), dtype=torch.bfloat16),
        f"varlen_{H3_TEACHER_TOKEN_TAGS_KEY}_int64": torch.zeros(3, dtype=torch.long),
        H3_TEACHER_CONDITIONS_KEY: torch.tensor(H3_TEACHER_CONDITION_IDS["first,last"], dtype=torch.long),
        H3_TEACHER_FINGERPRINT_CACHE_KEY: torch.zeros(32, dtype=torch.uint8),
    }


def test_teacher_pair_and_identity_are_required(monkeypatch):
    monkeypatch.setattr("musubi_tuner.minimax_h3.cache.save_text_encoder_output_cache_common", lambda *args, **kwargs: None)
    save_text_encoder_output_cache_minimax_h3(_Item(), _payload())
    for missing in (H3_TEACHER_CONDITIONS_KEY, f"varlen_{H3_TEACHER_TOKEN_TAGS_KEY}_int64"):
        payload = _payload()
        del payload[missing]
        with pytest.raises(ValueError, match="teacher conditioning cache"):
            save_text_encoder_output_cache_minimax_h3(_Item(), payload)


def test_teacher_caption_declarations():
    assert "<Audio 1>: fully_copy" in ref_teacher_caption("A scene")
    assert "<Audio 1>" not in ref_teacher_caption("A scene", has_audio=False)
    caption = subject_reference_caption("A portrait", 2, still_image=True)
    assert "<Picture 2>" in caption
    assert "A portrait" in caption
    with pytest.raises(ValueError):
        subject_reference_caption("", 0, still_image=False)


def _encoder_spy():
    encoder = object.__new__(MiniMaxH3ConditioningEncoder)
    encoder.task = "t2va"
    encoder.output_dtype = torch.bfloat16
    encoder.reference_image_short_edge = 2048
    encoder.reference_image_size_mode = "short_edge"
    encoder.reference_image_max_pixels = 0
    encoder.reference_video_short_edge = 768
    encoder.reference_video_max_pixels = 768 * 1344
    encoder.reference_video_fps = 0.0
    calls = []

    def encode(prompt, images=None, references=None):
        calls.append((prompt, images, references))
        return torch.zeros((3, 5120), dtype=torch.bfloat16), torch.zeros(3, dtype=torch.long)

    encoder._encode_prompt = encode
    return encoder, calls


def test_first_last_teacher_uses_crop_endpoints_and_authored_caption():
    encoder, calls = _encoder_spy()
    content = np.stack([np.full((8, 8, 3), value, dtype=np.uint8) for value in (4, 8, 12)])
    item = SimpleNamespace(caption="student", h3_teacher_caption="authored", content=content)
    tensors = encoder.encode_teacher(item, "first,last")
    prompt, images, references = calls[0]
    assert prompt == "authored" and references is None
    assert [int(np.asarray(image)[0, 0, 0]) for image in images] == [4, 12]
    assert int(tensors[H3_TEACHER_CONDITIONS_KEY]) == 1
    assert tensors[f"varlen_{H3_TEACHER_HIDDEN_KEY}_bfloat16"].shape == (3, 5120)


@pytest.mark.parametrize("target_mode, has_audio", [("av", True), ("video", False)])
def test_ref_teacher_uses_target_crop_video_and_conditional_audio(target_mode, has_audio):
    encoder, calls = _encoder_spy()
    content = np.ones((5, 8, 8, 3), dtype=np.uint8)
    item = SimpleNamespace(
        caption="student", h3_teacher_caption=None, content=content, h3_target_mode=target_mode, h3_target_fps=24
    )
    tensors = encoder.encode_teacher(item, "ref")
    prompt, images, references = calls[0]
    assert images is None and len(references) == 1
    assert references[0].kind is H3ReferenceKind.VIDEO
    np.testing.assert_array_equal(references[0].frames, content)
    assert references[0].has_audio is has_audio
    assert ("<Audio 1>" in prompt) is has_audio
    assert int(tensors[H3_TEACHER_CONDITIONS_KEY]) == 2


def test_subject_ref_accepts_one_frame_and_keeps_student_caption(monkeypatch):
    encoder, calls = _encoder_spy()
    references = (H3PreparedReference(kind=H3ReferenceKind.IMAGE),)
    monkeypatch.setattr("musubi_tuner.minimax_h3.conditioning.prepare_references", lambda *args: references)
    item = SimpleNamespace(caption="student", h3_teacher_caption=None, h3_one_frame=True)
    tensors = encoder.encode_teacher(item, "subject_ref")
    prompt, images, passed = calls[0]
    assert images is None and passed is references
    assert "still image" in prompt
    assert "student" in prompt and "<Picture 1>" in prompt
    assert int(tensors[H3_TEACHER_CONDITIONS_KEY]) == 3
