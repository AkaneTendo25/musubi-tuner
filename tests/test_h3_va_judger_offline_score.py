from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3_score_va_judger import (
    SCORING_IMPLEMENTATION_VERSION,
    SCORING_PROMPT,
    SCORING_PROMPT_SHA256,
    SCORING_PROMPT_VERSION,
    NativeVAJudger,
    _load_video_audio,
    _normalize_legacy_rope_config,
    _parse_scores,
    create_parser,
    score,
)


@pytest.mark.parametrize("total", ["Total", "Summed total", "Summed totals"])
def test_native_scores_accept_equivalent_labels_and_implicit_first_video(total):
    text = "\n".join(
        [
            "(A) video 1: 8/10 - hands; video 2: 9/10 - lively",
            "(B) audio-video alignment: 8/10 - aligned; video 2: audio-video alignment: 8/10 - aligned",
            "(C) audio quality: 9/10 - clear; video 2: audio quality: 9/10 - clear",
            "(D) video quality: 9/10 - clear; video 2: video quality: 9/10 - clear",
            "(E) completeness and coherence: 9/10 - coherent; video 2: completeness and coherence: 9/10 - coherent",
            f"{total}: video 1: 43; video 2: 44",
            "<answer>video 2 is better</answer>",
        ]
    )
    scores = _parse_scores(text)
    assert scores == {"A": {"1": 8, "2": 9}, **{d: {"1": v, "2": v} for d, v in zip("BCDE", [8, 9, 9, 9])}}


def test_native_scores_preserve_explicit_reverse_order():
    text = "\n".join(f"({d}) video 2: 7/10 - fine; video 1: 6/10 - fair compared to video 2." for d in "ABCDE")
    assert _parse_scores(text) == {d: {"1": 6, "2": 7} for d in "ABCDE"}


@pytest.mark.parametrize(
    "bad",
    [
        "8/10 - first; 9/10 - second",
        "video 1: 8/10 - first",
        "video 2: 8/10 - first; video 2: 9/10 - second",
        "rationale mentions 8/10; video 2: 9/10 - second",
        "video 1: 0/10 - first; video 2: 9/10 - second",
        "video 1: 11/10 - first; video 2: 9/10 - second",
        "video 1: nan/10 - first; video 2: 9/10 - second",
    ],
)
def test_native_scores_reject_ambiguous_missing_or_invalid_values(bad):
    from musubi_tuner.minimax_h3.va_judger import VAJudgerError

    text = "(A) " + bad + "\n" + "\n".join(f"({d}) video 1: 8/10; video 2: 9/10" for d in "BCDE")
    with pytest.raises(VAJudgerError):
        _parse_scores(text)


def test_native_scores_reject_duplicate_dimension():
    from musubi_tuner.minimax_h3.va_judger import VAJudgerError

    text = "\n".join(f"({d}) video 1: 8/10; video 2: 9/10" for d in "ABCDEA")
    with pytest.raises(VAJudgerError, match="repeats"):
        _parse_scores(text)


def test_transformers_457_rope_compatibility_preserves_checkpoint_parameters():
    text = SimpleNamespace(
        rope_scaling=None,
        rope_theta=None,
        head_dim=128,
        rope_parameters={
            "type": "default",
            "rope_type": "default",
            "rope_theta": 1_000_000,
            "mrope_section": [24, 20, 20],
            "interleaved": True,
            "mrope_interleaved": True,
        },
    )
    config = SimpleNamespace(text_config=text)
    assert _normalize_legacy_rope_config(config, "4.57.6") is config
    assert text.rope_theta == 1_000_000.0
    assert text.rope_scaling == {"type": "default", "rope_type": "default", "mrope_section": [24, 20, 20]}


def test_transformers_457_rope_compatibility_rejects_non_equivalent_layout():
    text = SimpleNamespace(
        rope_scaling=None,
        rope_theta=None,
        head_dim=128,
        rope_parameters={
            "type": "default",
            "rope_type": "default",
            "rope_theta": 1_000_000,
            "mrope_section": [24, 20, 19],
            "interleaved": True,
            "mrope_interleaved": True,
        },
    )
    with pytest.raises(ValueError, match="invalid multimodal RoPE sections"):
        _normalize_legacy_rope_config(SimpleNamespace(text_config=text), "4.57.6")


def test_transformers_5_rope_config_is_not_rewritten():
    text = SimpleNamespace(rope_parameters={"rope_type": "default"})
    config = SimpleNamespace(text_config=text)
    assert _normalize_legacy_rope_config(config, "5.14.1") is config
    assert not hasattr(text, "rope_scaling")


def test_native_judge_passes_normalized_thinker_config_to_model(tmp_path, monkeypatch):
    observed = {}
    text = SimpleNamespace(
        rope_scaling=None,
        rope_theta=None,
        head_dim=128,
        rope_parameters={
            "type": "default",
            "rope_type": "default",
            "rope_theta": 1_000_000,
            "mrope_section": [24, 20, 20],
            "interleaved": True,
            "mrope_interleaved": True,
        },
    )
    thinker = SimpleNamespace(text_config=text)

    class Config:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return SimpleNamespace(thinker_config=thinker)

    class Processor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class Model:
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            observed.update(kwargs)
            return cls()

        def eval(self):
            return self

    fake = types.ModuleType("transformers")
    fake.__version__ = "4.57.6"
    fake.Qwen3OmniMoeConfig = Config
    fake.Qwen3OmniMoeProcessor = Processor
    fake.Qwen3OmniMoeThinkerForConditionalGeneration = Model
    monkeypatch.setitem(sys.modules, "transformers", fake)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = SimpleNamespace(seed=42, reward_model=model_dir, video_max_pixels=1000, device="cpu", attn_implementation="sdpa")
    NativeVAJudger(args, "prompt")
    assert observed["config"] is thinker
    assert thinker.text_config.rope_theta == 1_000_000.0
    assert thinker.text_config.rope_scaling["mrope_section"] == [24, 20, 20]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeJudge:
    instances = []

    def __init__(self, args, prompt):
        self.args, self.prompt, self.closed, self.batch_sizes = args, prompt, False, []
        self.__class__.instances.append(self)

    def predict(self, pairs):
        self.batch_sizes.append(len(pairs))
        return [
            {
                "id": pair["id"],
                "completion": "fake",
                "dimension_scores": {
                    letter: {"1": float(n + pair["sample_index_1"]), "2": float(n + pair["sample_index_2"])}
                    for n, letter in enumerate("ABCDE", 1)
                },
            }
            for pair in pairs
        ]

    def close(self):
        self.closed = True


def _case(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    candidates = []
    for index in range(3):
        media = tmp_path / f"{index}.mp4"
        media.write_bytes(f"media-{index}".encode())
        latent = tmp_path / f"{index}.pt"
        latent.write_bytes(f"latent-{index}".encode())
        candidates.append(
            {
                "path": str(media.resolve()),
                "latent_path": str(latent.resolve()),
                "seed": index,
                "media_sha256": _hash(media),
                "latent_sha256": _hash(latent),
            }
        )
    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"resume")
    manifest = {
        "version": 1,
        "round_id": "r",
        "start_step": 0,
        "prompt_cursor": 0,
        "group_count": 1,
        "behavior_sha256": "behavior",
        "semantic_fingerprint": "semantic",
        "resume_path": str(resume.resolve()),
        "resume_sha256": _hash(resume),
        "groups": [{"group_id": 4, "prompt": "caption", "candidates": candidates}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = create_parser().parse_args(
        ["--round_manifest", str(manifest_path), "--reward_model", str(model), "--pair_batch_size", "2"]
    )
    return args, manifest_path


def test_scores_all_pairs_records_native_prompt_provenance(tmp_path):
    FakeJudge.instances.clear()
    args, manifest = _case(tmp_path)
    output_path = score(args, judge_factory=FakeJudge)
    output = json.loads(output_path.read_text())

    assert FakeJudge.instances[0].prompt == SCORING_PROMPT
    assert FakeJudge.instances[0].closed
    assert FakeJudge.instances[0].batch_sizes == [2, 2, 2]
    assert output["manifest_sha256"] == _hash(manifest)
    assert output["judge"]["backend"] == "native"
    assert output["judge"]["implementation_version"] == SCORING_IMPLEMENTATION_VERSION
    assert output["judge"]["scoring_prompt_version"] == SCORING_PROMPT_VERSION
    assert output["judge"]["scoring_prompt_sha256"] == SCORING_PROMPT_SHA256
    assert SCORING_PROMPT_SHA256 == hashlib.sha256(SCORING_PROMPT.encode()).hexdigest()
    assert output["judge"]["args"]["max_frames"] == 12
    assert output["judge"]["args"]["video_max_pixels"] == 602112
    assert output["groups"][0]["raw_scores"] == [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    assert len(output["groups"][0]["raw_pair_results"]) == 6
    assert [result["id"] for result in output["groups"][0]["raw_pair_results"]] == [
        "group0000_pair0000_0_1",
        "group0000_pair0001_1_0",
        "group0000_pair0002_0_2",
        "group0000_pair0003_2_0",
        "group0000_pair0004_1_2",
        "group0000_pair0005_2_1",
    ]
    with pytest.raises(FileExistsError, match="overwrite"):
        score(args, judge_factory=FakeJudge)


def test_embedded_rubric_has_versioned_five_dimension_contract():
    assert [f"({letter})" in SCORING_PROMPT for letter in "ABCDE"] == [True] * 5
    for phrase in ("Prompt adherence", "Audio-video alignment", "Audio quality", "Video quality", "Completeness and coherence"):
        assert phrase in SCORING_PROMPT
    assert "<think>" in SCORING_PROMPT and "video 1: N/10" in SCORING_PROMPT
    assert "<answer>video 1 is better</answer>" in SCORING_PROMPT
    assert "<answer>video 2 is better</answer>" in SCORING_PROMPT


def test_pyav_audio_loader_resamples_stereo_wave_to_finite_mono(tmp_path):
    pytest.importorskip("av")
    input_rate, output_rate, duration = 8000, 16000, 0.1
    audio_path = tmp_path / "stereo.wav"
    frames = int(input_rate * duration)
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(input_rate)
        samples = bytearray()
        for index in range(frames):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / input_rate))
            samples.extend(struct.pack("<hh", value, -value))
        output.writeframes(samples)

    decoded = _load_video_audio(audio_path, output_rate)

    assert decoded.dtype.name == "float32"
    assert decoded.ndim == 1
    assert abs(decoded.size - int(output_rate * duration)) <= 32
    assert bool(torch.isfinite(torch.from_numpy(decoded)).all())


def test_pyav_audio_loader_rejects_media_without_audio(tmp_path):
    av = pytest.importorskip("av")
    video_path = tmp_path / "silent.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = stream.height = 16
        frame = av.VideoFrame(16, 16, "rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with pytest.raises(ValueError, match="no audio track"):
        _load_video_audio(video_path, 16000)


def test_hash_failure_precedes_model_construction(tmp_path):
    args, manifest_path = _case(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["groups"][0]["candidates"][0]["media_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    FakeJudge.instances.clear()
    with pytest.raises(ValueError, match="hash mismatch"):
        score(args, judge_factory=FakeJudge)
    assert not FakeJudge.instances


def test_bad_protocol_fails_closed_and_closes_judge(tmp_path):
    class WrongId(FakeJudge):
        def predict(self, pairs):
            result = super().predict(pairs)
            result[0]["id"] = "wrong"
            return result

    args, _ = _case(tmp_path)
    WrongId.instances.clear()
    with pytest.raises(RuntimeError, match="identity mismatch"):
        score(args, judge_factory=WrongId)
    assert WrongId.instances[0].closed
    assert not (tmp_path / "rewards.json").exists()


def test_native_processor_receives_av_options_and_decode_uses_only_suffix():
    observed = {}

    class Inputs(dict):
        input_ids = torch.tensor([[10, 11, 12]])

        def to(self, device=None, dtype=None):
            observed["device"] = device
            observed["dtype"] = dtype
            return self

    class Processor:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        def apply_chat_template(self, messages, **kwargs):
            observed["messages"] = messages
            observed["template_kwargs"] = kwargs
            return ["rendered"]

        def __call__(self, **kwargs):
            observed["processor_call"] = kwargs
            return Inputs()

        def batch_decode(self, ids, **kwargs):
            observed["decoded"] = ids.clone()
            return ["done"]

    class Model:
        device = "cpu"
        dtype = torch.bfloat16

        def generate(self, **kwargs):
            observed["generation"] = kwargs
            return torch.tensor([[10, 11, 12, 90, 91]])

    class Torch:
        @staticmethod
        def inference_mode():
            return torch.inference_mode()

    judge = object.__new__(NativeVAJudger)
    judge.args = create_parser().parse_args(["--round_manifest", "x", "--reward_model", "x"])
    judge.prompt, judge.processor, judge.model, judge.torch = "prompt", Processor(), Model(), Torch()
    judge.load_audio = lambda path, sampling_rate: f"audio:{path}:{sampling_rate}"
    judge.predict = NativeVAJudger.predict.__get__(judge)
    with pytest.raises(RuntimeError, match="dimensions"):
        judge.predict([{"id": "p", "prompt": "caption", "video_1": "a.mp4", "video_2": "b.mp4"}])
    assert observed["template_kwargs"] == {"add_generation_prompt": True, "tokenize": False}
    processor_call = observed["processor_call"]
    assert processor_call["text"] == ["rendered"]
    assert processor_call["videos"] == [["a.mp4", "b.mp4"]]
    assert processor_call["audio"] == ["audio:a.mp4:16000", "audio:b.mp4:16000"]
    assert processor_call["use_audio_in_video"] is True
    assert processor_call["do_sample_frames"] is True
    assert processor_call["num_frames"] == 12
    assert "fps" not in processor_call and "max_frames" not in processor_call
    assert processor_call["max_pixels"] == 602112 and processor_call["padding"] is True
    assert observed["device"] == "cpu" and observed["dtype"] == torch.bfloat16
    assert observed["decoded"].tolist() == [[90, 91]]
    videos = [item for item in observed["messages"][0][1]["content"] if item["type"] == "video"]
    assert videos == [{"type": "video", "video": "a.mp4"}, {"type": "video", "video": "b.mp4"}]
    assert observed["generation"]["use_audio_in_video"] is True
    assert observed["generation"]["use_cache"] is True
