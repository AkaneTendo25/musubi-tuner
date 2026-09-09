from pathlib import Path

import pytest
import torch

from musubi_tuner.minimax_h3.architecture import temporal_shape
from musubi_tuner.minimax_h3.one_frame import fl_condition_entries, parse_one_frame_options
from musubi_tuner.minimax_h3.packing import build_t2va_packed_sequence
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Guide, H3Reference, ReferenceKind, ReferenceRole
import musubi_tuner.minimax_h3_generate_video as generate_cli
from musubi_tuner.minimax_h3_generate_video import apply_prompt_overrides, create_parser, parse_prompt_line, request_from_args


def test_one_frame_options_and_ordered_controls() -> None:
    assert parse_one_frame_options(None) == (0, None)
    assert parse_one_frame_options("target_index=24,control_index=0;48;96") == (24, (0, 48, 96))
    entries = fl_condition_entries(frame_count=1, condition_images=("a.png", "b.png", "c.png"))
    assert entries == (
        ("cond_000", Path("a.png")),
        ("cond_001", Path("b.png")),
        ("cond_002", Path("c.png")),
    )


@pytest.mark.parametrize("spec", ["target_index=-1", "control_index=0;-2", "unknown=1", "target_index=1,target_index=2"])
def test_one_frame_options_reject_invalid_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_one_frame_options(spec)


def test_true_one_frame_shape_and_explicit_rope_times() -> None:
    assert temporal_shape(1).video_latent_frames == 1
    assert temporal_shape(1).audio_latent_frames == 2
    tags = torch.tensor([1, 1, 1], dtype=torch.long)
    layout = build_t2va_packed_sequence(
        tags,
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        one_frame_target_index=24,
        one_frame_control_indices=(0, 48),
    )
    rows_per_frame = 4
    cursor = len(tags)
    assert torch.all(layout.position_ids[cursor : cursor + rows_per_frame, 0] == len(tags))
    assert torch.all(layout.position_ids[cursor + rows_per_frame : cursor + 2 * rows_per_frame, 0] == len(tags) + 80)
    assert torch.all(layout.position_ids[-rows_per_frame:, 0] == len(tags) + 40)
    audio_start = len(tags) + 2 * rows_per_frame
    assert torch.all(layout.position_ids[audio_start : audio_start + 4, 0] >= len(tags) + 40)


def test_request_requires_one_control_index_per_image() -> None:
    with pytest.raises(ValueError, match="one control index per image"):
        H3GenerationRequest(
            "prompt",
            Path("out.png"),
            frame_count_override=1,
            condition_images=(Path("a.png"), Path("b.png")),
            one_frame_target_index=24,
            one_frame_control_indices=(0,),
        )


def test_frame_count_one_is_authoritative_and_prompt_lines_preserve_control_order(tmp_path: Path) -> None:
    overrides = parse_prompt_line("caption --f 1 --ci a.png --ci b.png --of target_index=24,control_index=0;48 --o still.png")
    args = create_parser().parse_args(["--model", "model", "--output", str(tmp_path)])
    item = apply_prompt_overrides(args, overrides, 0)
    request = request_from_args(item)
    assert request.frame_count_override == 1
    assert request.one_frame_target_index == 24
    assert request.one_frame_control_indices == (0, 48)
    assert request.condition_images == (Path("a.png"), Path("b.png"))
    assert request.output == tmp_path / "still.png"


def test_one_frame_options_are_rejected_for_video() -> None:
    args = create_parser().parse_args(
        ["--model", "model", "--prompt", "caption", "--output", "out.mp4", "--one_frame", "target_index=1"]
    )
    with pytest.raises(ValueError, match="require --frame_count 1"):
        request_from_args(args)


def test_one_frame_rejects_keyframes_guides_and_mixed_reference_controls() -> None:
    keyframe = H3Reference(Path("key.png"), ReferenceKind.IMAGE, ReferenceRole.KEYFRAME, latent_index=0)
    with pytest.raises(ValueError, match="keyframes or guides"):
        H3GenerationRequest("prompt", Path("out.png"), references=(keyframe,), frame_count_override=1, one_frame_target_index=0)
    with pytest.raises(ValueError, match="keyframes or guides"):
        H3GenerationRequest(
            "prompt",
            Path("out.png"),
            frame_count_override=1,
            guides=(H3Guide(0, image=Path("guide.png")),),
            one_frame_target_index=0,
        )
    endpoint = H3Reference(Path("control.png"), ReferenceKind.IMAGE, ReferenceRole.FIRST_FRAME)
    reference = H3Reference(Path("reference.png"), ReferenceKind.IMAGE)
    with pytest.raises(ValueError, match="cannot mix FL2VA controls"):
        H3GenerationRequest(
            "prompt",
            Path("out.png"),
            references=(endpoint, reference),
            frame_count_override=1,
            one_frame_target_index=0,
            one_frame_control_indices=(0,),
        )


def test_from_file_honors_each_lines_runtime_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "first --f 1 --w 640 --h 384 --s 11\nsecond --f 1 --w 768 --h 512 --s 17\n",
        encoding="utf-8",
    )
    observed: list[tuple[int, int, int]] = []

    class FakeGenerator:
        def __init__(self, output: Path) -> None:
            self.output = output

        def generate(self, request: H3GenerationRequest) -> None:
            request.output.write_bytes(b"image")

    def fake_create_generator(**kwargs):
        observed.append((kwargs["width"], kwargs["height"], kwargs["num_inference_steps"]))
        return FakeGenerator(kwargs["request"].output)

    monkeypatch.setattr(generate_cli, "create_generator", fake_create_generator)
    generate_cli.main(
        [
            "--model",
            "model",
            "--text_encoder",
            "text-encoder",
            "--vae",
            "video-vae",
            "--from_file",
            str(prompts),
            "--output",
            str(tmp_path / "outputs"),
        ]
    )
    assert observed == [(640, 384, 11), (768, 512, 17)]
