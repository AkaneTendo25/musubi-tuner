from argparse import Namespace

import pytest
import torch

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_ONE_FRAME_CONTROL_INDICES_KEY,
    H3_ONE_FRAME_TARGET_INDEX_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3_train_network import (
    _one_frame_sample_inputs,
    _parse_one_frame_sample_options,
    _require_one_frame_opt_in,
    create_parser,
    MiniMaxH3NetworkTrainer,
)
from musubi_tuner.training.sampling_prompts import line_to_prompt_dict


def test_one_frame_flag_is_accepted_by_the_training_parser():
    args = create_parser().parse_args(["--one_frame"])

    assert args.one_frame is True


def test_training_sample_parser_preserves_repeatable_reference_order():
    parsed = line_to_prompt_dict("portrait --f 1 --ref subject.png --ref style.png --of target_index=24")

    assert parsed["reference_path"] == ["subject.png", "style.png"]
    assert parsed["one_frame"] == "target_index=24"


def test_ref2va_training_sample_gate_accepts_explicit_one_frame_mode():
    args = create_parser().parse_args(["--sdpa", "--one_frame"])
    args.h3_training_mode = "ref2va"
    args.sample_prompts = "samples.txt"
    args.text_encoder = "text.safetensors"
    args.vae = "video.safetensors"
    args.audio_vae = "audio.safetensors"

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, (0, None)),
        ("target_index=24", (24, None)),
        ("target_index=24,control_index=0;12;48", (24, (0, 12, 48))),
    ],
)
def test_parse_one_frame_sample_options(spec, expected):
    assert _parse_one_frame_sample_options(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "target_index=-1",
        "control_index=",
        "control_index=0;-1",
        "target_index=nope",
        "unknown=1",
        "target_index=1,target_index=2",
    ],
)
def test_parse_one_frame_sample_options_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="--of|nonnegative"):
        _parse_one_frame_sample_options(spec)


def test_one_frame_sample_preserves_repeatable_condition_order():
    target, indices, controls = _one_frame_sample_inputs(
        {
            "one_frame": "target_index=24,control_index=48;0;12",
            "control_image_path": ["late.png", "first.png", "middle.png"],
        }
    )

    assert target == 24
    assert indices == (48, 0, 12)
    assert controls == ["late.png", "first.png", "middle.png"]


def test_one_frame_sample_uses_i_and_ei_as_the_first_two_control_aliases():
    _, indices, controls = _one_frame_sample_inputs(
        {
            "one_frame": "target_index=24,control_index=0;48",
            "image_path": "first.png",
            "end_image_path": "last.png",
        }
    )

    assert indices == (0, 48)
    assert controls == ["first.png", "last.png"]


def test_one_frame_sample_rejects_aliases_mixed_with_ci():
    with pytest.raises(ValueError, match="cannot combine"):
        _one_frame_sample_inputs(
            {
                "one_frame": "target_index=24,control_index=0",
                "image_path": "first.png",
                "control_image_path": ["control.png"],
            }
        )


def test_indexed_one_frame_cache_requires_explicit_opt_in():
    video = torch.zeros(1, 24, 1, 4, 4)
    batch = {"one_frame_target_index": torch.tensor([24])}

    with pytest.raises(ValueError, match="requires --one_frame"):
        _require_one_frame_opt_in(Namespace(one_frame=False), batch, video)
    _require_one_frame_opt_in(Namespace(one_frame=True), batch, video)


def test_legacy_plain_image_cache_does_not_require_one_frame_opt_in():
    _require_one_frame_opt_in(Namespace(one_frame=False), {}, torch.zeros(1, 24, 1, 4, 4))


def test_one_frame_index_metadata_rejects_a_video_target():
    with pytest.raises(ValueError, match="single-frame"):
        _require_one_frame_opt_in(
            Namespace(one_frame=True),
            {"one_frame_target_index": torch.tensor([24])},
            torch.zeros(1, 24, 2, 4, 4),
        )


def test_native_backend_runs_k3_one_frame_prompt_and_empty_backward_at_identical_times():
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=24,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    video = torch.randn(1, 24, 1, 2, 2)
    # One-frame cache contract keeps the two-frame silent audio placeholder.
    audio = torch.randn(1, 2, 6, 2)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1])],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["fl2va"])],
        H3_ONE_FRAME_TARGET_INDEX_KEY: [torch.tensor(24, dtype=torch.int64)],
        H3_ONE_FRAME_CONTROL_INDICES_KEY: [torch.tensor([0, 12, 48], dtype=torch.int64)],
        "latents_cond_000": [torch.randn(24, 1, 2, 2)],
        "latents_cond_001": [torch.randn(24, 1, 2, 2)],
        "latents_cond_002": [torch.randn(24, 1, 2, 2)],
    }
    call = (transformer, batch, video, audio, torch.tensor([0.4]), torch.tensor([0.7]))

    torch.manual_seed(123)
    prompted_plan = backend._prepare_training_forward(*call, conditioning="prompt")
    torch.manual_seed(123)
    empty_plan = backend._prepare_training_forward(*call, conditioning="empty")
    torch.testing.assert_close(prompted_plan.kwargs["position_ids"], empty_plan.kwargs["position_ids"])
    torch.testing.assert_close(prompted_plan.kwargs["timestep"], empty_plan.kwargs["timestep"])
    assert prompted_plan.kwargs["video_hidden_states"].shape[1] == 4

    prompted = backend.predict_training(*call, conditioning="prompt")
    empty = backend.predict_training(*call, conditioning="empty")
    loss = prompted.video.square().mean() + prompted.audio.square().mean()
    loss = loss + empty.video.square().mean() + empty.audio.square().mean()
    loss.backward()

    assert prompted.video.shape == empty.video.shape == video.shape
    assert prompted.audio.shape == empty.audio.shape == audio.shape
    assert torch.isfinite(loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None
