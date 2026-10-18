import pytest
import torch

from musubi_tuner.minimax_h3.training import H3ModelPrediction
from musubi_tuner.minimax_h3_train_network import (
    _field_preservation_active,
    MiniMaxH3NetworkTrainer,
    create_parser,
    guidance_field_preservation_loss,
)


def test_field_preservation_grades_both_adapted_branches_with_joint_masks() -> None:
    prompt_video = torch.tensor([[4.0, 7.0, 99.0]], requires_grad=True)
    empty_video = torch.tensor([[1.0, 2.0, -99.0]], requires_grad=True)
    prompt_audio = torch.tensor([[5.0, 8.0]], requires_grad=True)
    empty_audio = torch.tensor([[2.0, 3.0]], requires_grad=True)
    base_prompt_video = torch.tensor([[3.0, 5.0, -40.0]], requires_grad=True)
    base_empty_video = torch.tensor([[1.0, 1.0, 40.0]], requires_grad=True)
    base_prompt_audio = torch.tensor([[4.0, 6.0]], requires_grad=True)
    base_empty_audio = torch.tensor([[2.0, 2.0]], requires_grad=True)

    result = guidance_field_preservation_loss(
        H3ModelPrediction(prompt_video, prompt_audio),
        H3ModelPrediction(empty_video, empty_audio),
        H3ModelPrediction(base_prompt_video, base_prompt_audio),
        H3ModelPrediction(base_empty_video, base_empty_audio),
        video_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        audio_mask=torch.tensor([[1.0, 1.0]]),
        balance="modality",
        mask_normalization="weighted",
    )
    result.loss.backward()

    assert result.loss.item() == pytest.approx(1.0)
    torch.testing.assert_close(prompt_video.grad, -empty_video.grad)
    torch.testing.assert_close(prompt_audio.grad, -empty_audio.grad)
    assert prompt_video.grad[0, 2] == 0
    assert base_prompt_video.grad is None and base_empty_video.grad is None
    assert base_prompt_audio.grad is None and base_empty_audio.grad is None


def test_field_preservation_helper_defaults_match_joint_loss_defaults() -> None:
    prompt = torch.tensor([[3.0]], requires_grad=True)
    empty = torch.tensor([[1.0]], requires_grad=True)
    result = guidance_field_preservation_loss(
        H3ModelPrediction(prompt, None),
        H3ModelPrediction(empty, None),
        H3ModelPrediction(torch.tensor([[2.0]]), None),
        H3ModelPrediction(torch.tensor([[1.0]]), None),
    )
    assert result.loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weight", "steps", "conditioning", "global_step", "expected"),
    [
        (0.0, 0, "prompt", 0, False),
        (0.1, 0, "prompt", 100_000, True),
        (0.1, 2, "prompt", 0, True),
        (0.1, 2, "prompt", 1, True),
        (0.1, 2, "prompt", 2, False),
        (0.1, 2, "empty", 0, False),
    ],
)
def test_field_preservation_active_window(weight: float, steps: int, conditioning: str, global_step: int, expected: bool) -> None:
    args = type(
        "Args",
        (),
        {
            "h3_field_preservation_weight": weight,
            "h3_field_preservation_steps": steps,
        },
    )()
    assert _field_preservation_active(args, conditioning, global_step) is expected


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--h3_field_preservation_weight", "nan"], "finite and non-negative"),
        (["--h3_field_preservation_steps", "2"], "requires --h3_field_preservation_weight"),
        (
            ["--h3_field_preservation_weight", "0.1", "--h3_adapter_prompt_only"],
            "both adapted prompt and empty",
        ),
        (
            [
                "--h3_field_preservation_weight",
                "0.1",
                "--gradient_checkpointing",
                "--blocks_to_swap",
                "1",
                "--fused_backward_pass",
                "--optimizer_type",
                "Adafactor",
                "--max_grad_norm",
                "0",
            ],
            "cannot use --fused_backward_pass",
        ),
        (
            [
                "--h3_guidance_null_anchor_weight",
                "0.5",
                "--blocks_to_swap",
                "1",
                "--fused_backward_pass",
                "--optimizer_type",
                "Adafactor",
                "--max_grad_norm",
                "0",
            ],
            "null_anchor_weight with classic block swap stages its backward",
        ),
    ],
)
def test_field_preservation_rejects_invalid_configuration(extra: list[str], message: str) -> None:
    args = create_parser().parse_args(["--sdpa", *extra])
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h2d_only_swap_does_not_reject_fused_null_anchor() -> None:
    args = create_parser().parse_args(
        [
            "--sdpa",
            "--h3_guidance_null_anchor_weight",
            "0.5",
            "--blocks_to_swap",
            "1",
            "--block_swap_h2d_only",
            "--use_pinned_memory_for_block_swap",
            "--fused_backward_pass",
            "--optimizer_type",
            "Adafactor",
            "--max_grad_norm",
            "0",
        ]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_field_preservation_parser_defaults_off() -> None:
    args = create_parser().parse_args(["--sdpa"])
    assert args.h3_field_preservation_weight == 0.0
    assert args.h3_field_preservation_steps == 0
