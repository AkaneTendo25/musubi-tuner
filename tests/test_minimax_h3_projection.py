import pytest
import torch

from musubi_tuner.minimax_h3.training import H3ModelPrediction, joint_prediction_loss, projection_guidance_target
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


def test_projection_target_is_per_sample_modality_and_supervised_mask() -> None:
    target_video = torch.tensor([[5.0, 7.0, 100.0], [2.0, -3.0, 8.0]])
    prompt_video = torch.tensor([[3.0, 4.0, 9.0], [4.0, 1.0, 5.0]])
    empty_video = torch.tensor([[1.0, 1.0, -7.0], [1.0, 2.0, 1.0]])
    target_audio = torch.tensor([[4.0, 8.0], [3.0, 7.0]])
    prompt_audio = torch.tensor([[2.0, 3.0], [5.0, 2.0]])
    empty_audio = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
    video_mask = torch.tensor([[1.0, 0.5, 0.0], [0.25, 1.0, 0.0]])
    audio_mask = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    projected = projection_guidance_target(
        H3ModelPrediction(target_video, target_audio),
        H3ModelPrediction(prompt_video, prompt_audio),
        H3ModelPrediction(empty_video, empty_audio),
        1.5,
        audio_scale=2.0,
        video_mask=video_mask,
        audio_mask=audio_mask,
    )

    for value, prompt, empty, mask, scale in (
        (projected.video, prompt_video, empty_video, video_mask, 1.5),
        (projected.audio, prompt_audio, empty_audio, audio_mask, 2.0),
    ):
        field = prompt - empty
        perpendicular = (value - prompt) / scale
        weighted_dot = (field * perpendicular * mask).sum(dim=1)
        torch.testing.assert_close(weighted_dot, torch.zeros_like(weighted_dot), atol=2e-5, rtol=0)


def test_projection_target_zero_field_and_zero_mask_leave_increment() -> None:
    target = torch.tensor([[4.0, -2.0]])
    base = torch.tensor([[1.0, 3.0]])
    expected = base + 2.0 * (target - base)
    zero_field = projection_guidance_target(
        H3ModelPrediction(target, None),
        H3ModelPrediction(base, None),
        H3ModelPrediction(base, None),
        2.0,
    )
    zero_mask = projection_guidance_target(
        H3ModelPrediction(target, None),
        H3ModelPrediction(torch.tensor([[2.0, 5.0]]), None),
        H3ModelPrediction(base, None),
        2.0,
        video_mask=torch.zeros_like(target),
    )
    torch.testing.assert_close(zero_field.video, expected)
    torch.testing.assert_close(zero_mask.video, torch.tensor([[2.0, 5.0]]) + 2.0 * (target - base))


def test_projection_widens_bfloat16_operands_before_differences() -> None:
    prompt = torch.tensor([[-314.0, 124.5]], dtype=torch.bfloat16)
    empty = torch.tensor([[-332.0, -408.0]], dtype=torch.bfloat16)
    target = torch.tensor([[-113.5, -442.0]], dtype=torch.bfloat16)
    projected = projection_guidance_target(
        H3ModelPrediction(target, None),
        H3ModelPrediction(prompt, None),
        H3ModelPrediction(empty, None),
        1.5,
    ).video

    field = prompt.float() - empty.float()
    increment = target.float() - empty.float()
    coefficient = (field * increment).sum(dim=1) / field.square().sum(dim=1)
    expected = (prompt.float() + 1.5 * (increment - coefficient[:, None] * field)).to(torch.bfloat16)
    naive_field = prompt - empty
    naive_increment = target - empty
    naive_coefficient = (naive_field * naive_increment).sum(dim=1) / naive_field.square().sum(dim=1)
    naive = prompt + 1.5 * (naive_increment - naive_coefficient[:, None] * naive_field)

    torch.testing.assert_close(projected, expected)
    assert not torch.equal(projected, naive)


def test_projection_overlay_context_restores_mixed_state_after_error() -> None:
    class Module:
        def __init__(self, enabled: bool):
            self.enabled = enabled

    class Overlay:
        def __init__(self):
            self.text_encoder_loras = []
            self.unet_loras = [Module(True), Module(False)]

        def set_enabled(self, enabled: bool) -> None:
            for module in self.unet_loras:
                module.enabled = enabled

    trainer = MiniMaxH3NetworkTrainer()
    trainer._overlay_network = Overlay()
    with pytest.raises(RuntimeError, match="teacher failed"):
        with trainer._overlay_disabled(True):
            assert [module.enabled for module in trainer._overlay_network.unet_loras] == [False, False]
            raise RuntimeError("teacher failed")
    assert [module.enabled for module in trainer._overlay_network.unet_loras] == [True, False]


def test_projection_loss_replaces_instead_of_adding_ordinary_loss() -> None:
    raw = H3ModelPrediction(torch.tensor([[1.0, 2.0]], requires_grad=True), None)
    data = H3ModelPrediction(torch.tensor([[4.0, 5.0]]), None)
    prompt = H3ModelPrediction(torch.tensor([[2.0, 4.0]]), None)
    empty = H3ModelPrediction(torch.tensor([[0.0, 1.0]]), None)
    target = projection_guidance_target(data, prompt, empty, 1.25)
    projected = joint_prediction_loss(raw, target).loss
    ordinary = joint_prediction_loss(raw, data).loss
    assert projected.item() != pytest.approx(ordinary.item())
    projected.backward()
    expected_grad = 2 * (raw.video.detach() - target.video) / raw.video.numel()
    torch.testing.assert_close(raw.video.grad, expected_grad)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--h3_projection_scale", "nan"], "finite and greater than 0"),
        (["--h3_projection_scale", "0"], "finite and greater than 0"),
        (
            ["--h3_projection_scale", "1.5", "--h3_guidance_distillation_scale", "4"],
            "cannot be combined with guidance distillation",
        ),
    ],
)
def test_projection_rejects_invalid_or_replacement_combinations(extra: list[str], message: str) -> None:
    args = create_parser().parse_args(["--sdpa", *extra])
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_projection_defaults_off_and_accepts_additive_anchors() -> None:
    defaults = create_parser().parse_args(["--sdpa"])
    assert defaults.h3_projection_scale is None

    standalone = create_parser().parse_args(["--sdpa", "--h3_projection_scale", "1.5", "--blocks_to_swap", "1"])
    MiniMaxH3NetworkTrainer().handle_model_specific_args(standalone)

    args = create_parser().parse_args(
        [
            "--sdpa",
            "--h3_projection_scale",
            "1.5",
            "--h3_base_preservation_loss_weight",
            "0.25",
            "--h3_guidance_null_anchor_weight",
            "0.5",
            "--h3_field_preservation_weight",
            "0.1",
        ]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
