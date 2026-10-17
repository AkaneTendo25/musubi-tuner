from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.cache import H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.marginal_repair import (
    guidance_recovered_target,
    interpolate_schedule,
    parse_scale_schedule,
    scale_at,
)
from musubi_tuner.minimax_h3.training import H3ModelPrediction
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser, read_config_from_file


def configured(*options):
    args = create_parser().parse_args(["--sdpa", *options])
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    return args, trainer


def test_reconstruction_formula_and_detached_teacher():
    reference = torch.tensor([[2.0, 4.0], [3.0, 5.0]], requires_grad=True)
    conditional = torch.tensor([[6.0, 8.0], [7.0, 9.0]])
    scales = torch.tensor([2.0, 4.0])
    distilled = reference.detach() + scales[:, None] * (conditional - reference.detach())
    recovered = guidance_recovered_target(distilled.requires_grad_(), reference, scales)
    torch.testing.assert_close(recovered, conditional)
    assert not recovered.requires_grad
    torch.testing.assert_close(guidance_recovered_target(distilled, reference, 1), distilled)


@pytest.mark.parametrize("spec", ["nan", "inf,2", "0.5", "2,3,4", "garbage"])
def test_invalid_scale(spec):
    with pytest.raises(ValueError):
        parse_scale_schedule(spec)


def test_schedules_and_no_extrapolation():
    sigma = torch.tensor([0.0, 0.4, 0.8, 0.95, 1.0])
    torch.testing.assert_close(interpolate_schedule((1, 3, 6), sigma), torch.tensor([1.0, 1.0, 3.0, 6.0, 6.0]))
    torch.testing.assert_close(interpolate_schedule((1, 3), sigma)[[0, 1, 3, 4]], torch.tensor([1.0, 1.0, 3.0, 3.0]))
    torch.testing.assert_close(scale_at((0, 4), sigma), (4 * sigma).clamp_min(1))
    with pytest.raises(ValueError, match="extrapolates"):
        guidance_recovered_target(torch.ones(1, 2), torch.zeros(1, 2), 4, torch.tensor([2.0]))


@pytest.mark.parametrize(
    "options,match",
    [
        (["--h3_repair_scale", "4"], "require"),
        (["--h3_repair_reference", "empty"], "requires"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_caption_dropout_rate", "0.1"], "caption_dropout"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_guidance_distillation_scale", "4"], "guidance"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_repair_sigma_weight", "0,1"], "positive"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_repair_target_multiplier", "2,2,2"], "extrapolates"),
    ],
)
def test_early_validation(options, match):
    with pytest.raises(ValueError, match=match):
        configured(*options)


def test_toml_and_metadata(tmp_path):
    path = tmp_path / "repair.toml"
    path.write_text('h3_repair_reference="empty"\nh3_repair_scale="0.86,4.24"\nh3_repair_sigma_weight=[1.0,2.0,3.0]\n')
    parser = create_parser()
    argv = ["--config_file", str(path), "--sdpa"]
    args = read_config_from_file(parser.parse_args(argv), parser, argv)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    metadata = trainer.extra_metadata(args)
    assert trainer._repair_schedule == (0.86, 4.24)
    assert args.h3_repair_sigma_weight == (1, 2, 3)
    assert metadata["ss_h3_repair_reference"] == "empty"
    assert metadata["ss_h3_repair_scale"] == "0.86,4.24"
    defaults, _ = configured()
    assert defaults.h3_repair_reference is None


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor(0.5))
        self.enabled = True
        self.events = []

    def set_enabled(self, enabled):
        self.enabled = enabled

    def switch_block_swap_for_inference(self):
        self.events.append("inference")

    def switch_block_swap_for_training(self):
        self.events.append("training")


class TinyBackend:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def predict_training(self, transformer, batch, video, audio, video_sigma, audio_sigma, *, conditioning="prompt"):
        self.calls.append((conditioning, torch.is_grad_enabled(), transformer.enabled))
        if conditioning == "empty" and self.fail:
            raise RuntimeError("teacher failure")
        offset = 4.0 if conditioning == "prompt" else 0.0
        delta = transformer.adapter if transformer.enabled else 0
        return H3ModelPrediction(
            None if video is None else torch.ones_like(video) * (offset + delta),
            None if audio is None else torch.ones_like(audio) * (offset + delta),
        )


def run_step(args, trainer, *, audio=False, swap=False, fail=False, missing=False, data=0.0):
    transformer = TinyTransformer()
    backend = TinyBackend(fail)
    trainer.backend = backend
    trainer.blocks_to_swap = 2 if swap else 0
    trainer.dit_dtype = torch.float32
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda x: x,
        autocast=nullcontext,
        print=lambda *a: None,
        backward=lambda loss: loss.backward(),
    )
    video = torch.full((1, 24, 2, 2, 2), data)
    batch = {
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 8)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }
    if missing:
        batch.pop(H3_EMPTY_TEXT_HIDDEN_KEY)
    if audio:
        from musubi_tuner.minimax_h3.cache import H3_AUDIO_LATENTS_KEY

        batch[H3_AUDIO_LATENTS_KEY] = torch.full((1, 4, 1, 4), data)
    result = trainer.process_batch(
        args,
        accelerator,
        transformer,
        transformer,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return result, transformer, backend


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_cpu_trainer_smoke_replaces_data_loss_and_updates_only_adapter(audio, swap):
    args, trainer = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    (loss, metrics), model, backend = run_step(args, trainer, audio=audio, swap=swap)
    # Frozen g=4, u=0, recovered c=1; live prediction=4.5.
    assert loss.item() == pytest.approx(3.5**2)
    loss.backward()
    assert model.adapter.grad.item() == pytest.approx(7)
    optimizer = torch.optim.SGD([model.adapter], lr=0.01)
    optimizer.step()
    assert model.adapter.item() == pytest.approx(0.43)
    assert backend.calls == [("prompt", False, False), ("empty", False, False), ("prompt", True, True)]
    assert model.events == (["inference", "training"] if swap else [])
    assert "loss/repair_video" in metrics
    args2, trainer2 = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    (changed, _), _, _ = run_step(args2, trainer2, audio=audio, data=50)
    torch.testing.assert_close(loss, changed)


def test_cpu_missing_cache_and_teacher_failure_restore_state(monkeypatch):
    import sys

    args, trainer = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    with pytest.raises(KeyError, match="cache_guidance_empty"):
        run_step(args, trainer, missing=True)
    model = TinyTransformer()
    monkeypatch.setattr(sys.modules[__name__], "TinyTransformer", lambda: model)
    with pytest.raises(RuntimeError, match="teacher failure"):
        run_step(args, trainer, swap=True, fail=True)
    assert model.enabled
    assert model.events == ["inference", "training"]


def test_cpu_av_schedule_uses_each_modalities_shifted_sigma():
    args, trainer = configured(
        "--h3_repair_reference",
        "empty",
        "--h3_repair_scale",
        "1,4",
        "--h3_loss_balance",
        "modality",
        "--h3_repair_sigma_weight",
        "2,2",
    )
    (loss, metrics), _, _ = run_step(args, trainer, audio=True)
    # t=.5, shift=12 for video and 3 for audio: sigma=12/13 and 3/4.
    video_error = (4.5 - 4 / (1 + 4 * 12 / 13)) ** 2
    audio_error = (4.5 - 4 / (1 + 4 * 3 / 4)) ** 2
    assert metrics["loss/repair_video"] == pytest.approx(2 * video_error)
    assert metrics["loss/repair_audio"] == pytest.approx(2 * audio_error)
    assert loss.item() == pytest.approx(video_error + audio_error)


def test_default_off_keeps_single_forward():
    args, trainer = configured()
    (_, metrics), _, backend = run_step(args, trainer)
    assert backend.calls == [("prompt", True, True)]
    assert "loss/repair_video" not in metrics


def test_cpu_native_transformer_real_lora_optimizer_smoke():
    from musubi_tuner.minimax_h3.cache import (
        H3_CONDITIONING_TASK_IDS,
        H3_CONDITIONING_TASK_KEY,
        H3_TEXT_HIDDEN_KEY,
        H3_TEXT_TOKEN_TAGS_KEY,
    )
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
    from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
    from musubi_tuner.networks import lora_minimax_h3

    torch.manual_seed(42)
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    model = MiniMaxH3Transformer(config).float().requires_grad_(False)
    original = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    args, trainer = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    trainer.dit_dtype = torch.float32
    backend = _NativeTrainingBackend(model, mode="fl2va")
    trainer.backend = backend
    teacher_grad = []
    original_predict = backend.predict_training

    def observed_predict(*values, **kwargs):
        prediction = original_predict(*values, **kwargs)
        teacher_grad.append(prediction.video.requires_grad)
        return prediction

    backend.predict_training = observed_predict
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda x: x,
        autocast=nullcontext,
        print=lambda *a: None,
        backward=lambda loss: loss.backward(),
    )
    video = torch.randn(1, 4, 2, 2, 2)
    batch = {
        "timesteps": [0.5],
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.randn(1, 8)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    loss, _ = trainer.process_batch(
        args, accelerator, model, network, batch, video, torch.randn_like(video), None, torch.float32, torch.float32, None, 0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert teacher_grad == [False, False, True]
    assert all(parameter.grad is None for parameter, _ in original)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in trainable)
    optimizer.step()
    assert any(not torch.equal(parameter, saved) for parameter, saved in zip(trainable, before))
    assert all(torch.equal(parameter, saved) for parameter, saved in original)
