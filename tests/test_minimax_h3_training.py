from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    guidance_consistent_prediction,
    joint_velocity_loss,
    map_sigma_between_shifts,
    prepare_joint_noisy_inputs,
    shift_sigma,
    unshift_sigma,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
import musubi_tuner.minimax_h3_train_network as h3_train_network
from musubi_tuner.networks import lora_minimax_h3


def test_h3_shift_round_trip_and_cross_modality_mapping():
    base = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    video = shift_sigma(base, 12.0)
    audio = shift_sigma(base, 3.0)

    torch.testing.assert_close(unshift_sigma(video, 12.0), base)
    torch.testing.assert_close(map_sigma_between_shifts(video, source_shift=12.0, target_shift=3.0), audio)
    assert video[0] == audio[0] == 0
    assert video[-1] == audio[-1] == 1


def test_h3_joint_noising_uses_data_ward_velocity_and_model_time():
    video = torch.tensor([[[[[2.0, 4.0]]]]])
    video_noise = torch.tensor([[[[[10.0, 20.0]]]]])
    audio = torch.tensor([[[[3.0, 6.0]]]])
    audio_noise = torch.tensor([[[[9.0, 12.0]]]])
    base_sigma = torch.tensor([0.25])
    video_sigma = shift_sigma(base_sigma, 12.0)
    audio_sigma = shift_sigma(base_sigma, 3.0)

    result = prepare_joint_noisy_inputs(video, audio, video_noise, audio_noise, video_sigma)

    torch.testing.assert_close(result.video, (1 - video_sigma.item()) * video + video_sigma.item() * video_noise)
    torch.testing.assert_close(result.audio, (1 - audio_sigma.item()) * audio + audio_sigma.item() * audio_noise)
    torch.testing.assert_close(result.video_target, video - video_noise)
    torch.testing.assert_close(result.audio_target, audio - audio_noise)
    torch.testing.assert_close(result.video_timestep, 1 - video_sigma)
    torch.testing.assert_close(result.audio_timestep, 1 - audio_sigma)


def test_guidance_consistent_prediction_reconstructs_conditional_and_stops_empty_gradient():
    scale = 4.0
    conditional_video = torch.tensor([2.0])
    conditional_audio = torch.tensor([5.0])
    unconditional_video = torch.tensor([-1.0], requires_grad=True)
    unconditional_audio = torch.tensor([1.0], requires_grad=True)
    guided_video = (unconditional_video.detach() + scale * (conditional_video - unconditional_video.detach())).requires_grad_()
    guided_audio = (unconditional_audio.detach() + scale * (conditional_audio - unconditional_audio.detach())).requires_grad_()

    reconstructed = guidance_consistent_prediction(
        H3ModelPrediction(guided_video, guided_audio),
        H3ModelPrediction(unconditional_video, unconditional_audio),
        scale,
    )

    torch.testing.assert_close(reconstructed.video, conditional_video)
    torch.testing.assert_close(reconstructed.audio, conditional_audio)
    (reconstructed.video.sum() + reconstructed.audio.sum()).backward()
    torch.testing.assert_close(guided_video.grad, torch.full_like(guided_video, 1 / scale))
    torch.testing.assert_close(guided_audio.grad, torch.full_like(guided_audio, 1 / scale))
    assert unconditional_video.grad is None
    assert unconditional_audio.grad is None


def test_joint_loss_masks_audio_padding_and_supports_both_balances():
    video = torch.zeros(1, 1, 1, 1, 2)
    audio = torch.zeros(1, 1, 1, 3)
    inputs = prepare_joint_noisy_inputs(video, audio, video.clone(), audio.clone(), torch.tensor([0.5]))
    prediction = H3ModelPrediction(
        video=torch.tensor([[[[[1.0, 3.0]]]]]),
        audio=torch.tensor([[[[2.0, 100.0, 100.0]]]]),
    )
    audio_mask = torch.tensor([[True, False, False]])

    token = joint_velocity_loss(prediction, inputs, audio_mask=audio_mask, balance="token")
    modality = joint_velocity_loss(prediction, inputs, audio_mask=audio_mask, balance="modality")

    assert token.video_elements == 2
    assert token.audio_elements == 1
    torch.testing.assert_close(token.video_loss, torch.tensor(5.0))
    torch.testing.assert_close(token.audio_loss, torch.tensor(4.0))
    torch.testing.assert_close(token.loss, torch.tensor(14.0 / 3.0))
    torch.testing.assert_close(modality.loss, torch.tensor(4.5))

    video_only = joint_velocity_loss(
        prediction,
        inputs,
        audio_mask=torch.zeros_like(audio_mask),
        balance="modality",
    )
    torch.testing.assert_close(video_only.loss, torch.tensor(5.0))
    with pytest.raises(ValueError, match="exclude every"):
        joint_velocity_loss(
            prediction,
            inputs,
            video_mask=torch.zeros(1, 1, 1, 2, dtype=torch.bool),
            audio_mask=torch.zeros_like(audio_mask),
        )


def test_h3_text_cache_contract_and_optional_empty_pair(tmp_path):
    path = tmp_path / "sample_h3_te.safetensors"
    item = ItemInfo("sample", "caption", (0, 0), (0, 0))
    item.text_encoder_output_cache_path = str(path)
    tensors = {
        "varlen_h3_text_hidden_float32": torch.zeros(3, 5120),
        "varlen_h3_text_token_tags_int64": torch.tensor([1, 1, 1]),
        "varlen_h3_empty_text_hidden_float32": torch.zeros(1, 5120),
        "varlen_h3_empty_text_token_tags_int64": torch.tensor([1]),
    }

    save_text_encoder_output_cache_minimax_h3(item, tensors)

    with safe_open(path, framework="pt") as handle:
        assert set(handle.keys()) == set(tensors)
        assert handle.metadata()["architecture"] == "minimax_h3"

    tensors.pop("varlen_h3_empty_text_token_tags_int64")
    with pytest.raises(ValueError, match="both hidden states and token tags"):
        save_text_encoder_output_cache_minimax_h3(item, tensors)


def test_standard_bucket_manager_loads_h3_joint_cache_without_shared_schema_changes(tmp_path):
    latent_path = tmp_path / "sample_h3.safetensors"
    text_path = tmp_path / "sample_h3_te.safetensors"
    item = ItemInfo("sample", "caption", (64, 64), (64, 64), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_path)
    from musubi_tuner.minimax_h3.cache import save_latent_cache_minimax_h3

    save_latent_cache_minimax_h3(
        item,
        {
            "latents_2x2x2_float32": torch.zeros(24, 2, 2, 2),
            "audio_latents_float32": torch.zeros(2, 32, 3),
            "audio_loss_mask": torch.tensor([True, True, False]),
        },
    )
    save_text_encoder_output_cache_minimax_h3(
        item,
        {
            "varlen_h3_text_hidden_float32": torch.zeros(2, 5120),
            "varlen_h3_text_token_tags_int64": torch.ones(2, dtype=torch.long),
        },
    )

    batch = BucketBatchManager({(64, 64): [item]}, batch_size=1)[0]

    assert batch["latents"].shape == (1, 24, 2, 2, 2)
    assert batch[H3_AUDIO_LATENTS_KEY].shape == (1, 2, 32, 3)
    assert batch[H3_AUDIO_LOSS_MASK_KEY].shape == (1, 3)
    assert isinstance(batch["h3_text_hidden"], list)
    assert batch["h3_text_hidden"][0].shape == (2, 5120)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4, bias=False)
        self.ff = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
        self.adaln_proj = nn.Linear(4, 4, bias=False)
        self.norm_probe = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        return hidden_states + self.attn(hidden_states) + self.ff(hidden_states)


class TinyH3Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([MiniMaxH3TransformerBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_h3_lora_targets_main_attention_and_ff_only():
    transformer = TinyH3Transformer().requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, [], transformer)
    names = {module.lora_name for module in network.unet_loras}

    assert any(name.endswith("_attn") for name in names)
    assert sum("_ff_" in name for name in names) == 2
    assert not any("adaln" in name or "norm" in name for name in names)

    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    transformer(torch.ones(1, 4)).sum().backward()
    adapter_grads = [parameter.grad for parameter in network.parameters()]
    assert any(gradient is not None and torch.isfinite(gradient).all() and bool(gradient.abs().sum()) for gradient in adapter_grads)
    assert all(parameter.grad is None for parameter in transformer.transformer_blocks[0].adaln_proj.parameters())


class _FakeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()


class _ScaleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def predict_training(
        self,
        transformer,
        batch,
        video_hidden_states,
        audio_hidden_states,
        video_timestep,
        audio_timestep,
        *,
        conditioning="prompt",
    ):
        del batch, video_timestep, audio_timestep
        self.calls.append((conditioning, torch.is_grad_enabled()))
        return H3ModelPrediction(video_hidden_states * transformer.scale, audio_hidden_states * transformer.scale)


@pytest.mark.parametrize(
    "guidance_scale,expected_calls",
    [(None, [("prompt", True)]), (4.0, [("empty", False), ("prompt", True)])],
)
def test_h3_trainer_joint_process_batch_routes_optional_guidance(guidance_scale, expected_calls):
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = guidance_scale
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    backend = _FakeBackend()
    trainer.backend = backend
    transformer = _ScaleTransformer()
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.ones(1, 3, dtype=torch.bool),
        "timesteps": [0.5],
    }
    if guidance_scale is not None:
        batch[H3_EMPTY_TEXT_HIDDEN_KEY] = [torch.zeros(1, 5120)]
        batch[H3_EMPTY_TEXT_TOKEN_TAGS_KEY] = [torch.ones(1, dtype=torch.long)]

    torch.manual_seed(0)
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    assert backend.calls == expected_calls
    assert torch.isfinite(loss)
    assert transformer.scale.grad is not None and torch.isfinite(transformer.scale.grad)
    assert set(metrics) == {"loss/video", "loss/audio", "h3/sigma_video", "h3/sigma_audio"}


def test_h3_trainer_build_dataset_routes_through_h3_adapter(monkeypatch):
    group = SimpleNamespace(num_train_items=1)
    captured = {}

    monkeypatch.setattr(h3_train_network.config_utils, "load_user_config", lambda path: {"source": path})

    def create_group(user_config, args, **kwargs):
        captured.update(user_config=user_config, args=args, kwargs=kwargs)
        return group, object()

    monkeypatch.setattr(h3_train_network, "create_h3_dataset_group", create_group)
    args = SimpleNamespace(
        dataset_config="dataset.toml",
        num_timestep_buckets=4,
        max_data_loader_n_workers=0,
    )

    trainer = MiniMaxH3NetworkTrainer()
    built_group, collator, current_epoch = trainer._build_dataset(args)

    assert built_group is group
    assert collator.dataset is group
    assert current_epoch.value == 0
    assert captured == {
        "user_config": {"source": "dataset.toml"},
        "args": args,
        "kwargs": {"training": True, "num_timestep_buckets": 4, "shared_epoch": current_epoch},
    }


class _IncompleteBlockSwapBackend:
    @staticmethod
    def get_training_transformer(mode):
        del mode
        transformer = nn.Linear(2, 2)
        transformer.enable_block_swap = lambda *args, **kwargs: None
        transformer.move_to_device_except_swap_blocks = lambda *args, **kwargs: None
        return transformer


def test_h3_trainer_rejects_incomplete_block_swap_backend(monkeypatch):
    monkeypatch.setattr(h3_train_network, "create_backend", lambda **kwargs: _IncompleteBlockSwapBackend())
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    args = SimpleNamespace(
        fp8_base=False,
        fp8_scaled=False,
        int8=False,
        allow_prequantized_fp8=False,
        blocks_to_swap=2,
        h3_training_mode="t2va",
    )

    with pytest.raises(ValueError, match="block swapping"):
        trainer.load_transformer(None, args, str(Path("model")), "torch", False, "cpu", None)


def test_h3_training_parser_defaults_to_native_t2va_contract():
    args = create_parser().parse_args([])
    assert args.network_module == "networks.lora_minimax_h3"
    assert args.h3_training_mode == "t2va"
    assert args.mixed_precision == "bf16"
    assert args.timestep_sampling == "shift"
    assert args.discrete_flow_shift == 12.0
    assert args.h3_guidance_distillation_scale is None
