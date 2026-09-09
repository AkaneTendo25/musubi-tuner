from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import musubi_tuner.minimax_h3_train_network as h3_train_network
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.networks import lora_minimax_h3


def _args(*extra):
    return create_parser().parse_args(["--sdpa", *extra])


class _Module:
    def __init__(self, enabled=True):
        self.enabled = enabled


class _Overlay:
    def __init__(self, states=(True, True)):
        self.text_encoder_loras = []
        self.unet_loras = [_Module(state) for state in states]

    def set_enabled(self, enabled):
        for module in self.unet_loras:
            module.enabled = enabled

    def is_enabled(self):
        return all(module.enabled for module in self.unet_loras)


class MiniMaxH3TransformerBlock(nn.Module):
    """Minimal block with the production class name used by H3 LoRA targeting."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4, bias=False)
        self.ff = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
        self.adaln_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        return hidden_states + self.attn(hidden_states) + self.ff(hidden_states)


class _RealOverlayTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TransformerBlock()])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _accelerator():
    return SimpleNamespace(device=torch.device("cpu"), print=lambda *_a, **_k: None, unwrap_model=lambda model: model)


def _sample_parameter():
    return {
        "height": 32,
        "width": 32,
        "frame_count": 5,
        "sample_steps": 3,
        "seed": 1,
        H3_TEXT_HIDDEN_KEY: torch.zeros(1, 4),
        H3_TEXT_TOKEN_TAGS_KEY: torch.ones(1, dtype=torch.long),
    }


def test_flag_defaults_to_live_overlay_and_is_recorded():
    args = _args()
    trainer = MiniMaxH3NetworkTrainer()

    assert args.h3_overlay_training_only is False
    assert trainer.extra_metadata(args)["ss_h3_overlay_training_only"] == "False"


def test_training_only_requires_overlay_weights():
    with pytest.raises(ValueError, match="requires --h3_overlay_weights"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_args("--h3_overlay_training_only"))


@pytest.mark.parametrize("probe", ["field", "rollout"])
def test_training_only_rejects_diagnostics_that_would_mislabel_the_overlay_as_stock(tmp_path, probe):
    overlay = tmp_path / "overlay.safetensors"
    overlay.touch()
    validation = tmp_path / "validation.toml"
    validation.write_text("", encoding="utf-8")
    extra = [
        "--h3_overlay_weights",
        str(overlay),
        "--h3_overlay_training_only",
        "--validation_dataset_config",
        str(validation),
    ]
    extra += ["--h3_validation_field_probe"] if probe == "field" else ["--h3_validation_rollout_probe", "2"]

    with pytest.raises(ValueError, match="stock checkpoint"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_args(*extra))


@pytest.mark.parametrize("route", ["fl2va", "ref2va"])
def test_preview_disables_only_overlay_then_restores_mixed_module_state(monkeypatch, route):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._overlay_training_only = True
    trainer._overlay_network = _Overlay((True, False))
    trainable = _Module(True)
    observed = []

    decoded = []

    def denoise(*_args, **_kwargs):
        observed.append((trainable.enabled, tuple(module.enabled for module in trainer._overlay_network.unet_loras)))
        # A tiny concrete composition: base=1, trainable=2, overlay=4. Preview
        # must receive base+trainable while leaving the trainable arm enabled.
        value = 1 + (2 if trainable.enabled else 0) + (4 if trainer._overlay_network.is_enabled() else 0)
        return torch.tensor([float(value)]), torch.zeros(1)

    monkeypatch.setattr(h3_train_network, "denoise_fl2va" if route == "fl2va" else "denoise_ref2va", denoise)

    def decode(_video_decoder, _audio_decoder, video_latents, *_args, **_kwargs):
        decoded.append(float(video_latents.item()))
        return SimpleNamespace(video=torch.zeros(1), audio=torch.zeros(1))

    monkeypatch.setattr(h3_train_network, "decode_latents_sequentially", decode)
    transformer = SimpleNamespace(to=lambda *_args: None)
    sample = _sample_parameter()
    if route == "ref2va":
        sample["_h3_encoded_references"] = object()

    trainer._generate_sample(
        SimpleNamespace(device=torch.device("cpu")),
        transformer,
        SimpleNamespace(video_decoder=object(), audio_decoder=object()),
        sample,
    )

    assert observed == [(True, (False, False))]
    assert decoded == [3.0]
    assert tuple(module.enabled for module in trainer._overlay_network.unet_loras) == (True, False)
    assert trainable.enabled is True


@pytest.mark.parametrize("route", ["fl2va", "ref2va"])
def test_preview_error_restores_exact_overlay_module_states(monkeypatch, route):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._overlay_training_only = True
    trainer._overlay_network = _Overlay((False, True))

    def fail(*_args, **_kwargs):
        assert all(not module.enabled for module in trainer._overlay_network.unet_loras)
        raise RuntimeError("denoise failed")

    monkeypatch.setattr(h3_train_network, "denoise_fl2va" if route == "fl2va" else "denoise_ref2va", fail)
    sample = _sample_parameter()
    if route == "ref2va":
        sample["_h3_encoded_references"] = object()

    with pytest.raises(RuntimeError, match="denoise failed"):
        trainer._generate_sample(
            SimpleNamespace(device=torch.device("cpu")),
            object(),
            SimpleNamespace(video_decoder=object(), audio_decoder=object()),
            sample,
        )

    assert tuple(module.enabled for module in trainer._overlay_network.unet_loras) == (False, True)


def test_default_preview_keeps_overlay_enabled(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._overlay_network = _Overlay()

    def fail(*_args, **_kwargs):
        assert trainer._overlay_network.is_enabled()
        raise RuntimeError("stop")

    monkeypatch.setattr(h3_train_network, "denoise_fl2va", fail)
    with pytest.raises(RuntimeError, match="stop"):
        trainer._generate_sample(
            SimpleNamespace(device=torch.device("cpu")),
            object(),
            SimpleNamespace(video_decoder=object(), audio_decoder=object()),
            _sample_parameter(),
        )


def test_real_trainable_and_frozen_loras_compose_while_preview_removes_only_overlay(tmp_path):
    torch.manual_seed(17)
    transformer = _RealOverlayTransformer()
    hidden = torch.randn(3, 4)
    base_weight = transformer.blocks[0].attn.weight.detach().clone()
    baseline = transformer(hidden).detach()

    trainable = lora_minimax_h3.create_arch_network(1.0, 2, 2, None, None, transformer)
    trainable.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    for module in trainable.unet_loras:
        nn.init.normal_(module.lora_up.weight)
    trainable_only = transformer(hidden).detach()
    assert not torch.allclose(trainable_only, baseline)

    overlay_path = tmp_path / "overlay.safetensors"
    down = torch.randn(2, 4)
    up = torch.randn(4, 2)
    save_file(
        {
            "lora_unet_blocks_0_attn.lora_down.weight": down,
            "lora_unet_blocks_0_attn.lora_up.weight": up,
            "lora_unet_blocks_0_attn.alpha": torch.tensor(1.0),
        },
        str(overlay_path),
    )
    args = _args("--h3_overlay_weights", str(overlay_path), "--h3_overlay_training_only")
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    overlay = trainer.install_overlay_weights(args, _accelerator(), transformer)

    both = transformer(hidden).detach()
    assert not torch.allclose(both, trainable_only)
    with trainer._preview_overlay_disabled():
        torch.testing.assert_close(transformer(hidden), trainable_only)
        assert trainable.is_enabled()
    torch.testing.assert_close(transformer(hidden), both)

    # Both adapters were installed as live wrappers; neither installation
    # rewrote the underlying checkpoint tensor.
    torch.testing.assert_close(transformer.blocks[0].attn.weight, base_weight)
    transformer.zero_grad(set_to_none=True)
    transformer(hidden).sum().backward()
    assert all(module.lora_up.weight.grad is not None for module in trainable.unet_loras)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in overlay.parameters())
