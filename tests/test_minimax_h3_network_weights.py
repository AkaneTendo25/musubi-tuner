from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.networks import lora_minimax_h3


class MiniMaxH3TransformerBlock(nn.Module):
    """Minimal block with the production class name used by H3 LoRA targeting."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4, bias=False)
        self.ff = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))

    def forward(self, hidden_states):
        return hidden_states + self.attn(hidden_states) + self.ff(hidden_states)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TransformerBlock()])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _accelerator():
    return SimpleNamespace(device=torch.device("cpu"), print=lambda *_a, **_k: None, unwrap_model=lambda model: model)


def _args(*extra):
    return create_parser().parse_args(["--sdpa", "--network_module", "networks.lora_minimax_h3", *extra])


def _comfy_adapter(path, rank=2):
    """A ComfyUI-keyed adapter on the attention projection, as published training adapters ship."""
    torch.manual_seed(3)
    save_file(
        {
            "diffusion_model.blocks.0.attn.lora_A.weight": torch.randn(rank, 4),
            "diffusion_model.blocks.0.attn.lora_B.weight": torch.randn(4, rank),
        },
        str(path),
    )


def test_network_weights_converts_comfy_keys_and_matches_the_overlay(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    _comfy_adapter(adapter)
    hidden = torch.randn(3, 4)

    overlay_model = _Transformer()
    trainer = MiniMaxH3NetworkTrainer()
    overlay_args = _args("--h3_overlay_weights", str(adapter))
    trainer.handle_model_specific_args(overlay_args)
    trainer.install_overlay_weights(overlay_args, _accelerator(), overlay_model)
    expected = overlay_model(hidden).detach()

    model = _Transformer()
    model.load_state_dict({k: v for k, v in overlay_model.state_dict().items() if "lora" not in k}, strict=False)
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    info = MiniMaxH3NetworkTrainer().load_trained_network_weights(network, str(adapter), lora_minimax_h3)

    assert not info.unexpected_keys
    torch.testing.assert_close(model(hidden), expected)


def test_network_weights_refuses_modules_this_network_does_not_target(tmp_path):
    weights = tmp_path / "wider.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_attn.lora_down.weight": torch.zeros(2, 4),
            "lora_unet_blocks_0_attn.lora_up.weight": torch.zeros(4, 2),
            "lora_unet_blocks_0_attn.alpha": torch.tensor(2.0),
            "lora_unet_token_refiner_blocks_0_attn_out_proj.lora_down.weight": torch.zeros(2, 4),
            "lora_unet_token_refiner_blocks_0_attn_out_proj.lora_up.weight": torch.zeros(4, 2),
        },
        str(weights),
    )
    model = _Transformer()
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)

    with pytest.raises(ValueError, match="match no module"):
        MiniMaxH3NetworkTrainer().load_trained_network_weights(network, str(weights), lora_minimax_h3)


def test_dim_from_weights_reads_ranks_from_network_weights(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    _comfy_adapter(adapter, rank=3)
    args = _args("--network_weights", str(adapter), "--dim_from_weights")

    network = MiniMaxH3NetworkTrainer()._build_network(args, _accelerator(), _Transformer(), None, torch.float32)

    ranks = {module.lora_dim for module in network.unet_loras}
    assert ranks == {3}


def test_dim_from_weights_without_network_weights_is_an_error():
    args = _args("--dim_from_weights")
    with pytest.raises(ValueError, match="--network_weights"):
        MiniMaxH3NetworkTrainer()._build_network(args, _accelerator(), _Transformer(), None, torch.float32)
