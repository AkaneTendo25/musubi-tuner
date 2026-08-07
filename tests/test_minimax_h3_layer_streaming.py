from types import SimpleNamespace

import torch
import pytest
from torch import nn

from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires split CPU/CUDA placement")
def test_layer_streaming_keeps_non_linear_block_parameters_on_compute_device(monkeypatch):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=1,
    )
    model = MiniMaxH3Transformer(config)
    args = SimpleNamespace(
        block_swap_h2d_only=True,
        block_swap_ring_size=2,
        block_swap_granularity="layer",
        use_pinned_memory_for_block_swap=False,
        gradient_checkpointing=True,
    )
    swap = BlockSwapConfig.from_args(args, torch.device("cuda"), supports_backward=True)
    monkeypatch.setattr("musubi_tuner.minimax_h3.model.create_offloader", lambda *_args, **_kwargs: object())
    model.enable_block_swap(2, swap)
    model.move_to_device_except_swap_blocks(torch.device("cuda"))

    for block in model.blocks:
        assert block.norm1.weight.device.type == "cuda"
        assert block.norm2.weight.device.type == "cuda"
        assert block.attn.q_norm.weight.device.type == "cuda"
        assert block.attn.k_norm.weight.device.type == "cuda"
        assert all(module.weight.device.type == "cpu" for module in block.modules() if isinstance(module, nn.Linear))
