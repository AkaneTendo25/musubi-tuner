from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.model_loader import resolve_transformer_checkpoint, validate_transformer_checkpoint


def _tiny_config() -> MiniMaxH3TransformerConfig:
    return MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )


def test_native_h3_model_matches_released_full_checkpoint_key_count():
    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer()
    state_dict = model.state_dict()

    assert len(state_dict) == 535
    assert state_dict["blocks.0.attn.qkv_proj.weight"].shape == (21504, 5376)
    assert state_dict["blocks.49.adaln_proj.linear.weight"].shape == (96768, 2688)
    assert state_dict["token_refiner.blocks.1.mlp.fc1.weight"].shape == (28672, 5376)
    assert state_dict["rope.inv_freq"].shape == (16,)


def test_native_h3_tiny_forward_and_backward():
    torch.manual_seed(1)
    config = _tiny_config()
    model = MiniMaxH3Transformer(config)
    text_indices = torch.tensor([0, 1, 2])
    audio_indices = torch.tensor([3, 4, 5, 6])
    video_indices = torch.tensor([7, 8])
    token_tags = torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0])
    timestep_indices = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0])
    position_ids = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10

    output = model(
        video_hidden_states=torch.randn(1, 2, 16),
        audio_hidden_states=torch.randn(1, 4, 8),
        encoder_hidden_states=torch.randn(1, 3, 12),
        timestep=torch.tensor([0.25, 0.75]),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )

    assert output.video.shape == (1, 2, 16)
    assert output.audio.shape == (1, 4, 8)
    loss = output.video.square().mean() + output.audio.square().mean()
    loss.backward()
    gradient = model.blocks[0].attn.qkv_proj.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_h3_checkpoint_validator_accepts_exact_native_mixed_precision_layout(tmp_path: Path):
    config = _tiny_config()
    model = MiniMaxH3Transformer(config)
    fp32_prefixes = (
        "video_patch_proj.",
        "audio_patch_proj.",
        "time_embedder.",
        "final_layer.video_out.",
        "final_layer.audio_out.",
        "rope.",
    )
    state_dict = {
        name: tensor.detach().to(torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    checkpoint_path = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    save_file(state_dict, checkpoint_path)

    tensor_count, parameter_count = validate_transformer_checkpoint(checkpoint_path, config)

    assert tensor_count == len(state_dict)
    assert parameter_count == sum(tensor.numel() for tensor in state_dict.values())
    assert resolve_transformer_checkpoint(checkpoint_path, "fl2va") == checkpoint_path
