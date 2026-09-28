"""Frozen blocks and parameter patterns in MiniMax H3 full fine-tuning.

``--h3_freeze_blocks`` and ``--h3_freeze_params`` switch gradients off after the
dense trainer unfreezes the transformer. Frozen tensors get no optimizer state
and no fused hook, so they must leave training bit-identical, including when
they stream through block swap and when a frozen prefix cuts the backward pass
short of the first swapped blocks.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
from transformers import Adafactor

from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3_train import (
    MiniMaxH3Trainer,
    create_parser,
    freeze_transformer_parameters,
    parse_block_spec,
)
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

# Everything upstream of the blocks: freezing it with blocks 0-2 leaves nothing
# trainable before block 3, so backward stops there.
_UPSTREAM = r"^(video_patch_proj|audio_patch_proj|condition_proj|time_embedder|token_refiner)\."


def _tiny_config(*, num_layers: int = 6) -> MiniMaxH3TransformerConfig:
    return MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=num_layers,
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


def _tiny_inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "video_hidden_states": torch.randn(1, 2, 16, generator=generator),
        "audio_hidden_states": torch.randn(1, 4, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, 3, 12, generator=generator),
        "timestep": torch.tensor([0.25, 0.75]),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0]),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0]),
        "position_ids": torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10,
        "video_indices": torch.tensor([7, 8]),
        "audio_indices": torch.tensor([3, 4, 5, 6]),
        "text_indices": torch.tensor([0, 1, 2]),
    }


def test_block_spec_parses_ranges_and_single_blocks():
    assert parse_block_spec("0-2,5") == {0, 1, 2, 5}
    assert parse_block_spec(" 7 ") == {7}
    assert parse_block_spec("3-3,1-2") == {1, 2, 3}
    for bad in ("", "1,", "a", "3-1", "-1", "1-x"):
        with pytest.raises(ValueError):
            parse_block_spec(bad)


def test_parser_accepts_freeze_options_and_validation_rejects_bad_syntax():
    args = create_parser().parse_args(["--sdpa", "--h3_freeze_blocks", "0-19", "--h3_freeze_params", "adaln_proj", "qkv"])
    assert args.h3_freeze_blocks == "0-19"
    assert args.h3_freeze_params == ["adaln_proj", "qkv"]
    MiniMaxH3Trainer()._validate_full_finetune_args(args)

    args = create_parser().parse_args(["--sdpa", "--h3_freeze_blocks", "5-2"])
    with pytest.raises(ValueError, match="invalid block range"):
        MiniMaxH3Trainer()._validate_full_finetune_args(args)
    args = create_parser().parse_args(["--sdpa", "--h3_freeze_params", "adaln_proj("])
    with pytest.raises(ValueError, match="invalid --h3_freeze_params pattern"):
        MiniMaxH3Trainer()._validate_full_finetune_args(args)


def test_freeze_selects_blocks_and_patterns_and_leaves_the_rest_trainable():
    model = MiniMaxH3Transformer(_tiny_config())
    model.requires_grad_(True)

    tensors, elements = freeze_transformer_parameters(model, "1,3-4", ["adaln_proj"])

    block = re.compile(r"^blocks\.(\d+)\.")
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    for name, parameter in model.named_parameters():
        match = block.match(name)
        expected = (match is not None and int(match.group(1)) in {1, 3, 4}) or "adaln_proj" in name
        assert (name in frozen) == expected, name
    assert tensors == len(frozen)
    assert elements == sum(p.numel() for n, p in model.named_parameters() if n in frozen)
    assert any(name.startswith("blocks.0.") for name, p in model.named_parameters() if p.requires_grad)
    assert "final_layer.adaln_proj.linear.weight" in frozen


def test_freeze_rejects_out_of_range_blocks_and_patterns_that_match_nothing():
    model = MiniMaxH3Transformer(_tiny_config())
    model.requires_grad_(True)
    with pytest.raises(ValueError, match=r"names blocks \[6, 7\], but the model has 6"):
        freeze_transformer_parameters(model, "5-7", None)
    with pytest.raises(ValueError, match="match no transformer parameter"):
        freeze_transformer_parameters(model, None, ["adaln_proj", "no_such_module"])


def test_dense_module_rejects_a_selection_that_freezes_everything():
    model = MiniMaxH3Transformer(_tiny_config(num_layers=2))
    args = create_parser().parse_args(["--sdpa", "--h3_freeze_params", "."])
    args.gradient_checkpointing = False
    with pytest.raises(ValueError, match="freeze every MiniMax H3 transformer parameter"):
        MiniMaxH3Trainer()._build_network(args, None, model, None, None)


def test_freeze_options_are_recorded_in_checkpoint_metadata(monkeypatch):
    monkeypatch.setattr("musubi_tuner.minimax_h3_train_network.MiniMaxH3NetworkTrainer.extra_metadata", lambda self, args: {})
    args = create_parser().parse_args(["--sdpa", "--h3_freeze_blocks", "0-19", "--h3_freeze_params", "adaln_proj", "fc2"])
    metadata = MiniMaxH3Trainer().extra_metadata(args)
    assert metadata["ss_h3_freeze_blocks"] == "0-19"
    assert metadata["ss_h3_freeze_params"] == "adaln_proj fc2"

    metadata = MiniMaxH3Trainer().extra_metadata(create_parser().parse_args(["--sdpa"]))
    assert "ss_h3_freeze_blocks" not in metadata and "ss_h3_freeze_params" not in metadata


def _train(model: MiniMaxH3Transformer, device: torch.device, steps: int) -> None:
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    MiniMaxH3Trainer._install_fused_optimizer(
        SimpleNamespace(fused_backward_pass=True, adafactor_triton=False),
        SimpleNamespace(sync_gradients=True, num_processes=1),
        optimizer,
    )
    # What process_batch does at the start of every step.
    finish_truncated_backward = getattr(getattr(model, "offloader", None), "finish_truncated_backward", None)
    for step in range(steps):
        if callable(finish_truncated_backward):
            finish_truncated_backward(model.blocks)
        inputs = {key: value.to(device) for key, value in _tiny_inputs(100 + step).items()}
        output = model(**inputs)
        (output.video.square().mean() + output.audio.square().mean()).backward()
    torch.cuda.synchronize(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the real block offloaders")
@pytest.mark.parametrize("ring", [True, False], ids=["trainable_ring", "backward_swap"])
@pytest.mark.parametrize(
    ("freeze_blocks", "freeze_params"),
    [("2", ["adaln_proj"]), ("0-2", [_UPSTREAM])],
    ids=["middle_block_and_adaln", "frozen_prefix_truncates_backward"],
)
def test_block_swapped_dense_training_keeps_frozen_weights_bit_exact(ring, freeze_blocks, freeze_params):
    torch.manual_seed(11)
    device = torch.device("cuda")
    config = _tiny_config(num_layers=6)
    initial = {name: tensor.clone() for name, tensor in MiniMaxH3Transformer(config).state_dict().items()}

    def build() -> MiniMaxH3Transformer:
        model = MiniMaxH3Transformer(config)
        model.load_state_dict(initial, strict=True)
        model.requires_grad_(True)
        freeze_transformer_parameters(model, freeze_blocks, freeze_params)
        model.enable_gradient_checkpointing()
        model.train()
        return model

    reference = build().to(device)
    _train(reference, device, steps=3)
    reference_state = {name: tensor.detach().cpu() for name, tensor in reference.state_dict().items()}
    frozen = {name for name, parameter in reference.named_parameters() if not parameter.requires_grad}
    trainable = {name for name, parameter in reference.named_parameters() if parameter.requires_grad}
    del reference
    torch.cuda.empty_cache()

    swapped = build()
    swap_args = SimpleNamespace(
        block_swap_trainable_ring=ring,
        use_pinned_memory_for_block_swap=ring,
        block_swap_ring_size=2,
        gradient_checkpointing=True,
        fused_backward_pass=True,
    )
    swapped.enable_block_swap(4, BlockSwapConfig.from_args(swap_args, device, supports_backward=True))
    swapped.move_to_device_except_swap_blocks(device)
    swapped.prepare_block_swap_before_forward()
    swapped.switch_block_swap_for_training()
    _train(swapped, device, steps=3)
    swapped.offload_block_swap_to_cpu()
    swapped_state = {name: tensor.detach().cpu() for name, tensor in swapped.state_dict().items()}

    assert frozen and trainable
    for name in frozen:
        assert torch.equal(reference_state[name], initial[name]), f"reference changed frozen {name}"
        assert torch.equal(swapped_state[name], initial[name]), f"block swap changed frozen {name}"
    moved = [name for name in trainable if not torch.equal(reference_state[name], initial[name])]
    assert moved, "no trainable parameter moved; the test would pass vacuously"
    trained_blocks = {int(name.split(".")[1]) for name in moved if name.startswith("blocks.")}
    assert trained_blocks == set(range(6)) - parse_block_spec(freeze_blocks)
    for name in trainable:
        torch.testing.assert_close(swapped_state[name], reference_state[name], msg=f"trainable {name} diverged under swap")
