from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.minimax_h3.model as h3_model
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.networks import lora_minimax_h3


def _tiny_block_model(attention_mode: str):
    config = h3_model.MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
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
    torch.manual_seed(0)
    model = h3_model.MiniMaxH3Transformer(config, attention_mode=attention_mode)
    model.requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(1.0, 4, 2.0, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    return model, network


def _block_inputs(sequence_length: int = 12):
    torch.manual_seed(1)
    frequencies = torch.randn(sequence_length, 6)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return (
        torch.randn(1, sequence_length, 32, requires_grad=True),
        torch.randn(2, 16),
        torch.randint(0, 6, (sequence_length,)),
        (frequencies.cos(), frequencies.sin()),
        None,
    )


def _untraceable_flash(qkv, attn_params=None):
    """Stands in for a FlashAttention binding Dynamo cannot trace: a graph break inside the call."""
    query, key, value = qkv
    qkv.clear()
    torch._dynamo.graph_break()
    out = torch.nn.functional.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)


def test_h3_block_with_lora_compiles_to_one_graph_with_zero_breaks_on_sdpa():
    model, _ = _tiny_block_model("torch")
    explanation = torch._dynamo.explain(model.blocks[0])(*_block_inputs())
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1


def test_h3_opaque_attention_is_exactly_one_graph_break_at_the_kernel_and_matches_eager(monkeypatch):
    monkeypatch.setattr(h3_model, "musubi_attention", _untraceable_flash)
    model, _ = _tiny_block_model("flash")
    block = model.blocks[0]
    inputs = _block_inputs()
    eager = block(*inputs)

    # The break inside the (inlined) attention call is replayed at every frame above it.
    torch._dynamo.reset()
    cascade = torch._dynamo.explain(block)(*inputs)
    assert cascade.graph_break_count > 1

    model.set_compile_opaque_attention(True)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(block)(*inputs)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    assert "torch.compiler.disable" in explanation.break_reasons[0].reason
    assert all(len(graph.graph.nodes) > 20 for graph in explanation.graphs)  # two real fragments, no stranded stubs

    torch._dynamo.reset()
    compiled = torch.compile(block, backend="eager", dynamic=False)
    output = compiled(*inputs)
    assert torch.equal(output, eager)
    output.square().mean().backward()
    assert inputs[0].grad is not None


def test_h3_opaque_attention_is_the_same_eager_code_outside_compile():
    model, _ = _tiny_block_model("torch")
    inputs = _block_inputs()
    inline = model.blocks[0](*inputs)
    model.set_compile_opaque_attention(True)
    assert all(module.opaque_attention for module in model.modules() if isinstance(module, h3_model.MiniMaxH3Attention))
    assert torch.equal(model.blocks[0](*inputs), inline)
    model.set_compile_opaque_attention(False)
    assert torch.equal(model.blocks[0](*inputs), inline)


@pytest.mark.parametrize(
    "flags, expected",
    [
        (["--sdpa"], False),  # the default is inline: an existing --compile run keeps its partitioning
        (["--flash3"], False),
        (["--flash_attn"], False),
        (["--flash3", "--h3_compile_attention", "auto"], True),
        (["--flash_attn", "--h3_compile_attention", "auto"], True),
        (["--sdpa", "--h3_compile_attention", "auto"], False),
        (["--flash3", "--compile_fullgraph", "--h3_compile_attention", "auto"], False),
        (["--sdpa", "--h3_compile_attention", "opaque"], True),
        (["--flash3", "--h3_compile_attention", "inline"], False),
    ],
)
def test_compile_attention_mode_resolution(flags, expected):
    args = create_parser().parse_args([*flags, "--compile"])
    assert args.h3_compile_attention == next(
        (flags[index + 1] for index, flag in enumerate(flags) if flag == "--h3_compile_attention"), "inline"
    )
    assert MiniMaxH3NetworkTrainer._compile_opaque_attention(args) is expected


def test_opaque_attention_is_refused_under_fullgraph():
    trainer = MiniMaxH3NetworkTrainer()
    args = create_parser().parse_args(["--flash3", "--compile", "--compile_fullgraph", "--h3_compile_attention", "opaque"])
    with pytest.raises(ValueError, match="--compile_fullgraph forbids"):
        trainer.handle_model_specific_args(args)


def test_compile_transformer_sets_the_opaque_wrapper_before_compiling(monkeypatch):
    calls = []
    model, _ = _tiny_block_model("flash")
    trainer = MiniMaxH3NetworkTrainer()
    trainer.blocks_to_swap = 0
    monkeypatch.setattr(
        "musubi_tuner.minimax_h3_train_network.model_utils.compile_transformer",
        lambda args, transformer, target_blocks, disable_linear: calls.append(transformer) or transformer,
    )
    args = SimpleNamespace(h3_compile_attention="auto", sdpa=False, compile_fullgraph=False)

    trainer.compile_transformer(args, model)

    assert calls == [model]
    assert all(module.opaque_attention for module in model.modules() if isinstance(module, h3_model.MiniMaxH3Attention))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SDPA")
def test_h3_compiled_attention_keeps_cudnn_auto_dispatch(monkeypatch):
    original_sdpa_kernel = h3_model.sdpa_kernel
    calls = []

    def recorded_priority(backends, *, set_priority=False):
        calls.append((backends, set_priority))
        return original_sdpa_kernel(backends, set_priority=set_priority)

    monkeypatch.setattr(h3_model, "sdpa_kernel", recorded_priority)
    monkeypatch.setattr(h3_model, "_CUDNN_AUTO_WORK_THRESHOLD", 1)
    monkeypatch.setattr(h3_model, "_CUDNN_AUTO_MIN_SEQUENCE", 1)
    module = h3_model.MiniMaxH3Attention(128, 1, 128, 1e-5).cuda().bfloat16().requires_grad_(False)
    module.auto_dispatch = True
    compiled = torch.compile(module, backend="eager", dynamic=False, fullgraph=True)
    hidden_states = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = compiled(hidden_states)
    output.float().square().mean().backward()

    assert calls and calls[0][1] is True
    assert hidden_states.grad is not None
