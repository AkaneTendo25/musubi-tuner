"""Numerical equivalence of the H3 elementwise rewrites (rotary, LoRA delta add, fused modulation)."""

from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.minimax_h3.model as h3_model
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig, _apply_rotary_emb
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.networks.lora import LoRAModule, _is_power_of_two


def _reference_rotary(hidden_states, cos, sin):
    """The rotary embedding as it was written before the rewrite."""
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


def _reference_fuse_delta(self, org_forwarded, delta, scale):
    """``_fuse_delta`` as it was before the fused ``add(alpha=)`` path: multiply, then accumulate."""
    owned = False
    for factor in (self.multiplier, scale):
        if factor == 1.0:
            continue
        delta = delta.mul_(factor) if owned else delta * factor
        owned = True
    if owned and delta.is_floating_point() and torch.promote_types(delta.dtype, org_forwarded.dtype) == delta.dtype:
        return delta.add_(org_forwarded)
    return org_forwarded + delta


def _rotary_tables(sequence_length: int, rope_freq_dim: int, dtype: torch.dtype):
    torch.manual_seed(3)
    frequencies = torch.randn(sequence_length, 3 * rope_freq_dim, dtype=torch.float32)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rotary_rewrite_is_bit_identical_to_the_reference_forward_and_backward(dtype):
    torch.manual_seed(0)
    hidden = (torch.randn(2, 37, 4, 32, dtype=dtype) * 3).requires_grad_(True)
    cos, sin = _rotary_tables(37, 2, dtype)
    grad_output = torch.randn(2, 37, 4, 32, dtype=dtype)

    rewritten = _apply_rotary_emb(hidden, cos, sin)
    (rewritten_grad,) = torch.autograd.grad(rewritten, hidden, grad_output)
    reference = _reference_rotary(hidden, cos, sin)
    (reference_grad,) = torch.autograd.grad(reference, hidden, grad_output)

    assert rewritten.is_contiguous()
    assert torch.equal(rewritten, reference)
    assert torch.equal(rewritten_grad, reference_grad)


def test_power_of_two_detection():
    assert all(_is_power_of_two(value) for value in (1, 1.0, 0.5, 2.0, 1 / 32, 2**-20, 1024))
    assert not any(_is_power_of_two(value) for value in (0, -1.0, 0.3, 1 / 3, 3.0, 0.75, float("inf"), float("nan"), None))


def _lora_pair(in_dim, out_dim, *, lora_dim, alpha, multiplier, dtype, lora_dtype):
    torch.manual_seed(1)
    base = torch.nn.Linear(in_dim, out_dim, bias=False).to(dtype)
    base.requires_grad_(False)
    lora = LoRAModule("blk", base, multiplier=multiplier, lora_dim=lora_dim, alpha=alpha)
    lora.apply_to()  # drops the base from the adapter's children before the cast below
    lora.to(lora_dtype)
    return base, lora


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("multiplier, alpha, lora_dim", [(1.0, 2.0, 4), (1.0, 1.0, 32), (0.5, 4.0, 8), (2.0, 1.0, 4)])
def test_lora_delta_add_with_power_of_two_factors_is_bit_identical_to_multiply_then_add(
    monkeypatch, dtype, multiplier, alpha, lora_dim
):
    """``add(org, delta, alpha=m*s)``: the scaling is exact, the sum rounded once -- as before."""
    base, lora = _lora_pair(24, 40, lora_dim=lora_dim, alpha=alpha, multiplier=multiplier, dtype=dtype, lora_dtype=dtype)
    assert _is_power_of_two(lora.scale) and _is_power_of_two(multiplier)
    x = torch.randn(3, 11, 24, dtype=dtype)
    grad_output = torch.randn(3, 11, 40, dtype=dtype)

    def run():
        lora.zero_grad(set_to_none=True)
        out = base(x)
        out.backward(grad_output)
        return out.detach().clone(), lora.lora_down.weight.grad.clone(), lora.lora_up.weight.grad.clone()

    fused = run()
    monkeypatch.setattr(LoRAModule, "_fuse_delta", _reference_fuse_delta)
    reference = run()

    for got, expected in zip(fused, reference):
        assert torch.equal(got, expected)


def test_lora_delta_add_keeps_the_multiply_then_add_order_for_other_factors_unless_opted_in(monkeypatch):
    base, lora = _lora_pair(24, 40, lora_dim=3, alpha=1.0, multiplier=1.0, dtype=torch.bfloat16, lora_dtype=torch.bfloat16)
    assert not _is_power_of_two(lora.scale)  # 1/3
    x = torch.randn(3, 11, 24, dtype=torch.bfloat16)
    default = base(x).detach()
    lora.fused_scale_add = True
    fused = base(x).detach()
    monkeypatch.setattr(LoRAModule, "_fuse_delta", _reference_fuse_delta)
    reference = base(x).detach()

    assert torch.equal(default, reference)
    # The opted-in form rounds the scaled delta once instead of twice (CUDA; the CPU kernel may
    # round twice on its scalar tail), so it is only guaranteed to agree within bf16 rounding.
    torch.testing.assert_close(fused, reference, rtol=2**-7, atol=2**-7)


def test_lora_delta_add_falls_back_when_dtypes_differ():
    """fp32 LoRA on a bf16 base without autocast: the delta is kept in fp32 and rounded once at the end."""
    base, lora = _lora_pair(24, 40, lora_dim=4, alpha=2.0, multiplier=1.0, dtype=torch.bfloat16, lora_dtype=torch.float32)
    x = torch.randn(3, 11, 24, dtype=torch.bfloat16)
    out = base(x)
    expected = (base.forward.__self__.org_forward(x).float() + lora.lora_up(lora.lora_down(x.float())) * lora.scale).to(
        torch.bfloat16
    )
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, expected)


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


def _tiny_inputs(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    return {
        "video_hidden_states": torch.randn(1, 2, 16, dtype=dtype),
        "audio_hidden_states": torch.randn(1, 4, 8, dtype=dtype),
        "encoder_hidden_states": torch.randn(1, 3, 12, dtype=dtype),
        "timestep": torch.tensor([0.25, 0.75]),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0]),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0]),
        "position_ids": torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10,
        "video_indices": torch.tensor([7, 8]),
        "audio_indices": torch.tensor([3, 4, 5, 6]),
        "text_indices": torch.tensor([0, 1, 2]),
    }


def _tiny_lora_model(dtype: torch.dtype, *, alpha: float = 2.0, lora_dim: int = 4):
    torch.manual_seed(7)
    model = MiniMaxH3Transformer(_tiny_config()).to(dtype)
    model.requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(1.0, lora_dim, alpha, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.to(dtype)
    network.requires_grad_(True)
    return model, network


def _loss_and_grads(model, network, dtype, *, checkpointing=False):
    if checkpointing:
        model.enable_gradient_checkpointing()
    network.zero_grad(set_to_none=True)
    output = model(**_tiny_inputs(dtype))
    loss = output.video.float().square().mean() + output.audio.float().square().mean()
    loss.backward()
    return loss.detach(), [parameter.grad.clone() for parameter in network.parameters()]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_tiny_model_loss_and_gradients_are_bit_identical_to_the_pre_rewrite_block(monkeypatch, dtype, checkpointing):
    """The default path retains the pre-optimization formulas exactly."""
    model, network = _tiny_lora_model(dtype)
    loss, grads = _loss_and_grads(model, network, dtype, checkpointing=checkpointing)

    monkeypatch.setattr(h3_model, "_apply_rotary_emb", _reference_rotary)
    monkeypatch.setattr(LoRAModule, "_fuse_delta", _reference_fuse_delta)
    reference_loss, reference_grads = _loss_and_grads(model, network, dtype, checkpointing=checkpointing)

    assert torch.equal(loss, reference_loss)
    assert len(grads) == len(reference_grads) == 2 * 4 * 2  # down/up for qkv, out, fc1, fc2 in two blocks
    for got, expected in zip(grads, reference_grads):
        assert torch.equal(got, expected)


@pytest.mark.parametrize("dtype, tolerance", [(torch.bfloat16, 2**-6), (torch.float32, 1e-5)])
def test_fused_elementwise_matches_the_default_within_rounding(dtype, tolerance):
    model, network = _tiny_lora_model(dtype, alpha=1.0, lora_dim=3)  # scale 1/3: the LoRA add also fuses
    loss, grads = _loss_and_grads(model, network, dtype)

    model.enable_fused_elementwise()
    for module in network.modules():
        if isinstance(module, LoRAModule):
            module.fused_scale_add = True
    assert all(block.fused_elementwise for block in model.blocks)
    fused_loss, fused_grads = _loss_and_grads(model, network, dtype)

    torch.testing.assert_close(fused_loss, loss, rtol=tolerance, atol=tolerance)
    for got, expected in zip(fused_grads, grads):
        torch.testing.assert_close(got, expected, rtol=tolerance, atol=tolerance * expected.abs().max().clamp(min=1.0))


def test_fused_elementwise_flag_opts_the_adapters_in_at_train_start(tmp_path):
    args = create_parser().parse_args(["--sdpa", "--output_dir", str(tmp_path), "--output_name", "run", "--h3_fused_elementwise"])
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    _, network = _tiny_lora_model(torch.float32)
    assert not any(module.fused_scale_add for module in network.modules() if isinstance(module, LoRAModule))

    trainer.on_train_start(args, SimpleNamespace(is_main_process=True, device=torch.device("cpu")), network, None, None)

    assert all(module.fused_scale_add for module in network.modules() if isinstance(module, LoRAModule))
    assert trainer.extra_metadata(args)["ss_h3_fused_elementwise"] == "True"


def test_fused_elementwise_is_off_by_default(tmp_path):
    args = create_parser().parse_args(["--sdpa", "--output_dir", str(tmp_path), "--output_name", "run"])
    assert args.h3_fused_elementwise is False
    assert args.h3_compile_attention == "inline"
    assert not any(block.fused_elementwise for block in MiniMaxH3Transformer(_tiny_config()).blocks)
    assert not any(
        module.opaque_attention
        for module in MiniMaxH3Transformer(_tiny_config()).modules()
        if isinstance(module, h3_model.MiniMaxH3Attention)
    )
