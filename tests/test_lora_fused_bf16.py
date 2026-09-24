"""Tests for the opt-in addmm-epilogue fused bf16 LoRA path (--h3_lora_fused_bf16).

The flag flips `LoRAModule.lora_fused_bf16`; gated modules compute
`out = x @ W^T + ((x @ A^T) * scale) @ B^T` via `_FusedLoraLinearBF16`, which rides the
base GEMM's cuBLAS epilogue (`addmm(delta, x, W^T)`) instead of a separate full-width
add. Unsupported configurations (dropouts, split_dims, bias, nora=forward, non-bf16,
non-Linear bases, compiled regions) fall back to the unfused path silently.
"""

import pytest
import torch
import torch.nn as nn

from musubi_tuner.networks import lora as lora_mod
from musubi_tuner.networks.lora import LoRAModule


def _make_lora(in_dim=64, out_dim=48, lora_dim=8, device="cpu", dtype=torch.float32, **kwargs):
    torch.manual_seed(0)
    base = nn.Linear(in_dim, out_dim, bias=kwargs.pop("bias", False)).to(device=device, dtype=dtype)
    lora = LoRAModule("blk", base, multiplier=kwargs.pop("multiplier", 1.0), lora_dim=lora_dim, alpha=lora_dim, **kwargs)
    for mod in (lora.lora_up, lora.lora_down):
        for sub in mod if isinstance(mod, nn.ModuleList) else [mod]:
            nn.init.normal_(sub.weight, std=0.05)
    lora.to(device=device, dtype=dtype)
    lora.apply_to()
    return base, lora


def _apply_spy(monkeypatch):
    calls = []
    original = lora_mod._FusedLoraLinearBF16.apply

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lora_mod._FusedLoraLinearBF16, "apply", staticmethod(spy))
    return calls


class TestFallbackPredicates:
    """Every unsupported configuration must silently take the unfused path."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"dropout": 0.1},
            {"rank_dropout": 0.1},
            {"module_dropout": 0.1},
            {"split_dims": [24, 24]},
            {"bias": True},
            {"nora": "forward"},
        ],
    )
    def test_cpu_never_fuses(self, monkeypatch, kwargs):
        # CPU tensors fail the `x.is_cuda` gate regardless of other flags.
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(**kwargs)
        lora.lora_fused_bf16 = True
        lora.train()
        x = torch.randn(2, 5, 64)
        base(x)
        assert not calls

    def test_flag_off_identity(self):
        torch.manual_seed(0)
        base, lora = _make_lora()
        x = torch.randn(2, 5, 64)
        lora.eval()
        out_ref = base(x)
        # Enabling the flag on an unsupported (CPU) module changes nothing.
        lora.lora_fused_bf16 = True
        out_flag = base(x)
        assert torch.equal(out_ref, out_flag)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused path requires CUDA")
class TestFusedParity:
    def _pair(self, **kwargs):
        torch.manual_seed(0)
        base_u, lora_u = _make_lora(device="cuda", dtype=torch.bfloat16, **kwargs)
        torch.manual_seed(0)
        base_f, lora_f = _make_lora(device="cuda", dtype=torch.bfloat16, **kwargs)
        lora_f.lora_fused_bf16 = True
        for m in (lora_u, lora_f):
            m.train()
        return (base_u, lora_u), (base_f, lora_f)

    def test_forward_backward_parity(self):
        (base_u, lora_u), (base_f, lora_f) = self._pair()
        x = torch.randn(2, 33, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_f = x.detach().clone().requires_grad_(True)

        out_u = base_u(x)
        out_f = base_f(x_f)
        # bf16 reassociation: epilogue add rounds once instead of twice.
        assert out_u.shape == out_f.shape
        rel = (out_f - out_u).abs().max() / out_u.abs().max().clamp_min(1e-6)
        assert rel < 0.05

        out_u.sum().backward()
        out_f.sum().backward()
        for name, gu, gf in [
            ("dx", x.grad, x_f.grad),
            ("dA", lora_u.lora_down.weight.grad, lora_f.lora_down.weight.grad),
            ("dB", lora_u.lora_up.weight.grad, lora_f.lora_up.weight.grad),
        ]:
            assert gf is not None, name
            denom = gu.abs().max().clamp_min(1e-6)
            assert (gf - gu).abs().max() / denom < 0.05, name

    def test_fused_path_actually_taken(self, monkeypatch):
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(device="cuda", dtype=torch.bfloat16)
        lora.lora_fused_bf16 = True
        lora.train()
        base(torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16))
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dropout": 0.1},
            {"rank_dropout": 0.1},
            {"module_dropout": 0.1},
            {"split_dims": [24, 24]},
            {"bias": True},
            {"nora": "forward"},
        ],
    )
    def test_unsupported_falls_back(self, monkeypatch, kwargs):
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(device="cuda", dtype=torch.bfloat16, **kwargs)
        lora.lora_fused_bf16 = True
        lora.train()
        base(torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16))
        assert not calls

    def test_non_bf16_falls_back(self, monkeypatch):
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(device="cuda", dtype=torch.float32)
        lora.lora_fused_bf16 = True
        lora.train()
        base(torch.randn(2, 5, 64, device="cuda", dtype=torch.float32))
        assert not calls

    def test_eval_mode_fuses(self, monkeypatch):
        # Fusion is not dropout-dependent; eval mode still takes the fused path.
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(device="cuda", dtype=torch.bfloat16)
        lora.lora_fused_bf16 = True
        lora.eval()
        with torch.no_grad():
            base(torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16))
        assert len(calls) == 1

    def test_3d_and_2d_inputs(self, monkeypatch):
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(device="cuda", dtype=torch.bfloat16)
        lora.lora_fused_bf16 = True
        lora.eval()
        o3 = base(torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16))
        o2 = base(torch.randn(10, 64, device="cuda", dtype=torch.bfloat16))
        assert o3.shape == (2, 5, 48) and o2.shape == (10, 48)
        assert len(calls) == 2

    def test_checkpoint_recompute(self, monkeypatch):
        # save_for_backward tensors must be consistent across checkpoint recompute.
        calls = _apply_spy(monkeypatch)
        base, lora = _make_lora(device="cuda", dtype=torch.bfloat16)
        lora.lora_fused_bf16 = True
        lora.train()
        x = torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = torch.utils.checkpoint.checkpoint(lambda t: base(t), x, use_reentrant=False)
        out.sum().backward()
        assert x.grad is not None
        # non-reentrant checkpointing invokes apply in the no-grad first pass and again in recompute
        assert len(calls) == 2
