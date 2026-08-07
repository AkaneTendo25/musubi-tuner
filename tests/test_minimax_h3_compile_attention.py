from __future__ import annotations

import pytest
import torch

import musubi_tuner.minimax_h3.model as h3_model


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
