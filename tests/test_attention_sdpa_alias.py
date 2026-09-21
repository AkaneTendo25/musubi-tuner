"""The public SDPA alias must match torch attention, including mask preparation."""

import pytest
import torch

from musubi_tuner.hunyuan_model.attention import attention as hunyuan_attention
from musubi_tuner.modules.attention import AttentionParams, attention


def test_sdpa_alias_in_direct_attention_params():
    assert AttentionParams("sdpa").attn_mode == "torch"


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_shared_sdpa_alias_matches_outputs_and_gradients(masked, split):
    torch.manual_seed(1)
    tensors = [torch.randn(2, 7, 2, 8, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]) if masked else None

    def run(mode):
        params = AttentionParams.create_attention_params_from_mask(mode, split, 3, mask)
        return attention(list(tensors), attn_params=params)

    expected = run("torch")
    actual = run("sdpa")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grads = torch.autograd.grad(expected.sum(), tensors)
    actual_grads = torch.autograd.grad(actual.sum(), tensors)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("split", [False, True])
def test_hunyuan_sdpa_alias_matches_torch(split):
    torch.manual_seed(2)
    tensors = [torch.randn(2, 7, 2, 8) for _ in range(3)]
    total_len = torch.tensor([5, 7]) if split else None
    expected = hunyuan_attention(list(tensors), mode="torch", total_len=total_len)
    actual = hunyuan_attention(list(tensors), mode="sdpa", total_len=total_len)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hunyuan_model_sdpa_alias_preserves_text_padding_mask(monkeypatch):
    from musubi_tuner.hunyuan_model import models

    # This CUDA-only helper is unused by torch attention; keep the mask test on CPU.
    monkeypatch.setattr(models, "get_cu_seqlens", lambda _mask, _length: None)
    model = models.HYVideoDiffusionTransformer(
        text_states_dim=12,
        text_states_dim_2=8,
        hidden_size=32,
        heads_num=2,
        mm_double_blocks_depth=1,
        mm_single_blocks_depth=1,
        rope_dim_list=[4, 6, 6],
        text_projection="linear",
        attn_mode="sdpa",
    )
    captured = []

    class ReachedAttention(Exception):
        pass

    def capture_mask(_module, args):
        captured.append(args[3])
        raise ReachedAttention

    model.double_blocks[0].register_forward_pre_hook(capture_mask)
    with pytest.raises(ReachedAttention):
        model(
            torch.zeros(2, 4, 1, 4, 4),
            torch.ones(2),
            text_states=torch.zeros(2, 4, 12),
            text_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
            text_states_2=torch.zeros(2, 8),
        )
    expected = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    expected[0, :, :6, :6] = True
    expected[1] = True
    assert captured[0] is not None
    torch.testing.assert_close(captured[0], expected)
