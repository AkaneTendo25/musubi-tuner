"""Real ModelOffloader / LoRAStreamOffloader sequences on both YuE2 stacks (CUDA, no weights).

no-grad AR prefill -> grad AR pass -> grad NAR pass -> backward -> no-grad pass, twice, must equal the same model
without block swap (outputs and trainable-parameter grads), for S < N/2 and S >= N/2.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Model  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CFG = YuE2Config.tiny(num_layers=8)


def _model():
    torch.manual_seed(0)
    model = YuE2Model(CFG)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape) / p.shape[1] ** 0.5)
            elif name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape))
            else:
                p.copy_(0.1 * torch.randn(p.shape))
    model = model.to(torch.bfloat16)
    model.requires_grad_(False)
    for name, p in model.named_parameters():  # LoRA-like: frozen Linears, trainable (resident) norms
        if name.endswith("norm.weight") and ".blocks." in name:
            p.requires_grad_(True)
    return model.train()


def _steps(model, device, n_steps=2):
    g = torch.Generator().manual_seed(1)
    records = []
    for _ in range(n_steps):
        ids = torch.randint(0, CFG.vocab_size, (1, 40), generator=g).to(device)
        x_t = torch.randn(1, 12, 64, generator=g).to(device)
        model.begin_train_step()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            kv = model.ar_forward(model.embed(ids), return_kv=True, return_hidden=False).kv
        hidden = model.ar_forward(model.embed(ids)).hidden
        v = model.nar_forward(x_t, torch.tensor([0.3], device=device), kv, rope_offset=40)
        loss = hidden.float().square().mean() + v.float().square().mean()
        loss.backward()
        grads = {n: p.grad.float().cpu() for n, p in model.named_parameters() if p.grad is not None}
        model.begin_train_step()  # a pass after backward starts a new phase (as the validator and sampler do)
        with torch.no_grad():
            after = model.ar_forward(model.embed(ids)).hidden.float().cpu()
        records.append((hidden.detach().float().cpu(), v.detach().float().cpu(), grads, after))
    return records


@pytest.mark.parametrize("h2d_only", [False, True])
@pytest.mark.parametrize("swap", [(2, 3), (5, 6)])
@pytest.mark.parametrize("gc", [False, True])
def test_block_swap_equals_no_swap(h2d_only, swap, gc):
    if h2d_only and not gc:
        pytest.skip("H2D-only block swap requires gradient checkpointing for training")
    device = torch.device("cuda")
    base = _model()
    swapped = copy.deepcopy(base)
    if gc:
        base.enable_gradient_checkpointing()
        swapped.enable_gradient_checkpointing()
    base.to(device)
    base._execution_device = device
    ref = _steps(base, device)

    ar_n, nar_n = swap
    swapped.set_block_swap_plan(ar_n, nar_n, ar_backward=True, nar_backward=True)
    swapped.enable_block_swap(max(swap), BlockSwapConfig(device=device, supports_backward=True, h2d_only=h2d_only))
    swapped.move_to_device_except_swap_blocks(device)
    swapped.prepare_block_swap_before_forward()
    got = _steps(swapped, device)

    for (h0, v0, g0, a0), (h1, v1, g1, a1) in zip(ref, got):
        # forwards are bitwise deterministic; SDPA backward may use atomics, so grads get a tolerance
        assert torch.equal(h0, h1) and torch.equal(v0, v1) and torch.equal(a0, a1)
        assert g0.keys() == g1.keys() and g0
        for k in g0:
            assert ((g0[k] - g1[k]).norm() / g0[k].norm().clamp_min(1e-12)).item() < 1e-2, k

    swapped.switch_block_swap_for_inference()
    with torch.no_grad():
        ids = torch.randint(0, CFG.vocab_size, (1, 9), device=device)
        cache = swapped.new_kv_cache(16)
        a = swapped.ar_prefill_into_cache(ids, cache)
        cache_b = base.new_kv_cache(16)
        b = base.ar_prefill_into_cache(ids, cache_b)
    assert torch.equal(a, b)
    swapped.switch_block_swap_for_training()
    again = _steps(swapped, device, n_steps=1)
    assert torch.isfinite(again[0][1]).all()


def test_ar_resident_toggle():
    device = torch.device("cuda")
    model = _model()
    ref = copy.deepcopy(model).to(device)
    ref._execution_device = device
    model.set_block_swap_plan(4, 4, ar_backward=False, nar_backward=True)
    model.enable_block_swap(4, BlockSwapConfig(device=device, supports_backward=True))
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    model.switch_block_swap_for_inference()
    model.set_ar_resident(True)
    assert all(p.device.type == "cuda" for p in model.ar.blocks.parameters())
    assert all(m.weight.device.type == "cpu" for m in model.nar.blocks.modules() if isinstance(m, torch.nn.Linear))
    ids = torch.randint(0, CFG.vocab_size, (1, 7), device=device)
    with torch.no_grad():
        c1, c2 = model.new_kv_cache(8), ref.new_kv_cache(8)
        assert torch.equal(model.ar_prefill_into_cache(ids, c1), ref.ar_prefill_into_cache(ids, c2))
    model.set_ar_resident(False)
    with torch.no_grad():
        kv = model.ar_forward(model.embed(ids), return_kv=True, return_hidden=False).kv
        kv_ref = ref.ar_forward(ref.embed(ids), return_kv=True, return_hidden=False).kv
        x = torch.randn(1, 5, 64, device=device)
        t = torch.tensor([0.5], device=device)
        assert torch.equal(model.nar_forward(x, t, kv, 7), ref.nar_forward(x, t, kv_ref, 7))
    model.switch_block_swap_for_training()
