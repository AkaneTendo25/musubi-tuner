"""Numerical SDS-LoRA checks against dense weights and ordinary LoRA consumers."""

import copy
import math

import pytest
import torch
from torch import nn

from musubi_tuner.networks.lora import create_network, create_network_from_weights


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(7, 9, bias=True)

    def forward(self, x):
        return self.proj(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()

    def forward(self, x):
        return self.block(x)


def _build(*, alpha=1.25, rank=3, multiplier=1.7, **kwargs):
    torch.manual_seed(123)
    model = _Model()
    baseline = copy.deepcopy(model)
    network = create_network(["_Block"], "lora_unet", multiplier, rank, alpha, None, None, model, **kwargs)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    return model, network, baseline


def _sds(*, warmup=2, phases=3, total=11, **kwargs):
    model, network, baseline = _build(sds="True", sds_warmup_steps=str(warmup), sds_update_phases=str(phases), **kwargs)
    network.configure_sds_training(total)
    params, _ = network.prepare_optimizer_params(unet_lr=0.01)
    optimizer = torch.optim.AdamW(params, weight_decay=0)
    return model, network, baseline, optimizer


def _update(model, network, optimizer, index=0):
    torch.manual_seed(40 + index)
    x = torch.randn(2, 4, 7)
    target = torch.randn(2, 4, 9)
    (model(x) - target).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    network.on_sds_optimizer_step(optimizer)


def _active(**kwargs):
    model, network, baseline, optimizer = _sds(**kwargs)
    for i in range(network.sds_warmup_steps):
        _update(model, network, optimizer, i)
    return model, network, baseline, optimizer


def test_sds_off_preserves_default_parameters_outputs_and_export():
    plain_model, plain, _ = _build()
    off_model, off, _ = _build(sds="False")
    for a, b in zip(plain.parameters(), off.parameters(), strict=True):
        assert torch.equal(a, b)
    with torch.no_grad():
        for network in (plain, off):
            network.unet_loras[0].lora_up.weight.fill_(0.2)
    x = torch.randn(2, 7)
    assert torch.equal(plain_model(x), off_model(x))
    assert plain.export_state_dict().keys() == off.export_state_dict().keys()
    for key, value in plain.export_state_dict().items():
        assert torch.equal(value, off.export_state_dict()[key])


def test_warmup_transition_preserves_delta_and_resets_adapter_optimizer_state(monkeypatch):
    model, network, _, optimizer = _sds()
    module = network.unet_loras[0]
    _update(model, network, optimizer)
    x = torch.randn(3, 7)
    model(x).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.state[module.lora_down.weight]
    assert optimizer.state[module.lora_up.weight]
    before = model(x).detach().clone()
    original_svd = torch.linalg.svd
    seen = []

    def small_svd(matrix, **kwargs):
        seen.append(matrix.shape)
        assert matrix.shape == (3, 3)
        return original_svd(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", small_svd)
    network.on_sds_optimizer_step(optimizer)
    torch.testing.assert_close(model(x), before, atol=2e-6, rtol=2e-6)
    assert seen == [(3, 3)]
    assert not optimizer.state
    assert module.sds_qa.requires_grad is False
    assert module.sds_qb.requires_grad is False
    assert network.sds_completed_steps == 2


@pytest.mark.parametrize("shape", [(5, 7), (2, 4, 7)])
def test_main_forward_and_all_gradients_match_independent_dense_reference(shape):
    model, network, baseline, _ = _active()
    module = network.unet_loras[0]
    with torch.no_grad():
        module.lora_down.weight.add_(torch.randn_like(module.lora_down.weight) * 0.03)
        module.lora_up.weight.add_(torch.randn_like(module.lora_up.weight) * 0.02)
    a, b = module.lora_down.weight, module.lora_up.weight
    qa, qb = module.sds_qa, module.sds_qb
    factor = module.multiplier * module.scale
    delta = factor * (qb @ a + b @ qa.T)
    x = torch.randn(shape, requires_grad=True)
    upstream = torch.randn(*shape[:-1], 9)
    expected = nn.functional.linear(x, baseline.block.proj.weight + delta, baseline.block.proj.bias)
    out = model(x)
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-6)
    out.backward(upstream)
    g = upstream.flatten(0, -2).T @ x.detach().flatten(0, -2)
    torch.testing.assert_close(a.grad, factor * qb.T @ g, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(b.grad, factor * g @ qa, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(x.grad, upstream @ (baseline.block.proj.weight + delta), atol=3e-6, rtol=3e-6)
    assert qa.grad is None and qb.grad is None


@pytest.mark.parametrize("active", [False, True])
def test_lossless_export_reloads_and_merges_in_ordinary_lora(active, tmp_path):
    model, network, baseline, optimizer = _sds()
    _update(model, network, optimizer)
    if active:
        _update(model, network, optimizer, 1)
        module = network.unet_loras[0]
        with torch.no_grad():
            module.lora_down.weight.add_(torch.randn_like(module.lora_down.weight) * 0.05)
            module.lora_up.weight.add_(torch.randn_like(module.lora_up.weight) * 0.05)
    raw_before = {key: value.clone() for key, value in network.state_dict().items()}
    exported = network.export_state_dict()
    rank = 6 if active else 3
    assert set(exported) == {
        "lora_unet_block_proj.lora_down.weight",
        "lora_unet_block_proj.lora_up.weight",
        "lora_unet_block_proj.alpha",
    }
    down, up = exported["lora_unet_block_proj.lora_down.weight"], exported["lora_unet_block_proj.lora_up.weight"]
    assert down.shape == (rank, 7) and up.shape == (9, rank)
    scale = exported["lora_unet_block_proj.alpha"].item() / rank
    assert scale == pytest.approx(1.25 / math.sqrt(3))
    if active:
        assert torch.linalg.matrix_rank(up @ down).item() > 3
    for key, value in network.state_dict().items():
        assert torch.equal(value, raw_before[key]), key
    path = tmp_path / "adapter.safetensors"
    network.save_weights(str(path), torch.float32, network.sds_export_metadata())
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt") as saved:
        assert saved.metadata()["ss_network_dim"] == str(rank)
    exported = load_file(path)
    infer_model = copy.deepcopy(baseline)
    infer = create_network_from_weights(["_Block"], 1.7, exported, unet=infer_model, for_inference=True)
    infer.apply_to(None, infer_model, apply_text_encoder=False, apply_unet=True)
    assert not infer.load_state_dict(exported).missing_keys
    merged_model = copy.deepcopy(baseline)
    merged = create_network_from_weights(["_Block"], 1.7, exported, unet=merged_model, for_inference=True)
    merged.merge_to(None, merged_model, exported, torch.float32, "cpu")
    x = torch.randn(2, 4, 7)
    torch.testing.assert_close(infer_model(x), model(x), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(merged_model(x), model(x), atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("checkpoint_step", [1, 2, 4])
def test_resume_restores_cached_bases_phase_and_optimizer_trajectory(checkpoint_step):
    model, network, _, optimizer = _sds()
    for i in range(checkpoint_step):
        _update(model, network, optimizer, i)
    raw = copy.deepcopy(network.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    resumed_model, resumed, _, resumed_optimizer = _sds()
    resumed.load_state_dict(raw)
    resumed_optimizer.load_state_dict(opt)
    resumed.configure_sds_training(11)
    assert resumed.sds_completed_steps == checkpoint_step
    for i in range(checkpoint_step, 8):
        _update(model, network, optimizer, i)
        _update(resumed_model, resumed, resumed_optimizer, i)
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
    x = torch.randn(2, 7)
    assert torch.equal(model(x), resumed_model(x))


def test_schedule_uses_main_training_steps_and_does_not_refresh_on_export(monkeypatch):
    model, network, _, optimizer = _active(warmup=2, phases=3, total=11)
    refresh_at = []
    module = network.unet_loras[0]
    original = module.refresh_sds_bases

    def refresh():
        refresh_at.append(network.sds_completed_steps)
        original()

    monkeypatch.setattr(module, "refresh_sds_bases", refresh)
    for i in range(2, 11):
        _update(model, network, optimizer, i)
        network.export_state_dict()
    # Nine main steps, split into three phases with intervals 1, 2, and 3.
    assert refresh_at == [3, 4, 5, 6, 8, 11]


def test_basis_refresh_keeps_the_svd_orientation_when_weights_have_not_changed():
    _, network, _, _ = _active()
    module = network.unet_loras[0]
    before = network.export_state_dict()
    delta = before["lora_unet_block_proj.lora_up.weight"] @ before["lora_unet_block_proj.lora_down.weight"]
    module.refresh_sds_bases()
    after = network.export_state_dict()
    refreshed = after["lora_unet_block_proj.lora_up.weight"] @ after["lora_unet_block_proj.lora_down.weight"]
    torch.testing.assert_close(refreshed, delta, atol=1e-6, rtol=1e-5)


def test_delta_norm_matches_effective_dense_update():
    _, network, _, _ = _active()
    module = network.unet_loras[0]
    with torch.no_grad():
        module.lora_down.weight.add_(0.1)
    sd = module.export_state_dict()
    dense = sd["lora_up.weight"] @ sd["lora_down.weight"] * module.scale * module.multiplier
    torch.testing.assert_close(module.delta_norm_sq(), dense.square().sum(), atol=2e-6, rtol=2e-6)


def test_cancelling_branches_cannot_produce_negative_squared_norm():
    _, network, _, _ = _active()
    module = network.unet_loras[0]
    with torch.no_grad():
        module.lora_down.weight.copy_(module.sds_qa.T)
        module.lora_up.weight.copy_(-module.sds_qb)
    assert module.delta_norm_sq().item() >= 0
    assert module.delta_norm_sq().item() < 1e-5


@pytest.mark.parametrize("named_targets", [False, True])
def test_h3_architecture_factory_routes_sds_to_the_selected_linear_targets(named_targets):
    from musubi_tuner.networks import lora_minimax_h3

    class MiniMaxH3TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Sequential(nn.Linear(8, 8))
            self.mlp = nn.Sequential(nn.Linear(8, 8))
            self.adaln_proj = nn.Linear(8, 8)

    model = nn.Module()
    model.blocks = nn.ModuleList([MiniMaxH3TransformerBlock()])
    options = {"sds": "True"}
    if named_targets:
        options["h3_target_modules"] = "attention"
    network = lora_minimax_h3.create_arch_network(1, 3, 4, None, None, model, **options)
    assert network.sds_enabled
    assert all(module.sds_enabled for module in network.unet_loras)
    names = {module.lora_name for module in network.unet_loras}
    assert names == ({"lora_unet_blocks_0_attn_0"} if named_targets else {"lora_unet_blocks_0_attn_0", "lora_unet_blocks_0_mlp_0"})


@pytest.mark.parametrize(
    "options,match",
    [
        ({"sds": "maybe"}, "sds"),
        ({"sds": "True", "sds_warmup_steps": "0"}, "warmup"),
        ({"sds": "True", "sds_update_phases": "0"}, "phases"),
        ({"sds": "True", "nora": "forward"}, "nora"),
        ({"sds": "True", "init": "bimi"}, "bimi"),
        ({"sds": "True", "rank_dropout": "0.1"}, "dropout"),
        ({"sds": "True", "module_dropout": "0.1"}, "dropout"),
    ],
)
def test_invalid_and_incompatible_factory_options_fail(options, match):
    with pytest.raises(ValueError, match=match):
        _build(**options)


def test_rank_and_training_budget_constraints():
    with pytest.raises(ValueError, match="rank"):
        _sds(rank=8)
    with pytest.raises(ValueError, match="warmup"):
        _sds(total=2)
    _, network, _, _ = _active()
    with pytest.raises(ValueError, match="mismatch"):
        network.configure_sds_training(12)


@pytest.mark.parametrize("rank", [0, -1])
def test_nonpositive_sds_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        _sds(rank=rank)


@pytest.mark.parametrize("change", [{"alpha": 2.0}, {"warmup": 3}, {"phases": 2}])
def test_resume_rejects_changed_sds_configuration(change):
    _, network, _, _ = _active()
    _, resumed, _, _ = _sds(**change)
    with pytest.raises((RuntimeError, ValueError), match="mismatch"):
        resumed.load_state_dict(network.state_dict())
        resumed.configure_sds_training(11)


def test_full_accelerate_state_resume_and_accumulation(tmp_path):
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model, network, _, optimizer = _sds()
    network, optimizer = accelerator.prepare(network, optimizer)

    def microsteps(current_model, current_network, current_optimizer, start, stop):
        for index in range(start, stop):
            torch.manual_seed(200 + index)
            x, target = torch.randn(2, 7), torch.randn(2, 9)
            with accelerator.accumulate(current_network):
                accelerator.backward((current_model(x) - target).square().mean())
                current_optimizer.step()
                current_optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    current_network.on_sds_optimizer_step(current_optimizer)
            assert current_network.sds_completed_steps == (index + 1) // 2

    microsteps(model, network, optimizer, 0, 6)
    state_dir = tmp_path / "state"
    accelerator.save_state(str(state_dir))
    microsteps(model, network, optimizer, 6, 10)
    reference = copy.deepcopy(network.state_dict())
    reference_output = model(torch.ones(2, 7)).detach()
    accelerator.free_memory()
    resumed_model, resumed, _, resumed_optimizer = _sds()
    resumed, resumed_optimizer = accelerator.prepare(resumed, resumed_optimizer)
    accelerator.load_state(str(state_dir))
    resumed.configure_sds_training(11)
    assert resumed.sds_completed_steps == 3
    microsteps(resumed_model, resumed, resumed_optimizer, 6, 10)
    for key, value in reference.items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(resumed_model(torch.ones(2, 7)), reference_output, atol=0, rtol=0)
    accelerator.free_memory()


def test_sds_preserves_loraplus_learning_rate_groups():
    _, network, _, _ = _sds(loraplus_lr_ratio="4")
    groups, descriptions = network.prepare_optimizer_params(unet_lr=0.01)
    assert descriptions == ["unet", "unet plus"]
    module = network.unet_loras[0]
    assert list(groups[0]["params"]) == [module.lora_down.weight]
    assert list(groups[1]["params"]) == [module.lora_up.weight]
    assert [group["lr"] for group in groups] == [0.01, 0.04]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
)
def test_bf16_autocast_forward_backward_transition_and_export(device):
    model, network, _, optimizer = _sds()
    model.to(device=device, dtype=torch.bfloat16)
    network.to(device=device)
    x = torch.randn(2, 4, 7, device=device, dtype=torch.bfloat16)
    for _ in range(4):
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(x)
            out.float().square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Deliberately run under autocast: decompositions and their small core stay FP32.
            network.on_sds_optimizer_step(optimizer)
        assert out.dtype == torch.bfloat16
        assert out.isfinite().all()
    for p in network.parameters():
        assert p.isfinite().all()
    assert network.export_state_dict()["lora_unet_block_proj.lora_down.weight"].shape[0] == 6
