"""NoRA (Normalized Low-Rank Adaptation, Kang et al. 2026, arXiv:2608.31036) in ``networks/lora.py``.

CPU-only. Covers the ``nora=off|forward|init`` modes and the ``init=bimi`` initialisation
exposed as ``network_args``, the adapter export (a standard LoRA whose ``lora_up @
lora_down`` is the training-time delta) and the save -> load round trip through the
inference module that has no NoRA code at all.
"""

import logging

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from musubi_tuner.networks.lora import (
    LoRAModule,
    LoRANetwork,
    bimi_init_,
    create_network,
    create_network_from_weights,
    normalize_lora_down,
)

RANK = 4


class _Block(nn.Module):
    """A Linear whose input size is not a multiple of the rank, plus a 1x1 Conv2d."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(10, 6, bias=True)
        self.conv = nn.Conv2d(5, 7, kernel_size=1, bias=False)

    def forward(self, x, img):
        return self.proj(x), self.conv(img)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()

    def forward(self, x, img):
        return self.block(x, img)


def _column_norms(weight):
    return weight.detach().float().flatten(1).norm(dim=0)


def _build(seed=0, alpha=RANK, **network_args):
    torch.manual_seed(seed)
    model = _Model()
    network = create_network(["_Block"], "lora_unet", 1.0, RANK, alpha, None, None, model, **network_args)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    return model, network


def _inputs(seed=1):
    torch.manual_seed(seed)
    return torch.randn(3, 10), torch.randn(2, 5, 4, 4)


def _loras(network):
    return {lora.lora_name: lora for lora in network.unet_loras}


def _perturb(network, seed=2):
    """Make lora_up non-zero and stretch lora_down's columns so raw and normalised A differ."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for lora in network.unet_loras:
            nn.init.normal_(lora.lora_up.weight, std=0.1)
            lora.lora_down.weight.mul_(torch.rand(1, *lora.lora_down.weight.shape[1:]) * 3 + 0.5)


def test_off_is_bit_identical_to_the_default():
    plain_model, plain = _build()
    off_model, off = _build(nora="off", init="default")
    assert all(lora.nora == "off" for lora in plain.unet_loras + off.unet_loras)
    for key, value in plain.state_dict().items():
        assert torch.equal(value, off.state_dict()[key])
    _perturb(plain)
    _perturb(off)
    x, img = _inputs()
    for a, b in zip(plain_model(x, img), off_model(x, img)):
        assert torch.equal(a, b)
    for key, value in plain.snapshot_weights(None).items():
        assert torch.equal(value, off.snapshot_weights(None)[key])
        assert torch.equal(value, plain.state_dict()[key])


def test_invalid_network_args_are_rejected():
    with pytest.raises(ValueError, match="nora"):
        _build(nora="sometimes")
    with pytest.raises(ValueError, match="init"):
        _build(init="orthogonal")


def test_alpha_not_equal_dim_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="musubi_tuner.networks.lora"):
        _build(alpha=1, nora="forward")
    assert any("network_alpha == network_dim" in record.getMessage() for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="musubi_tuner.networks.lora"):
        _build(alpha=RANK, nora="forward")
    assert not any("network_alpha" in record.getMessage() for record in caplog.records)


def test_forward_mode_uses_unit_columns_and_trains_a():
    model, network = _build(nora="forward")
    _perturb(network)
    x, img = _inputs()
    for lora in network.unet_loras:
        # The raw parameter is not unit-norm; the effective weight is, for Linear and Conv alike.
        assert not torch.allclose(_column_norms(lora.lora_down.weight), torch.ones(1))
        effective = lora.effective_down_weight(lora.lora_down)
        assert torch.allclose(_column_norms(effective), torch.ones(1), atol=1e-6)

    out, out_img = model(x, img)
    (out.sum() + out_img.sum()).backward()
    for lora in network.unet_loras:
        assert lora.lora_down.weight.grad is not None
        assert lora.lora_down.weight.grad.abs().sum() > 0
        # A gradient through the normalisation is orthogonal to the column direction.
        raw = lora.lora_down.weight.detach().flatten(1)
        grad = lora.lora_down.weight.grad.flatten(1)
        assert torch.allclose((raw * grad).sum(dim=0), torch.zeros(1), atol=1e-5)

    # The forward equals a plain LoRA delta computed with the normalised A.
    proj = model.block.proj
    lora = _loras(network)["lora_unet_block_proj"]
    with torch.no_grad():
        base = nn.functional.linear(x, proj.weight, proj.bias)
        delta = x @ (lora.lora_up.weight @ normalize_lora_down(lora.lora_down.weight)).T * lora.scale
        assert torch.allclose(model(x, img)[0], base + delta, atol=1e-6)


def test_forward_mode_saves_a_standard_lora_whose_delta_is_the_training_delta():
    model, network = _build(nora="forward")
    _perturb(network)
    x, img = _inputs()
    with torch.no_grad():
        out, out_img = model(x, img)
        network.set_enabled(False)
        base, base_img = model(x, img)
        network.set_enabled(True)
    saved = network.snapshot_weights(None)
    raw = network.state_dict()
    for name in _loras(network):
        down = saved[f"{name}.lora_down.weight"]
        assert torch.allclose(_column_norms(down), torch.ones(1), atol=1e-6)
        assert not torch.equal(down, raw[f"{name}.lora_down.weight"])  # raw parameter untouched
        assert torch.equal(saved[f"{name}.lora_up.weight"], raw[f"{name}.lora_up.weight"])
        assert torch.equal(saved[f"{name}.alpha"], raw[f"{name}.alpha"])
    proj = _loras(network)["lora_unet_block_proj"]
    delta = x @ (saved["lora_unet_block_proj.lora_up.weight"] @ saved["lora_unet_block_proj.lora_down.weight"]).T * proj.scale
    assert torch.allclose(out - base, delta, atol=1e-6)
    conv = _loras(network)["lora_unet_block_conv"]
    merged = saved["lora_unet_block_conv.lora_up.weight"].flatten(1) @ saved["lora_unet_block_conv.lora_down.weight"].flatten(1)
    delta_img = nn.functional.conv2d(img, merged.view(7, 5, 1, 1)) * conv.scale
    assert torch.allclose(out_img - base_img, delta_img, atol=1e-6)


def test_init_mode_normalises_once_then_trains_freely():
    _, network = _build(nora="init")
    for lora in network.unet_loras:
        assert torch.allclose(_column_norms(lora.lora_down.weight), torch.ones(1), atol=1e-6)
    _, plain = _build(nora="off")
    for lora, plain_lora in zip(network.unet_loras, plain.unet_loras):
        # Same directions as the default init, only the column magnitudes were removed.
        assert torch.allclose(lora.lora_down.weight, normalize_lora_down(plain_lora.lora_down.weight))

    model, network = _build(nora="init")
    _perturb(network)
    x, img = _inputs()
    lora = _loras(network)["lora_unet_block_proj"]
    assert not torch.allclose(_column_norms(lora.lora_down.weight), torch.ones(1))
    # After the one-time normalisation A is an ordinary parameter: the forward and the
    # saved adapter both use it as is.
    proj = model.block.proj
    with torch.no_grad():
        base = nn.functional.linear(x, proj.weight, proj.bias)
        delta = x @ (lora.lora_up.weight @ lora.lora_down.weight).T * lora.scale
        assert torch.allclose(model(x, img)[0], base + delta, atol=1e-6)
    saved = network.snapshot_weights(None)
    assert torch.equal(saved["lora_unet_block_proj.lora_down.weight"], lora.lora_down.weight)


def test_bimi_init_is_tiled_identity_with_zero_up():
    _, network = _build(nora="forward", init="bimi")
    loras = _loras(network)
    down = loras["lora_unet_block_proj"].lora_down.weight
    expected = torch.eye(RANK).repeat(1, 3)[:, :10]  # 10 inputs: two full blocks and a truncated one
    assert torch.equal(down, expected)
    assert torch.equal(loras["lora_unet_block_proj"].lora_up.weight, torch.zeros(6, RANK))
    conv_down = loras["lora_unet_block_conv"].lora_down.weight
    assert torch.equal(conv_down.flatten(1), torch.eye(RANK).repeat(1, 2)[:, :5])
    assert torch.equal(loras["lora_unet_block_conv"].lora_up.weight, torch.zeros(7, RANK, 1, 1))
    for lora in network.unet_loras:
        assert torch.allclose(_column_norms(lora.lora_down.weight), torch.ones(1))
        # Normalising is a no-op on bimi, so forward mode starts from exactly these blocks.
        assert torch.equal(lora.effective_down_weight(lora.lora_down), lora.lora_down.weight)

    weight = torch.empty(4, 8)
    bimi_init_(weight)
    assert torch.equal(weight, torch.cat([torch.eye(4), torch.eye(4)], dim=1))
    assert torch.equal(weight[:, :4] @ weight[:, :4].T, torch.eye(4))  # orthogonal within a block


def test_bimi_without_nora_is_plain_lora_with_the_block_init():
    _, network = _build(init="bimi")
    lora = _loras(network)["lora_unet_block_proj"]
    assert lora.nora == "off"
    assert torch.equal(lora.lora_down.weight, torch.eye(RANK).repeat(1, 3)[:, :10])


def test_round_trip_save_load_is_identical(tmp_path):
    model, network = _build(nora="forward")
    _perturb(network)
    x, img = _inputs()
    with torch.no_grad():
        expected = model(x, img)
    path = tmp_path / "nora.safetensors"
    network.save_weights(str(path), None, {})
    weights_sd = load_file(str(path))

    # Inference: a plain LoRAInfModule network built from the file, no NoRA involved.
    torch.manual_seed(0)
    inference_model = _Model()
    inference = create_network_from_weights(["_Block"], 1.0, weights_sd, None, inference_model, for_inference=True)
    assert all(lora.nora == "off" for lora in inference.unet_loras)
    inference.apply_to(None, inference_model, apply_text_encoder=False, apply_unet=True)
    info = inference.load_weights(str(path))
    assert not info.missing_keys and not info.unexpected_keys
    with torch.no_grad():
        for a, b in zip(inference_model(x, img), expected):
            assert torch.equal(a, b)

    # Merge: the same file folded into the base weights.
    torch.manual_seed(0)
    merged_model = _Model()
    merged = create_network_from_weights(["_Block"], 1.0, weights_sd, None, merged_model, for_inference=True)
    merged.merge_to(None, merged_model, weights_sd, torch.float32, "cpu")
    with torch.no_grad():
        for a, b in zip(merged_model(x, img), expected):
            assert torch.allclose(a, b, atol=1e-6)

    # Resume via --network_weights into a fresh nora=forward network: re-normalising the
    # saved (already unit-norm) A reproduces the same forward and the same export.
    resumed_model, resumed = _build(nora="forward")  # same base weights, LoRA state to be replaced
    _perturb(resumed, seed=9)
    resumed.load_weights(str(path))
    with torch.no_grad():
        for a, b in zip(resumed_model(x, img), expected):
            assert torch.allclose(a, b, atol=1e-6)
    for key, value in resumed.snapshot_weights(None).items():
        assert torch.allclose(value, weights_sd[key], atol=1e-6)


def test_loading_a_plain_lora_into_a_nora_run_warns(tmp_path, caplog):
    _, plain = _build(nora="off")
    _perturb(plain)
    path = tmp_path / "plain.safetensors"
    plain.save_weights(str(path), None, {})
    _, network = _build(nora="forward")
    with caplog.at_level(logging.WARNING, logger="musubi_tuner.networks.lora"):
        network.load_weights(str(path))
    assert any("not saved by a NoRA run" in record.getMessage() for record in caplog.records)


def test_split_dims_module_normalises_every_head():
    torch.manual_seed(0)
    base = nn.Linear(8, 12, bias=False)
    lora = LoRAModule("qkv", base, lora_dim=RANK, alpha=RANK, split_dims=[4, 4, 4], nora="forward", init="bimi")
    for down in lora.lora_down:
        assert torch.equal(down.weight, torch.eye(RANK).repeat(1, 2))
    with torch.no_grad():
        for down, up in zip(lora.lora_down, lora.lora_up):
            down.weight.mul_(torch.rand(1, 8) * 3 + 0.5)
            nn.init.normal_(up.weight, std=0.1)
    lora.apply_to()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = nn.functional.linear(x, base.weight)
        parts = [x @ (up.weight @ normalize_lora_down(down.weight)).T for down, up in zip(lora.lora_down, lora.lora_up)]
        expected = expected + torch.cat(parts, dim=-1) * lora.scale
        assert torch.allclose(base(x), expected, atol=1e-6)
    exported = lora.export_state_dict()
    for i in range(3):
        assert torch.allclose(_column_norms(exported[f"lora_down.{i}.weight"]), torch.ones(1), atol=1e-6)


class _QKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 12, bias=False)

    def forward(self, x):
        return self.qkv(x)


def _split_network(nora):
    torch.manual_seed(0)
    model = _QKV()
    module_kwargs = {"split_dims": [4, 4, 4]}
    if nora != "off":
        module_kwargs["nora"] = nora
    network = LoRANetwork(["_QKV"], "lora_unet", None, model, lora_dim=RANK, alpha=RANK, module_kwargs=module_kwargs)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    lora = network.unet_loras[0]
    with torch.no_grad():
        for down, up in zip(lora.lora_down, lora.lora_up):
            down.weight.mul_(torch.rand(1, 8) * 3 + 0.5)
            nn.init.normal_(up.weight, std=1.0)  # far above the max norm below
    return model, network


def _split_head_norms(lora):
    return [
        (up.weight @ lora.effective_down_weight(down) * lora.scale).norm().item() for down, up in zip(lora.lora_down, lora.lora_up)
    ]


@pytest.mark.parametrize("nora", ["off", "forward"])
def test_max_norm_regularization_handles_split_dims_keys(nora):
    model, network = _split_network(nora)
    lora = network.unet_loras[0]
    max_norm = 0.5
    assert all(norm > max_norm for norm in _split_head_norms(lora))
    x = torch.randn(2, 8)
    keys_scaled, mean_norm, max_norm_seen = network.apply_max_norm_regularization(max_norm, "cpu")
    assert keys_scaled == 3  # one lora_down.<i>.weight per head, each read its module's alpha
    # Every head's effective delta now sits on the cap, and the reported norms match it.
    for norm in _split_head_norms(lora):
        assert norm == pytest.approx(max_norm, rel=1e-5)
    assert mean_norm == pytest.approx(max_norm, rel=1e-5)
    assert max_norm_seen == pytest.approx(max_norm, rel=1e-5)
    with torch.no_grad():
        expected = nn.functional.linear(x, model.qkv.weight)
        parts = [x @ (up.weight @ lora.effective_down_weight(down)).T for down, up in zip(lora.lora_down, lora.lora_up)]
        assert torch.allclose(model(x), expected + torch.cat(parts, dim=-1) * lora.scale, atol=1e-6)
    if nora == "forward":
        # The raw directions were left alone; only lora_up carried the correction.
        for down in lora.lora_down:
            assert not torch.allclose(_column_norms(down.weight), torch.ones(1))
