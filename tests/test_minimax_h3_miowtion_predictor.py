"""CPU coverage for Miowtion bundle plans and learned block selection."""

import json
from copy import deepcopy

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.block_sparse_attention import _miowtion_keep, miowtion_block_sparse_attention
from musubi_tuner.minimax_h3.miowtion_predictor import (
    MIOWTION_FORMAT,
    MiowtionPredictor,
    build_miowtion_plan,
    load_miowtion_bundle,
    pool_miowtion_tiles,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig


def _bundle(tmp_path, *, grid=(2, 8, 8), layers=2, heads=4, dim=8):
    model = MiowtionPredictor(layers, heads, dim)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=1e-3)
    plan = {
        "tiny": {
            "geometry": "tiny",
            "grid": list(grid),
            "shapes": ["2x8x8", "1x8x16"],
            "head_shape": [[0, 0, 1, 1] for _ in range(layers)],
        }
    }
    metadata = {
        "format": MIOWTION_FORMAT,
        "num_layers": str(layers),
        "num_heads": str(heads),
        "head_dim": str(dim),
        "dtype": "float32",
        "keep_ratio": "0.25",
        "plans": json.dumps(plan),
    }
    path = tmp_path / "predictor.safetensors"
    save_file(model.state_dict(), path, metadata=metadata)
    return path


def _positions(grid=(2, 8, 8), context=7):
    video = torch.cartesian_prod(*(torch.arange(size) for size in grid))
    return torch.cat((torch.full((context, 3), -1), video), dim=0)


def test_bundle_is_frozen_and_plan_groups_heads(tmp_path):
    bundle = load_miowtion_bundle(_bundle(tmp_path))
    assert not any(parameter.requires_grad for parameter in bundle.predictor.parameters())
    plan = build_miowtion_plan(bundle, _positions(), 7)
    groups = plan.head_groups(0, "cpu")
    assert [group.heads.tolist() for group in groups] == [[0, 1], [2, 3]]
    assert plan.layout(groups[0].shape, "cpu").valid_count.tolist() == [128, 7]


def test_partial_tile_pooling_ignores_zero_padding(tmp_path):
    bundle = load_miowtion_bundle(_bundle(tmp_path))
    plan = build_miowtion_plan(bundle, _positions(), 7)
    layout = plan.layout(plan.shapes[0], "cpu")
    values = torch.arange(layout.n_tiles * 128 * 2, dtype=torch.float32).view(1, 1, -1, 2)
    values[..., ~layout.slot_valid, :] = 0
    features = pool_miowtion_tiles(values, layout)
    assert torch.equal(features[0, 0, 1, :2], values[0, 0, 128:135].mean(0))
    assert torch.equal(features[0, 0, 1, 2:4], values[0, 0, 128:135].amax(0))
    assert torch.equal(features[0, 0, 1, 4:], values[0, 0, 128:135].amin(0))


def test_predictor_scores_control_topk():
    scores = torch.tensor([[[[0.0, 9.0, 1.0], [2.0, 0.0, 8.0], [7.0, 1.0, 0.0]]]])
    keep = _miowtion_keep(scores, 3, 384, torch.ones(3, dtype=torch.bool), 1 / 3)
    # The diagonal is forced and consumes the one-tile budget.
    assert torch.equal(keep[0, 0], torch.eye(3, dtype=torch.bool))


def test_predictor_residual_projection_is_exact():
    predictor = MiowtionPredictor(1, 2, 2)
    layer = predictor.layers[0]
    with torch.no_grad():
        layer.proj_q.zero_()
        layer.proj_k.zero_()
        layer.proj_q[1, 2, 0] = 2.0
    features = torch.tensor([[[[1.0, 3.0, 5.0, 0.0, -1.0, -2.0]]]])
    query = layer.embed(features, torch.tensor([1]), layer.proj_q)
    key = layer.embed(features, torch.tensor([1]), layer.proj_k)
    assert torch.equal(query, torch.tensor([[[[11.0, 3.0]]]]))
    assert torch.equal(key, torch.tensor([[[[1.0, 3.0]]]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device staging needs CUDA")
def test_cpu_resident_predictor_stages_only_selected_heads():
    predictor = MiowtionPredictor(1, 2, 2)
    layer = predictor.layers[0]
    with torch.no_grad():
        layer.proj_q.zero_()
        layer.proj_q[1, 2, 0] = 2.0
    features = torch.tensor([[[[1.0, 3.0, 5.0, 0.0, -1.0, -2.0]]]], device="cuda")
    embedded = layer.embed(features, torch.tensor([1], device="cuda"), layer.proj_q)
    assert layer.proj_q.device.type == "cpu"
    assert embedded.device.type == "cuda"
    assert torch.equal(embedded.cpu(), torch.tensor([[[[11.0, 3.0]]]]))


def test_plan_rejects_non_raster_or_unknown_geometry(tmp_path):
    bundle = load_miowtion_bundle(_bundle(tmp_path))
    positions = _positions()
    positions[[7, 8]] = positions[[8, 7]]
    with pytest.raises(ValueError, match="raster order"):
        build_miowtion_plan(bundle, positions, 7)
    with pytest.raises(ValueError, match="exactly one"):
        build_miowtion_plan(bundle, _positions((1, 8, 8)), 7)


def test_bundle_rejects_unexpected_tensor(tmp_path):
    path = _bundle(tmp_path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
        metadata = dict(handle.metadata())
    tensors["surprise"] = torch.zeros(1)
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, bad, metadata=metadata)
    with pytest.raises(ValueError, match="predictor tensors"):
        load_miowtion_bundle(bad)


def test_fp8_bundle_dequantizes_per_head_scales(tmp_path):
    path = _bundle(tmp_path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        source = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
        metadata = dict(handle.metadata())
    metadata["dtype"] = "float8_e4m3fn"
    tensors = {}
    for name, value in source.items():
        scale = torch.linspace(0.25, 1.0, value.shape[0])
        tensors[name] = (value / scale.view(-1, 1, 1)).to(torch.float8_e4m3fn)
        tensors[name + ".__scale"] = scale
    fp8 = tmp_path / "fp8.safetensors"
    save_file(tensors, fp8, metadata=metadata)
    bundle = load_miowtion_bundle(fp8)
    expected = tensors["layers.0.proj_q"].float() * tensors["layers.0.proj_q.__scale"].view(-1, 1, 1)
    assert bundle.predictor.layers[0].proj_q.dtype == torch.bfloat16
    assert torch.equal(bundle.predictor.layers[0].proj_q.float(), expected.bfloat16().float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
@pytest.mark.parametrize("keep_fraction", [1.0, 0.5])
def test_attention_matches_dense_at_one_and_backpropagates(tmp_path, keep_fraction):
    grid, context, heads, dim = (4, 8, 8), 7, 4, 32
    bundle = load_miowtion_bundle(_bundle(tmp_path, grid=grid, heads=heads, dim=dim), device="cuda")
    plan = build_miowtion_plan(bundle, _positions(grid, context), context)
    generator = torch.Generator(device="cuda").manual_seed(4)
    shape = (1, heads, context + torch.tensor(grid).prod().item(), dim)
    query, key, value = (torch.randn(shape, generator=generator, device="cuda", requires_grad=True) for _ in range(3))
    output = miowtion_block_sparse_attention(
        query, key, value, bundle=bundle, plan=plan, layer_index=0, keep_fraction=keep_fraction
    )
    assert torch.isfinite(output).all()
    if keep_fraction == 1.0:
        dense = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        assert torch.allclose(output, dense, atol=2e-3, rtol=2e-3)
    output.square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_h3_model_routes_through_miowtion_without_saving_predictor(tmp_path):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=4,
        attention_head_dim=32,
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
    torch.manual_seed(11)
    sparse = MiniMaxH3Transformer(config).cuda()
    dense = deepcopy(sparse)
    bundle = load_miowtion_bundle(_bundle(tmp_path, grid=(1, 1, 2), layers=2, heads=4, dim=32), device="cuda")
    sparse.set_miowtion_sparse_attention(bundle, keep_fraction=1.0)
    assert not any("miowtion" in key for key in sparse.state_dict())
    inputs = {
        "video_hidden_states": torch.randn(1, 2, 16, device="cuda"),
        "audio_hidden_states": torch.randn(1, 4, 8, device="cuda"),
        "encoder_hidden_states": torch.randn(1, 3, 12, device="cuda"),
        "timestep": torch.tensor([0.25, 0.75], device="cuda"),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0], device="cuda"),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0], device="cuda"),
        "position_ids": torch.cat((torch.zeros(7, 3), _positions((1, 1, 2), 0)), dim=0).cuda(),
        "video_indices": torch.tensor([7, 8], device="cuda"),
        "audio_indices": torch.tensor([3, 4, 5, 6], device="cuda"),
        "text_indices": torch.tensor([0, 1, 2], device="cuda"),
    }
    sparse_output = sparse(**inputs)
    dense_output = dense(**inputs)
    assert torch.allclose(sparse_output.video, dense_output.video, atol=2e-3, rtol=2e-3)
    assert torch.allclose(sparse_output.audio, dense_output.audio, atol=2e-3, rtol=2e-3)
    sparse_output.video.square().mean().backward()
    assert sparse.blocks[0].attn.qkv_proj.weight.grad is not None
