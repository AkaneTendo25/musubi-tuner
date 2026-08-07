import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.minimax_h3.adaln_lowrank import (
    ANCHOR_TIMESTEPS,
    build_adaln_basis,
    default_timestep_grid,
    TABLE_KEY,
    build_timestep_table,
    factorize_adaln_weight,
    make_adaln_split_hook,
    reconstruction_error,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3TimeEmbedder


def _embedder(seed: int = 0) -> MiniMaxH3TimeEmbedder:
    torch.manual_seed(seed)
    module = MiniMaxH3TimeEmbedder(256, 64, 96)
    module.eval()
    return module


def test_grid_contains_pinned_conditioning_timesteps():
    grid = default_timestep_grid(129)

    for anchor in ANCHOR_TIMESTEPS:
        assert bool((grid == anchor).any()), f"grid must represent the pinned timestep {anchor}"
    assert bool((grid.diff() > 0).all()), "grid must be sorted and de-duplicated"


def test_basis_is_orthonormal_and_shaped_by_rank():
    basis = build_adaln_basis(_embedder(), rank=6)

    assert basis.basis.shape == (96, 6)
    assert basis.mean.shape == (96,)
    assert basis.rank == 6
    torch.testing.assert_close(basis.basis.T @ basis.basis, torch.eye(6, dtype=basis.basis.dtype), atol=1e-10, rtol=0)


def test_reconstruction_error_falls_as_rank_grows():
    embedder = _embedder()
    errors = [reconstruction_error(embedder, build_adaln_basis(embedder, rank=r)) for r in (2, 4, 8)]

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


def test_factorization_folds_the_mean_into_the_bias():
    # a(t) ~= mean + U c(t), so W a + b ~= (W U) c + (b + W mean). Dropping the
    # W mean term is a correctness bug, not an approximation: assert the folded
    # bias reproduces the exact projection far better than the raw bias does.
    embedder = _embedder()
    basis = build_adaln_basis(embedder, rank=16)
    torch.manual_seed(1)
    weight = torch.randn(32, 96) * 0.02
    bias = torch.randn(32) * 0.1

    reduced, folded = factorize_adaln_weight(weight, bias, basis)

    timesteps = torch.tensor(ANCHOR_TIMESTEPS + (0.123, 0.5))
    with torch.no_grad():
        activated = F.silu(embedder(timesteps).double())
    exact = activated @ weight.double().T + bias.double()
    approx = basis.coefficients(activated) @ reduced.double().T + folded.double()
    unfolded = basis.coefficients(activated) @ reduced.double().T + bias.double()

    folded_error = ((exact - approx).norm() / exact.norm()).item()
    unfolded_error = ((exact - unfolded).norm() / exact.norm()).item()
    assert folded_error < 1e-5
    assert unfolded_error > 100 * folded_error


def test_factorization_shapes_and_bias_free_projection():
    basis = build_adaln_basis(_embedder(), rank=4)
    weight = torch.randn(24, 96)

    reduced, folded = factorize_adaln_weight(weight, None, basis)

    assert reduced.shape == (24, 4)
    assert folded.shape == (24,)


def test_chunked_contraction_matches_single_pass():
    basis = build_adaln_basis(_embedder(), rank=5)
    torch.manual_seed(2)
    weight = torch.randn(48, 96)
    bias = torch.randn(48)

    whole = factorize_adaln_weight(weight, bias, basis, chunk_rows=1024)
    chunked = factorize_adaln_weight(weight, bias, basis, chunk_rows=7)

    torch.testing.assert_close(whole[0], chunked[0])
    torch.testing.assert_close(whole[1], chunked[1])


def test_embedder_dtype_is_restored_after_fitting():
    embedder = _embedder().to(torch.bfloat16)

    build_adaln_basis(embedder, rank=3)

    assert next(embedder.parameters()).dtype == torch.bfloat16


@pytest.mark.parametrize("rank", [0, -1])
def test_rejects_non_positive_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        build_adaln_basis(_embedder(), rank=rank)


def test_rejects_rank_above_feature_count():
    with pytest.raises(ValueError, match="exceeds"):
        build_adaln_basis(_embedder(), rank=97)


def test_rejects_mismatched_weight_width():
    basis = build_adaln_basis(_embedder(), rank=4)

    with pytest.raises(ValueError, match="input features"):
        factorize_adaln_weight(torch.randn(8, 95), None, basis)


def test_uncentered_basis_has_zero_mean_so_bias_needs_no_folding():
    embedder = _embedder()

    basis = build_adaln_basis(embedder, rank=8, center=False)

    torch.testing.assert_close(basis.mean, torch.zeros_like(basis.mean))
    weight = torch.randn(16, 96)
    bias = torch.randn(16)
    _, folded = factorize_adaln_weight(weight, bias, basis)
    torch.testing.assert_close(folded, bias)


def test_streaming_hook_reduces_weights_and_leaves_bias_values_alone():
    embedder = _embedder()
    basis = build_adaln_basis(embedder, rank=8, center=False)
    table = build_timestep_table(embedder, basis, 65)
    hook = make_adaln_split_hook(basis, table)

    weight = torch.randn(32, 96)
    keys, tensors = hook("blocks.0.adaln_proj.linear.weight", weight)
    assert keys == ["blocks.0.adaln_proj.linear.weight"]
    assert tensors[0].shape == (32, 8)

    # The bias value is unchanged (uncentered basis needs no fold), so its arrival
    # order relative to the weight is irrelevant; only its dtype is aligned.
    bias = torch.randn(32)
    _, out = hook("blocks.0.adaln_proj.linear.bias", bias)
    torch.testing.assert_close(out[0], bias.to(out[0].dtype))
    assert hook("blocks.0.attn.qkv_proj.weight", torch.randn(4, 4)) == (None, None)


def test_streaming_hook_swaps_the_embedder_for_the_table():
    embedder = _embedder()
    basis = build_adaln_basis(embedder, rank=8, center=False)
    table = build_timestep_table(embedder, basis, 65)
    hook = make_adaln_split_hook(basis, table)

    keys, tensors = hook("time_embedder.proj_out.weight", torch.randn(96, 64))
    assert keys == [TABLE_KEY]
    assert tensors[0].shape == (65, 8)

    # Every other embedder tensor disappears.
    assert hook("time_embedder.proj_in.weight", torch.randn(64, 256)) == ([], [])
    assert hook("time_embedder.proj_in.bias", None) == ([], None)


def test_streaming_hook_rejects_a_centered_basis():
    embedder = _embedder()
    basis = build_adaln_basis(embedder, rank=8, center=True)
    table = build_timestep_table(embedder, basis, 65)

    with pytest.raises(ValueError, match="uncentered"):
        make_adaln_split_hook(basis, table)


def test_factorization_follows_the_weight_device():
    # Weights stream in on the loader's calc device; the basis must follow them
    # rather than forcing a CPU matmul (or a device mismatch).
    basis = build_adaln_basis(_embedder(), rank=4)
    weight = torch.randn(8, 96)

    reduced, folded = factorize_adaln_weight(weight, None, basis)

    assert reduced.device == weight.device
    assert folded.device == weight.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA factorization")
def test_factorization_can_compute_on_cuda_and_return_to_source_device():
    basis = build_adaln_basis(_embedder(), rank=4)
    weight = torch.randn(8, 96)
    expected = factorize_adaln_weight(weight, None, basis)

    actual = factorize_adaln_weight(weight, None, basis, compute_device="cuda")

    assert actual[0].device.type == "cpu"
    assert actual[1].device.type == "cpu"
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_streaming_hook_stores_the_reduced_projection_in_the_default_dtype():
    # The stored dtype bounds the reduction: bf16's mantissa puts a floor under
    # the modulation orders of magnitude above the basis error, which would cap
    # every rank at the same accuracy. The reduced weight is small enough that
    # float32 costs nothing worth trading that away for.
    embedder = _embedder()
    basis = build_adaln_basis(embedder, rank=8, center=False)
    hook = make_adaln_split_hook(basis, build_timestep_table(embedder, basis, 65))

    _, tensors = hook("blocks.0.adaln_proj.linear.weight", torch.randn(32, 96).to(torch.bfloat16))

    assert tensors[0].dtype == torch.float32


def test_streaming_hook_aligns_bias_dtype_with_weight():
    # F.linear dispatches to addmm(bias, input, weight.T), so a bias left in the
    # checkpoint's dtype while the weight is promoted fails at the matmul.
    embedder = _embedder()
    basis = build_adaln_basis(embedder, rank=8, center=False)
    hook = make_adaln_split_hook(basis, build_timestep_table(embedder, basis, 65))

    _, w = hook("blocks.0.adaln_proj.linear.weight", torch.randn(32, 96).to(torch.bfloat16))
    keys, b = hook("blocks.0.adaln_proj.linear.bias", torch.randn(32).to(torch.bfloat16))

    assert keys == ["blocks.0.adaln_proj.linear.bias"]
    assert b[0].dtype == w[0].dtype
