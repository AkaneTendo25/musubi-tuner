import pytest
import torch
import torch.nn.functional as F

import musubi_tuner.minimax_h3.sparse_attention as sparse_attention
from musubi_tuner.minimax_h3.sparse_attention import (
    H3ReferenceSparseConfig,
    H3ReferenceSparseLayout,
    H3UniversalSparseRouter,
    _select_router_blocks,
    reference_sparse_attention,
)
from musubi_tuner.minimax_h3.triton_kernels import reference_block_sparse_forward


def test_universal_sparse_router_scores_each_head_and_block() -> None:
    router = H3UniversalSparseRouter(
        num_layers=3,
        num_heads=2,
        head_dim=8,
        index_dim=4,
        block_size=4,
    )
    query = torch.randn(1, 10, 2, 8, requires_grad=True)
    key = torch.randn(1, 10, 2, 8, requires_grad=True)

    scores = router(query, key, layer_index=1)

    assert scores.shape == (1, 2, 3, 3)
    scores.sum().backward()
    assert router.query_projection.weight.grad is not None
    assert router.key_projection.weight.grad is not None
    assert query.grad is None
    assert key.grad is None

    cached_scores = router.score_summaries(router.summarize(query), router.summarize(key), layer_index=1)
    torch.testing.assert_close(cached_scores, scores.detach())


def test_universal_sparse_router_teacher_loss_prefers_teacher_logits() -> None:
    teacher = torch.tensor([[[[0.8, 0.15, 0.05]]]])
    matching_scores = teacher.log()
    uniform_scores = torch.zeros_like(matching_scores)

    matching_loss = H3UniversalSparseRouter.teacher_kl_loss(matching_scores, teacher)
    uniform_loss = H3UniversalSparseRouter.teacher_kl_loss(uniform_scores, teacher)

    assert matching_loss < 1e-6
    assert uniform_loss > matching_loss


def test_universal_sparse_router_supports_layer_specific_index_projections() -> None:
    router = H3UniversalSparseRouter(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        index_dim=4,
        block_size=4,
        layer_specific_projections=True,
    )
    query = torch.randn(1, 8, 2, 8)
    key = torch.randn(1, 8, 2, 8)

    scores = router(query, key, layer_index=1)
    scores.sum().backward()

    assert scores.shape == (1, 2, 2, 2)
    assert router.query_layer_projection.grad is not None
    assert router.query_layer_projection.grad[1].abs().sum() > 0
    assert router.query_layer_projection.grad[0].abs().sum() == 0


def test_universal_sparse_router_preserves_multiple_subblock_summaries() -> None:
    router = H3UniversalSparseRouter(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        index_dim=4,
        block_size=8,
        summary_subblocks=4,
        layer_specific_projections=True,
    )
    query = torch.randn(1, 16, 2, 8)
    key = torch.randn(1, 16, 2, 8)

    summary = router.summarize(query)
    scores = router(query, key, layer_index=0)

    assert summary.shape == (1, 2, 2, 32)
    assert scores.shape == (1, 2, 2, 2)


def test_universal_sparse_router_supports_group_specific_projections() -> None:
    router = H3UniversalSparseRouter(
        num_layers=2,
        num_heads=4,
        head_dim=8,
        index_dim=4,
        block_size=8,
        layer_specific_projections=True,
        projection_groups=2,
    )
    query = torch.randn(1, 16, 4, 8)
    key = torch.randn(1, 16, 4, 8)

    scores = router(query, key, layer_index=1)
    scores.sum().backward()

    assert scores.shape == (1, 4, 2, 2)
    assert router.query_layer_projection.grad is not None
    assert router.query_layer_projection.grad[1, 0].abs().sum() > 0
    assert router.query_layer_projection.grad[1, 1].abs().sum() > 0
    assert router.query_layer_projection.grad[0].abs().sum() == 0


def test_positive_kernel_router_produces_finite_log_mass_scores() -> None:
    router = H3UniversalSparseRouter(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        index_dim=6,
        block_size=4,
        layer_specific_projections=True,
        positive_kernel=True,
    )
    query = torch.randn(1, 8, 2, 8)
    key = torch.randn(1, 8, 2, 8)

    scores = router(query, key, layer_index=0)
    loss = H3UniversalSparseRouter.teacher_kl_loss(scores, scores.detach().softmax(dim=-1))
    loss.backward()

    assert scores.shape == (1, 2, 2, 2)
    assert torch.isfinite(scores).all()
    assert router.query_layer_projection.grad is not None


def test_router_selection_is_deterministic_topk() -> None:
    scores = torch.tensor([[[0.7, 0.2, 0.8, 0.1], [0.4, 0.6, 0.3, 0.5]]])

    selected = _select_router_blocks(scores, maximum_blocks=2)

    assert selected.tolist() == [[[True, False, True, False], [False, True, False, True]]]


@pytest.mark.parametrize("reference_start", [0, 17])
def test_flex_fallback_uses_reference_relative_token_routes(monkeypatch, reference_start: int) -> None:
    rows = reference_start + 256 + 9
    query = torch.randn(1, rows, 1, 8)
    captured = {}

    def capture_flex(query, key, value, *, block_mask, score_mod):
        captured["mask"] = block_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(sparse_attention, "_compiled_flex_attention", capture_flex)
    layout = H3ReferenceSparseLayout(
        reference_start=reference_start,
        reference_rows=256,
        rows_per_frame=1,
        block_first_frames=torch.tensor([0, 10]),
        block_last_frames=torch.tensor([0, 10]),
    )
    config = H3ReferenceSparseConfig(topk_blocks=1, temporal_radius=2)
    routing_scores = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])
    reference_sparse_attention(query, query, query, layout, config, routing_scores=routing_scores)

    mask_mod = captured["mask"].mask_mod
    scalar = lambda value: torch.tensor(value, dtype=torch.int64)
    assert bool(mask_mod(scalar(0), scalar(0), scalar(reference_start), scalar(reference_start)))
    assert not bool(mask_mod(scalar(0), scalar(0), scalar(reference_start), scalar(reference_start + 128)))
    assert bool(mask_mod(scalar(0), scalar(0), scalar(rows - 1), scalar(reference_start + 128)))
    assert bool(mask_mod(scalar(0), scalar(0), scalar(reference_start + 128), scalar(rows - 1)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton sparse attention requires CUDA")
@pytest.mark.parametrize("data_path", ["packed", "mapped_int64"])
def test_triton_hybrid_matches_dense_with_misaligned_reference_and_exact_gradients(data_path: str) -> None:
    torch.manual_seed(9876)
    rows, reference_start, reference_rows, heads, head_dim = 290, 17, 256, 2, 128
    tensors = [torch.randn(1, rows, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    query, key, value = tensors
    layout = H3ReferenceSparseLayout(reference_start, reference_rows, rows_per_frame=128)
    config = H3ReferenceSparseConfig(topk_blocks=2, temporal_radius=0, data_path=data_path)
    routing_scores = torch.zeros(heads, 2, 2, device="cuda")
    actual = reference_sparse_attention(query, key, value, layout, config, routing_scores=routing_scores)
    expected = (
        F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        .transpose(1, 2)
        .flatten(2, 3)
    )
    relative_output_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_output_error < 4e-3

    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, tensors, upstream, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, tensors, upstream)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        relative_gradient_error = (
            actual_gradient.float() - expected_gradient.float()
        ).norm() / expected_gradient.float().norm().clamp_min(1e-12)
        assert relative_gradient_error < 4e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton sparse attention requires CUDA")
def test_attention_permutation_keeps_interleaved_reference_result_and_gradients_dense_exact() -> None:
    from musubi_tuner.minimax_h3.model import MiniMaxH3Attention

    torch.manual_seed(97531)
    rows, hidden = 280, 256
    dense = MiniMaxH3Attention(hidden, heads=2, head_dim=128, qk_norm_eps=1e-5).cuda().bfloat16()
    sparse = MiniMaxH3Attention(hidden, heads=2, head_dim=128, qk_norm_eps=1e-5).cuda().bfloat16()
    sparse.load_state_dict(dense.state_dict())
    reference_indices = torch.cat((torch.arange(2, 130), torch.arange(140, 268))).cuda()
    is_reference = torch.zeros(rows, dtype=torch.bool, device="cuda")
    is_reference[reference_indices] = True
    permutation = torch.cat((reference_indices, torch.arange(rows, device="cuda")[~is_reference]))
    inverse = torch.argsort(permutation)
    sparse.configure_reference_sparse_attention(
        H3ReferenceSparseConfig(topk_blocks=2, temporal_radius=0, layer_ranges=((0, 1),)),
        router=None,
        layer_index=0,
    )
    sparse.reference_sparse_layout = H3ReferenceSparseLayout(
        reference_start=0,
        reference_rows=256,
        rows_per_frame=1,
        permutation=permutation,
        inverse_permutation=inverse,
        block_first_frames=torch.tensor([0, 1], device="cuda"),
        block_last_frames=torch.tensor([0, 1], device="cuda"),
    )
    dense_hidden = torch.randn(1, rows, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    sparse_hidden = dense_hidden.detach().clone().requires_grad_(True)
    expected = dense(dense_hidden)
    actual = sparse(sparse_hidden)
    relative_output_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_output_error < 6e-3
    upstream = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, sparse_hidden, upstream)[0]
    expected_gradient = torch.autograd.grad(expected, dense_hidden, upstream)[0]
    relative_gradient_error = (actual_gradient.float() - expected_gradient.float()).norm() / expected_gradient.float().norm()
    assert relative_gradient_error < 6e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton sparse attention requires CUDA")
def test_triton_backward_route_buckets_match_routed_sdpa() -> None:
    """Cover both low-traffic and globally popular K/V blocks in dK/dV."""
    torch.manual_seed(2468)
    heads, blocks, block_rows, head_dim = 2, 32, 128, 128
    rows = blocks * block_rows
    tensors = [torch.randn(heads, rows, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    query, key, value = tensors
    routes = torch.empty(heads, blocks, 4, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(1357)
    for head in range(heads):
        for query_block in range(blocks):
            tail = torch.randperm(blocks - 1, generator=generator, device="cuda")[:3] + 1
            routes[head, query_block] = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64), tail)).int()
    weights = torch.ones_like(routes, dtype=torch.float32)
    actual = reference_block_sparse_forward(query, key, value, routes, weights)
    unbucketed = reference_block_sparse_forward(query, key, value, routes, weights, _bucket_dkv=False, _bucket_queries=False)
    torch.testing.assert_close(actual, unbucketed, rtol=0, atol=0)

    expected_heads = []
    for head in range(heads):
        expected_blocks = []
        for query_block in range(blocks):
            selected_rows = (
                (routes[head, query_block, :, None] * block_rows + torch.arange(block_rows, device="cuda")[None, :])
                .flatten()
                .long()
            )
            query_rows = slice(query_block * block_rows, (query_block + 1) * block_rows)
            expected_blocks.append(
                F.scaled_dot_product_attention(
                    query[head, query_rows].unsqueeze(0),
                    key[head, selected_rows].unsqueeze(0),
                    value[head, selected_rows].unsqueeze(0),
                )[0]
            )
        expected_heads.append(torch.cat(expected_blocks))
    expected = torch.stack(expected_heads)
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 4e-3

    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, tensors, upstream, retain_graph=True)
    unbucketed_gradients = torch.autograd.grad(unbucketed, tensors, upstream, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, tensors, upstream)
    for actual_gradient, unbucketed_gradient in zip(actual_gradients, unbucketed_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, unbucketed_gradient, rtol=0, atol=0)
    # The reference is 64 separate BF16 SDPA calls, so its K/V reduction order
    # differs from the single routed kernel. The optimization itself is guarded
    # more strictly above by exact equality with the unbucketed kernel.
    tolerances = {"dq": 4e-3, "dk": 6e-3, "dv": 6e-3}
    for name, actual_gradient, expected_gradient in zip(("dq", "dk", "dv"), actual_gradients, expected_gradients, strict=True):
        relative_error = (actual_gradient.float() - expected_gradient.float()).norm() / expected_gradient.float().norm()
        assert relative_error < tolerances[name], f"{name} relative error {relative_error}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton sparse attention requires CUDA")
def test_triton_strided_mapped_rows_match_routed_sdpa() -> None:
    torch.manual_seed(8642)
    heads, physical_rows, block_rows, head_dim = 2, 300, 128, 128
    query_start, query_rows = 16, 256
    actual_bases = [
        torch.randn(physical_rows, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    expected_bases = [tensor.detach().clone().requires_grad_(True) for tensor in actual_bases]

    first = torch.cat(
        (
            torch.arange(100, device="cuda", dtype=torch.int32),
            torch.full((28,), -1, device="cuda", dtype=torch.int32),
        )
    )
    second = torch.cat(
        (
            torch.arange(100, 300, device="cuda", dtype=torch.int32),
            torch.full((56,), -1, device="cuda", dtype=torch.int32),
        )
    )
    row_map = torch.cat((first, second))
    valid = row_map >= 0
    routes = torch.tensor([[[0, 1], [1, 2]], [[2, 0], [0, 2]]], device="cuda", dtype=torch.int32)
    weights = torch.ones_like(routes, dtype=torch.float32)

    actual_query = actual_bases[0][query_start : query_start + query_rows].permute(1, 0, 2)
    actual_key = actual_bases[1].permute(1, 0, 2)
    actual_value = actual_bases[2].permute(1, 0, 2)
    actual = reference_block_sparse_forward(
        actual_query,
        actual_key,
        actual_value,
        routes,
        weights,
        kv_valid_mask=valid,
        _kv_row_indices=row_map,
    )

    expected_query = expected_bases[0][query_start : query_start + query_rows].permute(1, 0, 2)
    expected_key = expected_bases[1].permute(1, 0, 2)
    expected_value = expected_bases[2].permute(1, 0, 2)
    expected_heads = []
    for head in range(heads):
        expected_blocks = []
        for query_block in range(query_rows // block_rows):
            logical_rows = (
                routes[head, query_block, :, None] * block_rows
                + torch.arange(block_rows, device="cuda", dtype=torch.int32)[None, :]
            ).flatten()
            physical = row_map[logical_rows.long()]
            physical = physical[physical >= 0].long()
            query_slice = slice(query_block * block_rows, (query_block + 1) * block_rows)
            expected_blocks.append(
                F.scaled_dot_product_attention(
                    expected_query[head, query_slice].unsqueeze(0),
                    expected_key[head, physical].unsqueeze(0),
                    expected_value[head, physical].unsqueeze(0),
                )[0]
            )
        expected_heads.append(torch.cat(expected_blocks))
    expected = torch.stack(expected_heads)

    output_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert output_error < 4e-3
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, actual_bases, upstream)
    expected_gradients = torch.autograd.grad(expected, expected_bases, upstream)
    for name, actual_gradient, expected_gradient in zip(("dq", "dk", "dv"), actual_gradients, expected_gradients, strict=True):
        relative_error = (actual_gradient.float() - expected_gradient.float()).norm() / expected_gradient.float().norm()
        assert relative_error < 7e-3, f"{name} relative error {relative_error}"
