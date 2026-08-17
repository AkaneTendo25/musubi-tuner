"""Synthetic probe for the LoRA dtype-bridging helpers added in PR #966.

`LoRAModule._lora_input` and `LoRAModule._match_org_dtype` were added to make LoRA
training work in an autocast-free regime where the frozen base weights and the LoRA
weights have different dtypes (the Ideogram 4 case: bf16 base from a dequantized FP8
checkpoint + fp32 LoRA, run with `--mixed_precision no`).

This test characterizes *exactly when each helper casts* and *what breaks if it is
removed*, deterministically and on CPU only (no real model / no CUDA needed). It exists
to decide whether the helpers can be removed from the shared `networks/lora.py` without
regressing other architectures.

Summary of the findings this test locks in:

  _lora_input(x):  casts x -> lora_down.weight.dtype  iff
                   (x is float) and (x.dtype != lora dtype) and (autocast disabled).
                   => no-op whenever autocast is enabled.

  _match_org_dtype(delta, org):  casts delta -> org.dtype  iff dtypes differ.
                   => no fire guard; always active.

  Regime matrix (base / LoRA / autocast):
    bf16 base, fp32 LoRA, autocast OFF  (Ideogram 4, mixed_precision=no, DEFAULT)
        -> BOTH helpers fire and are load-bearing.
    *,    *,         autocast ON         (Ideogram 4 bf16; other archs bf16/fp16)
        -> both no-op (autocast bridges the matmul dtype).
    fp32 base, fp32 LoRA, autocast OFF  (other archs, mixed_precision=no: --full_bf16
                                          disabled => weights are fp32)
        -> both no-op (everything is already fp32).
"""

import contextlib

import pytest
import torch
import torch.nn as nn

from musubi_tuner.networks.lora import LoRAModule, LoRANetwork


def _make_wrapped_lora(in_dim, out_dim, base_dtype, lora_dtype, *, strip_lora_input=False, strip_match=False):
    """Build a base Linear wrapped by a LoRAModule (forward hooked, as in production)."""
    torch.manual_seed(0)
    base = nn.Linear(in_dim, out_dim, bias=True).to(base_dtype)
    lora = LoRAModule("blk", base, multiplier=1.0, lora_dim=4, alpha=4)
    lora.lora_down.to(lora_dtype)
    lora.lora_up.to(lora_dtype)
    nn.init.normal_(lora.lora_up.weight, std=1e-2)  # nonzero so delta is a real contribution
    if strip_lora_input:
        lora._lora_input = lambda x: x  # simulate removing the input-side helper
    if strip_match:
        lora._match_org_dtype = lambda delta, org: delta  # simulate removing the output-side helper
    lora.apply_to()  # base.forward is now lora.forward
    return base, lora


def _autocast_ctx(enabled):
    return torch.autocast("cpu", dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


class _ToggleTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_lora_network_set_enabled_bypasses_training_adapters():
    torch.manual_seed(0)
    target = _ToggleTarget()
    network = LoRANetwork(["_ToggleTarget"], "lora_unet", None, target, lora_dim=4, alpha=4)
    network.apply_to(None, target, apply_text_encoder=False, apply_unet=True)
    nn.init.normal_(network.unet_loras[0].lora_up.weight, std=1e-2)
    x = torch.randn(2, 8)

    enabled_output = target(x)
    network.set_enabled(False)
    disabled_output = target(x)
    expected_base_output = network.unet_loras[0].org_forward(x)

    assert not torch.equal(enabled_output, disabled_output)
    torch.testing.assert_close(disabled_output, expected_base_output)

    network.set_enabled(True)
    torch.testing.assert_close(target(x), enabled_output)


# --------------------------------------------------------------------------------------
# Part 1: unit-level firing conditions of each helper
# --------------------------------------------------------------------------------------


def test_lora_input_casts_when_autocast_off_and_dtype_differs():
    _, lora = _make_wrapped_lora(8, 8, base_dtype=torch.bfloat16, lora_dtype=torch.float32)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    out = lora._lora_input(x)
    assert out.dtype == torch.float32  # bf16 input cast up to the fp32 LoRA dtype


def test_lora_input_noop_when_dtype_matches():
    _, lora = _make_wrapped_lora(8, 8, base_dtype=torch.float32, lora_dtype=torch.float32)
    x = torch.randn(2, 8, dtype=torch.float32)
    out = lora._lora_input(x)
    assert out is x  # untouched


def test_lora_input_noop_under_autocast_even_if_dtype_differs():
    _, lora = _make_wrapped_lora(8, 8, base_dtype=torch.bfloat16, lora_dtype=torch.float32)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    with _autocast_ctx(True):
        out = lora._lora_input(x)
    assert out is x  # autocast guard short-circuits the cast


def test_match_org_dtype_casts_delta_to_org():
    _, lora = _make_wrapped_lora(8, 8, base_dtype=torch.bfloat16, lora_dtype=torch.float32)
    delta = torch.randn(2, 8, dtype=torch.float32)
    org = torch.randn(2, 8, dtype=torch.bfloat16)
    out = lora._match_org_dtype(delta, org)
    assert out.dtype == torch.bfloat16  # fp32 delta forced down to the bf16 base output


def test_match_org_dtype_noop_when_equal():
    _, lora = _make_wrapped_lora(8, 8, base_dtype=torch.float32, lora_dtype=torch.float32)
    delta = torch.randn(2, 8, dtype=torch.float32)
    org = torch.randn(2, 8, dtype=torch.float32)
    out = lora._match_org_dtype(delta, org)
    assert out is delta


def test_add_then_cast_is_at_least_as_precise_as_cast_then_add():
    """The forward adds the fp32 delta to the bf16 base, then rounds the sum once.

    This keeps more of the high-precision delta than rounding the delta to bf16 before the
    add (the old order), so the result is at least as close to the fp32 reference.
    """
    torch.manual_seed(0)
    org = torch.randn(4096, dtype=torch.float32).to(torch.bfloat16)  # bf16 base output
    delta = torch.randn(4096, dtype=torch.float32) * 0.01  # small fp32 LoRA contribution
    reference = org.float() + delta  # high-precision target

    old_order = org + delta.to(torch.bfloat16)  # cast-then-add (two roundings)
    new_order = (org + delta).to(torch.bfloat16)  # add-then-cast (one rounding)

    err_old = (old_order.float() - reference).abs().mean().item()
    err_new = (new_order.float() - reference).abs().mean().item()
    assert err_new <= err_old


# --------------------------------------------------------------------------------------
# Part 2: end-to-end block behavior and what breaks if a helper is removed
#         block = [ LoRA-wrapped base Linear ] -> [ downstream base Linear ]
# --------------------------------------------------------------------------------------


def test_ideogram4_no_regime_helpers_are_load_bearing():
    """bf16 base + fp32 LoRA + autocast OFF (Ideogram 4 default). Both helpers fire."""
    in_dim = out_dim = 8
    base, _ = _make_wrapped_lora(in_dim, out_dim, base_dtype=torch.bfloat16, lora_dtype=torch.float32)
    downstream = nn.Linear(out_dim, out_dim, bias=True).to(torch.bfloat16)
    x = torch.randn(2, in_dim, dtype=torch.bfloat16)  # must match bf16 base for org_forward

    with _autocast_ctx(False):
        h = base(x)
        y = downstream(h)
    assert h.dtype == torch.bfloat16  # _match_org_dtype pinned output back to bf16
    assert y.dtype == torch.bfloat16  # downstream bf16 Linear is happy


def test_ideogram4_no_regime_breaks_without_match_org_dtype():
    """Removing _match_org_dtype leaks fp32 into the downstream bf16 Linear -> RuntimeError."""
    in_dim = out_dim = 8
    base, _ = _make_wrapped_lora(in_dim, out_dim, base_dtype=torch.bfloat16, lora_dtype=torch.float32, strip_match=True)
    downstream = nn.Linear(out_dim, out_dim, bias=True).to(torch.bfloat16)
    x = torch.randn(2, in_dim, dtype=torch.bfloat16)

    with _autocast_ctx(False):
        h = base(x)
        assert h.dtype == torch.float32  # fp32 delta promoted the residual sum
        with pytest.raises(RuntimeError):
            downstream(h)  # F.linear(fp32 input, bf16 weight) with autocast off


def test_ideogram4_no_regime_breaks_without_lora_input():
    """Removing _lora_input feeds bf16 x into the fp32 lora_down matmul -> RuntimeError."""
    in_dim = out_dim = 8
    base, _ = _make_wrapped_lora(in_dim, out_dim, base_dtype=torch.bfloat16, lora_dtype=torch.float32, strip_lora_input=True)
    x = torch.randn(2, in_dim, dtype=torch.bfloat16)
    with _autocast_ctx(False):
        with pytest.raises(RuntimeError):
            base(x)  # lora_down(bf16 x) with fp32 weight, autocast off


def test_autocast_on_regime_helpers_are_noop_and_removal_is_safe():
    """Any base/LoRA dtype + autocast ON (Ideogram 4 bf16; other archs bf16). Helpers no-op."""
    in_dim = out_dim = 8
    for strip in (False, True):
        base, _ = _make_wrapped_lora(
            in_dim,
            out_dim,
            base_dtype=torch.bfloat16,
            lora_dtype=torch.float32,
            strip_lora_input=strip,
            strip_match=strip,
        )
        downstream = nn.Linear(out_dim, out_dim, bias=True).to(torch.bfloat16)
        x = torch.randn(2, in_dim, dtype=torch.float32)
        with _autocast_ctx(True):
            h = base(x)
            y = downstream(h)
        # autocast drives the matmuls to bf16 regardless of the helpers
        assert h.dtype == torch.bfloat16
        assert y.dtype == torch.bfloat16


def test_other_arch_no_regime_all_fp32_helpers_are_noop_and_removal_is_safe():
    """fp32 base + fp32 LoRA + autocast OFF (other archs, mixed_precision=no). Helpers no-op."""
    in_dim = out_dim = 8
    for strip in (False, True):
        base, _ = _make_wrapped_lora(
            in_dim,
            out_dim,
            base_dtype=torch.float32,
            lora_dtype=torch.float32,
            strip_lora_input=strip,
            strip_match=strip,
        )
        downstream = nn.Linear(out_dim, out_dim, bias=True).to(torch.float32)
        x = torch.randn(2, in_dim, dtype=torch.float32)
        with _autocast_ctx(False):
            h = base(x)
            y = downstream(h)
        assert h.dtype == torch.float32
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# Delta accumulation: LoRAModule._fuse_delta must stay bit-identical to the
# naive `org_forwarded + delta * multiplier * scale` it replaced, while issuing
# fewer full-size elementwise allocations.
# ---------------------------------------------------------------------------


def _naive_fuse_delta(self, org_forwarded, delta, scale):
    """The pre-optimization expression, for bit-identity comparisons."""
    return org_forwarded + delta * self.multiplier * scale


def _build_module(in_dim, out_dim, base_dtype, lora_dtype, multiplier, alpha, lora_dim, split_dims=None, seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(in_dim, out_dim, bias=True).to(base_dtype)
    lora = LoRAModule("blk", base, multiplier=multiplier, lora_dim=lora_dim, alpha=alpha, split_dims=split_dims)
    lora.lora_down.to(lora_dtype)
    lora.lora_up.to(lora_dtype)
    ups = lora.lora_up if split_dims else [lora.lora_up]
    for up in ups:
        nn.init.normal_(up.weight, std=1e-2)
    lora.apply_to()
    return base, lora


def _run(lora, base, x):
    for p in lora.parameters():
        p.grad = None
    xg = x.detach().clone().requires_grad_(True)
    out = base(xg)
    out.float().pow(2).sum().backward()
    grads = [p.grad.detach().clone() for p in lora.parameters() if p.grad is not None]
    return out.detach().clone(), xg.grad.detach().clone(), grads


@pytest.mark.parametrize("shape", [(3, 8), (2, 5, 8)])
@pytest.mark.parametrize("multiplier", [1.0, 0.75])
@pytest.mark.parametrize("alpha,lora_dim", [(4, 4), (2, 4)])  # scale == 1.0 and scale != 1.0
@pytest.mark.parametrize(
    "base_dtype,lora_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),  # mixed regime: _match_org_dtype fires
        (torch.float32, torch.bfloat16),  # in-place accumulate must bail out (would downcast)
    ],
)
@pytest.mark.parametrize("split", [False, True])
def test_fuse_delta_is_bit_identical_to_naive_expression(shape, multiplier, alpha, lora_dim, base_dtype, lora_dtype, split):
    in_dim, out_dim = shape[-1], 12
    split_dims = [4, 8] if split else None
    kwargs = dict(
        in_dim=in_dim,
        out_dim=out_dim,
        base_dtype=base_dtype,
        lora_dtype=lora_dtype,
        multiplier=multiplier,
        alpha=alpha,
        lora_dim=lora_dim,
        split_dims=split_dims,
    )
    x = torch.randn(*shape).to(base_dtype)

    base_a, lora_a = _build_module(**kwargs)
    out_a, xgrad_a, grads_a = _run(lora_a, base_a, x)

    base_b, lora_b = _build_module(**kwargs)
    lora_b._fuse_delta = _naive_fuse_delta.__get__(lora_b, LoRAModule)
    out_b, xgrad_b, grads_b = _run(lora_b, base_b, x)

    assert out_a.dtype == out_b.dtype
    assert torch.equal(out_a, out_b), "forward is not bit-identical to the naive expression"
    assert torch.equal(xgrad_a, xgrad_b), "input gradient is not bit-identical"
    assert len(grads_a) == len(grads_b) and grads_a
    for ga, gb in zip(grads_a, grads_b):
        assert torch.equal(ga, gb), "LoRA weight gradient is not bit-identical"


class _CountFullSizeOps(torch.utils._python_dispatch.TorchDispatchMode):
    """Count elementwise mul/add kernels that produce a full-size [*, out_dim] tensor."""

    def __init__(self, out_dim):
        self.out_dim = out_dim
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in {"mul", "add", "mul_", "add_"} and isinstance(out, torch.Tensor):
            if out.dim() and out.shape[-1] == self.out_dim:
                self.count += 1
        return out


def test_fuse_delta_issues_fewer_full_size_elementwise_kernels():
    """Locks in the allocation reduction: 3 full-size mul/add kernels become 2 (or 1)."""
    in_dim, out_dim = 8, 12
    x = torch.randn(2, 5, in_dim)

    def count(mult, alpha, naive):
        base, lora = _build_module(in_dim, out_dim, torch.float32, torch.float32, mult, alpha, 4)
        if naive:
            lora._fuse_delta = _naive_fuse_delta.__get__(lora, LoRAModule)
        mode = _CountFullSizeOps(out_dim)
        with mode:
            base(x)
        return mode.count

    # multiplier != 1 and scale != 1: two muls + one add -> one mul, one in-place mul, one in-place add
    # (same kernel count but two fewer allocations); multiplier == 1 additionally drops a kernel.
    assert count(1.0, 4, naive=True) == 3  # naive always runs mul, mul, add
    assert count(1.0, 4, naive=False) == 1  # both factors are 1.0 -> plain add only
    assert count(1.0, 2, naive=False) == 2  # scale != 1.0 -> mul + in-place add
    assert count(0.75, 2, naive=False) == 3


def test_fuse_delta_does_not_break_the_dynamo_graph():
    """LoRA modules live inside the compiled ("blocks", ...) scope, so the delta path must
    trace in one graph. A `torch.result_type` call here graph-breaks; `promote_types` does not."""
    in_dim, out_dim = 8, 12
    base, lora = _build_module(in_dim, out_dim, torch.float32, torch.float32, 0.75, 2, 4)
    x = torch.randn(2, 5, in_dim, requires_grad=True)
    eager = base(x)
    eager.sum().backward()
    eager_grad = x.grad.detach().clone()

    compiled = torch.compile(base, backend="aot_eager", fullgraph=True)
    x2 = x.detach().clone().requires_grad_(True)
    out = compiled(x2)
    out.sum().backward()

    assert torch.equal(eager.detach(), out.detach())
    assert torch.equal(eager_grad, x2.grad)
