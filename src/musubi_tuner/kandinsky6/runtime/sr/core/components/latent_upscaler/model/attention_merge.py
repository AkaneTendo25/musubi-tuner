# Exact merging of flash-style attention branches with a closed-form null branch.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.autograd import Function

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def merge_math(
    outputs: Sequence[Tensor],
    lses: Sequence[Tensor],
    null_value: Tensor,
    null_logits: Tensor,
) -> tuple[Tensor, Tensor]:
    """Combine attention branches and the null token into one softmax, in fp32.

    Args:
        outputs: Branch outputs shaped ``(batch, seq, heads, dim)``.
        lses: Branch log-sum-exps shaped ``(batch, seq, heads)``.
        null_value: Null-token value shaped ``(heads, dim)``.
        null_logits: Null-token logits shaped ``(batch, seq, heads)``.

    Returns:
        Merged output ``(batch, seq, heads, dim)`` and total lse ``(batch, seq, heads)``.
    """
    lse_stack = torch.stack([*[lse.float() for lse in lses], null_logits.float()], dim=0)
    total_lse = torch.logsumexp(lse_stack, dim=0)
    weights = torch.exp(lse_stack - total_lse)
    merged = weights[-1].unsqueeze(-1) * null_value.float()
    for weight, output in zip(weights[:-1], outputs, strict=True):
        merged = merged + weight.unsqueeze(-1) * output.float()
    return merged, total_lse


@functools.cache
def compiled_merge_math() -> Callable[..., tuple[Tensor, Tensor]]:
    """Compile the merge math once and reuse it across calls."""
    return torch.compile(merge_math, fullgraph=True)


class MergeWithNullBranch(Function):
    """Merge kernel attention branches with the null branch, with exact gradients.

    Flash-style kernels (NATTEN ``na2d``) return ``(output, lse)`` but their
    backward ignores incoming lse gradients, so plain autograd through a softmax
    merge silently drops part of their q/k/v gradients. This op follows the
    ring-attention convention instead: each kernel branch receives the full
    upstream gradient while its saved output/lse tensors are overwritten in
    place with the merged values, letting the kernel backward reconstruct exact
    global gradients. The null branch is single-key attention, so its gradients
    have a closed form and are computed here directly.

    The in-place overwrite means branch tensors must not be read after backward,
    and the op is incompatible with gradient recomputation (activation
    checkpointing) and double backward.
    """

    @staticmethod
    def forward(ctx: Any, *args: Any) -> Tensor:
        """Merge branches; see :func:`merge_attention_branches` for the layout of ``args``."""
        null_value, null_logits, num_branches, use_compile, *branch_tensors = args
        outputs = branch_tensors[:num_branches]
        lses = branch_tensors[num_branches:]
        math_fn = compiled_merge_math() if use_compile else merge_math
        merged, total_lse = math_fn(outputs, lses, null_value, null_logits)
        ctx.num_branches = num_branches
        ctx.save_for_backward(null_value, null_logits, merged, total_lse, *branch_tensors)
        target = outputs[0] if outputs else null_value
        return merged.to(target.dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        """Route full gradients to kernel branches and closed-form ones to the null pair."""
        num_branches = ctx.num_branches
        null_value, null_logits, merged, total_lse, *branch_tensors = ctx.saved_tensors
        outputs = branch_tensors[:num_branches]
        lses = branch_tensors[num_branches:]
        grad = grad_out.float()
        null_weight = torch.exp(null_logits.float() - total_lse)
        d_null_value = torch.einsum("bsh,bshd->hd", null_weight, grad)
        d_null_logits = null_weight * (torch.einsum("bshd,hd->bsh", grad, null_value.float()) - (grad * merged).sum(dim=-1))
        for output, lse in zip(outputs, lses, strict=True):
            output.data.copy_(merged.to(output.dtype))
            lse.data.copy_(total_lse.to(lse.dtype))
        return (
            d_null_value.to(null_value.dtype),
            d_null_logits.to(null_logits.dtype),
            None,
            None,
            *(grad_out for _ in outputs),
            *(None for _ in lses),
        )


def merge_attention_branches(
    outputs: Sequence[Tensor],
    lses: Sequence[Tensor],
    null_value: Tensor,
    null_logits: Tensor,
    *,
    torch_compile: bool = False,
) -> Tensor:
    """Merge kernel attention branches with the closed-form null branch.

    Args:
        outputs: Kernel branch outputs shaped ``(batch, seq, heads, dim)``; may be empty.
        lses: Matching branch log-sum-exps shaped ``(batch, seq, heads)``.
        null_value: Null-token value shaped ``(heads, dim)``.
        null_logits: Null-token logits shaped ``(batch, seq, heads)``.
        torch_compile: Compile the merge math with ``torch.compile``.

    Returns:
        Merged attention output shaped ``(batch, seq, heads, dim)`` in the branch dtype
        (or the null-value dtype when no branches are given).
    """
    return MergeWithNullBranch.apply(null_value, null_logits, len(outputs), torch_compile, *outputs, *lses)
