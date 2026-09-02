"""Selective activation checkpointing for the H3 main blocks.

Plain gradient checkpointing re-runs a whole block in backward, attention
forward included, although every fused attention backward kernel (SDPA's
flash, cuDNN and efficient kernels, FlashAttention-2/3) reads only the
forward OUTPUT and the per-row logsumexp, never the score matrix. Saving those
outputs lets recomputation skip the attention forward -- the largest single
cost of a long packed sequence -- for one ``rows x hidden`` tensor per block.

Both levels are expressed as a ``torch.utils.checkpoint`` selective policy:

- ``attention`` saves the outputs of the fused attention kernels listed in
  :data:`ATTENTION_OUTPUT_OPS`;
- ``qkv`` additionally saves the base QKV projection's own matmul. The
  attention module wraps the frozen ``qkv_proj``'s ``forward`` -- the call a
  LoRA wrapper later captures as ``org_forward`` -- in :func:`projection_region`,
  so adapter terms run outside the region and are recomputed; inside it the
  matmul is recognised by its weight operand's ``(in_features, out_features)``
  shape, which ConvRot's group rotations do not have.

The policy sees dispatcher ops only. A kernel invoked through a plain Python
binding (an old ``flash_attn`` build without its ``torch.library`` ops, the
Triton INT8 attention, ``flex_attention``) never reaches it, and SDPA's math
backend is a composite of ordinary ops rather than one saved op. The first is
caught by :func:`attention_kernel_is_visible` at configuration time; the
second only shows once a real batch's mask and dtype pick a backend, so the
policy tallies what it saved per pass in :class:`SavedActivations` and the
transformer refuses to continue when a checkpointed block saved nothing.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

CHECKPOINT_KEEP_MODES = ("none", "attention", "qkv")

# Fused attention forwards whose backward needs only their own outputs. The
# ``flash_attn`` names are the ``torch.library`` ops flash-attn >= 2.7.1
# registers; ``flash_attn_3::fwd`` is the Hopper package's op. Names that are
# not registered in a given environment simply never match.
ATTENTION_OUTPUT_OPS = frozenset(
    {
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_cudnn_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "flash_attn::_flash_attn_forward",
        "flash_attn::_flash_attn_varlen_forward",
        "flash_attn_3::fwd",
        "flash_attn_3::_flash_attn_forward",
    }
)

# Matmuls a Linear (plain, scaled FP8 or ConvRot INT8) can dispatch for its
# weight, with the index of the weight operand, whose trailing two dimensions
# are ``(in_features, out_features)`` for the projection itself.
_PROJECTION_MATMUL_OPS = {
    "aten::mm": 1,
    "aten::addmm": 2,
    "aten::bmm": 1,
    "aten::baddbmm": 2,
    "aten::_scaled_mm": 1,
    "aten::_int_mm": 1,
}

_FLASH_OPS = {
    "flash": (("flash_attn", "_flash_attn_forward"),),
    "flash3": (("flash_attn_3", "fwd"), ("flash_attn_3", "_flash_attn_forward")),
}

# ``(in_features, out_features)`` of the base QKV projection whose forward is
# running, or ``None`` outside that call. Entered on whichever thread runs the
# block -- the forward's, and the autograd engine's for the recompute -- and
# left before that call returns, so the policy answers identically in both.
_projection_shape: ContextVar[tuple[int, int] | None] = ContextVar("h3_projection_shape", default=None)


@dataclass
class SavedActivations:
    """What the policy saved in one graph-carrying pass over the blocks."""

    attention: int = 0
    projection: int = 0

    def reset(self) -> None:
        self.attention = 0
        self.projection = 0


@contextmanager
def projection_region(in_features: int, out_features: int) -> Iterator[None]:
    """Mark the base QKV projection's own forward so the ``qkv`` level can find its matmul."""
    token = _projection_shape.set((int(in_features), int(out_features)))
    try:
        yield
    finally:
        _projection_shape.reset(token)


def op_name(func: Any) -> str:
    """``aten::mm`` for ``aten.mm.default``; the overload suffix is dropped."""
    return func.name().split(".", 1)[0]


def validate_checkpoint_keep(keep: str) -> str:
    if keep not in CHECKPOINT_KEEP_MODES:
        raise ValueError(f"H3 checkpoint keep mode must be one of {CHECKPOINT_KEEP_MODES}, got {keep!r}")
    return keep


def _op_is_registered(namespace: str, name: str) -> bool:
    try:
        getattr(getattr(torch.ops, namespace), name)
    except (AttributeError, RuntimeError):
        return False
    return True


def attention_kernel_is_visible(attention_mode: str) -> bool:
    """Whether the backend's attention forward can be a dispatcher op the policy saves.

    SDPA's fused backends are; whether a batch actually gets one is only known
    per pass (see :class:`SavedActivations`). The FlashAttention packages are
    only when they register their ``torch.library`` ops; a build that calls
    its CUDA binding directly runs below the dispatcher and keeps nothing.
    """
    if attention_mode == "torch":
        return True
    return any(_op_is_registered(namespace, name) for namespace, name in _FLASH_OPS.get(attention_mode, ()))


def _is_projection_matmul(name: str, args: tuple[Any, ...]) -> bool:
    shape = _projection_shape.get()
    if shape is None:
        return False
    operand = _PROJECTION_MATMUL_OPS.get(name)
    if operand is None or len(args) <= operand:
        return False
    weight = args[operand]
    return isinstance(weight, torch.Tensor) and weight.ndim >= 2 and tuple(weight.shape[-2:]) == shape


def checkpoint_policy(
    keep: str, saved: SavedActivations | None, ctx: Any, func: Any, *args: Any, **kwargs: Any
) -> CheckpointPolicy:
    """Selective-checkpoint policy: keep the attention output, and at ``qkv`` the base projection matmul."""
    del kwargs
    name = op_name(func)
    counting = saved is not None and not getattr(ctx, "is_recompute", False)
    if name in ATTENTION_OUTPUT_OPS:
        if counting:
            saved.attention += 1
        return CheckpointPolicy.MUST_SAVE
    if keep == "qkv" and _is_projection_matmul(name, args):
        if counting:
            saved.projection += 1
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def checkpoint_context_fn(keep: str, saved: SavedActivations | None = None) -> Callable[[], tuple[Any, Any]]:
    """``context_fn`` for ``torch.utils.checkpoint.checkpoint`` implementing ``keep``."""
    validate_checkpoint_keep(keep)
    if keep == "none":
        raise ValueError("keep='none' is plain checkpointing and needs no context_fn")
    return functools.partial(create_selective_checkpoint_contexts, functools.partial(checkpoint_policy, keep, saved))
