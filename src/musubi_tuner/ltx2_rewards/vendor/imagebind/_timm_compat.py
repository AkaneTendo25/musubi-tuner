"""Minimal inlined replacements for the ``timm.models.layers`` helpers ImageBind uses.

Vendored verbatim from ``timm`` (Apache-2.0, Ross Wightman) so the imagebind package
does not depend on ``timm`` for inference. Only ``trunc_normal_`` (weight init) and
``DropPath`` (stochastic depth) are needed. Both are init-/train-time only: at
``eval()`` ``DropPath`` is an identity passthrough and ``trunc_normal_`` only runs when
randomly initializing weights, so a loaded checkpoint's forward pass is bit-identical to
the timm-based version.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from timm/PyTorch _no_grad_trunc_normal_ (method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf).
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Cut & paste from timm.layers.drop. See discussion at:
    https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L209
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Vendored verbatim from ``timm.layers.DropPath``.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
