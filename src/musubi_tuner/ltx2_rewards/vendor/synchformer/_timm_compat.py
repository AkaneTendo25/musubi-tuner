"""Minimal inlined replacements for the two ``timm.layers`` helpers Synchformer uses.

Vendored verbatim from ``timm`` (Apache-2.0, Ross Wightman) so the synchformer
package does not depend on ``timm`` for inference. Only ``trunc_normal_`` and
``to_2tuple`` are needed (weight init + patch-size tupling); both are init-time only
and do not affect the loaded-checkpoint forward pass.
"""

from __future__ import annotations

import collections.abc
import math
import warnings
from itertools import repeat

import torch


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


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
