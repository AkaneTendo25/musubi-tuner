# Runtime helpers shared by latent-upscaler model components.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable


def forward_with_checkpointing(
    module: Callable[..., Tensor],
    *inputs: Tensor,
    use_checkpointing: bool = False,
) -> Tensor:
    """Run a callable directly or through non-reentrant activation checkpointing."""
    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(module, *inputs, use_reentrant=False)
    return module(*inputs)
