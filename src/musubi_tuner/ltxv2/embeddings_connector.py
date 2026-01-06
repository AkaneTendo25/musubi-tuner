from __future__ import annotations

import torch


def concat_av_context(context: torch.Tensor) -> torch.Tensor:
    return torch.cat([context, context], dim=-1)
