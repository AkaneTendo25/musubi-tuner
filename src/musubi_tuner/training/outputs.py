from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DiTOutput:
    pred: torch.Tensor
    target: torch.Tensor
    extra: dict = field(default_factory=dict)


def unpack_dit_output(output):
    if isinstance(output, DiTOutput):
        return output.pred, output.target
    return output
