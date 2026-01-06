from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class TextEncoderOutputs:
    hidden_state: torch.Tensor
    attention_mask: Optional[torch.Tensor]


class BaseTextEncoder:
    def text2tokens(self, text: str, data_type: str = "video") -> Any:
        raise NotImplementedError

    def encode(self, tokens: Any, data_type: str = "video") -> TextEncoderOutputs:
        raise NotImplementedError
