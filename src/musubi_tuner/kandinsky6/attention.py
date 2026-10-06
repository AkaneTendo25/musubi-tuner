from __future__ import annotations

import torch.nn.functional as F


def _sdpa(q, k, v, attn_mask=None):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(
        1, 2
    )


class SelfAttentionEngine:
    """Small upstream-compatible attention dispatcher using PyTorch SDPA."""

    def __init__(self, engine: str = "auto"):
        if engine not in ("auto", "sdpa"):
            raise ValueError(f"Vendored Kandinsky 6 supports attention_engine='auto' or 'sdpa', got {engine!r}")

    def get_attention(self):
        return _sdpa
