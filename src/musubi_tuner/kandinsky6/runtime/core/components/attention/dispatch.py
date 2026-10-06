"""Attention slot used by DiT blocks.

The kernels themselves live in ``musubi_tuner.kandinsky6.runtime.runtime.kernels``.
"""

from musubi_tuner.kandinsky6.runtime.runtime.kernels.attention_engine import SelfAttentionEngine, _sdpa, resolve_attention_engine

__all__ = ("SelfAttentionEngine", "_sdpa", "resolve_attention_engine")
