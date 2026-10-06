"""Load-time quantization. One module per method.

A model factory chooses which component receives a method. Removing a method
means reloading that component.
"""

from .nf4 import qwen_load_kwargs

__all__ = ("qwen_load_kwargs",)
