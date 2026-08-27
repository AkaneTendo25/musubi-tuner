from __future__ import annotations

import hashlib
import re

import torch


DOP_REWRITE_VERSION = 1


def rewrite_dop_caption(caption: str, trigger: str, class_prompt: str) -> str:
    """Replace one standalone trigger while avoiding duplicated class text."""
    trigger = trigger.strip()
    class_prompt = " ".join(class_prompt.split())
    if not trigger or not class_prompt:
        raise ValueError("DOP trigger and class prompt must be non-empty")
    pattern = rf"(?<!\w){re.escape(trigger)}(?!\w)"
    if re.search(pattern, caption) is None:
        raise ValueError(f"DOP trigger {trigger!r} is absent from caption {caption!r}")
    combined = rf"{pattern}\s+(?<!\w){re.escape(class_prompt)}(?!\w)"
    rewritten = re.sub(combined, class_prompt, caption)
    rewritten = re.sub(pattern, class_prompt, rewritten)
    return " ".join(rewritten.split())


def dop_config_identity(trigger: str, class_prompt: str) -> torch.Tensor:
    payload = f"h3-dop-v{DOP_REWRITE_VERSION}\0{trigger.strip()}\0{' '.join(class_prompt.split())}"
    return torch.tensor(list(hashlib.sha256(payload.encode("utf-8")).digest()), dtype=torch.uint8)
