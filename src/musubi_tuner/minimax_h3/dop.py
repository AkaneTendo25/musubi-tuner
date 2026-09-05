from __future__ import annotations

import hashlib
import re

import torch


DOP_REWRITE_VERSION = 1


DOP_CAPTION_MODES = ("class", "bare")


def rewrite_dop_caption(caption: str, trigger: str, class_prompt: str, mode: str = "class") -> str:
    """Rewrite one caption for the DOP reference forward.

    ``class`` replaces the standalone trigger with the class phrase, avoiding duplicated class text.
    ``bare`` removes the trigger and nothing else, so the reference prompt is the caption without the
    concept's name; punctuation and whitespace at the seam are tidied.
    """
    if mode not in DOP_CAPTION_MODES:
        raise ValueError(f"unknown DOP caption mode {mode!r}")
    trigger = trigger.strip()
    class_prompt = " ".join(class_prompt.split())
    if not trigger:
        raise ValueError("DOP trigger must be non-empty")
    pattern = rf"(?<!\w){re.escape(trigger)}(?!\w)"
    if re.search(pattern, caption) is None:
        raise ValueError(f"DOP trigger {trigger!r} is absent from caption {caption!r}")
    if mode == "bare":
        if class_prompt:
            raise ValueError("DOP caption mode 'bare' takes no class prompt")
        # Mark the trigger, then close the seam: punctuation on both sides collapses to the left one
        # ("First. sks. Second" -> "First. Second"), a leading trigger takes its punctuation with it
        # ("sks... Next" -> "Next"), brackets or quotes emptied by the removal disappear, and
        # punctuation elsewhere in the caption is left alone.
        marker = "\x00"
        # A possessive attached to the trigger goes with it ("sks's portrait" -> "portrait").
        rewritten = re.sub(rf"{pattern}(?:['\u2019]s)?(?!\w)", marker, caption)
        # Paired separators around the trigger collapse to one: "a -- sks -- b", "a / sks / b", em and en dashes.
        rewritten = re.sub(rf"(\s*(?:--|[\u2013\u2014/|]))\s*{marker}\s*(?:--|[\u2013\u2014/|])", r"\1", rewritten)
        rewritten = re.sub(rf"([,.;:!?])\s*{marker}\s*[,.;:!?]+", r"\1", rewritten)
        rewritten = re.sub(rf"^\s*{marker}\s*[,.;:!?]*", "", rewritten)
        rewritten = re.sub(rf"\s*{marker}\s*", " ", rewritten)
        rewritten = re.sub(r"\(\s*\)|\[\s*\]|\"\s*\"|'\s*'", "", rewritten)
        rewritten = re.sub(r"\s+([,.;:!?])", r"\1", rewritten)
        rewritten = " ".join(rewritten.split()).strip()
        if not rewritten:
            raise ValueError(f"DOP caption {caption!r} is empty without its trigger")
        return rewritten
    if not class_prompt:
        raise ValueError("DOP caption mode 'class' requires a class prompt")
    combined = rf"{pattern}\s+(?<!\w){re.escape(class_prompt)}(?!\w)"
    rewritten = re.sub(combined, class_prompt, caption)
    rewritten = re.sub(pattern, class_prompt, rewritten)
    return " ".join(rewritten.split())


def dop_config_identity(trigger: str, class_prompt: str, mode: str = "class") -> torch.Tensor:
    if mode not in DOP_CAPTION_MODES:
        raise ValueError(f"unknown DOP caption mode {mode!r}")
    payload = f"h3-dop-v{DOP_REWRITE_VERSION}\0{trigger.strip()}\0{' '.join(class_prompt.split())}"
    if mode != "class":
        # The class payload keeps its v1 bytes so existing caches stay valid.
        payload += f"\0mode={mode}"
    return torch.tensor(list(hashlib.sha256(payload.encode("utf-8")).digest()), dtype=torch.uint8)
