"""Inference-only vendored copy of the VideoScore2 reward model.

VideoScore2 (TIGER-Lab/VideoScore2, arxiv 2509.22799) is a generative Qwen2.5-VL-7B
VLM that scores a generated video on three dimensions -- visual quality (VQ),
text-to-video alignment (TA), and physical/common-sense consistency (PC) -- by
generating a chain-of-thought rationale ending in a ``visual quality: N; ...`` block
and reading a soft score off the digit logits. Unlike VideoReward / HPSv3 (which use a
trained reward head), VideoScore2 is a *stock* ``Qwen2_5_VLForConditionalGeneration``
checkpoint (4 safetensors shards, no custom modeling file, no LoRA, no special reward
tokens), so this copy needs no ``model.py``: it loads the model verbatim and reproduces
the model-card scoring pipeline (query template, ``.generate`` with
``output_scores=True``, the soft-score parsing) in-process.

Dependency surface: torch, transformers (>=4.56, the workbox pin), torchvision,
safetensors (the shard loader transformers uses), PIL, packaging, requests (http image
paths only) and numpy. ``qwen_vl_utils`` is PREFERRED when installed (it is exactly what
the model card uses, so it gives byte-identical preprocessing); a local
Qwen2.5-VL ``vision_process`` is vendored as a self-contained fallback. NO trl /
datasets / peft / deepspeed / fire. ``decord`` / ``torchcodec`` are optional video reader
backends (the vendored fallback drops to the torchvision reader).

Public API: ``VideoScore2Inferencer``.
"""

from .scorer import VideoScore2Inferencer

__all__ = ["VideoScore2Inferencer"]
