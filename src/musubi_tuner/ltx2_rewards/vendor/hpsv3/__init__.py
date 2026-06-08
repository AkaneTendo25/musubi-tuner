"""Inference-only vendored copy of the HPSv3 reward model (MizzenAI/HPSv3).

Faithful to the external ``hpsv3`` package for non-quantized, non-LoRA inference,
with the heavy training dependencies removed. Public API mirrors the external
package: ``HPSv3RewardInferencer``.

Dependency surface: torch, transformers, torchvision, safetensors, PIL,
huggingface_hub (only when ``checkpoint_path`` is not provided), and optionally
flash_attn (falls back to sdpa). NO trl / datasets / fire / peft / omegaconf /
matplotlib / tensorboard / timm / deepspeed.
"""

from .inferencer import HPSv3RewardInferencer

__all__ = ["HPSv3RewardInferencer"]
