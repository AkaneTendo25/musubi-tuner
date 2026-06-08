"""Inference-only vendored copy of the VideoReward (VideoAlign) reward model.

VideoReward (KwaiVGI/VideoAlign, arxiv 2501.13918) is a Qwen2-VL-based VLM reward
that scores generated video on three axes -- Visual Quality (VQ), Motion Quality
(MQ), and Text Alignment (TA) -- via three special reward tokens. This copy is
faithful to the upstream ``videoalign`` inference code for the released
full-model checkpoint, with the heavy training dependencies removed. Public API
mirrors the source: ``VideoVLMRewardInference``.

Dependency surface (INFERENCE path): torch, transformers, torchvision, safetensors
(LoRA fallback only), qwen_vl_utils is NOT required (a local ``vision_process`` is
vendored), PIL, packaging, requests (http image paths only), and optionally flash_attn
(falls back to sdpa) / decord (falls back to the torchvision video reader). NO trl /
datasets / peft / deepspeed / fire / pandas.

``peft`` is used ONLY by the one-time, offline :mod:`.merge_checkpoint` helper, which
folds the released UNMERGED LoRA checkpoint into a plain merged checkpoint. After that
merge the inference path (:class:`VideoVLMRewardInference`) loads the merged checkpoint
with a plain ``load_state_dict`` and never imports peft.
"""

from .inferencer import VideoVLMRewardInference

__all__ = ["VideoVLMRewardInference"]
