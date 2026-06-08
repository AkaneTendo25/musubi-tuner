"""Inference-only VideoReward (VideoAlign) inferencer.

Vendored / slimmed from github.com/KwaiVGI/VideoAlign
(``flow_grpo/videoalign/inference.py`` :: ``VideoVLMRewardInference`` +
``train_reward.py`` :: ``create_model_and_processor`` / ``find_target_linear_names`` +
``utils.py`` :: ``load_model_from_checkpoint``). Reproduces the inferencer's behavior
for the released checkpoint while dropping the training-only deps (trl, datasets,
deepspeed, fire, pandas).

Kept faithful to the source: ``model_config.json`` parsing, the model+processor
build, the LoRA reconstruction + checkpoint load, ``prepare_batch`` (file:// video +
``build_prompt``), ``reward`` (logits -> {VQ, MQ, TA}, optional normalization,
``Overall`` = VQ+MQ+TA).

Deviations from the source, all forced by the runtime:
  * trl helpers are inlined away: the released checkpoint is NOT quantized, so
    ``get_quantization_config`` -> None (no ``device_map`` / ``quantization_config``).
  * peft handling is checkpoint-driven (exact live-adapter by default, peft-free optional).
    The released ``checkpoint-*/model.pth`` is an UNMERGED PEFT/LoRA state dict (keys
    ``base_model.model.*`` with ``base_layer`` + ``lora_A`` / ``lora_B``). The DEFAULT path
    rebuilds the LoRA wrapping (``wrap_lora=True``) and loads it directly -- the EXACT
    live-adapter form, bit-identical to VideoAlign (which runs the live adapter);
    ``peft`` is a light runtime dep on this path. For a peft-free deployment,
    :mod:`.merge_checkpoint` folds the LoRA ONCE offline (``merge_and_unload``) into a plain
    MERGED checkpoint, which the runtime then loads without importing ``peft``.
    ``VideoVLMRewardInference`` auto-detects the checkpoint kind (mmap key peek) and sets
    ``wrap_lora`` accordingly. (The merge is fp32-exact but folds in bf16 -- a rounding-level
    shift vs the live adapter; the live-adapter default avoids it.)
  * transformers>=4.5x renamed the Qwen2-VL submodules (vision tower under
    ``model.visual.*``, language model under ``model.language_model.*``). The model
    ``forward`` handles this (see ``model.Qwen2VLRewardModelBT``); the legacy checkpoint
    keys are remapped to the new layout via ``_remap_legacy_state_dict`` before loading
    (same style as the vendored ``hpsv3`` copy).
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import AutoProcessor

from .model import Qwen2VLRewardModelBT
from .prompt_template import build_prompt
from .vision_process import process_vision_info


@dataclass
class _DataConfig:
    """Subset of the upstream ``videoalign/data.py::DataConfig`` used at inference time."""

    max_frame_pixels: int = 240 * 320
    num_frames: Optional[float] = None
    fps: float = 2.0
    eval_dim: Union[str, List[str]] = "VQ"
    prompt_template_type: str = "none"
    sample_type: str = "uniform"


def _load_configs_from_json(config_path):
    """Mirror of the upstream ``inference.load_configs_from_json`` (returns only what we use)."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    data_config = dict(config_dict["data_config"])
    data_config.pop("meta_data", None)
    data_config.pop("data_dir", None)

    model_config = dict(config_dict["model_config"])
    peft_lora_config = dict(config_dict.get("peft_lora_config", {}) or {})
    inference_config = config_dict.get("inference_config", None)
    return data_config, model_config, peft_lora_config, inference_config


def _find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None):
    """Verbatim port of the upstream ``train_reward.find_target_linear_names`` (torch-only)."""
    lora_namespan_exclude = lora_namespan_exclude or []
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names


# The released ``model_config.json`` hard-codes the trainer's local base-model path
# (``/pfs/.../Qwen2-VL-2B-Instruct``), which does not exist outside the original box. When
# it is missing we fall back to the public HF id (the checkpoint's ``base_model``), which
# resolves from the HF cache (works with ``HF_HUB_OFFLINE=1``). Only the architecture/config
# skeleton is taken from here; the actual weights come from the merged checkpoint.
_FALLBACK_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def _resolve_base_model_name_or_path(model_config):
    name_or_path = model_config["model_name_or_path"]
    if os.path.sep in name_or_path or "/" in name_or_path:
        if not os.path.isdir(name_or_path):
            return _FALLBACK_BASE_MODEL
    return name_or_path


def _build_model_and_processor(model_config, peft_lora_config, dtype, cache_dir=None, disable_flash_attn2=False, wrap_lora=None):
    """Inference-only replacement for ``train_reward.create_model_and_processor``.

    Non-quantized path (trl ``get_quantization_config`` -> None).

    ``wrap_lora`` controls the PEFT ``get_peft_model`` reconstruction (which lines the
    parameter names up with an UNMERGED released checkpoint and is the ONLY place ``peft``
    is imported). The inference path passes ``wrap_lora=False`` so it never touches peft;
    the one-time :mod:`.merge_checkpoint` helper passes ``wrap_lora=True`` to rebuild the
    PEFT model for folding. When ``wrap_lora`` is ``None`` it defaults to
    ``peft_lora_config.lora_enable`` (legacy behaviour).

    ``disable_flash_attn2=False`` matches the original (``inference.py`` builds with
    ``disable_flash_attn2=False``), i.e. flash_attention_2 is used when ``flash_attn`` is
    installed; otherwise it falls back to sdpa. The attention backend changes the bf16
    reductions, so for score parity with the original this MUST stay False when flash_attn
    is available.
    """
    if wrap_lora is None:
        wrap_lora = peft_lora_config.get("lora_enable", False)

    torch_dtype = (
        model_config["torch_dtype"]
        if model_config.get("torch_dtype") in ["auto", None]
        else getattr(torch, model_config["torch_dtype"])
    )

    base_model_name_or_path = _resolve_base_model_name_or_path(model_config)
    processor = AutoProcessor.from_pretrained(base_model_name_or_path, padding_side="right", cache_dir=cache_dir)

    special_token_ids = None
    if model_config.get("use_special_tokens", False):
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    # Match the original: flash_attention_2 when flash_attn is installed, else sdpa.
    try:
        import flash_attn  # noqa: F401

        has_flash_attn = True
    except ImportError:
        has_flash_attn = False
    attn_impl = "flash_attention_2" if (not disable_flash_attn2 and has_flash_attn) else "sdpa"

    model = Qwen2VLRewardModelBT.from_pretrained(
        base_model_name_or_path,
        output_dim=model_config.get("output_dim", 1),
        reward_token=model_config.get("reward_token", "special"),
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        cache_dir=cache_dir,
        revision=model_config.get("model_revision", "main"),
        device_map=None,
        quantization_config=None,
        use_cache=False,
    )
    if model_config.get("use_special_tokens", False):
        model.resize_token_embeddings(len(processor.tokenizer))

    if dtype == torch.bfloat16:
        model.to(torch.bfloat16)
    elif dtype == torch.float16:
        model.to(torch.float16)

    # Reproduce the original LoRA wrapping so parameter names match an UNMERGED checkpoint.
    # peft is imported ONLY here, and ONLY when wrap_lora is requested -- i.e. by the
    # one-time merge_checkpoint helper. The inference path passes wrap_lora=False.
    if wrap_lora and peft_lora_config.get("lora_enable", False):
        from peft import LoraConfig, get_peft_model

        exclude = peft_lora_config.get("lora_namespan_exclude") or []
        if not peft_lora_config.get("vision_lora", False) and "visual" not in exclude:
            exclude = list(exclude) + ["visual"]
        target_modules = _find_target_linear_names(
            model, num_lora_modules=peft_lora_config.get("num_lora_modules", -1), lora_namespan_exclude=exclude
        )
        lora_cfg = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.get("lora_r", 16),
            lora_alpha=peft_lora_config.get("lora_alpha", 32),
            lora_dropout=peft_lora_config.get("lora_dropout", 0.05),
            task_type=peft_lora_config.get("lora_task_type", "CAUSAL_LM"),
            use_rslora=peft_lora_config.get("use_rslora", False),
            bias="none",
            modules_to_save=peft_lora_config.get("lora_modules_to_save"),
        )
        model = get_peft_model(model, lora_cfg)

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor


def _remap_legacy_state_dict(state_dict):
    """Remap a legacy-layout Qwen2-VL (PEFT) checkpoint to the transformers>=4.5x layout.

    transformers folded the vision tower + language model into ``Qwen2VLModel`` so the
    decoder now lives at ``model.language_model.*`` and the vision tower at
    ``model.visual.*``. Older checkpoints store the decoder as ``model.*`` and the vision
    tower as ``visual.*``. Keys may be PEFT-wrapped (``base_model.model.<inner>``) or
    plain (``<inner>``); we rewrite the ``<inner>`` part and leave ``rm_head`` / ``lm_head``
    untouched.
    """

    def remap_inner(inner):
        if inner.startswith(("rm_head.", "lm_head.")):
            return inner
        if inner.startswith("visual."):
            return "model." + inner
        if inner.startswith(("model.visual.", "model.language_model.")):
            return inner
        if inner.startswith("model."):
            return "model.language_model." + inner[len("model.") :]
        return inner

    out = {}
    for k, v in state_dict.items():
        if k.startswith("base_model.model."):
            out["base_model.model." + remap_inner(k[len("base_model.model.") :])] = v
        else:
            out[remap_inner(k)] = v
    return out


def _insert_adapter_name_into_state_dict(state_dict, adapter_name, parameter_prefix):
    """Verbatim from the upstream ``utils._insert_adapter_name_into_state_dict`` (LoRA fallback)."""
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def _resolve_checkpoint_path(checkpoint_dir, checkpoint_step):
    """Resolve a ``checkpoint-*`` path from either a direct dir or a parent dir of them."""
    if os.path.basename(os.path.normpath(checkpoint_dir)).startswith("checkpoint-"):
        return checkpoint_dir
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    if checkpoint_step is None or checkpoint_step == -1:
        return checkpoint_paths[0]
    cand = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
    return cand if cand in checkpoint_paths else checkpoint_paths[0]


def _is_unmerged_lora_state_dict(state_dict):
    """True if ``state_dict`` still carries PEFT/LoRA structure (needs offline merge)."""
    return any(("lora_" in k) or k.startswith("base_model.model.") or ("base_layer" in k) for k in state_dict)


def _load_model_from_checkpoint(model, checkpoint_dir, checkpoint_step, allow_unmerged=False):
    """Slimmed from the upstream ``utils.load_model_from_checkpoint``.

    Loads the full ``model.pth`` into the (plain) model. With ``allow_unmerged=False``
    (the inference default) a still-PEFT/LoRA ``model.pth`` raises a clear error pointing
    at the one-time :mod:`.merge_checkpoint` step, so the inference path never needs peft.
    The merge helper passes ``allow_unmerged=True`` (its ``model`` IS the PEFT-wrapped one).
    The separate ``adapter_model.safetensors`` + ``non_lora_state_dict.pth`` branch is kept
    for completeness. A ``RuntimeError`` on load triggers the transformers>=4.5x legacy key
    remap and a retry.
    """
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir, checkpoint_step)
    checkpoint_step = os.path.basename(os.path.normpath(checkpoint_path)).split("checkpoint-")[-1]

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")

    if os.path.exists(full_ckpt):
        model_state_dict = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        if not allow_unmerged and _is_unmerged_lora_state_dict(model_state_dict):
            raise RuntimeError(
                f"videoreward: {full_ckpt} is an UNMERGED PEFT/LoRA checkpoint (keys like "
                "'base_model.model.*' / 'lora_A' / 'lora_B'), which the peft-free inference path "
                "cannot load. Run the one-time merge first:\n"
                "    python -m musubi_tuner.ltx2_rewards.vendor.videoreward.merge_checkpoint "
                "--src <.../VideoReward> --dst <.../VideoReward-merged>\n"
                "then point --reward_args checkpoint_path= at the merged dir."
            )
        try:
            model.load_state_dict(model_state_dict)
        except RuntimeError:
            model.load_state_dict(_remap_legacy_state_dict(model_state_dict))
    else:
        import safetensors.torch

        lora_state_dict = safetensors.torch.load_file(lora_ckpt)
        non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu", weights_only=True)
        lora_state_dict = _insert_adapter_name_into_state_dict(lora_state_dict, adapter_name="default", parameter_prefix="lora_")
        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        try:
            model.load_state_dict(model_state_dict)
        except RuntimeError:
            model.load_state_dict(_remap_legacy_state_dict(model_state_dict))

    return model, checkpoint_step


class VideoVLMRewardInference:
    """Inference-only port of the upstream ``videoalign.inference.VideoVLMRewardInference``.

    ``load_from_pretrained`` may be either the parent dir containing ``model_config.json``
    and ``checkpoint-*`` subdirs, OR a ``checkpoint-*`` dir directly (then the sibling
    ``model_config.json`` one level up is used).
    """

    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device="cuda", dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        if not os.path.exists(config_path):
            # checkpoint-* dir passed directly: model_config.json lives one level up.
            config_path = os.path.join(os.path.dirname(os.path.normpath(load_from_pretrained)), "model_config.json")

        data_config_dict, model_config, peft_lora_config, inference_config = _load_configs_from_json(config_path)
        self.data_config = _DataConfig(
            max_frame_pixels=data_config_dict.get("max_frame_pixels", 240 * 320),
            num_frames=data_config_dict.get("num_frames", None),
            fps=data_config_dict.get("fps", 2.0),
            eval_dim=data_config_dict.get("eval_dim", "VQ"),
            prompt_template_type=data_config_dict.get("prompt_template_type", "none"),
            sample_type=data_config_dict.get("sample_type", "uniform"),
        )
        self.inference_config = inference_config

        # Auto-detect the checkpoint kind. The RELEASED checkpoint is an UNMERGED LoRA -> use the
        # EXACT live-adapter (peft) build, bit-identical to VideoAlign (which runs the live
        # adapter); peft is a light runtime dep on this path. A MERGED checkpoint (produced once by
        # .merge_checkpoint for deployments that prefer no peft) -> peft-free plain load. Peek the
        # state-dict keys via mmap (no tensor materialization) to decide.
        _ckpt = _resolve_checkpoint_path(load_from_pretrained, load_from_pretrained_step)
        _full = os.path.join(_ckpt, "model.pth")
        is_unmerged = False
        if os.path.exists(_full):
            _sd = torch.load(_full, map_location="cpu", weights_only=True, mmap=True)
            is_unmerged = _is_unmerged_lora_state_dict(_sd)
            del _sd
        model, processor = _build_model_and_processor(model_config, peft_lora_config, dtype=dtype, wrap_lora=is_unmerged)

        self.device = device
        model, _checkpoint_step = _load_model_from_checkpoint(
            model, load_from_pretrained, load_from_pretrained_step, allow_unmerged=is_unmerged
        )
        model.eval()

        self.model = model
        self.processor = processor
        self.model.to(self.device)

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        reward["VQ"] = (reward["VQ"] - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
        reward["MQ"] = (reward["MQ"] - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
        reward["TA"] = (reward["TA"] - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
        return reward

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}",
                                "max_pixels": max_pixels,
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type),
                            },
                        ],
                    },
                ]
                for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}",
                                "max_pixels": max_pixels,
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type),
                            },
                        ],
                    },
                ]
                for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """Score videos. Returns ``List[dict]`` with keys VQ, MQ, TA, Overall (VQ+MQ+TA)."""
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."

        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(return_dict=True, **batch)["logits"]

        rewards = [{"VQ": reward[0].item(), "MQ": reward[1].item(), "TA": reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]["Overall"] = rewards[i]["VQ"] + rewards[i]["MQ"] + rewards[i]["TA"]

        return rewards
