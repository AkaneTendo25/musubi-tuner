"""YuE2 checkpoint layouts (HF, ComfyUI bf16 / pre-quantized int8, native), key remap, quantized loading, and
VAE / tokenizer extraction.

The merge-input key contract below is implemented here so that the loader and the LoRA format converters are tested
against the same rule.

Merge-input contract
--------------------
``load_yue2_model(lora_weights=[sd, ...], lora_multipliers=[m, ...])`` merges each ``sd`` into the base weights at
load time (before any quantization). Every ``sd`` must be a **native fused** dict holding only these keys:

* LoRA on block linears (fused layout, one adapter per fused Linear)::

      lora_unet_{ar|nar}_blocks_{N}_{self_attn_qkv_proj|self_attn_o_proj|mlp_gate_up_proj|mlp_down_proj}.lora_down.weight  [r, in]
      lora_unet_{ar|nar}_blocks_{N}_{...}.lora_up.weight    [out, r]
      lora_unet_{ar|nar}_blocks_{N}_{...}.alpha             scalar (required; scale = alpha / r)

  ``ΔW = m * (alpha / r) * up @ down`` is added to ``{ar|nar}.blocks.N.self_attn.qkv_proj.weight`` etc. Split
  (per q/k/v, per gate/up) adapters are converted to block-diagonal fused ones by the LoRA format converter before
  they reach the loader.
* Diffs on the NAR I/O modules::

      lora_unet_nar_{vae2llm|llm2vae|time_embedder_mlp_0|time_embedder_mlp_2}.diff    [out, in] (required)
      lora_unet_nar_{...}.diff_b  [out] (optional)

  ``W += m * diff``, ``b += m * diff_b`` on ``nar.vae2llm`` etc., computed in fp32 on the unquantized tensors. An I/O
  LoRA is passed as its product (``diff = (alpha / r) * up @ down``) and fully replaced ("full") I/O weights as
  ``full - base`` (``yue2_lora_formats.to_native(..., base_io=read_base_io(dit))``); neither form reaches the loader
  otherwise.

Anything else (split ``lora_down.0.weight``, key names of other LoRA file formats, I/O ``lora_down``, ``full``
weights) is rejected by ``check_merge_input``. Pre-quantized int8 checkpoints never accept merge input (merge before
quantization only).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

import torch

from musubi_tuner.modules.comfy_quant_utils import COMFY_QUANT_SUFFIX, COMFY_WEIGHT_SCALE_SUFFIX
from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import (
    ConvRotInt8Quantizer,
    apply_convrot_int8_monkey_patch,
    parse_comfy_quant_spec,
)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, WeightTransformHooks, get_split_weight_filenames
from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Int8Embedding, YuE2Model

logger = logging.getLogger(__name__)

YUE2_QUANT_TARGET_KEYS = ["ar.blocks.", "nar.blocks."]
YUE2_QUANT_EXCLUDE_KEYS = ["norm"]

# native LoRA module suffix -> module path inside a YuE2Block / YuE2NAR
YUE2_LORA_BLOCK_MODULES = {
    "self_attn_qkv_proj": "self_attn.qkv_proj",
    "self_attn_o_proj": "self_attn.o_proj",
    "mlp_gate_up_proj": "mlp.gate_up_proj",
    "mlp_down_proj": "mlp.down_proj",
}
YUE2_IO_MODULES = {
    "vae2llm": "vae2llm",
    "llm2vae": "llm2vae",
    "time_embedder_mlp_0": "time_embedder.mlp.0",
    "time_embedder_mlp_2": "time_embedder.mlp.2",
}

_BLOCK_KEY_RE = re.compile(
    r"^lora_unet_(ar|nar)_blocks_(\d+)_(" + "|".join(YUE2_LORA_BLOCK_MODULES) + r")\.(lora_down\.weight|lora_up\.weight|alpha)$"
)
_IO_KEY_RE = re.compile(r"^lora_unet_nar_(" + "|".join(YUE2_IO_MODULES) + r")\.(diff|diff_b)$")


class CheckpointLayout(str, Enum):
    HF = "hf"
    COMFY = "comfy"
    NATIVE = "native"


def merge_input_module_name(lora_name: str) -> str:
    """Native LoRA module name -> model module path, e.g. ``lora_unet_ar_blocks_3_self_attn_qkv_proj`` ->
    ``ar.blocks.3.self_attn.qkv_proj`` and ``lora_unet_nar_time_embedder_mlp_0`` -> ``nar.time_embedder.mlp.0``."""
    m = _BLOCK_KEY_RE.match(lora_name + ".alpha")
    if m is not None:
        branch, index, module = m.group(1), m.group(2), m.group(3)
        return f"{branch}.blocks.{index}.{YUE2_LORA_BLOCK_MODULES[module]}"
    m = _IO_KEY_RE.match(lora_name + ".diff")
    if m is not None:
        return f"nar.{YUE2_IO_MODULES[m.group(1)]}"
    raise ValueError(f"not a native YuE2 merge-input module name: {lora_name}")


def check_merge_input(sd: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """Validate a merge-input dict against the merge-input contract (module docstring).

    Returns ``{lora_name: "lora" | "diff"}``. Raises ``ValueError`` naming the offending keys on any unknown key, a
    LoRA module missing ``lora_down``/``lora_up``/``alpha``, rank or shape mismatches, or a diff without ``diff``.
    """
    unknown = []
    parts: dict[str, dict[str, torch.Tensor]] = {}
    kinds: dict[str, str] = {}
    for key, value in sd.items():
        m = _BLOCK_KEY_RE.match(key)
        if m is not None:
            name, part, kind = key[: -len(m.group(4)) - 1], m.group(4), "lora"
        else:
            m = _IO_KEY_RE.match(key)
            if m is None:
                unknown.append(key)
                continue
            name, part, kind = key[: -len(m.group(2)) - 1], m.group(2), "diff"
        parts.setdefault(name, {})[part] = value
        kinds[name] = kind
    if unknown:
        shown = ", ".join(sorted(unknown)[:8]) + (" ..." if len(unknown) > 8 else "")
        raise ValueError(f"YuE2 merge input must be a native fused dict; {len(unknown)} unexpected keys: {shown}")

    for name, kind in kinds.items():
        p = parts[name]
        if kind == "lora":
            missing = [k for k in ("lora_down.weight", "lora_up.weight", "alpha") if k not in p]
            if missing:
                raise ValueError(f"YuE2 merge input: {name} is missing {missing}")
            down, up, alpha = p["lora_down.weight"], p["lora_up.weight"], p["alpha"]
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(
                    f"YuE2 merge input: {name} needs lora_down [r, in] and lora_up [out, r], got {tuple(down.shape)} and {tuple(up.shape)}"
                )
            if alpha.numel() != 1:
                raise ValueError(f"YuE2 merge input: {name}.alpha must be a scalar")
        else:
            if "diff" not in p:
                raise ValueError(f"YuE2 merge input: {name} has diff_b without diff")
            diff = p["diff"]
            if diff.ndim != 2:
                raise ValueError(f"YuE2 merge input: {name}.diff must be [out, in], got {tuple(diff.shape)}")
            if "diff_b" in p and tuple(p["diff_b"].shape) != (diff.shape[0],):
                raise ValueError(f"YuE2 merge input: {name}.diff_b must be [{diff.shape[0]}], got {tuple(p['diff_b'].shape)}")
    return kinds


# region layouts and key remap

_DROP = object()  # split_hook sentinel: the key is not loaded

_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_HF_FUSED_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|nar_self_attn|mlp|nar_mlp)\.(q|k|v|gate|up)_proj\.weight$")
_FUSED_ROLES = {"self_attn": ("q", "k", "v"), "mlp": ("gate", "up")}
_FUSED_NATIVE = {"self_attn": "self_attn.qkv_proj", "mlp": "mlp.gate_up_proj"}
_NAR_IO_PREFIXES = ("vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed.")
YUE2_IO_NAMES = ("vae2llm", "llm2vae", "time_embedder.mlp.0", "time_embedder.mlp.2")
COMFY_TOKENIZER_KEY = "text_encoders.yue2_tokenizer_json"
# pre-quantized modules that are always dequantized at load; ar.lm_head depends on prequant_lm_head
PREQUANT_ALWAYS_DEQUANT = ("nar.llm2vae", "nar.time_embedder.mlp.0", "nar.time_embedder.mlp.2")


def _model_files(path: str) -> list[str]:
    return get_split_weight_filenames(path) or [path]


def _header_keys(path: str) -> list[str]:
    keys = []
    for file in _model_files(path):
        with MemoryEfficientSafeOpen(file) as f:
            keys.extend(f.keys())
    return keys


def detect_layout(path: str) -> tuple[CheckpointLayout, bool]:
    """``(layout, is_prequantized_convrot)`` from the safetensors header only (no tensor reads)."""
    keys = _header_keys(path)
    prequant = any(k.endswith(COMFY_QUANT_SUFFIX) for k in keys)
    if any(k.startswith(("model.diffusion_model.", "text_encoders.")) for k in keys):
        return CheckpointLayout.COMFY, prequant
    if any(k.startswith("model.layers.") for k in keys):
        return CheckpointLayout.HF, prequant
    if any(k.startswith(("ar.blocks.", "nar.blocks.")) for k in keys):
        return CheckpointLayout.NATIVE, prequant
    raise ValueError(f"{path} is not a YuE2 checkpoint (no Hugging Face, ComfyUI or native YuE2 keys)")


def _check_layer(index: str, config: YuE2Config, key: str) -> None:
    if int(index) >= config.num_layers:
        raise ValueError(f"YuE2 checkpoint key {key} has layer {index} >= num_layers {config.num_layers}")


def _hf_native_key(key: str, config: YuE2Config):
    """Native key for an HF key, ``None`` for a q/k/v or gate/up part that is fused, ``_DROP`` to skip."""
    m = _HF_LAYER_RE.match(key)
    if m is not None:
        index, rest = m.group(1), m.group(2)
        _check_layer(index, config, key)
        if _HF_FUSED_RE.match(key):
            return None
        branch = "ar"
        for src, dst in (
            ("nar_self_attn.", "self_attn."),
            ("nar_mlp.", "mlp."),
            ("nar_input_layernorm.", "input_layernorm."),
            ("nar_pre_mlp_layernorm.", "post_attention_layernorm."),
        ):
            if rest.startswith(src):
                branch, rest = "nar", dst + rest[len(src) :]
                break
        return f"{branch}.blocks.{index}.{rest}"
    if key.startswith("model.embed_tokens."):
        return "ar.embed_tokens." + key[len("model.embed_tokens.") :]
    if key.startswith("model.norm."):
        return "norm." + key[len("model.norm.") :]
    if key.startswith("lm_head."):
        return "ar." + key
    if key.startswith(_NAR_IO_PREFIXES):
        return "nar." + key
    raise ValueError(f"unexpected key in a Hugging Face YuE2 checkpoint: {key}")


_COMFY_RENAMES = (
    ("text_encoders.model.layers.", "ar.blocks."),
    ("text_encoders.model.embed_tokens.", "ar.embed_tokens."),
    ("text_encoders.model.norm.", "norm."),
    ("text_encoders.model.lm_head.", "ar.lm_head."),
    ("text_encoders.lm_head.", "ar.lm_head."),
    ("model.diffusion_model.model.layers.", "nar.blocks."),
) + tuple(("model.diffusion_model." + p, "nar." + p) for p in _NAR_IO_PREFIXES)
# the NAR copy of the final norm duplicates text_encoders.model.norm; vae.* and the tokenizer are read separately
_COMFY_DROPS = ("model.diffusion_model.model.norm.", "vae.", COMFY_TOKENIZER_KEY)


def _comfy_native_key(key: str, config: YuE2Config):
    if key.startswith(_COMFY_DROPS):
        return _DROP
    for src, dst in _COMFY_RENAMES:
        if key.startswith(src):
            native = dst + key[len(src) :]
            if dst.endswith("blocks."):
                _check_layer(native[len(dst) :].split(".", 1)[0], config, key)
            return native
    raise ValueError(f"unexpected key in a ComfyUI YuE2 checkpoint: {key}")


def _make_split_hook(rename):
    def split_hook(key: str, tensor: Optional[torch.Tensor]):
        native = rename(key)
        if native is None:
            return None, None  # fused: handled by the concat hook
        if native is _DROP:
            return [], []
        return [native], (None if tensor is None else [tensor])

    return split_hook


def _hf_concat_hook(key: str, tensors: Optional[dict[str, torch.Tensor]]):
    m = _HF_FUSED_RE.match(key)
    if m is None:
        return None, None
    index, module = m.group(1), m.group(2)
    branch = "nar" if module.startswith("nar_") else "ar"
    base = module[4:] if branch == "nar" else module
    native = f"{branch}.blocks.{index}.{_FUSED_NATIVE[base]}.weight"
    if tensors is None:
        return native, None
    # assemble by role, never by the (alphabetical) header order: q|k|v and gate|up
    by_role = {}
    for part_key, tensor in tensors.items():
        pm = _HF_FUSED_RE.match(part_key)
        if pm is None or pm.group(1) != index or pm.group(2) != module or pm.group(3) in by_role:
            raise ValueError(f"unexpected part {part_key} for fused YuE2 weight {native}")
        by_role[pm.group(3)] = tensor
    roles = _FUSED_ROLES[base]
    missing = [r for r in roles if r not in by_role]
    if missing or len(by_role) != len(roles):
        raise ValueError(f"fused YuE2 weight {native} is missing parts {missing}")
    return native, torch.cat([by_role[r] for r in roles], dim=0)


def build_weight_transform_hooks(layout: CheckpointLayout, config: YuE2Config) -> Optional[WeightTransformHooks]:
    """Source -> native key remap. ``split_hook`` renames/drops keys and returns ``None`` for every
    key that is fused; ``concat_hook`` assembles fused tensors **by role** (``q, k, v`` and ``gate, up``), never by the
    header (alphabetical) order. ``None`` for the native layout."""
    layout = CheckpointLayout(layout)
    if layout == CheckpointLayout.HF:
        return WeightTransformHooks(split_hook=_make_split_hook(lambda k: _hf_native_key(k, config)), concat_hook=_hf_concat_hook)
    if layout == CheckpointLayout.COMFY:
        return WeightTransformHooks(split_hook=_make_split_hook(lambda k: _comfy_native_key(k, config)))
    return None


# endregion

# region model loading


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


def _keeps_own_dtype(key: str, fp8: bool = False) -> bool:
    # ConvRot and int8-embedding scales stay fp32; fp8 scales follow the compute dtype (the fp8 forward dequantizes to it)
    return (key.endswith(".scale_weight") and not fp8) or key == "ar.embed_tokens.scale"


def _dequantize_rows(q: torch.Tensor, scale: torch.Tensor, group_size: int, calc_device, dtype: torch.dtype) -> torch.Tensor:
    """``dequantize_int8_convrot_weight`` in row chunks (bounded fp32 temporaries for the 184704-row heads)."""
    out = torch.empty(q.shape, dtype=dtype, device=q.device)
    rows = max(1, (1 << 24) // max(1, q.shape[1]))
    for start in range(0, q.shape[0], rows):
        end = min(start + rows, q.shape[0])
        part = dequantize_int8_convrot_weight(q[start:end].to(calc_device), scale[start:end].to(calc_device).float(), group_size)
        out[start:end] = part.to(device=out.device, dtype=dtype)
    return out


def _split_merge_input(lora_weights, lora_multipliers):
    """Merge-input dicts -> (LoRA dicts for the merge hook, their multipliers, I/O diffs [(module, diff, diff_b, m)])."""
    lora_sds, lora_mults, io_deltas = [], [], []
    for i, sd in enumerate(lora_weights or []):
        m = float(lora_multipliers[i]) if lora_multipliers is not None and len(lora_multipliers) > i else 1.0
        kinds = check_merge_input(sd)
        lora_part = {}
        for name, kind in kinds.items():
            if kind == "lora":
                for suffix in ("lora_down.weight", "lora_up.weight", "alpha"):
                    lora_part[f"{name}.{suffix}"] = sd[f"{name}.{suffix}"]
            else:
                io_deltas.append((merge_input_module_name(name), sd[f"{name}.diff"], sd.get(f"{name}.diff_b"), m))
        if lora_part:
            lora_sds.append(lora_part)
            lora_mults.append(m)
    return lora_sds, lora_mults, io_deltas


def load_yue2_model(
    path: str,
    *,
    device,
    loading_device,
    dtype: torch.dtype = torch.bfloat16,
    config: YuE2Config = YuE2Config(),
    attn_mode: str = "torch",
    split_attn: bool = False,
    fp8_scaled: bool = False,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    quantize_lm_head: bool = False,
    prequant_lm_head: str = "auto",
    lm_head_needed: bool = True,
    lora_weights: Optional[list[dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    disable_numpy_memmap: bool = False,
) -> YuE2Model:
    """Build ``YuE2Model`` on meta, stream the checkpoint through the key remap (merging ``lora_weights`` per the merge-input
    contract, then fp8/ConvRot quantizing ``ar.blocks.``/``nar.blocks.``), and assign-load it.

    Pre-quantized int8 files: ``nar.llm2vae`` and ``nar.time_embedder.mlp.{0,2}`` are always dequantized to bf16,
    ``ar.lm_head`` per ``prequant_lm_head`` (``auto`` = dequantize iff ``lm_head_needed``), ``ar.embed_tokens`` becomes a
    ``YuE2Int8Embedding``. ``lora_weights`` or ``fp8_scaled`` with a pre-quantized file raises. After load no tensor is
    on meta and every non-quantized floating tensor is ``dtype`` (bf16 for training and generation).
    """
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    if fp8_scaled and convrot_int8:
        raise ValueError("fp8_scaled and convrot_int8 are mutually exclusive")
    if prequant_lm_head not in ("auto", "keep", "dequant"):
        raise ValueError(f"invalid prequant_lm_head: {prequant_lm_head}")
    layout, prequant = detect_layout(path)
    if prequant and fp8_scaled:
        raise ValueError(f"{path} is a pre-quantized ConvRot int8 checkpoint; --fp8_scaled needs a bf16 checkpoint")
    if prequant and lora_weights:
        raise ValueError(f"cannot merge weights into the pre-quantized int8 checkpoint {path}; use a bf16 checkpoint")
    hooks = build_weight_transform_hooks(layout, config)
    logger.info(
        f"Loading YuE2 model from {path} (layout {layout.value}"
        + (", pre-quantized int8" if prequant else "")
        + (", fp8 scaled" if fp8_scaled else "")
        + (", ConvRot int8" if convrot_int8 else "")
        + (f", {len(lora_weights)} merge inputs" if lora_weights else "")
        + ")"
    )

    with torch.device("meta"):
        model = YuE2Model(config)

    lora_sds, lora_mults, io_deltas = _split_merge_input(lora_weights, lora_multipliers)
    targets = list(YUE2_QUANT_TARGET_KEYS) + (["ar.lm_head."] if quantize_lm_head else [])
    quantizer = ConvRotInt8Quantizer(targets, list(YUE2_QUANT_EXCLUDE_KEYS)) if (convrot_int8 or prequant) else None
    sd = load_safetensors_with_lora_and_fp8(
        model_files=path,
        lora_weights_list=lora_sds or None,
        lora_multipliers=lora_mults or None,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=None if (fp8_scaled or quantizer is not None) else dtype,
        target_keys=targets if fp8_scaled else None,
        exclude_keys=list(YUE2_QUANT_EXCLUDE_KEYS) if fp8_scaled else None,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=hooks,
        quantizer=quantizer,
    )

    if prequant:
        dequant_lm_head = prequant_lm_head == "dequant" or (prequant_lm_head == "auto" and lm_head_needed)
        modules = PREQUANT_ALWAYS_DEQUANT + (("ar.lm_head",) if dequant_lm_head else ())
        for module in modules:
            scale_key = module + ".scale_weight"
            if scale_key not in sd:
                continue
            group_size = quantizer.module_groupsizes.pop(module)
            sd[module + ".weight"] = _dequantize_rows(sd[module + ".weight"], sd.pop(scale_key), group_size, device, dtype)
        if "ar.embed_tokens.scale_weight" in sd:
            group_size = quantizer.module_groupsizes.pop("ar.embed_tokens")
            with torch.device("meta"):
                model.ar.embed_tokens = YuE2Int8Embedding(config.vocab_size, config.hidden_size, group_size, dtype=dtype)
            sd["ar.embed_tokens.scale"] = sd.pop("ar.embed_tokens.scale_weight")

    for module, diff, diff_b, m in io_deltas:
        for suffix, delta in ((".weight", diff), (".bias", diff_b)):
            if delta is None:
                continue
            key = module + suffix
            if key not in sd or module + ".scale_weight" in sd:
                raise ValueError(f"cannot apply a diff to {key}: missing or quantized in {path}")
            base = sd[key]
            if tuple(base.shape) != tuple(delta.shape):
                raise ValueError(f"diff for {key} has shape {tuple(delta.shape)}, base is {tuple(base.shape)}")
            sd[key] = (base.float() + m * delta.to(device=base.device, dtype=torch.float32)).to(base.dtype)

    for key, value in sd.items():
        if (
            value.is_floating_point()
            and not _is_fp8(value.dtype)
            and not _keeps_own_dtype(key, fp8_scaled)
            and value.dtype != dtype
        ):
            sd[key] = value.to(dtype)

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        base_quant = "fp8_scaled"
    elif quantizer is not None and any(k.endswith(".scale_weight") for k in sd):
        apply_convrot_int8_monkey_patch(model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=quantizer.module_groupsizes)
        # int8 tensors cannot become Parameters that require grad; the base is frozen anyway
        model.requires_grad_(False)
        base_quant = "prequant_int8" if prequant else "convrot_int8"
    else:
        base_quant = "prequant_int8" if prequant else "bf16"

    if loading_device.type != "cpu":
        for key in sd:
            sd[key] = sd[key].to(loading_device)
    model.load_state_dict(sd, strict=True, assign=True)
    del sd

    on_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if on_meta:
        raise RuntimeError(f"YuE2 load left tensors on meta: {on_meta[:8]}")
    wrong = [
        f"{n}:{t.dtype}"
        for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.is_floating_point() and not _is_fp8(t.dtype) and not _keeps_own_dtype(n, fp8_scaled) and t.dtype != dtype
    ]
    if wrong:
        raise RuntimeError(f"YuE2 tensors not in {dtype} after load: {wrong[:8]}")

    model.set_attention(attn_mode, split_attn)
    model.checkpoint_layout = layout.value
    model.base_quant = base_quant
    model.is_convrot_int8 = bool(getattr(model, "is_convrot_int8", False))
    model.eval()
    return model


def _find_tensor_file(path: str, key: str) -> Optional[str]:
    for file in _model_files(path):
        with MemoryEfficientSafeOpen(file) as f:
            if key in f.keys():
                return file
    return None


def _read_tensor(path: str, key: str) -> Optional[torch.Tensor]:
    file = _find_tensor_file(path, key)
    if file is None:
        return None
    with MemoryEfficientSafeOpen(file) as f:
        return f.get_tensor(key)


def read_base_io(path: str) -> dict[str, torch.Tensor]:
    """The four base NAR I/O modules of a checkpoint as fp32 tensors under native names
    ``nar.{vae2llm,llm2vae,time_embedder.mlp.0,time_embedder.mlp.2}.{weight,bias}`` (dequantized when pre-quantized).
    Header-indexed reads only; used by the LoRA format converter to turn "full" I/O weights into diffs."""
    layout, _ = detect_layout(path)
    prefix = {CheckpointLayout.HF: "", CheckpointLayout.COMFY: "model.diffusion_model.", CheckpointLayout.NATIVE: "nar."}[layout]
    out = {}
    for name in YUE2_IO_NAMES:
        src = prefix + name
        weight = _read_tensor(path, src + ".weight")
        if weight is None:
            raise KeyError(f"{path} has no {src}.weight")
        spec = _read_tensor(path, src + COMFY_QUANT_SUFFIX)
        if spec is not None:
            group_size = parse_comfy_quant_spec(src + COMFY_QUANT_SUFFIX, spec)["convrot_groupsize"]
            weight = dequantize_int8_convrot_weight(weight, _read_tensor(path, src + COMFY_WEIGHT_SCALE_SUFFIX).float(), group_size)
        out[f"nar.{name}.weight"] = weight.float()
        bias = _read_tensor(path, src + ".bias")
        if bias is not None:
            out[f"nar.{name}.bias"] = bias.float()
    return out


def read_tokenizer_json(path: str) -> bytes:
    """The embedded ``text_encoders.yue2_tokenizer_json`` uint8 tensor of a ComfyUI checkpoint as bytes."""
    tensor = _read_tensor(path, COMFY_TOKENIZER_KEY)
    if tensor is None:
        raise KeyError(f"{path} has no embedded tokenizer ({COMFY_TOKENIZER_KEY})")
    return bytes(tensor.numpy().tobytes())


# endregion

# region VAE and tokenizer

_ST_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16", "F64": "float64"}


def _vae_configs(config_dir: Optional[Path]) -> tuple[dict, dict]:
    from musubi_tuner.yue2.yue2_vae import default_vae_configs

    encoder_config, decoder_config = default_vae_configs()
    if config_dir is not None and (config_dir / "config.json").is_file():
        cfg = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        encoder_config = cfg.get("encoder_config") or encoder_config
        decoder_config = cfg.get("decoder_config") or decoder_config
    return encoder_config, decoder_config


def load_yue2_vae(
    vae_path: Optional[str],
    dit_path: Optional[str],
    *,
    device="cpu",
    decoder_only: bool = False,
    allow_fp16_source: bool = True,
    vae_configs: Optional[tuple[dict, dict]] = None,
):
    """FP32 ``YuE2VAE`` from a ``m-a-p/YuE2-Vae`` directory, an unprefixed or ``vae.``-prefixed file, or (``vae_path``
    None) the ``vae.*`` tensors of a ComfyUI ``dit_path``. A non-F32 source raises unless ``allow_fp16_source``;
    ``vae.source_dtype`` records it. ``vae_configs`` (encoder, decoder) overrides the config.json / released defaults."""
    from musubi_tuner.yue2.yue2_vae import YuE2VAE

    config_dir = None
    if vae_path is not None:
        p = Path(vae_path)
        if p.is_dir():
            config_dir = p
            index = p / "model.safetensors.index.json"
            if index.is_file():
                files = sorted({str(p / name) for name in json.loads(index.read_text())["weight_map"].values()})
            else:
                files = [str(p / "model.safetensors")]
        else:
            config_dir = p.parent
            files = [str(p)]
        source = vae_path
    elif dit_path is not None:
        files = _model_files(dit_path)
        source = dit_path
    else:
        raise ValueError("YuE2 VAE: pass --vae (m-a-p/YuE2-Vae directory or file) or a ComfyUI --dit that embeds vae.*")

    plan = []  # (file, source key, native key)
    for file in files:
        with MemoryEfficientSafeOpen(file) as f:
            keys = f.keys()
            prefixed = any(k.startswith("vae.") for k in keys)
            for key in keys:
                if prefixed and not key.startswith("vae."):
                    continue
                native = key[4:] if prefixed else key
                if not native.startswith(("encoder.", "decoder.")):
                    continue
                if decoder_only and not native.startswith("decoder."):
                    continue
                plan.append((file, key, native, f.header[key]["dtype"]))
    if not plan:
        raise ValueError(f"no YuE2 VAE tensors found in {source}")

    dtypes = sorted({dt for _, _, _, dt in plan})
    non_f32 = [dt for dt in dtypes if dt != "F32"]
    source_dtype = _ST_DTYPES.get(non_f32[0], non_f32[0]) if non_f32 else "float32"
    if non_f32:
        if not allow_fp16_source:
            raise ValueError(
                f"YuE2 VAE source {source} is {source_dtype}, not float32; encoded latents would shift."
                " Use the fp32 m-a-p/YuE2-Vae (--vae) or pass --allow_fp16_vae"
            )
        logger.warning(f"YuE2 VAE source {source} is {source_dtype}; it is upcast to float32")

    digest = hashlib.sha256()
    sd = {}
    for file in sorted({p[0] for p in plan}):
        with MemoryEfficientSafeOpen(file) as f:
            for _, key, native, _ in sorted(x for x in plan if x[0] == file):
                tensor = f.get_tensor(key)
                digest.update(f"{native}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
                digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                sd[native] = tensor.float()
    digest.update(f"source_dtype:{source_dtype}".encode())

    encoder_config, decoder_config = vae_configs if vae_configs is not None else _vae_configs(config_dir)
    vae = YuE2VAE(encoder_config, decoder_config, decoder_only=decoder_only)
    vae.load_state_dict_any(sd)
    vae.source_dtype = source_dtype
    vae.fingerprint = f"sha256:{digest.hexdigest()}"
    return vae.to(device).eval().requires_grad_(False)


def load_yue2_tokenizer(tokenizer: Optional[str], dit_path: Optional[str]):
    """``YuE2TextTokenizer`` from an explicit path (``.json`` -> tokenizers; ``qwen.tiktoken`` or a directory holding it
    -> tiktoken), else the ``text_encoders.yue2_tokenizer_json`` tensor of a ComfyUI ``dit_path``, else ``qwen.tiktoken``
    next to a Hugging Face ``model.safetensors``; a clear error otherwise."""
    from musubi_tuner.yue2.yue2_tokenizer import YuE2TextTokenizer

    if tokenizer is not None:
        p = Path(tokenizer)
        if p.is_dir():
            if (p / "qwen.tiktoken").is_file():
                return YuE2TextTokenizer.from_tiktoken(str(p / "qwen.tiktoken"))
            if (p / "tokenizer.json").is_file():
                return YuE2TextTokenizer.from_json_bytes((p / "tokenizer.json").read_bytes())
            raise ValueError(f"{tokenizer} holds neither qwen.tiktoken nor tokenizer.json")
        if not p.is_file():
            raise FileNotFoundError(f"YuE2 tokenizer not found: {tokenizer}")
        if p.suffix == ".json":
            return YuE2TextTokenizer.from_json_bytes(p.read_bytes())
        if p.suffix == ".safetensors":
            return YuE2TextTokenizer.from_json_bytes(read_tokenizer_json(str(p)))
        return YuE2TextTokenizer.from_tiktoken(str(p))
    if dit_path is not None:
        p = Path(dit_path)
        if p.is_dir():
            candidate = p / "qwen.tiktoken"
        else:
            if _find_tensor_file(dit_path, COMFY_TOKENIZER_KEY) is not None:
                return YuE2TextTokenizer.from_json_bytes(read_tokenizer_json(dit_path))
            candidate = p.parent / "qwen.tiktoken"
        if candidate.is_file():
            return YuE2TextTokenizer.from_tiktoken(str(candidate))
    raise ValueError(
        "YuE2 tokenizer not found: pass --tokenizer (tokenizer .json or qwen.tiktoken), or use a ComfyUI --dit that"
        " embeds it, or a Hugging Face model.safetensors with qwen.tiktoken next to it"
    )


# endregion
