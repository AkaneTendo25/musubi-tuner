"""Checkpoint loading helpers for MiniMax Music 3."""

from __future__ import annotations

import logging

import torch
from accelerate import init_empty_weights

from musubi_tuner.minimax_music3.model import MiniMaxMusic3DiT
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import load_safetensors


logger = logging.getLogger(__name__)


def load_comfy_dit(
    checkpoint: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    disable_mmap: bool = False,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "native",
    calc_device: str | torch.device | None = None,
) -> MiniMaxMusic3DiT:
    """Load a Comfy-Org bf16/fp16 DiT and optionally quantize its frozen blocks."""
    with init_empty_weights():
        model = MiniMaxMusic3DiT()

    if convrot_int8:
        linear_weight_keys = {f"{name}.weight" for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        state_dict = load_safetensors_with_lora_and_fp8(
            checkpoint, None, None, False, torch.device(calc_device or device),
            move_to_device=torch.device(device).type != "cpu", dit_weight_dtype=dtype,
            disable_numpy_memmap=disable_mmap,
            quantizer=ConvRotInt8Quantizer(exact_target_keys=linear_weight_keys),
        )
    else:
        state_dict = load_safetensors(checkpoint, device=device, disable_mmap=disable_mmap)

    if convrot_int8:
        apply_convrot_int8_monkey_patch(model, state_dict, bwd_mode=convrot_int8_bwd)
        model.requires_grad_(False)

    if dtype is not None:
        for key, value in state_dict.items():
            if value.is_floating_point() and not key.endswith(".weight_scale"):
                state_dict[key] = value.to(dtype=dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Invalid ComfyUI MiniMax Music 3 checkpoint; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    if not convrot_int8:
        model.to(device)
    model.requires_grad_(False)
    model.eval()
    logger.info("Loaded ComfyUI MiniMax Music 3 DiT (%s)", "ConvRot int8" if convrot_int8 else str(model.dtype))
    return model
