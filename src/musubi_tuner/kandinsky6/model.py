from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from .configs import MODEL_CONFIGS
from .dit import DiffusionTransformer3D
from .piflow_dit import PiFlowDiffusionTransformer3D


def _stream_assign_safetensors(
    model: torch.nn.Module,
    path: str | Path,
    *,
    dtype: torch.dtype | None,
    device: str | torch.device,
    strict: bool,
) -> tuple[list[str], list[str]]:
    """Assign one safetensors tensor at a time into a meta-instantiated module."""
    device = torch.device(device)
    expected = model.state_dict()
    expected_keys = set(expected)
    with safe_open(path, framework="pt", device="cpu") as handle:
        file_keys = set(handle.keys())
        missing = sorted(expected_keys - file_keys)
        unexpected = sorted(file_keys - expected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(f"State dict mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        for key in sorted(expected_keys & file_keys):
            tensor = handle.get_tensor(key)
            expected_tensor = expected[key]
            if tensor.shape != expected_tensor.shape:
                raise RuntimeError(
                    f"State dict shape mismatch for {key}: checkpoint={tuple(tensor.shape)}, model={tuple(expected_tensor.shape)}"
                )
            target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
            tensor = tensor.to(device=device, dtype=target_dtype)
            parent_name, _, leaf_name = key.rpartition(".")
            module = model.get_submodule(parent_name) if parent_name else model
            if leaf_name in module._parameters:
                old_parameter = module._parameters[leaf_name]
                requires_grad = old_parameter.requires_grad if old_parameter is not None else True
                module._parameters[leaf_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
            elif leaf_name in module._buffers:
                module._buffers[leaf_name] = tensor
            else:
                raise RuntimeError(f"State dict key {key} does not resolve to a parameter or buffer")
    return missing, unexpected


def _resolve_weight_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidates = (path / "transformer" / "diffusion_pytorch_model.safetensors", path / "diffusion_pytorch_model.safetensors")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No Kandinsky 6 transformer safetensors found under {path}")
    return path


def inspect_checkpoint(path: str | Path) -> tuple[str, int | None]:
    """Return (lite/pro, PiFlow grid count or None) without loading weights."""
    path = _resolve_weight_file(path)
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        sentinel = "visual_embeddings.in_layer.weight"
        if sentinel not in keys:
            raise ValueError(f"Not a Kandinsky 6 checkpoint: missing {sentinel}")
        hidden_size = f.get_slice(sentinel).get_shape()[0]
        if hidden_size == 1792:
            variant = "lite"
        elif hidden_size == 4096:
            variant = "pro"
        else:
            raise ValueError(f"Unknown Kandinsky 6 hidden size {hidden_size}; expected Lite 1792 or Pro 4096")
        out_key = "out_layer.out_layer.weight"
        output_width = f.get_slice(out_key).get_shape()[0]
    base_width = 16 * 4
    if output_width % base_width:
        raise ValueError(f"Unexpected Kandinsky 6 output head width {output_width}")
    n_grid = output_width // base_width
    return variant, (n_grid if n_grid > 1 else None)


def load_dit(
    checkpoint_path: str | Path,
    model_variant: str = "auto",
    dtype: torch.dtype | None = torch.bfloat16,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> DiffusionTransformer3D:
    """Load a published Lite/Pro or distilled PiFlow Kandinsky 6 transformer."""
    path = _resolve_weight_file(checkpoint_path)
    detected_variant, n_grid = inspect_checkpoint(path)
    if model_variant == "auto":
        model_variant = detected_variant
    if model_variant not in MODEL_CONFIGS:
        raise ValueError(f"model_variant must be 'auto', 'lite', or 'pro', got {model_variant!r}")
    if model_variant != detected_variant:
        raise ValueError(f"Checkpoint is {detected_variant}, requested {model_variant}")
    config = dict(MODEL_CONFIGS[model_variant])
    with torch.device("meta"):
        if n_grid is None:
            model = DiffusionTransformer3D(**config)
        else:
            model = PiFlowDiffusionTransformer3D(
                n_grid=n_grid, out_visual_dim=config.pop("out_visual_dim"), out_audio_dim=config.pop("out_audio_dim"), **config
            )
    _stream_assign_safetensors(model, path, dtype=dtype, device=device, strict=strict)
    for module in model.modules():
        touched = False
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                touched = True
        if touched:
            module.reset_parameters()
    model.eval()
    model.model_variant = model_variant
    model.piflow_n_grid = n_grid
    return model


def load_dit_convrot_int8(
    checkpoint_path: str | Path,
    model_variant: str = "auto",
    device: str | torch.device = "cpu",
    quant_device: str | torch.device = "cuda",
    bwd_mode: str = "bf16",
    disable_numpy_memmap: bool = False,
) -> DiffusionTransformer3D:
    """Stream and quantize the frozen DiT base for memory-efficient LoRA training."""
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    path = _resolve_weight_file(checkpoint_path)
    detected_variant, n_grid = inspect_checkpoint(path)
    if model_variant == "auto":
        model_variant = detected_variant
    if model_variant != detected_variant or model_variant not in MODEL_CONFIGS:
        raise ValueError(f"Checkpoint is {detected_variant}, requested {model_variant}")
    config = dict(MODEL_CONFIGS[model_variant])
    with torch.device("meta"):
        if n_grid is None:
            model = DiffusionTransformer3D(**config)
        else:
            model = PiFlowDiffusionTransformer3D(
                n_grid=n_grid, out_visual_dim=config.pop("out_visual_dim"), out_audio_dim=config.pop("out_audio_dim"), **config
            )
    quantizer = ConvRotInt8Quantizer(
        target_layer_keys=[
            "visual_transformer_blocks.",
            "video_text_transformer_blocks.",
            "audio_text_transformer_blocks.",
        ],
        exclude_layer_keys=["modulation", "norm"],
        allowed_groupsizes=(256, 64),
        allow_unrotated=True,
    )
    state = load_safetensors_with_lora_and_fp8(
        model_files=[str(path)],
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=False,
        calc_device=torch.device(quant_device),
        move_to_device=torch.device(device) == torch.device(quant_device),
        dit_weight_dtype=None,
        disable_numpy_memmap=disable_numpy_memmap,
        quantizer=quantizer,
    )
    apply_convrot_int8_monkey_patch(model, state, bwd_mode=bwd_mode, groupsize_map=quantizer.module_groupsizes)
    model.requires_grad_(False)
    for key, value in state.items():
        if value.dtype in (torch.float16, torch.float32):
            state[key] = value.to(torch.bfloat16) if not key.endswith(".scale_weight") else value.float()
        if torch.device(device).type != "cpu":
            state[key] = state[key].to(device)
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        touched = False
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                touched = True
        if touched:
            module.reset_parameters()
    model.eval()
    model.model_variant = model_variant
    model.piflow_n_grid = n_grid
    return model


def load_dit_fp8(
    checkpoint_path: str | Path,
    model_variant: str = "auto",
    device: str | torch.device = "cpu",
    quant_device: str | torch.device = "cuda",
    disable_numpy_memmap: bool = False,
) -> DiffusionTransformer3D:
    """Stream and scale-quantize transformer-block Linears to FP8."""
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    path = _resolve_weight_file(checkpoint_path)
    detected_variant, n_grid = inspect_checkpoint(path)
    if model_variant == "auto":
        model_variant = detected_variant
    if model_variant != detected_variant or model_variant not in MODEL_CONFIGS:
        raise ValueError(f"Checkpoint is {detected_variant}, requested {model_variant}")
    config = dict(MODEL_CONFIGS[model_variant])
    with torch.device("meta"):
        if n_grid is None:
            model = DiffusionTransformer3D(**config)
        else:
            model = PiFlowDiffusionTransformer3D(
                n_grid=n_grid, out_visual_dim=config.pop("out_visual_dim"), out_audio_dim=config.pop("out_audio_dim"), **config
            )
    state = load_safetensors_with_lora_and_fp8(
        model_files=[str(path)],
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=True,
        calc_device=torch.device(quant_device),
        move_to_device=torch.device(device) == torch.device(quant_device),
        dit_weight_dtype=None,
        target_keys=["visual_transformer_blocks.", "video_text_transformer_blocks.", "audio_text_transformer_blocks."],
        exclude_keys=["modulation", "norm"],
        disable_numpy_memmap=disable_numpy_memmap,
    )
    apply_fp8_monkey_patch(model, state, use_scaled_mm=False)
    model.requires_grad_(False)
    if torch.device(device).type != "cpu":
        state = {key: value.to(device) for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        touched = False
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                touched = True
        if touched:
            module.reset_parameters()
    model.eval()
    model.model_variant = model_variant
    model.piflow_n_grid = n_grid
    return model
