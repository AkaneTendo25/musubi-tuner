import contextlib
import os
import torch

_LOGGED_MISMATCH = False
_FP8_SYNC_ENABLED = True


def set_fp8_sync_enabled(enabled: bool) -> None:
    global _FP8_SYNC_ENABLED
    _FP8_SYNC_ENABLED = bool(enabled)


@contextlib.contextmanager
def fp8_sync_disabled():
    global _FP8_SYNC_ENABLED
    prev = _FP8_SYNC_ENABLED
    _FP8_SYNC_ENABLED = False
    try:
        yield
    finally:
        _FP8_SYNC_ENABLED = prev


def _is_norm_module(module: torch.nn.Module) -> bool:
    if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
        return True
    name = module.__class__.__name__
    return name.endswith("RMSNorm") or name.endswith("LayerNorm")


def ensure_fp8_modules_on_device(module: torch.nn.Module, target_device: torch.device, only_lora: bool = False, skip_trainable: bool = True) -> None:
    global _LOGGED_MISMATCH
    if not _FP8_SYNC_ENABLED:
        return
    # Only skip trainable parameters when moving TO CPU (offloading), not when loading TO GPU
    should_skip_trainable = skip_trainable and target_device.type == "cpu"
    
    allow_weight_move = isinstance(module, torch.nn.Linear) or module.__class__.__name__.endswith("Linear")
    for submodule in module.modules():
        if only_lora:
            allow_weight_move = False
            allow_norm_move = False
        else:
            allow_norm_move = _is_norm_module(submodule)

        if not only_lora:
            weight = getattr(submodule, "weight", None)
            if hasattr(submodule, "weight") and weight is not None and isinstance(weight, torch.Tensor):
                if (
                    weight.device == target_device
                    and target_device.type == "cuda"
                    and os.getenv("LTX2_FP8_FORCE_DTYPE", "1") == "1"
                    and hasattr(submodule, "scale_weight")
                    and isinstance(getattr(submodule, "scale_weight", None), torch.Tensor)
                    and weight.dtype not in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
                ):
                    try:
                        weight.data = weight.data.to(dtype=torch.float8_e4m3fn)
                    except Exception:
                        pass
                if weight.device != target_device:
                    # Skip trainable parameters only when offloading to CPU
                    if should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad:
                        pass  # Skip
                    elif not _LOGGED_MISMATCH:
                        _LOGGED_MISMATCH = True
                        print(
                            f"[LTX-2 fp8] weight on {weight.device}, target {target_device}: {submodule.__class__.__name__}"
                        )
                    if allow_weight_move or allow_norm_move:
                        if not (should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad):
                            submodule.to(target_device)
                            weight = submodule.weight
                            if (
                                target_device.type == "cuda"
                                and os.getenv("LTX2_FP8_RESTORE_ON_DEVICE", "1") == "1"
                                and hasattr(submodule, "scale_weight")
                                and isinstance(getattr(submodule, "scale_weight", None), torch.Tensor)
                                and weight.dtype not in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
                            ):
                                try:
                                    weight.data = weight.data.to(dtype=torch.float8_e4m3fn)
                                except Exception:
                                    pass
            scale_weight = getattr(submodule, "scale_weight", None)
            if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                if scale_weight.device != weight.device:
                    if not (should_skip_trainable and hasattr(scale_weight, 'requires_grad') and scale_weight.requires_grad):
                        submodule.scale_weight = scale_weight.to(device=weight.device)
            org_forward = getattr(submodule, "org_forward", None)
            if callable(org_forward):
                orig_module = getattr(org_forward, "__self__", None)
                if isinstance(orig_module, torch.nn.Module):
                    allow_norm_move = _is_norm_module(orig_module)
                    weight = getattr(orig_module, "weight", None)
                    if isinstance(weight, torch.Tensor) and weight.device != target_device and not _LOGGED_MISMATCH:
                        if not (should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad):
                            _LOGGED_MISMATCH = True
                            print(
                                f"[LTX-2 fp8] org_forward weight on {weight.device}, target {target_device}: {orig_module.__class__.__name__}"
                            )
                    if (allow_weight_move or allow_norm_move) and isinstance(weight, torch.Tensor) and weight.device != target_device:
                        if not (should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad):
                            orig_module.weight.data = weight.data.to(device=target_device)
                            weight = orig_module.weight
                    bias = getattr(orig_module, "bias", None)
                    if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
                        if bias.device != weight.device:
                            if not (should_skip_trainable and hasattr(bias, 'requires_grad') and bias.requires_grad):
                                bias.data = bias.data.to(device=weight.device)
                    scale_weight = getattr(orig_module, "scale_weight", None)
                    if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                        if scale_weight.device != weight.device:
                            if not (should_skip_trainable and hasattr(scale_weight, 'requires_grad') and scale_weight.requires_grad):
                                orig_module.scale_weight = scale_weight.to(device=weight.device)
        
        # LoRA replacement stores a bound forward on the original Linear.
        forward_self = getattr(getattr(submodule, "forward", None), "__self__", None)
        if forward_self is not None and forward_self is not submodule:
            if not only_lora:
                org_forward = getattr(forward_self, "org_forward", None)
                if callable(org_forward):
                    orig_module = getattr(org_forward, "__self__", None)
                    if isinstance(orig_module, torch.nn.Module):
                        allow_norm_move = _is_norm_module(orig_module)
                        weight = getattr(orig_module, "weight", None)
                        if (allow_weight_move or allow_norm_move) and isinstance(weight, torch.Tensor) and weight.device != target_device:
                            if not (should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad):
                                orig_module.weight.data = weight.data.to(device=target_device)
                                weight = orig_module.weight
                        bias = getattr(orig_module, "bias", None)
                        if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
                            if bias.device != weight.device:
                                if not (should_skip_trainable and hasattr(bias, 'requires_grad') and bias.requires_grad):
                                    bias.data = bias.data.to(device=weight.device)
                        scale_weight = getattr(orig_module, "scale_weight", None)
                        if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                            if scale_weight.device != weight.device:
                                if not (should_skip_trainable and hasattr(scale_weight, 'requires_grad') and scale_weight.requires_grad):
                                    orig_module.scale_weight = scale_weight.to(device=weight.device)
            # Move LoRA module weights (lora_down, lora_up) to target device
            # Skip if should_skip_trainable is True (LoRA weights are trainable, keep on GPU)
            if not should_skip_trainable:
                lora_down = getattr(forward_self, "lora_down", None)
                lora_up = getattr(forward_self, "lora_up", None)
                
                # Handle both single Linear and ModuleList (split_dims case)
                if isinstance(lora_down, torch.nn.ModuleList):
                    for ld in lora_down:
                        if hasattr(ld, 'weight') and ld.weight.device != target_device:
                            ld.to(target_device)
                elif isinstance(lora_down, torch.nn.Module):
                    lora_down_weight = getattr(lora_down, "weight", None)
                    if isinstance(lora_down_weight, torch.Tensor) and lora_down_weight.device != target_device:
                        lora_down.to(target_device)
                        
                if isinstance(lora_up, torch.nn.ModuleList):
                    for lu in lora_up:
                        if hasattr(lu, 'weight') and lu.weight.device != target_device:
                            lu.to(target_device)
                elif isinstance(lora_up, torch.nn.Module):
                    lora_up_weight = getattr(lora_up, "weight", None)
                    if isinstance(lora_up_weight, torch.Tensor) and lora_up_weight.device != target_device:
                        lora_up.to(target_device)



def move_fp8_scale_weights(module: torch.nn.Module, target_device: torch.device) -> None:
    non_blocking = target_device.type != "cpu"
    for submodule in module.modules():
        scale_weight = getattr(submodule, "scale_weight", None)
        if isinstance(scale_weight, torch.Tensor) and scale_weight.device != target_device:
            submodule.scale_weight = scale_weight.to(device=target_device, non_blocking=non_blocking)
        org_forward = getattr(submodule, "org_forward", None)
        if callable(org_forward):
            orig_module = getattr(org_forward, "__self__", None)
            if isinstance(orig_module, torch.nn.Module):
                scale_weight = getattr(orig_module, "scale_weight", None)
                if isinstance(scale_weight, torch.Tensor) and scale_weight.device != target_device:
                    orig_module.scale_weight = scale_weight.to(
                        device=target_device, non_blocking=non_blocking
                    )


def force_fp8_on_device(module: torch.nn.Module, device: torch.device) -> None:
    """Cast FP8-capable weights to FP8 if they already live on the target device."""
    if device.type != "cuda":
        return
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return
    for submodule in module.modules():
        weight = getattr(submodule, "weight", None)
        scale_weight = getattr(submodule, "scale_weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and isinstance(scale_weight, torch.Tensor)
            and weight.device == device
            and weight.dtype != fp8_dtype
        ):
            try:
                weight.data = weight.data.to(dtype=fp8_dtype)
            except Exception:
                pass
        if isinstance(scale_weight, torch.Tensor) and scale_weight.device == device and scale_weight.dtype == fp8_dtype:
            try:
                scale_weight.data = scale_weight.data.to(dtype=torch.float32)
            except Exception:
                pass
        org_forward = getattr(submodule, "org_forward", None)
        if callable(org_forward):
            orig_module = getattr(org_forward, "__self__", None)
            if isinstance(orig_module, torch.nn.Module):
                weight = getattr(orig_module, "weight", None)
                scale_weight = getattr(orig_module, "scale_weight", None)
                if (
                    isinstance(weight, torch.Tensor)
                    and isinstance(scale_weight, torch.Tensor)
                    and weight.device == device
                    and weight.dtype != fp8_dtype
                ):
                    try:
                        weight.data = weight.data.to(dtype=fp8_dtype)
                    except Exception:
                        pass
                if isinstance(scale_weight, torch.Tensor) and scale_weight.device == device and scale_weight.dtype == fp8_dtype:
                    try:
                        scale_weight.data = scale_weight.data.to(dtype=torch.float32)
                    except Exception:
                        pass
