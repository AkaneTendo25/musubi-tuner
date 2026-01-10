import torch

_LOGGED_MISMATCH = False


def _is_norm_module(module: torch.nn.Module) -> bool:
    if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
        return True
    name = module.__class__.__name__
    return name.endswith("RMSNorm") or name.endswith("LayerNorm")


def ensure_fp8_modules_on_device(module: torch.nn.Module, target_device: torch.device) -> None:
    global _LOGGED_MISMATCH
    allow_weight_move = isinstance(module, torch.nn.Linear) or module.__class__.__name__.endswith("Linear")
    for submodule in module.modules():
        allow_norm_move = _is_norm_module(submodule)
        weight = getattr(submodule, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device != target_device:
            if not _LOGGED_MISMATCH:
                _LOGGED_MISMATCH = True
                print(
                    f"[LTX-2 fp8] weight on {weight.device}, target {target_device}: {submodule.__class__.__name__}"
                )
            if allow_weight_move or allow_norm_move:
                submodule.to(target_device)
                weight = submodule.weight
        scale_weight = getattr(submodule, "scale_weight", None)
        if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
            if scale_weight.device != weight.device:
                submodule.scale_weight = scale_weight.to(device=weight.device)
        org_forward = getattr(submodule, "org_forward", None)
        if callable(org_forward):
            orig_module = getattr(org_forward, "__self__", None)
            if isinstance(orig_module, torch.nn.Module):
                allow_norm_move = _is_norm_module(orig_module)
                weight = getattr(orig_module, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.device != target_device and not _LOGGED_MISMATCH:
                    _LOGGED_MISMATCH = True
                    print(
                        f"[LTX-2 fp8] org_forward weight on {weight.device}, target {target_device}: {orig_module.__class__.__name__}"
                    )
                if (allow_weight_move or allow_norm_move) and isinstance(weight, torch.Tensor) and weight.device != target_device:
                    orig_module.weight.data = weight.data.to(device=target_device)
                    weight = orig_module.weight
                bias = getattr(orig_module, "bias", None)
                if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
                    if bias.device != weight.device:
                        bias.data = bias.data.to(device=weight.device)
                scale_weight = getattr(orig_module, "scale_weight", None)
                if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                    if scale_weight.device != weight.device:
                        orig_module.scale_weight = scale_weight.to(device=weight.device)
        # LoRA replacement stores a bound forward on the original Linear.
        forward_self = getattr(getattr(submodule, "forward", None), "__self__", None)
        if forward_self is not None and forward_self is not submodule:
            org_forward = getattr(forward_self, "org_forward", None)
            if callable(org_forward):
                orig_module = getattr(org_forward, "__self__", None)
                if isinstance(orig_module, torch.nn.Module):
                    allow_norm_move = _is_norm_module(orig_module)
                    weight = getattr(orig_module, "weight", None)
                    if (allow_weight_move or allow_norm_move) and isinstance(weight, torch.Tensor) and weight.device != target_device:
                        orig_module.weight.data = weight.data.to(device=target_device)
                        weight = orig_module.weight
                    bias = getattr(orig_module, "bias", None)
                    if isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor):
                        if bias.device != weight.device:
                            bias.data = bias.data.to(device=weight.device)
                    scale_weight = getattr(orig_module, "scale_weight", None)
                    if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor):
                        if scale_weight.device != weight.device:
                            orig_module.scale_weight = scale_weight.to(device=weight.device)


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
