import torch

_LOGGED_MISMATCH = False


def ensure_fp8_modules_on_device(module: torch.nn.Module, target_device: torch.device) -> None:
    global _LOGGED_MISMATCH
    for submodule in module.modules():
        weight = getattr(submodule, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device != target_device:
            submodule.to(target_device)
            weight = getattr(submodule, "weight", None)
        scale_weight = getattr(submodule, "scale_weight", None)
        if isinstance(scale_weight, torch.Tensor) and hasattr(submodule, "weight"):
            weight = getattr(submodule, "weight", None)
            if isinstance(weight, torch.Tensor) and scale_weight.device != weight.device:
                submodule.scale_weight = scale_weight.to(device=weight.device)

        org_forward = getattr(submodule, "org_forward", None)
        if callable(org_forward):
            orig_module = getattr(org_forward, "__self__", None)
            if isinstance(orig_module, torch.nn.Module):
                weight = getattr(orig_module, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.device != target_device and not _LOGGED_MISMATCH:
                    _LOGGED_MISMATCH = True
                    print(
                        f"[LTX-2 fp8] org_forward module on {weight.device}, target {target_device}: {orig_module.__class__.__name__}"
                    )
                if isinstance(weight, torch.Tensor) and weight.device != target_device:
                    orig_module.weight = torch.nn.Parameter(weight.to(device=target_device))
                bias = getattr(orig_module, "bias", None)
                if isinstance(bias, torch.Tensor) and bias.device != target_device:
                    orig_module.bias = torch.nn.Parameter(bias.to(device=target_device))
                orig_module.to(target_device)
                scale_weight = getattr(orig_module, "scale_weight", None)
                if isinstance(scale_weight, torch.Tensor) and hasattr(orig_module, "weight"):
                    weight = getattr(orig_module, "weight", None)
                    if isinstance(weight, torch.Tensor) and scale_weight.device != weight.device:
                        orig_module.scale_weight = scale_weight.to(device=weight.device)
        # LoRA replacement stores a bound forward on the original Linear.
        forward_self = getattr(getattr(submodule, "forward", None), "__self__", None)
        if forward_self is not None and forward_self is not submodule:
            org_forward = getattr(forward_self, "org_forward", None)
            if callable(org_forward):
                orig_module = getattr(org_forward, "__self__", None)
                if isinstance(orig_module, torch.nn.Module):
                    weight = getattr(orig_module, "weight", None)
                    if isinstance(weight, torch.Tensor) and weight.device != target_device:
                        orig_module.weight = torch.nn.Parameter(weight.to(device=target_device))
                    bias = getattr(orig_module, "bias", None)
                    if isinstance(bias, torch.Tensor) and bias.device != target_device:
                        orig_module.bias = torch.nn.Parameter(bias.to(device=target_device))
                    scale_weight = getattr(orig_module, "scale_weight", None)
                    if isinstance(scale_weight, torch.Tensor) and hasattr(orig_module, "weight"):
                        weight = getattr(orig_module, "weight", None)
                        if isinstance(weight, torch.Tensor) and scale_weight.device != weight.device:
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
