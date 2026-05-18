import torch

_LOGGED_MISMATCH = False
# Global flag to disable auto-moves when block swap is handling all moves
_BLOCK_SWAP_ACTIVE = False
# Track which block index is currently being processed by swap logic
# Only disable auto-moves for THIS block (others need auto-load for gradient checkpointing)
_CURRENT_SWAP_BLOCK_IDX = -1


def set_block_swap_active(active: bool) -> None:
    """Set global flag to disable ensure_fp8_modules_on_device auto-moves."""
    global _BLOCK_SWAP_ACTIVE
    _BLOCK_SWAP_ACTIVE = active


def set_current_swap_block(block_idx: int) -> None:
    """Track which block is currently being processed by swap logic."""
    global _CURRENT_SWAP_BLOCK_IDX
    _CURRENT_SWAP_BLOCK_IDX = block_idx


def get_current_swap_block() -> int:
    """Get the block index currently being processed by swap logic."""
    return _CURRENT_SWAP_BLOCK_IDX


def _is_norm_module(module: torch.nn.Module) -> bool:
    if isinstance(module, (torch.nn.RMSNorm, torch.nn.LayerNorm)):
        return True
    name = module.__class__.__name__
    return name.endswith("RMSNorm") or name.endswith("LayerNorm")


def ensure_fp8_modules_on_device(module: torch.nn.Module, target_device: torch.device, only_lora: bool = False, skip_trainable: bool = True) -> None:
    """Move FP8 module components to target device.

    Args:
        module: Module to process
        target_device: Target device
        only_lora: If True, only move LoRA modules
        skip_trainable: If True AND target is CPU, skip parameters with requires_grad=True

    NOTE: When block swap is active (_BLOCK_SWAP_ACTIVE=True), we do NOT globally skip auto-moves.
    Instead, we use per-submodule logic that checks both swap_weight_offload AND whether
    the weight is already on the target device. This allows gradient checkpointing
    recomputation to auto-load blocks from CPU when needed.
    """
    global _LOGGED_MISMATCH

    # Only skip trainable parameters when moving TO CPU (offloading), not when loading TO GPU
    should_skip_trainable = skip_trainable and target_device.type == "cpu"
    
    for submodule in module.modules():
        if only_lora:
            allow_weight_move = False
            allow_norm_move = False
        else:
            allow_weight_move = isinstance(submodule, torch.nn.Linear) or submodule.__class__.__name__.endswith("Linear")
            allow_norm_move = _is_norm_module(submodule)

        if not only_lora:
            weight = getattr(submodule, "weight", None)

            # Compute avoid_weight_move per-submodule based on actual weight location
            # Key insight: only avoid move if swap is managing AND weight is already on target device
            # If weight is on CPU and target is CUDA, we MUST allow the move (for gradient checkpointing recomputation)
            submodule_swap_attr = getattr(submodule, "swap_weight_offload", False)
            weight_already_on_target = weight is not None and isinstance(weight, torch.Tensor) and weight.device == target_device
            avoid_weight_move = bool(submodule_swap_attr) and target_device.type == "cuda" and weight_already_on_target

            if hasattr(submodule, "weight") and weight is not None and isinstance(weight, torch.Tensor) and weight.device != target_device:
                # Skip trainable parameters only when offloading to CPU
                if should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad:
                    pass  # Skip
                elif not _LOGGED_MISMATCH:
                    _LOGGED_MISMATCH = True
                    print(
                        f"[LTX-2 fp8] weight on {weight.device}, target {target_device}: {submodule.__class__.__name__}"
                    )
                if (allow_weight_move or allow_norm_move) and not avoid_weight_move:
                    if not (should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad):
                        submodule.to(target_device)
                        weight = submodule.weight
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
                    # Compute avoid_weight_move for orig_module
                    orig_swap_attr = getattr(orig_module, "swap_weight_offload", False)
                    orig_weight_on_target = weight is not None and isinstance(weight, torch.Tensor) and weight.device == target_device
                    orig_avoid_weight_move = bool(orig_swap_attr) and target_device.type == "cuda" and orig_weight_on_target

                    if isinstance(weight, torch.Tensor) and weight.device != target_device and not _LOGGED_MISMATCH:
                        if not (should_skip_trainable and hasattr(weight, 'requires_grad') and weight.requires_grad):
                            _LOGGED_MISMATCH = True
                            print(
                                f"[LTX-2 fp8] org_forward weight on {weight.device}, target {target_device}: {orig_module.__class__.__name__}"
                            )
                    if (allow_weight_move or allow_norm_move) and isinstance(weight, torch.Tensor) and weight.device != target_device and not orig_avoid_weight_move:
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
                        # Compute avoid_weight_move for orig_module in LoRA context
                        lora_orig_swap_attr = getattr(orig_module, "swap_weight_offload", False)
                        lora_orig_weight_on_target = weight is not None and isinstance(weight, torch.Tensor) and weight.device == target_device
                        lora_orig_avoid_weight_move = bool(lora_orig_swap_attr) and target_device.type == "cuda" and lora_orig_weight_on_target

                        if (allow_weight_move or allow_norm_move) and isinstance(weight, torch.Tensor) and weight.device != target_device and not lora_orig_avoid_weight_move:
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

                # Native LoKr/DoKr modules store adapter tensors directly as parameters.
                if forward_self.__class__.__module__.startswith("musubi_tuner.networks.lokr"):
                    for param in forward_self.parameters(recurse=True):
                        if isinstance(param, torch.Tensor) and param.device != target_device:
                            param.data = param.data.to(device=target_device)
                    for buf in forward_self.buffers(recurse=True):
                        if isinstance(buf, torch.Tensor) and buf.device != target_device:
                            buf.data = buf.data.to(device=target_device)



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
