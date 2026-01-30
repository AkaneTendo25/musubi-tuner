"""
Block-Level Gradient Checkpointing with CPU Offloading

This module provides custom gradient checkpointing that gives full control over
weight loading/offloading during both forward and backward passes. Unlike
torch.utils.checkpoint.checkpoint, this implementation ensures only one block's
weights are on GPU at any time during backward.

Key Features:
- Block-by-block processing during backward (not all at once)
- Automatic weight loading before each block's backward
- Automatic weight offloading after each block's backward  

Usage:
    from musubi_tuner.modules.block_level_checkpointing import block_checkpoint
    
    # In your model's forward method:
    output = block_checkpoint(
        forward_fn=lambda: self._forward(video, audio, perturbations),
        block=self,
        load_fn=load_weights_to_gpu,
        offload_fn=offload_weights_to_cpu,
        inputs=[video.x, audio.x] if audio else [video.x],
    )
"""

import torch
from typing import Callable, List, Any, Optional, Tuple
import logging
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device

logger = logging.getLogger(__name__)


def _get_device_from_args(args) -> torch.device:
    """Detect target device from input arguments."""
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.device
        elif isinstance(arg, (list, tuple)):
            for x in arg:
                if isinstance(x, torch.Tensor):
                    return x.device
    # Fallback to CUDA if available, else CPU
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _to_device_recursive(arg, device, non_blocking=True):
    """Recursively move tensors to device."""
    if isinstance(arg, torch.Tensor):
        return arg.to(device=device, non_blocking=non_blocking)
    elif isinstance(arg, (list, tuple)):
        return type(arg)(_to_device_recursive(x, device, non_blocking) for x in arg)
    elif isinstance(arg, dict):
        return {k: _to_device_recursive(v, device, non_blocking) for k, v in arg.items()}
    return arg


def _detach_cpu_recursive(arg):
    """Recursively detach tensors and move to CPU."""
    if isinstance(arg, torch.Tensor):
        return arg.detach().cpu()
    elif isinstance(arg, (list, tuple)):
        return type(arg)(_detach_cpu_recursive(x) for x in arg)
    elif isinstance(arg, dict):
        return {k: _detach_cpu_recursive(v) for k, v in arg.items()}
    return arg


class BlockCheckpointFunction(torch.autograd.Function):
    """
    Custom autograd function for block-level gradient checkpointing with CPU offloading.
    """
    
    @staticmethod
    def forward(ctx, run_forward, block, load_fn, offload_fn, preserve_rng_state, *args):
        """Forward pass: run the function and save context for backward."""
        ctx.run_forward = run_forward
        ctx.block = block
        ctx.load_fn = load_fn
        ctx.offload_fn = offload_fn
        ctx.preserve_rng_state = preserve_rng_state
        
        # Save RNG state if needed
        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.cuda_rng_state = torch.cuda.get_rng_state()
        
        # Save Autocast state
        ctx.autocast_enabled = torch.is_autocast_enabled('cuda') if torch.cuda.is_available() else False
        ctx.autocast_dtype = torch.get_autocast_dtype('cuda') if torch.cuda.is_available() else torch.float16

        # Detect target device - we need CUDA for computation even if inputs arrive on CPU
        # (which happens with activation_cpu_offloading)
        input_device = _get_device_from_args(args)
        if input_device.type == 'cpu' and torch.cuda.is_available():
            # Inputs are on CPU (from offloading), but computation must happen on GPU
            target_device = torch.device('cuda')
        else:
            target_device = input_device
        ctx.target_device = target_device

        # Save inputs for backward (move to CPU to save GPU memory)
        # Also save original devices to ensure backward returns grads on correct device.
        saved_args = []
        input_devices = []
        for arg in args:
            saved_args.append(_detach_cpu_recursive(arg))
            if isinstance(arg, torch.Tensor):
                input_devices.append(arg.device)
            else:
                input_devices.append(None)
                
        ctx.saved_args = saved_args
        ctx.input_devices = input_devices
        
        # Move inputs to target device (GPU) for computation
        gpu_args = [_to_device_recursive(arg, target_device) for arg in args]
        
        # === CRITICAL: Load weights for this block before forward ===
        if load_fn is not None:
            load_fn(block, target_device)
        # Ensure FP8 weights/scale are on compute device for recompute correctness.
        ensure_fp8_modules_on_device(block, target_device, skip_trainable=False)
        
        # Run forward with no_grad using GPU args
        with torch.no_grad():
            outputs = run_forward(*gpu_args)
        
        # === Offload weights after forward to save VRAM ===
        if offload_fn is not None:
            offload_fn(block, torch.device('cpu'))
        
        # Extract output tensors
        # Gradient checkpointing REQUIRES tensor outputs to track gradients.
        output_tensors = []
        if isinstance(outputs, tuple):
            ctx.num_outputs = len(outputs)
            for out in outputs:
                if out is None:
                    pass  # Skip None outputs
                elif hasattr(out, 'x') and isinstance(out.x, torch.Tensor):
                    output_tensors.append(out.x.detach().requires_grad_(out.x.requires_grad))
                elif isinstance(out, torch.Tensor):
                    output_tensors.append(out.detach().requires_grad_(out.requires_grad))
        else:
            ctx.num_outputs = 1
            if isinstance(outputs, torch.Tensor):
                output_tensors.append(outputs.detach().requires_grad_(outputs.requires_grad))
            elif hasattr(outputs, 'x') and isinstance(outputs.x, torch.Tensor):
                output_tensors.append(outputs.x.detach().requires_grad_(outputs.x.requires_grad))
        
        # FAIL FAST: If no tensor outputs, checkpointing cannot work correctly
        if not output_tensors:
            raise ValueError(
                f"block_checkpoint requires at least one tensor output for gradient tracking. "
                f"Got output type: {type(outputs)}. Ensure the forward function returns tensors "
                f"or objects with a .x tensor attribute."
            )
        
        return tuple(output_tensors)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass: load weights, recompute forward, compute grads, offload weights."""
        block = ctx.block
        run_forward = ctx.run_forward
        load_fn = ctx.load_fn
        offload_fn = ctx.offload_fn
        input_devices = ctx.input_devices
        
        # Determine target device - prefer saved device, fallback to grad_outputs
        target_device = ctx.target_device
        if target_device is None or target_device.type == 'cpu':
            for g in grad_outputs:
                if g is not None and isinstance(g, torch.Tensor) and g.device.type != 'cpu':
                    target_device = g.device
                    break
        
        # === CRITICAL: Load weights for this block ===
        if load_fn is not None:
            load_fn(block, target_device)
        # Ensure FP8 weights/scale are on compute device before recompute.
        ensure_fp8_modules_on_device(block, target_device, skip_trainable=False)
        
        # Restore inputs from CPU to Target Device and enable grad tracking
        inputs = []
        detached_inputs = []
        
        for arg in ctx.saved_args:
            if isinstance(arg, torch.Tensor):
                arg_gpu = arg.to(target_device)
                detached = arg_gpu.detach().requires_grad_(True)
                inputs.append(detached)
                detached_inputs.append(detached)
            else:
                # Handle recursive restore for tuples/lists
                arg_gpu = _to_device_recursive(arg, target_device)
                inputs.append(arg_gpu)
                detached_inputs.append(None)  # Non-tensors don't get grads
        
        # Restore RNG state
        if ctx.preserve_rng_state:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.set_rng_state(ctx.cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
        
        # Recompute forward with gradients
        with torch.enable_grad():
            if ctx.autocast_enabled:
                with torch.autocast('cuda', dtype=ctx.autocast_dtype):
                    outputs = run_forward(*inputs)
            else:
                outputs = run_forward(*inputs)
        
        # Restore RNG state
        if ctx.preserve_rng_state:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
        
        # Extract output tensors for grad computation
        output_tensors = []
        if isinstance(outputs, tuple):
            for out in outputs:
                if out is not None and hasattr(out, 'x') and isinstance(out.x, torch.Tensor):
                    output_tensors.append(out.x)
                elif isinstance(out, torch.Tensor):
                    output_tensors.append(out)
        elif isinstance(outputs, torch.Tensor):
            output_tensors.append(outputs)
        elif hasattr(outputs, 'x') and isinstance(outputs.x, torch.Tensor):
            output_tensors.append(outputs.x)
        
        # Compute gradients
        if output_tensors:
            valid_grads = []
            valid_outputs = []
            for i, out in enumerate(output_tensors):
                if out.requires_grad and i < len(grad_outputs) and grad_outputs[i] is not None:
                    valid_grads.append(grad_outputs[i])
                    valid_outputs.append(out)
            
            if valid_outputs:
                # Use torch.autograd.backward to populate .grad for Parameters (LoRA)
                torch.autograd.backward(
                    tensors=valid_outputs,
                    grad_tensors=valid_grads,
                    retain_graph=False,
                )
                
                # Collect gradients from input tensors
                full_grads = []
                for i, inp in enumerate(detached_inputs):
                    if inp is not None:
                        g = inp.grad
                        orig_device = input_devices[i]
                        if g is not None and orig_device is not None and g.device != orig_device:
                            g = g.to(orig_device)
                        full_grads.append(g)
                    else:
                        full_grads.append(None)
                input_grads = tuple(full_grads)
            else:
                input_grads = (None,) * len(detached_inputs)
        else:
            input_grads = (None,) * len(detached_inputs)
        
        # === CRITICAL: Offload weights after this block's backward ===
        if offload_fn is not None:
            offload_fn(block, torch.device('cpu'))
        
        # Return grads: (None for run_forward, block, load_fn, offload_fn, preserve_rng_state) + input_grads
        return (None, None, None, None, None) + input_grads


def block_checkpoint(
    function: Callable[..., Any],
    *args,
    block: torch.nn.Module,
    load_fn: Callable[[torch.nn.Module, torch.device], None],
    offload_fn: Callable[[torch.nn.Module, torch.device], None],
    preserve_rng_state: bool = True,
) -> Any:
    """
    Apply block-level gradient checkpointing with CPU offloading.
    
    Args:
        function: Function to run forward pass.
        *args: Argument list to pass to function.
        block: The neural network module (keyword arg).
        load_fn: Weight loading function (keyword arg).
        offload_fn: Weight offloading function (keyword arg).
        preserve_rng_state: (default True)
    
    Returns:
        Tuple of output tensors from the forward function.
    
    Raises:
        ValueError: If forward function returns no tensor outputs.
    """
    # Detect target device from args
    target_device = _get_device_from_args(args)
    
    # Check if we need checkpointing
    args_require_grad = any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args)
    
    if not args_require_grad and not torch.is_grad_enabled():
        # Truly inference / no-grad mode - skip checkpointing
        if load_fn is not None:
            load_fn(block, target_device)
            gpu_args = [_to_device_recursive(arg, target_device) for arg in args]
             
            with torch.no_grad():
                outputs = function(*gpu_args)
                 
            if offload_fn is not None:
                offload_fn(block, torch.device('cpu'))
            return outputs
        
        return function(*args)

    # If we are here, we need checkpointing/graph tracking.
    # If inputs don't require grad (e.g. frozen encoder), autograd.Function won't track.
    # We must force at least one input to require grad.
    if not args_require_grad:
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                new_args[i] = arg.detach().requires_grad_(True)
                break
        args = tuple(new_args)

    outputs = BlockCheckpointFunction.apply(
        function,
        block,
        load_fn,
        offload_fn,
        preserve_rng_state,
        *args
    )
    
    return outputs
