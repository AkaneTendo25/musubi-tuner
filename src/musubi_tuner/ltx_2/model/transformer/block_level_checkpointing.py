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

import os
import torch
from typing import Callable, Any
import logging
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device

logger = logging.getLogger(__name__)

# Opt-in: overlap the activation D2H/H2D of block-level checkpointing with block
# compute by using reused pinned CPU buffers + a dedicated CUDA stream, instead of
# the default synchronous .cpu()/.to(). Off by default — set LTX2_CKPT_ACT_PINNED=1
# to enable. Activation tensors are smaller than the swapped weights, so this is a
# modest win that mainly helps the squeezed-VRAM (--gradient_checkpointing_cpu_offload)
# path. Validate with a loss-parity A/B before trusting on a real run.
_ACT_PINNED_ENV = "LTX2_CKPT_ACT_PINNED"
_act_transfer_stream: "torch.cuda.Stream | None" = None
# (id(block), leaf_index, shape, dtype) -> reused pinned CPU buffer. Bounded by
# block_count x bucket_shapes; reused across steps so there is no per-step pinned alloc.
_act_pinned_cache: dict = {}


def _act_pinned_enabled() -> bool:
    return os.getenv(_ACT_PINNED_ENV, "0") == "1" and torch.cuda.is_available()


def _get_act_stream(device: torch.device) -> "torch.cuda.Stream":
    global _act_transfer_stream
    if _act_transfer_stream is None:
        _act_transfer_stream = torch.cuda.Stream(device=device)
    return _act_transfer_stream


def _acquire_act_pinned(key, shape, dtype) -> torch.Tensor:
    buf = _act_pinned_cache.get(key)
    if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
        buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        _act_pinned_cache[key] = buf
    return buf


def _find_tensor_device(arg) -> torch.device | None:
    """Find the first tensor device in a nested argument."""
    if isinstance(arg, torch.Tensor):
        return arg.device
    if isinstance(arg, (list, tuple)):
        for item in arg:
            device = _find_tensor_device(item)
            if device is not None:
                return device
    if isinstance(arg, dict):
        for item in arg.values():
            device = _find_tensor_device(item)
            if device is not None:
                return device
    return None


def _get_device_from_args(args) -> torch.device:
    """Detect target device from input arguments."""
    for arg in args:
        device = _find_tensor_device(arg)
        if device is not None:
            return device
    # Fallback to CUDA if available, else CPU
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_device_from_tensor_leaves(tensor_leaves) -> torch.device:
    """Detect target device from flattened tensor leaves."""
    for tensor in tensor_leaves:
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _to_device_recursive(arg, device, non_blocking=True):
    """Recursively move tensors to device."""
    if isinstance(arg, torch.Tensor):
        return arg.to(device=device, non_blocking=non_blocking)
    elif isinstance(arg, (list, tuple)):
        return type(arg)(_to_device_recursive(x, device, non_blocking) for x in arg)
    elif isinstance(arg, dict):
        return {k: _to_device_recursive(v, device, non_blocking) for k, v in arg.items()}
    return arg


def _pack_arg(arg, tensor_leaves):
    """Pack a nested argument while exposing tensor leaves to autograd."""
    if isinstance(arg, torch.Tensor):
        index = len(tensor_leaves)
        tensor_leaves.append(arg)
        return ("tensor", index)
    if isinstance(arg, tuple):
        return ("tuple", [_pack_arg(item, tensor_leaves) for item in arg])
    if isinstance(arg, list):
        return ("list", [_pack_arg(item, tensor_leaves) for item in arg])
    if isinstance(arg, dict):
        return ("dict", [(key, _pack_arg(value, tensor_leaves)) for key, value in arg.items()])
    return ("const", arg)


def _pack_args(args):
    tensor_leaves = []
    specs = [_pack_arg(arg, tensor_leaves) for arg in args]
    return specs, tensor_leaves


def _rebuild_arg(spec, tensor_leaves):
    kind = spec[0]
    if kind == "tensor":
        return tensor_leaves[spec[1]]
    if kind == "tuple":
        return tuple(_rebuild_arg(item, tensor_leaves) for item in spec[1])
    if kind == "list":
        return [_rebuild_arg(item, tensor_leaves) for item in spec[1]]
    if kind == "dict":
        return {key: _rebuild_arg(value, tensor_leaves) for key, value in spec[1]}
    if kind == "const":
        return spec[1]
    raise ValueError(f"Unknown checkpoint argument spec kind: {kind}")


def _rebuild_args(specs, tensor_leaves):
    return [_rebuild_arg(spec, tensor_leaves) for spec in specs]


def _tensor_can_require_grad(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


class BlockCheckpointFunction(torch.autograd.Function):
    """
    Custom autograd function for block-level gradient checkpointing with CPU offloading.
    """

    @staticmethod
    def forward(ctx, run_forward, block, load_fn, offload_fn, preserve_rng_state, arg_specs, *tensor_leaves):
        """Forward pass: run the function and save context for backward."""
        ctx.run_forward = run_forward
        ctx.block = block
        ctx.load_fn = load_fn
        ctx.offload_fn = offload_fn
        ctx.preserve_rng_state = preserve_rng_state
        ctx.arg_specs = arg_specs

        # Detect target device - we need CUDA for computation even if inputs arrive on CPU
        # (which happens with activation_cpu_offloading)
        input_device = _get_device_from_tensor_leaves(tensor_leaves)
        if input_device.type == "cpu" and torch.cuda.is_available():
            # Inputs are on CPU (from offloading), but computation must happen on GPU
            target_device = torch.device("cuda")
        else:
            target_device = input_device
        ctx.target_device = target_device

        # Save RNG state if needed. CUDA RNG is device-specific, so preserve the
        # same device that will run the recompute.
        ctx.cuda_rng_device = target_device if target_device.type == "cuda" else None
        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            if ctx.cuda_rng_device is not None:
                ctx.cuda_rng_state = torch.cuda.get_rng_state(ctx.cuda_rng_device)

        # Save Autocast state
        ctx.autocast_enabled = torch.is_autocast_enabled("cuda") if torch.cuda.is_available() else False
        ctx.autocast_dtype = torch.get_autocast_dtype("cuda") if torch.cuda.is_available() else torch.float16

        # Save tensor leaves for backward (move to CPU to save GPU memory)
        # Also save original devices to ensure backward returns grads on correct device.
        saved_tensor_leaves = []
        input_devices = []
        ctx.act_pinned = False
        if _act_pinned_enabled() and any(isinstance(t, torch.Tensor) and t.device.type == "cuda" for t in tensor_leaves):
            # Async path: copy each CUDA leaf into a reused pinned buffer on a dedicated
            # stream so the D2H overlaps with the block forward. record_stream keeps the
            # source memory alive until the copy completes; the H2D in backward runs on
            # the same stream, so stream ordering guarantees the pinned data is ready.
            stream = _get_act_stream(target_device)
            stream.wait_stream(torch.cuda.current_stream())
            block_id = id(block)
            with torch.cuda.stream(stream):
                for idx, tensor in enumerate(tensor_leaves):
                    input_devices.append(tensor.device)
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                        shape = tuple(tensor.shape)
                        buf = _acquire_act_pinned((block_id, idx, shape, tensor.dtype), shape, tensor.dtype)
                        buf.copy_(tensor.detach(), non_blocking=True)
                        tensor.record_stream(stream)
                        saved_tensor_leaves.append(buf)
                    else:
                        saved_tensor_leaves.append(tensor.detach().cpu())
            ctx.act_pinned = True
        else:
            for tensor in tensor_leaves:
                saved_tensor_leaves.append(tensor.detach().cpu())
                input_devices.append(tensor.device)

        ctx.saved_tensor_leaves = saved_tensor_leaves
        ctx.input_devices = input_devices

        # Move inputs to target device (GPU) for computation and rebuild nested args.
        gpu_tensor_leaves = [tensor.to(device=target_device, non_blocking=True) for tensor in tensor_leaves]
        gpu_args = _rebuild_args(arg_specs, gpu_tensor_leaves)

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
            offload_fn(block, torch.device("cpu"))

        # Extract output tensors
        # Gradient checkpointing REQUIRES tensor outputs to track gradients.
        output_tensors = []
        if isinstance(outputs, tuple):
            ctx.num_outputs = len(outputs)
            for out in outputs:
                if out is None:
                    pass  # Skip None outputs
                elif hasattr(out, "x") and isinstance(out.x, torch.Tensor):
                    output_tensors.append(out.x.detach().requires_grad_(out.x.requires_grad))
                elif isinstance(out, torch.Tensor):
                    output_tensors.append(out.detach().requires_grad_(out.requires_grad))
        else:
            ctx.num_outputs = 1
            if isinstance(outputs, torch.Tensor):
                output_tensors.append(outputs.detach().requires_grad_(outputs.requires_grad))
            elif hasattr(outputs, "x") and isinstance(outputs.x, torch.Tensor):
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
        if target_device is None or target_device.type == "cpu":
            for g in grad_outputs:
                if g is not None and isinstance(g, torch.Tensor) and g.device.type != "cpu":
                    target_device = g.device
                    break

        # === CRITICAL: Load weights for this block ===
        if load_fn is not None:
            load_fn(block, target_device)
        # Ensure FP8 weights/scale are on compute device before recompute.
        ensure_fp8_modules_on_device(block, target_device, skip_trainable=False)

        # Restore inputs from CPU to Target Device and enable grad tracking
        tensor_inputs = []
        detached_tensor_inputs = []

        if getattr(ctx, "act_pinned", False) and target_device.type == "cuda":
            # Async H2D from pinned buffers on the dedicated stream, then make the
            # compute stream wait so the recompute below sees the loaded tensors.
            stream = _get_act_stream(target_device)
            gpu_leaves = []
            with torch.cuda.stream(stream):
                for tensor in ctx.saved_tensor_leaves:
                    gpu = tensor.to(target_device, non_blocking=True)
                    gpu.record_stream(torch.cuda.current_stream())
                    gpu_leaves.append(gpu)
            torch.cuda.current_stream().wait_stream(stream)
            for gpu in gpu_leaves:
                detached = gpu.detach()
                if _tensor_can_require_grad(detached):
                    detached = detached.requires_grad_(True)
                tensor_inputs.append(detached)
                detached_tensor_inputs.append(detached)
        else:
            for tensor in ctx.saved_tensor_leaves:
                tensor_gpu = tensor.to(target_device)
                detached = tensor_gpu.detach()
                if _tensor_can_require_grad(detached):
                    detached = detached.requires_grad_(True)
                tensor_inputs.append(detached)
                detached_tensor_inputs.append(detached)

        inputs = _rebuild_args(ctx.arg_specs, tensor_inputs)

        # Restore RNG state
        if ctx.preserve_rng_state:
            rng_state = torch.get_rng_state()
            cuda_rng_device = getattr(ctx, "cuda_rng_device", None)
            cuda_rng_state = (
                torch.cuda.get_rng_state(cuda_rng_device) if cuda_rng_device is not None and torch.cuda.is_available() else None
            )
            torch.set_rng_state(ctx.cpu_rng_state)
            if cuda_rng_device is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ctx.cuda_rng_state, cuda_rng_device)

        # Recompute forward with gradients
        with torch.enable_grad():
            if ctx.autocast_enabled:
                with torch.autocast("cuda", dtype=ctx.autocast_dtype):
                    outputs = run_forward(*inputs)
            else:
                outputs = run_forward(*inputs)

        # Restore RNG state
        if ctx.preserve_rng_state:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, cuda_rng_device)

        # Extract output tensors for grad computation
        output_tensors = []
        if isinstance(outputs, tuple):
            for out in outputs:
                if out is not None and hasattr(out, "x") and isinstance(out.x, torch.Tensor):
                    output_tensors.append(out.x)
                elif isinstance(out, torch.Tensor):
                    output_tensors.append(out)
        elif isinstance(outputs, torch.Tensor):
            output_tensors.append(outputs)
        elif hasattr(outputs, "x") and isinstance(outputs.x, torch.Tensor):
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
                for i, inp in enumerate(detached_tensor_inputs):
                    if inp.requires_grad:
                        g = inp.grad
                    else:
                        g = None
                    orig_device = input_devices[i]
                    if g is not None and orig_device is not None and g.device != orig_device:
                        g = g.to(orig_device)
                    full_grads.append(g)
                tensor_grads = tuple(full_grads)
            else:
                tensor_grads = (None,) * len(detached_tensor_inputs)
        else:
            tensor_grads = (None,) * len(detached_tensor_inputs)

        # === CRITICAL: Offload weights after this block's backward ===
        if offload_fn is not None:
            offload_fn(block, torch.device("cpu"))

        # Return grads:
        # (None for run_forward, block, load_fn, offload_fn, preserve_rng_state, arg_specs) + tensor leaf grads
        return (None, None, None, None, None, None) + tensor_grads


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
    arg_specs, tensor_leaves = _pack_args(args)

    # Detect target device from args
    target_device = _get_device_from_args(args)

    # Check if we need checkpointing
    args_require_grad = any(tensor.requires_grad for tensor in tensor_leaves)

    if not args_require_grad and not torch.is_grad_enabled():
        # Truly inference / no-grad mode - skip checkpointing
        if load_fn is not None:
            load_fn(block, target_device)
            gpu_args = [_to_device_recursive(arg, target_device) for arg in args]

            with torch.no_grad():
                outputs = function(*gpu_args)

            if offload_fn is not None:
                offload_fn(block, torch.device("cpu"))
            return outputs

        return function(*args)

    # If we are here, we need checkpointing/graph tracking.
    # If inputs don't require grad (e.g. frozen encoder), autograd.Function won't track.
    # We must force at least one input to require grad.
    if not args_require_grad:
        new_tensor_leaves = list(tensor_leaves)
        for i, tensor in enumerate(new_tensor_leaves):
            if _tensor_can_require_grad(tensor):
                new_tensor_leaves[i] = tensor.detach().requires_grad_(True)
                tensor_leaves = new_tensor_leaves
                break

    outputs = BlockCheckpointFunction.apply(function, block, load_fn, offload_fn, preserve_rng_state, arg_specs, *tensor_leaves)

    return outputs
