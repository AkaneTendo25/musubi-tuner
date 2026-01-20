from dataclasses import fields, is_dataclass, replace as dataclass_replace
from typing import Any, Callable

import torch



def to_device(x: Any, device: torch.device) -> Any:
    """Recursively moves torch.Tensor objects (and containers thereof) to device.

    Supports: Tensor, list, tuple, dict, and frozen dataclass objects.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    if isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        field_updates = {f.name: to_device(getattr(x, f.name), device) for f in fields(x)}
        return dataclass_replace(x, **field_updates)
    return x


def to_cpu(x: Any) -> Any:
    """Recursively moves torch.Tensor objects (and containers thereof) to CPU."""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        field_updates = {f.name: to_cpu(getattr(x, f.name)) for f in fields(x)}
        return dataclass_replace(x, **field_updates)
    return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    
    NOTE: Modified to run in Float32 to prevent overflows/NaNs in mixed precision training.
    """
    input_dtype = x.dtype
    # Force Float32 for stability
    # This prevents 'inf' gradients caused by overflow in squared sum calculation
    x = x.to(torch.float32)
    if weight is not None:
        weight = weight.to(torch.float32)

    res = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)

    return res.to(input_dtype)




class RMSNorm(torch.nn.Module):
    """
    Robust RMSNorm module that uses the stabilized functional wrapper.
    Replaces torch.nn.RMSNorm to ensure mixed-precision compatibility (F8/F32/BF16).
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)
