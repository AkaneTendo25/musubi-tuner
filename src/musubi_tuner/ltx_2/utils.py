from typing import Any

import torch


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
