"""Small sampler helpers for LTX-2 inference."""

from __future__ import annotations

import torch


def resolve_ltx2_sampler(name: str | None, sampling_preset: str | None = None) -> str:
    sampler = str(name or "auto").lower()
    preset = str(sampling_preset or "").lower()
    if sampler == "auto":
        return "euler" if preset == "distilled_two_stage" else "res_2s"
    if sampler not in {"euler", "res_2s"}:
        raise ValueError("sample_sampler must be one of: auto, euler, res_2s")
    return sampler


def _phi(order: int, z: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(z.dtype).eps * 16
    if order == 1:
        series = 1 + z / 2 + z.square() / 6 + z.square() * z / 24
        return torch.where(z.abs() < eps, series, torch.expm1(z) / z)
    if order == 2:
        series = 0.5 + z / 6 + z.square() / 24 + z.square() * z / 120
        return torch.where(z.abs() < eps, series, (torch.expm1(z) - z) / z.square())
    raise ValueError("Only phi_1 and phi_2 are supported")


def res2s_midpoint(
    sample: torch.Tensor,
    denoised_sample: torch.Tensor,
    sigma_current: torch.Tensor,
    sigma_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the RES_2S midpoint sample and sigma, or None for the final zero step."""

    if float(sigma_next.detach().cpu()) <= 0.0:
        return None
    sigma_current_f = sigma_current.to(device=sample.device, dtype=torch.float32)
    sigma_next_f = sigma_next.to(device=sample.device, dtype=torch.float32)
    h = -torch.log(sigma_next_f / sigma_current_f)
    midpoint_fraction = torch.tensor(0.5, device=sample.device, dtype=torch.float32)
    phi1_mid = _phi(1, -h * midpoint_fraction)
    advance_weight = midpoint_fraction * phi1_mid
    midpoint = sample + (h * advance_weight).to(dtype=sample.dtype) * (denoised_sample - sample)
    sigma_midpoint = torch.exp(torch.log(sigma_current_f) - h * midpoint_fraction)
    return midpoint, sigma_midpoint


def res2s_step(
    sample: torch.Tensor,
    denoised_stage1: torch.Tensor,
    denoised_stage2: torch.Tensor,
    sigma_current: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Perform one ODE RES_2S step using x0/denoised predictions."""

    if float(sigma_next.detach().cpu()) <= 0.0:
        return denoised_stage1
    sigma_current_f = sigma_current.to(device=sample.device, dtype=torch.float32)
    sigma_next_f = sigma_next.to(device=sample.device, dtype=torch.float32)
    h = -torch.log(sigma_next_f / sigma_current_f)
    phi1 = _phi(1, -h)
    phi2 = _phi(2, -h)
    midpoint_fraction = torch.tensor(0.5, device=sample.device, dtype=torch.float32)
    weight_stage2 = phi2 / midpoint_fraction
    weight_stage1 = phi1 - weight_stage2
    update = h.to(dtype=sample.dtype) * (
        weight_stage1.to(dtype=sample.dtype) * (denoised_stage1 - sample)
        + weight_stage2.to(dtype=sample.dtype) * (denoised_stage2 - sample)
    )
    return sample + update
