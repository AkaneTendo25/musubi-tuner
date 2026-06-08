"""Stochastic (SDE) sampler step + per-step Gaussian log-prob for trajectory-faithful PPO/DDPO.

Per-step policy-gradient (DDPO, Black et al. 2023; DPOK, Fan et al. 2023; FlowGRPO for flow-matching
models) treats denoising as an MDP: each step's action is the next latent ``x_{t-1}`` drawn from a
Gaussian policy, and the PPO ratio is evaluated at the action actually sampled. This module provides
that Gaussian step (``sde_step``) and its log-prob (``step_log_ratio``). Phase A caches
``(x_t, x_next, x0, sigma, sigma_next)``; Phase B recomputes the policy mean from a fresh forward and
scores the cached action.

Flow-matching (LTX-2 convention, see ``ltx_2/model/ltx2_scheduler.py``):
    x_sigma  = (1 - sigma) * x0 + sigma * noise        # sigma: 1 (noise) -> 0 (clean)
    velocity = noise - x0 ;  x0 = x_t - sigma * velocity

Per-step transition (deterministic Euler/DDIM, then its DDIM-eta stochastic form):
    eps_pred = (x_t - (1 - sigma) * x0) / sigma
    mean = (1 - sigma_next) * x0 + sigma_next * sqrt(1 - eta**2) * eps_pred
    std  = sigma_next * eta                            # scalar, independent of x0 / the policy
    x_next ~ N(mean, std**2 * I)                       # eta = 0 -> deterministic Euler step

Only ``mean`` depends on the policy (through ``x0``); ``std`` is policy-independent and cancels in the
ratio ``exp(logpi_policy - logpi_old)``, so the ratio is exact and cache-replayable. ``eta in [0,1]``
sets per-step exploration.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

_LOG_RATIO_CLAMP = 10.0  # exp(+-10) keeps the ratio finite even before PPO-clip engages


def _as_tensor(sigma, ref: torch.Tensor) -> torch.Tensor:
    """Coerce a scalar/0-d/1-d sigma to a tensor broadcastable against ``ref`` ([B, ...] layout)."""
    if not torch.is_tensor(sigma):
        return torch.as_tensor(float(sigma), dtype=ref.dtype, device=ref.device)
    sigma = sigma.to(device=ref.device, dtype=ref.dtype)
    if sigma.ndim == 0:
        return sigma
    # per-sample sigma [B] -> [B, 1, 1, ...] so it broadcasts over the latent dims
    return sigma.view(sigma.shape[0], *([1] * (ref.ndim - 1)))


def implied_noise(x_t: torch.Tensor, x0: torch.Tensor, sigma) -> torch.Tensor:
    """eps_pred = (x_t - (1 - sigma) * x0) / sigma. Requires sigma > 0."""
    s = _as_tensor(sigma, x_t)
    return (x_t - (1.0 - s) * x0) / s.clamp_min(1e-8)


def sde_transition(x_t: torch.Tensor, x0: torch.Tensor, sigma, sigma_next, eta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, std)`` of the per-step Gaussian policy ``pi(x_next | x_t)``.

    ``mean`` carries the gradient of ``x0``; ``std`` is a (broadcastable) scalar independent of x0.
    """
    s = _as_tensor(sigma, x_t)
    s_next = _as_tensor(sigma_next, x_t)
    eta = float(eta)
    eps_pred = (x_t - (1.0 - s) * x0) / s.clamp_min(1e-8)
    mean = (1.0 - s_next) * x0 + s_next * math.sqrt(max(1.0 - eta * eta, 0.0)) * eps_pred
    std = (s_next * eta).clamp_min(0.0)
    return mean, std


def sde_step(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    sigma,
    sigma_next,
    eta: float,
    *,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One stochastic denoise step. Returns ``(x_next, mean, std)``.

    ``x_next = mean + std * noise`` with ``noise ~ N(0, I)`` (drawn if not supplied). At ``eta == 0``
    (or ``sigma_next == 0``) ``std == 0`` and ``x_next == mean`` is the deterministic Euler step.
    """
    mean, std = sde_transition(x_t, x0, sigma, sigma_next, eta)
    if torch.is_tensor(std) and bool((std > 0).any()):
        if noise is None:
            noise = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device, generator=generator)
        x_next = mean + std * noise
    else:
        x_next = mean
    return x_next, mean, std


def gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, reduce: str = "sum") -> torch.Tensor:
    """Diagonal-Gaussian log-density of ``x`` per sample, reduced over all non-batch dims.

    ``reduce='sum'`` is the true joint log-prob (DDPO convention). ``reduce='mean'`` divides by the
    element count (a fixed per-dim rescale; handy for tests / well-conditioned ratios). ``std`` is a
    scalar tensor (or [B,1,...]); it is broadcast. Returns shape ``[B]``.
    """
    std_t = std if torch.is_tensor(std) else torch.as_tensor(std, dtype=x.dtype, device=x.device)
    std_t = std_t.clamp_min(1e-12)
    dims = tuple(range(1, x.ndim))
    quad = ((x - mean) ** 2) / (std_t**2)
    log_norm = 2.0 * torch.log(std_t) + math.log(2.0 * math.pi)  # per-element normalizer
    per_elem = -0.5 * (quad + log_norm)
    return per_elem.mean(dim=dims) if reduce == "mean" else per_elem.sum(dim=dims)


def step_log_ratio(
    x_next: torch.Tensor,
    x0_policy: torch.Tensor,
    x0_old: torch.Tensor,
    x_t: torch.Tensor,
    sigma,
    sigma_next,
    eta: float,
) -> torch.Tensor:
    """Exact per-step PPO log importance-ratio ``log pi_policy(a|x_t) - log pi_old(a|x_t)`` at the
    taken action ``a = x_next``. The std (theta-independent, identical for both) cancels, so only the
    quadratic term remains -> numerically stable. ``x0_policy`` carries grad; ``x0_old`` is detached.
    Returns ``[B]`` (summed over latent dims), clamped to a finite range.
    """
    mean_p, std = sde_transition(x_t, x0_policy, sigma, sigma_next, eta)
    mean_o, _ = sde_transition(x_t, x0_old, sigma, sigma_next, eta)
    std_t = (std if torch.is_tensor(std) else torch.as_tensor(std, dtype=x_next.dtype, device=x_next.device)).clamp_min(1e-12)
    dims = tuple(range(1, x_next.ndim))
    # log p_policy - log p_old = -0.5/std^2 * ( ||a-mean_p||^2 - ||a-mean_o||^2 ); std_t (scalar or
    # [B,1,...]) broadcasts over the latent dims, so the same expression covers both shapes.
    quad_diff = ((x_next - mean_p) ** 2 - (x_next - mean_o) ** 2) / (std_t**2)
    log_ratio = (-0.5 * quad_diff).sum(dim=dims)
    return log_ratio.clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)


def step_is_stochastic(sigma, sigma_next, eta: float) -> bool:
    """A step carries a usable per-step log-prob only if it actually injected noise."""
    return float(eta) > 0.0 and float(sigma) > 0.0 and float(sigma_next) > 0.0
