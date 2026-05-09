import math
from functools import lru_cache

import numpy
import torch
from musubi_tuner.ltx_2.components.protocols import SchedulerProtocol

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# LTX-2.3 distilled single-stage workflow schedule.
# This is intentionally exact: the distilled model/LoRA was tuned for these
# sigmas rather than for the default token-shifted linear schedule.
LTX23_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


def build_ltx2_sigmas(
    steps: int,
    latent: torch.Tensor | None = None,
    *,
    sigma_schedule: str = "auto",
    sampling_preset: str | None = None,
) -> torch.FloatTensor:
    """Build LTX-2 sigmas from the configured sampling schedule.

    ``auto`` uses the exact LTX-2.3 distilled schedule for the distilled
    two-stage preset at 8 steps. Otherwise it uses ``LTX2Scheduler`` with the
    latent tensor for token-count-dependent shifting.
    """

    schedule = str(sigma_schedule or "auto").lower()
    preset = str(sampling_preset or "").lower()

    use_distilled = schedule == "ltx23_distilled" or (
        schedule == "auto" and preset == "distilled_two_stage" and int(steps) == len(LTX23_DISTILLED_SIGMAS) - 1
    )
    if use_distilled:
        expected_steps = len(LTX23_DISTILLED_SIGMAS) - 1
        if int(steps) != expected_steps:
            raise ValueError(
                f"LTX-2.3 distilled sigma schedule requires {expected_steps} steps, got {steps}. "
                "Use --sample_sigma_schedule ltx or --sample_steps 8."
            )
        return torch.tensor(LTX23_DISTILLED_SIGMAS, dtype=torch.float32)

    if schedule not in {"auto", "ltx"}:
        raise ValueError("sigma_schedule must be one of: auto, ltx, ltx23_distilled")

    return LTX2Scheduler().execute(steps=int(steps), latent=latent)


class LTX2Scheduler(SchedulerProtocol):
    """
    Default scheduler for LTX-2 diffusion sampling.
    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.
    """

    def execute(
        self,
        steps: int,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> torch.FloatTensor:
        tokens = math.prod(latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR
        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)


class LinearQuadraticScheduler(SchedulerProtocol):
    """
    Scheduler with linear steps followed by quadratic steps.
    Produces a sigma schedule that transitions linearly up to a threshold,
    then follows a quadratic curve for the remaining steps.
    """

    def execute(
        self, steps: int, threshold_noise: float = 0.025, linear_steps: int | None = None, **_kwargs
    ) -> torch.FloatTensor:
        if steps == 1:
            return torch.FloatTensor([1.0, 0.0])

        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_sigma_schedule = []
        if quadratic_steps > 0:
            quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
            linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
            const = quadratic_coef * (linear_steps**2)
            quadratic_sigma_schedule = [
                quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, steps)
            ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
        return torch.FloatTensor(sigma_schedule)


class BetaScheduler(SchedulerProtocol):
    """
    Scheduler using a beta distribution to sample timesteps.
    Based on: https://arxiv.org/abs/2407.12173
    """

    shift = 2.37
    timesteps_length = 10000

    def execute(self, steps: int, alpha: float = 0.6, beta: float = 0.6) -> torch.FloatTensor:
        """
        Execute the beta scheduler.
        Args:
            steps: The number of steps to execute the scheduler for.
            alpha: The alpha parameter for the beta distribution.
            beta: The beta parameter for the beta distribution.
        Warnings:
            The number of steps within `sigmas` theoretically might be less than `steps+1`,
            because of the deduplication of the identical timesteps
        Returns:
            A tensor of sigmas.
        """
        import scipy.stats  # lazy import - only needed for BetaScheduler

        model_sampling_sigmas = _precalculate_model_sampling_sigmas(self.shift, self.timesteps_length)
        total_timesteps = len(model_sampling_sigmas) - 1
        ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
        ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps).tolist()
        ts = list(dict.fromkeys(ts))

        sigmas = [float(model_sampling_sigmas[int(t)]) for t in ts] + [0.0]
        return torch.FloatTensor(sigmas)


@lru_cache(maxsize=5)
def _precalculate_model_sampling_sigmas(shift: float, timesteps_length: int) -> torch.Tensor:
    timesteps = torch.arange(1, timesteps_length + 1, 1) / timesteps_length
    return torch.Tensor([flux_time_shift(shift, 1.0, t) for t in timesteps])


def flux_time_shift(mu: float, sigma: float, t: float) -> float:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
