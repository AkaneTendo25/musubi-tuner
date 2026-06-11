"""Trainer timestep sampling helpers."""

from __future__ import annotations

import argparse
import logging
import math
import random
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training.timesteps import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, get_sigmas
from musubi_tuner.utils import train_utils

logger = logging.getLogger(__name__)


def get_bucketed_timestep(self) -> float:
    if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
        return random.random()

    if len(self.timestep_range_pool) == 0:
        bucket_size = 1.0 / self.num_timestep_buckets
        for i in range(self.num_timestep_buckets):
            self.timestep_range_pool.append((i * bucket_size, (i + 1) * bucket_size))
        random.shuffle(self.timestep_range_pool)

    a, b = self.timestep_range_pool.pop()
    return random.uniform(a, b)


def get_noisy_model_input_and_timesteps(
    self,
    args: argparse.Namespace,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: Optional[List[float]],
    noise_scheduler: FlowMatchDiscreteScheduler,
    device: torch.device,
    dtype: torch.dtype,
):
    batch_size = noise.shape[0]

    if timesteps is not None:
        timesteps = torch.tensor(timesteps, device=device)

    # This function converts uniform distribution samples to logistic distribution samples.
    # The final distribution of the samples after shifting significantly differs from the original normal distribution.
    # So we cannot use this.
    # def uniform_to_normal(t_samples: torch.Tensor) -> torch.Tensor:
    #     # Clip small values to prevent log(0)
    #     eps = 1e-7
    #     t_samples = torch.clamp(t_samples, eps, 1.0 - eps)
    #     # Convert to logit space with inverse function
    #     x_samples = torch.log(t_samples / (1.0 - t_samples))
    #     return x_samples

    def uniform_to_normal_ppF(t_uniform: torch.Tensor) -> torch.Tensor:
        """Use `torch.erfinv` to compute the inverse CDF to generate values from a normal distribution."""
        # Clip small values to prevent inf in erfinv
        eps = 1e-7
        t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

        # PPF of standard normal distribution: sqrt(2) * erfinv(2q - 1)
        term = 2.0 * t_uniform - 1.0
        x_normal = math.sqrt(2.0) * torch.erfinv(term)
        return x_normal

    def uniform_to_logsnr_ppF_pytorch(t_uniform: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Use erfinv to compute the inverse CDF."""
        # Clip small values to prevent inf in erfinv
        eps = 1e-7
        t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

        term = 2.0 * t_uniform - 1.0
        logsnr = mean + std * math.sqrt(2.0) * torch.erfinv(term)
        return logsnr

    if (
        args.timestep_sampling == "uniform"
        or args.timestep_sampling == "sigmoid"
        or args.timestep_sampling == "shift"
        or args.timestep_sampling == "flux_shift"
        or args.timestep_sampling == "qwen_shift"
        or args.timestep_sampling == "logsnr"
        or args.timestep_sampling == "qinglong_flux"
        or args.timestep_sampling == "qinglong_qwen"
        or args.timestep_sampling == "flux2_shift"
    ):

        def compute_sampling_timesteps(org_timesteps: Optional[torch.Tensor]) -> torch.Tensor:
            def rand(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                nonlocal device
                return torch.rand((bs,), device=device) if org_ts is None else org_ts

            def randn(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                nonlocal device
                return uniform_to_normal_ppF(org_ts) if org_ts is not None else torch.randn((bs,), device=device)

            def rand_logsnr(bs: int, mean: float, std: float, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                nonlocal device
                logsnr = (
                    uniform_to_logsnr_ppF_pytorch(org_ts, mean, std)
                    if org_ts is not None
                    else torch.normal(mean=mean, std=std, size=(bs,), device=device)
                )
                return logsnr

            if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
                # Simple random t-based noise sampling
                if args.timestep_sampling == "sigmoid":
                    t = torch.sigmoid(args.sigmoid_scale * randn(batch_size, org_timesteps))
                else:
                    t = rand(batch_size, org_timesteps)

            elif args.timestep_sampling.endswith("shift"):
                if args.timestep_sampling == "shift":
                    shift = args.discrete_flow_shift
                else:
                    h, w = latents.shape[-2:]
                    # we are pre-packed so must adjust for packed size
                    if args.timestep_sampling == "flux_shift":
                        mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                    elif args.timestep_sampling == "flux2_shift":
                        mu = train_utils.get_lin_function(y1=0.5, y2=1.15)(h * w)
                    elif args.timestep_sampling == "qwen_shift":
                        mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                    # def time_shift(mu: float, sigma: float, t: torch.Tensor):
                    #     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma) # sigma=1.0
                    shift = math.exp(mu)

                logits_norm = randn(batch_size, org_timesteps)
                logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
                t = logits_norm.sigmoid()
                t = (t * shift) / (1 + (shift - 1) * t)

            elif args.timestep_sampling == "logsnr":
                # https://arxiv.org/abs/2411.14793v3
                logsnr = rand_logsnr(batch_size, args.logit_mean, args.logit_std, org_timesteps)
                t = torch.sigmoid(-logsnr / 2)

            elif args.timestep_sampling.startswith("qinglong"):
                # Qinglong triple hybrid sampling: mid_shift:logsnr:logsnr2 = .80:.075:.125
                # First decide which method to use for each sample independently
                decision_t = torch.rand((batch_size,), device=device)

                # Create masks based on decision_t: .80 for mid_shift, 0.075 for logsnr, and 0.125 for logsnr2
                mid_mask = decision_t < 0.80  # 80% for mid_shift
                logsnr_mask = (decision_t >= 0.80) & (decision_t < 0.875)  # 7.5% for logsnr
                logsnr_mask2 = decision_t >= 0.875  # 12.5% for logsnr with -logit_mean

                # Initialize output tensor
                t = torch.zeros((batch_size,), device=device)

                # Generate mid_shift samples for selected indices (80%)
                if mid_mask.any():
                    mid_count = mid_mask.sum().item()
                    h, w = latents.shape[-2:]
                    if args.timestep_sampling == "qinglong_flux":
                        mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                    elif args.timestep_sampling == "qinglong_qwen":
                        mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                    shift = math.exp(mu)
                    logits_norm_mid = randn(mid_count, org_timesteps[mid_mask] if org_timesteps is not None else None)
                    logits_norm_mid = logits_norm_mid * args.sigmoid_scale
                    t_mid = logits_norm_mid.sigmoid()
                    t_mid = (t_mid * shift) / (1 + (shift - 1) * t_mid)

                    t[mid_mask] = t_mid

                # Generate logsnr samples for selected indices (7.5%)
                if logsnr_mask.any():
                    logsnr_count = logsnr_mask.sum().item()
                    logsnr = rand_logsnr(
                        logsnr_count,
                        args.logit_mean,
                        args.logit_std,
                        org_timesteps[logsnr_mask] if org_timesteps is not None else None,
                    )
                    t_logsnr = torch.sigmoid(-logsnr / 2)

                    t[logsnr_mask] = t_logsnr

                # Generate logsnr2 samples with -logit_mean for selected indices (12.5%)
                if logsnr_mask2.any():
                    logsnr2_count = logsnr_mask2.sum().item()
                    logsnr2 = rand_logsnr(
                        logsnr2_count, 5.36, 1.0, org_timesteps[logsnr_mask2] if org_timesteps is not None else None
                    )
                    t_logsnr2 = torch.sigmoid(-logsnr2 / 2)

                    t[logsnr_mask2] = t_logsnr2

            return t  # 0 to 1

        t_min = args.min_timestep if args.min_timestep is not None else 0
        t_max = args.max_timestep if args.max_timestep is not None else 1000.0
        t_min /= 1000.0
        t_max /= 1000.0

        if not args.preserve_distribution_shape:
            t = compute_sampling_timesteps(timesteps)
            t = t * (t_max - t_min) + t_min  # scale to [t_min, t_max], default [0, 1]
        else:
            max_loops = 1000
            available_t = []
            for i in range(max_loops):
                t = None
                if self.num_timestep_buckets is not None:
                    t = torch.tensor([self.get_bucketed_timestep() for _ in range(batch_size)], device=device)
                t = compute_sampling_timesteps(t)
                for t_i in t:
                    if t_min <= t_i <= t_max:
                        available_t.append(t_i)
                    if len(available_t) == batch_size:
                        break
                if len(available_t) == batch_size:
                    break
            if len(available_t) < batch_size:
                logger.warning(
                    f"Could not sample {batch_size} valid timesteps in {max_loops} loops / {max_loops}ループで{batch_size}個の有効なタイムステップをサンプリングできませんでした"
                )
                available_t = compute_sampling_timesteps(timesteps)
            else:
                t = torch.stack(available_t, dim=0)  # [batch_size, ]

        timesteps = t * 1000.0
        t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
        noisy_model_input = (1 - t) * latents + t * noise

        timesteps += 1  # 1 to 1000
    else:
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=batch_size,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        # indices = (u * noise_scheduler.config.num_train_timesteps).long()
        t_min = args.min_timestep if args.min_timestep is not None else 0
        t_max = args.max_timestep if args.max_timestep is not None else 1000
        indices = (u * (t_max - t_min) + t_min).long()

        timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000

        # Add noise according to flow matching.
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    # print(f"actual timesteps: {timesteps}")
    return noisy_model_input, timesteps


def show_timesteps(self, args: argparse.Namespace):
    N_TRY = 100000
    BATCH_SIZE = 1000
    CONSOLE_WIDTH = 64
    N_TIMESTEPS_PER_LINE = 25

    noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
    # print(f"Noise scheduler timesteps: {noise_scheduler.timesteps}")

    latents = torch.zeros(BATCH_SIZE, 1, 1, 1024 // 8, 1024 // 8, dtype=torch.float16)
    noise = torch.ones_like(latents)

    # sample timesteps
    sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
    for i in tqdm(range(N_TRY // BATCH_SIZE)):
        bucketed_timesteps = None
        if args.num_timestep_buckets is not None and args.num_timestep_buckets > 1:
            self.num_timestep_buckets = args.num_timestep_buckets
            bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

        # we use noise=1, so retured noisy_model_input is same as timestep, because `noisy_model_input = (1 - t) * latents + t * noise`
        actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16
        )
        actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
        for t in actual_timesteps:
            t = int(t.item())
            sampled_timesteps[t] += 1

    # sample weighting
    sampled_weighting = [0] * noise_scheduler.config.num_train_timesteps
    for i in tqdm(range(len(sampled_weighting))):
        timesteps = torch.tensor([i + 1], device="cpu")
        weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16)
        if weighting is None:
            weighting = torch.tensor(1.0, device="cpu")
        elif torch.isinf(weighting).any():
            weighting = torch.tensor(1.0, device="cpu")
        sampled_weighting[i] = weighting.item()

    # show results
    if args.show_timesteps == "image":
        # show timesteps with matplotlib
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
        plt.title("Sampled timesteps")
        plt.xlabel("Timestep")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
        plt.title("Sampled loss weighting")
        plt.xlabel("Timestep")
        plt.ylabel("Weighting")

        plt.tight_layout()
        plt.show()

    else:
        sampled_timesteps = np.array(sampled_timesteps)
        sampled_weighting = np.array(sampled_weighting)

        # average per line
        sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
        sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

        max_count = max(sampled_timesteps)
        print(f"Sampled timesteps: max count={max_count}")
        for i, t in enumerate(sampled_timesteps):
            line = f"{(i) * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
            line += "#" * int(t / max_count * CONSOLE_WIDTH)
            print(line)

        max_weighting = max(sampled_weighting)
        print(f"Sampled loss weighting: max weighting={max_weighting}")
        for i, w in enumerate(sampled_weighting):
            line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {w:8.2f} "
            line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
            print(line)
