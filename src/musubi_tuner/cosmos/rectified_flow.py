# Copied from https://github.com/nvidia-cosmos/cosmos-predict2.5
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0

from typing import Callable

import torch


class TrainTimeWeight:
    def __init__(
        self,
        noise_scheduler,
        weight: str = "uniform",
    ):
        if weight == "reweighting":
            weight = "uniform"

        self.weight = weight
        self.noise_scheduler = noise_scheduler

        assert self.weight == "uniform", "Only uniform loss weight is supported in RF"

    def __call__(self, t, tensor_kwargs) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")

        return wts


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.distribution == "uniform":
            t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
        elif self.distribution.startswith("waver_mode_"):
            s = float(self.distribution.split("_")[-1])
            assert s - 1.29 < 1e-4, "Waver's mode distribution is only supported with s = 1.29"
            u = torch.rand((batch_size,), dtype=torch.float32)
            t = 1.0 - u - s * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
            t = t.to(device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Time distribution '{self.distribution}' is not implemented.")

        return t


class RectifiedFlow:
    def __init__(
        self,
        train_time_distribution: str = "logitnormal",
        train_time_weight_method: str = "uniform",
        shift: int = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.train_time_sampler = TrainTimeSampler(train_time_distribution)
        self.shift = shift
        self.num_train_timesteps = 1000
        self.train_time_weight = TrainTimeWeight(None, train_time_weight_method)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    def sample_train_time(self, batch_size: int):
        time = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)
        return time

    def get_discrete_timestamp(self, u, tensor_kwargs):
        u = u.squeeze()
        timesteps = self.shift * u / (1 + (self.shift - 1) * u)
        timesteps = timesteps * self.num_train_timesteps
        return timesteps.unsqueeze(0) if timesteps.ndim == 0 else timesteps

    def get_sigmas(self, timesteps, tensor_kwargs):
        sigmas = (timesteps.to(**tensor_kwargs)) / self.num_train_timesteps
        return sigmas

    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Compute interpolation x_t and velocity dot_x_t.
        Note: x_0 is noise, x_1 is clean data (rectified flow convention).
        """
        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of x_0 and x_1 must match."
        assert t.shape[0] == x_1.shape[0], "Batch size of t must match x_1."
        t = t.view(t.shape[0], *([1] * (len(x_1.shape) - 1)))
        x_t = x_0 * t + x_1 * (1 - t)
        dot_x_t = x_0 - x_1
        return x_t, dot_x_t
