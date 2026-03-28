# Copied from https://github.com/nvidia-cosmos/cosmos-predict2.5
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0

from typing import Tuple

import torch


class RectifiedFlowScaling:
    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1.0, loss_weight_uniform: bool = True):
        assert abs(sigma_data - 1.0) < 1e-6, "sigma_data must be 1.0 for RectifiedFlowScaling"
        self.t_scaling_factor = t_scaling_factor
        self.loss_weight_uniform = loss_weight_uniform
        if loss_weight_uniform is False:
            self.num_steps = 1000
            t = torch.linspace(0, 1, self.num_steps)
            y = torch.exp(-2 * (t - 0.5) ** 2)
            shift = y - y.min()
            weights = shift * (self.num_steps / shift.sum())
            self.weights = weights

    def sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.loss_weight_uniform:
            return (1.0 + sigma) ** 2 / sigma**2
        else:
            t = sigma / (sigma + 1)
            index = (t * self.num_steps).round().long()
            index = torch.clamp(index, 0, self.num_steps - 1)
            weights_on_device = self.weights.to(sigma.device)
            return weights_on_device[index].type_as(sigma)

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = sigma / (sigma + 1)
        c_skip = 1.0 - t
        c_out = -t
        c_in = 1.0 - t
        c_noise = t * self.t_scaling_factor
        return c_skip, c_out, c_in, c_noise
