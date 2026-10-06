from typing import Any, Tuple

import torch
from torch import nn


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)  # .to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3, 4],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4],
                )

    def mode(self):
        return self.mean


class DiagonalGaussianRegularizer(nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def get_kl_loss(self, z: torch.Tensor) -> torch.Tensor:
        posterior = DiagonalGaussianDistribution(z)
        kl_loss = posterior.kl()
        kl_loss = torch.mean(kl_loss, dim=0)
        return kl_loss

    def get_sample(self, z: torch.Tensor) -> torch.Tensor:
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
