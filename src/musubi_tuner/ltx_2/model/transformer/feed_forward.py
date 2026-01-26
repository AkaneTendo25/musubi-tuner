import torch
from musubi_tuner.ltx_2.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4, chunk_size: int = 0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))
        self.chunk_size = int(chunk_size) if chunk_size else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size and x.dim() == 3 and x.shape[1] > self.chunk_size:
            outs = []
            for start in range(0, x.shape[1], self.chunk_size):
                end = min(start + self.chunk_size, x.shape[1])
                outs.append(self.net(x[:, start:end]))
            return torch.cat(outs, dim=1)
        return self.net(x)
