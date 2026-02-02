import torch
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
from musubi_tuner.ltx_2.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4, chunk_size: int = 0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))
        self.chunk_size = int(chunk_size) if chunk_size is not None else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensure_fp8_modules_on_device(self.net[0].proj, x.device)
        ensure_fp8_modules_on_device(self.net[-1], x.device)
        chunk_size = int(getattr(self, "chunk_size", 0) or 0)
        seq_len = x.shape[1]

        # No chunking if disabled, sequence is small, or empty tensor
        if chunk_size <= 0 or seq_len <= chunk_size or seq_len == 0:
            return self.net(x)

        chunks = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunks.append(self.net(x[:, start:end]))
        return torch.cat(chunks, dim=1)
