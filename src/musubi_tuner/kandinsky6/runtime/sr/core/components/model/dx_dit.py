"""DX π-Flow DiT wrapper (LAY-396).

Wraps the SR ``DiffusionTransformer3D`` so its output head emits ``n_grid``
velocity grids instead of one: the final projection width becomes
``out_visual_dim * n_grid`` and the forward reshapes the trailing channel axis
into ``(n_grid, out_visual_dim)``.

Warm-start replicates the teacher's single output head ``n_grid`` times (see
``musubi_tuner.kandinsky6.runtime.sr/train/checkpoint.py:load_and_replicate_fsdp_model``), so at step 0
every grid slot equals the teacher's velocity — which shows up as a low starting
training loss rather than a dedicated identity test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.dit import DiffusionTransformer3D

if TYPE_CHECKING:
    import torch


class DXDiTWrapper(DiffusionTransformer3D):
    """SR DiT whose output head emits ``n_grid`` velocity grids per token."""

    def __init__(self, conf: dict, out_visual_dim: int, n_grid: int) -> None:
        """Widen the DiT output head to ``out_visual_dim * n_grid`` velocity grids."""
        conf_modified = dict(conf)
        conf_modified["out_visual_dim"] = out_visual_dim * n_grid
        super().__init__(**conf_modified)
        self.n_grid = int(n_grid)
        self.base_out_visual_dim = int(out_visual_dim)
        self.dx_out_visual_dim = int(out_visual_dim)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        """Run the base DiT and split the head into ``(B, n_grid, *spatial, out_visual_dim)``."""
        # Base DiT returns (B, *spatial, n_grid * out_visual_dim) (4D for SR:
        # B, H, W, C). Split the trailing channel axis into (n_grid, out_visual_dim)
        # — the OutLayer unpatchify is visual_dim-major, so channel index ==
        # grid * out_visual_dim + out_channel, which this view recovers exactly
        # (matches the repeat(n_grid) warm-start layout). Then move the grid axis
        # to position 1 so the result is (B, n_grid, *spatial, out_visual_dim),
        # matching the n_grid=1 path (v0.unsqueeze(1)) and what DXPolicy expects.
        v = super().forward(*args, **kwargs)
        return v.view(*v.shape[:-1], self.n_grid, self.dx_out_visual_dim).movedim(-2, 1)
