from __future__ import annotations

from typing import Optional

import torch


def per_element_loss(pred: torch.Tensor, tgt: torch.Tensor, loss_type: str = "mse", huber_delta: float = 1.0) -> torch.Tensor:
    """Compute per-element unreduced loss based on loss_type."""
    if loss_type == "mae" or loss_type == "l1":
        return torch.nn.functional.l1_loss(pred.float(), tgt.float(), reduction="none")
    if loss_type == "huber" or loss_type == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(pred.float(), tgt.float(), reduction="none", beta=huber_delta)
    return torch.nn.functional.mse_loss(pred.float(), tgt.float(), reduction="none")


_per_element_loss = per_element_loss


def apply_loss_mask(
    per_elem: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce a per-element loss to a scalar with optional mask weighting."""
    if mask is None:
        return per_elem.mean(), {}

    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)

    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    metrics: dict[str, float] = {
        "mask_active": float(denom.detach().float().item()),
        "loss_unmasked": float(per_elem.detach().float().mean().item()),
    }
    if denom.item() == 0:
        loss = per_elem.mean()
    else:
        loss = (per_elem * mask_f).div(denom).mean()
    metrics["loss_masked"] = float(loss.detach().float().item())
    return loss, metrics
