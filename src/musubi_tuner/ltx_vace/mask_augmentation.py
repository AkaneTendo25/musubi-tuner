"""
Mask augmentation for VACE training.

Applies random augmentations to binary masks during training to improve
generalization. The VACE paper identifies this as critical — without mask
augmentation, the model overfits to specific mask shapes.

Supported augmentation modes:
  - original: keep mask as-is
  - hull: convex hull of mask contours
  - bbox: bounding box of mask
  - expand: dilate mask with random asymmetric expansion
  - hull_expand: convex hull + dilation
  - bbox_expand: bounding box + dilation

Reference: VACE paper (arXiv:2503.07598), Section 3.2
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MaskAugmentor:
    """Applies random augmentations to VACE binary masks.

    Args:
        modes: List of augmentation modes to sample from.
        weights: Sampling weights for each mode (must match len(modes)).
        expand_ratio_range: (min, max) range for dilation ratio relative to mask size.
        expand_iters_range: (min, max) range for dilation iterations.
        p: Probability of applying any augmentation (vs keeping original).
    """

    # Default mode weights from VACE paper
    DEFAULT_MODES = ["original", "expand", "hull", "hull_expand", "bbox", "bbox_expand"]
    DEFAULT_WEIGHTS = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]

    def __init__(
        self,
        modes: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        expand_ratio_range: tuple[float, float] = (0.01, 0.15),
        expand_iters_range: tuple[int, int] = (1, 8),
        p: float = 0.8,
    ):
        self.modes = modes or self.DEFAULT_MODES
        self.weights = weights or self.DEFAULT_WEIGHTS
        assert len(self.modes) == len(self.weights)
        self.expand_ratio_range = expand_ratio_range
        self.expand_iters_range = expand_iters_range
        self.p = p

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        """Augment a binary mask.

        Args:
            mask: Binary mask tensor (B, 1, F, H, W) or (1, F, H, W) or (F, H, W).
                  Values should be 0 or 1.

        Returns:
            Augmented mask with same shape and dtype.
        """
        if random.random() > self.p:
            return mask

        mode = random.choices(self.modes, weights=self.weights, k=1)[0]

        squeeze_dims = []
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 0]
        elif mask.dim() == 4:
            mask = mask.unsqueeze(0)
            squeeze_dims = [0]

        result = self._apply_mode(mask, mode)

        for _ in squeeze_dims:
            result = result.squeeze(0)

        return result

    def _apply_mode(self, mask: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply a specific augmentation mode. Mask shape: (B, 1, F, H, W)."""
        if mode == "original":
            return mask
        elif mode == "expand":
            return self._expand(mask)
        elif mode == "hull":
            return self._convex_hull(mask)
        elif mode == "hull_expand":
            return self._expand(self._convex_hull(mask))
        elif mode == "bbox":
            return self._bounding_box(mask)
        elif mode == "bbox_expand":
            return self._expand(self._bounding_box(mask))
        else:
            logger.warning("Unknown mask augmentation mode '%s', returning original", mode)
            return mask

    def _expand(self, mask: torch.Tensor) -> torch.Tensor:
        """Dilate mask with random asymmetric expansion.

        Uses max-pooling with asymmetric padding to simulate
        directional dilation (more expansion left/right/top/bottom).
        """
        B, C, F, H, W = mask.shape
        ratio = random.uniform(*self.expand_ratio_range)
        iters = random.randint(*self.expand_iters_range)

        # Compute per-direction expansion in pixels
        expand_h = max(1, int(H * ratio))
        expand_w = max(1, int(W * ratio))

        # Random asymmetric padding (different per direction)
        for _ in range(iters):
            top = random.randint(0, expand_h)
            bottom = expand_h - top
            left = random.randint(0, expand_w)
            right = expand_w - left

            # Process each frame with 2D max-pool dilation
            frames = []
            for f in range(F):
                frame = mask[:, :, f]  # (B, 1, H, W)
                padded = F.pad(frame, (left, right, top, bottom), mode="constant", value=0)
                kernel_h = top + bottom + 1
                kernel_w = left + right + 1
                dilated = F.max_pool2d(
                    padded, kernel_size=(kernel_h, kernel_w), stride=1, padding=0
                )
                # Crop back to original size if needed
                if dilated.shape[-2] > H:
                    dilated = dilated[:, :, :H, :]
                if dilated.shape[-1] > W:
                    dilated = dilated[:, :, :, :W]
                frames.append(dilated)

            mask = torch.stack(frames, dim=2)

        return (mask > 0.5).to(mask.dtype)

    def _convex_hull(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute per-frame convex hull of mask.

        Approximates convex hull using row/column spans:
        for each row with any active pixel, fills from leftmost to rightmost;
        for each column with any active pixel, fills from topmost to bottommost.
        This is fast and gives a reasonable convex-hull-like result.
        """
        B, C, F, H, W = mask.shape
        result = mask.clone()

        for b in range(B):
            for f in range(F):
                frame = result[b, 0, f]  # (H, W)
                if frame.sum() == 0:
                    continue

                # Row-wise fill: for each row, fill between first and last active pixel
                for row in range(H):
                    active = torch.where(frame[row] > 0.5)[0]
                    if len(active) > 0:
                        frame[row, active[0]:active[-1] + 1] = 1.0

                # Column-wise fill: for each column, fill between first and last active pixel
                for col in range(W):
                    active = torch.where(frame[:, col] > 0.5)[0]
                    if len(active) > 0:
                        frame[active[0]:active[-1] + 1, col] = 1.0

                result[b, 0, f] = frame

        return result

    def _bounding_box(self, mask: torch.Tensor) -> torch.Tensor:
        """Fill the bounding box of the mask per frame."""
        B, C, F, H, W = mask.shape
        result = torch.zeros_like(mask)

        for b in range(B):
            for f in range(F):
                frame = mask[b, 0, f]  # (H, W)
                if frame.sum() == 0:
                    continue

                rows = torch.where(frame.any(dim=1))[0]
                cols = torch.where(frame.any(dim=0))[0]
                if len(rows) > 0 and len(cols) > 0:
                    result[b, 0, f, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = 1.0

        return result


def augment_mask_batch(
    masks: torch.Tensor,
    augmentor: Optional[MaskAugmentor] = None,
) -> torch.Tensor:
    """Augment a batch of masks independently.

    Each mask in the batch gets a different random augmentation.

    Args:
        masks: (B, 1, F, H, W) binary masks.
        augmentor: MaskAugmentor instance (creates default if None).

    Returns:
        Augmented masks with same shape.
    """
    if augmentor is None:
        augmentor = MaskAugmentor()

    results = []
    for i in range(masks.shape[0]):
        results.append(augmentor(masks[i:i + 1]))
    return torch.cat(results, dim=0)
