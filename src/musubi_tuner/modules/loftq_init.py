"""LoftQ initialization for LoRA + NF4 quantization.

Pre-computes LoRA A/B matrices that compensate for NF4 quantization error via
truncated SVD of the residual ``W - dequant(Q(W))``.  This yields better quality
than random LoRA init when training on a heavily-quantized base model.

Reference: Li et al., "LoftQ: LoRA-Fine-Tuning-Aware Quantization" (2023).
"""

from typing import Callable, Tuple

import torch

import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def loftq_initialize(
    weight: torch.Tensor,
    quantize_fn: Callable,
    dequantize_fn: Callable,
    lora_rank: int,
    block_size: int = 64,
    num_iterations: int = 1,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LoftQ-initialized LoRA matrices for a single weight.

    Args:
        weight: ``[out_features, in_features]`` full-precision original weight.
        quantize_fn: ``quantize_nf4_block(tensor, block_size) -> (packed, scale)``.
        dequantize_fn: ``dequantize_nf4_block(packed, scale, out, in, bs, dtype) -> tensor``.
        lora_rank: target LoRA rank (``network_dim``).
        block_size: NF4 block size.
        num_iterations: number of alternating quantize-SVD iterations (1 is usually enough).
        device: device for SVD computation (GPU recommended).

    Returns:
        ``(lora_A, lora_B)`` — tensors of shape ``[rank, in]`` and ``[out, rank]``.
    """
    out_features, in_features = weight.shape
    W = weight.float()
    if device is not None:
        W = W.to(device)

    # Initial quantize → dequantize
    packed, scale = quantize_fn(W, block_size)
    W_q = dequantize_fn(packed, scale, out_features, in_features, block_size, torch.float32)

    lora_A = None
    lora_B = None

    for i in range(num_iterations):
        residual = W - W_q

        # Use randomized low-rank SVD — only computes top-k singular values.
        # Orders of magnitude faster than full SVD for large matrices.
        U, S, V = torch.svd_lowrank(residual, q=lora_rank)
        # U: [out, rank], S: [rank], V: [in, rank]

        sqrt_S = S.sqrt()
        lora_B = U * sqrt_S  # [out, rank]
        lora_A = (V * sqrt_S).T  # [rank, in]

        if i < num_iterations - 1:
            # Refine: re-quantize W - B@A and recompute residual
            approx = W - lora_B @ lora_A
            packed, scale = quantize_fn(approx, block_size)
            W_q = dequantize_fn(packed, scale, out_features, in_features, block_size, torch.float32)

    return lora_A, lora_B
