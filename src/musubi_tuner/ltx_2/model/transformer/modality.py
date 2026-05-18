from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    sigma: torch.Tensor | None = None  # Shape: (B,). Current sigma for prompt/cross-attn timestep conditioning.
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None  # Shape: (B, T, T), values in [0, 1]
    a2v_cross_attention_mask: torch.Tensor | None = None  # Shape: (B, Tq_video, Tk_audio)
    v2a_cross_attention_mask: torch.Tensor | None = None  # Shape: (B, Tq_audio, Tk_video)
    dcr_detach_mask: torch.Tensor | None = None  # DCR: Shape (B, 1, 1), 1.0=normal 0.0=detach
    force_keep_mask: torch.Tensor | None = None  # TREAD: Shape (B, T), True keeps the token routable
