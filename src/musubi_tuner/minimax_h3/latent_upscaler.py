"""Trained upscaler for H3 video latents.

MiniMax generate at one megapixel and reach 1440p with a second pass through
their own upscaler, which is not released. The community trained a replacement
that resizes the 24-channel video latent directly, so a second pass never leaves
latent space — which is what keeps speech and lip movement intact, unlike a
round trip out to pixels and back.

The network interleaves 3D residual blocks with depthwise temporal convolutions
and carries the requested scale as a conditioning embedding. Half the blocks run
before the resize and half after, so the resize itself is plain trilinear
interpolation while the blocks on either side decide what the new samples hold.

The released weights expect the latent normalized by the VAE channel statistics
*a second time*. ComfyUI hands out H3 latents already divided by those
statistics, exactly as :meth:`MiniMaxH3VideoVAE.encode` does, and the node these
weights ship with divides by them again before the network — so that twice
normalized space is what the training saw. Passing an ordinary normalized latent
straight in inflates the output about three times over and produces noise, which
is why the statistics are applied here and undone afterwards.
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

#: Channels in an H3 video latent.
LATENT_CHANNELS = 24

#: Frames per temporal chunk, and the overlap that hides the seams. Long clips
#: are resized in chunks because the residual stack holds every frame at full
#: width at once.
_CHUNK_FRAMES = 16


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _ResBlock(nn.Module):
    """Residual block whose normalization is shifted and scaled by the embedding."""

    def __init__(self, channels: int, embed_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_channels, 2 * channels))
        self.out_norm = _group_norm(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.in_layers(hidden)
        scale, shift = self.emb_layers(embedding).type(residual.dtype)[..., None, None, None].chunk(2, dim=1)
        residual = self.out_layers(self.out_norm(residual) * (1 + scale) + shift)
        return self.skip(hidden) + residual


class _TemporalConv(nn.Module):
    """Depthwise convolution along time only, added as a residual."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.pwconv(self.dwconv(F.silu(self.norm(hidden))))


class LatentUpscaler(nn.Module):
    """Resize an H3 video latent with a trained network instead of interpolation."""

    def __init__(
        self,
        in_channels: int = LATENT_CHANNELS,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ) -> None:
        super().__init__()
        embed_dim = 64
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        def stack(count: int) -> nn.ModuleList:
            blocks: list[nn.Module] = []
            for index in range(count):
                blocks.append(_ResBlock(channels, embed_dim, dropout))
                if temporal_every > 0 and index % temporal_every == 0:
                    blocks.append(_TemporalConv(channels, temporal_kernel))
            return nn.ModuleList(blocks)

        self.in_blocks = stack(in_blocks)
        self.out_blocks = stack(out_blocks)
        self.norm_out = _group_norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _run(self, latent: torch.Tensor, scale: float, size: tuple[int, int, int]) -> torch.Tensor:
        embedding = self.embed(torch.tensor([[scale - 1.0]], dtype=latent.dtype, device=latent.device))
        hidden = self.conv_in(latent)
        for block in self.in_blocks:
            hidden = block(hidden, embedding.expand(hidden.shape[0], -1)) if isinstance(block, _ResBlock) else block(hidden)
        hidden = F.interpolate(hidden, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            hidden = block(hidden, embedding.expand(hidden.shape[0], -1)) if isinstance(block, _ResBlock) else block(hidden)
        return self.conv_out(F.silu(self.norm_out(hidden)))

    def set_latent_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store the VAE channel statistics this network normalizes by."""
        self.register_buffer("latents_mean", mean.reshape(1, -1, 1, 1, 1).clone(), persistent=False)
        self.register_buffer("latents_std", std.reshape(1, -1, 1, 1, 1).clone(), persistent=False)

    def forward(self, latent: torch.Tensor, *, scale: float | None = None, size: tuple[int, int] | None = None) -> torch.Tensor:
        """Resize ``(B, C, T, H, W)`` spatially, leaving the frame count alone.

        Args:
            latent: video latent in the ordinary normalized space this pipeline
                uses; the second normalization the weights expect is applied here.
            scale: multiplier for height and width. Ignored when ``size`` is given.
            size: explicit latent height and width.
        """
        if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"latent must have shape [B, {LATENT_CHANNELS}, T, H, W], got {tuple(latent.shape)}")
        frames, height, width = latent.shape[-3:]
        if size is not None:
            target = (frames, int(size[0]), int(size[1]))
            scale = target[-1] / width
        elif scale is not None:
            target = (frames, int(round(height * scale)), int(round(width * scale)))
        else:
            raise ValueError("latent upscaling needs either a scale or a size")
        if target[-2:] == (height, width):
            return latent

        # Follow the latent rather than assume the buffers were moved with the
        # module: the statistics are registered by a separate call, and one made
        # after the module reached its device would otherwise leave them behind.
        mean = self.latents_mean.to(device=latent.device, dtype=latent.dtype)
        std = self.latents_std.to(device=latent.device, dtype=latent.dtype)
        latent = (latent - mean) / std

        if frames <= _CHUNK_FRAMES:
            return self._run(latent, scale, target) * std + mean

        # Chunks overlap by half the temporal kernel so a frame near a seam still
        # sees the neighbours the convolution expects; the overlap is then cut.
        overlap = next(b.dwconv.kernel_size[0] for b in self.in_blocks if isinstance(b, _TemporalConv)) // 2
        pieces = []
        start = 0
        while start < frames:
            low = max(0, start - overlap)
            high = min(frames, start + _CHUNK_FRAMES + overlap)
            piece = self._run(latent[:, :, low:high], scale, (high - low, target[1], target[2]))
            pieces.append(piece[:, :, start - low : start - low + min(_CHUNK_FRAMES, frames - start)])
            start += _CHUNK_FRAMES
        return torch.cat(pieces, dim=2) * std + mean


def load_latent_upscaler(
    path,
    device,
    dtype: torch.dtype = torch.bfloat16,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> LatentUpscaler:
    """Build the network, load released weights, and record the VAE statistics.

    The statistics come from the video VAE rather than a copy kept here, so a
    checkpoint with different ones cannot silently disagree with the upscaler.
    """
    weights = load_file(str(path))
    model = LatentUpscaler()
    model.set_latent_statistics(latents_mean, latents_std)
    info = model.load_state_dict(weights, strict=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"strict latent upscaler load failed: {info}")
    return model.to(device=device, dtype=dtype).eval()
