# Convolution primitives shared by the latent upsampler architectures.

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn
from torch.nn import functional

DIMS_2 = 2
DIMS_3 = 3

TemporalPadding = Literal["zeros", "replicate", "causal"]
UpsamplePaddingMode = Literal["reflect", "zeros"]


def as_triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Expand a scalar kernel/padding spec into an explicit ``(t, h, w)`` triple."""
    return (value, value, value) if isinstance(value, int) else value


class TemporalReplicateConv3d(nn.Conv3d):
    """``Conv3d`` that repeats the edge frame along T and zero-pads H/W.

    K-VAE extends the temporal axis by repeating a boundary frame — the encoder
    and decoder both seed their causal padding with a replica of frame 0 — while
    padding H/W with zeros (``padding_mode: zeros`` in the shipped sidecar).
    ``nn.Conv3d`` cannot express that split, because ``padding_mode`` applies to
    every padded dim at once; here T is padded explicitly and H/W is left to the
    convolution.

    Without this, a zero-padded temporal axis makes the first and last latent
    frames see a hole where a neighbour should be — on K-VAE latents that hole
    lands on frame 0, which is already the odd one out (it encodes a single
    pixel frame, while every later latent aggregates four).

    Parameter names and shapes are those of ``nn.Conv3d``, so checkpoints
    trained before this padding change still load.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel as ``(kt, kh, kw)`` or a single int applied to all dims.
        padding: The "same" padding the caller would have passed to ``nn.Conv3d``.
            Its H/W entries go to the convolution; its T entry becomes the width
            of the replicate pad.
        groups: Convolution groups (``in_channels`` for a depthwise conv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        padding: int | tuple[int, int, int] = 0,
        *,
        groups: int = 1,
    ) -> None:
        """Initialize the convolution with H/W padding only, keeping T for forward."""
        pad_t, pad_h, pad_w = as_triple(padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=(0, pad_h, pad_w),
            groups=groups,
        )
        self.temporal_pad = pad_t

    def temporal_pad_lr(self) -> tuple[int, int]:
        """Split the temporal pad budget into ``(before, after)`` frame counts."""
        return (self.temporal_pad, self.temporal_pad)

    def forward(self, x: Tensor) -> Tensor:
        """Replicate-pad T, then convolve with zero padding on H and W."""
        before, after = self.temporal_pad_lr()
        if before or after:
            x = functional.pad(x, (0, 0, 0, 0, before, after), mode="replicate")
        return super().forward(x)


class TemporalCausalConv3d(TemporalReplicateConv3d):
    """``TemporalReplicateConv3d`` with K-VAE's causal temporal window.

    K-VAE's ``CausalConv3d`` spends the whole temporal pad budget in front:
    ``kT - 1`` replicas of frame 0 precede the clip and nothing follows it, so
    each 3-tap kernel reads ``(t-2, t-1, t)``. Moving the symmetric budget
    (``pad_t`` per side) to the front reproduces that window exactly, which is
    what lets decoder kernels load without recentring — and keeps every frame's
    output independent of its future, the property a chunked streaming
    inference would rely on.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        padding: int | tuple[int, int, int] = 0,
        *,
        groups: int = 1,
    ) -> None:
        """Validate that the pad budget covers exactly the causal window."""
        super().__init__(in_channels, out_channels, kernel_size, padding, groups=groups)
        kernel_t = self.kernel_size[0]
        if 2 * self.temporal_pad != kernel_t - 1:
            msg = (
                f"a causal window needs the full kT-1 = {kernel_t - 1} pad budget in front, "
                f"but padding supplies 2 * {self.temporal_pad}"
            )
            raise ValueError(msg)

    def temporal_pad_lr(self) -> tuple[int, int]:
        """Put the whole budget before the clip: taps end on the current frame."""
        return (2 * self.temporal_pad, 0)


def make_conv(dims: int, temporal_padding: TemporalPadding) -> type[nn.Module]:
    """Pick the convolution class for a stack of ``dims``-dimensional convs.

    Args:
        dims: 2 for ``Conv2d``, 3 for ``Conv3d``.
        temporal_padding: How ``dims == 3`` convolutions extend T. ``"zeros"``
            keeps the stock convolution; ``"replicate"`` repeats the edge frame;
            ``"causal"`` pads the past only (K-VAE semantics).
            Ignored for ``dims == 2``, which has no temporal axis.

    Returns:
        The convolution class to instantiate.
    """
    if dims == DIMS_2:
        return nn.Conv2d
    if temporal_padding == "replicate":
        return TemporalReplicateConv3d
    if temporal_padding == "causal":
        return TemporalCausalConv3d
    return nn.Conv3d
