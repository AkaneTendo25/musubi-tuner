from typing import Union, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SafeConv3d as Conv3d
from .utils import cast_tuple


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        padding_mode=None,
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert (height_kernel_size % 2) and (width_kernel_size % 2)

        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 2

        self.stride = stride
        self.conv = Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_padding = None
        self.padding_mode = padding_mode

    def forward(self, input_):
        input_parallel = input_

        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        input_parallel = F.pad(input_parallel, padding_3d, mode=self.padding_mode or "replicate")

        output = self.conv(input_parallel)
        return output


def RMSNorm(in_channels, *args, **kwargs):
    return WanRMS_norm(n_ch=in_channels, bias=False)


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, n_ch: int, bias: bool = False) -> None:
        super().__init__()
        shape = (n_ch, 1, 1, 1)

        self.scale = n_ch**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x, *args, **kwargs):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim, normalization):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = normalization(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, zq=None):
        identity = x
        x = self.norm(x, zq)

        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity
