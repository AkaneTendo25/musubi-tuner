import torch
import torch.nn as nn

from .ctx_normalize import Normalize
from .ctx_parallel_causal_conv import ContextParallelCausalConv3d
from .utils_layers import SafeConv3d as Conv3d
from .utils_layers import nonlinearity
from ..utils import ckpt_wrap, identity


class ContextParallelResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        gather_norm=False,
        norm_chunks_num=1,
        modulated_norm=Normalize,
        checkpoint=True,
        padding_mode=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = modulated_norm(
            in_channels, zq_ch=zq_ch, add_conv=add_conv, gather=gather_norm, chunks_num=norm_chunks_num, padding_mode=padding_mode
        )

        self.conv1 = ContextParallelCausalConv3d(
            chan_in=in_channels, chan_out=out_channels, kernel_size=3, padding_mode=padding_mode
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = modulated_norm(
            out_channels, zq_ch=zq_ch, add_conv=add_conv, gather=gather_norm, chunks_num=norm_chunks_num, padding_mode=padding_mode
        )
        # self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ContextParallelCausalConv3d(
            chan_in=out_channels, chan_out=out_channels, kernel_size=3, padding_mode=padding_mode
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ContextParallelCausalConv3d(
                    chan_in=in_channels, chan_out=out_channels, kernel_size=3, padding_mode=padding_mode
                )
            else:
                self.nin_shortcut = Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

        self.wrapper = identity
        self.wrapper_2 = ckpt_wrap if checkpoint else identity

    def forward_resnet_block(self, x, zq=None, clear_fake_cp_cache=True):
        h = x

        if zq is not None:
            h = self.wrapper(self.norm1, (h, zq, clear_fake_cp_cache))
        else:
            h = self.wrapper(self.norm1, h)

        h = self.wrapper(nonlinearity, h)
        h = self.wrapper(self.conv1, (h, clear_fake_cp_cache))

        if zq is not None:
            h = self.wrapper(self.norm2, (h, zq, clear_fake_cp_cache))
        else:
            h = self.wrapper(self.norm2, h)

        h = self.wrapper(nonlinearity, h)
        h = self.wrapper(self.conv2, (h, clear_fake_cp_cache))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.wrapper(self.conv_shortcut, (x, clear_fake_cp_cache))
            else:
                x = self.wrapper(self.nin_shortcut, x)

        return x + h

    def forward(self, x, zq=None, clear_fake_cp_cache=True):
        return self.wrapper_2(self.forward_resnet_block, (x, zq, clear_fake_cp_cache))
