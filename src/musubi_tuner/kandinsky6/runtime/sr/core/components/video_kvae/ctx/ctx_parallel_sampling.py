from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_cp_internals import get_context_parallel_rank
from .ctx_parallel_causal_conv import ContextParallelCausalConv3d


class PXSDownsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2, fix_stride=False, version=1, padding_mode=None):
        super().__init__()
        self.factor = factor
        self.temporal_compress = compress_time
        self.unshuffle = nn.PixelUnshuffle(self.factor)

        self.version = version
        out_channels = in_channels * 2 if version > 1 else in_channels

        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode=padding_mode or "reflect",
        )
        if self.temporal_compress:
            if version == 2:
                self.temporal_conv = nn.Sequential(
                    ContextParallelCausalConv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(2, 1, 1),
                        stride=(1, 1, 1),
                        dilation=(1, 1, 1),
                        fix_stride=True,
                        padding_mode=padding_mode,
                    ),
                    ContextParallelCausalConv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(2, 1, 1),
                        stride=(2, 1, 1),
                        dilation=(1, 1, 1),
                        fix_stride=True,
                        padding_mode=padding_mode,
                    ),
                )
            else:
                self.temporal_conv = ContextParallelCausalConv3d(
                    out_channels,
                    out_channels,  # ContextParallelCausalConv3d
                    kernel_size=(3, 1, 1),
                    stride=(2, 1, 1),
                    dilation=(1, 1, 1),
                    fix_stride=fix_stride,
                    padding_mode=padding_mode,
                )

        self.linear = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def spatial_downsample(self, input_):
        # PixelShuffle part
        pxs_input = rearrange(input_, "b c t h w -> (b t) c h w")
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        if self.version > 1:
            pxs_interm_view = pxs_interm.view(b, c // self.factor, self.factor, h, w)
        else:
            pxs_interm_view = pxs_interm.view(b, c // self.factor**2, self.factor**2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = rearrange(pxs_out, "(b t) c h w -> b c t h w", t=input_.size(2))

        # Downsampling by 3D-convolution
        conv_out = self.spatial_conv(input_)

        # adding it all together
        return conv_out + pxs_out

    def temporal_downsample(self, input_, fake_cp=True):
        # Interpolation part
        permuted = rearrange(input_, "b c t h w -> (b h w) c t")

        if get_context_parallel_rank() == 0 and fake_cp:
            # split first frame
            first, rest = permuted[..., :1], permuted[..., 1:]
            if rest.size(-1) > 0:
                rest = F.avg_pool1d(rest, kernel_size=2, stride=2)
            permuted = torch.cat([first, rest], dim=-1)
        else:
            permuted = F.avg_pool1d(permuted, kernel_size=2, stride=2)

        full_interp = rearrange(permuted, "(b h w) c t -> b c t h w", h=input_.size(-2), w=input_.size(-1))

        # Downsampling by 3D-convolution
        conv_out = self.temporal_conv(input_)

        return conv_out + full_interp

    def forward(self, x, fake_cp=True):
        # SPATIAL DOWNSAMPLE
        out = self.spatial_downsample(x)
        # TEMPORAL DOWNSAMPLE
        if self.temporal_compress:
            out = self.temporal_downsample(out, fake_cp)
        return self.linear(out)


class PXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int = 2, version=1, padding_mode=None):
        super().__init__()
        self.factor = factor
        self.temporal_compress = compress_time
        self.shuffle = nn.PixelShuffle(self.factor)
        out_channels = in_channels // 2 if version > 1 else in_channels
        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode=padding_mode or "reflect",
        )

        if self.temporal_compress:
            self.temporal_conv = ContextParallelCausalConv3d(
                in_channels,
                in_channels,  # ContextParallelCausalConv3d TODO: check paddings
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                padding_mode=padding_mode,
            )

        self.linear = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)

    def spatial_upsample(self, input_):
        image_like = rearrange(input_, "b c t h w -> (b t) c h w")

        # PixelShuffle part
        repeated = image_like.repeat_interleave(self.factor**2, dim=1)
        pxs_interm = self.shuffle(repeated)
        pxs_out = rearrange(pxs_interm, "(b t) c h w -> b c t h w", t=input_.size(2))

        # Upsampling by 3D-convolution
        image_like_ups = F.interpolate(image_like, scale_factor=2, mode="nearest")
        video_like_ups = rearrange(image_like_ups, "(b t) c h w -> b c t h w", t=input_.size(2))
        conv_out = self.spatial_conv(video_like_ups)

        # adding it all together
        return conv_out + pxs_out

    def temporal_upsample(self, input_, fake_cp=True, time_factor=2):
        # HERE: only one transformation (via Conv3d)

        if get_context_parallel_rank() == 0 and fake_cp:
            if isinstance(time_factor, torch.Tensor):
                time_factor = time_factor.item()
            # input_ : (1 + T) x H x W
            repeated = input_.repeat_interleave(int(time_factor), dim=2)
            # repeated: (2 + 2T) x H x W
            tail = repeated[..., int(time_factor - 1) :, :, :]
            # tail: (1 + 2T) x H x W
        else:
            # input_ : T x H x W
            tail = input_.repeat_interleave(int(time_factor), dim=2)
            # tail: 2T x H x W

        conv_out = self.temporal_conv(tail)
        return conv_out + tail

    def forward(self, x, fake_cp=True, time_factor=2):
        # TEMPORAL UPSAMPLE
        if self.temporal_compress:
            x = self.temporal_upsample(x, fake_cp=fake_cp, time_factor=time_factor)
        # SPATIAL UPSAMPLE
        out = self.spatial_upsample(x)
        return self.linear(out)
