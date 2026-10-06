import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CausalConv3d
from .utils import cast_tuple, nonlinearity, SafeConv3d as Conv3d


class CachedGroupNorm(nn.GroupNorm):
    def group_forward(self, x_input, expectation=None, variance=None, return_stat=False):
        input_dtype = x_input.dtype
        x = x_input.to(torch.float32)
        chunks = torch.chunk(x, self.num_groups, dim=1)
        if expectation is None:
            ch_mean = [torch.mean(chunk, dim=(1, 2, 3, 4), keepdim=True) for chunk in chunks]
        else:
            ch_mean = expectation

        if variance is None:
            ch_var = [torch.var(chunk, dim=(1, 2, 3, 4), keepdim=True, unbiased=False) for chunk in chunks]
        else:
            ch_var = variance

        x_norm = [(chunk - mean) / torch.sqrt(var + self.eps) for chunk, mean, var in zip(chunks, ch_mean, ch_var)]
        x_norm = torch.cat(x_norm, dim=1)

        x_norm.mul_(self.weight.data.view(1, -1, 1, 1, 1))
        x_norm.add_(self.bias.data.view(1, -1, 1, 1, 1))

        x_out = x_norm.to(input_dtype)
        if return_stat:
            return x_out, ch_mean, ch_var
        return x_out

    def forward(self, x, cache: dict):
        out = super().forward(x)
        if cache.get('mean') is None and cache.get('var') is None:
            cache['mean'] = 1
            cache['var'] = 1
        return out


def Normalize(in_channels, gather=False, **kwargs):
    return CachedGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CachedCausalConv3d(CausalConv3d):
    def forward(self, input_, cache: dict, fixed_stride=False):
        t_stride = self.stride[0]
        padding_3d = (self.height_pad, self.height_pad, self.width_pad, self.width_pad, 0, 0)
        input_parallel = F.pad(input_, padding_3d,
            mode="constant" if self.padding_mode == 'zeros' else (self.padding_mode or 'replicate'),
            value=0 if self.padding_mode in ['constant', 'zeros'] else None)

        if cache['padding'] is None:
            first_frame = input_parallel[:, :, :1]
            time_pad_shape = [i for i in first_frame.shape]
            time_pad_shape[2] = self.time_pad
            padding = first_frame.expand(time_pad_shape)
        else:
            padding = cache['padding']

        out_size = [i for i in input_.shape]
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (input_.size(2) + 1) // 2
        output = torch.empty(tuple(out_size), dtype=input_.dtype, device=input_.device)

        offset_out = math.ceil(padding.size(2) / t_stride)  # forward on `padding_poisoned` should take exactly this range
        offset_in = offset_out * t_stride - padding.size(2) # to make forward on `input_parallel` take slice starting with this index

        if offset_out > 0:
            padding_poisoned = torch.cat([padding, input_parallel[:, :, :offset_in + self.time_kernel_size - t_stride]], dim=2)
            output[:, :, :offset_out] = self.conv(padding_poisoned)

        if offset_out < output.size(2):
            output[:, :, offset_out:] = self.conv(input_parallel[:, :, offset_in:])

        if t_stride == 2 and not fixed_stride and cache['padding'] is not None:
            expected_pad_size = padding.size(2) - 1
            offset_out = math.ceil(expected_pad_size / t_stride)
            offset_in = offset_out * t_stride - expected_pad_size

        # exact formula, doesn't depend on size of segments
        pad_offset = offset_in + t_stride * math.trunc((input_parallel.size(2) - offset_in - self.time_kernel_size) / t_stride) + t_stride

        # this condition will be executed ONLY for old models
        if t_stride == 2 and not fixed_stride:
            pad_offset -= 1

        if pad_offset < 0: # specific to small chunks (for inference on high resolution videos)
            cache['padding'] = torch.cat([padding[:, :, pad_offset:], input_parallel], dim=2)
        else:
            cache['padding'] = torch.clone(input_parallel[:, :, pad_offset:])

        return output


class CachedCausalResnetBlock3D(nn.Module):
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
        normalization=Normalize,
        padding_mode=None
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv
        )

        self.conv1 = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
            padding_mode=padding_mode
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv
        )
        # self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CachedCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
            padding_mode=padding_mode
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CachedCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                    padding_mode=padding_mode
                )
            else:
                self.nin_shortcut = Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb, layer_cache, zq=None):
        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()
        h = x

        if zq is None:
            h = self.norm1(h, cache=layer_cache['norm1'])
        else:
            h = self.norm1(h, zq, cache=layer_cache['norm1'])

        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()

        # h = nonlinearity(h)
        h = F.silu(h, inplace=True)
        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()

        h = self.conv1(h, cache=layer_cache['conv1'])
        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        if zq is None:
            h = self.norm2(h, cache=layer_cache['norm2'])
        else:
            h = self.norm2(h, zq, cache=layer_cache['norm2'])

        # h = nonlinearity(h)
        h = F.silu(h, inplace=True)
        # h = self.dropout(h)
        h = self.conv2(h, cache=layer_cache['conv2'])

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, cache=layer_cache['conv_shortcut'])
            else:
                x = self.nin_shortcut(x)

        return x + h


class CachedPXSDownsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2, version=1, fixed_stride=False, padding_mode=None):
        super().__init__()
        self.temporal_compress = compress_time
        self.fixed_stride = fixed_stride
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.version = version
        out_channels = in_channels * 2 if version > 1 else in_channels

        self.spatial_conv = Conv3d(in_channels, out_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2),
                                      padding=(0, 1, 1),
                                      padding_mode=padding_mode or 'reflect')

        if self.temporal_compress:
            if version == 2:
                self.temporal_conv = nn.Sequential(CachedCausalConv3d(out_channels, out_channels,
                                                                      kernel_size=(2, 1, 1),
                                                                      stride=(1, 1, 1),
                                                                      dilation=(1, 1, 1),
                                                                      padding_mode=padding_mode),
                                                   CachedCausalConv3d(out_channels, out_channels,
                                                                      kernel_size=(2, 1, 1),
                                                                      stride=(2, 1, 1),
                                                                      dilation=(1, 1, 1),
                                                                      padding_mode=padding_mode))
            else: # 1 or 3
                self.temporal_conv = CachedCausalConv3d(out_channels, out_channels,
                                        kernel_size=(3, 1, 1),
                                        stride=(2, 1, 1),
                                        dilation=(1, 1, 1),
                                        padding_mode=padding_mode)

        self.linear = nn.Conv3d(out_channels, out_channels,
                                kernel_size=1,
                                stride=1)

    def spatial_downsample(self, input_):
        # PixelShuffle part
        pxs_input = rearrange(input_, 'b c t h w -> (b t) c h w')
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        if self.version > 1:
            pxs_interm_view = pxs_interm.view(b, c // self.factor, self.factor, h, w)
        else: #
            pxs_interm_view = pxs_interm.view(b, c // self.factor ** 2, self.factor ** 2, h, w)
        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = rearrange(pxs_out, '(b t) c h w -> b c t h w', t=input_.size(2))

        # Downsampling by 3D-convolution
        conv_out = self.spatial_conv(input_)

        # adding it all together
        return conv_out + pxs_out

    def temporal_downsample(self, input_, cache):
        # Interpolation part
        permuted = rearrange(input_, "b c t h w -> (b h w) c t")
        if cache[0]['padding'] is None:
            first, rest = permuted[..., :1], permuted[..., 1:]

            if rest.size(-1) > 0:
                rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
                full_interp = torch.cat([first, rest_interp], dim=-1)
            else:
                full_interp = first
        else:
            rest = permuted
            if rest.size(-1) > 0:
                full_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)

        full_interp = rearrange(full_interp, "(b h w) c t -> b c t h w", h=input_.size(-2), w=input_.size(-1))

        # Downsampling by convolution
        if self.version == 1:
            conv_out = self.temporal_conv(input_, cache[0], fixed_stride=self.fixed_stride)
        elif self.version == 2:
            conv_out = self.temporal_conv[0](input_, cache[0], fixed_stride=self.fixed_stride)
            conv_out = self.temporal_conv[1](conv_out, cache[1], fixed_stride=self.fixed_stride)

        return conv_out + full_interp

    def forward(self, x, cache):
        # SPATIAL DOWNSAMPLE
        out = self.spatial_downsample(x)

        if self.temporal_compress:
            # TEMPORAL DOWNSAMPLE
            out = self.temporal_downsample(out, cache=cache)

        return self.linear(out)


class CachedSpatialNorm3D(nn.Module):
    """
      Looking at `forward`, it seems this class should be renamed to `DecoderSpatialNorm` or something similar,
      because it's not usual forward, but a conditional one.
    """
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        padding_mode=None,
        normalization=Normalize,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = normalization(in_channels=f_channels, **norm_layer_params)

        self.add_conv = add_conv
        if add_conv:
            self.conv = CachedCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
                padding_mode=padding_mode
            )

        self.conv_y = Conv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )
        self.conv_b = Conv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )

    def forward(self, f, zq, cache):
        f_shape = [s for s in f.shape]

        if cache['norm']['mean'] is None and cache['norm']['var'] is None:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]

            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")

            if zq.size(2) > 1:
                zq_rest_splits = torch.split(zq_rest, 32, dim=1)
                interpolated_splits = [
                    F.interpolate(split, size=f_rest_size, mode="nearest") for split in zq_rest_splits
                ]

                zq_rest = torch.cat(interpolated_splits, dim=1)
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            f_size = f.shape[-3:]
            zq_splits = torch.split(zq, 32, dim=1)
            interpolated_splits = [
                F.interpolate(split, size=f_size, mode="nearest") for split in zq_splits
            ]
            zq = torch.cat(interpolated_splits, dim=1)


        if self.add_conv:
            zq = self.conv(zq, cache['add_conv'])

        norm_f = self.norm_layer(f, cache['norm'])
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))

        if cache['norm']['mean'] is None and cache['norm']['var'] is None:
            cache['norm']['mean'] = 1
            cache['norm']['var'] = 1

        return norm_f


def Normalize3D(
    in_channels,
    zq_ch,
    add_conv,
    normalization=Normalize
):
    return CachedSpatialNorm3D(
        in_channels,
        zq_ch,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
        normalization=normalization
    )


class CachedPXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2, padding_mode=None):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = Conv3d(in_channels, in_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 1, 1),
                                      padding=(0, 1, 1),
                                      padding_mode=padding_mode or 'reflect')

        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(in_channels, in_channels,
                                                    kernel_size=(3, 1, 1),
                                                    stride=(1, 1, 1),
                                                    dilation=(1, 1, 1),
                                                    padding_mode=padding_mode)

        self.linear = Conv3d(in_channels, in_channels,
                                kernel_size=1,
                                stride=1)

    def spatial_upsample_NEW(self, input_):
        def conv_part(x):
            to = torch.empty_like(x)
            out = self.spatial_conv(x, write_to=to)
            return out

        # 5D interpolate keeps channels_last_3d layout intact; merging dims
        # via view() is impossible for channels_last strides (dim 1 is innermost)
        input_interp = F.interpolate(input_, scale_factor=(1, 2, 2), mode='nearest')
        input_interp.add_(conv_part(input_interp))
        return input_interp

    def temporal_upsample(self, input_, cache):
        # input_ : (T + 1) x H x W

        repeated = input_.repeat_interleave(2, dim=2)
        # repeated: (2T + 2) x H x W

        if cache['padding'] is None:
            tail = repeated[..., 1:, :, :] # tail: (2T + 1) x H x W
        else:
            tail = repeated

        conv_out = self.temporal_conv(tail, cache)
        return conv_out + tail

    def forward(self, x, cache):
        if self.temporal_compress:
            # TEMPORAL UPSAMPLE
            x = self.temporal_upsample(x, cache)

        # SPATIAL UPSAMPLE
        s_out = self.spatial_upsample_NEW(x)
        to = torch.empty_like(s_out)

        lin_out = self.linear(s_out, write_to=to)

        return lin_out
