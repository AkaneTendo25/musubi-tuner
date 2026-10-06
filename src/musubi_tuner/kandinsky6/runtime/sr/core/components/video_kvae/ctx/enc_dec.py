import functools

import numpy as np
import torch.nn as nn

from ..utils import nonlinearity

from .ctx_parallel_causal_conv import ContextParallelCausalConv3d
from .ctx_normalize import Normalize, Normalize3D
from .ctx_parallel_resnet import ContextParallelResnetBlock3D
from .ctx_parallel_sampling import PXSDownsample, PXSUpsample
from ..layers import RMSNorm, AttentionBlock


class Encoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=None,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=None,
        attn_mid=False,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=0,
        z_channels=16,
        double_z=True,
        padding_mode=None,
        temporal_compress_times=4,
        gather_norm=False,
        norm_chunks_num=1,
        fix_pxs=False,
        freeze=False,
        checkpoint_list=None,
        norm_type="group_norm",
        downsample_version=1,
        temporal_compress_start_level=0,
        skip_last_resolution=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.skip_last_resolution = skip_last_resolution

        if checkpoint_list is None:
            checkpoint_list = [True] * (self.num_resolutions + 1)

        assert len(checkpoint_list) == (self.num_resolutions + 1), (
            f"Checkpoint list filled wrong, expected length {self.num_resolutions + 1}, found {len(checkpoint_list)}"
        )

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compress_times)) + temporal_compress_start_level

        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)

        self.conv_in = ContextParallelCausalConv3d(  # CausalConv3d
            chan_in=in_channels, chan_out=int(in_ch_mult[0] * self.ch), kernel_size=3, padding_mode=padding_mode
        )

        normalization = Normalize if norm_type == "group_norm" else RMSNorm

        curr_res = resolution
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = round(ch * in_ch_mult[i_level])
            block_out = round(ch * ch_mult[i_level])

            if downsample_version > 1 and i_level > 0:
                block_in *= 2

            for i_block in range(self.num_res_blocks):
                block.append(
                    ContextParallelResnetBlock3D(  # CheckpointedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=self.temb_ch,
                        gather_norm=gather_norm,
                        norm_chunks_num=norm_chunks_num,
                        modulated_norm=normalization,
                        checkpoint=checkpoint_list[i_level],
                        padding_mode=padding_mode,
                    )
                )
                if attn_resolutions and i_level in attn_resolutions:
                    attn.append(AttentionBlock(block_out, normalization=normalization))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if temporal_compress_start_level <= i_level < temporal_compress_level:
                    down.downsample = PXSDownsample(
                        block_in, compress_time=True, fix_stride=fix_pxs, version=downsample_version, padding_mode=padding_mode
                    )  # DownSample3D(block_in, resamp_with_conv, compress_time=True)
                else:
                    down.downsample = PXSDownsample(
                        block_in, compress_time=False, version=downsample_version, padding_mode=padding_mode
                    )  # DownSample3D(block_in, resamp_with_conv, compress_time=False)
                curr_res = curr_res // 2
            if not skip_last_resolution or i_level != self.num_resolutions - 1:
                self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ContextParallelResnetBlock3D(  # CheckpointedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
            norm_chunks_num=norm_chunks_num,
            modulated_norm=normalization,
            checkpoint=checkpoint_list[-1],
            padding_mode=padding_mode,
        )

        if attn_mid:
            self.mid.attn = AttentionBlock(block_in, normalization=normalization)
        else:
            self.mid.attn = None

        self.mid.block_2 = ContextParallelResnetBlock3D(  # CheckpointedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
            norm_chunks_num=norm_chunks_num,
            modulated_norm=normalization,
            checkpoint=checkpoint_list[-1],
            padding_mode=padding_mode,
        )

        # end
        self.norm_out = normalization(block_in, gather=gather_norm)

        self.conv_out = ContextParallelCausalConv3d(  # CausalConv3d(
            chan_in=block_in, chan_out=2 * z_channels if double_z else z_channels, kernel_size=3, padding_mode=padding_mode
        )

    def forward(self, x, return_features=False):

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            if not self.skip_last_resolution or i_level != self.num_resolutions - 1:
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](h)
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        if self.mid.attn is not None:
            h = self.mid.attn(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)

        if return_features:
            return h

        h = self.conv_out(h)

        return h

    def get_last_layer(self):
        return self.conv_out.conv.weight


class Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=None,
        attn_mid=False,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=None,
        resolution=0,
        z_channels=16,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        padding_mode=None,
        temporal_compress_times=4,
        gather_norm=False,
        norm_chunks_num=1,
        checkpoint_list=None,
        norm_type="group_norm",
        temporal_compress_start_level=0,
        upsample_version=1,
        skip_last_resolution=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.skip_last_resolution = skip_last_resolution

        if checkpoint_list is None:
            checkpoint_list = [True] * (self.num_resolutions + 1)

        assert len(checkpoint_list) == (self.num_resolutions + 1), (
            f"Checkpoint list filled wrong, expected length {self.num_resolutions + 1}, found {len(checkpoint_list)}"
        )
        checkpoint_list.reverse()

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compress_times)) + temporal_compress_start_level

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = round(ch * ch_mult[self.num_resolutions - 1])
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = ContextParallelCausalConv3d(  # CausalConv3d(
            chan_in=z_channels, chan_out=block_in, kernel_size=3, padding_mode=padding_mode
        )

        normalization = Normalize if norm_type == "group_norm" else RMSNorm
        modulated_norm = functools.partial(Normalize3D, normalization=normalization)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ContextParallelResnetBlock3D(  # CheckpointedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            modulated_norm=modulated_norm,
            gather_norm=gather_norm,
            norm_chunks_num=norm_chunks_num,
            checkpoint=checkpoint_list[-1],
            padding_mode=padding_mode,
        )

        if attn_mid:
            self.mid.attn = AttentionBlock(block_in, normalization=normalization)
        else:
            self.mid.attn = None

        self.mid.block_2 = ContextParallelResnetBlock3D(  # CheckpointedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            modulated_norm=modulated_norm,
            gather_norm=gather_norm,
            norm_chunks_num=norm_chunks_num,
            checkpoint=checkpoint_list[-1],
            padding_mode=padding_mode,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            if upsample_version > 1 and i_level < self.num_resolutions - 1:
                block_in //= 2
            block_out = round(ch * ch_mult[i_level])

            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ContextParallelResnetBlock3D(  # CheckpointedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        modulated_norm=modulated_norm,
                        gather_norm=gather_norm,
                        norm_chunks_num=norm_chunks_num,
                        checkpoint=checkpoint_list[i_level],
                        padding_mode=padding_mode,
                    )
                )
                if attn_resolutions and i_level in attn_resolutions:
                    attn.append(AttentionBlock(block_out, normalization=normalization))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if self.num_resolutions - temporal_compress_start_level > i_level >= self.num_resolutions - temporal_compress_level:
                    up.upsample = PXSUpsample(
                        block_in, compress_time=True, version=upsample_version, padding_mode=padding_mode
                    )  # Upsample3D(block_in, with_conv=resamp_with_conv, compress_time=True)
                else:
                    up.upsample = PXSUpsample(
                        block_in, compress_time=False, version=upsample_version, padding_mode=padding_mode
                    )  # Upsample3D(block_in, with_conv=resamp_with_conv, compress_time=False)
            self.up.insert(0, up)

        self.norm_out = modulated_norm(block_in, zq_ch, add_conv=add_conv)  # , gather=gather_norm)

        self.conv_out = ContextParallelCausalConv3d(  # CausalConv3d(
            chan_in=block_in, chan_out=out_ch, kernel_size=3, padding_mode=padding_mode
        )

    def forward(self, z, up_time=None):
        self.last_z_shape = z.shape

        # z to block_in

        zq = z
        # h = self.conv_in(z, clear_cache=clear_fake_cp_cache)  # SPECIAL LINE FOR CAUSAL-RES
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, zq)  # SPECIAL LINE FOR CAUSAL-RES
        # h = self.mid.block_1(h)
        if self.mid.attn is not None:
            h = self.mid.attn(h, zq)
        h = self.mid.block_2(h, zq)  # SPECIAL LINE FOR CAUSAL-RES
        # h = self.mid.block_2(h)

        # upsampling
        if up_time is None:
            up_time = z.shape[2] > 1
        time_factor = 2 if up_time else 1
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq)

                # h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, time_factor=time_factor)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)  # , fake_cp=use_cp)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    def get_last_layer(self):
        return self.conv_out.conv.weight
