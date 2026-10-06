import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctx_parallel_causal_conv import ContextParallelCausalConv3d
from .utils_distributed import _conv_gather, _conv_split
from .utils_cp_internals import get_context_parallel_rank, get_context_parallel_world_size, get_context_parallel_group


def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)


def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)


class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        cp_world_size = get_context_parallel_world_size()
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size) / cp_world_size, None, None


class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_gather(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        cp_world_size = get_context_parallel_world_size()
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size) * cp_world_size, None, None


class ContextParallelGroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs) -> None:
        self.chunks_num = kwargs.pop('chunks_num', 1)
        self.gather_flag = kwargs.pop('gather_flag', False)
        super().__init__(*args, **kwargs)

    def forward(self, input_):

        cp_rank = get_context_parallel_rank()

        if self.gather_flag :
            input_ = conv_gather_from_context_parallel_region(input_, dim=2, kernel_size=1)
            output = super().forward(input_)
            output = conv_scatter_to_context_parallel_region(output, dim=2, kernel_size=1)
        else:
            # calc statistics by chunks
            dim_size = input_.size()[2]
            if cp_rank == 0:
                dim_size -= 1
            chunk_sizes = [dim_size // self.chunks_num] * self.chunks_num
            if cp_rank == 0:
                chunk_sizes[0] += 1
            chunks = torch.split(input_, chunk_sizes, dim=2)
            output_chunks = list()
            for chunk in chunks:
                output_chunks.append(super().forward(chunk))
            output = torch.concat(output_chunks, dim=2)

        # TODO: check exactness
        # if gather_flag:
        #     cp_rank = get_context_parallel_rank()
        #     cp_world_size = get_context_parallel_world_size()
        #     cp_group = get_context_parallel_group()
        #     input_dtype = input_.dtype
        #     x = input_.to(torch.float32)

        #     temporal_len = x.shape[2]
        #     if cp_rank == 0:
        #         temporal_len -= 1
        #     total_temporal_len = temporal_len * cp_world_size + 1
        #     group_size = x.shape[1] // self.num_groups

        #     # Cacl mean over whole tensor
        #     ch_mean_sum = torch.sum(torch.mean(input_, dim=(3, 4), keepdim=True), dim=2, keepdim=True) # mean by spatial, sum by temporal
        #     torch.distributed.all_reduce(ch_mean_sum, group=cp_group) # sum by temporal over all ranks
        #     ch_mean = ch_mean_sum / total_temporal_len # mean by temporal
        #     ch_mean_chunks = torch.chunk(ch_mean, self.num_groups, dim=1) # split to groups
        #     ch_mean_chunks = [torch.mean(chunk, dim=1, keepdim=True) for chunk in ch_mean_chunks] # mean by channel in each chunk
        #     ch_mean_chunks_expanded = [torch.repeat_interleave(chunk, group_size, dim=1) for chunk in ch_mean_chunks] # expand to initial channel num
        #     ch_mean = torch.cat(ch_mean_chunks_expanded, dim=1) # concat by channel

        #     # Calc var over whole tensor
        #     repeat_shape = list(x.shape)
        #     repeat_shape[1] = 1
        #     ch_mean = ch_mean.repeat(*repeat_shape) # expand to initial shape
        #     ch_var_sum = torch.sum(torch.mean(torch.square(x - ch_mean), dim=(3, 4), keepdim=True), dim=2, keepdim=True) # mean by spatial, sum by temporal
        #     torch.distributed.all_reduce(ch_var_sum, group=cp_group) # sum by temporal over all ranks
        #     ch_var = ch_var_sum / total_temporal_len # mean by temporal
        #     ch_var_chunks = torch.chunk(ch_var, self.num_groups, dim=1) # split to groups
        #     ch_var_chunks = [torch.mean(chunk, dim=1, keepdim=True) for chunk in ch_var_chunks] # mean by channel in each chunk

        #     # Split x to groups
        #     chunks = torch.chunk(x, self.num_groups, dim=1)

        #     x_norm = [(chunk - mean) / torch.sqrt(var + self.eps) for chunk, mean, var in zip(chunks, ch_mean_chunks, ch_var_chunks)]
        #     x_norm = torch.cat(x_norm, dim=1)

        #     x_norm.mul_(self.weight.data.view(1, -1, 1, 1, 1))
        #     x_norm.add_(self.bias.data.view(1, -1, 1, 1, 1))

        #     output = x_norm.to(input_dtype)
        # else:
        #     output = super().forward(input_)

        return output

def Normalize(in_channels, gather=False, chunks_num=1, **kwargs):
    return ContextParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, gather_flag=gather, chunks_num=chunks_num)

class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        padding_mode=None,
        gather=False,
        chunks_num=1,
        normalization=Normalize,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = normalization(in_channels=f_channels, gather=gather, chunks_num=chunks_num, **norm_layer_params)

        # self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False

        self.add_conv = add_conv
        if add_conv:
            self.conv = ContextParallelCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
                padding_mode=padding_mode
            )

        self.conv_y = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
            padding_mode=padding_mode
        )
        self.conv_b = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
            padding_mode=padding_mode
        )

    def forward(self, f, zq, clear_fake_cp_cache=True):
        if f.shape[2] > 1 and get_context_parallel_rank() == 0:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")
            zq_rest = F.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            zq = torch.cat([zq_first, zq_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:], mode="nearest")

        if self.add_conv:
            zq = self.conv(zq, clear_cache=clear_fake_cp_cache)

        # f = conv_gather_from_context_parallel_region(f, dim=2, kernel_size=1)
        norm_f = self.norm_layer(f)
        # norm_f = conv_scatter_to_context_parallel_region(norm_f, dim=2, kernel_size=1)

        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


def Normalize3D(
    in_channels,
    zq_ch,
    add_conv,
    gather=False,
    chunks_num=1,
    normalization=Normalize,
    padding_mode=None
):
    return SpatialNorm3D(
        in_channels,
        zq_ch,
        gather=gather,
        chunks_num=chunks_num,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
        normalization=normalization,
        padding_mode=padding_mode
    )
