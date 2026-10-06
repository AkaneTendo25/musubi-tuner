from typing import Union, Tuple

import torch.nn as nn
import torch.nn.functional as F

from .utils_layers import SafeConv3d as Conv3d
from .utils_layers import cast_tuple
from .utils_distributed import fake_cp_pass_from_previous_rank


class ContextParallelCausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride=1,
        dilation=1,
        fix_stride=False,
        padding_mode=None,
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        t_kernel, h_kernel, w_kernel = kernel_size

        assert h_kernel & w_kernel & 1

        self.t_pad = t_kernel - 1
        self.h_pad = h_kernel // 2
        self.w_pad = w_kernel // 2

        self.t_kernel = t_kernel
        self.temporal_dim = 2

        stride = cast_tuple(stride, 3)
        dilation = cast_tuple(dilation, 3)
        self.conv = Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_padding = None
        self.fix_stride = stride[0] if fix_stride else 1
        self.padding_mode = padding_mode

    def forward(self, input_, clear_cache=True):

        input_parallel = fake_cp_pass_from_previous_rank(
            input_, self.temporal_dim, self.t_kernel, self.fix_stride, self.cache_padding
        )
        padding_3d = (self.w_pad, self.w_pad, self.h_pad, self.h_pad, 0, 0)

        input_parallel = F.pad(
            input_parallel,
            padding_3d,
            mode="constant" if self.padding_mode == "zeros" else (self.padding_mode or "replicate"),
            value=0 if self.padding_mode in ["zeros", "constant"] else None,
        )

        output = self.conv(input_parallel)
        return output
