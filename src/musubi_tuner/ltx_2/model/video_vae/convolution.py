from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from musubi_tuner.ltx_2.model.video_vae.enums import PaddingModeType


def make_conv_nd(  # noqa: PLR0913
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    temporal_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    if not (spatial_padding_mode == temporal_padding_mode or causal):
        raise NotImplementedError("spatial and temporal padding modes must be equal")
    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    if dims == 2:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    elif dims in (3, (2, 1)):
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


class DualConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super(DualConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        # Ensure kernel_size, stride, padding, and dilation are tuples of length 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError("kernel_size must be greater than 1. Use make_linear_nd instead.")
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        # Set parameters for convolutions
        self.groups = groups
        self.bias = bias

        # Define the size of the channels after the first convolution
        intermediate_channels = out_channels if in_channels < out_channels else in_channels

        # Define parameters for the first convolution
        self.weight1 = nn.Parameter(
            torch.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Define parameters for the second convolution
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, intermediate_channels // groups, kernel_size[0], 1, 1))
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight1, a=torch.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=torch.sqrt(5))
        if self.bias:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / torch.sqrt(fan_in1)
            nn.init.uniform_(self.bias1, -bound1, bound1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / torch.sqrt(fan_in2)
            nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(
        self,
        x: torch.Tensor,
        use_conv3d: bool = False,
        skip_time_conv: bool = False,
    ) -> torch.Tensor:
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        # First convolution
        x = F.conv3d(
            x,
            self.weight1,
            self.bias1,
            self.stride1,
            self.padding1,
            self.dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        if skip_time_conv:
            return x

        # Second convolution
        x = F.conv3d(
            x,
            self.weight2,
            self.bias2,
            self.stride2,
            self.padding2,
            self.dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )

        return x

    def forward_with_2d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        b, _, _, h, w = x.shape

        # First 2D convolution
        x = rearrange(x, "b c d h w -> (b d) c h w")
        # Squeeze the depth dimension out of weight1 since it's 1
        weight1 = self.weight1.squeeze(2)
        # Select stride, padding, and dilation for the 2D convolution
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(
            x,
            weight1,
            self.bias1,
            stride1,
            padding1,
            dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        _, _, h, w = x.shape

        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x

        # Second convolution which is essentially treated as a 1D convolution across the 'd' dimension
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)

        # Reshape weight2 to match the expected dimensions for conv1d
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        # Use only the relevant dimension for stride, padding, and dilation for the 1D convolution
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]
        x = F.conv1d(
            x,
            weight2,
            self.bias2,
            stride2,
            padding2,
            dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)

        return x

    @property
    def weight(self) -> torch.Tensor:
        return self.weight2


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chunk_size = None

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode.value,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.chunk_size is None:
            if causal:
                first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
                x = torch.concatenate((first_frame_pad, x), dim=2)
            else:
                first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
                last_frame_pad = x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
                x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)
            x = self.conv(x)
            return x

        # Chunked execution
        if not causal:
             # Non-causal chunking (halo exchange)
             outputs = []
             frames = x.shape[2]
             
             # Calculate halo size based on kernel size
             # For k=3, halo is 1 frame on each side. For k=5, halo is 2 frames.
             halo_size = (self.time_kernel_size - 1) // 2
             
             # Pad the entire input first to handle boundary conditions simply
             # Pad left with first frame, right with last frame
             first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, halo_size, 1, 1))
             last_frame_pad = x[:, :, -1:, :, :].repeat((1, 1, halo_size, 1, 1))
             padded_x = torch.cat((first_frame_pad, x, last_frame_pad), dim=2)
             
             # Adjust total frames to iterate over the *original* x
             chunk_size = self.chunk_size
             stride_t = self.conv.stride[0]
             
             # Validate chunk size
             if chunk_size % stride_t != 0:
                 chunk_size = (chunk_size // stride_t) * stride_t
                 if chunk_size == 0: 
                    chunk_size = stride_t # fallback
             
             for i in range(0, frames, chunk_size):
                 # Define the window on the original x
                 start_idx = i
                 end_idx = min(i + chunk_size, frames)
                 
                 # Map these indices to the padded_x
                 # padded_x starts at -halo_size relative to x
                 # So x[0] is at padded_x[halo_size]
                 
                 # We need a chunk that includes the halo context
                 pad_start = start_idx # + halo_size - halo_size
                 pad_end = end_idx + 2 * halo_size # + halo_size + halo_size
                 
                 # Extract chunk with halos
                 # Dimensions: [B, C, chunk_size + 2*halo, H, W]
                 chunk_with_halo = padded_x[:, :, pad_start : pad_end, :, :]
                 
                 # Convolve
                 out_chunk = self.conv(chunk_with_halo)
                 
                 # Append (no cropping needed if stride=1 and padding=0, effectively 'valid' mode on time)
                 # Wait, self.conv has padding=(0, H_pad, W_pad). It doesn't pad time (dim 0 of kernel).
                 # So output depth = (Input - Kernel) / Stride + 1
                 # Input depth = chunk_size + 2*halo
                 # Kernel depth = 2*halo + 1
                 # Result depth = (chunk_size + 2*halo - (2*halo + 1)) / 1 + 1 = chunk_size
                 # This holds for stride=1.
                 # If stride > 1, we trust PyTorch striding.
                 outputs.append(out_chunk)
                 
             return torch.cat(outputs, dim=2)

        # Causal Chunking
        outputs = []
        frames = x.shape[2]
        # Pad with first frame for the very beginning
        # We maintain a 'buffer' of previous frames needed for the next chunk
        # Initial buffer is the first frame repeated
        buffer = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
        
        # Ensure chunk size is valid w.r.t stride
        stride_t = self.conv.stride[0]
        chunk_size = self.chunk_size
        if chunk_size % stride_t != 0:
            chunk_size = (chunk_size // stride_t) * stride_t
            if chunk_size == 0: 
                chunk_size = stride_t
        
        for i in range(0, frames, chunk_size):
            chunk = x[:, :, i : i + chunk_size, :, :]
            
            # Concat with buffer (previous context)
            input_chunk = torch.cat([buffer, chunk], dim=2)
            
            # Run conv
            out_chunk = self.conv(input_chunk)
            outputs.append(out_chunk)
            
            # Update buffer for next iteration
            # We need the last (time_kernel_size - 1) frames of the current *original input* chunk
            # to serve as padding for the next chunk.
            # input_chunk has shape [B, C, buffer_len + chunk_len, H, W]
            # We want the last (k-1) frames of input_chunk to be the buffer for the next.
            buffer = input_chunk[:, :, -(self.time_kernel_size - 1):, :, :]
            
        return torch.cat(outputs, dim=2)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight
