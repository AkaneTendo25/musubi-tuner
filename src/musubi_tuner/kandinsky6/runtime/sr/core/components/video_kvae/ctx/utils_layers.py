import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from einops import rearrange


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

class SafeConv3d(nn.Conv3d):
    def forward(self, input):
        memory_count = input.numel() * 2 / 1024**3
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)  # NCTHW
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((prev[:, :, -kernel_size + 1 :], cur), dim=2)
                    for prev, cur in zip(input_chunks, input_chunks[1:])
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super(SafeConv3d, self).forward(input)
