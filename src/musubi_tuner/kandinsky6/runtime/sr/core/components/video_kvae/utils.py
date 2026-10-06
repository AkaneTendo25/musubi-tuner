import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class SafeConv3d(nn.Conv3d):
    def forward(self, x, write_to=None, transform=None):
        if transform is None:
            transform = lambda x: x

        memory_count = x.numel() / (10**9)
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)  # NCTHW

            if any(ch.size(2) < kernel_size for ch in input_chunks) and kernel_size > 1:
                assert input_chunks[0].numel() * (kernel_size / input_chunks[0].size(2)) < (2 * 10**9), (
                    "frames are too big for Conv3d"
                )

                t_stride, output = self.stride[0], []
                for i in range(0, x.size(2) - kernel_size + 1, t_stride):
                    chunk = transform(x[:, :, i : i + kernel_size])
                    output.append(super(SafeConv3d, self).forward(chunk))
                output = torch.cat(output, dim=2)
                return output

            if write_to is None:
                output = []
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1 :], chunk], dim=2)
                    output.append(super(SafeConv3d, self).forward(transform(z)))
                output = torch.cat(output, dim=2)
                return output
            else:
                time_offset = 0
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1 :], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size - 1)
                    write_to[:, :, time_offset : time_offset + z_time] = super(SafeConv3d, self).forward(transform(z))
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super(SafeConv3d, self).forward(transform(x))
            else:
                write_to[...] = super(SafeConv3d, self).forward(transform(x))
                return write_to


def nonlinearity(x):
    # Apply the SiLU activation.
    return x * torch.sigmoid(x)


def custom_forward(module):
    def inside_fn(args):
        if isinstance(args, tuple):
            return module(*args)
        else:
            return module(args)

    return inside_fn


def ckpt_wrap(module, args):
    fn = custom_forward(module)
    return checkpoint(fn, args=args, use_reentrant=False)


def identity(module, args):
    if not isinstance(args, tuple):
        return module(args)
    return module(*args)
