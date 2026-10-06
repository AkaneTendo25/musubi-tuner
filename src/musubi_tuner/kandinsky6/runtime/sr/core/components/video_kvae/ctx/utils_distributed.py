import torch
import os
import torch.distributed as tdi

from .utils_cp_internals import (
    get_context_parallel_rank,
    get_context_parallel_group,
    get_context_parallel_group_rank,
    get_context_parallel_world_size,
)


def fake_cp_pass_from_previous_rank(input_, dim, kernel_size, stride, cache_padding):
    return _FakeCPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size, stride, cache_padding)


class _FakeCPConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size, stride, cache_padding):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, stride, cache_padding)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size, ctx.stride), None, None, None, None


def _drop_from_previous_rank(grad_output, dim, kernel_size, stride=1):

    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return grad_output

    cp_rank = get_context_parallel_rank()

    group = get_context_parallel_group()
    group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    grad_output = grad_output.transpose(0, dim)

    # cut first kernel_size - 1 gradients to transfer (change by stride)
    pad = (kernel_size - 1) - (stride - 1)
    transfer_shape = list(grad_output.shape)
    transfer_shape[0] = pad

    if pad <= 0:
        # Split grad_output to (grad_to_transfer, grad_output)
        grad_to_transfer = grad_output[:pad].contiguous() if cp_rank > 0 else grad_output[: (kernel_size - 1)]
        grad_output = grad_output[pad:] if cp_rank > 0 else grad_output[(kernel_size - 1) :]

        if cp_rank == 0:
            grad_output[0] = grad_output[0] + torch.sum(grad_to_transfer, dim=0)

        grad_output = grad_output.transpose(0, dim)
        return grad_output

    # print('in _drop_from_previous_rank,', 'cp_rank:', cp_rank, 'input_size:', grad_output.shape, 'pad:', pad)

    # transfer these gradients to previous rank
    send_to_rank = cp_world_size * group_rank + cp_rank - 1
    recv_from_rank = cp_world_size * group_rank + cp_rank + 1

    do_transfer = True

    if do_transfer:
        if cp_rank < cp_world_size - 1:
            recv_buffer = torch.empty(transfer_shape, device=grad_output.device, dtype=grad_output.dtype).contiguous()
            if os.environ.get("CP_DEBUG") == "INFO":
                print(
                    "CP Rank %d, receive grad from %d in group %s"
                    % (cp_world_size * group_rank + cp_rank, recv_from_rank, group.group_name if group else None),
                    recv_buffer.shape,
                )
            req_recv = tdi.irecv(recv_buffer, recv_from_rank, group=group)
            if os.environ.get("CP_DEBUG") == "INFO":
                print("CP Rank %d, grad received" % (cp_world_size * group_rank + cp_rank))
        else:
            req_recv = None

        # Wait receive complete before send
        if transfer_shape[0] * 2 > grad_output.shape[0] and req_recv is not None:
            req_recv.wait()
            # Add received gradients to last frames
            grad_output[-pad:] = grad_output[-pad:] + recv_buffer
            req_recv = None

        # Split grad_output to (grad_to_transfer, grad_output)
        grad_to_transfer = grad_output[:pad].contiguous() if cp_rank > 0 else grad_output[: (kernel_size - 1)]
        grad_output = grad_output[pad:] if cp_rank > 0 else grad_output[(kernel_size - 1) :]
        if cp_rank > 0:
            if os.environ.get("CP_DEBUG") == "INFO":
                print(
                    "CP Rank %d, send grad to %d in group %s"
                    % (cp_world_size * group_rank + cp_rank, send_to_rank, group.group_name if group else None),
                    grad_to_transfer.shape,
                )
            req_send = tdi.isend(grad_to_transfer, send_to_rank, group=group)
        else:
            # On rank 0 just add grads to first frame
            grad_output[0] = grad_output[0] + torch.sum(grad_to_transfer, dim=0)
            req_send = None

        if req_recv is not None:
            req_recv.wait()
            # Add received gradients to last frames
            grad_output[-pad:] = grad_output[-pad:] + recv_buffer

        # if req_send is not None:
        #    req_send.wait()

        # if cp_world_size > 1:
        #     if os.environ.get("CP_DEBUG") == "INFO":
        #         print("Rank %d, grad barrier reached" % global_rank)
        #     tdi.barrier(group=group)
    else:
        # Split grad_output to (grad_to_transfer, grad_output)
        grad_to_transfer = grad_output[:pad].contiguous() if cp_rank > 0 else grad_output[: (kernel_size - 1)]
        grad_output = grad_output[pad:] if cp_rank > 0 else grad_output[(kernel_size - 1) :]
        # Add grads to first frame
        grad_output[0] = grad_output[0] + torch.sum(grad_to_transfer, dim=0)

    grad_output = grad_output.transpose(0, dim)

    # print('in _drop_from_previous_rank, global_rank:', global_rank, 'cp_rank:', cp_rank, 'output_size:', grad_output.shape)

    return grad_output


def _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, stride=1, cache_padding=None):
    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    group = get_context_parallel_group()
    group_rank = get_context_parallel_group_rank()
    cp_rank = get_context_parallel_rank()
    cp_world_size = get_context_parallel_world_size()

    input_ = input_.transpose(0, dim)

    # pass from last rank
    send_to_rank = cp_world_size * group_rank + cp_rank + 1
    recv_from_rank = cp_world_size * group_rank + cp_rank - 1

    pad = (kernel_size - 1) - (stride - 1)
    if pad <= 0:
        if cp_rank == 0:
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
        input_ = input_.transpose(0, dim).contiguous()
        return input_

    # print('in _pass_from_previous_rank', 'cp_rank:', cp_rank, 'input_size:', input_.shape, ', pad:', pad)

    buffer_shape = (pad,) + input_.shape[1:]

    if cp_rank > 0:
        recv_buffer = torch.empty(buffer_shape, device=input_.device, dtype=input_.dtype).contiguous()
        if os.environ.get("CP_DEBUG") == "INFO":
            print(
                "Rank %d, receive %d from %d in group %s"
                % (cp_world_size * group_rank + cp_rank, recv_buffer.shape[0], recv_from_rank, group.group_name if group else None),
                recv_buffer.shape,
            )
        req_recv = tdi.irecv(recv_buffer, recv_from_rank, group=group)
        if os.environ.get("CP_DEBUG") == "INFO":
            print("Rank %d, received" % (cp_world_size * group_rank + cp_rank))
    else:
        # First chunk, just replicate first point
        if cache_padding is not None:
            input_ = torch.cat([cache_padding.transpose(0, dim).to(input_.device), input_], dim=0)
        else:
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
        req_recv = None

    if input_.shape[0] < buffer_shape[0] and req_recv is not None:
        # Too low chunk, wait data from previous rank before sending to next rank
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)
        req_recv = None

    if cp_rank < cp_world_size - 1:
        if os.environ.get("CP_DEBUG") == "INFO":
            print(
                "Rank %d, send %d to %d in group %s"
                % (cp_world_size * group_rank + cp_rank, input_[-pad:].shape[0], send_to_rank, group.group_name if group else None),
                input_[-pad:].shape,
            )
        send_buffer = input_[-pad:].contiguous()
        req_send = tdi.isend(send_buffer, send_to_rank, group=group)
    else:
        req_send = None

    # Wait async requests done
    if req_recv is not None:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)
    # if req_send is not None:
    #    req_send.wait()

    # if cp_world_size > 1:
    #     if os.environ.get("CP_DEBUG") == "INFO":
    #         print("Rank %d, barrier reached" % global_rank)
    #     tdi.barrier(group=group)

    input_ = input_.transpose(0, dim).contiguous()

    # print('out _pass_from_previous_rank', 'cp_rank:', cp_rank, 'out_size:', input_.shape)

    return input_


def _conv_gather(input_, dim, kernel_size):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()

    global_rank = tdi.get_rank()
    # print('in _conv_gather, global_rank:', global_rank, 'cp_rank:', cp_rank, 'input_size:', input_.shape)

    input_ = input_.contiguous()
    input_first_kernel = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    if cp_rank == 0:
        fake_input = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        fake_input = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim).contiguous()

    tensor_list = [torch.empty_like(torch.cat([input_first_kernel, fake_input], dim=dim))] + [
        torch.empty_like(fake_input) for _ in range(cp_world_size - 1)
    ]

    tensor_list[cp_rank] = input_
    tdi.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # print('out _conv_gather, global_rank:', global_rank, 'cp_rank:', cp_rank, 'input_size:', output.shape)

    return output


def _conv_split(input_, dim, kernel_size):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    global_rank = tdi.get_rank()
    cp_rank = get_context_parallel_rank()
    # print('in _conv_split, global_rank:', global_rank, 'cp_rank:', cp_rank, 'input_size:', input_.shape)

    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # output = input_.transpose(dim, 0)[cp_rank * dim_size + 1:(cp_rank + 1) * dim_size + kernel_size].transpose(dim, 0)
        output = input_.transpose(dim, 0)[cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size].transpose(
            dim, 0
        )
    output = output.contiguous()

    # print('out _conv_split, global_rank:', global_rank, 'cp_rank:', cp_rank, 'out_size:', output.shape)

    return output
