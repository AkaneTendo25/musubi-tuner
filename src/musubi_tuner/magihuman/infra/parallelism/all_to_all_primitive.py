"""Context-parallel collectives for a single-process context parallel group.

With a context-parallel world size of one there is nothing to exchange, so both
primitives return their input unchanged and keep the vendored attention code's
call sites intact.
"""


def batch_scatter_head_gather_seqlen(tensors, cp_split_sizes, cp_group):
    del cp_split_sizes, cp_group
    return tensors


def scatter_seqlen_gather_head(tensor, cp_split_sizes, cp_group, async_op=False):
    del cp_split_sizes, cp_group, async_op
    return tensor
