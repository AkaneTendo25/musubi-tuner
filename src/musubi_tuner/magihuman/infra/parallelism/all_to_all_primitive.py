def batch_scatter_head_gather_seqlen(tensors, cp_split_sizes, cp_group):
    del cp_split_sizes, cp_group
    return tensors


def scatter_seqlen_gather_head(tensor, cp_split_sizes, cp_group, async_op=False):
    del cp_split_sizes, cp_group, async_op
    return tensor
