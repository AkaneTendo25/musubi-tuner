"""Single-process topology accessors.

This package targets single-GPU training, so every parallelism axis reports a
world size of one and rank zero. The functions exist so that vendored model code
written against a distributed runtime imports and runs unchanged.
"""


def initialize_infra() -> None:
    return None


def get_tp_rank() -> int:
    return 0


def get_cp_rank() -> int:
    return 0


def get_pp_rank() -> int:
    return 0


def get_cp_group():
    return None


def get_cp_world_size() -> int:
    return 1
