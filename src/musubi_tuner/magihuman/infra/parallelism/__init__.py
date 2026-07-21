"""Sequence-parallel scheduling for a single-process run.

Ulysses sequence parallelism splits attention inputs across a context-parallel
group; with a group of one, dispatch and undispatch are identities.
"""


class _IdentityUlyssesScheduler:
    cp_split_sizes = None

    def dispatch(self, tensor):
        return tensor

    def undispatch(self, tensor):
        return tensor


_SCHEDULER = _IdentityUlyssesScheduler()


def ulysses_scheduler() -> _IdentityUlyssesScheduler:
    return _SCHEDULER
