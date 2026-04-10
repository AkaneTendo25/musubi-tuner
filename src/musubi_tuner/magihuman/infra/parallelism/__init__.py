class _IdentityUlyssesScheduler:
    cp_split_sizes = None

    def dispatch(self, tensor):
        return tensor

    def undispatch(self, tensor):
        return tensor


_SCHEDULER = _IdentityUlyssesScheduler()


def ulysses_scheduler() -> _IdentityUlyssesScheduler:
    return _SCHEDULER
