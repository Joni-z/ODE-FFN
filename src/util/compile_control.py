import functools

import torch


_COMPILE_ENABLED = True


def set_compile_enabled(enabled: bool) -> None:
    global _COMPILE_ENABLED
    _COMPILE_ENABLED = enabled


def maybe_compile(fn):
    compiled_fn = torch.compile(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if _COMPILE_ENABLED:
            return compiled_fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapped
