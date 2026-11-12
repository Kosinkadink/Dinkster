"""Operation-scoped observations of completed preparation work."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

Progress = Callable[[str, int, int], None]
_observer: ContextVar[Progress | None] = ContextVar("preparation_observer", default=None)


@contextmanager
def observe_preparation(observer: Progress) -> Generator[None]:
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def preparation_progress(stage: str, completed: int = 0, total: int = 0) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(stage, completed, total)
