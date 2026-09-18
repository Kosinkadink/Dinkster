"""Block-loop prefetch for routed Aimdo weights."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, cast, runtime_checkable

import torch

__all__ = [
    "PrefetchPlan",
    "PrefetchQueue",
    "close_prefetch_queue",
    "cleanup_prefetch_queues",
    "make_prefetch_queue",
    "prefetch_queue_pop",
]


class _PrefetchHandle(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class _PrefetchMechanism(Protocol):
    def prefetch_enabled(self) -> bool: ...

    def prefetch(
        self, requests: Sequence[tuple[str, torch.dtype | None]]
    ) -> _PrefetchHandle | None: ...


_PrefetchRoute = Callable[[], tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None]


class _Target:
    __slots__ = ("module", "requests")

    def __init__(
        self,
        module: torch.nn.Module,
        requests: dict[_PrefetchMechanism, list[tuple[str, torch.dtype | None]]],
    ) -> None:
        self.module = module
        self.requests = requests


class _Prepared:
    __slots__ = ("target", "handles")

    def __init__(self, target: _Target, handles: list[_PrefetchHandle]) -> None:
        self.target = target
        self.handles = handles

    def close(self) -> None:
        first_error: BaseException | None = None
        for handle in self.handles:
            try:
                handle.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


_Entry = _Target | _Prepared | None
_queue_state = threading.local()


def _queues() -> list[PrefetchQueue]:
    queues = getattr(_queue_state, "queues", None)
    if queues is None:
        queues = []
        _queue_state.queues = queues
    return queues


class PrefetchQueue:
    def __init__(self, targets: Sequence[_Target]) -> None:
        self._entries: list[_Entry] = [None, *targets, None]
        self._closed = False

    def pop(self, module: torch.nn.Module | None) -> bool:
        if self._closed:
            return False
        consumed = self._entries.pop(0)
        if isinstance(consumed, _Prepared):
            consumed.close()

        next_entry = self._entries[0]
        expected = (
            next_entry.target.module
            if isinstance(next_entry, _Prepared)
            else None
            if next_entry is None
            else next_entry.module
        )
        if expected is not module:
            raise RuntimeError("prefetch queue block order mismatch")
        if isinstance(next_entry, _Target):
            handles: list[_PrefetchHandle] = []
            try:
                for mechanism, requests in next_entry.requests.items():
                    handle = mechanism.prefetch(requests)
                    if handle is not None:
                        handles.append(handle)
            except BaseException as error:
                for handle in reversed(handles):
                    try:
                        handle.close()
                    except BaseException as cleanup_error:
                        error.add_note(f"partial prefetch cleanup also failed: {cleanup_error!r}")
                raise
            self._entries[0] = _Prepared(next_entry, handles)
            fully_prepared = len(handles) == len(next_entry.requests)
        else:
            fully_prepared = False
        if module is None:
            self.close()
        return fully_prepared

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for entry in self._entries:
            if not isinstance(entry, _Prepared):
                continue
            try:
                entry.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._entries.clear()
        queues = _queues()
        if self in queues:
            queues.remove(self)
        if first_error is not None:
            raise first_error


def _target(block: torch.nn.Module) -> _Target:
    return _target_from_routes(block, _routes(block))


def _routes(block: torch.nn.Module) -> tuple[_PrefetchRoute, ...]:
    routes: list[_PrefetchRoute] = []
    for module in block.modules():
        residency_prefetch = getattr(module, "residency_prefetch", None)
        if callable(residency_prefetch):
            routes.append(cast("_PrefetchRoute", residency_prefetch))
    return tuple(routes)


def _bound_mechanisms(
    blocks: Sequence[tuple[torch.nn.Module, tuple[_PrefetchRoute, ...]]],
) -> tuple[_PrefetchMechanism, ...] | None:
    mechanisms: list[_PrefetchMechanism] = []
    for _block, routes in blocks:
        for route in routes:
            owner = getattr(route, "__self__", None)
            binding = getattr(owner, "_residency", None)
            mechanism = getattr(binding, "mechanism", None)
            if not isinstance(mechanism, _PrefetchMechanism):
                return None
            if not any(mechanism is existing for existing in mechanisms):
                mechanisms.append(mechanism)
    return tuple(mechanisms)


def _target_from_routes(block: torch.nn.Module, routes: Sequence[_PrefetchRoute]) -> _Target:
    grouped: dict[_PrefetchMechanism, list[tuple[str, torch.dtype | None]]] = {}
    for residency_prefetch in routes:
        route = residency_prefetch()
        if route is None:
            continue
        mechanism, requests = route
        if not isinstance(mechanism, _PrefetchMechanism):
            continue
        grouped.setdefault(mechanism, []).extend(requests)
    disabled: list[_PrefetchMechanism] | None = None
    for mechanism in grouped:
        if not mechanism.prefetch_enabled():
            if disabled is None:
                disabled = []
            disabled.append(mechanism)
    if disabled is not None:
        for mechanism in disabled:
            del grouped[mechanism]
    return _Target(block, grouped)


def _make_queue(targets: Sequence[_Target]) -> PrefetchQueue | None:
    if not targets or not any(target.requests for target in targets):
        return None
    queue = PrefetchQueue(targets)
    _queues().append(queue)
    return queue


class PrefetchPlan:
    """Static block traversal with dynamically evaluated residency routes."""

    def __init__(self, blocks: Iterable[torch.nn.Module]) -> None:
        self._blocks = tuple((block, _routes(block)) for block in blocks)
        # Enrollment bindings are immutable for the model lifetime; eviction
        # changes only the mechanism's dynamic prefetch capability.
        self._mechanisms: tuple[_PrefetchMechanism, ...] | None = None

    def make_queue(self) -> PrefetchQueue | None:
        if torch.compiler.is_compiling():
            return None
        mechanisms = self._mechanisms
        if mechanisms is None:
            mechanisms = _bound_mechanisms(self._blocks)
            if mechanisms is not None:
                self._mechanisms = mechanisms
        if mechanisms is not None and not any(
            mechanism.prefetch_enabled() for mechanism in mechanisms
        ):
            return None
        return _make_queue(
            tuple(_target_from_routes(block, routes) for block, routes in self._blocks)
        )


def make_prefetch_queue(
    blocks: Iterable[torch.nn.Module],
) -> PrefetchQueue | None:
    if torch.compiler.is_compiling():
        return None
    targets = tuple(_target(block) for block in blocks)
    return _make_queue(targets)


def prefetch_queue_pop(queue: PrefetchQueue | None, module: torch.nn.Module | None) -> bool:
    return queue is not None and queue.pop(module)


def close_prefetch_queue(queue: PrefetchQueue | None) -> None:
    if queue is None:
        return
    active_error = sys.exception()
    try:
        queue.close()
    except BaseException as cleanup_error:
        if active_error is None:
            raise
        active_error.add_note(f"prefetch cleanup also failed: {cleanup_error!r}")


def cleanup_prefetch_queues() -> None:
    queues = tuple(_queues())
    first_error: BaseException | None = None
    for queue in queues:
        try:
            queue.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
