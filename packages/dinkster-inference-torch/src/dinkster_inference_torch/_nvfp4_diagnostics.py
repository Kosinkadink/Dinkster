"""Private, bounded and data-free NVFP4 runtime diagnostics."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

_EVENTS = frozenset(
    {
        "quantize_success",
        "quantize_error",
        "scaled_mm_success",
        "scaled_mm_error",
        "dequantize_success",
        "dequantize_error",
        "requantize_success",
        "requantize_error",
        "route_native",
        "route_full_precision",
        "route_non_cuda",
        "route_hip",
        "route_rank",
        "route_pre_sm10",
        "route_no_quantize_backend",
        "route_no_scaled_mm_backend",
        "route_backend_fallback",
        "route_deferred_patch",
        "observed_loaded",
        "observed_offloaded",
        "observed_state_change",
    }
)


@dataclass(frozen=True)
class Nvfp4InvocationSnapshot:
    counters: Mapping[str, int]
    terminal: str


@dataclass(frozen=True)
class Nvfp4RuntimeSnapshot:
    lifetime: Mapping[str, int]
    completed: tuple[Nvfp4InvocationSnapshot, ...]
    active: int
    unscoped: Mapping[str, int]


class Nvfp4DiagnosticsRecorder:
    def __init__(self) -> None:
        self._lock = RLock()
        self._lifetime: Counter[str] = Counter()
        self._unscoped: Counter[str] = Counter()
        self._completed: deque[Nvfp4InvocationSnapshot] = deque(maxlen=16)
        self._active = 0
        self._observed_state: dict[int, str] = {}
        self._scope: ContextVar[Counter[str] | None] = ContextVar(
            f"nvfp4_scope_{id(self)}", default=None
        )

    def record(self, event: str) -> None:
        if event not in _EVENTS:
            raise ValueError("unknown NVFP4 diagnostic event")
        scope = self._scope.get()
        with self._lock:
            self._lifetime[event] += 1
            (self._unscoped if scope is None else scope)[event] += 1

    def observe_state(self, owner: object, state: str) -> None:
        if state not in {"loaded", "offloaded"}:
            raise ValueError("unknown NVFP4 observed residency state")
        scope = self._scope.get()
        event = "observed_" + state
        with self._lock:
            previous = self._observed_state.get(id(owner))
            self._observed_state[id(owner)] = state
            self._lifetime[event] += 1
            target = self._unscoped if scope is None else scope
            target[event] += 1
            if previous is not None and previous != state:
                self._lifetime["observed_state_change"] += 1
                target["observed_state_change"] += 1

    @contextmanager
    def invocation(self) -> Generator[None]:
        current = self._scope.get()
        if current is not None:
            yield
            return
        counters: Counter[str] = Counter()
        token: Token[Counter[str] | None] = self._scope.set(counters)
        with self._lock:
            self._active += 1
        terminal = "success"
        try:
            yield
        except BaseException:
            terminal = "error"
            raise
        finally:
            with self._lock:
                self._active -= 1
                self._completed.append(
                    Nvfp4InvocationSnapshot(MappingProxyType(dict(counters)), terminal)
                )
            self._scope.reset(token)

    def snapshot(self) -> Nvfp4RuntimeSnapshot:
        with self._lock:
            return Nvfp4RuntimeSnapshot(
                MappingProxyType(dict(self._lifetime)),
                tuple(self._completed),
                self._active,
                MappingProxyType(dict(self._unscoped)),
            )


def nvfp4_runtime_status(runtime: object) -> Nvfp4RuntimeSnapshot:
    """Return package-internal status for a Flux runtime or assembly."""
    assembled = getattr(runtime, "assembled", runtime)
    diffusion = getattr(assembled, "diffusion", None)
    recorder = getattr(diffusion, "_nvfp4_diagnostics", None)
    if not isinstance(recorder, Nvfp4DiagnosticsRecorder):
        raise TypeError("object has no NVFP4 runtime diagnostics")
    return recorder.snapshot()
