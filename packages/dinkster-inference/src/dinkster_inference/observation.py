"""Invocation-local stage spans for timing and memory instrumentation."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

ExecutionStage = Literal["load", "condition", "sample", "encode", "decode"]
ExecutionSpanPhase = Literal["begin", "end"]


@dataclass(frozen=True, slots=True)
class ExecutionSpanEvent:
    invocation_id: str
    span_id: int
    parent_span_id: int | None
    phase: ExecutionSpanPhase
    stage: ExecutionStage
    operation: str
    monotonic_ns: int
    component_role: str | None = None
    device: str | None = None
    storage_dtype: str | None = None
    compute_dtype: str | None = None
    resident_bytes: int | None = None


ExecutionObserver = Callable[[ExecutionSpanEvent], None]


@dataclass(frozen=True, slots=True)
class ExecutionSpan:
    invocation_id: str
    span_id: int
    parent_span_id: int | None
    stage: ExecutionStage
    operation: str
    component_role: str | None
    device: str | None
    storage_dtype: str | None
    compute_dtype: str | None
    resident_bytes: int | None


class ExecutionObserverAttachment:
    """Concurrency-safe span authority for one observed invocation."""

    def __init__(self, observer: ExecutionObserver, invocation_id: str | None = None) -> None:
        if not callable(observer):
            raise TypeError("execution observer must be callable")
        self.invocation_id = invocation_id or uuid.uuid4().hex
        if type(self.invocation_id) is not str or not self.invocation_id:
            raise ValueError("execution observer invocation_id must be a nonempty string")
        self._observer = observer
        self._lock = threading.Lock()
        self._next_span_id = 1

    def begin(
        self,
        stage: ExecutionStage,
        operation: str,
        *,
        parent_span_id: int | None = None,
        component_role: str | None = None,
        device: str | None = None,
        storage_dtype: str | None = None,
        compute_dtype: str | None = None,
        resident_bytes: int | None = None,
    ) -> ExecutionSpan:
        if type(operation) is not str or not operation:
            raise ValueError("execution span operation must be a nonempty string")
        with self._lock:
            span_id = self._next_span_id
            self._next_span_id += 1
        span = ExecutionSpan(
            self.invocation_id,
            span_id,
            parent_span_id,
            stage,
            operation,
            component_role,
            device,
            storage_dtype,
            compute_dtype,
            resident_bytes,
        )
        self._emit(span, "begin")
        return span

    def end(self, span: ExecutionSpan) -> None:
        if span.invocation_id != self.invocation_id:
            raise ValueError("execution span belongs to a different observer attachment")
        self._emit(span, "end")

    def _emit(self, span: ExecutionSpan, phase: ExecutionSpanPhase) -> None:
        try:
            self._observer(
                ExecutionSpanEvent(
                    invocation_id=span.invocation_id,
                    span_id=span.span_id,
                    parent_span_id=span.parent_span_id,
                    phase=phase,
                    stage=span.stage,
                    operation=span.operation,
                    monotonic_ns=time.perf_counter_ns(),
                    component_role=span.component_role,
                    device=span.device,
                    storage_dtype=span.storage_dtype,
                    compute_dtype=span.compute_dtype,
                    resident_bytes=span.resident_bytes,
                )
            )
        except Exception:
            pass


@contextmanager
def execution_span(
    attachment: ExecutionObserverAttachment | None,
    stage: ExecutionStage,
    operation: str,
    *,
    parent_span_id: int | None = None,
    component_role: str | None = None,
    device: str | None = None,
) -> Generator[ExecutionSpan | None, None, None]:
    if attachment is None:
        yield None
        return
    span = attachment.begin(
        stage,
        operation,
        parent_span_id=parent_span_id,
        component_role=component_role,
        device=device,
    )
    try:
        yield span
    finally:
        attachment.end(span)


__all__ = [
    "ExecutionObserver",
    "ExecutionObserverAttachment",
    "ExecutionSpan",
    "ExecutionSpanEvent",
    "ExecutionSpanPhase",
    "ExecutionStage",
    "execution_span",
]
