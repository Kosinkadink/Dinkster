"""Invocation-local stage spans for timing and memory instrumentation."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, cast

ExecutionStage = Literal["load", "condition", "sample", "encode", "decode"]
ExecutionSpanPhase = Literal["begin", "end", "snapshot", "decision"]
MemoryBoundary = Literal[
    "post-load",
    "first-sampling-seam",
    "stage-end",
    "peak-residency",
    "post-offload",
    "release",
]
MemoryPageClass = Literal[
    "weights",
    "activation-runtime-workspace",
    "execution-result-cache",
    "other-reclaimable",
    "unknown",
]
MemoryDecisionAction = Literal["place", "retain", "offload", "evict"]
MemoryCompilerState = Literal["active", "disabled", "unavailable"]

MEMORY_PAGE_CLASSES: tuple[MemoryPageClass, ...] = (
    "weights",
    "activation-runtime-workspace",
    "execution-result-cache",
    "other-reclaimable",
    "unknown",
)
_MEMORY_BOUNDARIES: tuple[MemoryBoundary, ...] = (
    "post-load",
    "first-sampling-seam",
    "stage-end",
    "peak-residency",
    "post-offload",
    "release",
)
_MEMORY_COMPILER_STATES: tuple[MemoryCompilerState, ...] = (
    "active",
    "disabled",
    "unavailable",
)
_MEMORY_DECISION_ACTIONS: tuple[MemoryDecisionAction, ...] = (
    "place",
    "retain",
    "offload",
    "evict",
)


def _validate_bytes(label: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class ComponentMemorySnapshot:
    """One component's logical placement and physical device-page ownership."""

    component_id: str
    component_role: str
    storage_id: str
    device: str
    total_bytes: int
    loaded_bytes: int
    offloaded_bytes: int
    resident_bytes: int
    bytes_by_page_class: Mapping[MemoryPageClass, int]

    def __post_init__(self) -> None:
        for label in ("component_id", "component_role", "storage_id", "device"):
            if not getattr(self, label):
                raise ValueError(f"memory component {label} must be nonempty")
        for label in ("total_bytes", "loaded_bytes", "offloaded_bytes", "resident_bytes"):
            _validate_bytes(f"memory component {label}", cast(int, getattr(self, label)))
        if self.loaded_bytes + self.offloaded_bytes != self.total_bytes:
            raise ValueError("memory component loaded and offloaded bytes must equal total bytes")
        if self.resident_bytes < self.loaded_bytes:
            raise ValueError("memory component resident bytes cannot be less than loaded bytes")
        classes = dict(self.bytes_by_page_class)
        if set(classes) != set(MEMORY_PAGE_CLASSES):
            raise ValueError("memory component must report every page class exactly once")
        for page_class, nbytes in classes.items():
            _validate_bytes(f"memory page class {page_class}", nbytes)
        if sum(classes.values()) != self.resident_bytes:
            raise ValueError("memory component page classes must equal resident bytes")
        object.__setattr__(self, "bytes_by_page_class", classes)


@dataclass(frozen=True, slots=True)
class DeviceMemorySnapshot:
    """Measured device usage reconciled against uniquely owned component pages."""

    device: str
    measured_bytes: int
    reconciliation_bound_bytes: int
    unknown_bytes: int

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("memory device must be nonempty")
        for label in ("measured_bytes", "reconciliation_bound_bytes", "unknown_bytes"):
            _validate_bytes(f"memory device {label}", cast(int, getattr(self, label)))


@dataclass(frozen=True, slots=True)
class ExecutionMemorySnapshot:
    """A stable execution boundary with complete component and device accounting."""

    boundary: MemoryBoundary
    components: tuple[ComponentMemorySnapshot, ...]
    devices: tuple[DeviceMemorySnapshot, ...]
    memory_compiler: MemoryCompilerState

    def __post_init__(self) -> None:
        if self.boundary not in _MEMORY_BOUNDARIES:
            raise ValueError(f"unknown memory snapshot boundary {self.boundary!r}")
        if self.memory_compiler not in _MEMORY_COMPILER_STATES:
            raise ValueError(f"unknown memory compiler state {self.memory_compiler!r}")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("memory snapshot component identities must be distinct")
        device_ids = [device.device for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("memory snapshot device identities must be distinct")
        component_devices = {component.device for component in self.components}
        if component_devices != set(device_ids):
            raise ValueError("memory snapshot must account for every component device exactly once")
        for device in self.devices:
            owned: dict[str, ComponentMemorySnapshot] = {}
            for component in self.components:
                if component.device != device.device:
                    continue
                previous = owned.setdefault(component.storage_id, component)
                if (
                    previous.total_bytes,
                    previous.loaded_bytes,
                    previous.offloaded_bytes,
                ) != (
                    component.total_bytes,
                    component.loaded_bytes,
                    component.offloaded_bytes,
                ):
                    raise ValueError("shared storage must report identical logical placement")
                if previous.resident_bytes != component.resident_bytes:
                    raise ValueError("shared storage must report identical resident bytes")
                if previous.bytes_by_page_class != component.bytes_by_page_class:
                    raise ValueError("shared storage must report identical page classes")
            classified = sum(component.resident_bytes for component in owned.values())
            reconciled = classified + device.unknown_bytes
            if abs(device.measured_bytes - reconciled) > device.reconciliation_bound_bytes:
                raise ValueError("memory device totals exceed the reconciliation bound")


@dataclass(frozen=True, slots=True)
class ExecutionMemoryDecision:
    """One observed placement-policy or memory-compiler action."""

    component_id: str
    component_role: str
    device: str
    source: Literal["residency-policy", "memory-compiler"]
    action: MemoryDecisionAction
    byte_count: int
    reason: str

    def __post_init__(self) -> None:
        for label in ("component_id", "component_role", "device", "reason"):
            if not getattr(self, label):
                raise ValueError(f"memory decision {label} must be nonempty")
        if self.source not in ("residency-policy", "memory-compiler"):
            raise ValueError(f"unknown memory decision source {self.source!r}")
        if self.action not in _MEMORY_DECISION_ACTIONS:
            raise ValueError(f"unknown memory decision action {self.action!r}")
        _validate_bytes("memory decision byte_count", self.byte_count)


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
    memory_snapshot: ExecutionMemorySnapshot | None = None
    memory_decision: ExecutionMemoryDecision | None = None

    def __post_init__(self) -> None:
        if self.phase == "snapshot":
            if self.memory_snapshot is None or self.memory_decision is not None:
                raise ValueError("memory snapshot events require only a snapshot")
        elif self.phase == "decision":
            if self.memory_decision is None or self.memory_snapshot is not None:
                raise ValueError("memory decision events require only a decision")
        elif self.memory_snapshot is not None or self.memory_decision is not None:
            raise ValueError("span begin/end events cannot carry memory records")


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
        self._markers: set[str] = set()

    def _next_id(self) -> int:
        with self._lock:
            span_id = self._next_span_id
            self._next_span_id += 1
        return span_id

    def claim_once(self, marker: str) -> bool:
        if not marker:
            raise ValueError("execution observer marker must be nonempty")
        with self._lock:
            if marker in self._markers:
                return False
            self._markers.add(marker)
            return True

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
        span_id = self._next_id()
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

    def record_memory_snapshot(
        self,
        stage: ExecutionStage,
        operation: str,
        snapshot: ExecutionMemorySnapshot,
        *,
        parent_span_id: int | None = None,
    ) -> None:
        self._emit_memory(
            stage,
            operation,
            "snapshot",
            parent_span_id=parent_span_id,
            snapshot=snapshot,
        )

    def record_memory_decision(
        self,
        stage: ExecutionStage,
        operation: str,
        decision: ExecutionMemoryDecision,
        *,
        parent_span_id: int | None = None,
    ) -> None:
        self._emit_memory(
            stage,
            operation,
            "decision",
            parent_span_id=parent_span_id,
            decision=decision,
        )

    def _emit_memory(
        self,
        stage: ExecutionStage,
        operation: str,
        phase: Literal["snapshot", "decision"],
        *,
        parent_span_id: int | None,
        snapshot: ExecutionMemorySnapshot | None = None,
        decision: ExecutionMemoryDecision | None = None,
    ) -> None:
        if type(operation) is not str or not operation:
            raise ValueError("execution memory operation must be a nonempty string")
        event = ExecutionSpanEvent(
            invocation_id=self.invocation_id,
            span_id=self._next_id(),
            parent_span_id=parent_span_id,
            phase=phase,
            stage=stage,
            operation=operation,
            monotonic_ns=time.perf_counter_ns(),
            memory_snapshot=snapshot,
            memory_decision=decision,
        )
        try:
            self._observer(event)
        except Exception:
            pass

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


def _snapshot_list() -> list[ExecutionMemorySnapshot]:
    return []


def _decision_list() -> list[ExecutionMemoryDecision]:
    return []


def _error_list() -> list[str]:
    return []


@dataclass(slots=True)
class ExecutionMemoryReceipt:
    """Collect memory events into a deterministic machine-readable receipt."""

    expected_components: frozenset[str] = frozenset()
    _invocation_id: str | None = field(default=None, init=False)
    _snapshots: list[ExecutionMemorySnapshot] = field(default_factory=_snapshot_list, init=False)
    _decisions: list[ExecutionMemoryDecision] = field(default_factory=_decision_list, init=False)
    _errors: list[str] = field(default_factory=_error_list, init=False)

    def observe(self, event: ExecutionSpanEvent) -> None:
        if self._invocation_id is None:
            self._invocation_id = event.invocation_id
        elif event.invocation_id != self._invocation_id:
            self._errors.append("execution memory receipt cannot mix invocations")
            return
        if event.memory_snapshot is not None:
            present = {component.component_id for component in event.memory_snapshot.components}
            missing = self.expected_components - present
            if missing:
                self._errors.append(
                    "memory snapshot is missing components: " + ", ".join(sorted(missing))
                )
                return
            self._snapshots.append(event.memory_snapshot)
        if event.memory_decision is not None:
            self._decisions.append(event.memory_decision)

    def to_dict(self) -> dict[str, object]:
        if self._errors:
            raise ValueError("; ".join(self._errors))
        if self.expected_components and not self._snapshots:
            raise ValueError("execution memory receipt has no snapshots")
        peaks: dict[str, int] = {}
        for snapshot in self._snapshots:
            for device in snapshot.devices:
                peaks[device.device] = max(peaks.get(device.device, 0), device.measured_bytes)
        return {
            "schema": "dinkster.execution-memory.v1",
            "invocationId": self._invocation_id,
            "snapshots": [_memory_snapshot_dict(snapshot) for snapshot in self._snapshots],
            "decisions": [_memory_decision_dict(decision) for decision in self._decisions],
            "peakResidentBytesByDevice": dict(sorted(peaks.items())),
        }


def _memory_snapshot_dict(snapshot: ExecutionMemorySnapshot) -> dict[str, object]:
    return {
        "boundary": snapshot.boundary,
        "memoryCompiler": snapshot.memory_compiler,
        "components": [
            {
                "componentId": component.component_id,
                "componentRole": component.component_role,
                "storageId": component.storage_id,
                "device": component.device,
                "totalBytes": component.total_bytes,
                "loadedBytes": component.loaded_bytes,
                "offloadedBytes": component.offloaded_bytes,
                "residentBytes": component.resident_bytes,
                "bytesByPageClass": {
                    page_class: component.bytes_by_page_class[page_class]
                    for page_class in MEMORY_PAGE_CLASSES
                },
            }
            for component in snapshot.components
        ],
        "devices": [
            {
                "device": device.device,
                "measuredBytes": device.measured_bytes,
                "reconciliationBoundBytes": device.reconciliation_bound_bytes,
                "unknownBytes": device.unknown_bytes,
            }
            for device in snapshot.devices
        ],
    }


def _memory_decision_dict(decision: ExecutionMemoryDecision) -> dict[str, object]:
    return {
        "componentId": decision.component_id,
        "componentRole": decision.component_role,
        "device": decision.device,
        "source": decision.source,
        "action": decision.action,
        "byteCount": decision.byte_count,
        "reason": decision.reason,
    }


__all__ = [
    "ComponentMemorySnapshot",
    "DeviceMemorySnapshot",
    "ExecutionMemoryDecision",
    "ExecutionMemoryReceipt",
    "ExecutionMemorySnapshot",
    "ExecutionObserver",
    "ExecutionObserverAttachment",
    "ExecutionSpan",
    "ExecutionSpanEvent",
    "ExecutionSpanPhase",
    "ExecutionStage",
    "MEMORY_PAGE_CLASSES",
    "MemoryBoundary",
    "MemoryCompilerState",
    "MemoryDecisionAction",
    "MemoryPageClass",
    "execution_span",
]
