"""Timing receipts for partial-residency forwards.

A leased forward on an offloaded unit does three kinds of work: it
transfers stored weights to the load device, optionally decodes or
casts them, and runs the consuming operation. This module measures
those phases so a run can prove whether transfer and decode work
overlapped compute or stalled it.

Collection is explicit and thread-local: a weight lease (eager or
aimdo-managed) captures the innermost collector opened by
:func:`collect_partial_residency_timing` on the same thread when the
lease opens (one thread-local read; with no active collector the
instrumented paths run their uncollected form unchanged). Leased
consumers record their own phases into the same collector via
``WeightLease.timing_collector``, so module receipts stay coherent
with lease transfers.

On CUDA load devices, phases are bracketed with CUDA events recorded
on the measured device's current stream and resolved to GPU-timeline
durations at :meth:`PartialResidencyTiming.report`; the exposed-stall
phase brackets the consumer stream's wait on the producer (transfer)
stream, so transfer time hidden under compute never appears in it.
On other devices phases are wall-clock and transfers are fully
exposed, so exposed stall equals transfer time.

Reports are measurement evidence only: nothing here feeds runtime
identity or route facts.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

__all__ = [
    "PartialResidencyTiming",
    "PartialResidencyTimingReport",
    "collect_partial_residency_timing",
]

TRANSFER = "transfer"
EXPOSED_STALL = "exposed_stall"
DEQUANT = "dequant"
COMPUTE = "compute"

_PHASES = (TRANSFER, EXPOSED_STALL, DEQUANT, COMPUTE)


@dataclass(frozen=True, slots=True)
class PartialResidencyTimingReport:
    """Resolved phase durations for one collection window.

    ``transfer_ms`` is total producer-stream copy time whether or not
    it was hidden under compute; ``exposed_stall_ms`` is the part the
    consumer stream actually waited. ``exposed_stall_ms`` much smaller
    than ``transfer_ms`` proves the copies overlapped compute.

    ``transfer_bytes`` counts every stored byte copied to the load
    device; ``prefetch_bytes`` is the subset copied ahead of the
    consuming lease by mechanism prefetch, and ``prefetched_transfers``
    counts those moves (``leased_transfers`` counts moves the lease
    itself had to start). A consuming lease that finds its value
    prefetched records only its wait as exposed stall, never a second
    transfer.
    """

    transfer_ms: float
    exposed_stall_ms: float
    dequant_ms: float
    compute_ms: float
    transfer_bytes: int
    leased_transfers: int
    leased_forwards: int
    prefetched_transfers: int
    prefetch_bytes: int


class PartialResidencyTiming:
    """Accumulates phase samples; resolve with :meth:`report`.

    CUDA samples hold event pairs whose ``elapsed_time`` is only valid
    after both events complete; ``report()`` synchronizes each end
    event once, so call it after the measured work, not inside it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cuda_samples: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            phase: [] for phase in _PHASES
        }
        self._wall_ns: dict[str, int] = {phase: 0 for phase in _PHASES}
        self._transfer_bytes = 0
        self._leased_transfers = 0
        self._leased_forwards = 0
        self._prefetched_transfers = 0
        self._prefetch_bytes = 0

    def add_cuda_sample(self, phase: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        with self._lock:
            self._cuda_samples[phase].append((start, end))

    def add_wall_ns(self, phase: str, elapsed_ns: int) -> None:
        with self._lock:
            self._wall_ns[phase] += elapsed_ns

    def count_transfer(self, nbytes: int) -> None:
        with self._lock:
            self._transfer_bytes += nbytes
            self._leased_transfers += 1

    def count_forward(self) -> None:
        with self._lock:
            self._leased_forwards += 1

    def count_prefetch(self, nbytes: int) -> None:
        with self._lock:
            self._transfer_bytes += nbytes
            self._prefetch_bytes += nbytes
            self._prefetched_transfers += 1

    def report(self) -> PartialResidencyTimingReport:
        with self._lock:
            totals: dict[str, float] = {}
            for phase in _PHASES:
                milliseconds = self._wall_ns[phase] / 1e6
                for start, end in self._cuda_samples[phase]:
                    end.synchronize()
                    milliseconds += start.elapsed_time(end)
                totals[phase] = milliseconds
            return PartialResidencyTimingReport(
                transfer_ms=totals[TRANSFER],
                exposed_stall_ms=totals[EXPOSED_STALL],
                dequant_ms=totals[DEQUANT],
                compute_ms=totals[COMPUTE],
                transfer_bytes=self._transfer_bytes,
                leased_transfers=self._leased_transfers,
                leased_forwards=self._leased_forwards,
                prefetched_transfers=self._prefetched_transfers,
                prefetch_bytes=self._prefetch_bytes,
            )


_active = threading.local()


def active_partial_residency_timing() -> PartialResidencyTiming | None:
    """The innermost collector opened on this thread, if any."""
    stack: list[PartialResidencyTiming] | None = getattr(_active, "stack", None)
    if not stack:
        return None
    return stack[-1]


@contextmanager
def collect_partial_residency_timing() -> Generator[PartialResidencyTiming]:
    """Collect partial-residency phase timings on this thread."""
    stack: list[PartialResidencyTiming] | None = getattr(_active, "stack", None)
    if stack is None:
        stack = []
        _active.stack = stack
    collector = PartialResidencyTiming()
    stack.append(collector)
    try:
        yield collector
    finally:
        stack.pop()


@contextmanager
def timed_phase(
    collector: PartialResidencyTiming | None, phase: str, device: torch.device
) -> Generator[None]:
    """Bracket one phase on ``device``'s current stream or wall clock.

    Events are recorded explicitly on ``device``'s current stream, so
    the measurement stays on the measured GPU even when a different
    CUDA device is the process's current one.
    """
    if collector is None:
        yield
        return
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            collector.add_cuda_sample(phase, start, end)
        return
    began = time.perf_counter_ns()
    try:
        yield
    finally:
        collector.add_wall_ns(phase, time.perf_counter_ns() - began)
