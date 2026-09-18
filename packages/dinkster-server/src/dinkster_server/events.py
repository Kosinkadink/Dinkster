"""Event hub: fans engine + queue events out to WebSocket subscribers.

Publication never blocks the engine: `publish` is synchronous and appends to
per-subscriber deques. Slow subscribers lose *droppable* events (per-node
progress chatter - the next event supersedes it anyway); terminal events
(job state transitions, failures, run completion) are never dropped - when a
buffer is full, the oldest droppable event is evicted to make room, and if
nothing is droppable the buffer grows past its soft bound rather than lose a
terminal event silently.

Wire form is camelCase and typed by "type". Node progress is a state map,
not a single cursor: graph execution is parallel, so "the current node" is
not a coherent concept (DESIGN 3.5).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from collections.abc import Callable, Collection, Mapping

from dinkster_engine import EngineEvent

from .redaction import PathRedactor

# Per-node chatter: superseded by later events, safe to shed under pressure.
# node_event (progress/previews/pack events) is chatter by the reporting
# contract: nothing correctness-critical rides on it.
DROPPABLE_KINDS = frozenset(
    {
        "node_started",
        "node_cached",
        "cache_miss",
        "node_finished",
        "run_started",
        "node_event",
    }
)

DEFAULT_BUFFER = 256

BINARY_BLOB_KEY = "_blob"
"""Reserved wire-record key holding raw bytes (preview payloads). Records
carrying it must go out as a binary WebSocket frame (encode_binary_event),
never through JSON serialization."""


def encode_binary_event(event: Mapping[str, object]) -> bytes:
    """One self-describing binary frame: 4-byte big-endian JSON-header
    length, the JSON header (the event minus its blob), then the raw
    payload. Self-contained by design - no pairing with a preceding text
    frame for the client to lose under reconnects or interleaving."""
    header = {key: value for key, value in event.items() if key != BINARY_BLOB_KEY}
    blob = event[BINARY_BLOB_KEY]
    assert isinstance(blob, bytes)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(header_bytes).to_bytes(4, "big") + header_bytes + blob


def engine_event_to_wire(
    event: EngineEvent,
    *,
    client_id: str | None,
    job_id: str | None,
    redactor: PathRedactor | None = None,
) -> dict[str, object]:
    wire: dict[str, object] = {
        "type": event.kind,
        "runId": event.run_id,
    }
    if event.node_id is not None:
        wire["nodeId"] = event.node_id
    if event.kind == "node_event":
        # A node's typed report (engine detail: name/data/blob). Flatten to
        # first-class wire fields; the blob moves under BINARY_BLOB_KEY so
        # the WS sender ships it as a binary frame, never as JSON.
        detail = dict(event.detail)
        wire["event"] = detail.get("name", "")
        # Node-reported data is an arbitrary channel (report_event payloads,
        # captured log lines) and can carry resolved filesystem paths.
        data = detail.get("data", {})
        wire["data"] = redactor.redact_value(data) if redactor is not None else data
        worker = detail.get("worker")
        if isinstance(worker, str):
            wire["worker"] = worker
        execution_arm = detail.get("executionArm")
        if execution_arm in ("native", "comfyui"):
            wire["executionArm"] = execution_arm
        provider = detail.get("provider")
        if isinstance(provider, str):
            wire["provider"] = provider
        pack = detail.get("pack")
        if isinstance(pack, str):
            wire["pack"] = pack
        attention_diagnostic = detail.get("attentionDiagnostic")
        if isinstance(attention_diagnostic, str):
            wire["attentionDiagnostic"] = (
                redactor.redact_text(attention_diagnostic)
                if redactor is not None
                else attention_diagnostic
            )
        if detail.get("schemaVersion") == 1:
            wire["schemaVersion"] = 1
            wire["extensionSnapshotDigest"] = detail["extensionSnapshotDigest"]
        blob = detail.get("blob")
        if blob is not None:
            wire[BINARY_BLOB_KEY] = blob
    elif event.detail:
        # Lifecycle detail carries raiser-written text (node_failed message).
        detail_wire = dict(event.detail)
        wire["detail"] = redactor.redact_value(detail_wire) if redactor is not None else detail_wire
    if client_id is not None:
        wire["clientId"] = client_id
    if job_id is not None:
        wire["jobId"] = job_id
    return wire


class ProgressThrottle:
    """Coalesces per-node progress chatter at ingestion, before the hub,
    the replay buffer, and the execution journal see it (ComfyUI paces its
    progress bar the same way). A progress event is kept when it is the
    node's first, its first to reach the total, a phase-text or total
    change, or arrives at least ``min_interval`` seconds and ``min_delta``
    fraction after the last kept one. Everything in between is superseded
    chatter no consumer needs."""

    def __init__(
        self,
        *,
        min_interval: float = 0.1,
        min_delta: float = 0.005,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval = min_interval
        self._min_delta = min_delta
        self._clock = clock
        # run_id -> node_id -> (kept_at, fraction, text, total)
        self._kept: dict[str, dict[str, tuple[float, float, object, object]]] = {}

    def admit(self, run_id: str, node_id: str, data: Mapping[str, object]) -> bool:
        step = data.get("step")
        total = data.get("total")
        if (
            not isinstance(step, (int, float))
            or not isinstance(total, (int, float))
            or not math.isfinite(step)
            or not math.isfinite(total)
            or total <= 0
        ):
            return True  # malformed or indeterminate: not this policy's call
        fraction = min(max(step / total, 0.0), 1.0)
        text = data.get("text")
        nodes = self._kept.setdefault(run_id, {})
        last = nodes.get(node_id)
        now = self._clock()
        keep = (
            last is None
            or text != last[2]
            or total != last[3]
            or (fraction >= 1.0 and last[1] < 1.0)
            or (
                fraction < 1.0
                and now - last[0] >= self._min_interval
                and abs(fraction - last[1]) >= self._min_delta
            )
        )
        if keep:
            nodes[node_id] = (now, fraction, text, total)
        return keep

    def retain(self, run_ids: Collection[str]) -> None:
        """Drop throttle state for every run not in ``run_ids``."""
        for run_id in [r for r in self._kept if r not in run_ids]:
            del self._kept[run_id]


class Subscription:
    """One subscriber's bounded buffer. The hub pushes; the consumer gets."""

    def __init__(
        self,
        client_id: str | None,
        buffer: int,
        event_filter: Callable[[Mapping[str, object]], bool] | None = None,
    ) -> None:
        self.client_id = client_id
        self.event_filter = event_filter
        self._buffer = buffer
        self._events: deque[tuple[dict[str, object], bool]] = deque()
        self._ready = asyncio.Event()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, event: dict[str, object], droppable: bool) -> None:
        """Hub-internal: deliver one event into this subscriber's buffer."""
        if self._closed:
            return
        if len(self._events) >= self._buffer:
            # Evict the oldest queued droppable: the newest event is the one
            # that matters (the last progress update before a stream goes
            # quiet has no later superseder), staleness is what gets shed.
            for i, (_queued, queued_droppable) in enumerate(self._events):
                if queued_droppable:
                    del self._events[i]
                    break
            else:
                if droppable:
                    return  # only protected events queued; shed the chatter
                # exceed the soft bound: terminal events must not silently
                # disappear
        self._events.append((event, droppable))
        self._ready.set()

    async def get(self) -> dict[str, object] | None:
        """Next event, or None once closed and drained."""
        while True:
            if self._events:
                return self._events.popleft()[0]
            if self._closed:
                return None
            self._ready.clear()
            await self._ready.wait()

    def close(self) -> None:
        """Stop receiving; get() drains what is buffered, then returns None.
        Unsubscribing from the hub is the hub's job (EventHub.unsubscribe)."""
        self._closed = True
        self._ready.set()


class EventHub:
    def __init__(self, *, buffer: int = DEFAULT_BUFFER) -> None:
        self._buffer = buffer
        self._subscriptions: set[Subscription] = set()

    def subscribe(
        self,
        client_id: str | None = None,
        *,
        event_filter: Callable[[Mapping[str, object]], bool] | None = None,
    ) -> Subscription:
        """client_id filters to that client's jobs; None observes everything."""
        sub = Subscription(client_id, self._buffer, event_filter)
        self._subscriptions.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.close()
        self._subscriptions.discard(sub)

    def publish(
        self,
        event: Mapping[str, object],
        *,
        client_id: str | None,
        droppable: bool,
    ) -> None:
        """client_id routes the event to that client's subscribers (and to
        unfiltered observers); None broadcasts - instance-scoped telemetry
        (memory status) belongs to no client and reaches everyone."""
        wire = dict(event)
        for sub in tuple(self._subscriptions):
            if sub.closed:
                self._subscriptions.discard(sub)
                continue
            if sub.client_id is not None and client_id is not None and client_id != sub.client_id:
                continue
            if sub.event_filter is not None and not sub.event_filter(wire):
                continue
            sub.push(wire, droppable)

    def close(self) -> None:
        for sub in tuple(self._subscriptions):
            self.unsubscribe(sub)
