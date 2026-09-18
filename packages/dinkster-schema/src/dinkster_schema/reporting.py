"""Node-facing reporting: events and persistent value diagnostics.

Nodes report through ambient module functions, never injected parameters:
execute() signatures stay plain (hazard H9 - node authors see values, not
machinery), and a valid call made outside an execution context is a silent
no-op, so unit tests can call execute() directly. The executing worker
installs a Reporter with use_reporter() for transient events;
where the event goes - same process, another venv, another machine - is the
worker's business, invisible from here.

Event names: bare names are the core vocabulary (``progress``, ``preview``),
emitted only through their typed helpers so their payload shape is a
contract, not a convention. Pack-defined events must be dot-namespaced
(``mypack.stage``) - the dot is what keeps a pack from squatting on a core
name the frontend already interprets.

Delivery semantics a node author may rely on:

- Emission never blocks on transport and never raises for transport
  reasons; a slow or absent consumer cannot fail a node.
- Events are chatter, not results: consumers may drop or coalesce them
  under pressure. Anything correctness-critical belongs in outputs.
- Ordering is preserved per invocation; events always reach the engine
  before the invocation's result does.

``report_value_diagnostic`` is not chatter: the worker captures these
records independently of the event observer and attaches them to successful
outputs' valueDiagnostics metadata for transport and cache replay.

Context propagation follows contextvars: the reporter is visible across
awaits and through ``asyncio.to_thread`` / tasks (which copy context), but
NOT inside a raw ``threading.Thread`` - hand off work with to_thread if a
background thread needs to report.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, cast

PROGRESS_EVENT = "progress"
PREVIEW_EVENT = "preview"
LOG_EVENT = "log"

LOG_LEVELS = ("info", "warning")
"""Log levels a node may report. There is deliberately no "error" level:
a node that failed raises, and the failure travels through the single
existing error report (NodeError -> node_failed) that drives error
surfaces - the log never carries a second copy of it."""

_LOG_RESERVED_KEYS = frozenset({"level", "message", "ts"})

_RESERVED_EVENTS = frozenset({PROGRESS_EVENT, PREVIEW_EVENT, LOG_EVENT})


class Reporter(Protocol):
    """Where emitted events go. Implementations must be safe to call from
    any thread and must never raise back into node code."""

    def __call__(self, name: str, data: Mapping[str, object], blob: bytes | None) -> None: ...


_reporter: ContextVar[Reporter | None] = ContextVar("dinkster_reporter", default=None)

_value_diagnostic_sink: ContextVar[Callable[[dict[str, object]], None] | None] = ContextVar(
    "dinkster_value_diagnostic_sink", default=None
)

_suppress_capture: ContextVar[bool] = ContextVar("dinkster_suppress_capture", default=False)


@contextmanager
def capture_value_diagnostics() -> Generator[list[dict[str, object]]]:
    """Worker-only invocation scope, closed even in copied task/thread contexts."""
    diagnostics: list[dict[str, object]] = []
    lock = threading.Lock()
    open_ = True

    def capture(diagnostic: dict[str, object]) -> None:
        with lock:
            if open_:
                diagnostics.append(diagnostic)

    token = _value_diagnostic_sink.set(capture)
    try:
        yield diagnostics
    finally:
        with lock:
            open_ = False
        _value_diagnostic_sink.reset(token)


def report_value_diagnostic(code: str, data: Mapping[str, object] | None = None) -> None:
    """Record a nonblocking fact about this invocation's output values.

    Codes and JSON-representable details are pack-defined. ``code`` and
    ``nodeId`` are reserved keys; the engine supplies the current node ID,
    including on cache hits. Details are snapshotted before returning.
    Invalid arguments raise ValueError; valid calls outside execution are
    silent no-ops, as are reports from work that outlives its invocation.
    """
    if not isinstance(cast("object", code), str) or not code.strip():
        raise ValueError("value diagnostic code must be a non-empty string")
    if data is not None and not isinstance(cast("object", data), Mapping):
        raise ValueError("value diagnostic data must be a mapping")
    payload: dict[str, object] = dict(data) if data is not None else {}
    reserved = {"code", "nodeId"} & payload.keys()
    if reserved:
        raise ValueError(f"value diagnostic data may not use reserved keys: {sorted(reserved)}")
    payload["code"] = code
    try:
        snapshot = cast("dict[str, object]", json.loads(json.dumps(payload, allow_nan=False)))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("value diagnostic data must be JSON-representable") from exc
    sink = _value_diagnostic_sink.get()
    if sink is not None:
        sink(snapshot)


@contextmanager
def suppress_capture() -> Generator[None]:
    """Mark the current context as reporting/delivery machinery: execution
    output capture (capture.py) ignores stream writes and log records made
    while this is active, so a listener that prints or logs can never feed
    its own output back into the log stream. ``_emit`` sets it around every
    delivery; the standard log handler sets it around its terminal write."""
    token = _suppress_capture.set(True)
    try:
        yield
    finally:
        _suppress_capture.reset(token)


def capture_suppressed() -> bool:
    """Whether the current context is inside ``suppress_capture()``."""
    return _suppress_capture.get()


@contextmanager
def use_reporter(reporter: Reporter) -> Generator[None]:
    """Install a reporter for the current context. Worker-side machinery
    only; node code never calls this."""
    token = _reporter.set(reporter)
    try:
        yield
    finally:
        _reporter.reset(token)


def _emit(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
    reporter = _reporter.get()
    if reporter is not None:
        with suppress_capture():
            reporter(name, data, blob)


def report_progress(step: int, total: int, *, text: str = "") -> None:
    """Report how far along this node is: ``step`` of ``total`` units done.

    Units are whatever the node counts - sampler steps, frames, files. An
    optional ``text`` labels the phase ("decoding", "upscaling tile 3/9").
    """
    payload: dict[str, object] = {"step": int(step), "total": int(total)}
    if text:
        payload["text"] = text
    _emit(PROGRESS_EVENT, payload, None)


def report_log(level: str, message: str, *, data: Mapping[str, object] | None = None) -> None:
    """Say something to the user about this node's execution: an ``info``
    note or a ``warning``, shown in the frontend's execution log attributed
    to this node.

    Not an error channel: a node that cannot continue raises, and the
    exception travels through the normal failure report. ``data`` carries
    small JSON-representable extras (it may not use the reserved keys
    ``level``/``message``/``ts``).
    """
    if level not in LOG_LEVELS:
        raise ValueError(
            f"unknown log level: {level!r} (expected one of {LOG_LEVELS}); "
            "a failing node should raise, not log an error"
        )
    payload: dict[str, object] = dict(data) if data else {}
    reserved = _LOG_RESERVED_KEYS & payload.keys()
    if reserved:
        raise ValueError(f"log data may not use reserved keys: {sorted(reserved)}")
    payload["level"] = level
    payload["message"] = str(message)
    payload["ts"] = time.time()
    _emit(LOG_EVENT, payload, None)


def report_preview(
    data: bytes,
    *,
    mime: str = "image/jpeg",
    width: int | None = None,
    height: int | None = None,
    stream: str | None = None,
    frame_index: int | None = None,
    frame_count: int | None = None,
    fps: float | None = None,
) -> None:
    """Ship an intermediate visual: encoded image bytes plus their MIME
    type. Encode small (a preview is a glance, not an output) - the bytes
    cross every boundary between the node and the frontend.

    ``stream`` names the latent stream a multi-stream family previews.
    Animated previews address each frame into a fixed ring: ``frame_index``
    is the frame's slot in [0, ``frame_count``) and ``fps`` the ring's
    display rate; a frame without them replaces the node's single image."""
    payload: dict[str, object] = {"mime": mime}
    if width is not None:
        payload["width"] = int(width)
    if height is not None:
        payload["height"] = int(height)
    if stream is not None:
        payload["stream"] = str(stream)
    if frame_index is not None:
        payload["frameIndex"] = int(frame_index)
    if frame_count is not None:
        payload["frameCount"] = int(frame_count)
    if fps is not None:
        payload["fps"] = float(fps)
    _emit(PREVIEW_EVENT, payload, bytes(data))


def report_event(
    name: str,
    data: Mapping[str, object] | None = None,
    *,
    blob: bytes | None = None,
) -> None:
    """Emit a pack-defined event. ``name`` must be dot-namespaced
    (``mypack.stage``); ``data`` must be JSON-representable - an event
    whose payload cannot cross a process boundary is dropped there, not
    delivered by luck when the worker happens to be in-process."""
    if name in _RESERVED_EVENTS or "." not in name:
        raise ValueError(
            f"custom event name {name!r} must be dot-namespaced "
            "(e.g. 'mypack.stage'); bare names are reserved for core events"
        )
    _emit(name, dict(data) if data else {}, blob)
