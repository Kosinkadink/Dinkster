"""Execution output capture: stdout/stderr and python logging produced
while a node executes become attributed execution log events.

ComfyUI users watch the server terminal to see what a workflow is doing;
Dinkster attributes that output instead. While an invocation is in flight (an
ambient Reporter from ``use_reporter`` plus a ``CaptureBudget`` from
``use_capture_budget`` are set), text a node writes to stdout/stderr and
python logging records are forwarded through ``report_log`` - stdout as
``info``, stderr as ``warning``, logging by its own level, with
ERROR/CRITICAL forwarded as ``warning`` because a log record is chatter
from a node that kept running, never a node failure: failures raise and
travel through the one existing error report (NodeError -> node_failed).

The logging contract is scoped by who handles the record: records the
configured ``dinkster`` handler emits (core and ``pack_logger`` loggers)
forward at INFO+ with their origin logger named; loggers outside that tree
with no handler of their own reach the log at WARNING+ through
``logging.lastResort``'s stderr write; a foreign logger with its own
handlers is captured only as whatever text those handlers write to
stdout/stderr.

Capture adds forwarding, it never swallows: every captured write still
reaches the original stream, and logging handlers keep printing to the
terminal. Attribution rides the same contextvars as reporting itself, so
concurrent invocations (each in its own context) capture independently and
code outside any invocation is untouched.

Two global installations cooperate here:

- ``install_stream_capture()`` (idempotent, worker startup) wraps
  ``sys.stdout``/``sys.stderr`` in pass-through proxies. Because
  ``logging.lastResort`` resolves ``sys.stderr`` dynamically, WARNING+
  records from loggers with no configured handler are captured through the
  stderr proxy with no root handler installed - installing one would
  disable lastResort and silently swallow terminal output.
- the standard handler ``configure_logging`` installs calls
  ``forward_log_record`` after writing the terminal line inside
  ``suppress_capture()``, so the stderr proxy never re-captures the
  formatted line the handler just wrote (one record, one log event).

The ``suppress_capture`` guard (owned by reporting.py, set around every
report delivery) also stops feedback loops: anything a delivery listener
prints or logs while an event is in flight passes through to the terminal
only, never back into the log stream.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, TextIO, cast

from .reporting import capture_suppressed, report_log, suppress_capture

MAX_CAPTURED_RECORDS = 200
"""Per-invocation cap on forwarded records; the cap being hit is announced
by one final truncation record, then further output goes terminal-only."""

MAX_CAPTURED_MESSAGE_CHARS = 2000
"""Per-record message cap; longer messages are cut with a marker."""

_TRUNCATION_MARKER = " ... [truncated]"

_budget: ContextVar[CaptureBudget | None] = ContextVar("dinkster_capture_budget", default=None)


class CaptureBudget:
    """One invocation's capture allowance and line-assembly state.

    Thread-safe: a node may write from the loop thread and worker threads
    (asyncio.to_thread copies the context, so both see this budget).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records = 0
        self._notice_sent = False
        self._partial: dict[str, str] = {}

    def take(self) -> Literal["ok", "notify", "spent"]:
        """Reserve one record: ``ok`` to forward it, ``notify`` exactly once
        when the budget just ran out (the caller emits the truncation
        notice instead), ``spent`` afterwards."""
        with self._lock:
            if self._records < MAX_CAPTURED_RECORDS:
                self._records += 1
                return "ok"
            if not self._notice_sent:
                self._notice_sent = True
                return "notify"
            return "spent"

    def split_lines(self, stream: str, text: str) -> list[str]:
        """Fold ``text`` into the stream's pending partial line and return
        the complete lines it produced. An overlong partial (no newline in
        sight) is cut into a line of its own so it cannot grow unboundedly."""
        with self._lock:
            pending = self._partial.get(stream, "") + text
            lines = pending.split("\n")
            partial = lines.pop()
            if len(partial) > MAX_CAPTURED_MESSAGE_CHARS:
                lines.append(partial)
                partial = ""
            self._partial[stream] = partial
            return lines

    def drain_partials(self) -> list[tuple[str, str]]:
        """The unterminated tail of each stream, taken exactly once (called
        when the invocation ends, so a final print without a newline still
        becomes a record)."""
        with self._lock:
            drained = [(stream, line) for stream, line in self._partial.items() if line]
            self._partial.clear()
            return drained


@contextmanager
def use_capture_budget() -> Generator[None]:
    """Scope one invocation's capture. Worker-side machinery only, installed
    alongside ``use_reporter``; on exit, unterminated stream tails are
    flushed as records (the reporter is still open - this context must exit
    before the reporter does)."""
    budget = CaptureBudget()
    token = _budget.set(budget)
    try:
        yield
    finally:
        _budget.reset(token)
        try:
            for stream, line in budget.drain_partials():
                if line.strip():
                    _forward(budget, _STREAM_LEVELS[stream], line, {"origin": stream})
        except Exception:
            # Flushing tails is chatter: it must never fail the invocation
            # that is ending.
            pass


def _forward(budget: CaptureBudget, level: str, message: str, data: Mapping[str, object]) -> None:
    verdict = budget.take()
    if verdict == "spent":
        return
    with suppress_capture():
        if verdict == "notify":
            report_log(
                "warning",
                f"captured output truncated after {MAX_CAPTURED_RECORDS} records",
                data={"origin": "capture"},
            )
            return
        if len(message) > MAX_CAPTURED_MESSAGE_CHARS:
            message = message[:MAX_CAPTURED_MESSAGE_CHARS] + _TRUNCATION_MARKER
        report_log(level, message, data=data)


def forward_log_record(record: logging.LogRecord) -> None:
    """Forward one python logging record as an execution log event, when an
    invocation is being captured. Called by the standard handler after its
    terminal write; a no-op outside execution or under the guard."""
    if record.levelno < logging.INFO:
        return
    budget = _budget.get()
    if budget is None or capture_suppressed():
        return
    level = "warning" if record.levelno >= logging.WARNING else "info"
    data: dict[str, object] = {"origin": "logging", "logger": record.name}
    if record.levelno >= logging.ERROR:
        # Forwarded as "warning": error is not a logging level here - node
        # failures raise and use the single existing error report. The
        # python level is preserved so nothing is lost.
        data["pythonLevel"] = record.levelname
    message = record.getMessage()
    if record.exc_info is not None and record.exc_info != (None, None, None):
        exc = record.exc_info[1]
        if exc is not None:
            message = f"{message}\n{type(exc).__name__}: {exc}"
    _forward(budget, level, message, data)


_STREAM_LEVELS = {"stdout": "info", "stderr": "warning"}


class _CaptureStream:
    """Pass-through stream proxy: every write reaches the original stream
    first, then complete non-blank lines are forwarded as log records when
    an invocation is being captured."""

    def __init__(self, original: TextIO, name: str) -> None:
        self._original = original
        self._name = name
        self._level = _STREAM_LEVELS[name]
        # Serializes terminal write, line folding, and forwarding for THIS
        # stream, so captured records match the terminal's order. Reentrant
        # as insurance (delivery output takes the suppressed lock-free path,
        # but a custom original stream could write back into the proxy).
        self._lock = threading.RLock()

    def write(self, text: str) -> int:
        if capture_suppressed():
            return self._original.write(text)
        budget = _budget.get()
        if budget is None or not text:
            return self._original.write(text)
        with self._lock:
            result = self._original.write(text)
            lines = budget.split_lines(self._name, text)
            try:
                for line in lines:
                    if line.strip():
                        _forward(budget, self._level, line, {"origin": self._name})
            except Exception:
                # Forwarding is chatter: a broken delivery path must never
                # break the node's own write (which already reached the
                # terminal above).
                pass
            return result

    def flush(self) -> None:
        self._original.flush()

    @property
    def dinkster_original(self) -> TextIO:
        return self._original

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


def install_stream_capture() -> None:
    """Wrap ``sys.stdout``/``sys.stderr`` in capture proxies. Idempotent and
    process-global; worker hosts call it at startup. Everything written
    still reaches the real streams."""
    if sys.stdout is not None and not isinstance(sys.stdout, _CaptureStream):
        sys.stdout = cast("TextIO", _CaptureStream(sys.stdout, "stdout"))
    if sys.stderr is not None and not isinstance(sys.stderr, _CaptureStream):
        sys.stderr = cast("TextIO", _CaptureStream(sys.stderr, "stderr"))
