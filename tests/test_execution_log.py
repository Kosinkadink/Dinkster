"""Execution log events: report_log plus stdout/stderr/logging capture.

The contract under test (issue #368, slice 1):

- ``report_log(level, message)`` is the explicit voice: ``info`` or
  ``warning``, never "error" - a failing node raises, and the failure
  travels through the single existing error report (NodeError ->
  node_failed), not the log.
- While an invocation is captured (ambient reporter + capture budget),
  stdout becomes ``info`` records, stderr ``warning`` records, and python
  logging records INFO+ forward with their origin logger named -
  ERROR/CRITICAL as ``warning`` with the python level preserved.
- Capture forwards, never swallows: every write still reaches the real
  stream; the terminal keeps working.
- Delivery machinery output is never re-captured (no feedback loops), a
  configured-handler record forwards exactly once, and a per-invocation
  budget bounds record count and message size.
- Through the engine, log records arrive as ``node_event`` events named
  ``log`` with run/node provenance - identically for in-process and
  isolated workers.
"""

from __future__ import annotations

import asyncio
import contextvars
import io
import logging
import sys
import threading
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    configure_logging,
    install_stream_capture,
    report_event,
    report_log,
    use_capture_budget,
    use_reporter,
)
from dinkster_schema.capture import MAX_CAPTURED_MESSAGE_CHARS, MAX_CAPTURED_RECORDS
from dinkster_schema.log import ROOT_LOGGER_NAME
from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker

STRING = TypeExpr.concrete(CORE_STRING)
TESTS_DIR = Path(__file__).parent

Captured = tuple[str, Mapping[str, object], bytes | None]


@pytest.fixture(autouse=True)
def _reset_dinkster_logging():
    """Restore the dinkster logger tree after each test."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved = (list(root.handlers), root.level, root.propagate)
    yield
    root.handlers[:], root.level, root.propagate = saved[0], saved[1], saved[2]


@contextmanager
def proxied_streams() -> Generator[tuple[io.StringIO, io.StringIO]]:
    """Fresh StringIO stdout/stderr wrapped by the capture proxies.

    Must run INSIDE the test body: pytest's capture machinery re-assigns
    ``sys.stdout``/``sys.stderr`` between the fixture and call phases, so a
    fixture-time swap would not survive into the test.
    """
    out, err = io.StringIO(), io.StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        install_stream_capture()
        yield out, err
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def recorder(captured: list[Captured]):
    def report(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        captured.append((name, data, blob))

    return report


def log_events(captured: list[Captured]) -> list[Mapping[str, object]]:
    return [data for name, data, _ in captured if name == "log"]


# -- report_log's own contract ------------------------------------------------


def test_report_log_payload_shape() -> None:
    captured: list[Captured] = []
    before = time.time()
    with use_reporter(recorder(captured)):
        report_log("info", "hello")
        report_log("warning", "careful", data={"detail": 7})
    after = time.time()

    assert [name for name, _, _ in captured] == ["log", "log"]
    first, second = log_events(captured)
    assert first["level"] == "info"
    assert first["message"] == "hello"
    assert second["level"] == "warning"
    assert second["message"] == "careful"
    assert second["detail"] == 7
    for event in (first, second):
        ts = event["ts"]
        assert isinstance(ts, float)
        assert before <= ts <= after
        assert "origin" not in event  # explicit calls carry no origin
    assert all(blob is None for _, _, blob in captured)


def test_report_log_rejects_non_log_levels() -> None:
    # "error" is deliberately not a log level: failures raise and travel
    # through the one existing error report.
    with pytest.raises(ValueError, match="raise"):
        report_log("error", "boom")
    with pytest.raises(ValueError, match="unknown log level"):
        report_log("debug", "chatter")


def test_report_log_rejects_reserved_data_keys() -> None:
    with pytest.raises(ValueError, match="reserved"):
        report_log("info", "x", data={"level": "warning"})
    with pytest.raises(ValueError, match="reserved"):
        report_log("info", "x", data={"message": "smuggled", "ts": 1.0})


def test_report_log_is_a_noop_without_a_reporter() -> None:
    report_log("info", "nobody listening")


def test_log_event_name_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved"):
        report_event("log")


# -- stream capture ------------------------------------------------------------


def test_stdout_and_stderr_capture_attributes_and_passes_through() -> None:
    captured: list[Captured] = []
    with proxied_streams() as (out, err):
        with use_reporter(recorder(captured)), use_capture_budget():
            print("plain line")
            sys.stderr.write("worry line\n")
            print("")  # blank lines are not records

        assert out.getvalue() == "plain line\n\n"
        assert err.getvalue() == "worry line\n"
    events = log_events(captured)
    assert [(e["level"], e["message"], e["origin"]) for e in events] == [
        ("info", "plain line", "stdout"),
        ("warning", "worry line", "stderr"),
    ]


def test_stream_writes_outside_an_invocation_are_not_captured() -> None:
    captured: list[Captured] = []
    with proxied_streams() as (out, _):
        print("before any invocation")
        with use_reporter(recorder(captured)):
            print("reporter but no budget")
        assert "before any invocation" in out.getvalue()
        assert "reporter but no budget" in out.getvalue()
    assert captured == []


def test_partial_line_flushes_when_the_invocation_ends() -> None:
    captured: list[Captured] = []
    with proxied_streams() as (out, _):
        with use_reporter(recorder(captured)):
            with use_capture_budget():
                sys.stdout.write("no newline yet")
            # Budget exit flushed the tail while the reporter was open.
        assert out.getvalue() == "no newline yet"
    events = log_events(captured)
    assert [(e["level"], e["message"]) for e in events] == [("info", "no newline yet")]


def test_capture_stops_after_the_record_budget_with_one_notice() -> None:
    captured: list[Captured] = []
    total_lines = MAX_CAPTURED_RECORDS + 5
    with proxied_streams() as (out, _):
        with use_reporter(recorder(captured)), use_capture_budget():
            for i in range(total_lines):
                print(f"line {i}")
        # Terminal output is never rationed.
        assert out.getvalue().count("\n") == total_lines
    events = log_events(captured)
    assert len(events) == MAX_CAPTURED_RECORDS + 1
    assert [e["message"] for e in events[:MAX_CAPTURED_RECORDS]] == [
        f"line {i}" for i in range(MAX_CAPTURED_RECORDS)
    ]
    notice = events[-1]
    assert notice["level"] == "warning"
    assert notice["origin"] == "capture"
    assert "truncated" in str(notice["message"])


def test_overlong_messages_are_cut() -> None:
    captured: list[Captured] = []
    long = "x" * (MAX_CAPTURED_MESSAGE_CHARS + 500)
    with proxied_streams() as (out, _):
        with use_reporter(recorder(captured)), use_capture_budget():
            print(long)
        assert long in out.getvalue()  # pass-through is untouched
    (event,) = log_events(captured)
    message = str(event["message"])
    assert message.startswith("x" * MAX_CAPTURED_MESSAGE_CHARS)
    assert message.endswith("[truncated]")
    assert len(message) < len(long)


def test_overlong_partial_line_becomes_a_record_without_a_newline() -> None:
    captured: list[Captured] = []
    with proxied_streams():
        with use_reporter(recorder(captured)), use_capture_budget():
            sys.stdout.write("y" * (MAX_CAPTURED_MESSAGE_CHARS + 100))
            events_before_exit = len(log_events(captured))
    assert events_before_exit == 1  # cut immediately, not held until exit
    (event,) = log_events(captured)
    assert str(event["message"]).endswith("[truncated]")


# -- python logging capture -----------------------------------------------------


def test_configured_logging_forwards_once_with_origin() -> None:
    captured: list[Captured] = []
    with proxied_streams() as (_, err):
        # The handler writes its terminal line through the stderr PROXY: the
        # single-capture guarantee under test is that the formatted line the
        # handler just wrote is not re-captured by the proxy.
        configure_logging("info", stream=sys.stderr)
        with use_reporter(recorder(captured)), use_capture_budget():
            logging.getLogger("dinkster.pack.mypack").warning("pack says %s", "hi")
            logging.getLogger("dinkster.server").info("serving")
        terminal = err.getvalue()
    assert "pack says hi" in terminal and "serving" in terminal
    events = log_events(captured)
    assert [(e["level"], e["message"], e["origin"], e["logger"]) for e in events] == [
        ("warning", "pack says hi", "logging", "dinkster.pack.mypack"),
        ("info", "serving", "logging", "dinkster.server"),
    ]
    assert all("pythonLevel" not in e for e in events)


def test_debug_records_are_not_forwarded() -> None:
    configure_logging("debug", stream=io.StringIO())
    captured: list[Captured] = []
    with use_reporter(recorder(captured)), use_capture_budget():
        logging.getLogger("dinkster.server").debug("wire noise")
    assert log_events(captured) == []


def test_error_records_forward_as_warning_with_python_level() -> None:
    configure_logging("info", stream=io.StringIO())
    captured: list[Captured] = []
    with use_reporter(recorder(captured)), use_capture_budget():
        try:
            raise ValueError("nope")
        except ValueError:
            logging.getLogger("dinkster.pack.mypack").error("stage failed", exc_info=True)
        logging.getLogger("dinkster.pack.mypack").critical("still going")

    events = log_events(captured)
    assert [e["level"] for e in events] == ["warning", "warning"]
    assert [e["pythonLevel"] for e in events] == ["ERROR", "CRITICAL"]
    first_message = str(events[0]["message"])
    assert first_message.startswith("stage failed")
    assert "ValueError: nope" in first_message


def test_unconfigured_logger_warning_is_captured_via_stderr() -> None:
    # No dinkster handler and no propagation: logging falls back to
    # logging.lastResort, which resolves sys.stderr dynamically - so the
    # stderr proxy sees the message with no root handler installed.
    orphan = logging.getLogger("execlogtest.orphan")
    orphan.propagate = False
    captured: list[Captured] = []
    with proxied_streams() as (_, err):
        with use_reporter(recorder(captured)), use_capture_budget():
            orphan.warning("orphan warning")
        assert "orphan warning" in err.getvalue()
    events = log_events(captured)
    assert [(e["level"], e["origin"]) for e in events] == [("warning", "stderr")]
    assert "orphan warning" in str(events[0]["message"])


def test_malformed_logging_call_does_not_raise() -> None:
    # A bad %-format call is handled by logging itself for the terminal
    # write; forwarding must be equally forgiving and never raise into
    # whatever logged it - under production settings (raiseExceptions
    # True) and with the handler writing through the stderr proxy, whose
    # capture must not pick up the stdlib's error diagnostic either.
    assert logging.raiseExceptions
    captured: list[Captured] = []
    with proxied_streams():
        configure_logging("info", stream=sys.stderr)
        with use_reporter(recorder(captured)), use_capture_budget():
            logging.getLogger("dinkster.server").info("%s %s", "only-one-arg")
            logging.getLogger("dinkster.server").info("fine afterwards")
    assert [e["message"] for e in log_events(captured)] == ["fine afterwards"]


def test_broken_reporter_does_not_break_node_writes() -> None:
    def broken_reporter(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        raise RuntimeError("delivery exploded")

    with proxied_streams() as (out, _):
        with use_reporter(broken_reporter), use_capture_budget():
            print("still fine")
        assert "still fine" in out.getvalue()


def test_cross_thread_writes_assemble_consistent_lines() -> None:
    # Two threads share one invocation's budget (asyncio.to_thread copies
    # the context). The terminal write and line assembly are atomic, so a
    # partial line one thread left open folds with the next write in
    # terminal order.
    captured: list[Captured] = []
    with proxied_streams() as (out, _):
        with use_reporter(recorder(captured)), use_capture_budget():
            source = contextvars.copy_context()
            a_wrote = threading.Event()
            b_wrote = threading.Event()

            def writer_a() -> None:
                sys.stdout.write("foo")
                a_wrote.set()
                b_wrote.wait(timeout=5)
                sys.stdout.write("baz\n")

            def writer_b() -> None:
                a_wrote.wait(timeout=5)
                sys.stdout.write("bar\n")
                b_wrote.set()

            threads = [
                threading.Thread(target=source.copy().run, args=(writer,))
                for writer in (writer_a, writer_b)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        assert out.getvalue() == "foobar\nbaz\n"
    events = log_events(captured)
    # Exact order: the per-stream lock serializes write, folding, and
    # forwarding, so records match the terminal's order.
    assert [str(e["message"]) for e in events] == ["foobar", "baz"]


def test_delivery_output_is_never_recaptured() -> None:
    captured: list[Captured] = []

    def noisy_reporter(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        captured.append((name, data, blob))
        print(f"delivering {name}")  # listener chatter must not loop back

    with proxied_streams() as (out, _):
        with use_reporter(noisy_reporter), use_capture_budget():
            report_log("info", "one event only")
        assert "delivering log" in out.getvalue()
    assert len(log_events(captured)) == 1


# -- through the engine, in-process ---------------------------------------------


class LogChatty(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="execlog.chatty",
            display_name="Log Chatty",
            category="test",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        report_log("info", f"explicit:{value}", data={"detail": "d"})
        print(f"printed:{value}")
        return cls.outputs(value=value.upper())


def test_engine_surfaces_log_events_with_provenance() -> None:
    async def scenario() -> None:
        install_stream_capture()
        events: list[EngineEvent] = []
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas={s.node_type: s for s in [LogChatty.schema()]},
            registry=registry,
            worker=InProcessWorker({"execlog.chatty": LogChatty}, registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph(nodes={"c": GraphNode("execlog.chatty", {"value": "hi"})})
        result = await engine.run(graph, ["c"])
        assert result.outputs["c"]["value"].resolve() == "HI"

        reports = [e for e in events if e.kind == "node_event" and e.node_id == "c"]
        assert [e.detail["name"] for e in reports] == ["log", "log"]
        explicit, printed = [cast("Mapping[str, object]", e.detail["data"]) for e in reports]
        assert explicit["level"] == "info"
        assert explicit["message"] == "explicit:hi"
        assert explicit["detail"] == "d"
        assert "origin" not in explicit
        assert printed["level"] == "info"
        assert printed["message"] == "printed:hi"
        assert printed["origin"] == "stdout"
        assert all(e.run_id == result.run_id for e in reports)

        # Log events land between the node's started and finished events.
        kinds = [e.kind for e in events if e.node_id == "c" or e.kind == "node_event"]
        assert kinds[0] == "node_started"
        assert kinds[-1] == "node_finished"

    asyncio.run(scenario())


# -- across the process boundary --------------------------------------------------


def write_iso_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\n\n[pack.entry]\n'
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
    )
    return manifest


def test_log_events_stream_across_the_process_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            write_iso_manifest(tmp_path),
            registry,
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        await worker.start()
        try:
            events: list[EngineEvent] = []
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=events.append,
            )
            graph = Graph(nodes={"t": GraphNode("iso.talky", {"value": "hi"})})
            result = await engine.run(graph, ["t"])
            assert result.outputs["t"]["value"].resolve() == "HI"

            reports = [e for e in events if e.kind == "node_event" and e.node_id == "t"]
            assert [e.detail["name"] for e in reports] == ["log"] * 4
            data = [cast("Mapping[str, object]", e.detail["data"]) for e in reports]
            assert [(d["level"], d["message"], d.get("origin")) for d in data] == [
                ("info", "explicit:hi", None),
                ("info", "printed:hi", "stdout"),
                ("warning", "errline:hi", "stderr"),
                ("warning", "logged:hi", "logging"),
            ]
            assert data[0]["detail"] == "d"
            assert data[3]["logger"] == "dinkster.pack.isopack"
            assert all(isinstance(d["ts"], float) for d in data)
            assert all(e.run_id == result.run_id for e in reports)
        finally:
            await worker.close()

    asyncio.run(scenario())
