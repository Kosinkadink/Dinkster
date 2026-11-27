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

import contextvars
import io
import logging
import sys
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from dinkster_caches import DiskCacheStore, MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import CacheStore
from dinkster_schema import (
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    configure_logging,
    install_stream_capture,
    report_event,
    report_log,
    report_preview,
    report_progress,
    report_value_diagnostic,
    use_capture_budget,
    use_reporter,
)
from dinkster_schema.capture import MAX_CAPTURED_RECORDS
from dinkster_schema.log import ROOT_LOGGER_NAME
from dinkster_schema.reporting import capture_value_diagnostics
from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker

from tests.async_scenarios import run_scenario

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


def test_private_invocation_hides_node_text_but_keeps_numeric_progress() -> None:
    captured: list[Captured] = []
    configured_stream = io.StringIO()
    configure_logging("info", stream=configured_stream)
    with proxied_streams() as (out, err):
        with (
            use_reporter(recorder(captured)),
            capture_value_diagnostics() as diagnostics,
            use_capture_budget(private=True),
        ):
            sys.stdout.write("private-")
            sys.stdout.flush()
            sys.stdout.write("credential\n")
            sys.stderr.write("private-credential\n")
            try:
                raise ValueError("private-credential")
            except ValueError:
                logging.getLogger("dinkster.pack.test").warning("private-credential", exc_info=True)
            report_log("warning", "private-credential")
            report_event("test.private-credential", {"value": "private-credential"})
            report_preview(b"private-credential")
            report_value_diagnostic("private-credential", {"value": "private-credential"})
            report_progress(1, 4, text="private-credential")
            with use_capture_budget():
                print("nested private-credential")
            copied = contextvars.copy_context()
            thread = threading.Thread(target=copied.run, args=(print, "private-credential"))
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
            contextvars.Context().run(print, "unrelated invocation")
        print("ordinary output")
        assert out.getvalue() == "unrelated invocation\nordinary output\n"
        assert err.getvalue() == configured_stream.getvalue() == ""
    assert diagnostics == []
    assert captured == [("progress", {"step": 1, "total": 4}, None)]


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
            body=(Input("value", STRING), Output("value", STRING)),
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

    run_scenario(scenario())


class LogWarner(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="execlog.warner",
            category="test",
            body=(Input("value", STRING), Output("value", STRING)),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        report_log("info", f"working on {value}")
        if len(value) > 3:
            report_log("warning", f"{value} is long", data={"length": len(value)})
            report_log("warning", f"{value} shows an earlier view")
        return cls.outputs(value=value.upper())


def test_a_cached_result_shows_the_warnings_its_node_gave(tmp_path: Path) -> None:
    """A run that takes a node from the cache shows, on that node, the
    warnings the node gave when it made the result, and no info notes; a
    result made without warnings shows none, and a node that runs again
    shows only its new warnings. The warnings survive the disk cache."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        events: list[EngineEvent] = []

        def engine(cache: CacheStore) -> Engine:
            return Engine(
                schemas={"execlog.warner": LogWarner.schema()},
                registry=registry,
                worker=InProcessWorker({"execlog.warner": LogWarner}, registry),
                cache=cache,
                on_event=events.append,
            )

        async def run(target: Engine, value: str) -> tuple[str, list[dict[str, object]]]:
            events.clear()
            graph = Graph(nodes={"w": GraphNode("execlog.warner", {"value": value})})
            result = await target.run(graph, ["w"])
            kinds = [e.kind for e in events if e.node_id == "w"]
            ending = "node_cached" if "node_cached" in kinds else "node_finished"
            assert kinds[-1] == ending, kinds
            logs = [
                {k: v for k, v in cast("Mapping[str, object]", e.detail["data"]).items()}
                for e in events
                if e.kind == "node_event" and e.node_id == "w" and e.detail["name"] == "log"
            ]
            assert all(isinstance(log.pop("ts"), float) for log in logs)
            assert all(e.run_id == result.run_id for e in events if e.node_id == "w")
            return ending, logs

        warned = [
            {"level": "warning", "message": "hello is long", "length": 5},
            {"level": "warning", "message": "hello shows an earlier view"},
        ]
        memory = engine(MemoryLRUCache())
        first = await run(memory, "hello")
        assert first == (
            "node_finished",
            [{"level": "info", "message": "working on hello"}, *warned],
        )
        assert await run(memory, "hello") == ("node_cached", warned)
        assert await run(memory, "hi") == (
            "node_finished",
            [{"level": "info", "message": "working on hi"}],
        )
        assert await run(memory, "hi") == ("node_cached", [])
        assert await run(memory, "howdy") == (
            "node_finished",
            [
                {"level": "info", "message": "working on howdy"},
                {"level": "warning", "message": "howdy is long", "length": 5},
                {"level": "warning", "message": "howdy shows an earlier view"},
            ],
        )

        assert (await run(engine(DiskCacheStore(tmp_path, registry)), "hello"))[0] == (
            "node_finished"
        )
        # A fresh engine reads the entry back from disk.
        assert await run(engine(DiskCacheStore(tmp_path, registry)), "hello") == (
            "node_cached",
            warned,
        )

    run_scenario(scenario())


# -- across the process boundary --------------------------------------------------


def write_iso_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\npublisher = "dinkster"\n'
        'type-namespaces = ["iso"]\n\n[pack.entry]\n'
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

    run_scenario(scenario())
