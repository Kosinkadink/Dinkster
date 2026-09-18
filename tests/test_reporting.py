"""Typed node events (DESIGN 3.5): report_progress/report_preview/
report_event flow from node code to engine listeners with run/node
provenance, in order, before the node's terminal event - identically for
in-process and isolated workers. Reporting is chatter: it never blocks a
node, never fails one, and asks nothing of nodes that stay silent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import Context
from pathlib import Path
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import InvocationEvent
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    report_event,
    report_preview,
    report_progress,
    schema_to_wire,
    use_reporter,
)
from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker

STRING = TypeExpr.concrete(CORE_STRING)
TESTS_DIR = Path(__file__).parent

Captured = tuple[str, Mapping[str, object], bytes | None]


class Chatty(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rep.chatty",
            display_name="Chatty",
            category="test",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        report_progress(1, 2, text=value)
        report_progress(2, 2)
        report_preview(b"\x89fake", mime="image/png", width=4, height=2)
        report_event("rep.stage", {"stage": "done"})
        return cls.outputs(value=value.upper())


class Threaded(Node):
    """Reports from a worker thread via asyncio.to_thread - the contextvar
    crosses (to_thread copies context) and delivery marshals to the loop."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rep.threaded",
            display_name="Threaded",
            category="test",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        await asyncio.to_thread(report_progress, 1, 1)
        return cls.outputs(value=value)


class Silent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rep.silent",
            display_name="Silent",
            category="test",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class PreviewChannels(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rep.preview-channels",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("final", STRING, preview=True),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        report_preview(b"frame-1", mime="image/png", width=1, height=1)
        report_preview(b"frame-2", mime="image/webp", width=2, height=1)
        return cls.outputs(final=f"final:{value}")


class TerminalOrder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rep.terminal-order",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        report_progress(1, 1)
        return cls.outputs(value=value)


LOCAL_NODES = [Chatty, Threaded, Silent, PreviewChannels, TerminalOrder]


def make_engine(events: list[EngineEvent]) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=build_schemas(LOCAL_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(LOCAL_NODES), registry),
        cache=MemoryLRUCache(),
        on_event=events.append,
    )


def node_events(events: list[EngineEvent], node_id: str) -> list[EngineEvent]:
    return [e for e in events if e.kind == "node_event" and e.node_id == node_id]


# --- the ambient API's own contract ---


def test_report_functions_are_noops_without_a_reporter() -> None:
    # Unit-testing a node by calling execute() directly must just work.
    report_progress(3, 10, text="fine")
    report_preview(b"bytes")
    report_event("some.event", {"k": 1}, blob=b"b")


def test_custom_event_names_must_be_namespaced() -> None:
    with pytest.raises(ValueError, match="reserved"):
        report_event("progress")
    with pytest.raises(ValueError, match="reserved"):
        report_event("preview")
    with pytest.raises(ValueError, match="dot-namespaced"):
        report_event("stage")


def test_helper_payload_shapes() -> None:
    captured: list[Captured] = []

    def capture(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        captured.append((name, data, blob))

    with use_reporter(capture):
        report_progress(2, 5, text="phase")
        report_progress(3, 5)
        report_preview(b"img", mime="image/webp", width=8, height=6)
        report_event("my.thing", {"a": 1})
    # Outside the context: silent again.
    report_progress(4, 5)

    assert captured == [
        ("progress", {"step": 2, "total": 5, "text": "phase"}, None),
        ("progress", {"step": 3, "total": 5}, None),
        ("preview", {"mime": "image/webp", "width": 8, "height": 6}, b"img"),
        ("my.thing", {"a": 1}, None),
    ]


# --- through the engine, in-process ---


def test_engine_surfaces_node_events_with_provenance_and_order() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        graph = Graph(nodes={"c": GraphNode("rep.chatty", {"value": "hi"})})
        result = await engine.run(graph, ["c"])
        assert result.outputs["c"]["value"].resolve() == "HI"

        reports = node_events(events, "c")
        assert [e.detail["name"] for e in reports] == [
            "progress",
            "progress",
            "preview",
            "rep.stage",
        ]
        preview = reports[2]
        assert preview.detail["blob"] == b"\x89fake"
        assert preview.detail["data"] == {"mime": "image/png", "width": 4, "height": 2}
        assert all(e.run_id == result.run_id for e in reports)

        # Reports land between the node's started and finished events.
        kinds = [e.kind for e in events if e.node_id == "c" or e.kind == "node_event"]
        assert kinds[0] == "node_started"
        assert kinds[-1] == "node_finished"
        assert kinds[1:-1] == ["node_event"] * 4

    asyncio.run(scenario())


def test_runtime_previews_and_preview_marked_final_are_distinct() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        result = await engine.run(
            Graph(nodes={"p": GraphNode("rep.preview-channels", {"value": "x"})}), ["p"]
        )
        reports = node_events(events, "p")
        assert [event.detail["blob"] for event in reports] == [b"frame-1", b"frame-2"]
        assert [event.detail["data"] for event in reports] == [
            {"mime": "image/png", "width": 1, "height": 1},
            {"mime": "image/webp", "width": 2, "height": 1},
        ]
        assert all(event.run_id == result.run_id and event.node_id == "p" for event in reports)
        assert result.outputs["p"]["final"].resolve() == "final:x"
        output_wire = cast(
            "list[dict[str, object]]", schema_to_wire(PreviewChannels.schema())["interface"]
        )[1]
        assert output_wire["preview"] is True
        kinds = [event.kind for event in events if event.node_id == "p"]
        assert kinds == ["node_started", "node_event", "node_event", "node_finished"]

    asyncio.run(scenario())


def test_thread_emission_reaches_the_listener() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        graph = Graph(nodes={"t": GraphNode("rep.threaded", {"value": "x"})})
        await engine.run(graph, ["t"])
        reports = node_events(events, "t")
        assert [e.detail["name"] for e in reports] == ["progress"]

    asyncio.run(scenario())


def test_in_process_invoke_waits_for_accepted_cross_thread_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        call_soon_threadsafe = loop.call_soon_threadsafe
        submit = ThreadPoolExecutor.submit

        def submit_completed(
            executor: ThreadPoolExecutor,
            fn: Callable[..., object],
            /,
            *args: object,
            **kwargs: object,
        ) -> Future[object]:
            future = submit(executor, fn, *args, **kwargs)
            # Force run_in_executor to wrap an already-completed future.
            future.result()
            return future

        monkeypatch.setattr(ThreadPoolExecutor, "submit", submit_completed)

        for _ in range(20):
            events: list[EngineEvent] = []
            engine = make_engine(events)
            report_accepted = asyncio.Event()
            held: list[tuple[Callable[..., object], tuple[object, ...], Context | None]] = []

            def intercept(
                callback: Callable[..., object],
                *args: object,
                context: Context | None = None,
                held_reports: list[
                    tuple[Callable[..., object], tuple[object, ...], Context | None]
                ] = held,
                accepted: asyncio.Event = report_accepted,
            ) -> asyncio.Handle | None:
                if not held_reports and len(args) == 1 and isinstance(args[0], InvocationEvent):
                    held_reports.append((callback, args, context))
                    call_soon_threadsafe(accepted.set)
                    return None
                return call_soon_threadsafe(callback, *args, context=context)

            monkeypatch.setattr(loop, "call_soon_threadsafe", intercept)
            task = asyncio.create_task(
                engine.run(
                    Graph(nodes={"t": GraphNode("rep.terminal-order", {"value": "x"})}), ["t"]
                )
            )
            await report_accepted.wait()
            barrier = loop.create_future()
            loop.call_soon(barrier.set_result, None)
            await barrier
            assert not task.done()

            callback, args, context = held.pop()
            call_soon_threadsafe(callback, *args, context=context)
            result = await task
            assert result.outputs["t"]["value"].resolve() == "x"
            assert [event.kind for event in events if event.node_id == "t"] == [
                "node_started",
                "node_event",
                "node_finished",
            ]

    asyncio.run(scenario())


def test_in_process_report_schedule_failure_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        loop = asyncio.get_running_loop()
        call_soon_threadsafe = loop.call_soon_threadsafe

        def intercept(
            callback: Callable[..., object], *args: object, context: Context | None = None
        ) -> asyncio.Handle:
            if len(args) == 1 and isinstance(args[0], InvocationEvent):
                raise RuntimeError("report scheduling failed")
            return call_soon_threadsafe(callback, *args, context=context)

        monkeypatch.setattr(loop, "call_soon_threadsafe", intercept)
        result = await engine.run(
            Graph(nodes={"t": GraphNode("rep.terminal-order", {"value": "x"})}), ["t"]
        )
        assert result.outputs["t"]["value"].resolve() == "x"
        assert [event.kind for event in events if event.node_id == "t"] == [
            "node_started",
            "node_finished",
        ]

    asyncio.run(scenario())


def test_broken_listener_does_not_fail_the_node() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []

        def listener(event: EngineEvent) -> None:
            events.append(event)
            if event.kind == "node_event":
                raise RuntimeError("observer bug")

        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(LOCAL_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(LOCAL_NODES), registry),
            cache=MemoryLRUCache(),
            on_event=listener,
        )
        graph = Graph(nodes={"c": GraphNode("rep.chatty", {"value": "ok"})})
        result = await engine.run(graph, ["c"])
        assert result.outputs["c"]["value"].resolve() == "OK"
        assert node_events(events, "c")  # events were still delivered

    asyncio.run(scenario())


def test_silent_node_pays_nothing() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        graph = Graph(nodes={"s": GraphNode("rep.silent", {"value": "q"})})
        await engine.run(graph, ["s"])
        assert not [e for e in events if e.kind == "node_event"]

    asyncio.run(scenario())


# --- across the process boundary ---


def write_iso_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\n\n[pack.entry]\n'
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
    )
    return manifest


def test_events_stream_across_the_process_boundary(tmp_path: Path) -> None:
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
            graph = Graph(nodes={"c": GraphNode("iso.chatty", {"value": "hi"})})
            result = await engine.run(graph, ["c"])
            assert result.outputs["c"]["value"].resolve() == "HI"

            reports = node_events(events, "c")
            assert [e.detail["name"] for e in reports] == [
                "progress",
                "progress",
                "preview",
                "isopack.stage",
            ]
            assert reports[0].detail["data"] == {"step": 1, "total": 2, "text": "hi"}
            assert reports[2].detail["blob"] == b"\x89fakepng:hi"
            assert reports[3].detail["data"] == {"stage": "done", "value": "hi"}

            # Ordering guarantee: every report precedes the terminal event.
            index = {id(e): i for i, e in enumerate(events)}
            finished = next(e for e in events if e.kind == "node_finished" and e.node_id == "c")
            assert all(index[id(e)] < index[id(finished)] for e in reports)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_concurrent_invocations_route_events_to_their_own_nodes(
    tmp_path: Path,
) -> None:
    """Two chatty nodes overlap in one graph (io_bound + sleep): each
    node_event must carry the node that emitted it, never its neighbor."""

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
            graph = Graph(
                nodes={
                    "a": GraphNode("iso.chatty", {"value": "aa", "seconds": 0.05}),
                    "b": GraphNode("iso.chatty", {"value": "bb", "seconds": 0.05}),
                }
            )
            await engine.run(graph, ["a", "b"])

            for node_id, value in (("a", "aa"), ("b", "bb")):
                reports = node_events(events, node_id)
                assert [e.detail["name"] for e in reports] == [
                    "progress",
                    "progress",
                    "preview",
                    "isopack.stage",
                ], node_id
                # Payloads prove provenance: each node's events carry its
                # own input, so no frame was routed to the wrong sink.
                data = reports[0].detail["data"]
                assert isinstance(data, Mapping) and data["text"] == value
                assert reports[2].detail["blob"] == b"\x89fakepng:" + value.encode()
        finally:
            await worker.close()

    asyncio.run(scenario())
