"""Automatic values: a field that declares ``automatic`` is worked out again
when a run is queued, the value used is recorded with the run, and a value
that cannot be worked out gives the field's default without blocking the run."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets.model import AssetRef
from dinkster_assets.value import ASSET_TYPE
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_engine.automatic import resolve_automatic
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_schema import (
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_schema.body import Formula, Lookup, State, lookup
from dinkster_schema.model import AssetWidget, NumberWidget
from dinkster_server import STATE_KEY, create_app
from dinkster_server.history import HistoryStore, record_from_job
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

import dinkster.compose
from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

DEV_PACK = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-dev/dinkster-pack.toml"

INT = TypeExpr.concrete("INT")
FLOAT = TypeExpr.concrete("FLOAT")
STRING = TypeExpr.concrete("STRING")


@lookup
def source_rate(rate: float) -> float:
    return rate / 4


@lookup
def rate_name(rate: float) -> str:
    return str(rate)


class FrameRate(Node):
    received: list[float] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.frame_rate",
            output_node=True,
            body=(
                Input("rate", FLOAT, default=48.0),
                Input("divisor", INT, default=2),
                Input("fps", FLOAT, default=24.0, automatic=Formula("rate / divisor")),
                Output("fps_out", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, rate: float, divisor: int, fps: float) -> Mapping[str, object]:
        cls.received.append(fps)
        return cls.outputs(fps_out=fps)


class Number(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.number",
            body=(Input("value", FLOAT, default=1.0), Output("out", FLOAT)),
        )

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        return cls.outputs(out=value)


class LookedUp(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.looked_up",
            output_node=True,
            body=(
                Input("rate", FLOAT, default=48.0),
                Input("fps", FLOAT, default=24.0, automatic=Lookup(source_rate, args=("rate",))),
                Output("fps_out", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, rate: float, fps: float) -> Mapping[str, object]:
        return cls.outputs(fps_out=fps)


@lookup
def file_rate(video: AssetRef) -> float:
    return 30.0


class FromFile(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.from_file",
            output_node=True,
            body=(
                Input(
                    "video",
                    TypeExpr.concrete(ASSET_TYPE),
                    required=False,
                    widget=AssetWidget(accept=("video/mp4",)),
                ),
                Input("fps", FLOAT, default=24.0, automatic=Lookup(file_rate, args=("video",))),
                Output("fps_out", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, fps: float, video: object = None) -> Mapping[str, object]:
        return cls.outputs(fps_out=fps)


class Unreadable(Node):
    """A lookup read back from the wire names an image socket, which the
    load check cannot type and a formula cannot read."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.unreadable",
            output_node=True,
            body=(
                Input("image", TypeExpr.concrete("IMAGE"), required=False),
                Input("fps", FLOAT, default=24.0, automatic=Lookup("pack.rate", args=("image",))),
                Output("fps_out", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, fps: float, image: object = None) -> Mapping[str, object]:
        return cls.outputs(fps_out=fps)


NODES = [FrameRate, Number, LookedUp, FromFile, Unreadable]
SCHEMAS = build_schemas(NODES)


def make_engine(on_event: object = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            lambda: (Input("fps", FLOAT, automatic=Formula("1.0")),),
            "automatic needs a default",
        ),
        (
            lambda: (
                Input("image", TypeExpr.concrete("IMAGE"), default=1, automatic=Formula("1")),
            ),
            "automatic needs a whole number, decimal number, text, choice or yes or no field",
        ),
        (
            lambda: (
                State("view", FLOAT, default=1.0, part_of_run=False),
                Input("fps", FLOAT, default=1.0, automatic=Formula("view * 2")),
            ),
            "a State with part_of_run=False",
        ),
        (
            lambda: (
                Input("note", FLOAT, default=1.0, pin=False, sent=False),
                Input("fps", FLOAT, default=1.0, automatic=Formula("note * 2")),
            ),
            "a field with sent=False",
        ),
        (
            lambda: (
                Input("note", FLOAT, default=1.0, pin=False, sent=False),
                Input("fps", FLOAT, default=1.0, automatic=Lookup(source_rate, args=("note",))),
            ),
            r"reads 'note', which is not part of a run \(a State with part_of_run=False or a "
            r"field with sent=False\)",
        ),
        (
            lambda: (
                Input("rate", FLOAT, default=1.0),
                Input("fps", FLOAT, default=1.0, automatic=Lookup(rate_name, args=("rate",))),
            ),
            "answers str where float or int is used",
        ),
        (
            lambda: (
                Input("a", FLOAT, default=1.0, automatic=Formula("b + 1")),
                Input("b", FLOAT, default=1.0, automatic=Formula("a * 2")),
            ),
            r"automatic fields read each other \(a -> b -> a\)",
        ),
        (
            lambda: (Input("a", FLOAT, default=1.0, automatic=Formula("a + 1")),),
            r"automatic fields read each other \(a -> a\)",
        ),
    ],
)
def test_a_pack_with_an_automatic_field_it_cannot_hold_is_refused(
    rows: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NodeSchema("test.refused", body=rows())  # type: ignore[operator]


def _submit(graph: Graph, job_id: str, targets: Sequence[str] = ("n",)) -> dict[str, object]:
    return {
        "clientId": "c",
        "jobId": job_id,
        "graph": graph_to_wire(graph),
        "targets": list(targets),
    }


async def _finished(client: TestClient, job_id: str) -> dict[str, object]:
    async with asyncio.timeout(10):
        while True:
            job = await (await client.get(f"/api/jobs/c/{job_id}")).json()
            if job["state"] in ("completed", "failed", "cancelled"):
                return job
            await asyncio.sleep(0.01)


def test_a_queued_run_uses_records_and_caches_the_worked_out_value(tmp_path: Path) -> None:
    async def scenario() -> None:
        FrameRate.received.clear()
        async with TestClient(TestServer(create_app(make_engine, SCHEMAS))) as client:
            first = GraphNode("test.frame_rate", {"rate": 60.0})
            response = await client.post("/api/jobs", json=_submit(Graph({"n": first}), "1"))
            assert response.status == 202, await response.text()
            job = await _finished(client, "1")
            assert job["state"] == "completed", job
            assert job["automaticValues"] == [
                {"nodeId": "n", "inputId": "fps", "value": 30.0, "source": "worked-out"}
            ]
            assert FrameRate.received == [30.0]

            # The same worked-out value is served from the cache.
            await client.post("/api/jobs", json=_submit(Graph({"n": first}), "2"))
            assert (await _finished(client, "2"))["state"] == "completed"
            assert FrameRate.received == [30.0]

            # A changed worked-out value runs the node again.
            changed = GraphNode("test.frame_rate", {"rate": 80.0})
            await client.post("/api/jobs", json=_submit(Graph({"n": changed}), "3"))
            assert (await _finished(client, "3"))["automaticValues"] == [
                {"nodeId": "n", "inputId": "fps", "value": 40.0, "source": "worked-out"}
            ]
            assert FrameRate.received == [30.0, 40.0]

            # A sent value is the value; nothing is worked out.
            fixed = GraphNode("test.frame_rate", {"rate": 80.0, "fps": 12.0})
            await client.post("/api/jobs", json=_submit(Graph({"n": fixed}), "4"))
            assert "automaticValues" not in await _finished(client, "4")
            assert FrameRate.received == [30.0, 40.0, 12.0]

            job = client.app[STATE_KEY].queue.get("c", "3")
            assert job is not None
            store = HistoryStore(tmp_path / "history.sqlite")
            store.put(record_from_job(job))
            kept = store.get(job.scope, job.run_id)
            assert kept is not None
            assert kept.to_wire()["automaticValues"] == [
                {"nodeId": "n", "inputId": "fps", "value": 40.0, "source": "worked-out"}
            ]

    run_scenario(scenario())


def test_a_value_that_cannot_be_worked_out_runs_with_the_default() -> None:
    async def scenario() -> None:
        FrameRate.received.clear()
        async with TestClient(TestServer(create_app(make_engine, SCHEMAS))) as client:
            failing = Graph({"n": GraphNode("test.frame_rate", {"divisor": 0})})
            await client.post("/api/jobs", json=_submit(failing, "1"))
            job = await _finished(client, "1")
            assert job["state"] == "completed", job
            assert job["automaticValues"] == [
                {
                    "nodeId": "n",
                    "inputId": "fps",
                    "value": 24.0,
                    "source": "default",
                    "reason": "failed",
                }
            ]
            wired = Graph(
                {
                    "v": GraphNode("test.number", {"value": 90.0}),
                    "n": GraphNode("test.frame_rate", {"rate": Link("v", "out")}),
                }
            )
            await client.post("/api/jobs", json=_submit(wired, "2"))
            job = await _finished(client, "2")
            assert job["automaticValues"][0]["reason"] == "unknown"  # type: ignore[index]
            assert FrameRate.received == [24.0, 24.0]

    run_scenario(scenario())


def test_unanswered_numeric_fix_uses_and_keeps_only_the_runs_corrected_value(
    tmp_path: Path,
) -> None:
    received: list[tuple[int, int, int, int]] = []

    class Limited(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                "test.limited",
                output_node=True,
                body=(
                    Input("cap", INT, default=2, automatic=Formula("4")),
                    Input("amount", INT, default=1, widget=NumberWidget(min=1, max=Formula("cap"))),
                    Input(
                        "grid",
                        INT,
                        default=1,
                        widget=NumberWidget(min=-5, max=12, step=4, offset=-3),
                    ),
                    Input("double", INT, default=0, automatic=Formula("amount * 2")),
                    Output("out", INT),
                ),
            )

        @classmethod
        def execute(cls, *, amount: int, grid: int, cap: int, double: int) -> Mapping[str, object]:
            received.append((amount, grid, cap, double))
            return cls.outputs(out=amount)

    schemas = build_schemas([Limited])

    def engine(on_event: object = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types([Limited]), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,  # type: ignore[arg-type]
        )

    async def scenario() -> None:
        store = HistoryStore(tmp_path / "numeric-history.sqlite")
        original = Graph({"n": GraphNode("test.limited", {"amount": 7, "grid": 3})})
        submitted = _submit(original, "corrected")
        expected = [
            {"nodeId": "n", "inputId": "amount", "value": 4, "source": "number-outside-limits"},
            {"nodeId": "n", "inputId": "grid", "value": 1, "source": "number-outside-limits"},
        ]
        async with TestClient(TestServer(create_app(engine, schemas, history=store))) as client:
            response = await client.post("/api/jobs", json=submitted)
            assert response.status == 202, await response.text()
            accepted = await response.json()
            assert accepted["correctedValues"] == expected
            job = await _finished(client, "corrected")
            assert job["state"] == "completed", job
            assert received == [(4, 1, 4, 8)]
            assert job["correctedValues"] == expected
            assert job["automaticValues"] == [
                {"nodeId": "n", "inputId": "cap", "value": 4, "source": "worked-out"},
                {"nodeId": "n", "inputId": "double", "value": 8, "source": "worked-out"},
            ]
            by_ref = await (await client.get(f"/api/jobs/by-ref/{job['jobRef']}")).json()
            assert by_ref["correctedValues"] == expected
            assert original.nodes["n"].inputs == {"amount": 7, "grid": 3}  # type: ignore[union-attr]
            assert submitted == _submit(original, "corrected")

            # Equivalent valid input reuses the effective value's cache entry,
            # without claiming a correction that did not occur in this run.
            await client.post(
                "/api/jobs",
                json=_submit(Graph({"n": GraphNode("test.limited", {"amount": 4})}), "valid"),
            )
            valid = await _finished(client, "valid")
            assert valid["state"] == "completed", valid
            assert "correctedValues" not in valid
            assert received == [(4, 1, 4, 8)]

            await client.post(
                "/api/jobs",
                json=_submit(Graph({"n": GraphNode("test.limited", {"amount": -2})}), "lower"),
            )
            lower = await _finished(client, "lower")
            assert lower["state"] == "completed", lower
            assert received == [(4, 1, 4, 8), (1, 1, 4, 2)]
            assert lower["correctedValues"] == [{**expected[0], "value": 1}]
            run_id = job["jobRef"]
            assert isinstance(run_id, str)

        reopened = HistoryStore(tmp_path / "numeric-history.sqlite")
        try:
            record = reopened.get("local", run_id)
            assert record is not None
            assert record.to_wire()["correctedValues"] == expected
        finally:
            reopened.close()

    run_scenario(scenario())


def test_circular_numeric_limit_is_unknown_and_other_limits_settle_first() -> None:
    received: list[tuple[int, int, int, int]] = []

    class DependentLimits(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                "test.dependent_limits",
                output_node=True,
                body=(
                    Input("amount", INT, default=1, widget=NumberWidget(max=Formula("cap"))),
                    Input("cap", INT, default=2, automatic=Formula("amount - 1")),
                    Input("dependent", INT, default=1, widget=NumberWidget(max=Formula("other"))),
                    Input("other", INT, default=1, widget=NumberWidget(max=4)),
                    Output("out", INT),
                ),
            )

        @classmethod
        def execute(
            cls, *, amount: int, cap: int, dependent: int, other: int
        ) -> Mapping[str, object]:
            received.append((amount, cap, dependent, other))
            return cls.outputs(out=amount)

    schemas = build_schemas([DependentLimits])

    def engine(on_event: object = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types([DependentLimits]), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,  # type: ignore[arg-type]
        )

    async def scenario() -> None:
        original = Graph(
            {"n": GraphNode("test.dependent_limits", {"amount": 7, "dependent": 6, "other": 7})}
        )
        async with TestClient(TestServer(create_app(engine, schemas))) as client:
            response = await client.post("/api/jobs", json=_submit(original, "circular"))
            assert response.status == 202, await response.text()
            job = await _finished(client, "circular")
            assert job["state"] == "completed", job
            assert received == [(7, 6, 4, 4)]
            assert job["automaticValues"] == [
                {"nodeId": "n", "inputId": "cap", "value": 6, "source": "worked-out"}
            ]
            assert job["correctedValues"] == [
                {"nodeId": "n", "inputId": name, "value": 4, "source": "number-outside-limits"}
                for name in ("dependent", "other")
            ]
            assert original.nodes["n"].inputs == {  # type: ignore[union-attr]
                "amount": 7,
                "dependent": 6,
                "other": 7,
            }

    run_scenario(scenario())


def test_only_nodes_of_the_requested_run_are_worked_out() -> None:
    graph = Graph(
        {
            "n": GraphNode("test.frame_rate", {}),
            "other": GraphNode("test.frame_rate", {}),
        }
    )
    resolved, records = asyncio.run(resolve_automatic(graph, ["n"], SCHEMAS))
    assert [(record.node_id, record.value) for record in records] == [("n", 24.0)]
    assert dict(resolved.nodes["n"].inputs) == {"fps": 24.0}  # type: ignore[union-attr]
    assert dict(resolved.nodes["other"].inputs) == {}  # type: ignore[union-attr]


def _looked_up(*rates: float) -> Graph:
    return Graph(
        {
            f"n{index}": GraphNode("test.looked_up", {"rate": rate})
            for index, rate in enumerate(rates)
        }
    )


def test_a_lookup_answer_is_used_and_a_failing_lookup_gives_the_default() -> None:
    calls: list[tuple[str, str, tuple[object, ...]]] = []

    async def answer(node_type: str, function: str, args: Sequence[object]) -> object:
        calls.append((node_type, function, tuple(args)))
        return source_rate(*args)  # type: ignore[arg-type]

    async def broken(node_type: str, function: str, args: Sequence[object]) -> object:
        raise RuntimeError("the pack's lookup raised")

    graph = _looked_up(40.0)
    _, records = asyncio.run(resolve_automatic(graph, ["n0"], SCHEMAS, call_lookup=answer))
    assert calls == [("test.looked_up", f"{source_rate.__module__}.source_rate", (40.0,))]
    assert [(record.value, record.worked_out) for record in records] == [(10.0, True)]
    _, records = asyncio.run(resolve_automatic(graph, ["n0"], SCHEMAS, call_lookup=broken))
    assert [(record.value, record.reason) for record in records] == [(24.0, "lookup-failed")]

    class HostTimeout(Exception):
        code = "lookup-timeout"

    async def timed_out(node_type: str, function: str, args: Sequence[object]) -> object:
        raise HostTimeout()

    _, records = asyncio.run(resolve_automatic(graph, ["n0"], SCHEMAS, call_lookup=timed_out))
    assert [(record.value, record.reason) for record in records] == [(24.0, "lookup-timeout")]


def test_a_lookup_argument_of_none_gives_the_default_without_a_call() -> None:
    calls: list[tuple[object, ...]] = []

    async def answer(node_type: str, function: str, args: Sequence[object]) -> object:
        calls.append(tuple(args))
        return 30.0

    # A lookup cannot take None, so the engine does not send it.
    graph = Graph({"n0": GraphNode("test.looked_up", {"rate": None})})
    _, records = asyncio.run(resolve_automatic(graph, ["n0"], SCHEMAS, call_lookup=answer))
    assert calls == []
    assert [(record.value, record.worked_out, record.reason) for record in records] == [
        (24.0, False, "argument-unknown")
    ]


def test_the_lookups_of_a_run_are_called_side_by_side_within_the_limit() -> None:
    async def slow(node_type: str, function: str, args: Sequence[object]) -> object:
        await asyncio.sleep(0.3 if args[0] == 40.0 else 30.0)
        return 1.5

    graph = _looked_up(40.0, 41.0, 40.0)
    started = time.monotonic()
    _, records = asyncio.run(
        resolve_automatic(graph, ["n0", "n1", "n2"], SCHEMAS, call_lookup=slow, time_limit=0.8)
    )
    elapsed = time.monotonic() - started
    assert 0.7 < elapsed < 1.5
    assert [(record.node_id, record.value, record.reason) for record in records] == [
        ("n0", 1.5, None),
        ("n1", 24.0, "lookup-timeout"),
        ("n2", 1.5, None),
    ]


def test_a_queued_run_asks_the_packs_lookup_and_records_its_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(dinkster.compose, "LOOKUP_TIME_LIMIT", 1.0)

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(DEV_PACK))
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                schema_owners=composition.schema_owners,
                choice_owners=composition.choice_owners,
                lookup_dispatch=composer.call_lookup,
                lookup_marks=composer.lookup_marks,
            )
            async with TestClient(TestServer(app)) as client:
                rate = "dev.automatic.lookup_rate"
                video_rate = "dev.automatic.video_rate"
                clip = {**VIDEO, "name": "clip_25fps.mp4"}
                graph = Graph(
                    {
                        "answered": GraphNode(rate, {"rate": 100.0, "delay": 0.0}),
                        "raised": GraphNode(rate, {"rate": -4.0, "delay": 0.0}),
                        "slow": GraphNode(rate, {"rate": 100.0, "delay": 3.0}),
                        "video": GraphNode(video_rate, {"video": clip}),
                        "no_video": GraphNode(video_rate, {}),
                    }
                )
                response = await client.post(
                    "/api/jobs", json=_submit(graph, "1", list(graph.nodes))
                )
                assert response.status == 202, await response.text()
                job = await _finished(client, "1")
                assert job["state"] == "completed", job
                assert job["automaticValues"] == [
                    {
                        "nodeId": "answered",
                        "inputId": "frame_rate",
                        "value": 25.0,
                        "source": "worked-out",
                    },
                    {
                        "nodeId": "no_video",
                        "inputId": "frame_rate",
                        "value": 24.0,
                        "source": "default",
                        "reason": "argument-unknown",
                    },
                    {
                        "nodeId": "raised",
                        "inputId": "frame_rate",
                        "value": 24.0,
                        "source": "default",
                        "reason": "lookup-failed",
                    },
                    {
                        "nodeId": "slow",
                        "inputId": "frame_rate",
                        "value": 24.0,
                        "source": "default",
                        "reason": "lookup-timeout",
                    },
                    {
                        "nodeId": "video",
                        "inputId": "frame_rate",
                        "value": 25.0,
                        "source": "worked-out",
                    },
                ]
                # The lookup route binds the same file the same way.
                asked = await client.post(
                    "/api/lookups",
                    json={
                        "nodeType": video_rate,
                        "function": "dinkster_nodes_dev.automatic.dev_video_rate",
                        "args": [clip],
                    },
                )
                assert await asked.json() == {
                    "answer": 25.0,
                    "typedAnswer": {"$float": 25},
                }
        finally:
            await composer.close()

    run_scenario(scenario())


VIDEO = {
    "digest": "blake3:" + "a" * 64,
    "name": "clip.mp4",
    "size": 12,
    "mediaType": "video/mp4",
    "virtualPath": "",
}


def test_a_lookup_that_takes_a_file_is_handed_the_picked_file() -> None:
    calls: list[tuple[object, ...]] = []

    async def answer(node_type: str, function: str, args: Sequence[object]) -> object:
        calls.append(tuple(args))
        return 30.0

    graph = Graph(
        {
            "a": GraphNode("test.from_file", {"video": VIDEO}),
            "b": GraphNode("test.from_file", {"video": dict(VIDEO)}),
            "none": GraphNode("test.from_file", {}),
            "wired": GraphNode("test.from_file", {"video": Link("a", "fps_out")}),
        }
    )
    resolved, records = asyncio.run(
        resolve_automatic(graph, list(graph.nodes), SCHEMAS, call_lookup=answer)
    )
    # Two nodes with the same file share one call, handed the file's wire form.
    assert calls == [(VIDEO,)]
    assert [(record.node_id, record.value, record.reason) for record in records] == [
        ("a", 30.0, None),
        ("b", 30.0, None),
        ("none", 24.0, "argument-unknown"),
        ("wired", 24.0, "argument-unknown"),
    ]
    assert dict(resolved.nodes["none"].inputs) == {"fps": 24.0}  # type: ignore[union-attr]


def test_a_formula_refused_while_working_out_gives_the_default_and_the_run_goes_on() -> None:
    graph = Graph({"n": GraphNode("test.unreadable", {})})
    _, records = asyncio.run(resolve_automatic(graph, ["n"], SCHEMAS))
    assert [(record.value, record.reason) for record in records] == [(24.0, "failed")]

    async def scenario() -> None:
        async with TestClient(TestServer(create_app(make_engine, SCHEMAS))) as client:
            response = await client.post("/api/jobs", json=_submit(graph, "1"))
            assert response.status == 202, await response.text()
            job = await _finished(client, "1")
            assert job["state"] == "completed", job
            assert job["automaticValues"] == [
                {
                    "nodeId": "n",
                    "inputId": "fps",
                    "value": 24.0,
                    "source": "default",
                    "reason": "failed",
                }
            ]

    run_scenario(scenario())
