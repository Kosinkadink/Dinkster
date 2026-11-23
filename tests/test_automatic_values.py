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
from dinkster_server import STATE_KEY, create_app
from dinkster_server.history import HistoryStore, record_from_job
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

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


NODES = [FrameRate, Number, LookedUp]
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
                Input("rate", FLOAT, default=1.0),
                Input("fps", FLOAT, default=1.0, automatic=Lookup(rate_name, args=("rate",))),
            ),
            "answers str where float or int is used",
        ),
    ],
)
def test_a_pack_with_an_automatic_field_it_cannot_hold_is_refused(
    rows: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NodeSchema("test.refused", body=rows())  # type: ignore[operator]


def _submit(graph: Graph, job_id: str) -> dict[str, object]:
    return {"clientId": "c", "jobId": job_id, "graph": graph_to_wire(graph), "targets": ["n"]}


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
