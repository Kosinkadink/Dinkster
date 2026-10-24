"""Run control frees admission while waiting and cannot escape its invocation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import GovernorReservationService, MemoryGovernor, ReservationRequest
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    RunExecutionContext,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import TypeRegistry, Value, is_absent, register_core_types
from dinkster_workers import GovernedWorker, InProcessWorker

INT = TypeExpr.concrete("core.int")


class Waiter(Node):
    saved: RunExecutionContext | None = None
    started: asyncio.Event | None = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "control.wait", outputs=(OutputSpec("value", INT),), execution_context=True
        )

    @classmethod
    async def execute(cls, execution_context: RunExecutionContext) -> Mapping[str, Value]:
        cls.saved = execution_context
        if cls.started is not None:
            cls.started.set()
        return await execution_context.await_run_value(execution_context.node_id)


class Publisher(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "control.publish",
            inputs=(InputSpec("value", INT),),
            outputs=(),
            execution_context=True,
        )

    @classmethod
    async def execute(cls, value: int, execution_context: RunExecutionContext) -> dict[str, object]:
        with pytest.raises(ValueError, match="only expanding"):
            await execution_context.inhibit(["waiter"])
        with pytest.raises(ValueError, match="only expanding"):
            await execution_context.override_node("waiter", {})
        with pytest.raises(ValueError, match="does not match"):
            await execution_context.publish_run_value("waiter", {})
        await execution_context.publish_run_value("waiter", execution_context.inputs)
        with pytest.raises(ValueError, match="already published"):
            await execution_context.publish_run_value("waiter", execution_context.inputs)
        return {}


class Passthrough(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "control.pass", inputs=(InputSpec("value", INT),), outputs=(OutputSpec("value", INT),)
        )

    @classmethod
    def execute(cls, value: int) -> dict[str, object]:
        return {"value": value}


class Controller(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "control.edit",
            outputs=(OutputSpec("value", INT),),
            execution_context=True,
            may_expand_graph=True,
        )

    @classmethod
    async def execute(cls, execution_context: RunExecutionContext) -> dict[str, object]:
        before = cast(Any, await execution_context.get_graph())
        with pytest.raises(TypeError):
            before["nodes"]["victim"]["inputs"]["value"] = 90
        with pytest.raises(ValueError, match="started node"):
            await execution_context.override_node("controller", {})
        with pytest.raises(ValueError, match="invalid run override"):
            await execution_context.override_node(
                "victim", {"value": {"$link": {"node": "victim", "output": "value"}}}
            )
        await execution_context.inhibit(["muted"])
        await execution_context.override_node(
            "victim", {"value": {"$link": {"node": "source", "output": "value"}}}
        )
        after = cast(Any, await execution_context.get_graph())
        assert after["nodes"]["victim"]["inputs"]["value"] == {
            "$link": {"node": "source", "output": "value"}
        }
        assert (
            before["nodes"]["victim"]["inputs"]["value"]
            != after["nodes"]["victim"]["inputs"]["value"]
        )
        return {"value": 7}


NODES = [Waiter, Publisher, Passthrough, Controller]


def make_engine(governed: bool = False) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    worker = InProcessWorker(build_node_types(NODES), registry)
    if governed:
        worker = GovernedWorker(
            worker,
            planner=lambda _: (ReservationRequest("ram", 1),),
            reservations=GovernorReservationService(MemoryGovernor({"ram": 1}), timeout=1),
        )
    return Engine(
        schemas=build_schemas(NODES),
        worker=worker,
        cache=MemoryLRUCache(),
        registry=registry,
        max_concurrency=1,
    )


@pytest.mark.parametrize("governed", [False, True])
def test_run_value_wait_releases_execution_and_memory_admission(governed: bool) -> None:
    async def scenario() -> None:
        engine = make_engine(governed)
        for seed in (19, -43):
            graph = Graph(
                {
                    "waiter": GraphNode("control.wait"),
                    "publisher": GraphNode("control.publish", {"value": seed}),
                }
            )
            result = await asyncio.wait_for(engine.run(graph, ["waiter", "publisher"]), 3)
            assert result.outputs["waiter"]["value"].resolve() == seed
            assert not result.cached
            assert Waiter.saved is not None
            with pytest.raises(RuntimeError, match="no longer active"):
                await Waiter.saved.publish_run_value("waiter", {})

    asyncio.run(scenario())


def test_control_graph_edits_are_validated_immutable_and_run_local() -> None:
    async def scenario() -> None:
        graph = Graph(
            {
                "controller": GraphNode("control.edit"),
                "source": GraphNode("control.pass", {"value": 45}),
                "victim": GraphNode("control.pass", {"value": Link("controller", "value")}),
                "muted": GraphNode("control.pass", {"value": Link("controller", "value")}),
            }
        )
        engine = make_engine()
        for _ in range(2):
            result = await engine.run(graph, ["victim", "muted"])
            assert result.outputs["victim"]["value"].resolve() == 45
            assert is_absent(result.outputs["muted"]["value"])
            assert "muted" in result.skipped
            assert graph.nodes["victim"].inputs["value"] == Link("controller", "value")

    asyncio.run(scenario())


def test_cancelling_waiter_releases_admission_for_the_next_run() -> None:
    async def scenario() -> None:
        engine = make_engine(True)
        Waiter.started = asyncio.Event()
        try:
            graph = Graph({"waiter": GraphNode("control.wait")})
            task = asyncio.create_task(engine.run(graph, ["waiter"]))
            await asyncio.wait_for(Waiter.started.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            result = await asyncio.wait_for(
                engine.run(Graph({"pass": GraphNode("control.pass", {"value": 31})}), ["pass"]), 3
            )
            assert result.outputs["pass"]["value"].resolve() == 31
        finally:
            Waiter.started = None

    asyncio.run(scenario())


def test_execution_context_schema_is_explicit_and_not_a_socket() -> None:
    schema = Waiter.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    with pytest.raises(ValueError, match="reserved input"):
        NodeSchema("bad", inputs=(InputSpec("execution_context", INT),))
