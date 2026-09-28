"""Produced envelopes outlive consumers, not the entire execution plan."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, ExecutionError
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_protocol import Invocation, InvocationResult, LazyStatusResult, NodeError
from dinkster_schema import InputSpec, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    ResourcePins,
    TypeRegistry,
    Value,
    ValueMeta,
    make_absent_value,
    register_core_types,
)

INT = TypeExpr.concrete("core.int")
SCHEMAS = {
    "step": NodeSchema(
        node_type="step",
        inputs=(InputSpec("a", INT, default=0), InputSpec("b", INT, default=0)),
        outputs=(OutputSpec("out", INT),),
        io_bound=True,
    ),
    "lazy": NodeSchema(
        node_type="lazy",
        inputs=(
            InputSpec("a", INT),
            InputSpec("b", INT, lazy=True),
            InputSpec("demand", TypeExpr.concrete("core.boolean")),
        ),
        outputs=(OutputSpec("out", INT),),
        io_bound=True,
    ),
    "condition": NodeSchema(
        node_type="condition",
        inputs=(InputSpec("a", INT),),
        outputs=(OutputSpec("out", TypeExpr.concrete("core.boolean")),),
    ),
}


class NoCache:
    async def get(self, key: str) -> Mapping[str, Value] | None:
        return None

    async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
        pass


class LifetimeWorker:
    """Use real envelopes without retaining invocations or payloads in a spy."""

    def __init__(self, registry, inspect) -> None:
        self.registry = registry
        self.inspect = inspect
        self.refs: dict[str, weakref.ReferenceType[Value]] = {}
        self.calls: list[str] = []

    async def prepare(self, node_types) -> None:
        pass

    async def check_lazy_status(self, invocation) -> LazyStatusResult:
        inputs = invocation.available_inputs
        return LazyStatusResult(
            requested_inputs=("b",) if inputs["demand"].resolve() and "b" not in inputs else ()
        )

    async def invoke(self, invocation: Invocation, on_event=None) -> InvocationResult:
        self.calls.append(invocation.node_id)
        result = await self.inspect(invocation)
        if result is None:
            result = self.registry.wrap(
                "core.int",
                sum(
                    cast(int, invocation.inputs[name].resolve())
                    for name in ("a", "b")
                    if name in invocation.inputs
                )
                + 1,
            )
        self.refs[invocation.node_id] = weakref.ref(result)
        return InvocationResult(outputs={"out": result})


def make_engine(inspect, cache=None, pins=None):
    registry = TypeRegistry()
    register_core_types(registry)
    worker = LifetimeWorker(registry, inspect)
    events: list[EngineEvent] = []
    engine = Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=worker,
        cache=NoCache() if cache is None else cache,
        on_event=events.append,
        pins=pins,
        explain_misses=True,
    )
    return engine, worker, events


@pytest.mark.parametrize("retain_source", [False, True])
@pytest.mark.parametrize("cache_hit", [False, True])
def test_branch_consumers_and_targets_release_after_completion(retain_source, cache_hit) -> None:
    async def scenario() -> None:
        observed_fast = asyncio.Event()
        slow_started = asyncio.Event()

        async def inspect(invocation) -> None:
            if invocation.node_id == "slow":
                slow_started.set()
                await observed_fast.wait()
                assert worker.refs["source"]() is invocation.inputs["a"]
            elif invocation.node_id == "witness":
                await slow_started.wait()
                assert worker.refs["source"]() is not None
                observed_fast.set()
            elif invocation.node_id == "join":
                assert (worker.refs["source"]() is not None) == retain_source
                assert worker.refs["fast"]() is None

        cache = MemoryLRUCache(max_entries=1) if cache_hit else None
        engine, worker, events = make_engine(inspect, cache)
        graph = Graph(
            nodes={
                "source": GraphNode("step", {"a": 10}),
                "fast": GraphNode("step", {"a": Link("source", "out"), "b": 20}),
                "slow": GraphNode("step", {"a": Link("source", "out"), "b": 30}),
                "witness": GraphNode("step", {"a": Link("fast", "out")}),
                "join": GraphNode("step", {"a": Link("witness", "out"), "b": Link("slow", "out")}),
                "unreachable": GraphNode("step", {"a": Link("source", "out")}),
            }
        )
        if cache_hit:
            await engine.run(graph, ["source"])
        result = await engine.run(graph, ["join", "source"] if retain_source else ["join"])
        assert result.outputs["join"]["out"].resolve() == 76
        assert "unreachable" not in worker.calls
        assert ("source" in result.cached) == cache_hit
        assert worker.calls.count("source") == 1
        assert (worker.refs["source"]() is not None) == retain_source
        assert events  # Retained events and miss explanations must not retain envelopes.

    asyncio.run(scenario())


@pytest.mark.parametrize("demand", [False, True])
@pytest.mark.parametrize("cache_hit", [False, True])
def test_lazy_possible_consumers_keep_shared_ancestor_until_decision(demand, cache_hit) -> None:
    async def scenario() -> None:
        async def inspect(invocation) -> None:
            if invocation.node_id == "gate":
                # The dormant lazy branch may still need this completed producer.
                assert worker.refs["source"]() is not None
            elif invocation.node_id == "candidate":
                assert invocation.inputs["a"] is worker.refs["source"]()
            elif invocation.node_id == "tail":
                assert worker.refs["source"]() is None
                assert worker.refs["ordinary"]() is None
                if demand:
                    assert worker.refs["candidate"]() is None

        cache = MemoryLRUCache(max_entries=1) if cache_hit else None
        engine, worker, _ = make_engine(inspect, cache)
        graph = Graph(
            nodes={
                "source": GraphNode("step", {"a": 10}),
                "ordinary": GraphNode("step", {"a": Link("source", "out")}),
                "gate": GraphNode("step", {"a": Link("ordinary", "out")}),
                "candidate": GraphNode("step", {"a": Link("source", "out"), "b": 20}),
                "choose": GraphNode(
                    "lazy",
                    {
                        "a": Link("gate", "out"),
                        "b": Link("candidate", "out"),
                        "demand": demand,
                    },
                ),
                "tail": GraphNode("step", {"a": Link("choose", "out")}),
            }
        )
        if cache_hit:
            await engine.run(graph, ["source"])
        result = await engine.run(graph, ["tail"])
        assert result.outputs["tail"]["out"].resolve() == (47 if demand else 15)
        assert ("candidate" in worker.calls) == demand
        assert ("source" in result.cached) == cache_hit
        assert worker.calls.count("source") == 1

    asyncio.run(scenario())


def test_repeated_input_links_count_one_consumer() -> None:
    async def scenario() -> None:
        async def inspect(invocation) -> None:
            if invocation.node_id == "twice":
                assert invocation.inputs["a"] is invocation.inputs["b"]
            elif invocation.node_id == "tail":
                assert worker.refs["source"]() is None

        engine, worker, _ = make_engine(inspect)
        result = await engine.run(
            Graph(
                nodes={
                    "source": GraphNode("step", {"a": 10}),
                    "twice": GraphNode(
                        "step", {"a": Link("source", "out"), "b": Link("source", "out")}
                    ),
                    "tail": GraphNode("step", {"a": Link("twice", "out")}),
                }
            ),
            ["tail"],
        )
        assert result.outputs["tail"]["out"].resolve() == 24

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["map", "fold"])
def test_regions_release_intermediates_but_keep_exports_and_state(kind) -> None:
    async def scenario() -> None:
        async def inspect(invocation) -> None:
            if invocation.node_id.endswith("/export"):
                prefix = invocation.node_id.removesuffix("export")
                assert worker.refs[prefix + "source"]() is None
            if invocation.node_id == "region[2]/source" and kind == "fold":
                # The previous iteration's ports must not retain superseded state.
                assert worker.refs["region[0]/export"]() is None

        engine, worker, _ = make_engine(inspect)
        state = kind == "fold"
        region = RegionNode(
            kind=kind,
            ports={"item": INT, **({"total": INT} if state else {})},
            inputs={"item": [1, 2, 3], **({"total": 0} if state else {})},
            element_ports=("item",),
            state_ports=("total",) if state else (),
            body=Graph(
                nodes={
                    "source": GraphNode(
                        "step",
                        {
                            "a": Link(PORTS_NODE_ID, "item"),
                            **({"b": Link(PORTS_NODE_ID, "total")} if state else {}),
                        },
                    ),
                    "middle": GraphNode("step", {"a": Link("source", "out")}),
                    "export": GraphNode("step", {"a": Link("middle", "out")}),
                }
            ),
            outputs={
                "total" if state else "items": RegionOutput(
                    Link("export", "out"), "state" if state else "gather"
                ),
                # An exported node may itself feed another body node.
                **({} if state else {"middles": RegionOutput(Link("middle", "out"), "gather")}),
            },
        )
        result = await engine.run(Graph(nodes={"region": region}), ["region"])
        if state:
            assert result.outputs["region"]["total"].resolve() == 15
            assert worker.refs["region[1]/export"]() is None
        else:
            assert result.outputs["region"]["items"].resolve() == [4, 5, 6]
            assert result.outputs["region"]["middles"].resolve() == [3, 4, 5]

    asyncio.run(scenario())


@pytest.mark.parametrize("lazy", [False, True])
def test_failure_cancels_consumers_without_dropping_live_inputs(lazy) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def inspect(invocation) -> None:
            if invocation.node_id == "slow":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    assert invocation.inputs["a"] is worker.refs["source"]()
                    cancelled.set()
            elif invocation.node_id == "fail":
                await started.wait()
                raise ExecutionError(NodeError("fail", "step", "expected failure"))

        engine, worker, _ = make_engine(inspect)
        graph = Graph(
            nodes={
                "source": GraphNode("step", {"a": 10}),
                "slow": GraphNode("step", {"a": Link("source", "out"), "b": 20}),
                "fail": GraphNode("step", {"a": Link("source", "out"), "b": 30}),
                **(
                    {
                        "choose": GraphNode(
                            "lazy",
                            {
                                "a": Link("slow", "out"),
                                "b": Link("fail", "out"),
                                "demand": True,
                            },
                        )
                    }
                    if lazy
                    else {}
                ),
            }
        )
        with pytest.raises(ExecutionError, match="expected failure"):
            await engine.run(graph, ["choose", "fail"] if lazy else ["slow", "fail"])
        assert cancelled.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True])
def test_produced_release_does_not_shorten_resource_pin_lifetime(fail) -> None:
    async def scenario() -> None:
        pins = ResourcePins()

        async def inspect(invocation):
            if invocation.node_id == "source":
                return replace(
                    worker.registry.wrap("core.int", 10),
                    meta=ValueMeta({RESOURCE_ID_META_KEY: "resident"}),
                )
            if invocation.node_id == "tail":
                assert worker.refs["source"]() is None
                assert pins.pinned("resident")
                assert not pins.condemn("resident")
                if fail:
                    raise ExecutionError(NodeError("tail", "step", "expected failure"))

        engine, worker, _ = make_engine(inspect, pins=pins)
        graph = Graph(
            nodes={
                "source": GraphNode("step", {}),
                "middle": GraphNode("step", {"a": Link("source", "out")}),
                "tail": GraphNode("step", {"a": Link("middle", "out")}),
            }
        )
        if fail:
            with pytest.raises(ExecutionError, match="expected failure"):
                await engine.run(graph, ["tail"])
        else:
            assert (await engine.run(graph, ["tail"])).outputs["tail"]["out"].resolve() == 12
        assert not pins.pinned("resident")
        assert pins.condemn("resident")

    asyncio.run(scenario())


def test_while_releases_previous_iteration_ports_and_keeps_continue_export() -> None:
    async def scenario() -> None:
        async def inspect(invocation):
            if invocation.node_id.endswith("/continue"):
                return worker.registry.wrap("core.boolean", invocation.inputs["a"].resolve() < 3)
            if invocation.node_id == "region[2]/step":
                assert worker.refs["region[0]/step"]() is None
                assert worker.refs["region[1]/continue"]() is None

        engine, worker, _ = make_engine(inspect)
        region = RegionNode(
            kind="while",
            ports={"count": INT},
            inputs={"count": 0},
            state_ports=("count",),
            body=Graph(
                nodes={
                    "step": GraphNode("step", {"a": Link(PORTS_NODE_ID, "count")}),
                    "continue": GraphNode("condition", {"a": Link("step", "out")}),
                }
            ),
            outputs={"count": RegionOutput(Link("step", "out"), "state")},
            continue_source=Link("continue", "out"),
            max_iterations=3,
        )
        result = await engine.run(Graph(nodes={"region": region}), ["region"])
        assert result.outputs["region"]["count"].resolve() == 3

    asyncio.run(scenario())


def test_failed_region_does_not_retain_other_exports_in_scheduler_tracebacks() -> None:
    async def scenario() -> None:
        source_started = asyncio.Event()

        async def inspect(invocation) -> None:
            if invocation.node_id.endswith("/source"):
                source_started.set()
            elif invocation.node_id.endswith("/fail"):
                await source_started.wait()
                assert worker.refs["region[0]/source"]() is not None
                raise ExecutionError(NodeError(invocation.node_id, "step", "expected failure"))

        engine, worker, _ = make_engine(inspect)
        region = RegionNode(
            kind="map",
            ports={"item": INT},
            inputs={"item": [1]},
            element_ports=("item",),
            body=Graph(
                nodes={
                    "source": GraphNode("step", {"a": Link(PORTS_NODE_ID, "item")}),
                    "fail": GraphNode("step", {"a": 20}),
                }
            ),
            outputs={
                "items": RegionOutput(Link("source", "out"), "gather"),
                "failure": RegionOutput(Link("fail", "out"), "gather"),
            },
        )
        with pytest.raises(ExecutionError, match="expected failure") as error:
            await engine.run(Graph(nodes={"region": region}), ["region"])
        assert error.value.__traceback__ is not None
        assert worker.refs["region[0]/source"]() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["absent", "non-boolean", "limit", "worker"])
def test_while_errors_release_iteration_exports_and_prior_state(failure: str) -> None:
    async def scenario() -> None:
        async def inspect(invocation):
            if invocation.node_id.endswith("/continue"):
                if invocation.node_id.startswith("region[1]/"):
                    if failure == "absent":
                        return make_absent_value(origin=invocation.node_id)
                    if failure == "non-boolean":
                        return worker.registry.wrap("core.int", 1)
                    if failure == "worker":
                        raise ExecutionError(NodeError(invocation.node_id, "condition", "failure"))
                return worker.registry.wrap("core.boolean", True)

        engine, worker, _ = make_engine(inspect)
        region = RegionNode(
            kind="while",
            ports={"count": INT},
            inputs={"count": 0},
            state_ports=("count",),
            body=Graph(
                nodes={
                    "step": GraphNode("step", {"a": Link(PORTS_NODE_ID, "count")}),
                    "continue": GraphNode("condition", {"a": Link("step", "out")}),
                }
            ),
            outputs={
                "count": RegionOutput(Link("step", "out"), "state"),
                "items": RegionOutput(Link("step", "out"), "gather"),
            },
            continue_source=Link("continue", "out"),
            max_iterations=2,
        )
        with pytest.raises(ExecutionError) as error:
            await engine.run(Graph(nodes={"region": region}), ["region"])
        assert error.value.__traceback__ is not None
        assert "region[1]/step" in worker.refs
        assert all(ref() is None for ref in worker.refs.values())

    asyncio.run(scenario())
