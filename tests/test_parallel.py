"""Parallel execution (hazard H12): ready-set scheduling, concurrency bounds,
engine-wide single-flight, and schedule-independence of results."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import (
    Engine,
    ExecutionError,
    GraphValidationError,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_protocol import LazyStatusInvocation, LazyStatusResult
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

STRING = TypeExpr.concrete("core.string")


class Sleeper(Node):
    """Async node that records how many of itself run concurrently."""

    active = 0
    peak = 0
    log: list[str] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.sleeper",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            io_bound=True,  # a wait, not compute: overlaps freely by default
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        Sleeper.active += 1
        Sleeper.peak = max(Sleeper.peak, Sleeper.active)
        Sleeper.log.append(tag)
        await asyncio.sleep(0.02)
        Sleeper.active -= 1
        return cls.outputs(out=tag)

    @classmethod
    def reset(cls) -> None:
        cls.active = 0
        cls.peak = 0
        cls.log = []


class SlowFail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.slow_fail",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        await asyncio.sleep(0.01)
        raise ValueError(f"boom: {tag}")


class Effect(Node):
    """Non-idempotent: never cached, so never coalesced."""

    invocations = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.effect",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        Effect.invocations += 1
        await asyncio.sleep(0.01)
        return cls.outputs(out=tag)


class LazySleeper(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.lazy-sleeper",
            inputs=(
                InputSpec("tag", STRING),
                InputSpec("trigger", STRING, required=False, lazy=True),
            ),
            outputs=(OutputSpec("out", STRING),),
            io_bound=True,
        )

    @classmethod
    def check_lazy_status(cls, **_inputs: object) -> tuple[()]:
        return ()

    @classmethod
    async def execute(cls, *, tag: str, **_inputs: object) -> Mapping[str, object]:
        await asyncio.sleep(0.01)
        return cls.outputs(out=tag)


class CountingWorker:
    """Delegates to InProcessWorker, counting real invocations."""

    def __init__(self, inner: InProcessWorker) -> None:
        self._inner = inner
        self.invocations = 0

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._inner.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        self.invocations += 1
        return await self._inner.invoke(invocation)


EXTRA_NODES: list[type[Node]] = [Sleeper, SlowFail, Effect, LazySleeper]


def make_engine(max_concurrency: int | None = None) -> tuple[Engine, CountingWorker]:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    nodes = list(SCAFFOLD_NODES) + EXTRA_NODES
    worker = CountingWorker(InProcessWorker(build_node_types(nodes), registry))
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
        max_concurrency=max_concurrency,
    )
    return engine, worker


def sleeper_fanout(*tags: str) -> Graph:
    return Graph(nodes={tag: GraphNode("test.sleeper", {"tag": tag}) for tag in tags})


def test_independent_nodes_execute_concurrently() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, worker = make_engine()
        graph = sleeper_fanout("a", "b", "c")
        result = await engine.run(graph, ["a", "b", "c"])
        assert {r for r in result.executed} == {"a", "b", "c"}
        assert Sleeper.peak == 3  # all three overlapped
        assert worker.invocations == 3
        for tag in "abc":
            assert result.outputs[tag]["out"].resolve() == tag

    asyncio.run(scenario())


def test_max_concurrency_bounds_parallelism() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, _ = make_engine(max_concurrency=1)
        await engine.run(sleeper_fanout("a", "b", "c"), ["a", "b", "c"])
        assert Sleeper.peak == 1  # serialized by the slot budget

    asyncio.run(scenario())


def test_dependencies_are_respected_under_parallelism() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, _ = make_engine()
        # Diamond: root -> left/right (parallel) -> join.
        graph = Graph(
            nodes={
                "root": GraphNode("test.sleeper", {"tag": "root"}),
                "left": GraphNode("test.sleeper", {"tag": Link("root", "out")}),
                "right": GraphNode(
                    "std.string.concat",
                    {"a": Link("root", "out"), "b": "!", "separator": ""},
                ),
                "join": GraphNode(
                    "std.string.concat",
                    {"a": Link("left", "out"), "b": Link("right", "text"), "separator": " "},
                ),
            }
        )
        result = await engine.run(graph, ["join"])
        assert result.outputs["join"]["text"].resolve() == "root root!"
        assert result.executed[-1] == "join"  # join completes last
        assert result.executed[0] == "root"  # root completes first

    asyncio.run(scenario())


def test_parallel_and_serial_schedules_produce_identical_results() -> None:
    def diamond() -> Graph:
        return Graph(
            nodes={
                "root": GraphNode("std.string.concat", {"a": "x", "b": "y", "separator": ""}),
                "l": GraphNode(
                    "std.string.concat",
                    {"a": Link("root", "text"), "b": "l", "separator": "-"},
                ),
                "r": GraphNode(
                    "std.string.concat",
                    {"a": Link("root", "text"), "b": "r", "separator": "-"},
                ),
                "join": GraphNode(
                    "std.string.concat",
                    {"a": Link("l", "text"), "b": Link("r", "text"), "separator": " "},
                ),
            }
        )

    async def run_with(bound: int | None) -> str:
        engine, _ = make_engine(max_concurrency=bound)
        result = await engine.run(diamond(), ["join"])
        text = result.outputs["join"]["text"].resolve()
        assert isinstance(text, str)
        return text

    assert asyncio.run(run_with(None)) == asyncio.run(run_with(1)) == "xy-l xy-r"


def test_single_flight_within_one_run() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, worker = make_engine()
        # Two nodes, same type, same inputs -> same cache key.
        graph = Graph(
            nodes={
                "n1": GraphNode("test.sleeper", {"tag": "same"}),
                "n2": GraphNode("test.sleeper", {"tag": "same"}),
            }
        )
        result = await engine.run(graph, ["n1", "n2"])
        assert worker.invocations == 1  # computed once
        assert len(result.executed) == 1
        assert len(result.cached) == 1  # the coalesced twin
        assert result.outputs["n1"]["out"].resolve() == "same"
        assert result.outputs["n2"]["out"].resolve() == "same"

    asyncio.run(scenario())


def test_single_flight_across_concurrent_runs() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, worker = make_engine()
        g1 = sleeper_fanout("shared")
        g2 = sleeper_fanout("shared")
        r1, r2 = await asyncio.gather(engine.run(g1, ["shared"]), engine.run(g2, ["shared"]))
        assert worker.invocations == 1  # both workflows consumed one execution
        assert r1.outputs["shared"]["out"].resolve() == "shared"
        assert r2.outputs["shared"]["out"].resolve() == "shared"
        assert sorted([len(r1.executed) + len(r2.executed), len(r1.cached) + len(r2.cached)]) == [
            1,
            1,
        ]

    asyncio.run(scenario())


def test_non_idempotent_nodes_are_never_coalesced() -> None:
    async def scenario() -> None:
        Effect.invocations = 0
        engine, worker = make_engine()
        g1 = Graph(nodes={"e": GraphNode("test.effect", {"tag": "t"})})
        g2 = Graph(nodes={"e": GraphNode("test.effect", {"tag": "t"})})
        r1, r2 = await asyncio.gather(engine.run(g1, ["e"]), engine.run(g2, ["e"]))
        assert Effect.invocations == 2  # identical inputs, still both ran
        assert r1.executed == ("e",)
        assert r2.executed == ("e",)

    asyncio.run(scenario())


def test_failure_fails_the_run_with_the_node_error() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, _ = make_engine()
        graph = Graph(
            nodes={
                "ok": GraphNode("test.sleeper", {"tag": "ok"}),
                "bad": GraphNode("test.slow_fail", {"tag": "b"}),
            }
        )
        with pytest.raises(ExecutionError) as excinfo:
            await engine.run(graph, ["ok", "bad"])
        assert excinfo.value.error.node_id == "bad"
        assert "boom" in excinfo.value.error.message

    asyncio.run(scenario())


def test_coalesced_runs_fail_identically() -> None:
    async def scenario() -> None:
        engine, worker = make_engine()
        g1 = Graph(nodes={"bad": GraphNode("test.slow_fail", {"tag": "x"})})
        g2 = Graph(nodes={"bad": GraphNode("test.slow_fail", {"tag": "x"})})
        results = await asyncio.gather(
            engine.run(g1, ["bad"]), engine.run(g2, ["bad"]), return_exceptions=True
        )
        assert worker.invocations == 1  # one execution, both runs see its failure
        for outcome in results:
            assert isinstance(outcome, ExecutionError)
            assert "boom" in outcome.error.message

    asyncio.run(scenario())


def test_failed_computation_is_not_poisoned_for_later_runs() -> None:
    async def scenario() -> None:
        engine, worker = make_engine()
        graph = Graph(nodes={"bad": GraphNode("test.slow_fail", {"tag": "x"})})
        with pytest.raises(ExecutionError):
            await engine.run(graph, ["bad"])
        # The inflight entry is gone; a retry executes again (and fails again).
        with pytest.raises(ExecutionError):
            await engine.run(graph, ["bad"])
        assert worker.invocations == 2

    asyncio.run(scenario())


def test_waiter_takes_over_when_owning_run_is_cancelled() -> None:
    """Run 1 owns the shared computation but dies (sibling failure cancels
    it mid-flight); run 2, waiting on the same key, must take over ownership
    and complete instead of hanging or failing with run 1."""

    async def scenario() -> None:
        Sleeper.reset()
        engine, worker = make_engine()
        doomed = Graph(
            nodes={
                "slow": GraphNode("test.sleeper", {"tag": "takeover"}),
                "bad": GraphNode("test.slow_fail", {"tag": "x"}),
            }
        )
        survivor = Graph(nodes={"slow": GraphNode("test.sleeper", {"tag": "takeover"})})

        r1, r2 = await asyncio.gather(
            engine.run(doomed, ["slow", "bad"]),
            engine.run(survivor, ["slow"]),
            return_exceptions=True,
        )
        assert isinstance(r1, ExecutionError)  # its own node failed
        assert not isinstance(r2, BaseException)
        assert r2.outputs["slow"]["out"].resolve() == "takeover"
        assert not engine._inflight  # no leaked inflight entries

    asyncio.run(scenario())


class GatedWorker:
    """Blocks the Nth invocation on an event; later ones pass through.
    Makes ownership/waiter states deterministic in cancellation tests."""

    def __init__(self, inner: InProcessWorker, gate_first: int = 1) -> None:
        self._inner = inner
        self._gate_first = gate_first
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.second_lazy_hook = asyncio.Event()
        self.invocations = 0
        self.lazy_hooks = 0

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._inner.prepare(node_types)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        result = await self._inner.check_lazy_status(invocation, on_event)
        self.lazy_hooks += 1
        if self.lazy_hooks == 2:
            self.second_lazy_hook.set()
        return result

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        self.invocations += 1
        if self.invocations <= self._gate_first:
            self.entered.set()
            await self.gate.wait()
        return await self._inner.invoke(invocation)


def make_gated_engine(gate_first: int = 1) -> tuple[Engine, GatedWorker]:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    nodes = list(SCAFFOLD_NODES) + EXTRA_NODES
    worker = GatedWorker(InProcessWorker(build_node_types(nodes), registry), gate_first)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
    )
    return engine, worker


def test_cancelling_owner_and_waiter_cancels_both_cleanly() -> None:
    """Regression: a waiter whose own run is being cancelled must not
    mistake the owner's cancellation for a takeover signal - it must stay
    cancelled, register no replacement computation, and leak nothing."""

    async def scenario() -> None:
        engine, worker = make_gated_engine()
        graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "both"})})

        owner = asyncio.create_task(engine.run(graph, ["n"]))
        await asyncio.wait_for(worker.entered.wait(), 1)  # owner holds the key
        waiter = asyncio.create_task(engine.run(graph, ["n"]))
        for _ in range(5):  # let the waiter reach the shield
            await asyncio.sleep(0)

        owner.cancel()
        waiter.cancel()
        results = await asyncio.wait_for(asyncio.gather(owner, waiter, return_exceptions=True), 1)
        assert all(isinstance(r, asyncio.CancelledError) for r in results)
        assert not engine._inflight  # nothing leaked
        assert worker.invocations == 1  # no replacement computation started

    asyncio.run(scenario())


def test_waiter_survives_owner_only_cancellation() -> None:
    """Owner is cancelled while blocked in the worker; a healthy waiter must
    take over and complete (only the first invocation is gated)."""

    async def scenario() -> None:
        engine, worker = make_gated_engine(gate_first=1)
        graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "takeover2"})})

        owner = asyncio.create_task(engine.run(graph, ["n"]))
        await asyncio.wait_for(worker.entered.wait(), 1)
        waiter = asyncio.create_task(engine.run(graph, ["n"]))
        for _ in range(5):
            await asyncio.sleep(0)

        owner.cancel()
        result = await asyncio.wait_for(waiter, 1)
        assert result.outputs["n"]["out"].resolve() == "takeover2"
        assert result.executed == ("n",)  # the waiter became the owner
        assert not engine._inflight
        with pytest.raises(asyncio.CancelledError):
            await owner

    asyncio.run(scenario())


def test_lazy_waiter_takes_over_after_cancelled_owner_body() -> None:
    """Lazy hooks run independently, but the finalized idempotent body uses
    the ordinary cancellation/takeover single-flight contract."""

    async def scenario() -> None:
        engine, worker = make_gated_engine(gate_first=1)
        graph = Graph(nodes={"n": GraphNode("test.lazy-sleeper", {"tag": "lazy"})})

        owner = asyncio.create_task(engine.run(graph, ["n"]))
        await asyncio.wait_for(worker.entered.wait(), 1)
        waiter = asyncio.create_task(engine.run(graph, ["n"]))
        await asyncio.wait_for(worker.second_lazy_hook.wait(), 1)

        owner.cancel()
        result = await asyncio.wait_for(waiter, 1)
        assert result.outputs["n"]["out"].resolve() == "lazy"
        assert result.executed == ("n",)
        assert worker.invocations == 2
        assert worker.lazy_hooks == 2
        assert not engine._inflight
        with pytest.raises(asyncio.CancelledError):
            await owner

    asyncio.run(scenario())


def test_max_concurrency_bounds_across_concurrent_runs() -> None:
    async def scenario() -> None:
        Sleeper.reset()
        engine, _ = make_engine(max_concurrency=1)
        await asyncio.gather(
            engine.run(sleeper_fanout("r1a", "r1b"), ["r1a", "r1b"]),
            engine.run(sleeper_fanout("r2a", "r2b"), ["r2a", "r2b"]),
        )
        assert Sleeper.peak == 1  # the bound is engine-wide, not per-run

    asyncio.run(scenario())


def test_validation_failure_has_no_scheduling_side_effects() -> None:
    async def scenario() -> None:
        engine, worker = make_engine()
        bad = Graph(nodes={"n": GraphNode("test.sleeper", {"nope": "x"})})
        with pytest.raises(GraphValidationError):
            await engine.run(bad, ["n"])
        assert worker.invocations == 0
        assert not engine._inflight

    asyncio.run(scenario())
