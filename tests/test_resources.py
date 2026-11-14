"""Resource admission (hazard H12): schemas declare what a node occupies,
the engine owns capacity. Default capacity 1 per kind serializes GPU nodes;
configuration raises it; admission never touches cache identity (hazard H4)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import (
    Engine,
    ExecutionError,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import RESOURCES_META_KEY, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")
MODEL = TypeExpr.concrete("test.model")


class Tracked(Node):
    """Base for nodes that record how many of their type run concurrently."""

    active = 0
    peak = 0

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        await asyncio.sleep(0.02)
        cls.active -= 1
        return cls.outputs(out=tag)

    @classmethod
    def reset(cls) -> None:
        cls.active = 0
        cls.peak = 0


class GpuSleeper(Tracked):
    active = 0
    peak = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.gpu_sleeper",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            occupies=("gpu",),
        )


class CpuSleeper(Tracked):
    """Plain local node: shares the implicit compute lane."""

    active = 0
    peak = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.cpu_sleeper",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )


class IoSleeper(Tracked):
    """Wait-bound (the partner/API-node case): exempt from the compute lane."""

    active = 0
    peak = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.io_sleeper",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            io_bound=True,
        )


class DualResource(Tracked):
    """Occupies two kinds at once - exercises multi-kind acquisition."""

    active = 0
    peak = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.dual",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            occupies=("gpu", "npu"),
        )


class GpuFail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.gpu_fail",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            occupies=("gpu",),
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        await asyncio.sleep(0.01)
        raise ValueError(f"boom: {tag}")


class LoadModel(Node):
    """Produces a model value that knows which device it lives on - the
    Dinkster analog of a ModelPatcher carrying its load device."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.load_model",
            inputs=(InputSpec("name", STRING), InputSpec("device", STRING)),
            outputs=(OutputSpec("model", MODEL),),
        )

    @classmethod
    async def execute(cls, *, name: str, device: str) -> Mapping[str, object]:
        return cls.outputs(model={"name": name, "device": device})


class Sampler(Tracked):
    """Occupies "gpu" abstractly; WHICH gpu is a fact of the model input."""

    active = 0
    peak = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.sampler",
            inputs=(InputSpec("model", MODEL), InputSpec("tag", STRING)),
            outputs=(OutputSpec("out", STRING),),
            occupies=("gpu",),
        )

    @classmethod
    async def execute(  # pyright: ignore[reportIncompatibleMethodOverride] - schema-declared inputs, never called via Tracked
        cls, *, model: Mapping[str, str], tag: str
    ) -> Mapping[str, object]:
        Sampler.active += 1
        Sampler.peak = max(Sampler.peak, Sampler.active)
        await asyncio.sleep(0.02)
        Sampler.active -= 1
        return cls.outputs(out=f"{model['name']}:{tag}")


NODES: list[type[Node]] = [
    GpuSleeper,
    CpuSleeper,
    IoSleeper,
    DualResource,
    GpuFail,
    LoadModel,
    Sampler,
]


class GatedWorker:
    """Blocks the first invocation on an event; later ones pass through."""

    def __init__(self, inner: InProcessWorker) -> None:
        self._inner = inner
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.invocations = 0

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._inner.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        self.invocations += 1
        if self.invocations == 1:
            self.entered.set()
            await self.gate.wait()
        return await self._inner.invoke(invocation)


def make_engine(
    resource_capacities: Mapping[str, int] | None = None,
    max_concurrency: int | None = None,
    gated: bool = False,
) -> tuple[Engine, GatedWorker | InProcessWorker]:
    registry = TypeRegistry()
    register_core_types(registry)

    # The model type declares its device(s) as envelope meta (scheduling),
    # while its fingerprint is content-only (identity): residency never
    # touches cache keys, exactly like a real model type would register.
    # "cuda:0+cuda:1" loads a model spanning both devices (multigpu).
    def model_meta(obj: object) -> Mapping[str, object]:
        devices = obj["device"].split("+")  # type: ignore[index]
        return {RESOURCES_META_KEY: {"gpu": devices[0] if len(devices) == 1 else devices}}

    registry.register(
        "test.model",
        fingerprint=lambda obj: "model:" + obj["name"],  # type: ignore[index]
        meta=model_meta,
    )
    inner = InProcessWorker(build_node_types(NODES), registry)
    worker: GatedWorker | InProcessWorker = GatedWorker(inner) if gated else inner
    engine = Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
        max_concurrency=max_concurrency,
        resource_capacities=resource_capacities,
    )
    return engine, worker


def fanout(node_type: str, *tags: str) -> Graph:
    return Graph(nodes={tag: GraphNode(node_type, {"tag": tag}) for tag in tags})


def test_gpu_nodes_serialize_by_default() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, _ = make_engine()  # "gpu" never configured -> capacity 1
        await engine.run(fanout("test.gpu_sleeper", "a", "b", "c"), ["a", "b", "c"])
        assert GpuSleeper.peak == 1  # one occupying invocation at a time

    run_scenario(scenario())


def test_io_nodes_overlap_while_gpu_serializes() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        IoSleeper.reset()
        engine, _ = make_engine()
        graph = Graph(
            nodes={
                "g1": GraphNode("test.gpu_sleeper", {"tag": "g1"}),
                "g2": GraphNode("test.gpu_sleeper", {"tag": "g2"}),
                "i1": GraphNode("test.io_sleeper", {"tag": "i1"}),
                "i2": GraphNode("test.io_sleeper", {"tag": "i2"}),
            }
        )
        await engine.run(graph, ["g1", "g2", "i1", "i2"])
        assert GpuSleeper.peak == 1  # gpu lane serialized
        assert IoSleeper.peak == 2  # io nodes unaffected by gpu admission

    run_scenario(scenario())


def test_admission_is_engine_wide_across_runs() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, _ = make_engine()
        await asyncio.gather(
            engine.run(fanout("test.gpu_sleeper", "r1a", "r1b"), ["r1a", "r1b"]),
            engine.run(fanout("test.gpu_sleeper", "r2a", "r2b"), ["r2a", "r2b"]),
        )
        assert GpuSleeper.peak == 1  # capacity holds across concurrent workflows

    run_scenario(scenario())


def test_cache_hits_need_no_permit() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, _ = make_engine()
        graph = fanout("test.gpu_sleeper", "warm")
        await engine.run(graph, ["warm"])  # populate the cache
        # Exhaust the gpu permit by hand; a cache hit must still complete.
        await engine._resource_slot("gpu").acquire()
        try:
            result = await asyncio.wait_for(engine.run(graph, ["warm"]), 1)
            assert result.cached == ("warm",)
        finally:
            engine._resource_slot("gpu").release()

    run_scenario(scenario())


def test_permits_released_after_worker_error() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, _ = make_engine()
        with pytest.raises(ExecutionError):
            await engine.run(fanout("test.gpu_fail", "bad"), ["bad"])
        # The permit is free again: a healthy gpu node completes promptly.
        result = await asyncio.wait_for(engine.run(fanout("test.gpu_sleeper", "ok"), ["ok"]), 1)
        assert result.executed == ("ok",)
        assert engine._resource_slots["gpu"]._value == 1

    run_scenario(scenario())


def test_permits_released_after_cancellation() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, worker = make_engine(gated=True)
        assert isinstance(worker, GatedWorker)

        owner = asyncio.create_task(engine.run(fanout("test.gpu_sleeper", "doomed"), ["doomed"]))
        await asyncio.wait_for(worker.entered.wait(), 1)  # blocked holding the permit
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner

        result = await asyncio.wait_for(engine.run(fanout("test.gpu_sleeper", "next"), ["next"]), 1)
        assert result.executed == ("next",)
        assert engine._resource_slots["gpu"]._value == 1

    run_scenario(scenario())


def sampler_graph(*pairs: tuple[str, str]) -> Graph:
    """One loader + one sampler per (device, tag) pair."""
    nodes: dict[str, GraphNode] = {}
    for device, tag in pairs:
        nodes[f"load_{tag}"] = GraphNode("test.load_model", {"name": "sdxl", "device": device})
        nodes[f"sample_{tag}"] = GraphNode(
            "test.sampler", {"model": Link(f"load_{tag}", "model"), "tag": tag}
        )
    return Graph(nodes=nodes)


def test_samplers_on_different_devices_overlap() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine()
        graph = sampler_graph(("cuda:0", "a"), ("cuda:1", "b"))
        await engine.run(graph, ["sample_a", "sample_b"])
        assert Sampler.peak == 2  # separate devices are separate lanes
        assert "gpu:cuda:0" in engine._resource_slots
        assert "gpu:cuda:1" in engine._resource_slots
        assert "gpu" not in engine._resource_slots  # every kind was bound

    run_scenario(scenario())


def test_samplers_on_the_same_device_serialize() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine()
        graph = sampler_graph(("cuda:0", "a"), ("cuda:0", "b"))
        await engine.run(graph, ["sample_a", "sample_b"])
        assert Sampler.peak == 1  # one lane, default capacity 1

    run_scenario(scenario())


def test_device_never_changes_cache_identity() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine()
        first = await engine.run(sampler_graph(("cuda:0", "t")), ["sample_t"])
        assert "sample_t" in first.executed
        # Same model content, different device: the sampler's inputs
        # fingerprint identically, so its computation is a cache hit even
        # though it would occupy a different lane if it ran.
        second = await engine.run(sampler_graph(("cuda:1", "t")), ["sample_t"])
        assert "sample_t" in second.cached

    run_scenario(scenario())


def test_resource_status_idle_then_active_then_released() -> None:
    """resource_status() reports occupancy honestly: lanes appear once first
    used, count while an invocation holds them, and return to zero after."""

    async def scenario() -> None:
        engine, worker = make_engine(gated=True)
        assert isinstance(worker, GatedWorker)
        assert engine.resource_status() == {}  # nothing used yet
        run = asyncio.create_task(engine.run(fanout("test.gpu_sleeper", "a"), ["a"]))
        await worker.entered.wait()
        status = engine.resource_status()
        assert status["gpu"] == {"executionCapacity": 1, "executionInUse": 1}
        worker.gate.set()
        await run
        assert engine.resource_status()["gpu"]["executionInUse"] == 0

    run_scenario(scenario())
