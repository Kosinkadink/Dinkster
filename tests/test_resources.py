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
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import RESOURCES_META_KEY, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

STRING = TypeExpr.concrete("core.string")
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

    asyncio.run(scenario())


def test_configured_capacity_allows_overlap() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, _ = make_engine(resource_capacities={"gpu": 2})
        await engine.run(fanout("test.gpu_sleeper", "a", "b", "c"), ["a", "b", "c"])
        assert GpuSleeper.peak == 2  # a powerful GPU runs two, never three

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_plain_nodes_serialize_by_default() -> None:
    async def scenario() -> None:
        CpuSleeper.reset()
        engine, _ = make_engine()
        await engine.run(fanout("test.cpu_sleeper", "a", "b", "c"), ["a", "b", "c"])
        assert CpuSleeper.peak == 1  # implicit compute lane, capacity 1
        assert "compute" in engine._resource_slots

    asyncio.run(scenario())


def test_compute_capacity_widens_plain_parallelism() -> None:
    async def scenario() -> None:
        CpuSleeper.reset()
        engine, _ = make_engine(resource_capacities={"compute": 2})
        await engine.run(fanout("test.cpu_sleeper", "a", "b", "c"), ["a", "b", "c"])
        assert CpuSleeper.peak == 2  # widened deliberately, still bounded

    asyncio.run(scenario())


def test_io_nodes_overlap_freely() -> None:
    async def scenario() -> None:
        IoSleeper.reset()
        engine, _ = make_engine()
        await engine.run(fanout("test.io_sleeper", "a", "b", "c"), ["a", "b", "c"])
        assert IoSleeper.peak == 3  # exempt from the compute lane

    asyncio.run(scenario())


def test_admission_is_engine_wide_across_runs() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, _ = make_engine()
        await asyncio.gather(
            engine.run(fanout("test.gpu_sleeper", "r1a", "r1b"), ["r1a", "r1b"]),
            engine.run(fanout("test.gpu_sleeper", "r2a", "r2b"), ["r2a", "r2b"]),
        )
        assert GpuSleeper.peak == 1  # capacity holds across concurrent workflows

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_single_flight_waiter_consumes_no_permit() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        engine, worker = make_engine(resource_capacities={"gpu": 2}, gated=True)
        assert isinstance(worker, GatedWorker)
        graph = fanout("test.gpu_sleeper", "shared")

        owner = asyncio.create_task(engine.run(graph, ["shared"]))
        await asyncio.wait_for(worker.entered.wait(), 1)  # owner holds one permit
        waiter = asyncio.create_task(engine.run(graph, ["shared"]))
        for _ in range(5):  # let the waiter reach the shield
            await asyncio.sleep(0)

        assert worker.invocations == 1  # waiter never invoked
        assert engine._resource_slots["gpu"]._value == 1  # only the owner's permit

        worker.gate.set()
        r1, r2 = await asyncio.wait_for(asyncio.gather(owner, waiter), 1)
        assert worker.invocations == 1
        assert engine._resource_slots["gpu"]._value == 2  # everything released

    asyncio.run(scenario())


def test_multiple_kinds_acquired_and_released() -> None:
    async def scenario() -> None:
        DualResource.reset()
        engine, _ = make_engine()
        await engine.run(fanout("test.dual", "a", "b"), ["a", "b"])
        assert DualResource.peak == 1  # both kinds default to capacity 1
        assert engine._resource_slots["gpu"]._value == 1
        assert engine._resource_slots["npu"]._value == 1

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        make_engine(resource_capacities={"gpu": 0})
    with pytest.raises(ValueError, match="non-empty"):
        make_engine(resource_capacities={"": 1})


def test_invalid_occupies_declarations_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate occupies"):
        NodeSchema(node_type="t", occupies=("gpu", "gpu"))
    with pytest.raises(ValueError, match="non-empty strings"):
        NodeSchema(node_type="t", occupies=("",))
    with pytest.raises(ValueError, match="mutually exclusive"):
        NodeSchema(node_type="t", occupies=("gpu",), io_bound=True)


def test_wire_round_trip_preserves_admission_hints() -> None:
    schema = GpuSleeper.define_schema()
    restored = schema_from_wire(schema_to_wire(schema))
    assert restored.occupies == ("gpu",)
    io_restored = schema_from_wire(schema_to_wire(IoSleeper.define_schema()))
    assert io_restored.io_bound is True
    # Absent on the wire means the defaults - no phantom keys.
    plain_wire = schema_to_wire(CpuSleeper.define_schema())
    assert "occupies" not in plain_wire
    assert "ioBound" not in plain_wire


def test_occupies_never_changes_the_signature() -> None:
    plain = CpuSleeper.define_schema()
    occupying = NodeSchema(
        node_type=plain.node_type,
        inputs=plain.inputs,
        outputs=plain.outputs,
        occupies=("gpu", "npu"),
    )
    waiting = NodeSchema(
        node_type=plain.node_type,
        inputs=plain.inputs,
        outputs=plain.outputs,
        io_bound=True,
    )
    # Admission metadata is scheduling, not computation (hazard H4): the
    # same cache entries stay valid on differently configured hosts.
    assert schema_signature(plain) == schema_signature(occupying)
    assert schema_signature(plain) == schema_signature(waiting)


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

    asyncio.run(scenario())


def test_samplers_on_the_same_device_serialize() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine()
        graph = sampler_graph(("cuda:0", "a"), ("cuda:0", "b"))
        await engine.run(graph, ["sample_a", "sample_b"])
        assert Sampler.peak == 1  # one lane, default capacity 1

    asyncio.run(scenario())


def test_per_device_capacity_override() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine(resource_capacities={"gpu:cuda:0": 2})
        graph = sampler_graph(("cuda:0", "a"), ("cuda:0", "b"))
        await engine.run(graph, ["sample_a", "sample_b"])
        assert Sampler.peak == 2  # this one device admits two

    asyncio.run(scenario())


def test_bound_lane_inherits_kind_capacity() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine(resource_capacities={"gpu": 2})
        graph = sampler_graph(("cuda:0", "a"), ("cuda:0", "b"))
        await engine.run(graph, ["sample_a", "sample_b"])
        assert Sampler.peak == 2  # "gpu:cuda:0" falls back to the "gpu" figure

    asyncio.run(scenario())


def test_spanning_model_occupies_every_device() -> None:
    """Multigpu: a model living on two devices occupies both lanes, so its
    sampler contends with anything on either device but not with a third."""

    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine()
        graph = sampler_graph(
            ("cuda:0+cuda:1", "span"),  # occupies gpu:cuda:0 AND gpu:cuda:1
            ("cuda:1", "b"),  # contends with span on cuda:1
            ("cuda:2", "c"),  # untouched device: free to overlap
        )
        await engine.run(graph, ["sample_span", "sample_b", "sample_c"])
        assert Sampler.peak == 2  # never all three: span and b share cuda:1
        for lane in ("gpu:cuda:0", "gpu:cuda:1", "gpu:cuda:2"):
            assert lane in engine._resource_slots

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_elaboration_preserves_occupies() -> None:
    dynamic = NodeSchema(
        node_type="test.dyn_gpu",
        input_families=(InputFamilySpec("items", STRING),),
        outputs=(OutputSpec("out", STRING),),
        occupies=("gpu",),
    )
    effective = elaborate(dynamic, ["items.0", "items.1"])
    assert effective.occupies == ("gpu",)


class BlockAllWorker:
    """Blocks every invocation on one gate; counts concurrent entries."""

    def __init__(self, inner: InProcessWorker) -> None:
        self._inner = inner
        self.gate = asyncio.Event()
        self.entered = 0
        self.entered_changed = asyncio.Event()

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._inner.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        self.entered += 1
        self.entered_changed.set()
        await self.gate.wait()
        return await self._inner.invoke(invocation)

    async def wait_for_entered(self, count: int) -> None:
        while self.entered < count:
            self.entered_changed.clear()
            await self.entered_changed.wait()


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

    asyncio.run(scenario())


def test_resource_status_exact_count_with_wider_capacity() -> None:
    async def scenario() -> None:
        GpuSleeper.reset()
        registry = TypeRegistry()
        register_core_types(registry)
        worker = BlockAllWorker(InProcessWorker(build_node_types(NODES), registry))
        engine = Engine(
            schemas=build_schemas(NODES),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
            resource_capacities={"gpu": 2},
        )
        run = asyncio.create_task(
            engine.run(fanout("test.gpu_sleeper", "a", "b", "c"), ["a", "b", "c"])
        )
        # Capacity 2: exactly two invocations get lanes; the third is gated
        # at admission, so it must NOT be counted as in use.
        await worker.wait_for_entered(2)
        await asyncio.sleep(0)  # let any (wrongly) admitted third node surface
        status = engine.resource_status()
        assert status["gpu"] == {"executionCapacity": 2, "executionInUse": 2}
        worker.gate.set()
        await run
        assert engine.resource_status()["gpu"]["executionInUse"] == 0

    asyncio.run(scenario())


def test_resource_status_released_after_error() -> None:
    async def scenario() -> None:
        engine, _ = make_engine()
        with pytest.raises(ExecutionError):
            await engine.run(fanout("test.gpu_fail", "x"), ["x"])
        assert engine.resource_status()["gpu"]["executionInUse"] == 0

    asyncio.run(scenario())


def test_resource_status_released_after_cancellation() -> None:
    async def scenario() -> None:
        engine, worker = make_engine(gated=True)
        assert isinstance(worker, GatedWorker)
        run = asyncio.create_task(engine.run(fanout("test.gpu_sleeper", "a"), ["a"]))
        await worker.entered.wait()
        assert engine.resource_status()["gpu"]["executionInUse"] == 1
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert engine.resource_status()["gpu"]["executionInUse"] == 0

    asyncio.run(scenario())


def test_resource_status_reports_bound_lanes() -> None:
    async def scenario() -> None:
        Sampler.reset()
        engine, _ = make_engine(resource_capacities={"gpu": 2})
        await engine.run(sampler_graph(("cuda:0", "t")), ["sample_t"])
        status = engine.resource_status()
        # The bound lane inherits its kind's configured capacity.
        assert status["gpu:cuda:0"] == {"executionCapacity": 2, "executionInUse": 0}

    asyncio.run(scenario())
