"""ResourcePins and the engine's run-lifetime pinning (DESIGN 3.10).

The property under test: a resource reference held only in a run's Python
variables is visible to a ram-lane release (pinned), a condemned resource
can never acquire a new pin (late arrivals are detected and recomputed,
never dangled), and pins always unwind when the run ends - success,
failure, or cancellation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, EventListener
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import PressureSignal
from dinkster_memory.resident import ResidentPool
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    CORE_INT,
    CORE_STRING,
    COST_META_KEY,
    ResourcePins,
    TypeRegistry,
    register_core_types,
    register_resident_type,
    resident_resource_id,
)
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

# --- ResourcePins semantics ----------------------------------------------


def test_condemn_refuses_while_pinned() -> None:
    pins = ResourcePins()
    pins.pin("r1")
    assert not pins.condemn("r1")  # a live run holds it
    pins.unpin("r1")
    assert pins.condemn("r1")


def test_condemned_resource_refuses_new_pins() -> None:
    pins = ResourcePins()
    assert pins.condemn("r1")
    assert not pins.pin("r1")  # late arrival detected, not counted
    assert not pins.pinned("r1")


# --- engine run-lifetime pinning ------------------------------------------

MODEL = TypeExpr.concrete("pins.model")
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)


class FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name


class PinLoad(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pins.load",
            display_name="Load",
            category="test",
            inputs=(InputSpec("name", STRING, default="m"),),
            outputs=(OutputSpec("model", MODEL),),
        )

    @classmethod
    def execute(cls, *, name: str) -> Mapping[str, object]:
        return cls.outputs(model=FakeModel(name))


class PinUse(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="pins.use",
            display_name="Use",
            category="test",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("name", STRING),),
        )

    @classmethod
    def execute(cls, *, model: FakeModel) -> Mapping[str, object]:
        return cls.outputs(name=model.name)


NODES = [PinLoad, PinUse]


def build(
    pins: ResourcePins | None,
    cache: MemoryLRUCache,
    on_event: EventListener | None = None,
) -> tuple[Engine, ResidentPool]:
    pool = ResidentPool(
        cost_of=lambda obj: {COST_META_KEY: {"ram": 100, "vram:cuda:0": 400, "vram:cuda:1": 200}}
    )
    if pins is not None:
        pool.register_pins(pins)
    registry = TypeRegistry()
    register_core_types(registry)
    register_resident_type(registry, "pins.model", table=pool)
    engine = Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=cache,
        pins=pins,
        on_event=on_event,
    )
    return engine, pool


def graph(name: str = "m") -> Graph:
    return Graph(
        nodes={
            "load": GraphNode("pins.load", {"name": name}),
            "use": GraphNode("pins.use", {"model": Link("load", "model")}),
        }
    )


def resource_id_of(pool: ResidentPool) -> str:
    (item,) = pool.details()
    return resident_resource_id(item.item_id)


def test_run_pins_references_and_unpins_at_run_end() -> None:
    async def scenario() -> None:
        pins = ResourcePins()
        observed: list[bool] = []
        pool_box: list[ResidentPool] = []

        def on_event(event: EngineEvent) -> None:
            # At the downstream node's completion the loader's output is
            # held only in the run's `produced` dict - exactly the state
            # the pin must make visible.
            if event.kind == "node_finished" and event.node_id == "use":
                observed.append(pins.pinned(resource_id_of(pool_box[0])))

        engine, pool = build(pins, MemoryLRUCache(), on_event)
        pool_box.append(pool)
        await engine.run(graph("during"), ["use"])
        assert observed == [True]
        assert not pins.pinned(resource_id_of(pool))  # unwound at run end

    run_scenario(scenario())


def test_manual_unload_preserves_models_held_by_a_live_run() -> None:
    async def scenario() -> None:
        pins = ResourcePins()
        engine, pool = build(pins, MemoryLRUCache())
        await engine.run(graph("held"), ["use"])
        (before,) = pool.details()
        assert before.in_use is False
        await engine.run(graph("unused"), ["use"])
        entered, finish = asyncio.Event(), asyncio.Event()

        async def publish(run_id: str, node_id: str, outputs: Mapping[str, object]) -> None:
            if node_id == "load":
                entered.set()
                await finish.wait()

        execution = replace(engine.pin_execution(), publish_outputs=publish)
        running = asyncio.create_task(engine.run(graph("held"), ["use"], execution=execution))
        try:
            async with asyncio.timeout(10):
                await entered.wait()
                for device in ("ram", "vram:cuda:0", "vram:cuda:1"):
                    assert (
                        await pool.shed(PressureSignal(device, 1000, items=(before.item_id,))) == 0
                    )
                assert await pool.shed(PressureSignal("vram:cuda:0", 1000)) == 400
                assert await pool.shed(PressureSignal("vram:cuda:1", 1000)) == 0
                assert await pool.shed(PressureSignal("ram", 1000)) == 100
                (during,) = pool.details()
                assert during.in_use is True
                assert during.bytes_by_residency == before.bytes_by_residency
                finish.set()
                result = await running
                assert result.outputs["use"]["name"].resolve() == "held"
            assert await pool.shed(PressureSignal("vram:cuda:0", 400)) == 400
            assert pool.footprint("vram:cuda:1") == 0
            assert pool.footprint("ram") == 100
            assert await pool.shed(PressureSignal("ram", 100)) == 100
            assert pool.details() == []
        finally:
            finish.set()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    run_scenario(scenario())


def test_condemned_cache_hit_is_a_miss_and_recomputes() -> None:
    async def scenario() -> None:
        pins = ResourcePins()
        cache = MemoryLRUCache()
        engine, pool = build(pins, cache)

        first = await engine.run(graph("stale"), ["use"])
        assert "load" in first.executed
        resource_id = resource_id_of(pool)

        # A release happened: caches invalidated, resource condemned. (The
        # invalidation half is what drop_referencing does; condemning while
        # the entry survives simulates the stale-read window of a cache
        # whose get() truly suspends.)
        assert pins.condemn(resource_id)

        second = await engine.run(graph("stale"), ["use"])
        # The hit still referencing the condemned resource must not be
        # served; the loader reruns and produces a fresh resident.
        assert "load" in second.executed

    run_scenario(scenario())
