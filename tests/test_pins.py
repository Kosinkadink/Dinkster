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

from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import (
    ResidentPool,
    register_resident_type,
    resident_resource_id,
)
from dinkster_engine import Engine, EngineEvent, EventListener
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
from dinkster_values import (
    CORE_INT,
    CORE_STRING,
    COST_META_KEY,
    ResourcePins,
    TypeRegistry,
    register_core_types,
)
from dinkster_workers import InProcessWorker

# --- ResourcePins semantics ----------------------------------------------


def test_pins_count_references() -> None:
    pins = ResourcePins()
    assert not pins.pinned("r1")
    assert pins.pin("r1")
    assert pins.pin("r1")
    pins.unpin("r1")
    assert pins.pinned("r1")  # one reference still live
    pins.unpin("r1")
    assert not pins.pinned("r1")


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


def test_absolve_rolls_back_a_refused_release() -> None:
    pins = ResourcePins()
    assert pins.condemn("r1")
    pins.absolve("r1")
    assert pins.pin("r1")  # references are valid again


def test_concurrent_releases_cannot_both_condemn() -> None:
    pins = ResourcePins()
    assert pins.condemn("r1")
    assert not pins.condemn("r1")  # the second release must skip it


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
    pool = ResidentPool(cost_of=lambda obj: {COST_META_KEY: {"ram": 100}})
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

    asyncio.run(scenario())


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

    asyncio.run(scenario())
