import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import (
    ActiveRunIdError,
    Engine,
    EngineEvent,
    ExecutionError,
    ExecutionSelection,
    GraphValidationError,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_protocol import (
    Invocation,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    OnInvocationEvent,
)
from dinkster_schema import (
    InputFamilySpec,
    InputSpec,
    MultiComboWidget,
    Node,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    RESOURCE_HANDLE_TYPE,
    ResourceHandle,
    TypeRegistry,
    Value,
    register_core_types,
    register_resource_handle_type,
)
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types


def make_engine(
    events: list[EngineEvent] | None = None,
    *,
    explain_misses: bool = False,
    cache: MemoryLRUCache | None = None,
) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    return Engine(
        schemas=build_schemas(SCAFFOLD_NODES),
        registry=registry,
        worker=InProcessWorkerFactory(registry),
        cache=cache if cache is not None else MemoryLRUCache(),
        on_event=events.append if events is not None else None,
        explain_misses=explain_misses,
    )


def InProcessWorkerFactory(registry: TypeRegistry):
    from dinkster_workers import InProcessWorker

    return InProcessWorker(build_node_types(SCAFFOLD_NODES), registry)


def image_graph(ratio: float = 0.5) -> Graph:
    return Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 16, "height": 8}),
            "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
            "b": GraphNode(
                "dev.image.blend",
                {"a": Link("g", "image"), "b": Link("i", "image"), "ratio": ratio},
            ),
            "s": GraphNode("dev.image.stats", {"image": Link("b", "image")}),
        }
    )


def test_end_to_end_and_cache_behavior() -> None:
    async def scenario() -> None:
        engine = make_engine()

        first = await engine.run(image_graph(), ["s"])
        assert set(first.executed) == {"g", "i", "b", "s"}
        assert first.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
        assert first.outputs["s"]["mean"].type_id == "core.float"

        second = await engine.run(image_graph(), ["s"])
        assert second.executed == ()
        assert set(second.cached) == {"g", "i", "b", "s"}

        # Changing one literal re-executes only the dirty suffix.
        third = await engine.run(image_graph(ratio=0.9), ["s"])
        assert set(third.executed) == {"b", "s"}
        assert set(third.cached) == {"g", "i"}

    asyncio.run(scenario())


def test_provider_resolution_reads_immutable_run_entry_snapshot() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = image_graph()
        original_inputs = cast("dict[str, object]", graph.nodes["g"].inputs)
        calls = 0

        def resolve_providers(candidate: Graph) -> Graph:
            nonlocal calls
            calls += 1
            assert candidate is not graph
            with pytest.raises(TypeError):
                cast("dict[str, object]", candidate.nodes["g"].inputs)["width"] = 32
            original_inputs["width"] = "invalid-after-snapshot"
            return candidate

        runtime = replace(engine.pin_execution(), resolve_providers=resolve_providers)
        result = await engine.run(graph, ["s"], execution=runtime)

        assert result.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
        assert calls == 2

    asyncio.run(scenario())


def test_computed_lazy_selector_runs_only_the_selected_branch_cold_and_warm() -> None:
    executed_sources: list[str] = []

    class StringSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-string-source",
                inputs=(InputSpec("value", TypeExpr.concrete(CORE_STRING)),),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls, value: str) -> Mapping[str, object]:
            executed_sources.append(value)
            return cls.outputs(value=value)

    class BoolSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-bool-source",
                inputs=(InputSpec("value", TypeExpr.concrete(CORE_BOOLEAN)),),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_BOOLEAN)),),
            )

        @classmethod
        def execute(cls, value: bool) -> Mapping[str, object]:
            return cls.outputs(value=value)

    class LazySwitch(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-switch",
                inputs=(
                    InputSpec("switch", TypeExpr.concrete(CORE_BOOLEAN)),
                    InputSpec("off", TypeExpr.concrete(CORE_STRING), lazy=True),
                    InputSpec("on", TypeExpr.concrete(CORE_STRING), lazy=True),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
                selector=SelectorSpec("switch", {"false": "off", "true": "on"}),
            )

        @classmethod
        def check_lazy_status(
            cls, switch: bool, off: str | None, on: str | None
        ) -> tuple[str, ...]:
            del cls
            if switch and on is None:
                return ("on",)
            if not switch and off is None:
                return ("off",)
            return ()

        @classmethod
        def execute(cls, switch: bool, off: str | None, on: str | None) -> Mapping[str, object]:
            return cls.outputs(value=on if switch else off)

    node_types = (StringSource, BoolSource, LazySwitch)
    registry = TypeRegistry()
    register_core_types(registry)
    events: list[EngineEvent] = []
    from dinkster_workers import DispatchWorker, InProcessWorker

    owner = InProcessWorker(build_node_types(node_types), registry)
    owner_domain = object()
    worker = DispatchWorker(
        {"owner": owner},
        resolve_owner=lambda _token: None,
        arm_domains={"owner": owner_domain},
        default_arms={"owner": "owner"},
    )

    async def plan_owner(
        _node_id: str,
        _node_type: str,
        _schema: NodeSchema,
        _inputs: Mapping[str, object],
        _run_id: str,
        _attention_config: object,
    ) -> ExecutionSelection:
        return ExecutionSelection(target="owner", cache_tag="owner@1")

    engine = Engine(
        schemas=build_schemas(node_types),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
        on_event=events.append,
        plan_execution=plan_owner,
    )
    graph = Graph(
        {
            "selector": GraphNode("test.lazy-bool-source", {"value": True}),
            "off": GraphNode("test.lazy-string-source", {"value": "OFF"}),
            "on": GraphNode("test.lazy-string-source", {"value": "ON"}),
            "switch": GraphNode(
                "test.lazy-switch",
                {
                    "switch": Link("selector", "value"),
                    "off": Link("off", "value"),
                    "on": Link("on", "value"),
                },
            ),
        }
    )

    async def scenario() -> None:
        cold = await engine.run(graph, ["switch"])
        warm = await engine.run(graph, ["switch"])
        assert cold.outputs["switch"]["value"].resolve() == "ON"
        assert warm.outputs["switch"]["value"].resolve() == "ON"
        assert executed_sources == ["ON"]
        assert "off" not in cold.executed and "off" not in warm.executed
        assert "on" in cold.executed and "on" in warm.cached
        assert "switch" in cold.executed and "switch" in warm.cached
        demands = [
            event.detail
            for event in events
            if event.kind == "node_event" and event.detail.get("name") == "lazy_demand"
        ]
        assert len(demands) == 4
        first = cast("Mapping[str, object]", demands[0]["data"])
        warm_first = cast("Mapping[str, object]", demands[2]["data"])
        assert first["producerNodes"] == ["on"]
        assert warm_first["producerNodes"] == ["on"]

    asyncio.run(scenario())


def test_computed_selector_keeps_resident_value_in_producer_domain() -> None:
    class BoolSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.selector-resident-bool",
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_BOOLEAN)),),
            )

    class ResidentSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.selector-resident-source",
                inputs=(InputSpec("resource_id", TypeExpr.concrete(CORE_STRING)),),
                outputs=(OutputSpec("value", TypeExpr.concrete(RESOURCE_HANDLE_TYPE)),),
            )

    class ResidentSelector(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            resource = TypeExpr.concrete(RESOURCE_HANDLE_TYPE)
            return NodeSchema(
                node_type="test.selector-resident-select",
                inputs=(
                    InputSpec("condition", TypeExpr.concrete(CORE_BOOLEAN)),
                    InputSpec("on_false", resource, lazy=True),
                    InputSpec("on_true", resource, lazy=True),
                ),
                outputs=(OutputSpec("value", resource),),
                selector=SelectorSpec("condition", {"false": "on_false", "true": "on_true"}),
            )

    registry = TypeRegistry()
    register_core_types(registry)
    register_resource_handle_type(registry)
    foundation_domain = object()
    model_domain = object()
    invocations: list[tuple[str, str]] = []

    class Arm:
        def __init__(self, name: str) -> None:
            self.name = name

        async def prepare(self, node_types: Sequence[str]) -> None:
            del node_types
            return None

        async def invoke(
            self, invocation: Invocation, on_event: OnInvocationEvent | None = None
        ) -> InvocationResult:
            del on_event
            invocations.append((self.name, invocation.node_type))
            if invocation.node_type == BoolSource.schema().node_type:
                return InvocationResult(outputs={"value": registry.wrap(CORE_BOOLEAN, True)})
            if invocation.node_type == ResidentSource.schema().node_type:
                resource_id = cast("str", invocation.inputs["resource_id"].resolve())
                return InvocationResult(
                    outputs={
                        "value": registry.wrap(
                            RESOURCE_HANDLE_TYPE,
                            ResourceHandle(resource_id, "model", owner="model-life"),
                        )
                    }
                )
            raise AssertionError("selectors must not execute in a worker")

        async def check_lazy_status(
            self,
            _invocation: LazyStatusInvocation,
            on_event: OnInvocationEvent | None = None,
        ) -> LazyStatusResult:
            del on_event
            raise AssertionError("selector demand must not enter a worker residency domain")

    from dinkster_workers import DispatchWorker

    worker = DispatchWorker(
        {"foundation": Arm("foundation"), "models": Arm("models")},
        resolve_owner=lambda owner: model_domain if owner == "model-life" else None,
        arm_domains={"foundation": foundation_domain, "models": model_domain},
        default_arms={"foundation": "foundation", "models": "models"},
    )
    planned: list[str] = []

    async def plan_execution(
        _node_id: str,
        node_type: str,
        _schema: NodeSchema,
        _inputs: Mapping[str, object],
        _run_id: str,
        _attention_config: object,
    ) -> ExecutionSelection:
        planned.append(node_type)
        target = "models" if node_type == ResidentSource.schema().node_type else "foundation"
        return ExecutionSelection(target=target, cache_tag=target)

    nodes = (BoolSource, ResidentSource, ResidentSelector)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
        plan_execution=plan_execution,
    )
    graph = Graph(
        {
            "condition": GraphNode(BoolSource.schema().node_type),
            "inactive": GraphNode(ResidentSource.schema().node_type, {"resource_id": "inactive"}),
            "selected": GraphNode(ResidentSource.schema().node_type, {"resource_id": "selected"}),
            "selector": GraphNode(
                ResidentSelector.schema().node_type,
                {
                    "condition": Link("condition", "value"),
                    "on_false": Link("inactive", "value"),
                    "on_true": Link("selected", "value"),
                },
            ),
        }
    )

    async def scenario() -> None:
        result = await engine.run(graph, ["selector"])
        selected = result.outputs["selector"]["value"].resolve()
        assert isinstance(selected, ResourceHandle)
        assert selected.resource_id == "selected"
        assert invocations == [
            ("foundation", BoolSource.schema().node_type),
            ("models", ResidentSource.schema().node_type),
        ]
        assert ResidentSelector.schema().node_type not in planned
        assert ResidentSource.schema().node_type in planned
        assert "inactive" not in result.executed

    asyncio.run(scenario())


def test_demanded_producer_can_run_its_own_lazy_fixpoint() -> None:
    class Source(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.nested-lazy-source",
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(value="nested")

    class LazyPass(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.nested-lazy-pass",
                inputs=(InputSpec("value", TypeExpr.concrete(CORE_STRING), lazy=True),),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(cls, value: str | None) -> tuple[str, ...]:
            return ("value",) if value is None else ()

        @classmethod
        def execute(cls, value: str | None) -> Mapping[str, object]:
            return cls.outputs(value=value)

    nodes = (Source, LazyPass)
    registry = TypeRegistry()
    register_core_types(registry)
    events: list[EngineEvent] = []
    from dinkster_workers import InProcessWorker

    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
        on_event=events.append,
    )
    graph = Graph(
        {
            "source": GraphNode("test.nested-lazy-source"),
            "inner": GraphNode("test.nested-lazy-pass", {"value": Link("source", "value")}),
            "outer": GraphNode("test.nested-lazy-pass", {"value": Link("inner", "value")}),
        }
    )

    result = asyncio.run(engine.run(graph, ["outer"]))
    assert result.outputs["outer"]["value"].resolve() == "nested"
    assert set(result.executed) == {"source", "inner"}
    assert result.cached == ("outer",)
    demand_nodes = [
        event.node_id
        for event in events
        if event.kind == "node_event" and event.detail.get("name") == "lazy_demand"
    ]
    assert demand_nodes == ["outer", "inner", "inner", "outer"]


def test_lazy_consumer_hooks_run_before_single_flight_coalesces_the_body() -> None:
    schema = NodeSchema(
        node_type="test.lazy-cache-bypass",
        inputs=(InputSpec("value", TypeExpr.concrete(CORE_STRING), lazy=True),),
        outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
        io_bound=True,
    )
    registry = TypeRegistry()
    register_core_types(registry)

    class CountingCache(MemoryLRUCache):
        def __init__(self) -> None:
            super().__init__()
            self.get_calls = 0
            self.put_calls = 0

        async def get(self, key: str) -> Mapping[str, Value] | None:
            self.get_calls += 1
            return await super().get(key)

        async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
            self.put_calls += 1
            await super().put(key, outputs)

    class GatedWorker:
        def __init__(self) -> None:
            self.invocations = 0
            self.hooks = 0
            self.first_started = asyncio.Event()
            self.both_hooks = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare(self, node_types: Sequence[str]) -> None:
            del node_types
            return None

        async def check_lazy_status(
            self,
            _invocation: LazyStatusInvocation,
            on_event: OnInvocationEvent | None = None,
        ) -> LazyStatusResult:
            del on_event
            self.hooks += 1
            if self.hooks == 2:
                self.both_hooks.set()
            return LazyStatusResult(requested_inputs=())

        async def invoke(
            self,
            invocation: Invocation,
            on_event: OnInvocationEvent | None = None,
        ) -> InvocationResult:
            del invocation, on_event
            self.invocations += 1
            if self.invocations == 1:
                self.first_started.set()
            await self.release.wait()
            return InvocationResult(outputs={"value": registry.wrap(CORE_STRING, "ready")})

    async def scenario() -> None:
        cache = CountingCache()
        worker = GatedWorker()
        engine = Engine(
            schemas={schema.node_type: schema},
            registry=registry,
            worker=worker,
            cache=cache,
        )
        graph = Graph({"lazy": GraphNode(schema.node_type, {"value": "literal"})})
        first = asyncio.create_task(engine.run(graph, ["lazy"]))
        await asyncio.wait_for(worker.first_started.wait(), 2)
        second = asyncio.create_task(engine.run(graph, ["lazy"]))
        await asyncio.wait_for(worker.both_hooks.wait(), 2)
        worker.release.set()
        results = await asyncio.gather(first, second)

        assert worker.hooks == 2
        assert worker.invocations == 1
        assert cache.get_calls == 1
        assert cache.put_calls == 1
        assert sum(result.executed == ("lazy",) for result in results) == 1
        assert sum(result.cached == ("lazy",) for result in results) == 1

    asyncio.run(scenario())


def test_lazy_cache_identity_tracks_final_demand_and_ignores_undemanded_values() -> None:
    source_runs: list[str] = []

    class Source(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-identity-source",
                inputs=(InputSpec("value", TypeExpr.concrete(CORE_STRING)),),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls, value: str) -> Mapping[str, object]:
            source_runs.append(value)
            return cls.outputs(value=value)

    class Pick(Node):
        hooks = 0
        bodies = 0

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-demand-aware-identity",
                inputs=(
                    InputSpec("selector", TypeExpr.concrete(CORE_BOOLEAN)),
                    InputSpec("first", TypeExpr.concrete(CORE_STRING), lazy=True),
                    InputSpec("second", TypeExpr.concrete(CORE_STRING), lazy=True),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(
            cls, selector: bool, first: str | None, second: str | None
        ) -> tuple[str, ...]:
            cls.hooks += 1
            return (
                ("first",)
                if selector and first is None
                else (("second",) if not selector and second is None else ())
            )

        @classmethod
        def execute(
            cls, selector: bool, first: str | None, second: str | None
        ) -> Mapping[str, object]:
            cls.bodies += 1
            return cls.outputs(value=first if selector else second)

    registry = TypeRegistry()
    register_core_types(registry)
    nodes = (Source, Pick)
    events: list[EngineEvent] = []
    from dinkster_workers import InProcessWorker

    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
        on_event=events.append,
        explain_misses=True,
    )

    def graph(selector: bool, first: str, second: str) -> Graph:
        return Graph(
            {
                "first": GraphNode(Source.schema().node_type, {"value": first}),
                "second": GraphNode(Source.schema().node_type, {"value": second}),
                "pick": GraphNode(
                    Pick.schema().node_type,
                    {
                        "selector": selector,
                        "first": Link("first", "value"),
                        "second": Link("second", "value"),
                    },
                ),
            }
        )

    async def scenario() -> None:
        cold = await engine.run(graph(True, "A", "B"), ["pick"])
        assert cold.outputs["pick"]["value"].resolve() == "A"

        undemanded_changed = await engine.run(graph(True, "A", "B2"), ["pick"])
        assert undemanded_changed.cached == ("first", "pick")
        assert "B2" not in source_runs

        demanded_changed = await engine.run(graph(True, "A2", "B2"), ["pick"])
        assert demanded_changed.outputs["pick"]["value"].resolve() == "A2"
        assert "pick" in demanded_changed.executed

        events.clear()
        demand_set_changed = await engine.run(graph(False, "A2", "B2"), ["pick"])
        assert demand_set_changed.outputs["pick"]["value"].resolve() == "B2"
        assert "pick" in demand_set_changed.executed
        miss = _misses(events)["pick"]
        assert miss["reason"] == "lazy-state-changed"
        assert miss["lazy_state"] == {"connected_undemanded_inputs": ["first"]}
        assert miss["previous_lazy_state"] == {"connected_undemanded_inputs": ["second"]}

        undemanded_changed_again = await engine.run(graph(False, "never-run", "B2"), ["pick"])
        assert "pick" in undemanded_changed_again.cached
        assert "never-run" not in source_runs
        assert Pick.hooks == 10  # demand round plus final ready round per run
        assert Pick.bodies == 3

    asyncio.run(scenario())


def test_connected_undemanded_identity_differs_from_omission_and_default() -> None:
    class Source(Node):
        runs = 0

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-shape-source",
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            cls.runs += 1
            return cls.outputs(value=f"source-{cls.runs}")

    class Shape(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-shape-identity",
                inputs=(
                    InputSpec(
                        "omittable",
                        TypeExpr.concrete(CORE_STRING),
                        required=False,
                        lazy=True,
                    ),
                    InputSpec(
                        "defaulted",
                        TypeExpr.concrete(CORE_STRING),
                        required=False,
                        default="schema-default",
                        lazy=True,
                    ),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(cls, **_inputs: object) -> tuple[()]:
            return ()

        @classmethod
        def execute(
            cls,
            omittable: str | None = "omitted",
            defaulted: str | None = "python-default",
        ) -> Mapping[str, object]:
            return cls.outputs(value=f"{omittable}|{defaulted}")

    class MarkerOnly(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-marker-only-identity",
                inputs=(
                    InputSpec(
                        "value",
                        TypeExpr.concrete(CORE_STRING),
                        required=False,
                        lazy=True,
                    ),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(cls, **_inputs: object) -> tuple[()]:
            return ()

        @classmethod
        def execute(cls, value: str | None = "omitted") -> Mapping[str, object]:
            return cls.outputs(value=str(value))

    registry = TypeRegistry()
    register_core_types(registry)
    nodes = (Source, Shape, MarkerOnly)
    events: list[EngineEvent] = []
    from dinkster_workers import InProcessWorker

    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
        on_event=events.append,
        explain_misses=True,
    )
    connected = Graph(
        {
            "source": GraphNode(Source.schema().node_type),
            "shape": GraphNode(
                Shape.schema().node_type,
                {
                    "omittable": Link("source", "value"),
                    "defaulted": Link("source", "value"),
                },
            ),
        }
    )
    unconnected = Graph({"shape": GraphNode(Shape.schema().node_type)})
    marker_connected = Graph(
        {
            "source": GraphNode(Source.schema().node_type),
            "marker": GraphNode(
                MarkerOnly.schema().node_type,
                {"value": Link("source", "value")},
            ),
        }
    )
    marker_unconnected = Graph({"marker": GraphNode(MarkerOnly.schema().node_type)})

    async def scenario() -> None:
        marker_first = await engine.run(marker_connected, ["marker"])
        assert marker_first.outputs["marker"]["value"].resolve() == "None"
        assert Source.runs == 0

        events.clear()
        marker_second = await engine.run(marker_unconnected, ["marker"])
        assert marker_second.outputs["marker"]["value"].resolve() == "omitted"
        marker_miss = _misses(events)["marker"]
        assert marker_miss["reason"] == "lazy-state-changed"
        assert marker_miss["lazy_state"] == {"connected_undemanded_inputs": []}
        assert marker_miss["previous_lazy_state"] == {"connected_undemanded_inputs": ["value"]}
        marker_replay = await engine.run(marker_unconnected, ["marker"])
        assert marker_replay.cached == ("marker",)

        first = await engine.run(connected, ["shape"])
        assert first.outputs["shape"]["value"].resolve() == "None|None"
        assert Source.runs == 0

        events.clear()
        second = await engine.run(unconnected, ["shape"])
        assert second.outputs["shape"]["value"].resolve() == "omitted|schema-default"
        assert second.executed == ("shape",)
        miss = _misses(events)["shape"]
        assert miss["reason"] == "lazy-state-changed"
        assert miss["lazy_state"] == {"connected_undemanded_inputs": []}
        assert miss["previous_lazy_state"] == {
            "connected_undemanded_inputs": ["omittable", "defaulted"]
        }

        replay = await engine.run(unconnected, ["shape"])
        assert replay.cached == ("shape",)
        connected_replay = await engine.run(connected, ["shape"])
        assert connected_replay.cached == ("shape",)
        assert connected_replay.outputs["shape"]["value"].resolve() == "None|None"
        assert Source.runs == 0

    asyncio.run(scenario())


def test_non_idempotent_lazy_consumer_remains_uncached_and_uncoalesced() -> None:
    class Effect(Node):
        hooks = 0
        bodies = 0

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-effect",
                inputs=(
                    InputSpec(
                        "trigger",
                        TypeExpr.concrete(CORE_INT),
                        required=False,
                        lazy=True,
                    ),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_INT)),),
                idempotent=False,
            )

        @classmethod
        def check_lazy_status(cls, **_inputs: object) -> tuple[()]:
            cls.hooks += 1
            return ()

        @classmethod
        def execute(cls, **_inputs: object) -> Mapping[str, object]:
            cls.bodies += 1
            return cls.outputs(value=cls.bodies)

    class CountingCache(MemoryLRUCache):
        def __init__(self) -> None:
            super().__init__()
            self.get_calls = 0
            self.put_calls = 0

        async def get(self, key: str) -> Mapping[str, Value] | None:
            self.get_calls += 1
            return await super().get(key)

        async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
            self.put_calls += 1
            await super().put(key, outputs)

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        from dinkster_workers import InProcessWorker

        cache = CountingCache()
        engine = Engine(
            schemas=build_schemas((Effect,)),
            registry=registry,
            worker=InProcessWorker(build_node_types((Effect,)), registry),
            cache=cache,
        )
        graph = Graph({"effect": GraphNode(Effect.schema().node_type)})
        first, second = await asyncio.gather(
            engine.run(graph, ["effect"]), engine.run(graph, ["effect"])
        )
        values = {
            first.outputs["effect"]["value"].resolve(),
            second.outputs["effect"]["value"].resolve(),
        }
        assert values == {1, 2}
        assert Effect.hooks == 2
        assert Effect.bodies == 2
        assert cache.get_calls == 0
        assert cache.put_calls == 0
        assert not engine._inflight

    asyncio.run(scenario())


def test_lazy_dynamic_members_select_only_one_source_and_cache() -> None:
    source_runs: list[str] = []

    class Source(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-dynamic-source",
                inputs=(InputSpec("value", TypeExpr.concrete(CORE_STRING)),),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls, *, value: str) -> Mapping[str, object]:
            source_runs.append(value)
            return cls.outputs(value=value)

    class LazyDynamic(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-dynamic-select",
                inputs=(InputSpec("index", TypeExpr.concrete(CORE_INT)),),
                input_families=(
                    InputFamilySpec(
                        "parts",
                        (
                            InputSpec(
                                "value",
                                TypeExpr.concrete(CORE_STRING),
                                lazy=True,
                            ),
                        ),
                        min_members=1,
                    ),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(
            cls, *, index: int, parts: Mapping[str, str | None]
        ) -> tuple[str, ...]:
            suffix, value = tuple(parts.items())[index]
            return (f"parts.{suffix}",) if value is None else ()

        @classmethod
        def execute(cls, *, index: int, parts: Mapping[str, str | None]) -> Mapping[str, object]:
            value = tuple(parts.values())[index]
            assert value is not None
            return cls.outputs(value=value)

    async def scenario() -> None:
        from dinkster_workers import InProcessWorker

        nodes = (Source, LazyDynamic)
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )

        def graph(last: str) -> Graph:
            return Graph(
                {
                    "a": GraphNode("test.lazy-dynamic-source", {"value": "A"}),
                    "b": GraphNode("test.lazy-dynamic-source", {"value": "B"}),
                    "c": GraphNode("test.lazy-dynamic-source", {"value": last}),
                    "lazy": GraphNode(
                        "test.lazy-dynamic-select",
                        {
                            "index": 1,
                            "parts.a": Link("a", "value"),
                            "parts.b": Link("b", "value"),
                            "parts.c": Link("c", "value"),
                        },
                    ),
                }
            )

        cold = await engine.run(graph("C"), ["lazy"])
        warm = await engine.run(graph("C"), ["lazy"])
        undemanded_changed = await engine.run(graph("C2"), ["lazy"])
        assert cold.outputs["lazy"]["value"].resolve() == "B"
        assert warm.outputs["lazy"]["value"].resolve() == "B"
        assert undemanded_changed.outputs["lazy"]["value"].resolve() == "B"
        assert set(cold.executed) == {"b", "lazy"}
        assert set(warm.cached) == {"b", "lazy"}
        assert set(undemanded_changed.cached) == {"b", "lazy"}
        assert source_runs == ["B"]

    asyncio.run(scenario())


def test_lazy_grouped_dynamic_members_are_loud_refusals() -> None:
    class LazyGrouped(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.lazy-grouped-dynamic-refusal",
                input_families=(
                    InputFamilySpec(
                        "parts",
                        (
                            InputSpec("value", TypeExpr.concrete(CORE_STRING), lazy=True),
                            InputSpec("label", TypeExpr.concrete(CORE_STRING)),
                        ),
                        min_members=1,
                    ),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(cls, **_inputs: object) -> tuple[()]:
            return ()

        @classmethod
        def execute(cls, **_inputs: object) -> Mapping[str, object]:
            return cls.outputs(value="unreachable")

    async def scenario() -> None:
        from dinkster_workers import InProcessWorker

        nodes = (LazyGrouped,)
        registry = TypeRegistry()
        register_core_types(registry)
        events: list[EngineEvent] = []
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph(
            {
                "lazy": GraphNode(
                    "test.lazy-grouped-dynamic-refusal",
                    {"parts.a.value": "x", "parts.a.label": "X"},
                )
            }
        )
        with pytest.raises(ExecutionError, match="lazy-dynamic-unsupported"):
            await engine.run(graph, ["lazy"])
        failures = [event for event in events if event.kind == "node_failed"]
        assert len(failures) == 1
        assert failures[0].detail["message"] == "lazy-dynamic-unsupported"

    asyncio.run(scenario())


def grant_save_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DINKSTER_MOUNTS_SNAPSHOT at a one-mount snapshot so save_pgm may
    write - into this folder, nowhere else. Returns the writable root."""
    root = tmp_path / "out"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "out", "root": str(root), "mode": "readwrite"}]}),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root


def save_graph() -> Graph:
    return Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 4, "height": 4}),
            "save": GraphNode(
                "dev.image.save_pgm",
                {
                    "image": Link("g", "image"),
                    "target": {"mount": "out", "prefix": "result"},
                },
            ),
        }
    )


def test_non_idempotent_never_cached(tmp_path, monkeypatch) -> None:
    root = grant_save_mount(tmp_path, monkeypatch)

    async def scenario() -> None:
        engine = make_engine()
        graph = save_graph()
        first = await engine.run(graph, ["save"])
        second = await engine.run(graph, ["save"])
        assert "save" in first.executed
        assert "save" in second.executed  # re-ran despite identical inputs
        assert "g" in second.cached
        # Both runs landed distinct counter-named files in the mount.
        assert (root / "result_00001.pgm").exists()
        assert (root / "result_00002.pgm").exists()
        assert first.outputs["save"]["path"].resolve() == "mounts/out/result_00001.pgm"

    asyncio.run(scenario())


def test_default_inputs_applied() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={"m": GraphNode("std.math.multiply_floats", {"a": 3.0})}  # b defaults to 1.0
        )
        result = await engine.run(graph, ["m"])
        assert result.outputs["m"]["product"].resolve() == pytest.approx(3.0)

    asyncio.run(scenario())


def test_validation_failure_raises() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(nodes={"x": GraphNode("no.such.type", {})})
        with pytest.raises(GraphValidationError):
            await engine.run(graph, ["x"])

    asyncio.run(scenario())


def test_typed_literal_executes_into_nonconcrete_inputs() -> None:
    """The joint-contract case end to end: typed literals wrap with their
    stamp, so variable-, wildcard- and list<T>-typed inputs execute from
    inline values (and the stamp feeds variable solving downstream)."""
    from dinkster_graph import TypedLiteral

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                # variable T: the stamp binds T=core.int at the worker.
                "m": GraphNode("dev.gallery.match", {"var_in": TypedLiteral("core.int", 7)}),
                # list<T>: a recursively concrete list stamp wraps children.
                "l": GraphNode(
                    "std.list.length",
                    {"list": TypedLiteral("list<core.float>", [1.0, 2.0, 3.0])},
                ),
                # wildcard: the PreviewAny-shaped case that drove the form.
                "g": GraphNode("dev.image.gradient", {"width": 4, "height": 4}),
                "s": GraphNode(
                    "dev.gallery.sockets",
                    {
                        "req_image": Link("g", "image"),
                        "union2": Link("g", "image"),
                        "union3": Link("g", "image"),
                        "union4": Link("g", "image"),
                        "any_in": TypedLiteral("core.int", 7),
                    },
                ),
            }
        )
        result = await engine.run(graph, ["m", "l", "s"])
        assert result.outputs["m"]["var_out"].resolve() == 7
        assert result.outputs["m"]["var_out"].type_id == "core.int"
        assert result.outputs["l"]["length"].resolve() == 3
        assert result.outputs["s"]["image"].type_id == "dev.image"

    asyncio.run(scenario())


def test_typed_literal_unknown_type_fails_validation_not_execution() -> None:
    from dinkster_graph import TypedLiteral

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "m": GraphNode(
                    "dev.gallery.match",
                    {"var_in": TypedLiteral("no.such.type", 7)},
                )
            }
        )
        with pytest.raises(GraphValidationError) as err:
            await engine.run(graph, ["m"])
        assert any(d.code == "typed-literal-unknown-type" for d in err.value.diagnostics)

    asyncio.run(scenario())


def test_runtime_combo_admission_catches_variable_passthrough() -> None:
    """A generic output has no static runtime id, so validation cannot see
    the hidden string stamp. Engine admission rejects it before core.combo
    node code runs; a plain literal on the concrete combo input stays legal."""

    class StringSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.runtime_string",
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(value="string-payload")

    class GenericPassthrough(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            generic = TypeExpr.variable("T")
            return NodeSchema(
                node_type="test.runtime_generic",
                inputs=(InputSpec("value", generic),),
                outputs=(OutputSpec("value", generic),),
            )

        @classmethod
        def execute(cls, *, value: object) -> Mapping[str, object]:
            return cls.outputs(value=value)

    class ComboTap(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            combo = TypeExpr.concrete(CORE_COMBO)
            return NodeSchema(
                node_type="test.runtime_combo",
                inputs=(InputSpec("value", combo),),
                outputs=(OutputSpec("value", combo),),
            )

        @classmethod
        def execute(cls, *, value: str) -> Mapping[str, object]:
            return cls.outputs(value=value)

    nodes = (StringSource, GenericPassthrough, ComboTap)
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorkerFactoryFor(nodes, registry),
        cache=MemoryLRUCache(),
    )

    async def scenario() -> None:
        literal = await engine.run(
            Graph(nodes={"combo": GraphNode("test.runtime_combo", {"value": "a"})}),
            ["combo"],
        )
        assert literal.outputs["combo"]["value"].type_id == CORE_COMBO
        assert literal.outputs["combo"]["value"].resolve() == "a"

        hidden = Graph(
            nodes={
                "source": GraphNode("test.runtime_string", {}),
                "generic": GraphNode(
                    "test.runtime_generic",
                    {"value": Link("source", "value")},
                ),
                "combo": GraphNode(
                    "test.runtime_combo",
                    {"value": Link("generic", "value")},
                ),
            }
        )
        with pytest.raises(ExecutionError, match="explicit converter"):
            await engine.run(hidden, ["combo"])

    asyncio.run(scenario())


def test_combo_list_input_forms_converge_on_one_envelope_and_cache_identity() -> None:
    from dinkster_graph import TypedLiteral

    values = ["b", "outside-current-options", "b"]
    combo_list = TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))

    class ComboListSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_list_source",
                outputs=(OutputSpec("values", combo_list),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(values=list(values))

    class ComboListEcho(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.combo_list_echo",
                inputs=(
                    InputSpec(
                        "values",
                        combo_list,
                        required=False,
                        default=list(values),
                        widget=MultiComboWidget(options=("b", "a", "b")),
                    ),
                ),
                outputs=(OutputSpec("values", combo_list),),
            )

        @classmethod
        def execute(cls, *, values: list[str]) -> Mapping[str, object]:
            return cls.outputs(values=values)

    nodes = (ComboListSource, ComboListEcho)
    registry = TypeRegistry()
    register_core_types(registry)
    engine = Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorkerFactoryFor(nodes, registry),
        cache=MemoryLRUCache(),
    )

    graphs = (
        Graph(nodes={"echo": GraphNode("test.combo_list_echo", {})}),
        Graph(nodes={"echo": GraphNode("test.combo_list_echo", {"values": values})}),
        Graph(
            nodes={
                "echo": GraphNode(
                    "test.combo_list_echo",
                    {"values": TypedLiteral("list<core.combo>", values)},
                )
            }
        ),
        Graph(
            nodes={
                "source": GraphNode("test.combo_list_source", {}),
                "echo": GraphNode("test.combo_list_echo", {"values": Link("source", "values")}),
            }
        ),
    )

    async def scenario() -> None:
        envelopes = [
            (await engine.run(graph, ["echo"])).outputs["echo"]["values"] for graph in graphs
        ]
        assert [value.resolve() for value in envelopes] == [values] * len(graphs)
        assert {value.type_id for value in envelopes} == {"list<core.combo>"}
        assert len({value.fingerprint for value in envelopes}) == 1

        replay = await engine.run(graphs[1], ["echo"])
        assert replay.cached == ("echo",)
        assert replay.outputs["echo"]["values"].fingerprint == envelopes[0].fingerprint

    asyncio.run(scenario())


def InProcessWorkerFactoryFor(nodes: tuple[type[Node], ...], registry: TypeRegistry):
    from dinkster_workers import InProcessWorker

    return InProcessWorker(build_node_types(nodes), registry)


def test_node_exception_becomes_execution_error() -> None:
    async def scenario() -> None:
        engine = make_engine()
        # width=-5 makes numpy raise inside execute()
        graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": -5, "height": 4})})
        with pytest.raises(ExecutionError) as excinfo:
            await engine.run(graph, ["g"])
        assert excinfo.value.error.node_id == "g"

    asyncio.run(scenario())


def _misses(events: list[EngineEvent]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for event in events:
        if event.kind == "cache_miss":
            assert event.node_id is not None
            found[event.node_id] = dict(event.detail)
    return found


def test_cache_miss_explanations() -> None:
    """Dev-mode misses name their cause from the key's composite parts
    (DESIGN 3.9), not two opaque hashes."""

    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events, explain_misses=True)

        await engine.run(image_graph(), ["s"])
        first = _misses(events)
        assert set(first) == {"g", "i", "b", "s"}
        assert all(d["reason"] == "first-seen" for d in first.values())
        assert all("cache_key" in d for d in first.values())

        events.clear()
        await engine.run(image_graph(), ["s"])
        assert _misses(events) == {}  # all hits: nothing to explain

        events.clear()
        await engine.run(image_graph(ratio=0.9), ["s"])
        third = _misses(events)
        # Only the dirty suffix misses; each miss names the changed input.
        assert set(third) == {"b", "s"}
        assert third["b"]["reason"] == "inputs-changed"
        assert third["b"]["changed_inputs"] == ["ratio"]
        assert third["s"]["reason"] == "inputs-changed"
        assert third["s"]["changed_inputs"] == ["image"]

    asyncio.run(scenario())


def test_cache_miss_never_cacheable_and_evicted(tmp_path, monkeypatch) -> None:
    grant_save_mount(tmp_path, monkeypatch)

    async def scenario() -> None:
        events: list[EngineEvent] = []
        # One-entry cache: each stored node evicts the previous one.
        engine = make_engine(events, explain_misses=True, cache=MemoryLRUCache(1))
        graph = save_graph()
        await engine.run(graph, ["save"])
        assert _misses(events)["save"]["reason"] == "never-cacheable"

        events.clear()
        await engine.run(graph, ["save"])
        second = _misses(events)
        # Same components as last run, but the one-slot cache lost g's
        # entry: the explanation blames the store, not the computation.
        assert second["g"]["reason"] == "evicted"
        assert second["save"]["reason"] == "never-cacheable"

    asyncio.run(scenario())


def test_cache_misses_silent_by_default() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        await engine.run(image_graph(), ["s"])
        assert _misses(events) == {}

    asyncio.run(scenario())


def test_events_emitted_in_order() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        await engine.run(image_graph(), ["s"])
        kinds = [e.kind for e in events]
        assert kinds[0] == "run_started"
        assert kinds[-1] == "run_finished"
        assert kinds.count("node_finished") == 4

    asyncio.run(scenario())


def test_node_finished_and_cached_carry_output_summary() -> None:
    """node_finished/node_cached details carry {outputId: {typeId, length?,
    value?}} so frontends render live type/element-count badges on
    intermediate edges mid-run. length is present exactly for list values;
    value is the declared-safe inline scalar (DESIGN 3.5) - lists never
    inline (their elements would through descriptor recursion)."""

    graph = Graph(
        nodes={
            "x": GraphNode("std.math.add_ints", {"a": 3, "b": 0}),
            "y": GraphNode("std.math.add_ints", {"a": 4, "b": 0}),
            "mk": GraphNode(
                "std.list.make",
                {"items.a": Link("x", "sum"), "items.b": Link("y", "sum")},
            ),
            "len": GraphNode("std.list.length", {"list": Link("mk", "list")}),
        }
    )

    def summaries(events: list[EngineEvent], kind: str) -> dict[str, object]:
        return {e.node_id or "": e.detail["outputs"] for e in events if e.kind == kind}

    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        await engine.run(graph, ["len"])
        fresh = summaries(events, "node_finished")
        assert fresh["x"] == {"sum": {"typeId": "core.int", "value": 3}}
        assert fresh["mk"] == {"list": {"typeId": "list<core.int>", "length": 2}}
        assert fresh["len"] == {"length": {"typeId": "core.int", "value": 2}}

        events.clear()
        await engine.run(graph, ["len"])  # everything cached
        hits = summaries(events, "node_cached")
        assert hits["x"] == {"sum": {"typeId": "core.int", "value": 3}}
        assert hits["mk"] == {"list": {"typeId": "list<core.int>", "length": 2}}
        assert hits["len"] == {"length": {"typeId": "core.int", "value": 2}}

    asyncio.run(scenario())


def test_caller_supplied_run_identity_flows_through_events_result_and_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server's queue owns job identity: a supplied run_id must appear on
    the result and on every emitted event, with no side channel."""

    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        invocations: list[Invocation] = []
        invoke = engine._worker.invoke

        async def recording_invoke(
            invocation: Invocation, on_event: OnInvocationEvent | None = None
        ) -> InvocationResult:
            invocations.append(invocation)
            return await invoke(invocation, on_event)

        monkeypatch.setattr(engine._worker, "invoke", recording_invoke)
        result = await engine.run(image_graph(), ["s"], run_id="job-abc123", attempt_id=4)
        assert result.run_id == "job-abc123"
        assert {e.run_id for e in events} == {"job-abc123"}
        assert invocations
        assert {(invocation.job_ref, invocation.attempt_id) for invocation in invocations} == {
            ("job-abc123", 4)
        }

    asyncio.run(scenario())


def test_default_run_ids_are_unique() -> None:
    async def scenario() -> None:
        engine = make_engine()
        first = await engine.run(image_graph(), ["s"])
        second = await engine.run(image_graph(), ["s"])
        assert first.run_id
        assert first.run_id != second.run_id

    asyncio.run(scenario())


def test_empty_run_id_rejected() -> None:
    async def scenario() -> None:
        engine = make_engine()
        with pytest.raises(ValueError, match="run_id"):
            await engine.run(image_graph(), ["s"], run_id="")

    asyncio.run(scenario())


@pytest.mark.parametrize("attempt_id", (0, -1, True, 1.5, "1"))
def test_invalid_attempt_id_rejected(attempt_id: object) -> None:
    async def scenario() -> None:
        with pytest.raises(ValueError, match="attempt_id"):
            await make_engine().run(image_graph(), ["s"], attempt_id=attempt_id)  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_concurrent_duplicate_run_id_is_rejected() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = make_engine(events)
        first = asyncio.create_task(engine.run(image_graph(), ["s"], run_id="shared"))
        while not any(event.kind == "run_started" for event in events):
            await asyncio.sleep(0)
        with pytest.raises(ActiveRunIdError, match="already active"):
            await engine.run(image_graph(), ["s"], run_id="shared")
        assert (await first).run_id == "shared"

    asyncio.run(scenario())


def test_raising_listener_is_nonfatal_for_owner_and_coalesced_waiter() -> None:
    async def scenario() -> None:
        def raising_listener(_event: EngineEvent) -> None:
            raise RuntimeError("listener broke")

        engine = make_engine()
        engine._on_event = raising_listener
        owner, waiter = await asyncio.gather(
            engine.run(image_graph(), ["s"]),
            engine.run(image_graph(), ["s"]),
        )
        assert owner.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
        assert waiter.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
        assert owner.executed or waiter.executed
        assert owner.cached or waiter.cached

    asyncio.run(scenario())


def test_announce_schemas_is_additive_and_atomic() -> None:
    """Progressive announcement grows the engine's surface additively:
    redefining an existing type is refused, and a refused delta merges
    NOTHING - announcing the non-colliding half afterwards succeeds,
    which it could not if the failed call had half-applied."""
    from dinkster_schema import InputSpec, NodeSchema, OutputSpec, TypeExpr

    engine = make_engine()
    late = NodeSchema(
        node_type="late.echo",
        inputs=(InputSpec("text", TypeExpr.concrete("core.string")),),
        outputs=(OutputSpec("out", TypeExpr.concrete("core.string")),),
    )
    existing = build_schemas(SCAFFOLD_NODES)["dev.image.gradient"]

    with pytest.raises(ValueError, match="redefine"):
        engine.announce_schemas({"late.echo": late, "dev.image.gradient": existing})
    engine.announce_schemas({"late.echo": late})  # atomic: nothing merged above
    with pytest.raises(ValueError, match="redefine"):
        engine.announce_schemas({"late.echo": late})
