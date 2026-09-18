from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields, replace
from typing import cast

import numpy as np
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import (
    ActiveRunIdError,
    AdmittedGraph,
    CompiledGraph,
    Engine,
    EngineEvent,
    ExecutionRuntime,
    GraphAdmissionError,
    GraphAdmissionOrigin,
    GraphCompileError,
    GraphValidationError,
    ParentGraphBudget,
    VirtualGraphJoin,
    admit_parent_graph,
)
from dinkster_graph import Graph, GraphNode, Link, RegionNode, graph_to_wire
from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_ERROR_DEPTH_LIMIT,
    GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
    GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
    GRAPH_COMPILE_ERROR_ID_COLLISION,
    GRAPH_COMPILE_ERROR_ID_FORMAT,
    GRAPH_COMPILE_ERROR_LINK_LIMIT,
    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
    GRAPH_COMPILE_ERROR_NODE_LIMIT,
    GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
    GRAPH_COMPILE_ERROR_PASS_LIMIT,
    GRAPH_COMPILE_ERROR_REPLY_OVERSIZE,
    GRAPH_COMPILE_ERROR_SELECTOR_EMITTED,
    GRAPH_COMPILE_ERROR_SELECTOR_INPUT,
    GRAPH_COMPILE_ERROR_TARGET_MISMATCH,
    GRAPH_COMPILE_ERROR_TIMEOUT,
    GRAPH_COMPILE_MAX_NODES,
    GRAPH_COMPILE_MAX_REPLY_BYTES,
    GRAPH_COMPILE_RESULT_TYPE,
    GRAPH_COMPILERS_SURFACE,
    GraphCompilerRegistrySnapshot,
    KeyedContribution,
    generated_node_id,
)
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types


def _compiler_registry() -> GraphCompilerRegistrySnapshot:
    contribution = KeyedContribution(
        surface_id=GRAPH_COMPILERS_SURFACE,
        id="test.compiler",
        behavior_metadata=(("contractVersion", 1), ("order", 0)),
    )
    return GraphCompilerRegistrySnapshot((contribution,))


def _engine(events: list[EngineEvent] | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    return Engine(
        schemas=build_schemas(SCAFFOLD_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(SCAFFOLD_NODES), registry),
        cache=MemoryLRUCache(),
        on_event=events.append if events is not None else None,
    )


def _graph() -> Graph:
    return Graph({"g": GraphNode("dev.image.gradient", {"width": 4, "height": 2})})


def _success(
    digest: str,
    graph: Graph,
    targets: tuple[str, ...] = ("g",),
    **changes: object,
) -> dict[str, object]:
    reply: dict[str, object] = {
        "type": GRAPH_COMPILE_RESULT_TYPE,
        "requestId": "request-1",
        "blobs": [],
        "generationKey": digest,
        "graph": graph_to_wire(graph),
        "targets": list(targets),
        "passCount": 0,
        "attemptedGeneratedCounts": [],
        "origins": [],
    }
    reply.update(changes)
    return reply


def _runtime(engine: Engine, transport) -> ExecutionRuntime:
    return replace(
        engine.pin_execution(),
        graph_compiler_registry=_compiler_registry(),
        graph_compile_transport=transport,
    )


def test_execution_runtime_compiler_invariant_defaults_schema_normalization_and_replace() -> None:
    engine = _engine()
    base = engine.pin_execution()
    assert base.graph_compiler_registry == GraphCompilerRegistrySnapshot()
    assert base.graph_compile_transport is None
    assert base.schemas is not None

    async def transport(*_args):
        return {}

    with pytest.raises(ValueError, match="present together"):
        replace(base, graph_compile_transport=transport)
    with pytest.raises(ValueError, match="present together"):
        replace(base, graph_compiler_registry=_compiler_registry())
    raw = replace(base, schemas=None)
    normalized = engine._normalize_execution(raw)
    assert normalized.schemas is not None
    assert replace(normalized, schemas=dict(normalized.schemas)).schemas == normalized.schemas
    assert engine._normalize_execution(replace(base, schemas={})).schemas == base.schemas
    with pytest.raises(FrozenInstanceError):
        normalized.schemas = {}  # type: ignore[misc]


def test_compiled_graph_is_deeply_immutable() -> None:
    graph = Graph(
        {
            "g": GraphNode(
                "dev.image.gradient",
                {"nested": [{"value": 1}]},
            )
        }
    )
    digest = _engine().extension_snapshot_digest
    sources = ["source"]
    origins = {
        "generated": {
            "nodeId": "generated",
            "sources": sources,
            "compilerId": "test.compiler",
            "passIndex": 0,
            "localKey": "one",
        }
    }
    compiled = CompiledGraph(graph, ["g"], digest, origins)
    cast(dict[str, GraphNode], graph.nodes)["x"] = GraphNode("bad", {})
    sources.append("later")
    assert set(compiled.graph.nodes) == {"g"}
    assert compiled.targets == ("g",)
    assert compiled.origins["generated"]["sources"] == ("source",)
    exposed = cast("list[dict[str, int]]", compiled.graph.nodes["g"].inputs["nested"])
    exposed[0]["value"] = 2
    assert compiled.graph.nodes["g"].inputs["nested"] == [{"value": 1}]
    with pytest.raises(FrozenInstanceError):
        compiled.targets = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        compiled.origins["x"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        compiled.origins["generated"]["sources"] = ()  # type: ignore[index]


def test_parent_graph_admission_is_deterministic_complete_and_deeply_immutable() -> None:
    parent = _graph()
    parent_wire = graph_to_wire(parent)
    namespace = "parent.expansion"
    node_id = generated_node_id(namespace, ("g",), "invert")
    delta = graph_to_wire(Graph({node_id: GraphNode("dev.image.invert", {})}))
    origins = (GraphAdmissionOrigin(node_id, ("g",), "invert"),)
    joins = (VirtualGraphJoin(node_id, "image", "g", "image"),)

    admitted = admit_parent_graph(
        parent,
        delta,
        namespace=namespace,
        origins=origins,
        virtual_joins=joins,
        budget=ParentGraphBudget(),
        schemas=build_schemas(SCAFFOLD_NODES),
    )
    repeated = admit_parent_graph(
        parent,
        delta,
        namespace=namespace,
        origins=origins,
        virtual_joins=joins,
        budget=ParentGraphBudget(),
        schemas=build_schemas(SCAFFOLD_NODES),
    )

    assert isinstance(admitted, AdmittedGraph)
    assert graph_to_wire(admitted.graph) == graph_to_wire(repeated.graph)
    assert admitted.origins[node_id].sources == ("g",)
    assert admitted.graph.nodes[node_id].inputs["image"] == Link("g", "image")
    assert graph_to_wire(parent) == parent_wire
    cast(dict[str, object], delta["nodes"])[node_id] = {
        "nodeType": "bad",
        "inputs": {},
    }
    assert cast(GraphNode, admitted.graph.nodes[node_id]).node_type == "dev.image.invert"
    with pytest.raises(TypeError):
        admitted.origins[node_id] = origins[0]  # type: ignore[index]


@pytest.mark.parametrize(
    ("delta", "origins", "joins", "budget", "code"),
    [
        (
            {"nodes": {"g": {"nodeType": "dev.image.invert", "inputs": {}}}},
            (GraphAdmissionOrigin("g", ("g",), "mutate"),),
            (),
            ParentGraphBudget(),
            "id-collision",
        ),
        (
            {"nodes": {"$gen-0000000000000000": {"nodeType": "dev.image.invert", "inputs": {}}}},
            (),
            (),
            ParentGraphBudget(),
            "origin-coverage",
        ),
        (
            {"nodes": {"$gen-0000000000000000": {"nodeType": "dev.image.invert", "inputs": {}}}},
            (
                GraphAdmissionOrigin("$gen-0000000000000000", ("g",), "duplicate"),
                GraphAdmissionOrigin("$gen-0000000000000000", ("g",), "duplicate"),
            ),
            (),
            ParentGraphBudget(),
            "id-collision",
        ),
        (
            {"nodes": {"$gen-0000000000000000": {"nodeType": "dev.image.invert", "inputs": {}}}},
            (GraphAdmissionOrigin("$gen-0000000000000000", ("g",), "wrong"),),
            (),
            ParentGraphBudget(),
            "id-format",
        ),
        (
            {
                "nodes": {
                    "$gen-0000000000000000": {
                        "nodeType": "dev.image.invert",
                        "inputs": {"image": {"$link": {"node": "g", "output": "image"}}},
                    }
                }
            },
            (GraphAdmissionOrigin("$gen-0000000000000000", ("g",), "direct"),),
            (),
            ParentGraphBudget(),
            "cross-boundary-link",
        ),
        (
            {"nodes": {}},
            (),
            (),
            ParentGraphBudget(max_header_bytes=1),
            "header-budget",
        ),
    ],
)
def test_parent_graph_admission_refuses_mutation_origin_cross_boundary_and_header(
    delta: dict[str, object],
    origins: tuple[GraphAdmissionOrigin, ...],
    joins: tuple[VirtualGraphJoin, ...],
    budget: ParentGraphBudget,
    code: str,
) -> None:
    parent = _graph()
    before = graph_to_wire(parent)
    with pytest.raises(GraphAdmissionError) as raised:
        admit_parent_graph(
            parent,
            delta,
            namespace="parent.expansion",
            origins=origins,
            virtual_joins=joins,
            budget=budget,
            schemas=build_schemas(SCAFFOLD_NODES),
        )
    assert raised.value.code == code
    assert graph_to_wire(parent) == before


def test_parent_graph_admission_refuses_join_cycles_interfaces_types_and_budgets() -> None:
    parent = _graph()
    namespace = "parent.expansion"
    first = generated_node_id(namespace, ("g",), "first")
    second = generated_node_id(namespace, ("g",), "second")
    schemas = build_schemas(SCAFFOLD_NODES)

    def refuse(
        delta: Graph,
        origins: tuple[GraphAdmissionOrigin, ...],
        joins: tuple[VirtualGraphJoin, ...],
        budget: ParentGraphBudget | None = None,
    ) -> GraphAdmissionError:
        before = graph_to_wire(parent)
        with pytest.raises(GraphAdmissionError) as raised:
            admit_parent_graph(
                parent,
                graph_to_wire(delta),
                namespace=namespace,
                origins=origins,
                virtual_joins=joins,
                budget=budget or ParentGraphBudget(),
                schemas=schemas,
            )
        assert graph_to_wire(parent) == before
        return raised.value

    two_origins = (
        GraphAdmissionOrigin(first, ("g",), "first"),
        GraphAdmissionOrigin(second, ("g",), "second"),
    )
    cycle = refuse(
        Graph(
            {
                first: GraphNode("dev.image.invert", {}),
                second: GraphNode("dev.image.invert", {}),
            }
        ),
        two_origins,
        (
            VirtualGraphJoin(first, "image", second, "image"),
            VirtualGraphJoin(second, "image", first, "image"),
        ),
    )
    assert cycle.code == "graph-validation"
    assert {diagnostic.code for diagnostic in cycle.diagnostics} == {"cycle"}

    missing_output = refuse(
        Graph({first: GraphNode("dev.image.invert", {})}),
        (two_origins[0],),
        (VirtualGraphJoin(first, "image", "g", "missing"),),
    )
    assert missing_output.code == "graph-validation"
    assert "dangling-output" in {diagnostic.code for diagnostic in missing_output.diagnostics}

    mismatch = refuse(
        Graph({first: GraphNode("dev.image.gradient", {"height": 1})}),
        (two_origins[0],),
        (VirtualGraphJoin(first, "width", "g", "image"),),
    )
    assert mismatch.code == "graph-validation"
    assert "type-mismatch" in {diagnostic.code for diagnostic in mismatch.diagnostics}

    node_budget = refuse(
        Graph({first: GraphNode("dev.image.invert", {})}),
        (two_origins[0],),
        (VirtualGraphJoin(first, "image", "g", "image"),),
        ParentGraphBudget(max_nodes=1),
    )
    assert node_budget.code == "node-budget"

    edge_budget = refuse(
        Graph({first: GraphNode("dev.image.invert", {})}),
        (two_origins[0],),
        (VirtualGraphJoin(first, "image", "g", "image"),),
        ParentGraphBudget(max_edges=0),
    )
    assert edge_budget.code == "edge-budget"

    recursion_budget = refuse(
        Graph(
            {
                second: RegionNode(
                    kind="map",
                    body=Graph({first: GraphNode("dev.image.invert", {})}),
                )
            }
        ),
        (),
        (),
        ParentGraphBudget(max_depth=0),
    )
    assert recursion_budget.code == "recursion-budget"


def test_parent_graph_admission_malformed_values_always_return_typed_refusals() -> None:
    parent = _graph()
    schemas = build_schemas(SCAFFOLD_NODES)

    with pytest.raises(GraphAdmissionError) as bad_namespace:
        admit_parent_graph(
            parent,
            {"nodes": {}},
            namespace="unqualified",
            origins=(),
            virtual_joins=(),
            budget=ParentGraphBudget(),
            schemas=schemas,
        )
    assert bad_namespace.value.code == "id-format"

    namespace = "parent.expansion"
    node_id = generated_node_id(namespace, ("g",), "invert")
    with pytest.raises(GraphAdmissionError) as bad_join:
        admit_parent_graph(
            parent,
            graph_to_wire(Graph({node_id: GraphNode("dev.image.invert", {})})),
            namespace=namespace,
            origins=(GraphAdmissionOrigin(node_id, ("g",), "invert"),),
            virtual_joins=(VirtualGraphJoin(node_id, cast(str, 1), "g", "image"),),
            budget=ParentGraphBudget(),
            schemas=schemas,
        )
    assert bad_join.value.code == "virtual-join"

    with pytest.raises(GraphAdmissionError) as bad_origin:
        admit_parent_graph(
            parent,
            graph_to_wire(Graph({node_id: GraphNode("dev.image.invert", {})})),
            namespace=namespace,
            origins=(GraphAdmissionOrigin(cast(str, []), ("g",), "invert"),),
            virtual_joins=(),
            budget=ParentGraphBudget(),
            schemas=schemas,
        )
    assert bad_origin.value.code == "origin-coverage"

    nested: object = 0
    for _ in range(1200):
        nested = [nested]
    malformed_literal = {
        "nodes": {
            node_id: {
                "nodeType": "dev.image.invert",
                "inputs": {"image": nested},
            }
        }
    }
    with pytest.raises(GraphAdmissionError) as deep_literal:
        admit_parent_graph(
            parent,
            malformed_literal,
            namespace=namespace,
            origins=(GraphAdmissionOrigin(node_id, ("g",), "invert"),),
            virtual_joins=(),
            budget=ParentGraphBudget(),
            schemas=schemas,
        )
    assert deep_literal.value.code == "malformed-delta"

    deep_region: object = {"nodes": {}}
    for _ in range(1200):
        deep_region = {
            "nodes": {
                node_id: {
                    "region": {
                        "kind": "map",
                        "ports": {},
                        "inputs": {},
                        "body": deep_region,
                        "outputs": {},
                    }
                }
            }
        }
    with pytest.raises(GraphAdmissionError) as deep_graph:
        admit_parent_graph(
            parent,
            cast(dict[str, object], deep_region),
            namespace=namespace,
            origins=(),
            virtual_joins=(),
            budget=ParentGraphBudget(),
            schemas=schemas,
        )
    assert deep_graph.value.code == "recursion-budget"


def test_empty_registry_run_skips_compile_and_preserves_outputs_cache_and_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[EngineEvent] = []
    engine = _engine(events)

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("compile_for_execution must not be called")

    monkeypatch.setattr(engine, "compile_for_execution", forbidden)

    async def scenario() -> None:
        first = await engine.run(_graph(), ["g"], run_id="first")
        second = await engine.run(_graph(), ["g"], run_id="second")
        assert first.executed == ("g",) and first.cached == ()
        assert second.executed == () and second.cached == ("g",)
        assert first.outputs["g"]["image"].fingerprint == second.outputs["g"]["image"].fingerprint
        started = [event for event in events if event.kind == "run_started"]
        assert [event.detail["planned"] for event in started] == [["g"], ["g"]]
        assert [(event.kind, event.node_id) for event in events if event.node_id] == [
            ("node_started", "g"),
            ("node_finished", "g"),
            ("node_cached", "g"),
        ]

    asyncio.run(scenario())


def test_empty_registry_run_id_refusal_still_precedes_runtime_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()

    def forbidden_pin():
        raise AssertionError("legacy run-id refusal must happen before pinning")

    monkeypatch.setattr(engine, "pin_execution", forbidden_pin)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await engine.run(_graph(), ["g"], run_id="")

    asyncio.run(scenario())


def test_empty_registry_compile_returns_typed_snapshot_without_rpc() -> None:
    async def scenario() -> None:
        engine = _engine()
        graph = _graph()
        compiled = await engine.compile_for_execution(graph, ["g"])
        cast(dict[str, GraphNode], graph.nodes)["late"] = GraphNode("bad", {})
        assert isinstance(compiled, CompiledGraph)
        assert set(compiled.graph.nodes) == {"g"}
        assert compiled.origins == {}

    asyncio.run(scenario())


def test_empty_registry_run_preserves_submitted_generated_prefix_id() -> None:
    async def scenario() -> None:
        events: list[EngineEvent] = []
        engine = _engine(events)
        node_id = "$gen-submitted"
        graph = Graph({node_id: GraphNode("dev.image.gradient", {"width": 1, "height": 1})})
        result = await engine.run(graph, [node_id])
        assert tuple(result.outputs) == (node_id,)
        assert result.executed == (node_id,)
        assert [event.detail["planned"] for event in events if event.kind == "run_started"] == [
            [node_id]
        ]
        assert [event.node_id for event in events if event.node_id is not None] == [
            node_id,
            node_id,
        ]

    asyncio.run(scenario())


def test_nonempty_run_calls_transport_once_exactly_and_reuses_one_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = _engine()
        calls: list[tuple[str, object, tuple[str, ...]]] = []

        async def transport(digest, wire, targets):
            calls.append((digest, wire, tuple(targets)))
            changed = _graph()
            cast(dict[str, GraphNode], changed.nodes)["g"] = GraphNode(
                "dev.image.gradient", {"width": 2, "height": 2}
            )
            return _success(
                digest,
                changed,
                passCount=1,
                attemptedGeneratedCounts=[0],
            )

        runtime = _runtime(engine, transport)
        pins = 0

        def forbidden_pin():
            nonlocal pins
            pins += 1
            raise AssertionError("already pinned runtime must be reused")

        monkeypatch.setattr(engine, "pin_execution", forbidden_pin)
        result = await engine.run(_graph(), ["g"], execution=runtime)
        image = cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.float32]]",
            result.outputs["g"]["image"].resolve(),
        )
        assert image.shape[:2] == (2, 2)
        assert calls == [(runtime.extension_snapshot_digest, graph_to_wire(_graph()), ("g",))]
        assert pins == 0

    asyncio.run(scenario())


def test_nonempty_run_reserves_run_id_while_compiling() -> None:
    async def scenario() -> None:
        engine = _engine()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def transport(digest, _wire, targets):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return _success(digest, _graph(), tuple(targets))

        runtime = _runtime(engine, transport)
        first = asyncio.create_task(engine.run(_graph(), ["g"], run_id="shared", execution=runtime))
        await entered.wait()
        with pytest.raises(ActiveRunIdError):
            await engine.run(_graph(), ["g"], run_id="shared", execution=runtime)
        assert calls == 1
        release.set()
        assert (await first).executed == ("g",)

    asyncio.run(scenario())


def test_raw_mutation_after_transport_entry_cannot_change_request_or_compiled_bytes() -> None:
    async def scenario() -> None:
        engine = _engine()
        graph = _graph()
        entered = asyncio.Event()
        release = asyncio.Event()
        seen: dict[str, object] = {}

        async def transport(digest, wire, targets):
            seen["wire"] = wire
            entered.set()
            await release.wait()
            return _success(digest, _graph(), tuple(targets))

        task = asyncio.create_task(
            engine.compile_for_execution(graph, ["g"], execution=_runtime(engine, transport))
        )
        await entered.wait()
        cast(dict[str, GraphNode], graph.nodes)["late"] = GraphNode("bad", {})
        release.set()
        compiled = await task
        assert seen["wire"] == graph_to_wire(_graph())
        assert graph_to_wire(compiled.graph) == graph_to_wire(_graph())

    asyncio.run(scenario())


def test_run_compiled_matches_digest_never_recompiles_and_refuses_drift_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = _engine()
        runtime = engine.pin_execution()
        compiled = CompiledGraph(_graph(), ("g",), runtime.extension_snapshot_digest, {})

        async def forbidden(*_args, **_kwargs):
            raise AssertionError("precompiled execution must not compile")

        monkeypatch.setattr(engine, "compile_for_execution", forbidden)
        result = await engine.run_compiled(compiled, execution=runtime)
        assert result.executed == ("g",)
        drift = CompiledGraph(
            compiled.graph,
            compiled.targets,
            "sha256:" + "f" * 64,
            compiled.origins,
        )
        with pytest.raises(GraphCompileError) as caught:
            await engine.run_compiled(drift, run_id="", execution=runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_GENERATION_MISMATCH

    asyncio.run(scenario())


def test_recursive_input_selector_generated_prefix_and_output_selector_refusals() -> None:
    async def scenario() -> None:
        engine = _engine()
        base = engine.pin_execution()
        assert base.schemas is not None
        integer = TypeExpr.concrete("core.int")
        selector = NodeSchema(
            "test.selector",
            inputs=(
                InputSpec("switch", TypeExpr.concrete("core.boolean")),
                InputSpec("off", integer),
                InputSpec("on", integer),
            ),
            outputs=(OutputSpec("out", integer),),
            selector=SelectorSpec("switch", {"false": "off", "true": "on"}),
        )
        region = RegionNode("map", Graph({"inside": GraphNode(selector.node_type, {})}))
        schemas = {**base.schemas, selector.node_type: selector}

        async def unchanged(digest, _wire, targets):
            return _success(digest, Graph({"r": region}), tuple(targets))

        runtime = replace(
            base,
            schemas=schemas,
            graph_compiler_registry=_compiler_registry(),
            graph_compile_transport=unchanged,
        )
        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(Graph({"r": region}), ["r"], execution=runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_SELECTOR_INPUT

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(
                Graph({"$gen-deadbeefdeadbeef": GraphNode("unknown", {})}),
                ["$gen-deadbeefdeadbeef"],
                execution=runtime,
            )
        assert caught.value.code == GRAPH_COMPILE_ERROR_ID_COLLISION

        plain = Graph({"g": GraphNode("dev.image.gradient", {})})

        async def emits_selector(digest, _wire, targets):
            return _success(
                digest,
                Graph({"g": GraphNode(selector.node_type, {})}),
                tuple(targets),
                passCount=1,
                attemptedGeneratedCounts=[0],
            )

        emitted_runtime = replace(runtime, graph_compile_transport=emits_selector)
        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(plain, ["g"], execution=emitted_runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_SELECTOR_EMITTED

    asyncio.run(scenario())


def test_recursive_limits_accept_depth_16_and_refuse_depth_17_and_node_4097() -> None:
    def nested(depth: int) -> Graph:
        graph = Graph({"leaf": GraphNode("unknown", {})})
        for index in range(depth):
            graph = Graph({f"r{index}": RegionNode("map", graph)})
        return graph

    async def scenario() -> None:
        engine = _engine()

        async def unchanged(digest, wire, targets):
            return _success(digest, nested(16), tuple(targets))

        runtime = _runtime(engine, unchanged)
        compiled = await engine.compile_for_execution(nested(16), ["r15"], execution=runtime)
        assert compiled.targets == ("r15",)
        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(nested(17), ["r16"], execution=runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_DEPTH_LIMIT
        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(nested(500), ["r499"], execution=runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_DEPTH_LIMIT

        async def emits_deep(digest, _wire, targets):
            output = Graph({"g": GraphNode("unknown", {}), "deep": nested(17).nodes["r16"]})
            return _success(
                digest,
                output,
                tuple(targets),
                passCount=1,
                attemptedGeneratedCounts=[0],
            )

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(
                Graph({"g": GraphNode("unknown", {})}),
                ["g"],
                execution=_runtime(engine, emits_deep),
            )
        assert caught.value.code == GRAPH_COMPILE_ERROR_DEPTH_LIMIT

        deep_wire: dict[str, object] = {"nodes": {"leaf": {"nodeType": "unknown", "inputs": {}}}}
        for index in range(1500):
            deep_wire = {
                "nodes": {
                    f"r{index}": {
                        "region": {
                            "kind": "map",
                            "ports": {},
                            "inputs": {},
                            "body": deep_wire,
                            "outputs": {},
                        }
                    }
                }
            }

        async def emits_extremely_deep(digest, _wire, targets):
            reply = _success(
                digest,
                Graph({"g": GraphNode("unknown", {})}),
                tuple(targets),
                passCount=1,
                attemptedGeneratedCounts=[0],
            )
            reply["graph"] = deep_wire
            return reply

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(
                Graph({"g": GraphNode("unknown", {})}),
                ["g"],
                execution=_runtime(engine, emits_extremely_deep),
            )
        assert caught.value.code == GRAPH_COMPILE_ERROR_DEPTH_LIMIT
        oversized = Graph(
            {f"n{index}": GraphNode("unknown", {}) for index in range(GRAPH_COMPILE_MAX_NODES + 1)}
        )
        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(oversized, ["n0"], execution=runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_NODE_LIMIT

    asyncio.run(scenario())


def test_link_limit_accepts_16384_and_refuses_16385() -> None:
    async def scenario() -> None:
        engine = _engine()

        async def unchanged(digest, wire, targets):
            return {
                "type": GRAPH_COMPILE_RESULT_TYPE,
                "requestId": "links",
                "blobs": [],
                "generationKey": digest,
                "graph": wire,
                "targets": list(targets),
                "passCount": 0,
                "attemptedGeneratedCounts": [],
                "origins": [],
            }

        runtime = _runtime(engine, unchanged)
        legal = Graph(
            {
                "source": GraphNode("unknown", {}),
                "target": GraphNode(
                    "unknown",
                    {f"i{index}": Link("source", "out") for index in range(16384)},
                ),
            }
        )
        compiled = await engine.compile_for_execution(legal, ["target"], execution=runtime)
        assert compiled.targets == ("target",)
        illegal = Graph(
            {
                "source": GraphNode("unknown", {}),
                "target": GraphNode(
                    "unknown",
                    {f"i{index}": Link("source", "out") for index in range(16385)},
                ),
            }
        )
        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(illegal, ["target"], execution=runtime)
        assert caught.value.code == GRAPH_COMPILE_ERROR_LINK_LIMIT

    asyncio.run(scenario())


def test_pass_generated_origin_identity_coverage_and_zero_pass_change_refusals() -> None:
    async def scenario() -> None:
        engine = _engine()
        source = _graph()
        digest = engine.extension_snapshot_digest
        generated = generated_node_id("test.compiler", ("g",), "one")
        output = Graph({**source.nodes, generated: GraphNode("dev.image.gradient", {})})
        origin = {
            "nodeId": generated,
            "compilerId": "test.compiler",
            "passIndex": 0,
            "sources": ["g"],
            "localKey": "one",
        }
        valid = _success(
            digest,
            output,
            passCount=1,
            attemptedGeneratedCounts=[1],
            origins=[origin],
        )

        async def transport(*_args):
            return valid

        compiled = await engine.compile_for_execution(
            source, ["g"], execution=_runtime(engine, transport)
        )
        assert set(compiled.origins) == {generated}

        async def legal_maxima(*_args):
            return _success(
                digest,
                source,
                passCount=32,
                attemptedGeneratedCounts=[1024] * 32,
            )

        maxima = await engine.compile_for_execution(
            source, ["g"], execution=_runtime(engine, legal_maxima)
        )
        assert maxima.targets == ("g",)

        cases = [
            ({**valid, "origins": []}, GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE),
            ({**valid, "origins": [origin, origin]}, GRAPH_COMPILE_ERROR_ID_COLLISION),
            (
                {**valid, "origins": [{**origin, "localKey": "wrong"}]},
                GRAPH_COMPILE_ERROR_ID_FORMAT,
            ),
            ({**valid, "attemptedGeneratedCounts": [0]}, GRAPH_COMPILE_ERROR_GENERATED_LIMIT),
            (
                {**valid, "attemptedGeneratedCounts": [1025]},
                GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
            ),
            (
                {**valid, "origins": [{**origin, "compilerId": "other.compiler"}]},
                GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
            ),
            (
                {**valid, "origins": [{**origin, "sources": ["bad//path"]}]},
                GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
            ),
            (
                {**valid, "origins": [{**origin, "sources": ["ghost"]}]},
                GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
            ),
            (
                {**valid, "origins": [{**origin, "sources": [generated]}]},
                GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
            ),
            (
                {**valid, "origins": [{**origin, "passIndex": True}]},
                GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
            ),
            (
                {
                    **valid,
                    "origins": [{key: value for key, value in origin.items() if key != "localKey"}],
                },
                GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
            ),
            (
                {**valid, "origins": [{**origin, "nodeId": "g"}]},
                GRAPH_COMPILE_ERROR_ID_COLLISION,
            ),
            (
                {**valid, "passCount": 33, "attemptedGeneratedCounts": [0] * 33},
                GRAPH_COMPILE_ERROR_PASS_LIMIT,
            ),
            (_success(digest, output), GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE),
        ]
        for reply, code in cases:

            async def invalid(*_args, reply=reply):
                return reply

            with pytest.raises(GraphCompileError) as caught:
                await engine.compile_for_execution(
                    source, ["g"], execution=_runtime(engine, invalid)
                )
            assert caught.value.code == code

    asyncio.run(scenario())


def test_strict_generated_region_descendants_require_independent_full_path_origins() -> None:
    async def scenario() -> None:
        engine = _engine()
        source = _graph()
        outer = generated_node_id("test.compiler", ("g",), "outer")
        inner = generated_node_id("test.compiler", ("g",), "inner")
        output = Graph(
            {
                **source.nodes,
                outer: RegionNode("map", Graph({inner: GraphNode("unknown", {})})),
            }
        )
        digest = engine.extension_snapshot_digest
        origins = [
            {
                "nodeId": outer,
                "compilerId": "test.compiler",
                "passIndex": 0,
                "sources": ["g"],
                "localKey": "outer",
            }
        ]

        async def transport(*_args):
            return _success(
                digest,
                output,
                passCount=1,
                attemptedGeneratedCounts=[2],
                origins=origins,
            )

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(source, ["g"], execution=_runtime(engine, transport))
        assert caught.value.code == GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE
        origins.append(
            {
                "nodeId": f"{outer}/{inner}",
                "compilerId": "test.compiler",
                "passIndex": 0,
                "sources": ["g"],
                "localKey": "inner",
            }
        )
        compiled = await engine.compile_for_execution(
            source, ["g"], execution=_runtime(engine, transport)
        )
        assert set(compiled.origins) == {outer, f"{outer}/{inner}"}

    asyncio.run(scenario())


def test_target_order_generation_framing_blobs_unknown_keys_and_non_json_are_checked() -> None:
    async def scenario() -> None:
        engine = _engine()
        digest = engine.extension_snapshot_digest
        graph = Graph({"a": GraphNode("unknown", {}), "b": GraphNode("unknown", {})})
        base = _success(digest, graph, ("a", "b"))
        cases = [
            ({**base, "targets": ["b", "a"]}, GRAPH_COMPILE_ERROR_TARGET_MISMATCH),
            (
                {**base, "graph": graph_to_wire(Graph({"a": GraphNode("unknown", {})}))},
                GRAPH_COMPILE_ERROR_TARGET_MISMATCH,
            ),
            (
                {**base, "generationKey": "sha256:" + "f" * 64},
                GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
            ),
            ({**base, "blobs": [b"x"]}, GRAPH_COMPILE_ERROR_MALFORMED_REPLY),
            (
                {key: value for key, value in base.items() if key != "blobs"},
                GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
            ),
            ({**base, "extra": True}, GRAPH_COMPILE_ERROR_MALFORMED_REPLY),
            ({**base, "graph": {"nodes": {}, "bad": {1}}}, GRAPH_COMPILE_ERROR_MALFORMED_REPLY),
            ({**base, "graph": {"value": float("nan")}}, GRAPH_COMPILE_ERROR_MALFORMED_REPLY),
        ]
        for reply, code in cases:

            async def transport(*_args, reply=reply):
                return reply

            runtime = _runtime(engine, transport)
            with pytest.raises(GraphCompileError) as caught:
                await engine.compile_for_execution(graph, ["a", "b"], execution=runtime)
            assert caught.value.code == code

        async def accepts_blobs(*_args):
            return {**base, "blobs": []}

        compiled = await engine.compile_for_execution(
            graph, ["a", "b"], execution=_runtime(engine, accepts_blobs)
        )
        assert compiled.targets == ("a", "b")

    asyncio.run(scenario())


def test_full_envelope_canonical_size_limit_is_measured_before_framing() -> None:
    async def scenario() -> None:
        engine = _engine()

        async def oversized(*_args):
            return {
                "type": "wrong",
                "requestId": "x" * GRAPH_COMPILE_MAX_REPLY_BYTES,
            }

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(
                _graph(), ["g"], execution=_runtime(engine, oversized)
            )
        assert caught.value.code == GRAPH_COMPILE_ERROR_REPLY_OVERSIZE

    asyncio.run(scenario())


def test_remote_errors_transport_failure_timeout_and_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = _engine()
        digest = engine.extension_snapshot_digest

        for name, expected in [
            (GRAPH_COMPILE_ERROR_NODE_LIMIT, GRAPH_COMPILE_ERROR_NODE_LIMIT),
            ("future-remote-error", GRAPH_COMPILE_ERROR_COMPILER_FAILURE),
        ]:

            async def remote(*_args, name=name):
                return {
                    "type": GRAPH_COMPILE_RESULT_TYPE,
                    "requestId": "r",
                    "blobs": [],
                    "errorName": name,
                    "error": "remote detail",
                }

            with pytest.raises(GraphCompileError) as caught:
                await engine.compile_for_execution(
                    _graph(), ["g"], execution=_runtime(engine, remote)
                )
            assert caught.value.code == expected
            assert "remote detail" in str(caught.value)

        marker = RuntimeError("transport detail")

        async def fails(*_args):
            raise marker

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(_graph(), ["g"], execution=_runtime(engine, fails))
        assert caught.value.code == GRAPH_COMPILE_ERROR_COMPILER_FAILURE
        assert caught.value.__cause__ is marker

        import dinkster_engine.compile as compile_module

        monkeypatch.setattr(compile_module, "GRAPH_COMPILE_TIMEOUT_SECONDS", 0.001)

        async def sleeps(*_args):
            await asyncio.sleep(10)
            return _success(digest, _graph())

        with pytest.raises(GraphCompileError) as caught:
            await engine.compile_for_execution(_graph(), ["g"], execution=_runtime(engine, sleeps))
        assert caught.value.code == GRAPH_COMPILE_ERROR_TIMEOUT

        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def waits(*_args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(
            engine.compile_for_execution(_graph(), ["g"], execution=_runtime(engine, waits))
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_returned_unknown_node_type_is_graph_validation_not_malformed_reply() -> None:
    async def scenario() -> None:
        engine = _engine()

        async def transport(digest, _wire, targets):
            return _success(
                digest,
                Graph({"g": GraphNode("unknown.returned", {})}),
                tuple(targets),
                passCount=1,
                attemptedGeneratedCounts=[0],
            )

        with pytest.raises(GraphValidationError):
            await engine.run(_graph(), ["g"], execution=_runtime(engine, transport))

    asyncio.run(scenario())


def test_engine_event_public_schema_is_unchanged() -> None:
    assert [field.name for field in fields(EngineEvent)] == ["kind", "run_id", "node_id", "detail"]
