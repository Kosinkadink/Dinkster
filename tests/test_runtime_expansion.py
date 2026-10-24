"""Runtime expansion uses the same invocation and value ownership on every worker."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import dinkster_api.v1 as api
import pytest
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, ExecutionError
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_memory import (
    GovernorReservationService,
    MemoryGovernor,
    ReservationRequest,
)
from dinkster_protocol import (
    BlockedOutput,
    CachedResult,
    CurrentNodeRef,
    CurrentOutputRef,
    ExpandedReturn,
    ExternalNodeRef,
    ExternalOutputRef,
    Invocation,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    PresentOutput,
    Worker,
)
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
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    TypeRegistry,
    Value,
    ValueMeta,
    list_children,
    register_core_types,
)
from dinkster_workers import (
    GovernedWorker,
    InProcessWorker,
    IsolatedWorker,
    RemoteWorker,
    RoutingWorker,
)


class WireExpansionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "expansion.test",
            outputs=(
                OutputSpec("result", TypeExpr.concrete("core.int")),
                OutputSpec("seed", TypeExpr.concrete("core.int")),
            ),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls) -> Any:
        values = TypeRegistry()
        register_core_types(values)
        return ExpandedReturn(
            LocalExpansion(
                (LocalNode("child", "expansion.child", (("value", CurrentOutputRef("seed")),)),)
            ),
            (
                ("result", LocalOutputRef("child", "value")),
                ("seed", PresentOutput(values.wrap("core.int", 17))),
            ),
        )


class ExpansionChild(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "expansion.child",
            inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
            outputs=(OutputSpec("value", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, value: int) -> Any:
        return cls.outputs(value=value * 3 + 2)


class ExpansionOutput(ExpansionChild):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="expansion.output",
            output_node=True,
            idempotent=False,
        )


class UnusedExpansionChild(ExpansionChild):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(super().define_schema(), node_type="expansion.unused")

    @classmethod
    def execute(cls, value: int) -> Any:
        raise AssertionError("an undemanded expansion child must not execute")


class GraphExpansionNode(WireExpansionNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(super().define_schema(), node_type="expansion.graph")

    @classmethod
    def execute(cls) -> Any:
        graph = api.ExpansionBuilder()
        child = graph.node("expansion.child", "child", value=graph.current("seed"))
        graph.node("expansion.output", "output", value=child.out("value"))
        graph.node("expansion.unused", "unused", value=graph.current("seed"))
        return graph.finalize(result=child.out(0), seed=17)


class NestedExpansionNode(GraphExpansionNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(super().define_schema(), node_type="expansion.nested")

    @classmethod
    def execute(cls) -> Any:
        return ExpandedReturn(
            LocalExpansion((LocalNode("nested", "expansion.graph"),)),
            (
                ("result", LocalOutputRef("nested", "result")),
                ("seed", LocalOutputRef("nested", "seed")),
            ),
        )


class NonIdempotentExpansionNode(GraphExpansionNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(super().define_schema(), node_type="expansion.repeat", idempotent=False)


class TwoOutputChild(ExpansionChild):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="expansion.pair",
            outputs=(
                OutputSpec("decoy", TypeExpr.concrete("core.int")),
                OutputSpec("value", TypeExpr.concrete("core.int")),
            ),
        )

    @classmethod
    def execute(cls, value: int) -> Any:
        return cls.outputs(decoy=-101, value=value * 3 + 2)


class IndexedExpansionNode(GraphExpansionNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="expansion.indexed",
            inputs=(InputSpec("index", TypeExpr.concrete("core.int"), required=False, default=1),),
        )

    @classmethod
    def execute(cls, index: int = 1) -> api.Expansion:
        graph = api.ExpansionBuilder()
        child = graph.node("expansion.pair", "pair", value=17)
        output = graph.node("expansion.output", "output", value=child.out(1))
        output.set_override_display_id(child.id)
        child.set_input("unused", 9)
        child.set_input("unused", None)
        result = graph.finalize(result=child.out(index), seed=23)
        child.set_input("value", 999)
        return result


class InvalidExpansionNode(WireExpansionNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="expansion.invalid",
            inputs=(InputSpec("problem", TypeExpr.concrete("core.string")),),
        )

    @classmethod
    def execute(cls, problem: str = "missing") -> Any:
        original = super().execute()
        if problem == "cycle":
            nodes = (
                LocalNode(
                    "child", "expansion.child", (("value", LocalOutputRef("child", "value")),)
                ),
            )
        elif problem == "type":
            nodes = (
                LocalNode("child", "expansion.string", (("value", CurrentOutputRef("seed")),)),
            )
        else:
            nodes = (LocalNode("child", "expansion.missing"),)
        return ExpandedReturn(LocalExpansion(nodes), original.bindings)


class StringExpansionChild(ExpansionChild):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="expansion.string",
            inputs=(InputSpec("value", TypeExpr.concrete("core.string")),),
        )


class ExternalExpansionNode(GraphExpansionNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="expansion.external",
            inputs=(
                InputSpec("source", TypeExpr.concrete("core.string"), default="source"),
                InputSpec("output", TypeExpr.concrete("core.int"), default=1),
                InputSpec("display", TypeExpr.concrete("core.string"), default="source"),
            ),
        )

    @classmethod
    def execute(cls, source: str = "source", output: int = 1, display: str = "source") -> Any:
        graph = api.ExpansionBuilder()
        child = graph.node("expansion.child", "child", value=graph.external(source, output))
        generated = graph.node("expansion.output", "output", value=child.out(0))
        generated.set_override_display_id(display, external=True)
        return graph.finalize(result=child.out(0), seed=graph.external(source, output))


PACK_NODES = [
    WireExpansionNode,
    ExpansionChild,
    ExpansionOutput,
    UnusedExpansionChild,
    GraphExpansionNode,
    NestedExpansionNode,
    NonIdempotentExpansionNode,
    InvalidExpansionNode,
    StringExpansionChild,
    TwoOutputChild,
    IndexedExpansionNode,
    ExternalExpansionNode,
]


@pytest.mark.parametrize(
    ("worker_kind", "use_shm"),
    [
        ("in-process", False),
        ("governed", False),
        ("isolated", False),
        ("isolated", True),
        ("remote", False),
    ],
)
def test_expansion_result_survives_worker_round_trip(
    tmp_path: Path, worker_kind: str, use_shm: bool
) -> None:
    from test_remote import TOKEN, start_service, stop_service

    async def scenario() -> None:
        values = TypeRegistry()
        register_core_types(values)
        boundary_worker: IsolatedWorker | RemoteWorker | None = None
        service: asyncio.subprocess.Process | None = None
        worker: Worker
        if worker_kind in ("isolated", "remote"):
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "expansion"\nnamespaces = ["expansion"]\n'
                '[pack.entry]\nnodes = "tests.test_runtime_expansion:PACK_NODES"\n',
                encoding="ascii",
            )
            root = Path(__file__).resolve().parent.parent
            if worker_kind == "remote":
                service, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
                boundary_worker = RemoteWorker(host, port, TOKEN, values, name="expansion")
            else:
                boundary_worker = IsolatedWorker(
                    manifest,
                    values,
                    use_shm=use_shm,
                    shm_threshold=1,
                    extra_env={"PYTHONPATH": str(root)},
                )
            worker = boundary_worker
        else:
            worker = InProcessWorker(build_node_types(PACK_NODES), values)
            if worker_kind == "governed":
                worker = GovernedWorker(
                    worker,
                    planner=lambda _: (ReservationRequest("ram", 1),),
                    reservations=GovernorReservationService(MemoryGovernor({"ram": 16})),
                )
        try:
            if boundary_worker is not None:
                await boundary_worker.start()
            for index in range(2):
                result = await worker.invoke(
                    Invocation(
                        f"invocation-{index}",
                        "parent",
                        "expansion.test",
                        {},
                        WireExpansionNode.schema(),
                        job_ref="run",
                    )
                )
                assert result.error is None
                assert result.outputs is not None
                assert result.outputs["seed"].resolve() == 17
                assert result.expansion is not None
                bindings = dict(result.expansion.bindings)
                assert bindings["result"] == LocalOutputRef("child", "value")
                assert isinstance(bindings["seed"], PresentOutput)
                assert bindings["seed"].value is result.outputs["seed"]
                assert result.expansion.expansion.nodes == (
                    LocalNode("child", "expansion.child", (("value", CurrentOutputRef("seed")),)),
                )
        finally:
            try:
                if boundary_worker is not None:
                    await boundary_worker.close()
            finally:
                if service is not None:
                    await stop_service(service)

    asyncio.run(scenario())


def test_v1_client_cannot_execute_expanding_nodes_on_current_host(tmp_path: Path) -> None:
    from dinkster_workers import PROTOCOL_VERSION
    from dinkster_workers.boundary import (
        ValueCodec,
        decode_result,
        encode_invocation,
        read_frame,
        write_frame,
    )
    from test_remote import TOKEN, start_service, stop_service

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "expansion"\nnamespaces = ["expansion"]\n'
            '[pack.entry]\nnodes = "tests.test_runtime_expansion:PACK_NODES"\n',
            encoding="ascii",
        )
        root = Path(__file__).resolve().parent.parent
        service, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
        try:
            reader, writer = await asyncio.open_connection(host, port)
            try:
                writer.write(TOKEN.encode("ascii"))
                await writer.drain()
                await write_frame(
                    writer,
                    {
                        "type": "clientHello",
                        "protocol": PROTOCOL_VERSION,
                        "payloadTransports": ["inline"],
                        "engine": {"instanceId": "version-test", "label": "version-test"},
                    },
                    [],
                )
                frame = await asyncio.wait_for(read_frame(reader), 10)
                assert frame is not None and frame[0]["type"] == "resumeAccepted"
                frame = await asyncio.wait_for(read_frame(reader), 10)
                assert frame is not None and frame[0]["type"] == "hello"
                codec = ValueCodec(registry, use_shm=False)
                for node_type, schema, inputs in (
                    (
                        "expansion.external",
                        replace(ExternalExpansionNode.schema(), may_expand_graph=False),
                        {},
                    ),
                    (
                        "expansion.child",
                        ExpansionChild.schema(),
                        {"value": registry.wrap("core.int", 5)},
                    ),
                ):
                    invocation = Invocation(node_type, "node", node_type, inputs, schema)
                    header, blobs, segments, _ = encode_invocation(codec, invocation)
                    assert not segments
                    header["resultCapabilities"] = ["dinkster.invocation-result-algebra.v1"]
                    await write_frame(writer, header, blobs)
                    while True:
                        frame = await asyncio.wait_for(read_frame(reader), 10)
                        assert frame is not None
                        if frame[0]["type"] == "result":
                            break
                    assert "resultAlgebra" not in frame[0]
                    if node_type == "expansion.external":
                        error = frame[0]["error"]
                        assert (
                            "runtime expansion requires negotiated capability" in error["message"]
                        )
                        assert "v2" in error["message"]
                    else:
                        result, _ = decode_result(codec, frame[0], frame[1], [])
                        assert result.error is None and result.outputs is not None
                        assert result.outputs["value"].resolve() == 17
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            await stop_service(service)

    asyncio.run(scenario())


def cached_expansion(seed: Value) -> CachedResult:
    registry = TypeRegistry()
    register_core_types(registry)
    other = registry.wrap("core.int", -4)
    return CachedResult(
        {"seed": seed, "other": other},
        ExpandedReturn(
            LocalExpansion(
                (
                    LocalNode(
                        "child",
                        "expansion.child",
                        (
                            ("external", ExternalOutputRef("existing", 1)),
                            ("value", CurrentOutputRef("seed")),
                        ),
                        parent=CurrentNodeRef(),
                        display=LocalNodeRef("display"),
                    ),
                    LocalNode("display", "expansion.display", parent=ExternalNodeRef("existing")),
                )
            ),
            (
                ("blocked", BlockedOutput("stop here")),
                ("other", PresentOutput(other)),
                ("result", LocalOutputRef("child", "value")),
                ("seed", PresentOutput(seed)),
            ),
        ),
    )


@pytest.mark.parametrize("cache_kind", ["memory", "disk", "layered", "spill", "peer"])
def test_cached_expansion_survives_persistence_and_promotion(
    tmp_path: Path, cache_kind: str
) -> None:
    from dinkster_server import PeerCacheStore
    from test_cache_share import serve_cache

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        original = cached_expansion(registry.wrap("core.int", 17))
        memory = MemoryLRUCache(max_bytes=1 if cache_kind == "spill" else 1024)
        disk = DiskCacheStore(tmp_path / "disk", registry)
        server = None
        peer = None
        try:
            if cache_kind == "memory":
                await memory.put("parent", original)
                cache = memory
            elif cache_kind == "disk":
                await disk.put("parent", original)
                cache = DiskCacheStore(tmp_path / "disk", registry)
            elif cache_kind == "peer":
                await disk.put("parent", original)
                server = await serve_cache(disk)
                peer = PeerCacheStore(str(server.make_url("")), registry)
                cache = LayeredCache(memory, DiskCacheStore(tmp_path / "local", registry), peer)
            elif cache_kind == "spill":
                costed = replace(original["seed"], meta=ValueMeta({COST_META_KEY: {"ram": 50}}))
                original = CachedResult({**original, "seed": costed}, original.expansion)
                cache = LayeredCache(memory, disk, disk_spill_min_bytes=1)
                await cache.put("parent", original)
                assert len(memory) == 0
            else:
                await disk.put("parent", original)
                cache = LayeredCache(memory, DiskCacheStore(tmp_path / "disk", registry))

            for _ in range(2):
                hit = await cache.get("parent")
                assert hit is not None and hit.expansion is not None
                assert set(hit) == {"seed", "other"}
                assert hit["seed"].resolve() == 17
                assert hit["other"].resolve() == -4
                bindings = dict(hit.expansion.bindings)
                assert bindings["blocked"] == BlockedOutput("stop here")
                assert bindings["result"] == LocalOutputRef("child", "value")
                assert isinstance(bindings["seed"], PresentOutput)
                assert bindings["seed"].value is hit["seed"]
                assert original.expansion is not None
                assert hit.expansion.expansion == original.expansion.expansion
            if peer is not None:
                assert peer.hits == 1
                restarted = await DiskCacheStore(tmp_path / "local", registry).get("parent")
                assert restarted is not None and restarted.expansion is not None
                assert restarted["seed"].resolve() == 17
        finally:
            if peer is not None:
                await peer.close()
            if server is not None:
                await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "damage", ["document", "missing-output", "extra-output", "missing-field", "index"]
)
def test_incomplete_cached_expansion_is_a_whole_entry_miss(tmp_path: Path, damage: str) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        disk = DiskCacheStore(tmp_path, registry)
        await disk.put("parent", cached_expansion(registry.wrap("core.int", 17)))
        path = next((tmp_path / "entries").glob("*.json"))
        manifest = json.loads(path.read_text())
        if damage == "document":
            manifest["expansion"]["expansion"]["nodes"][0]["nodeType"] = "bad type"
        elif damage == "missing-output":
            del manifest["outputs"]["seed"]
        elif damage == "extra-output":
            manifest["outputs"]["extra"] = manifest["outputs"]["seed"]
        elif damage == "index":
            manifest["expansion"]["bindings"][3]["value"]["valueIndex"] = True
        else:
            del manifest["expansion"]
        path.write_text(json.dumps(manifest))
        cache = LayeredCache(MemoryLRUCache(), DiskCacheStore(tmp_path, registry))
        assert await cache.get("parent") is None
        assert not path.exists()

    asyncio.run(scenario())


def test_cached_expansion_accounts_and_invalidates_present_values_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        seed = registry.wrap("core.int", 17)
        seed = replace(seed, meta=ValueMeta({COST_META_KEY: {"ram": 50}}))
        memory = MemoryLRUCache(max_bytes=100)
        disk = DiskCacheStore(tmp_path, registry)
        cache = LayeredCache(memory, disk)
        await cache.put("parent", cached_expansion(seed))
        assert memory.footprint("ram") == 50
        resident = replace(seed, meta=ValueMeta({RESOURCE_ID_META_KEY: "resident"}))
        await cache.put("parent", cached_expansion(resident))
        assert memory.footprint("ram") == 0
        assert await disk.get("parent") is None
        assert cache.drop_referencing("resident") == 1
        assert await cache.get("parent") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("worker_kind", ["in-process", "governed", "isolated", "remote"])
@pytest.mark.parametrize("cache_enabled", [True, False])
@pytest.mark.parametrize(
    "node_type", ["expansion.graph", "expansion.indexed", "expansion.external"]
)
def test_expansion_executes_children_through_each_worker(
    tmp_path: Path, worker_kind: str, cache_enabled: bool, node_type: str
) -> None:
    from test_remote import TOKEN, start_service, stop_service

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        boundary: IsolatedWorker | RemoteWorker | None = None
        service: asyncio.subprocess.Process | None = None
        worker: Worker = InProcessWorker(build_node_types(PACK_NODES), registry)
        if worker_kind == "governed":
            worker = GovernedWorker(
                worker,
                planner=lambda _: (ReservationRequest("ram", 1),),
                reservations=GovernorReservationService(MemoryGovernor({"ram": 16})),
            )
        elif worker_kind in ("isolated", "remote"):
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "expansion"\nnamespaces = ["expansion"]\n'
                '[pack.entry]\nnodes = "tests.test_runtime_expansion:PACK_NODES"\n',
                encoding="ascii",
            )
            root = Path(__file__).resolve().parent.parent
            if worker_kind == "remote":
                service, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
                boundary = RemoteWorker(host, port, TOKEN, registry, name="expansion")
            else:
                boundary = IsolatedWorker(
                    manifest,
                    registry,
                    extra_env={"PYTHONPATH": str(root)},
                )
            await boundary.start()
            worker = boundary
        events: list[EngineEvent] = []
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph(
            {
                "parent": GraphNode(node_type),
                "consumer": GraphNode("expansion.child", {"value": Link("parent", "result")}),
                "source": GraphNode("expansion.pair", {"value": 5}),
            }
        )
        try:
            for index in range(2):
                events.clear()
                result = await asyncio.wait_for(
                    engine.run(graph, ["consumer"], cache_enabled=cache_enabled),
                    10,
                )
                assert result.outputs["consumer"]["value"].resolve() == 161
                assert "parent.0.0.output" in result.executed
                assert not any("unused" in node_id for node_id in result.executed)
                if index and cache_enabled:
                    assert "parent" in result.cached and "parent" not in result.executed
                else:
                    assert "parent" in result.executed
                expanded = [event for event in events if event.kind == "node_expanded"]
                count = 3 if node_type == "expansion.graph" else 2
                assert [(e.node_id, e.detail["childCount"]) for e in expanded] == [
                    ("parent", count)
                ]
                child = next(
                    e
                    for e in events
                    if e.kind == "node_started" and e.node_id == "parent.0.0.output"
                )
                assert child.detail["displayNodeId"] == (
                    "source" if node_type == "expansion.external" else "parent"
                )
                assert child.detail["parentId"] == "parent"
            assert set(graph.nodes) == {"parent", "consumer", "source"}
        finally:
            try:
                if boundary is not None:
                    await boundary.close()
            finally:
                if service is not None:
                    await stop_service(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_kind", ["memory", "disk", "layered"])
@pytest.mark.parametrize(
    "node_type",
    [
        "expansion.graph",
        "expansion.nested",
        "expansion.repeat",
        "expansion.indexed",
        "expansion.external",
    ],
)
def test_engine_replays_cached_expansion_declarations(
    tmp_path: Path, cache_kind: str, node_type: str
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        memory = MemoryLRUCache()
        cache = memory if cache_kind == "memory" else DiskCacheStore(tmp_path, registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=cache,
        )
        graph = Graph(
            {
                "parent": GraphNode(node_type),
                "source": GraphNode("expansion.pair", {"value": 5}),
            }
        )
        first = await engine.run(graph, ["parent"])
        assert first.outputs["parent"]["result"].resolve() == 53
        if cache_kind != "memory":
            disk = DiskCacheStore(tmp_path, registry)
            engine = Engine(
                schemas=build_schemas(PACK_NODES),
                registry=registry,
                worker=InProcessWorker(build_node_types(PACK_NODES), registry),
                cache=LayeredCache(MemoryLRUCache(), disk) if cache_kind == "layered" else disk,
            )
        second = await engine.run(graph, ["parent"])
        assert second.outputs["parent"]["seed"].resolve() == (
            23 if node_type == "expansion.indexed" else 17
        )
        assert second.outputs["parent"]["result"].resolve() == 53
        assert any(node_id.endswith(".output") for node_id in second.executed)
        if node_type == "expansion.repeat":
            assert "parent" in second.executed
        else:
            assert "parent" in second.cached and "parent" not in second.executed
        if node_type == "expansion.nested":
            assert "parent.0.0.nested" in second.cached

    asyncio.run(scenario())


@pytest.mark.parametrize("index", [-1, 2])
def test_builder_output_index_is_validated_at_parent(index: int) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        with pytest.raises(ExecutionError) as error:
            await engine.run(
                Graph({"parent": GraphNode("expansion.indexed", {"index": index})}), ["parent"]
            )
        assert error.value.error.node_id == "parent"
        assert "index" in error.value.error.message

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "inputs",
    [
        {"source": "missing"},
        {"output": 2},
        {"source": "text"},
        {"display": "missing"},
        {"source": "consumer", "output": 0},
    ],
)
def test_external_reference_failures_name_the_parent(inputs: dict[str, object]) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        events: list[EngineEvent] = []
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph(
            {
                "parent": GraphNode("expansion.external", inputs),
                "source": GraphNode("expansion.pair", {"value": 5}),
                "text": GraphNode("expansion.string", {"value": "not an integer"}),
                "consumer": GraphNode("expansion.child", {"value": Link("parent", "result")}),
            }
        )
        with pytest.raises(ExecutionError, match="invalid expansion") as error:
            await asyncio.wait_for(engine.run(graph, ["parent"]), 5)
        assert error.value.error.node_id == "parent"
        if inputs.get("source") == "consumer":
            assert "dynamic cycle" in str(error.value)
        assert not any(event.kind == "node_expanded" for event in events)

    asyncio.run(scenario())


def test_external_reference_reuses_a_completed_non_idempotent_source() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        calls: list[str] = []

        class CountingWorker(InProcessWorker):
            async def invoke(self, invocation: Invocation, on_event: Any = None) -> Any:
                calls.append(invocation.node_id)
                return await super().invoke(invocation, on_event=on_event)

        class OrderedExpansion(ExternalExpansionNode):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return replace(
                    super().define_schema(),
                    node_type="expansion.ordered",
                    inputs=super().define_schema().inputs
                    + (InputSpec("dependency", TypeExpr.concrete("core.int")),),
                )

            @classmethod
            def execute(
                cls, source: str = "source", output: int = 1, display: str = "source", **_: Any
            ) -> Any:
                return super().execute(source, output, display)

        class NonIdempotentSource(TwoOutputChild):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return replace(super().define_schema(), idempotent=False)

        nodes = [node for node in PACK_NODES if node is not TwoOutputChild]
        nodes += [OrderedExpansion, NonIdempotentSource]
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=CountingWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(
                {
                    "parent": GraphNode(
                        "expansion.ordered", {"dependency": Link("bridge", "value")}
                    ),
                    "bridge": GraphNode("expansion.child", {"value": Link("source", "value")}),
                    "source": GraphNode("expansion.pair", {"value": 5}),
                }
            ),
            ["parent"],
        )
        assert result.outputs["parent"]["result"].resolve() == 53
        assert result.outputs["parent"]["seed"].resolve() == 17
        assert calls.count("source") == 1

    asyncio.run(scenario())


def test_external_references_detect_a_cycle_across_two_expansions() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            {
                "left": GraphNode("expansion.external", {"source": "right", "output": 0}),
                "right": GraphNode("expansion.external", {"source": "left", "output": 0}),
                "source": GraphNode("expansion.pair", {"value": 5}),
            }
        )
        with pytest.raises(ExecutionError, match="dynamic cycle"):
            await asyncio.wait_for(engine.run(graph, ["left", "right"]), 5)

    asyncio.run(scenario())


def test_cached_expansion_reads_current_external_values() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        for source, expected, seed in ((5, 53, 17), (11, 107, 35)):
            graph = Graph(
                {
                    "parent": GraphNode("expansion.external"),
                    "source": GraphNode("expansion.pair", {"value": source}),
                }
            )
            result = await engine.run(graph, ["parent"])
            assert result.outputs["parent"]["result"].resolve() == expected
            assert result.outputs["parent"]["seed"].resolve() == seed
            if source == 11:
                assert "parent" in result.cached
                assert "parent.0.0.child" in result.executed

    asyncio.run(scenario())


def test_expansion_can_reference_an_existing_region_output() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        region = RegionNode(
            kind="map",
            body=Graph({"body": GraphNode("expansion.child", {"value": Link("$region", "value")})}),
            ports={"value": TypeExpr.concrete("core.int")},
            inputs={"value": [5, 11]},
            element_ports=("value",),
            outputs={"last": RegionOutput(Link("body", "value"), "last")},
        )
        graph = Graph(
            {
                "parent": GraphNode("expansion.external", {"source": "source", "output": 0}),
                "source": region,
            }
        )
        result = await asyncio.wait_for(engine.run(graph, ["parent"]), 5)
        assert result.outputs["parent"]["seed"].resolve() == 35
        assert result.outputs["parent"]["result"].resolve() == 107

    asyncio.run(scenario())


def test_builder_can_reference_a_node_outside_its_worker() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        producer = InProcessWorker(build_node_types([IndexedExpansionNode]), registry)
        children = InProcessWorker(build_node_types([TwoOutputChild, ExpansionOutput]), registry)
        engine = Engine(
            schemas=build_schemas([IndexedExpansionNode, TwoOutputChild, ExpansionOutput]),
            registry=registry,
            worker=RoutingWorker({"expansion.indexed": producer}, default=children),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(Graph({"parent": GraphNode("expansion.indexed")}), ["parent"])
        assert result.outputs["parent"]["result"].resolve() == 53
        assert "parent.0.0.output" in result.executed

    asyncio.run(scenario())


@pytest.mark.parametrize("problem", ["cycle", "type", "missing"])
def test_invalid_expansion_names_its_parent(problem: str) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph({"parent": GraphNode("expansion.invalid", {"problem": problem})})
        with pytest.raises(ExecutionError) as error:
            await engine.run(graph, ["parent"])
        assert error.value.error.node_id == "parent"
        expected = "cycle" if problem == "cycle" else "invalid expansion"
        assert expected in error.value.error.message

    asyncio.run(scenario())


def test_coalesced_expansion_parents_have_independent_children() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        calls: list[str] = []

        class CountingWorker(InProcessWorker):
            async def invoke(self, invocation: Invocation, on_event: Any = None) -> Any:
                calls.append(invocation.node_type)
                await asyncio.sleep(0)
                return await super().invoke(invocation, on_event=on_event)

        events: list[EngineEvent] = []
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=CountingWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph({name: GraphNode("expansion.graph") for name in ("left", "right")})
        result = await asyncio.wait_for(engine.run(graph, ["left", "right"]), 5)
        assert calls.count("expansion.graph") == 1
        assert calls.count("expansion.output") == 2
        assert calls.count("expansion.child") == 2
        assert result.outputs["left"]["result"].resolve() == 53
        assert result.outputs["right"]["result"].resolve() == 53
        assert {event.node_id for event in events if event.kind == "node_expanded"} == {
            "left",
            "right",
        }
        for parent in ("left", "right"):
            child = next(
                event
                for event in events
                if event.kind == "node_started" and event.node_id == f"{parent}.0.0.output"
            )
            assert child.detail["parentId"] == parent
            assert child.detail["displayNodeId"] == parent

    asyncio.run(scenario())


def test_cancelled_expansion_releases_children_and_can_run_again() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        started = asyncio.Event()
        stopped = asyncio.Event()
        pause = True

        class PausingWorker(InProcessWorker):
            async def invoke(self, invocation: Invocation, on_event: Any = None) -> Any:
                if invocation.node_type == "expansion.output" and pause:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()
                return await super().invoke(invocation, on_event=on_event)

        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=PausingWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph({"parent": GraphNode("expansion.graph")})
        pending = asyncio.create_task(engine.run(graph, ["parent"]))
        await asyncio.wait_for(started.wait(), 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 5)
        assert stopped.is_set()
        pause = False
        result = await asyncio.wait_for(engine.run(graph, ["parent"]), 5)
        assert result.outputs["parent"]["result"].resolve() == 53
        assert "parent.0.0.output" in result.executed

    asyncio.run(scenario())


def test_expansion_preserves_enclosing_region_ports() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), registry),
            cache=MemoryLRUCache(),
        )
        body = Graph(
            {
                "parent": GraphNode("expansion.graph"),
                "port": GraphNode("expansion.child", {"value": Link("$region", "value")}),
            }
        )
        graph = Graph(
            {
                "region": RegionNode(
                    kind="map",
                    body=body,
                    ports={"value": TypeExpr.concrete("core.int")},
                    inputs={"value": [4, 9]},
                    element_ports=("value",),
                    outputs={
                        "expanded": RegionOutput(Link("parent", "result"), "gather"),
                        "port": RegionOutput(Link("port", "value"), "gather"),
                    },
                )
            }
        )
        for _ in range(2):
            result = await asyncio.wait_for(engine.run(graph, ["region"]), 5)
            expanded = list_children(result.outputs["region"]["expanded"])
            ports = list_children(result.outputs["region"]["port"])
            assert expanded is not None and ports is not None
            assert [v.resolve() for v in expanded] == [53, 53]
            assert [v.resolve() for v in ports] == [14, 29]

    asyncio.run(scenario())
