"""list<T> values, schema, engine, boundary, and cache behavior (DESIGN 3.13).

Slice 1-2 of collections: lists as ordinary typed data. Envelope children
stay full Values (resource visibility through trees), cardinality is
structural at validation time, and node authors see plain Python lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import pytest
from dinkster_caches import MemoryLRUCache, encode_entry, entry_from_wire, entry_to_wire
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import (
    CORE_COMBO,
    CORE_INT,
    RESOURCE_HANDLE_TYPE,
    ResourceHandle,
    TypeRegistry,
    ValueMeta,
    iter_value_tree,
    list_children,
    list_type_id,
    make_list_value,
    parse_list_type_id,
    register_core_types,
    register_resource_handle_type,
    value_resource_ids,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import ValueCodec
from dinkster_workers.devices import DeviceMap

# -- grammar and TypeExpr ---------------------------------------------------


def test_list_type_id_grammar() -> None:
    assert list_type_id("core.int") == "list<core.int>"
    assert parse_list_type_id("list<core.int>") == "core.int"
    assert parse_list_type_id("list<list<core.int>>") == "list<core.int>"
    assert parse_list_type_id("core.int") is None
    assert parse_list_type_id("list<>") is None


def test_list_types_have_one_schema_spelling() -> None:
    with pytest.raises(ValueError):
        TypeExpr.concrete("list<core.int>")


def test_type_expr_list_kind() -> None:
    expr = TypeExpr.list_of(TypeExpr.concrete("core.int"))
    assert expr.runtime_type_id() == "list<core.int>"
    nested = TypeExpr.list_of(expr)
    assert nested.runtime_type_id() == "list<list<core.int>>"
    assert TypeExpr.list_of(TypeExpr.wildcard()).runtime_type_id() is None

    assert expr.cardinality() == "list"
    assert TypeExpr.concrete("core.int").cardinality() == "scalar"
    assert TypeExpr.wildcard().cardinality() == "unknown"

    assert expr.accepts_concrete("list<core.int>")
    assert not expr.accepts_concrete("core.int")
    assert not expr.accepts_concrete("list<core.float>")
    assert TypeExpr.list_of(TypeExpr.wildcard()).accepts_concrete("list<anything>")

    with pytest.raises(ValueError):
        TypeExpr(kind="list")  # no element
    with pytest.raises(ValueError):
        TypeExpr(kind="concrete", types=("core.int",), element=TypeExpr.wildcard())


def test_schema_wire_roundtrip_with_lists() -> None:
    schema = NodeSchema(
        node_type="test.lists",
        inputs=(
            InputSpec("values", TypeExpr.list_of(TypeExpr.concrete("core.int"))),
            InputSpec(
                "matrix",
                TypeExpr.list_of(TypeExpr.list_of(TypeExpr.concrete("core.float"))),
                required=False,
            ),
        ),
        outputs=(OutputSpec("out", TypeExpr.list_of(TypeExpr.concrete("core.string"))),),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema


# -- value envelopes --------------------------------------------------------


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_wrap_list_builds_child_envelopes() -> None:
    registry = make_registry()
    value = registry.wrap("list<core.int>", [1, 2, 3])
    assert value.type_id == "list<core.int>"
    assert value.meta.get("length") == 3
    assert value.resolve() == [1, 2, 3]
    children = list_children(value)
    assert children is not None
    assert [c.type_id for c in children] == [CORE_INT] * 3
    assert children[0].fingerprint == registry.wrap(CORE_INT, 1).fingerprint
    assert len(list(iter_value_tree(value))) == 4  # the list + 3 children


def test_list_fingerprint_is_order_sensitive_and_stable() -> None:
    registry = make_registry()
    a = registry.wrap("list<core.int>", [1, 2, 3])
    b = registry.wrap("list<core.int>", [1, 2, 3])
    c = registry.wrap("list<core.int>", [3, 2, 1])
    d = registry.wrap("list<core.int>", [1, 2])
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.fingerprint != d.fingerprint


def test_wrap_nested_lists() -> None:
    registry = make_registry()
    value = registry.wrap("list<list<core.int>>", [[1, 2], [3]])
    assert value.resolve() == [[1, 2], [3]]
    children = list_children(value)
    assert children is not None
    assert [c.type_id for c in children] == ["list<core.int>"] * 2


def test_wrap_combo_lists_preserves_empty_order_and_duplicates() -> None:
    registry = make_registry()
    empty = registry.wrap("list<core.combo>", [])
    ordered = registry.wrap("list<core.combo>", ["b", "a", "b"])

    assert empty.resolve() == []
    assert ordered.resolve() == ["b", "a", "b"]
    children = list_children(ordered)
    assert children is not None
    assert [child.type_id for child in children] == [CORE_COMBO] * 3
    assert children[0].fingerprint == children[2].fingerprint
    assert children[0].fingerprint != children[1].fingerprint
    assert ordered.fingerprint != registry.wrap("list<core.combo>", ["a", "b", "b"]).fingerprint


@pytest.mark.parametrize("invalid", [["ok", 1], [None], [["nested"]]])
def test_wrap_combo_lists_recursively_rejects_non_strings(invalid: object) -> None:
    registry = make_registry()
    with pytest.raises(TypeError, match="core.combo expects a string"):
        registry.wrap("list<core.combo>", invalid)


def test_wrap_list_rejects_non_sequences() -> None:
    registry = make_registry()
    with pytest.raises(TypeError):
        registry.wrap("list<core.int>", 5)
    with pytest.raises(TypeError):
        registry.wrap("list<core.string>", "not-a-list-of-strings")


def test_make_list_value_rejects_mismatched_children() -> None:
    registry = make_registry()
    child = registry.wrap(CORE_INT, 1)
    with pytest.raises(ValueError):
        make_list_value("core.float", [child])


def _handle_value(registry: TypeRegistry, rid: str, device: str = "cuda:0"):
    handle = ResourceHandle(
        resource_id=rid,
        kind="gpu",
        residency={"gpu": device},
        cost={f"vram:{device}": 100},
        obj=object(),
    )
    return registry.wrap(RESOURCE_HANDLE_TYPE, handle)


def test_resource_ids_visible_through_lists() -> None:
    registry = make_registry()
    register_resource_handle_type(registry)
    v1 = _handle_value(registry, "model-a")
    v2 = _handle_value(registry, "model-b")
    lst = make_list_value(RESOURCE_HANDLE_TYPE, [v1, v2, v1])
    assert value_resource_ids(lst) == ("model-a", "model-b")  # deduped, ordered
    nested = make_list_value(list_type_id(RESOURCE_HANDLE_TYPE), [lst])
    assert value_resource_ids(nested) == ("model-a", "model-b")


# -- engine end to end ------------------------------------------------------


class MakeRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.make_range",
            inputs=(InputSpec("count", TypeExpr.concrete("core.int")),),
            outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.concrete("core.int"))),),
        )

    @classmethod
    def execute(cls, count: int) -> Mapping[str, object]:
        return cls.outputs(values=list(range(count)))


class SumList(Node):
    saw: list[object] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.sum_list",
            inputs=(InputSpec("values", TypeExpr.list_of(TypeExpr.concrete("core.int"))),),
            outputs=(OutputSpec("total", TypeExpr.concrete("core.int")),),
        )

    @classmethod
    def execute(cls, values: list[int]) -> Mapping[str, object]:
        cls.saw.append(values)
        return cls.outputs(total=sum(values))


class BadListOutput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.bad_list_output",
            inputs=(),
            outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.concrete("core.int"))),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(values=7)  # not a sequence: contract error


LIST_NODES: tuple[type[Node], ...] = (MakeRange, SumList, BadListOutput)


def make_engine() -> Engine:
    registry = make_registry()
    return Engine(
        schemas=build_schemas(LIST_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(LIST_NODES), registry),
        cache=MemoryLRUCache(),
    )


def test_engine_runs_list_edges_and_caches() -> None:
    async def scenario() -> None:
        SumList.saw.clear()
        engine = make_engine()
        graph = Graph(
            nodes={
                "r": GraphNode("test.make_range", {"count": 4}),
                "s": GraphNode("test.sum_list", {"values": Link("r", "values")}),
            }
        )
        first = await engine.run(graph, ["s"])
        assert first.outputs["s"]["total"].resolve() == 6
        # Node code saw a plain Python list, no envelopes (hazard H9).
        assert SumList.saw == [[0, 1, 2, 3]]

        second = await engine.run(graph, ["s"])
        assert second.executed == ()
        assert set(second.cached) == {"r", "s"}

    asyncio.run(scenario())


def test_engine_wraps_list_literals() -> None:
    async def scenario() -> None:
        SumList.saw.clear()
        engine = make_engine()
        graph = Graph(nodes={"s": GraphNode("test.sum_list", {"values": [5, 6]})})
        result = await engine.run(graph, ["s"])
        assert result.outputs["s"]["total"].resolve() == 11
        assert SumList.saw == [[5, 6]]

    asyncio.run(scenario())


def test_bad_list_output_is_contract_error() -> None:
    async def scenario() -> None:
        registry = make_registry()
        worker = InProcessWorker(build_node_types(LIST_NODES), registry)
        engine = Engine(
            schemas=build_schemas(LIST_NODES),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
        )
        graph = Graph(nodes={"b": GraphNode("test.bad_list_output", {})})
        from dinkster_engine import ExecutionError

        with pytest.raises(ExecutionError, match="values"):
            await engine.run(graph, ["b"])

    asyncio.run(scenario())


# -- validation diagnostics -------------------------------------------------


class TakesScalar(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.takes_scalar",
            inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )


def _diag_codes(graph: Graph, nodes: tuple[type[Node], ...], targets: list[str]):
    return [d.code for d in validate(graph, build_schemas(nodes), targets)]


def test_list_into_scalar_is_an_error_with_guidance() -> None:
    graph = Graph(
        nodes={
            "r": GraphNode("test.make_range", {"count": 3}),
            "t": GraphNode("test.takes_scalar", {"value": Link("r", "values")}),
        }
    )
    nodes = (MakeRange, TakesScalar)
    diags = validate(graph, build_schemas(nodes), ["t"])
    codes = {d.code for d in diags}
    assert "list-into-scalar" in codes
    message = next(d.message for d in diags if d.code == "list-into-scalar")
    assert "Map" in message  # the diagnostic names the explicit fix


def test_scalar_into_list_is_an_error() -> None:
    graph = Graph(
        nodes={
            "t": GraphNode("test.takes_scalar", {"value": 1}),
            "s": GraphNode("test.sum_list", {"values": Link("t", "out")}),
        }
    )
    assert "scalar-into-list" in _diag_codes(graph, (TakesScalar, SumList), ["s"])


def test_list_element_type_mismatch_is_advisory() -> None:
    class MakeFloats(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.make_floats",
                inputs=(),
                outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.concrete("core.float"))),),
            )

    graph = Graph(
        nodes={
            "f": GraphNode("test.make_floats", {}),
            "s": GraphNode("test.sum_list", {"values": Link("f", "values")}),
        }
    )
    diags = validate(graph, build_schemas((MakeFloats, SumList)), ["s"])
    assert [d.code for d in diags] == ["type-mismatch"]
    assert all(d.severity == "warning" for d in diags)


def test_non_list_literal_on_list_input_is_an_error() -> None:
    graph = Graph(nodes={"s": GraphNode("test.sum_list", {"values": 5})})
    assert "literal-shape" in _diag_codes(graph, (SumList,), ["s"])


# -- boundary codec ---------------------------------------------------------


def test_codec_roundtrips_lists_with_child_envelopes() -> None:
    registry = make_registry()
    register_resource_handle_type(registry)
    sender = ValueCodec(registry, use_shm=False, accept_shm=False)
    receiver = ValueCodec(registry, use_shm=False, accept_shm=False)

    stub = _handle_value(registry, "model-x", device="cuda:1")
    inner = registry.wrap("list<core.int>", [1, 2])
    value = make_list_value("list<core.int>", [inner, registry.wrap("list<core.int>", [3])])

    blobs: list[bytes] = []
    wire, stat = sender.encode(value, blobs, [])
    assert stat.transport == "list"
    decoded, _ = receiver.decode(wire, blobs, [])
    assert decoded.type_id == value.type_id
    assert decoded.fingerprint == value.fingerprint
    assert decoded.resolve() == [[1, 2], [3]]

    # A resource stub inside a list keeps its identity across the boundary.
    lst = make_list_value(RESOURCE_HANDLE_TYPE, [stub])
    blobs2: list[bytes] = []
    wire2, _ = sender.encode(lst, blobs2, [])
    decoded2, _ = receiver.decode(wire2, blobs2, [])
    assert value_resource_ids(decoded2) == ("model-x",)
    children = list_children(decoded2)
    assert children is not None
    assert children[0].meta.get("resources") == {"gpu": "cuda:1"}


def test_device_map_rewrites_list_children() -> None:
    registry = make_registry()
    register_resource_handle_type(registry)
    stub = _handle_value(registry, "model-y", device="cuda:0")
    lst = make_list_value(RESOURCE_HANDLE_TYPE, [stub])
    mapped = DeviceMap(mapping={"cuda:0": "cuda:1"}).value(lst)
    assert mapped.fingerprint == lst.fingerprint  # identity is device-free
    children = list_children(mapped)
    assert children is not None
    assert children[0].meta.get("resources") == {"gpu": "cuda:1"}
    # Identity map: the very same object comes back.
    assert DeviceMap(mapping={}).value(lst) is lst


# -- persisted cache manifests ----------------------------------------------


def test_cache_manifest_roundtrips_lists() -> None:
    async def scenario() -> None:
        registry = make_registry()
        outputs = {"values": registry.wrap("list<list<core.int>>", [[1], [2, 3]])}
        encoded = encode_entry(outputs, registry)
        assert encoded is not None

        blob_store: dict[str, bytes] = {}

        def store(data: bytes) -> str:
            digest = f"d{len(blob_store)}"
            blob_store[digest] = data
            return digest

        manifest = entry_to_wire("key1", encoded, store)
        assert manifest["version"] == 2

        async def fetch(digest: str, size: int) -> bytes | None:
            return blob_store.get(digest)

        entry = await entry_from_wire(manifest, fetch, registry)
        assert entry is not None
        rehydrated = entry["values"]
        assert rehydrated.fingerprint == outputs["values"].fingerprint
        assert rehydrated.resolve() == [[1], [2, 3]]

    asyncio.run(scenario())


def test_cache_refuses_resource_stubs_inside_lists() -> None:
    registry = make_registry()
    register_resource_handle_type(registry)
    stub = _handle_value(registry, "model-z")
    outputs = {"models": make_list_value(RESOURCE_HANDLE_TYPE, [stub])}
    assert encode_entry(outputs, registry) is None


def test_memory_cache_sees_costs_and_references_through_lists() -> None:
    async def scenario() -> None:
        registry = make_registry()
        register_resource_handle_type(registry)
        cache = MemoryLRUCache()

        costed = registry.wrap(CORE_INT, 1)
        costed = type(costed)(
            type_id=costed.type_id,
            fingerprint=costed.fingerprint,
            meta=ValueMeta({"cost": {"ram": 64}}),
            payload=costed.payload,
        )
        stub = _handle_value(registry, "model-q")
        entry = {
            "mixed": make_list_value(CORE_INT, [costed, registry.wrap(CORE_INT, 2)]),
            "models": make_list_value(RESOURCE_HANDLE_TYPE, [stub]),
        }
        # make_list_value requires matching child types; adjust: separate keys
        await cache.put("k", cast(dict[str, Any], entry))
        assert cache.footprint("ram") == 64  # child cost counted, stub excluded
        assert cache.drop_referencing("model-q") == 1
        assert await cache.get("k") is None

    asyncio.run(scenario())
