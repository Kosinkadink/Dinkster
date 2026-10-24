"""list<T> values, schema, engine, boundary, and cache behavior (DESIGN 3.13).

Slice 1-2 of collections: lists as ordinary typed data. Envelope children
stay full Values (resource visibility through trees), cardinality is
structural at validation time, and node authors see plain Python lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from dinkster_caches import MemoryLRUCache, encode_entry, entry_from_wire, entry_to_wire
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput, validate
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    combo_type_mismatch_is_error,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import (
    CORE_COMBO,
    CORE_INT,
    RESOURCE_HANDLE_TYPE,
    RESOURCE_OWNER_META_KEY,
    ResourceHandle,
    TypeRegistry,
    ValueMeta,
    iter_value_tree,
    list_children,
    list_type_id,
    make_list_value,
    make_tuple_value,
    parse_list_type_id,
    register_core_types,
    register_resource_handle_type,
    runtime_type_atoms,
    stamp_resource_producer_arm,
    value_children,
    value_resource_ids,
    value_resource_provenance_refs,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import BoundaryError, ValueCodec
from dinkster_workers.devices import DeviceMap

# -- grammar and TypeExpr ---------------------------------------------------


def test_list_type_id_grammar() -> None:
    assert list_type_id("INT") == "list<INT>"
    assert parse_list_type_id("list<INT>") == "INT"
    assert parse_list_type_id("list<list<INT>>") == "list<INT>"
    assert parse_list_type_id("INT") is None
    assert parse_list_type_id("list<>") is None


def test_list_types_have_one_schema_spelling() -> None:
    with pytest.raises(ValueError):
        TypeExpr.concrete("list<INT>")


def test_type_expr_list_kind() -> None:
    expr = TypeExpr.list_of(TypeExpr.concrete("INT"))
    assert expr.runtime_type_id() == "list<INT>"
    nested = TypeExpr.list_of(expr)
    assert nested.runtime_type_id() == "list<list<INT>>"
    assert TypeExpr.list_of(TypeExpr.wildcard()).runtime_type_id() is None

    assert expr.cardinality() == "list"
    assert TypeExpr.concrete("INT").cardinality() == "scalar"
    assert TypeExpr.wildcard().cardinality() == "unknown"

    assert expr.accepts_concrete("list<INT>")
    assert not expr.accepts_concrete("INT")
    assert not expr.accepts_concrete("list<FLOAT>")
    assert TypeExpr.list_of(TypeExpr.wildcard()).accepts_concrete("list<anything>")

    with pytest.raises(ValueError):
        TypeExpr(kind="list")  # no element
    with pytest.raises(ValueError):
        TypeExpr(kind="concrete", types=("INT",), element=TypeExpr.wildcard())


def test_schema_wire_roundtrip_with_lists() -> None:
    schema = NodeSchema(
        node_type="test.lists",
        inputs=(
            InputSpec("values", TypeExpr.list_of(TypeExpr.concrete("INT"))),
            InputSpec(
                "matrix",
                TypeExpr.list_of(TypeExpr.list_of(TypeExpr.concrete("FLOAT"))),
                required=False,
            ),
        ),
        outputs=(OutputSpec("out", TypeExpr.list_of(TypeExpr.concrete("STRING"))),),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema


# -- value envelopes --------------------------------------------------------


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_wrap_list_builds_child_envelopes() -> None:
    registry = make_registry()
    value = registry.wrap("list<INT>", [1, 2, 3])
    assert value.type_id == "list<INT>"
    assert value.meta.get("length") == 3
    assert value.resolve() == [1, 2, 3]
    children = list_children(value)
    assert children is not None
    assert [c.type_id for c in children] == [CORE_INT] * 3
    assert children[0].fingerprint == registry.wrap(CORE_INT, 1).fingerprint
    assert len(list(iter_value_tree(value))) == 4  # the list + 3 children


def test_list_fingerprint_is_order_sensitive_and_stable() -> None:
    registry = make_registry()
    a = registry.wrap("list<INT>", [1, 2, 3])
    b = registry.wrap("list<INT>", [1, 2, 3])
    c = registry.wrap("list<INT>", [3, 2, 1])
    d = registry.wrap("list<INT>", [1, 2])
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.fingerprint != d.fingerprint


def test_wrap_nested_lists() -> None:
    registry = make_registry()
    value = registry.wrap("list<list<INT>>", [[1, 2], [3]])
    assert value.resolve() == [[1, 2], [3]]
    children = list_children(value)
    assert children is not None
    assert [c.type_id for c in children] == ["list<INT>"] * 2


def test_wrap_combo_lists_preserves_empty_order_and_duplicates() -> None:
    registry = make_registry()
    empty = registry.wrap("list<COMBO>", [])
    ordered = registry.wrap("list<COMBO>", ["b", "a", "b"])

    assert empty.resolve() == []
    assert ordered.resolve() == ["b", "a", "b"]
    children = list_children(ordered)
    assert children is not None
    assert [child.type_id for child in children] == [CORE_COMBO] * 3
    assert children[0].fingerprint == children[2].fingerprint
    assert children[0].fingerprint != children[1].fingerprint
    assert ordered.fingerprint != registry.wrap("list<COMBO>", ["a", "b", "b"]).fingerprint


@pytest.mark.parametrize("invalid", [["ok", 1], [None], [["nested"]]])
def test_wrap_combo_lists_recursively_rejects_non_strings(invalid: object) -> None:
    registry = make_registry()
    with pytest.raises(TypeError, match="COMBO expects a string"):
        registry.wrap("list<COMBO>", invalid)


def test_wrap_list_rejects_non_sequences() -> None:
    registry = make_registry()
    with pytest.raises(TypeError):
        registry.wrap("list<INT>", 5)
    with pytest.raises(TypeError):
        registry.wrap("list<STRING>", "not-a-list-of-strings")


def test_make_list_value_rejects_mismatched_children() -> None:
    registry = make_registry()
    child = registry.wrap(CORE_INT, 1)
    with pytest.raises(ValueError):
        make_list_value("FLOAT", [child])


def test_tuple_values_preserve_position_types_and_nested_containers() -> None:
    registry = make_registry()
    assert runtime_type_atoms("tuple<IMAGE,list<INT>,asset<tuple<INT,STRING>>>") == (
        "IMAGE",
        "INT",
        "STRING",
    )
    assert runtime_type_atoms("tuple<>") == ()
    for malformed in ("tuple<INT,>", "tuple<INT,STRING>>", "list<tuple<INT,>>", "tuple<,INT>"):
        assert runtime_type_atoms(malformed) is None
    assert combo_type_mismatch_is_error("tuple<INT,COMBO>", TypeExpr.concrete("STRING"))
    assert combo_type_mismatch_is_error(
        "tuple<INT,STRING>", TypeExpr.tuple_of(TypeExpr.concrete("INT"), TypeExpr.concrete("COMBO"))
    )
    assert not combo_type_mismatch_is_error(
        "tuple<INT,STRING>", TypeExpr.tuple_of(TypeExpr.concrete("INT"), TypeExpr.concrete("FLOAT"))
    )
    value = registry.wrap(
        "tuple<INT,STRING,list<INT>,tuple<STRING,INT>>", (7, "a", [9, 2], ("b", 4))
    )
    assert value.resolve() == (7, "a", [9, 2], ("b", 4))
    assert list_children(value) is None
    children = value_children(value)
    assert children is not None
    assert tuple(child.type_id for child in children) == (
        "INT",
        "STRING",
        "list<INT>",
        "tuple<STRING,INT>",
    )
    assert value.fingerprint != registry.wrap(value.type_id, (8, "a", [9, 2], ("b", 4))).fingerprint
    assert registry.wrap("tuple<>", ()).resolve() == ()
    for invalid in ([7, "a"], (7,), (7, "a", 3)):
        with pytest.raises(TypeError, match="expects a tuple of 2 positions"):
            registry.wrap("tuple<INT,STRING>", invalid)
    with pytest.raises(KeyError, match="unregistered"):
        registry.wrap("tuple<INT,unknown>", (7, "a"))


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
            inputs=(InputSpec("count", TypeExpr.concrete("INT")),),
            outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.concrete("INT"))),),
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
            inputs=(InputSpec("values", TypeExpr.list_of(TypeExpr.concrete("INT"))),),
            outputs=(OutputSpec("total", TypeExpr.concrete("INT")),),
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
            outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.concrete("INT"))),),
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


def test_engine_runs_independently_typed_tuple_nodes_and_caches() -> None:
    from dinkster_nodes_foundation import (
        AddInts,
        GetTupleItem,
        IntPrimitive,
        MakeTuple,
        StringPrimitive,
    )

    async def scenario() -> None:
        nodes = (MakeTuple, GetTupleItem, IntPrimitive, StringPrimitive, AddInts)
        registry = make_registry()
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "integer": GraphNode("dinkster.int", {"value": 7}),
                "text": GraphNode("dinkster.string", {"value": "label"}),
                "pair": GraphNode(
                    "std.tuple.make",
                    {
                        "items.first": Link("integer", "value"),
                        "items.second": Link("text", "value"),
                    },
                ),
                "first": GraphNode("std.tuple.get", {"tuple": Link("pair", "tuple"), "index": 0}),
                "last": GraphNode("std.tuple.get", {"tuple": Link("pair", "tuple"), "index": -1}),
            }
        )
        first = await engine.run(graph, ["pair", "first", "last"])
        assert first.outputs["pair"]["tuple"].type_id == "tuple<INT,STRING>"
        assert first.outputs["first"]["item"].type_id == "INT"
        assert first.outputs["first"]["item"].resolve() == 7
        assert first.outputs["last"]["item"].type_id == "STRING"
        assert first.outputs["last"]["item"].resolve() == "label"
        second = await engine.run(graph, ["pair", "first", "last"])
        assert second.executed == ()
        assert set(second.cached) == set(graph.nodes)

        folded = RegionNode(
            kind="fold",
            ports={
                "item": TypeExpr.concrete("INT"),
                "carry": TypeExpr.tuple_of(TypeExpr.concrete("INT"), TypeExpr.concrete("STRING")),
            },
            inputs={"item": [2, 5], "carry": Link("pair", "tuple")},
            element_ports=("item",),
            state_ports=("carry",),
            body=Graph(
                nodes={
                    "count": GraphNode(
                        "std.tuple.get", {"tuple": Link("$region", "carry"), "index": 0}
                    ),
                    "label": GraphNode(
                        "std.tuple.get", {"tuple": Link("$region", "carry"), "index": 1}
                    ),
                    "sum": GraphNode(
                        "std.math.add_ints",
                        {"a": Link("count", "item"), "b": Link("$region", "item")},
                    ),
                    "next": GraphNode(
                        "std.tuple.make",
                        {"items.count": Link("sum", "sum"), "items.label": Link("label", "item")},
                    ),
                }
            ),
            outputs={
                "carry": RegionOutput(Link("next", "tuple"), mode="state"),
                "observed": RegionOutput(Link("next", "tuple")),
            },
        )
        carried = await engine.run(Graph(nodes={**graph.nodes, "fold": folded}), ["fold"])
        assert carried.outputs["fold"]["carry"].type_id == "tuple<INT,STRING>"
        assert carried.outputs["fold"]["carry"].resolve() == (14, "label")
        assert carried.outputs["fold"]["observed"].type_id == "list<tuple<INT,STRING>>"
        assert carried.outputs["fold"]["observed"].resolve() == [(9, "label"), (14, "label")]

        connected_selector = replace(
            graph.nodes["first"],
            inputs={"tuple": Link("pair", "tuple"), "index": Link("integer", "value")},
        )
        diagnostics = validate(
            Graph(nodes={**graph.nodes, "first": connected_selector}),
            build_schemas(nodes),
            ["first"],
        )
        assert any(d.severity == "error" and "constant INT" in d.message for d in diagnostics)
        for index in (-3, 2):
            invalid_selector = replace(
                graph.nodes["first"], inputs={"tuple": Link("pair", "tuple"), "index": index}
            )
            diagnostics = validate(
                Graph(nodes={**graph.nodes, "first": invalid_selector}),
                build_schemas(nodes),
                ["first"],
            )
            assert [(d.node_id, d.code) for d in diagnostics if d.severity == "error"] == [
                ("first", "output-type-invalid")
            ]
            assert "out of range" in diagnostics[0].message
        invalid_source = replace(graph.nodes["first"], inputs={"tuple": Link("integer", "value")})
        diagnostics = validate(
            Graph(nodes={**graph.nodes, "first": invalid_source}), build_schemas(nodes), ["first"]
        )
        assert any(
            d.code == "output-type-invalid" and "requires a tuple" in d.message for d in diagnostics
        )
        invalid_body = replace(
            folded,
            body=Graph(
                nodes={
                    **folded.body.nodes,
                    "label": GraphNode(
                        "std.tuple.get", {"tuple": Link("$region", "carry"), "index": 2}
                    ),
                }
            ),
        )
        diagnostics = validate(
            Graph(nodes={**graph.nodes, "fold": invalid_body}), build_schemas(nodes), ["fold"]
        )
        assert any(
            d.code == "output-type-invalid" and d.node_id == "fold/label" for d in diagnostics
        )

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


def test_tuple_positions_prepare_their_host_types_before_output_wrapping() -> None:
    class TupleSource(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            pair = TypeExpr.tuple_of(
                TypeExpr.concrete("test.first"), TypeExpr.concrete("test.second")
            )
            return NodeSchema(
                "test.tuple_source",
                outputs=(OutputSpec("value", pair),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(value=(7, "label"))

    async def scenario() -> None:
        registry = make_registry()
        nodes = (TupleSource,)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )

        async def prepare(atoms: set[str]) -> None:
            for type_id in ("test.first", "test.second"):
                if type_id in atoms and type_id not in registry:
                    registry.register(type_id)

        runtime = replace(engine.pin_execution(), prepare_host_types=prepare)
        result = await engine.run(
            Graph(nodes={"source": GraphNode("test.tuple_source", {})}),
            ["source"],
            execution=runtime,
        )
        assert result.outputs["source"]["value"].type_id == "tuple<test.first,test.second>"
        assert result.outputs["source"]["value"].resolve() == (7, "label")

    asyncio.run(scenario())


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("cache_mode", ["memory", "layered"])
def test_tuple_nodes_preserve_position_types_through_production_hosts(
    tmp_path: Path, in_process: bool, cache_mode: Literal["memory", "layered"]
) -> None:
    from dinkster.compose import PackSpec, compose_serving

    async def scenario() -> None:
        manifest = (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-foundation/dinkster-pack.toml"
        )
        composition = await compose_serving(
            [PackSpec(manifest, in_process=in_process, trust_reserved=True)],
            include_default_packs=False,
            cache_mode=cache_mode,
            cache_dir=tmp_path / "cache",
        )
        try:
            engine = composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "text": GraphNode("dinkster.string", {"value": "first"}),
                    "integer": GraphNode("dinkster.int", {"value": 29}),
                    "pair": GraphNode(
                        "std.tuple.make",
                        {
                            "items.label": Link("text", "value"),
                            "items.count": Link("integer", "value"),
                        },
                    ),
                    "selected": GraphNode(
                        "std.tuple.get", {"tuple": Link("pair", "tuple"), "index": -1}
                    ),
                }
            )
            for run in range(2):
                if run and cache_mode == "layered":
                    engine = composition.make_engine(lambda _event: None)
                result = await engine.run(graph, ["pair", "selected"])
                pair = result.outputs["pair"]["tuple"]
                assert pair.type_id == "tuple<STRING,INT>"
                assert pair.resolve() == ("first", 29)
                assert result.outputs["selected"]["item"].type_id == "INT"
                assert result.outputs["selected"]["item"].resolve() == 29
                if run:
                    assert result.executed == ()
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composition.close()

    asyncio.run(scenario())


# -- validation diagnostics -------------------------------------------------


class TakesScalar(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.takes_scalar",
            inputs=(InputSpec("value", TypeExpr.concrete("INT")),),
            outputs=(OutputSpec("out", TypeExpr.concrete("INT")),),
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
                outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.concrete("FLOAT"))),),
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
    inner = registry.wrap("list<INT>", [1, 2])
    value = make_list_value("list<INT>", [inner, registry.wrap("list<INT>", [3])])

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


def test_tuple_transport_retains_resource_identity_and_projection() -> None:
    registry = make_registry()
    register_resource_handle_type(registry)
    stub = _handle_value(registry, "tuple-model", device="cuda:0")
    stub = replace(stub, meta=ValueMeta({**stub.meta.entries, RESOURCE_OWNER_META_KEY: "owner"}))
    value = make_tuple_value(
        (registry.wrap("STRING", "label"), make_list_value(RESOURCE_HANDLE_TYPE, [stub]))
    )
    stamped = stamp_resource_producer_arm(value, "owner", "arm")
    assert value_resource_provenance_refs(stamped) == (("tuple-model", "owner", "arm", True),)
    mapped = DeviceMap(mapping={"cuda:0": "cuda:1"}).value(stamped)
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, stat = codec.encode(mapped, blobs, [])
    decoded, _ = codec.decode(wire, blobs, [])
    assert stat.transport == "tuple"
    assert decoded.type_id == value.type_id
    assert decoded.fingerprint == value.fingerprint
    assert list_children(decoded) is None
    assert value_resource_provenance_refs(decoded) == (("tuple-model", "owner", "arm", True),)
    leaves = tuple(iter_value_tree(decoded))
    assert leaves[-1].meta.get("resources") == {"gpu": "cuda:1"}
    assert encode_entry({"value": decoded}, registry) is None
    malformed = {**wire, "typeId": "tuple<INT,STRING>"}
    with pytest.raises(BoundaryError, match="tuple child types"):
        codec.decode(malformed, blobs, [])


# -- persisted cache manifests ----------------------------------------------


@pytest.mark.parametrize(
    ("type_id", "payload"),
    [
        ("list<list<INT>>", [[1], [2, 3]]),
        ("tuple<INT,STRING,list<INT>>", (7, "label", [2, 3])),
        ("list<tuple<STRING,INT>>", [("a", 3), ("b", 9)]),
        ("tuple<>", ()),
    ],
)
def test_cache_manifest_roundtrips_lists(type_id: str, payload: object) -> None:
    async def scenario() -> None:
        registry = make_registry()
        outputs = {"values": registry.wrap(type_id, payload)}
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
        assert rehydrated.resolve() == payload

    asyncio.run(scenario())


def test_cache_refuses_resource_stubs_inside_lists() -> None:
    registry = make_registry()
    register_resource_handle_type(registry)
    stub = _handle_value(registry, "model-z")
    outputs = {"models": make_list_value(RESOURCE_HANDLE_TYPE, [stub])}
    assert encode_entry(outputs, registry) is None


@pytest.mark.parametrize("container", ["list", "tuple"])
def test_memory_cache_sees_costs_and_references_through_lists(container: str) -> None:
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
        if container == "tuple":
            entry = {"tuple": make_tuple_value((entry["mixed"], stub))}
        # make_list_value requires matching child types; adjust: separate keys
        await cache.put("k", cast(dict[str, Any], entry))
        assert cache.footprint("ram") == 64  # child cost counted, stub excluded
        assert cache.drop_referencing("model-q") == 1
        assert await cache.get("k") is None

    asyncio.run(scenario())
