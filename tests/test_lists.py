"""List and tuple values through schemas, engines, worker boundaries and caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from dinkster_caches import MemoryLRUCache, encode_entry
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput, validate
from dinkster_schema import (
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    CORE_COMBO,
    RESOURCE_HANDLE_TYPE,
    ResourceHandle,
    TypeRegistry,
    Value,
    list_children,
    list_type_id,
    make_list_value,
    register_core_types,
    register_resource_handle_type,
    value_resource_ids,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import ValueCodec

from tests.async_scenarios import run_scenario

# -- grammar and TypeExpr ---------------------------------------------------


# -- value envelopes --------------------------------------------------------


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_list_fingerprint_is_order_sensitive_and_stable() -> None:
    registry = make_registry()
    a = registry.wrap("list<INT>", [1, 2, 3])
    b = registry.wrap("list<INT>", [1, 2, 3])
    c = registry.wrap("list<INT>", [3, 2, 1])
    d = registry.wrap("list<INT>", [1, 2])
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.fingerprint != d.fingerprint


@pytest.mark.parametrize("nested", [False, True])
def test_wrap_combo_lists_preserves_scalar_types_across_boundary(nested: bool) -> None:
    registry = make_registry()
    selected = ["ok", 1, True, 1.5, "1", False, 0]
    children = [registry.wrap(CORE_COMBO, item) for item in selected]
    element_type = "list<COMBO>" if nested else CORE_COMBO
    raw = [selected] if nested else selected
    wrapped = [make_list_value(CORE_COMBO, children)] if nested else children
    value = registry.wrap(list_type_id(element_type), raw)
    assert value.fingerprint == make_list_value(element_type, wrapped).fingerprint
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    decoded, _ = codec.decode(wire, blobs, [])
    result = decoded.resolve()
    assert isinstance(result, list)
    if nested:
        result = result[0]
        assert isinstance(result, list)
    assert result == selected
    assert [type(item) for item in result] == [type(item) for item in selected]
    assert decoded.fingerprint == value.fingerprint


def _handle_value(registry: TypeRegistry, rid: str, device: str = "cuda:0"):
    handle = ResourceHandle(
        resource_id=rid,
        kind="gpu",
        residency={"gpu": device},
        cost={f"vram:{device}": 100},
        obj=object(),
    )
    return registry.wrap(RESOURCE_HANDLE_TYPE, handle)


# -- engine end to end ------------------------------------------------------


class IntPrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete("INT")
        return NodeSchema(
            "dinkster.int",
            inputs=(InputSpec("value", integer),),
            outputs=(OutputSpec("value", integer),),
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(value=value)


class StringPrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete("STRING")
        return NodeSchema(
            "dinkster.string",
            inputs=(InputSpec("value", string),),
            outputs=(OutputSpec("value", string),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class MakeTuple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "std.tuple.make",
            input_families=(
                InputFamilySpec("items", TypeExpr.variable("T"), independent_types=True),
            ),
            outputs=(OutputSpec("tuple", TypeExpr.tuple_from_family("items")),),
        )

    @classmethod
    def execute(cls, *, items: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(tuple=tuple(items.values()))


class GetTupleItem(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "std.tuple.get",
            inputs=(
                InputSpec("tuple", TypeExpr.variable("Tuple")),
                InputSpec("index", TypeExpr.concrete("INT"), default=0),
            ),
            outputs=(OutputSpec("item", TypeExpr.tuple_element("tuple", "index")),),
        )

    @classmethod
    def execute(cls, *, tuple: tuple[object, ...], index: int) -> Mapping[str, object]:
        return cls.outputs(item=tuple[index])


class AddInts(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete("INT")
        return NodeSchema(
            "std.math.add_ints",
            inputs=(InputSpec("a", integer), InputSpec("b", integer)),
            outputs=(OutputSpec("sum", integer),),
        )

    @classmethod
    def execute(cls, *, a: int, b: int) -> Mapping[str, object]:
        return cls.outputs(sum=a + b)


TUPLE_NODES = (MakeTuple, GetTupleItem, IntPrimitive, StringPrimitive, AddInts)


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


class CollectIndependentValues(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.collect-independent",
            inputs=(
                InputSpec("left", TypeExpr.wildcard()),
                InputSpec("right", TypeExpr.wildcard()),
            ),
            outputs=(OutputSpec("values", TypeExpr.list_of(TypeExpr.wildcard())),),
            input_values=True,
        )

    @classmethod
    def execute(
        cls, *, left: object, right: object, input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        return cls.outputs(values=[input_values["left"], input_values["right"]])


class PickIndependentValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.pick-independent",
            inputs=(InputSpec("values", TypeExpr.list_of(TypeExpr.wildcard())),),
            outputs=(OutputSpec("item", TypeExpr.wildcard()),),
            input_values=True,
        )

    @classmethod
    def execute(
        cls, *, values: list[object], input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        children = list_children(input_values["values"])
        assert children is not None
        return cls.outputs(item=children[-1])


HETEROGENEOUS_NODES = (
    CollectIndependentValues,
    PickIndependentValue,
    IntPrimitive,
    StringPrimitive,
)


@pytest.mark.parametrize("in_process", [False, True])
def test_heterogeneous_lists_cross_ordinary_hosts_and_persistent_cache(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster.compose import PackSpec, compose_serving

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        module = f"mixed_list_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text("from tests.test_lists import HETEROGENEOUS_NODES\n")
        manifest.write_text(
            '[pack]\nname="mixed-list-test"\npublisher="dinkster"\n'
            'namespaces=["test","dinkster"]\n[pack.entry]\n'
            f'nodes="{module}:HETEROGENEOUS_NODES"\n'
        )
        composition = await compose_serving(
            [PackSpec(manifest, in_process=in_process, trust_reserved=True)],
            include_default_packs=False,
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
        )
        try:
            graph = Graph(
                nodes={
                    "number": GraphNode("dinkster.int", {"value": 29}),
                    "text": GraphNode("dinkster.string", {"value": "seven"}),
                    "mixed": GraphNode(
                        "test.collect-independent",
                        {"left": Link("text", "value"), "right": Link("number", "value")},
                    ),
                    "reverse": GraphNode(
                        "test.collect-independent",
                        {"left": Link("number", "value"), "right": Link("text", "value")},
                    ),
                    "selected": GraphNode(
                        "test.pick-independent", {"values": Link("mixed", "values")}
                    ),
                }
            )
            for run in range(2):
                engine = composition.make_engine(lambda _event: None)
                result = await engine.run(graph, ["mixed", "reverse", "selected"])
                assert result.outputs["mixed"]["values"].type_id == "list<*>"
                assert result.outputs["mixed"]["values"].resolve() == ["seven", 29]
                assert result.outputs["reverse"]["values"].resolve() == [29, "seven"]
                assert result.outputs["selected"]["item"].type_id == "INT"
                assert result.outputs["selected"]["item"].resolve() == 29
                if run:
                    assert result.executed == ()
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composition.close()

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_engine_wraps_list_literals() -> None:
    async def scenario() -> None:
        SumList.saw.clear()
        engine = make_engine()
        graph = Graph(nodes={"s": GraphNode("test.sum_list", {"values": [5, 6]})})
        result = await engine.run(graph, ["s"])
        assert result.outputs["s"]["total"].resolve() == 11
        assert SumList.saw == [[5, 6]]

    run_scenario(scenario())


def test_engine_runs_independently_typed_tuple_nodes_and_caches() -> None:
    async def scenario() -> None:
        nodes = TUPLE_NODES
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

    run_scenario(scenario())


@pytest.mark.parametrize(("index", "expected", "type_id"), [(0, 7, "INT"), (1, "label", "STRING")])
def test_worker_projects_tuple_selectors_in_output_families(
    index: int, expected: object, type_id: str
) -> None:
    from dinkster_protocol import Invocation
    from dinkster_schema import OutputFamilySpec, OutputInterface, elaborate

    class Pick(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                "test.family-pick",
                inputs=(
                    InputSpec(
                        "values",
                        TypeExpr.tuple_of(TypeExpr.concrete("INT"), TypeExpr.concrete("STRING")),
                    ),
                    InputSpec("index", TypeExpr.concrete("INT")),
                ),
                output_families=(
                    OutputFamilySpec("picked", TypeExpr.tuple_element("values", "index")),
                ),
            )

        @classmethod
        def execute(
            cls, *, values: tuple[int, str], index: int, output_spec: OutputInterface
        ) -> Mapping[str, object]:
            return cls.outputs(
                picked={name: values[index] for name in output_spec.family("picked")}
            )

    async def scenario() -> None:
        registry = make_registry()
        schema = elaborate(
            Pick.define_schema(),
            {"values": (7, "label"), "index": index},
            {"picked": ("right",)},
        )
        worker = InProcessWorker(build_node_types((Pick,)), registry)
        try:
            result = await worker.invoke(
                Invocation(
                    invocation_id="family-selector",
                    node_id="pick",
                    node_type="test.family-pick",
                    inputs={
                        "values": registry.wrap("tuple<INT,STRING>", (7, "label")),
                        "index": registry.wrap("INT", index),
                    },
                    effective_schema=schema,
                    output_members=(("picked", ("right",)),),
                )
            )
            assert result.error is None
            assert result.outputs is not None
            assert result.outputs["picked.right"].type_id == type_id
            assert result.outputs["picked.right"].resolve() == expected
        finally:
            await worker.close()

    run_scenario(scenario())


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

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("cache_mode", ["memory", "layered"])
def test_tuple_nodes_preserve_position_types_through_production_hosts(
    tmp_path: Path, in_process: bool, cache_mode: Literal["memory", "layered"]
) -> None:
    from dinkster.compose import PackSpec, compose_serving

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        module = f"tuple_fixture_{in_process}_{cache_mode}"
        (tmp_path / f"{module}.py").write_text("from tests.test_lists import TUPLE_NODES\n")
        manifest.write_text(
            '[pack]\nname="tuple-test"\npublisher="dinkster"\n'
            'namespaces=["std","dinkster"]\n[pack.entry]\n'
            f'nodes="{module}:TUPLE_NODES"\n'
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

    run_scenario(scenario())


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


def test_non_list_literal_on_list_input_is_an_error() -> None:
    graph = Graph(nodes={"s": GraphNode("test.sum_list", {"values": 5})})
    assert "literal-shape" in _diag_codes(graph, (SumList,), ["s"])


# -- boundary codec ---------------------------------------------------------


def test_heterogeneous_transport_keeps_child_conversion_and_resources() -> None:
    import numpy as np
    from dinkster_api.v1 import image_input

    registry = make_registry()
    registry.register("IMAGE", input_convert=image_input)
    register_resource_handle_type(registry)
    image = registry.wrap("IMAGE", np.array([0, 255, 64], dtype=np.uint8).reshape(1, 1, 1, 3))
    stub = _handle_value(registry, "model-mixed", device="cuda:1")
    value = registry.wrap("list<*>", [registry.wrap("STRING", "image"), image, stub])
    sender = ValueCodec(registry, use_shm=False, accept_shm=False)
    receiver = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = sender.encode(value, blobs, [])
    decoded, _ = receiver.decode(wire, blobs, [])
    children = list_children(decoded)
    assert children is not None
    assert [child.type_id for child in children] == ["STRING", "IMAGE", RESOURCE_HANDLE_TYPE]
    assert children[1].fingerprint == image.fingerprint
    assert value_resource_ids(decoded) == ("model-mixed",)
    converted = cast(list[Any], registry.input_object(decoded.type_id, decoded))
    assert converted[0] == "image"
    np.testing.assert_array_equal(
        converted[1], np.array([0, 1, 64 / 255], dtype=np.float32).reshape(1, 1, 1, 3)
    )
    assert cast(Any, image.resolve()).dtype == np.uint8
    assert encode_entry({"mixed": decoded}, registry) is None


# -- persisted cache manifests ----------------------------------------------
