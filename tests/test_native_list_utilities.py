"""Public list-node ordering, indexing, and schema contracts."""

from __future__ import annotations

import asyncio

import pytest
from dinkster_api.v1 import (
    InputSpec,
    TypeExpr,
    TypeRegistry,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_nodes_comfy.list_utilities import CreateList, GetItemFromList
from dinkster_nodes_foundation import IntPrimitive
from dinkster_schema import bind_type_variables, build_node_types, build_schemas
from dinkster_values import register_core_types
from dinkster_workers import InProcessWorker


def test_list_nodes_preserve_member_order_and_nested_items() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    first = registry.wrap("STRING", "first")
    nested = registry.wrap("list<*>", [registry.wrap("STRING", "nested"), registry.wrap("INT", 13)])
    number = registry.wrap("INT", 7)
    registry.register("OPAQUE")
    opaque = registry.wrap("OPAQUE", [2, 5])
    value = CreateList.execute(
        inputs={"input2": ["first", ["nested", 13]], "input0": [], "input1": 7, "input3": [2, 5]},
        input_values={
            "inputs.input2": registry.wrap("list<*>", [first, nested]),
            "inputs.input0": registry.wrap("list<STRING>", []),
            "inputs.input1": number,
            "inputs.input3": opaque,
        },
    )
    assert value == {"list": [first, nested, number, opaque]}
    items = value["list"]
    envelope = registry.wrap("list<*>", items)
    assert GetItemFromList.execute(
        list=envelope.resolve(),
        index=-3,
        input_values={"list": envelope, "index": registry.wrap("INT", -3)},
    ) == {"item": nested}
    assert GetItemFromList.execute(
        list=envelope.resolve(),
        index=0,
        input_values={"list": envelope, "index": registry.wrap("INT", 0)},
    ) == {"item": first}
    with pytest.raises(IndexError):
        GetItemFromList.execute(
            list=envelope.resolve(),
            index=4,
            input_values={"list": envelope, "index": registry.wrap("INT", 4)},
        )


def test_list_schemas_roundtrip_without_collapsing_list_types() -> None:
    for node in (CreateList, GetItemFromList):
        schema = node.schema()
        assert schema_from_wire(schema_to_wire(schema)) == schema
    family = CreateList.schema().input_families[0]
    assert (family.id, family.member_prefix, family.min_members, family.max_members) == (
        "inputs",
        "input",
        1,
        10,
    )
    assert GetItemFromList.schema().inputs[1].default == 0
    assert GetItemFromList.schema().inputs[1].accepts_list
    assert CreateList.schema().outputs[0].type == TypeExpr.list_of(TypeExpr.wildcard())
    assert GetItemFromList.schema().outputs[0].type == TypeExpr.wildcard()


def test_native_list_nodes_execute_scalar_and_list_envelopes_without_wrappers() -> None:
    async def run() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        nodes = (CreateList, GetItemFromList, IntPrimitive)
        schemas = build_schemas(nodes)
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "items": GraphNode(
                    "CreateList",
                    {
                        "inputs.input2": TypedLiteral("STRING", "first"),
                        "inputs.input0": TypedLiteral("list<INT>", [7, 13]),
                        "inputs.input1": TypedLiteral("list<STRING>", []),
                    },
                ),
                "selected": GraphNode(
                    "GetItemFromList",
                    {
                        "list": Link("items", "list"),
                        "index": TypedLiteral("list<INT>", [-1, 0]),
                    },
                ),
                "indices": GraphNode(
                    "CreateList",
                    {"inputs.input0": TypedLiteral("list<INT>", [-2, 0])},
                ),
                "linked": GraphNode(
                    "GetItemFromList",
                    {"list": Link("items", "list"), "index": Link("indices", "list")},
                ),
                "scalar": GraphNode(
                    "GetItemFromList",
                    {
                        "list": TypedLiteral("STRING", "singleton"),
                        "index": 0,
                    },
                ),
            }
        )
        result = await engine.run(graph, ["items", "selected", "linked", "scalar"])
        assert result.outputs["items"]["list"].resolve() == ["first", 7, 13]
        assert result.outputs["selected"]["item"].type_id == "INT"
        assert result.outputs["selected"]["item"].resolve() == 13
        assert result.outputs["linked"]["item"].resolve() == 7
        assert result.outputs["scalar"]["item"].resolve() == "singleton"
        rejected = Graph(
            nodes={
                "plain": GraphNode("dinkster.int", {"value": TypedLiteral("list<INT>", [2])}),
            }
        )
        assert any(d.severity == "error" for d in validate(rejected, schemas, ["plain"]))

    asyncio.run(run())


def test_whole_list_admission_binds_the_declared_element_type() -> None:
    assert bind_type_variables(
        [InputSpec("item", TypeExpr.variable("T"), accepts_list=True)],
        {"item": "list<INT>"},
    ) == {"T": "INT"}
