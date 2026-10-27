"""Public list-node ordering, indexing, and schema contracts."""

from __future__ import annotations

import pytest
from dinkster_api.v1 import (
    TypeExpr,
    TypeRegistry,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy.list_utilities import CreateList, GetItemFromList
from dinkster_values import register_core_types


def test_list_nodes_preserve_member_order_and_nested_items() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    first = registry.wrap("STRING", "first")
    nested = registry.wrap("list<*>", [registry.wrap("STRING", "nested"), registry.wrap("INT", 13)])
    number = registry.wrap("INT", 7)
    value = CreateList.execute(
        inputs={"input2": ["first", ["nested", 13]], "input0": [], "input1": [7]},
        input_values={
            "inputs.input2": registry.wrap("list<*>", [first, nested]),
            "inputs.input0": registry.wrap("list<STRING>", []),
            "inputs.input1": registry.wrap("list<INT>", [number]),
        },
    )
    assert value == {"list": [first, nested, number]}
    items = value["list"]
    envelope = registry.wrap("list<*>", items)
    assert GetItemFromList.execute(
        list=["first", ["nested", 13], 7], index=-2, input_values={"list": envelope}
    ) == {"item": nested}
    assert GetItemFromList.execute(
        list=["first", ["nested", 13], 7], index=0, input_values={"list": envelope}
    ) == {"item": first}
    with pytest.raises(IndexError):
        GetItemFromList.execute(
            list=["first", ["nested", 13], 7], index=3, input_values={"list": envelope}
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
    assert CreateList.schema().outputs[0].type == TypeExpr.list_of(TypeExpr.wildcard())
    assert GetItemFromList.schema().outputs[0].type == TypeExpr.wildcard()
