"""Public list-node ordering, indexing, and schema contracts."""

from __future__ import annotations

import pytest
from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_nodes_comfy.list_utilities import CreateList, GetItemFromList


def test_list_nodes_preserve_member_order_and_nested_items() -> None:
    nested = ["nested", 13]
    value = CreateList.execute(inputs={"input2": ["first", nested], "input0": [], "input1": [7]})
    assert value == {"list": ["first", nested, 7]}
    items = value["list"]
    assert isinstance(items, list)
    assert GetItemFromList.execute(list=items, index=-2) == {"item": nested}
    assert GetItemFromList.execute(list=items, index=0) == {"item": "first"}
    with pytest.raises(IndexError):
        GetItemFromList.execute(list=items, index=3)


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
