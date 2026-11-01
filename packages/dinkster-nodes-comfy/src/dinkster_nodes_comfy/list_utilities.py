"""Native list construction and indexing nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    CORE_INT,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    Value,
    list_children,
)

from .family_migrations import ordered_family_migration

_ITEM = TypeExpr.wildcard()
_ITEMS = TypeExpr.list_of(_ITEM)


def _execution_items(value: Value) -> tuple[Value, ...]:
    if TypeExpr.runtime_cardinality(value.type_id) != "list":
        return (value,)
    children = list_children(value)
    assert children is not None
    return children


class CreateList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CreateList",
            display_name="Create List",
            category="utilities",
            input_families=(
                InputFamilySpec(
                    "inputs",
                    (InputSpec("input", _ITEM),),
                    min_members=1,
                    max_members=10,
                    member_prefix="input",
                    optional_after_minimum=True,
                ),
            ),
            outputs=(OutputSpec("list", _ITEMS),),
            search_terms=("Image Iterator", "Text Iterator", "Iterator"),
            input_values=True,
            replacements=(ordered_family_migration("CreateList", "inputs", "input", ("list",)),),
        )

    @classmethod
    def execute(
        cls, *, inputs: Mapping[str, object], input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        items: list[Value] = []
        for name in inputs:
            items.extend(_execution_items(input_values[f"inputs.{name}"]))
        return cls.outputs(list=items)


class GetItemFromList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetItemFromList",
            display_name="Get Item From List",
            category="utilities",
            inputs=(
                InputSpec("list", _ITEM),
                InputSpec(
                    "index",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    accepts_list=True,
                ),
            ),
            outputs=(OutputSpec("item", _ITEM, display_name="*"),),
            input_values=True,
        )

    @classmethod
    def execute(
        cls, *, list: object, index: object, input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        selected_index = _execution_items(input_values["index"])[0]
        if selected_index.type_id != CORE_INT:
            raise TypeError("index requires an INT or a list of INT values")
        children = _execution_items(input_values["list"])
        return cls.outputs(item=children[cast("int", selected_index.resolve())])


LIST_UTILITY_NODES = (CreateList, GetItemFromList)
