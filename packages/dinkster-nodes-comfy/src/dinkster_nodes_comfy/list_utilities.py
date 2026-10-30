"""Native list construction and indexing nodes."""

from __future__ import annotations

from collections.abc import Mapping

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
                    (InputSpec("input", _ITEMS),),
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
        cls, *, inputs: Mapping[str, list[object]], input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        items: list[Value] = []
        for name in inputs:
            children = list_children(input_values[f"inputs.{name}"])
            assert children is not None
            items.extend(children)
        return cls.outputs(list=items)


class GetItemFromList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetItemFromList",
            display_name="Get Item From List",
            category="utilities",
            inputs=(
                InputSpec("list", _ITEMS),
                InputSpec(
                    "index",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                ),
            ),
            outputs=(OutputSpec("item", _ITEM, display_name="*"),),
            input_values=True,
        )

    @classmethod
    def execute(
        cls, *, list: list[object], index: int, input_values: Mapping[str, Value]
    ) -> Mapping[str, object]:
        children = list_children(input_values["list"])
        assert children is not None
        return cls.outputs(item=children[index])


LIST_UTILITY_NODES = (CreateList, GetItemFromList)
