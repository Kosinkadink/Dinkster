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
)

_ITEM = TypeExpr.variable("T")
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
                ),
            ),
            outputs=(OutputSpec("list", _ITEMS),),
            search_terms=("Image Iterator", "Text Iterator", "Iterator"),
        )

    @classmethod
    def execute(cls, *, inputs: Mapping[str, list[object]]) -> Mapping[str, object]:
        return cls.outputs(list=[item for values in inputs.values() for item in values])


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
        )

    @classmethod
    def execute(cls, *, list: list[object], index: int) -> Mapping[str, object]:
        return cls.outputs(item=list[index])


LIST_UTILITY_NODES = (CreateList, GetItemFromList)
