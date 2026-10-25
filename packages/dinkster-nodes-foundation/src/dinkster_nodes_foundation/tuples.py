"""Fixed positional products with independently typed elements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    CORE_INT,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)


class MakeTuple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.tuple.make",
            display_name="Create Tuple",
            category="tuple",
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
            node_type="std.tuple.get",
            display_name="Tuple Item",
            category="tuple",
            inputs=(
                InputSpec("tuple", TypeExpr.variable("Tuple")),
                InputSpec("index", TypeExpr.concrete(CORE_INT), default=0),
            ),
            outputs=(OutputSpec("item", TypeExpr.tuple_element("tuple", "index")),),
        )

    @classmethod
    def execute(cls, *, tuple: Sequence[object], index: int) -> Mapping[str, object]:
        return cls.outputs(item=tuple[index])
