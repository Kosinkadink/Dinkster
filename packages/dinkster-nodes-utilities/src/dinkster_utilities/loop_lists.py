"""Typed list construction and selection used by loop expansion."""

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

INT = TypeExpr.concrete(CORE_INT)
T = TypeExpr.variable("T")
LIST_T = TypeExpr.list_of(T)


class MakeList(Node):
    """Collect same-typed values as whole elements in document order."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.utilities.make_list",
            display_name="Create List",
            category="list",
            input_families=(InputFamilySpec("items", T, min_members=1),),
            outputs=(OutputSpec("list", LIST_T),),
            input_values=True,
        )

    @classmethod
    def execute(cls, *, input_values: Mapping[str, Value]) -> Mapping[str, object]:
        return cls.outputs(list=list(input_values.values()))


class ListElement(Node):
    """Select a typed child with Python index semantics."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.utilities.list_element",
            display_name="List Element",
            category="list",
            inputs=(InputSpec("list", LIST_T), InputSpec("index", INT, default=0)),
            outputs=(OutputSpec("item", T),),
            input_values=True,
        )

    @classmethod
    def execute(cls, *, index: int, input_values: Mapping[str, Value]) -> Mapping[str, object]:
        children = list_children(input_values["list"])
        assert children is not None
        try:
            item = children[index]
        except IndexError:
            raise ValueError(f"index {index} out of range for a list of {len(children)}") from None
        return cls.outputs(item=item)
