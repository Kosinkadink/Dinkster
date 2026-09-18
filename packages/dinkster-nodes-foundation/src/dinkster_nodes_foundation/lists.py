"""Core list combinators (DESIGN 3.13): dumb nodes producing list values.

Zero engine involvement - where v1's executor guessed (implicit zipping,
last-element clamping), the graph says it with a visible node and a
predictable output count. Element types are template variables solved per
invocation at the worker boundary (dinkster_schema.solve), so one MakeList
serves every value type; variable-typed inputs are connection-fed (literals
need a concrete-typed input to wrap with).

Zip is deliberately NOT a node: without a tuple constructor there is no
honest value for it to produce, and zip semantics already live where they
belong - a region's zip binding over multiple element ports. CrossProduct
produces two ALIGNED lists (same length, index-matched) for the same reason;
feed them to a zip-binding region's element ports.
"""

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

INT = TypeExpr.concrete(CORE_INT)
T = TypeExpr.variable("T")
LIST_T = TypeExpr.list_of(T)


class MakeList(Node):
    """Collect same-typed values as whole elements in document order."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.make",
            display_name="Create List",
            category="list",
            input_families=(InputFamilySpec("items", T, min_members=1),),
            outputs=(OutputSpec("list", LIST_T),),
            aliases=("CreateList",),
        )

    @classmethod
    def execute(cls, *, items: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(list=list(items.values()))


class AppendToList(Node):
    """Append each same-typed item as one element after an existing list."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.append",
            display_name="Append to List",
            category="list",
            inputs=(InputSpec("list", LIST_T),),
            input_families=(InputFamilySpec("items", T, min_members=1),),
            outputs=(OutputSpec("list", LIST_T),),
        )

    @classmethod
    def execute(
        cls, *, list: Sequence[object], items: Mapping[str, object]
    ) -> Mapping[str, object]:
        return cls.outputs(list=[*list, *items.values()])


class ListLength(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.length",
            display_name="List Length",
            category="list",
            inputs=(InputSpec("list", LIST_T),),
            outputs=(OutputSpec("length", INT),),
        )

    @classmethod
    def execute(cls, *, list: Sequence[object]) -> Mapping[str, object]:
        return cls.outputs(length=len(list))


class ListElement(Node):
    """Explicit element selection (the list-into-scalar fix). Python index
    semantics, negatives included; out of range is a loud node error."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.element",
            display_name="List Element",
            category="list",
            inputs=(InputSpec("list", LIST_T), InputSpec("index", INT, default=0)),
            outputs=(OutputSpec("item", T),),
        )

    @classmethod
    def execute(cls, *, list: Sequence[object], index: int) -> Mapping[str, object]:
        try:
            item = list[index]
        except IndexError:
            raise ValueError(f"index {index} out of range for a list of {len(list)}") from None
        return cls.outputs(item=item)


class ConcatLists(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.concat",
            display_name="Concat Lists",
            category="list",
            input_families=(InputFamilySpec("lists", LIST_T, min_members=1),),
            outputs=(OutputSpec("list", LIST_T),),
        )

    @classmethod
    def execute(cls, *, lists: Mapping[str, Sequence[object]]) -> Mapping[str, object]:
        combined: list[object] = []
        for part in lists.values():
            combined.extend(part)
        return cls.outputs(list=combined)


class ReverseList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.reverse",
            display_name="Reverse List",
            category="list",
            inputs=(InputSpec("list", LIST_T),),
            outputs=(OutputSpec("list", LIST_T),),
        )

    @classmethod
    def execute(cls, *, list: Sequence[object]) -> Mapping[str, object]:
        return cls.outputs(list=[*reversed(list)])


class SliceList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.slice",
            display_name="Slice List",
            category="list",
            inputs=(
                InputSpec("list", LIST_T),
                InputSpec("start", INT, required=False),
                InputSpec("stop", INT, required=False),
                InputSpec("step", INT, default=1),
            ),
            outputs=(OutputSpec("list", LIST_T),),
        )

    @classmethod
    def execute(
        cls,
        *,
        list: Sequence[object],
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
    ) -> Mapping[str, object]:
        if step == 0:
            raise ValueError("step must not be zero")
        return cls.outputs(list=[*list[slice(start, stop, step)]])


class IntegerRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.range",
            display_name="Integer Range",
            category="list",
            inputs=(
                InputSpec("start", INT, default=0),
                InputSpec("stop", INT),
                InputSpec("step", INT, default=1),
            ),
            outputs=(OutputSpec("list", TypeExpr.list_of(INT)),),
        )

    @classmethod
    def execute(cls, *, start: int, stop: int, step: int) -> Mapping[str, object]:
        if step == 0:
            raise ValueError("step must not be zero")
        return cls.outputs(list=list(range(start, stop, step)))


class RepeatItem(Node):
    """item x count -> list<T>. count=0 is a legal empty list (the
    zero-iteration conditional story), negative counts are node errors."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.repeat",
            display_name="Repeat Item",
            category="list",
            inputs=(InputSpec("item", T), InputSpec("count", INT, default=1)),
            outputs=(OutputSpec("list", LIST_T),),
        )

    @classmethod
    def execute(cls, *, item: object, count: int) -> Mapping[str, object]:
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        return cls.outputs(list=[item] * count)


class CrossProduct(Node):
    """The explicit 3x4=12 sweep: two aligned lists of length |a|*|b|, the
    last input varying fastest - the exact ordering of a region's cross
    binding, so results are interchangeable. Feed both outputs to a
    zip-binding map region's element ports to consume pairs."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.list.cross_product",
            display_name="Cross Product",
            category="list",
            inputs=(
                InputSpec("a", TypeExpr.list_of(TypeExpr.variable("A"))),
                InputSpec("b", TypeExpr.list_of(TypeExpr.variable("B"))),
            ),
            outputs=(
                OutputSpec("a", TypeExpr.list_of(TypeExpr.variable("A"))),
                OutputSpec("b", TypeExpr.list_of(TypeExpr.variable("B"))),
            ),
        )

    @classmethod
    def execute(cls, *, a: Sequence[object], b: Sequence[object]) -> Mapping[str, object]:
        return cls.outputs(
            a=[x for x in a for _ in b],
            b=[y for _ in a for y in b],
        )
