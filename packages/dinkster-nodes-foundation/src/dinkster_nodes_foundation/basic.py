"""Basic math/string nodes. Note what node authoring looks like (hazard H9):
plain values in, plain values out, keyed by output id. No envelopes anywhere."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputFamilySpec,
    OutputInterface,
    OutputSpec,
    TypeExpr,
)

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)


class AddInts(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.math.add_ints",
            display_name="Add Ints",
            category="math",
            inputs=(InputSpec("a", INT), InputSpec("b", INT)),
            outputs=(OutputSpec("sum", INT),),
        )

    @classmethod
    def execute(cls, *, a: int, b: int) -> Mapping[str, object]:
        return cls.outputs(sum=a + b)


class MultiplyFloats(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.math.multiply_floats",
            display_name="Multiply Floats",
            category="math",
            inputs=(InputSpec("a", FLOAT), InputSpec("b", FLOAT, default=1.0)),
            outputs=(OutputSpec("product", FLOAT),),
        )

    @classmethod
    def execute(cls, *, a: float, b: float) -> Mapping[str, object]:
        return cls.outputs(product=a * b)


class ConcatStrings(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.string.concat",
            display_name="Concat Strings",
            category="string",
            inputs=(
                InputSpec("a", STRING),
                InputSpec("b", STRING),
                InputSpec("separator", STRING, default=" "),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, a: str, b: str, separator: str) -> Mapping[str, object]:
        return cls.outputs(text=a + separator + b)


class JoinStrings(Node):
    """Autogrow input family (hazard H10): the document decides how many
    "pieces" members exist; execute() receives them as one suffix-keyed
    mapping in document order and stays boundary-oblivious (hazard H9)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.string.join",
            display_name="Join Strings",
            category="string",
            inputs=(InputSpec("separator", STRING, default=", "),),
            input_families=(InputFamilySpec("pieces", STRING, min_members=1),),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, separator: str, pieces: Mapping[str, str]) -> Mapping[str, object]:
        return cls.outputs(text=separator.join(pieces.values()))


class SplitString(Node):
    """Dynamic outputs (hazard H10): the document stores which "parts" members
    exist; execute() reads the membership from the reserved output_spec
    parameter and returns one value per member, grouped under the family id.
    Membership never depends on what execute() computes."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="std.string.split",
            display_name="Split String",
            category="string",
            inputs=(
                InputSpec("text", STRING),
                InputSpec("separator", STRING, default=" "),
            ),
            output_families=(OutputFamilySpec("parts", STRING, min_members=1),),
        )

    @classmethod
    def execute(
        cls, *, text: str, separator: str, output_spec: OutputInterface
    ) -> Mapping[str, object]:
        members = output_spec.family("parts")
        pieces = text.split(separator, len(members) - 1)
        pieces += [""] * (len(members) - len(pieces))
        return cls.outputs(parts=dict(zip(members, pieces, strict=True)))
