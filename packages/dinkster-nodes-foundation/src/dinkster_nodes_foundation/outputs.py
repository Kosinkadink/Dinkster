"""Edit-time output declarations for expressions and list fan-out."""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputInterface,
    OutputSpec,
    StringWidget,
    TypeExpr,
    output_descriptor_entries,
)

from .expression import BOOLEAN, FLOAT, INT, STRING, evaluate_expression

SCALAR_OUTPUT_CHOICES = (
    OutputSpec("float", FLOAT),
    OutputSpec("int", INT),
    OutputSpec("boolean", BOOLEAN),
    OutputSpec("string", STRING),
)


class MathExpressions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.math.expressions",
            display_name="Declared Expressions",
            category="math",
            inputs=(
                InputSpec(
                    "entries",
                    STRING,
                    default='{"entries":[{"id":"m0","name":"Result","type":"float","expression":"1+2"}]}',
                    widget=StringWidget(multiline=True),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "values",
                    TypeExpr.union("core.float", "core.int", "core.boolean"),
                    member_names=tuple(string.ascii_lowercase),
                ),
            ),
            output_descriptors=OutputDescriptorsSpec("entries", SCALAR_OUTPUT_CHOICES, 32),
        )

    @classmethod
    def execute(
        cls, *, entries: str, values: Mapping[str, object], output_spec: OutputInterface
    ) -> Mapping[str, object]:
        declarations = output_descriptor_entries(entries)
        results: dict[str, object] = {}
        for output, entry in zip(output_spec.outputs, declarations, strict=True):
            expression = entry.get("expression")
            if not isinstance(expression, str):
                raise ValueError(f"output '{output.id}' requires an expression string")
            floating, integer, boolean = evaluate_expression(expression, values)
            type_id = output.type.runtime_type_id()
            assert type_id is not None
            results[output.id] = {
                "core.float": floating,
                "core.int": integer,
                "core.boolean": boolean,
                "core.string": str(floating),
            }[type_id]
        return results


class EntryFanOut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.list.fan_out",
            display_name="Entry List Fan-out",
            category="list",
            inputs=(
                InputSpec(
                    "entries",
                    STRING,
                    default='{"entries":[{"id":"m0","name":"Entry","type":"int","value":0}]}',
                    widget=StringWidget(multiline=True),
                ),
                InputSpec("items", TypeExpr.list_of(TypeExpr.variable("T")), required=False),
            ),
            output_descriptors=OutputDescriptorsSpec("entries", SCALAR_OUTPUT_CHOICES, 512),
        )

    @classmethod
    def execute(
        cls, *, entries: str, output_spec: OutputInterface, items: Sequence[object] = ()
    ) -> Mapping[str, object]:
        results: dict[str, object] = {}
        for output, entry in zip(
            output_spec.outputs, output_descriptor_entries(entries), strict=True
        ):
            if "index" in entry:
                index = entry["index"]
                if type(index) is not int or not -len(items) <= index < len(items):
                    raise ValueError(f"output '{output.id}' index is outside the input list")
                results[output.id] = items[index]
            elif "value" in entry:
                results[output.id] = entry["value"]
            else:
                raise ValueError(f"output '{output.id}' requires a value or list index")
        return results
