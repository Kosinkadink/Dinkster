"""Primitive value nodes with the interfaces of ComfyUI core nodes."""

from __future__ import annotations

import sys
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    with_identity_replacement,
)


class PrimitiveString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.primitive_string",
            display_name="Text",
            category="utilities/primitive",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=False),
                ),
            ),
            outputs=(OutputSpec("STRING", TypeExpr.concrete(CORE_STRING)),),
        )
        return with_identity_replacement(schema, "PrimitiveString")

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(STRING=value)


class PrimitiveStringMultiline(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.primitive_string_multiline",
            display_name="Text (Multiline)",
            category="utilities/primitive",
            inputs=(
                InputSpec(
                    "value", TypeExpr.concrete(CORE_STRING), widget=StringWidget(multiline=True)
                ),
            ),
            outputs=(OutputSpec("STRING", TypeExpr.concrete(CORE_STRING)),),
        )
        return with_identity_replacement(schema, "PrimitiveStringMultiline")

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(STRING=value)


class PrimitiveInt(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.primitive_int",
            display_name="Int",
            category="utilities/primitive",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete(CORE_INT),
                    widget=NumberWidget(
                        min=-sys.maxsize, max=sys.maxsize, control_after_generate="fixed"
                    ),
                ),
            ),
            outputs=(OutputSpec("INT", TypeExpr.concrete(CORE_INT)),),
        )
        return with_identity_replacement(schema, "PrimitiveInt")

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(INT=value)


class PrimitiveFloat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.primitive_float",
            display_name="Float",
            category="utilities/primitive",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete(CORE_FLOAT),
                    widget=NumberWidget(min=-float(sys.maxsize), max=float(sys.maxsize), step=0.1),
                ),
            ),
            outputs=(OutputSpec("FLOAT", TypeExpr.concrete(CORE_FLOAT)),),
        )
        return with_identity_replacement(schema, "PrimitiveFloat")

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        return cls.outputs(FLOAT=value)


class PrimitiveBoolean(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.primitive_boolean",
            display_name="Boolean",
            category="utilities/primitive",
            inputs=(InputSpec("value", TypeExpr.concrete(CORE_BOOLEAN)),),
            outputs=(OutputSpec("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),),
        )
        return with_identity_replacement(schema, "PrimitiveBoolean")

    @classmethod
    def execute(cls, *, value: bool) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=value)
