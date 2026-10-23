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
)


class PrimitiveString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PrimitiveString",
            display_name="Text",
            category="utilities/primitive",
            inputs=(InputSpec("value", TypeExpr.concrete(CORE_STRING), widget=StringWidget()),),
            outputs=(OutputSpec("STRING", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(STRING=value)


class PrimitiveStringMultiline(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PrimitiveStringMultiline",
            display_name="Text (Multiline)",
            category="utilities/primitive",
            inputs=(
                InputSpec(
                    "value", TypeExpr.concrete(CORE_STRING), widget=StringWidget(multiline=True)
                ),
            ),
            outputs=(OutputSpec("STRING", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(STRING=value)


class PrimitiveInt(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PrimitiveInt",
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

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(INT=value)


class PrimitiveFloat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PrimitiveFloat",
            display_name="Float",
            category="utilities/primitive",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.concrete(CORE_FLOAT),
                    widget=NumberWidget(min=float(-sys.maxsize), max=float(sys.maxsize), step=0.1),
                ),
            ),
            outputs=(OutputSpec("FLOAT", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        return cls.outputs(FLOAT=value)


class PrimitiveBoolean(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PrimitiveBoolean",
            display_name="Boolean",
            category="utilities/primitive",
            inputs=(InputSpec("value", TypeExpr.concrete(CORE_BOOLEAN)),),
            outputs=(OutputSpec("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, value: bool) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=value)
