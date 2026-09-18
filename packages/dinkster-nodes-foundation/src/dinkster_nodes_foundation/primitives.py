"""Native primitive value nodes and the text preview sink.

The Dinkster-native successors to ComfyUI's Primitive* nodes: always
composed (zero dependence on a ComfyUI install), semantic output ids,
and first-class presentation metadata (wire v11 NUMBER/STRING widgets)
instead of the compat translation's stripped-down twins. Each value
primitive claims its legacy v1 class name as an alias; when the compat
pack composes, merge_native_nodes evicts the translated twin so the
alias resolves here, unambiguously.

dinkster.preview_any deliberately does NOT claim ComfyUI's "PreviewAny"
name: it executes in the engine process, where only engine-decodable
payloads (core scalars/strings/lists, comfy.IMAGE as numpy) resolve.
The translated comfy.PreviewAny runs in the compat worker with torch
and can stringify resident/tensor values, so both coexist - the alias
claim is ledgered with its revival trigger in ROADMAP.
"""

from __future__ import annotations

import json
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
    OutputKnownValue,
    OutputSpec,
    StringWidget,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
)

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)

_PRIMITIVE_CATEGORY = "utilities/primitive"


def _text_representations(*, default: str) -> WidgetRepresentations:
    return WidgetRepresentations(
        representations=(
            WidgetRepresentation(
                "single-line", StringWidget(multiline=False), display_name="Single line"
            ),
            WidgetRepresentation(
                "multiline", StringWidget(multiline=True), display_name="Multiline"
            ),
        ),
        default=default,
        user_switchable=True,
    )


class IntPrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.int",
            display_name="Int",
            category=_PRIMITIVE_CATEGORY,
            inputs=(
                InputSpec(
                    "value",
                    INT,
                    required=False,
                    default=0,
                    display_name="Value",
                    # No min/max: unbounded, unlike ComfyUI's +/-sys.maxsize
                    # sentinels (which exceed the JS safe-integer range).
                    widget=NumberWidget(control_after_generate="fixed"),
                ),
            ),
            outputs=(OutputSpec("value", INT, known_value=OutputKnownValue(input="value")),),
            aliases=("PrimitiveInt",),
            search_terms=("PrimitiveInt", "integer", "number"),
        )

    @classmethod
    def execute(cls, *, value: int = 0) -> Mapping[str, object]:
        return cls.outputs(value=value)


class FloatPrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.float",
            display_name="Float",
            category=_PRIMITIVE_CATEGORY,
            inputs=(
                InputSpec(
                    "value",
                    FLOAT,
                    required=False,
                    default=0.0,
                    display_name="Value",
                    widget=NumberWidget(step=0.1),
                ),
            ),
            outputs=(OutputSpec("value", FLOAT, known_value=OutputKnownValue(input="value")),),
            aliases=("PrimitiveFloat",),
            search_terms=("PrimitiveFloat", "number"),
        )

    @classmethod
    def execute(cls, *, value: float = 0.0) -> Mapping[str, object]:
        return cls.outputs(value=value)


class StringPrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string",
            display_name="Text",
            category=_PRIMITIVE_CATEGORY,
            inputs=(
                InputSpec(
                    "value",
                    STRING,
                    required=False,
                    default="",
                    display_name="Value",
                    widget=_text_representations(default="single-line"),
                ),
            ),
            outputs=(OutputSpec("value", STRING, known_value=OutputKnownValue(input="value")),),
            aliases=("PrimitiveString",),
            search_terms=("PrimitiveString", "text", "string", "text box", "prompt"),
        )

    @classmethod
    def execute(cls, *, value: str = "") -> Mapping[str, object]:
        return cls.outputs(value=value)


class StringMultilinePrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string_multiline",
            display_name="Text (Multiline)",
            category=_PRIMITIVE_CATEGORY,
            inputs=(
                InputSpec(
                    "value",
                    STRING,
                    required=False,
                    default="",
                    display_name="Value",
                    widget=_text_representations(default="multiline"),
                ),
            ),
            outputs=(OutputSpec("value", STRING, known_value=OutputKnownValue(input="value")),),
            aliases=("PrimitiveStringMultiline",),
            search_terms=(
                "PrimitiveStringMultiline",
                "text multiline",
                "string multiline",
                "text box",
                "prompt",
            ),
            search_visibility="hidden",
        )

    @classmethod
    def execute(cls, *, value: str = "") -> Mapping[str, object]:
        return cls.outputs(value=value)


class BooleanPrimitive(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.boolean",
            display_name="Boolean",
            category=_PRIMITIVE_CATEGORY,
            inputs=(
                # No widget: the core.boolean type alone means "render a
                # toggle" (BooleanWidget exists only for custom labels).
                InputSpec(
                    "value",
                    BOOLEAN,
                    required=False,
                    default=False,
                    display_name="Value",
                ),
            ),
            outputs=(OutputSpec("value", BOOLEAN, known_value=OutputKnownValue(input="value")),),
            aliases=("PrimitiveBoolean",),
            search_terms=("PrimitiveBoolean", "toggle", "bool"),
        )

    @classmethod
    def execute(cls, *, value: bool = False) -> Mapping[str, object]:
        return cls.outputs(value=value)


def preview_text(source: object) -> str:
    """ComfyUI PreviewAny's formatting, torch-free: scalars verbatim,
    everything else pretty JSON, then str(), then a plain apology."""
    if source is None:
        return "None"
    if isinstance(source, str):
        return source
    if isinstance(source, (int, float, bool)):
        return str(source)
    try:
        return json.dumps(source, indent=4)
    except Exception:
        try:
            return str(source)
        except Exception:
            return "source exists, but could not be serialized."


class PreviewAsText(Node):
    """Preview any input value as text (the native PreviewAny cousin).

    The source socket is wildcard-typed, so anything links in; the value
    must be decodable in the engine process (see the module docstring for
    why this does not claim the "PreviewAny" alias)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_any",
            display_name="Preview as Text",
            category="utilities",
            description="Preview any input value as text.",
            inputs=(
                InputSpec(
                    "source",
                    TypeExpr.wildcard(),
                    required=False,
                    on_absent="accept",
                    display_name="Source",
                ),
            ),
            outputs=(OutputSpec("text", STRING),),
            output_node=True,
            search_terms=(
                "PreviewAny",
                "show output",
                "inspect",
                "debug",
                "print value",
                "show text",
            ),
        )

    @classmethod
    def execute(cls, *, source: object = None) -> Mapping[str, object]:
        return cls.outputs(text=preview_text(source))


PRIMITIVE_NODES: tuple[type[Node], ...] = (
    IntPrimitive,
    FloatPrimitive,
    StringPrimitive,
    StringMultilinePrimitive,
    BooleanPrimitive,
    PreviewAsText,
)
