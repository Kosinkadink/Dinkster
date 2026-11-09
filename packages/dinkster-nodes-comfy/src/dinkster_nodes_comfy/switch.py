"""Lazy, type-matched selection with ComfyUI's public switch interface."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
)


class ComfySwitchNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        value_type = TypeExpr.variable("switch")
        return NodeSchema(
            node_type="ComfySwitchNode",
            display_name="If/Else Switch",
            category="utilities/logic",
            inputs=(
                InputSpec("switch", TypeExpr.concrete(CORE_BOOLEAN)),
                InputSpec("on_false", value_type, required=False, lazy=True, on_absent="skip"),
                InputSpec("on_true", value_type, required=False, lazy=True, on_absent="skip"),
            ),
            outputs=(OutputSpec("output", value_type, optional=True),),
            selector=SelectorSpec("switch", {"false": "on_false", "true": "on_true"}),
        )

    @classmethod
    def check_lazy_status(
        cls, *, switch: bool, on_false: object = ABSENT, on_true: object = ABSENT
    ) -> tuple[str, ...]:
        selected = "on_true" if switch else "on_false"
        return (selected,) if (on_true if switch else on_false) is None else ()

    @classmethod
    def execute(
        cls, *, switch: bool, on_false: object = ABSENT, on_true: object = ABSENT
    ) -> Mapping[str, object]:
        value = on_true if switch else on_false
        return cls.outputs(output=ABSENT if value is None else value)
