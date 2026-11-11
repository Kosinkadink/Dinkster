"""Eager boolean nodes using Python truthiness."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    with_identity_replacement,
)


class ComfyNotNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_not_node",
            display_name="Not",
            category="utilities/logic",
            description=(
                "Logical NOT operation. Returns true if the value is falsy. "
                "Uses Python's rules for truthiness."
            ),
            inputs=(InputSpec("value", TypeExpr.wildcard()),),
            outputs=(OutputSpec("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),),
        )
        return with_identity_replacement(schema, "ComfyNotNode")

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=not value)


class ComfyAndNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_and_node",
            display_name="And",
            category="utilities/logic",
            description=(
                "Logical AND operation. Returns true if all of the values are truthy. "
                "Uses Python's rules for truthiness."
            ),
            input_families=(
                InputFamilySpec(
                    "values",
                    TypeExpr.wildcard(),
                    min_members=1,
                    max_members=10,
                    member_prefix="value",
                    optional_after_minimum=True,
                ),
            ),
            outputs=(OutputSpec("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),),
        )
        return with_identity_replacement(schema, "ComfyAndNode")

    @classmethod
    def execute(cls, *, values: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=all(values.values()))


class ComfyOrNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.comfy_or_node",
            display_name="Or",
            category="utilities/logic",
            description=(
                "Logical OR operation. Returns true if any of the values are truthy. "
                "Uses Python's rules for truthiness."
            ),
            input_families=(
                InputFamilySpec(
                    "values",
                    TypeExpr.wildcard(),
                    min_members=1,
                    max_members=10,
                    member_prefix="value",
                    optional_after_minimum=True,
                ),
            ),
            outputs=(OutputSpec("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),),
        )
        return with_identity_replacement(schema, "ComfyOrNode")

    @classmethod
    def execute(cls, *, values: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=any(values.values()))


class CustomCombo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.custom_combo",
            display_name="Custom Combo",
            category="utilities",
            inputs=(
                InputSpec(
                    "choice",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(options=(), editable_options=True),
                ),
            ),
            outputs=(
                OutputSpec("STRING", TypeExpr.concrete(CORE_STRING)),
                OutputSpec("INDEX", TypeExpr.concrete(CORE_INT)),
            ),
            additional_inputs=TypeExpr.wildcard(),
        )
        return with_identity_replacement(schema, "CustomCombo")

    @classmethod
    def execute(cls, *, choice: str, index: int = 0, **kwargs: object) -> Mapping[str, object]:
        return cls.outputs(STRING=choice, INDEX=index)
