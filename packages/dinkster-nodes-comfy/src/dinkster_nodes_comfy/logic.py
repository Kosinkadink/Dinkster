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
)

from .family_migrations import ordered_family_migration


class ComfyNotNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ComfyNotNode",
            display_name="Not",
            category="utilities/logic",
            description=(
                "Logical NOT operation. Returns true if the value is falsy. "
                "Uses Python's rules for truthiness."
            ),
            inputs=(InputSpec("value", TypeExpr.wildcard()),),
            outputs=(OutputSpec("BOOLEAN", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=not value)


class ComfyAndNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ComfyAndNode",
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
            replacements=(
                ordered_family_migration(
                    "ComfyAndNode",
                    "values",
                    "value",
                    ("BOOLEAN",),
                    historical_member_port="values.values",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, values: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=all(values.values()))


class ComfyOrNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ComfyOrNode",
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
            replacements=(
                ordered_family_migration(
                    "ComfyOrNode",
                    "values",
                    "value",
                    ("BOOLEAN",),
                    historical_member_port="values.values",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, values: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=any(values.values()))


class CustomCombo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CustomCombo",
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

    @classmethod
    def execute(cls, *, choice: str, index: int = 0, **kwargs: object) -> Mapping[str, object]:
        return cls.outputs(STRING=choice, INDEX=index)


LOGIC_NODES = (ComfyNotNode, ComfyAndNode, ComfyOrNode, CustomCombo)
