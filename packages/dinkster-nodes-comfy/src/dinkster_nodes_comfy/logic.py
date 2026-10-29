"""Eager boolean nodes using Python truthiness."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)


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
        )

    @classmethod
    def execute(cls, *, values: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(BOOLEAN=any(values.values()))


LOGIC_NODES = (ComfyNotNode, ComfyAndNode, ComfyOrNode)
