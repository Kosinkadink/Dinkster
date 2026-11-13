from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import CustomWidgetDescriptor

STRING = TypeExpr.concrete("STRING")
INT = TypeExpr.concrete("INT")


def test_custom_widget_descriptor_roundtrips_verbatim() -> None:
    schema = NodeSchema(
        "test.custom_widget",
        inputs=(
            InputSpec(
                "value",
                STRING,
                widget=CustomWidgetDescriptor(
                    "example.gradient",
                    {"stops": ["#112233", "#abcdef"], "vertical": True},
                ),
            ),
        ),
        outputs=(OutputSpec("value", STRING),),
    )

    wire = schema_to_wire(schema)
    interface = cast("list[dict[str, object]]", wire["interface"])
    assert interface[0]["widget"] == {
        "type": "example.gradient",
        "stops": ["#112233", "#abcdef"],
        "vertical": True,
    }
    assert schema_from_wire(wire).inputs[0].widget == CustomWidgetDescriptor(
        "example.gradient",
        {"stops": ["#112233", "#abcdef"], "vertical": True},
    )


@pytest.fixture
def schema() -> NodeSchema:
    return NodeSchema(
        "test.composed",
        inputs=(InputSpec("entries", STRING),),
        outputs=(OutputSpec("value", INT),),
    )


@pytest.mark.parametrize("storage", [False, True])
@pytest.mark.parametrize("descriptors", [False, True])
def test_composed_roundtrip(schema: NodeSchema, storage: bool, descriptors: bool) -> None:
    schema = replace(
        schema,
        inputs=(replace(schema.inputs[0], accepts_storage=storage),),
        output_descriptors=(
            OutputDescriptorsSpec("entries", (OutputSpec("int", INT),), 4) if descriptors else None
        ),
    )
    wire: dict[str, Any] = schema_to_wire(schema)
    assert wire["schemaVersion"] == 1
    assert ("acceptsStorage" in wire["interface"][0]) == storage
    assert sum(entry["role"] == "outputDescriptors" for entry in wire["interface"]) == descriptors
    assert schema_from_wire(wire) == schema
    assert schema_to_wire(schema_from_wire(wire)) == wire
