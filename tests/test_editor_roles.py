from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    TypeExpr,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import CustomWidgetDescriptor


def test_editor_metadata_roundtrips_through_the_schema_wire() -> None:
    schema = NodeSchema(
        node_type="fixture.editor",
        display_name="Fixture editor",
        category="test",
        editor_role="fixture-editor",
        inputs=(
            InputSpec(
                "custom",
                TypeExpr.concrete("extension.value"),
                widget=CustomWidgetDescriptor(
                    "extension.widget",
                    cast("dict[str, Any]", {"nested": {"enabled": True, "values": [1, None]}}),
                ),
            ),
        ),
    )
    wire = schema_to_wire(schema)
    assert wire["editorRole"] == "fixture-editor"
    assert cast("list[dict[str, object]]", wire["interface"])[0]["widget"] == {
        "type": "extension.widget",
        "nested": {"enabled": True, "values": [1, None]},
    }
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(
        dataclasses.replace(
            schema,
            editor_role=None,
            inputs=(dataclasses.replace(schema.inputs[0], widget=None),),
        )
    )


@pytest.mark.parametrize(
    "widget_type",
    [
        "ASSET",
        "BOOLEAN",
        "COLOR",
        "COMBO",
        "COMPOSITOR",
        "CURVE",
        "MULTI_COMBO",
        "NUMBER",
        "REPRESENTATIONS",
        "SAVE_TARGET",
        "STRING",
    ],
)
def test_custom_widget_descriptors_cannot_impersonate_builtins(widget_type: str) -> None:
    with pytest.raises(ValueError, match="is reserved"):
        CustomWidgetDescriptor(widget_type)


def test_custom_widget_descriptor_freezes_nested_json() -> None:
    source = {"features": ["trim"], "options": {"snap": True}}
    descriptor = CustomWidgetDescriptor("VIDEO_EDIT", cast("dict[str, Any]", source))
    cast("list[str]", source["features"]).append("crop")
    cast("dict[str, bool]", source["options"])["snap"] = False
    assert descriptor.params == {"features": ("trim",), "options": {"snap": True}}
