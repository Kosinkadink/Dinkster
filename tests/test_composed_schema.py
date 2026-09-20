from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from dinkster_schema import (
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputSpec,
    SlotVariant,
    TypeExpr,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import CustomWidgetDescriptor, JsonValue

STRING = TypeExpr.concrete("core.string")
INT = TypeExpr.concrete("core.int")


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


def test_custom_widget_descriptor_reserves_wire_type_field() -> None:
    with pytest.raises(ValueError, match="params must not contain 'type'"):
        CustomWidgetDescriptor("example.gradient", {"type": "other.widget"})


def test_custom_widget_descriptor_copies_nested_json_and_rejects_nonfinite_numbers() -> None:
    stops: list[JsonValue] = ["#112233"]
    descriptor = CustomWidgetDescriptor("example.gradient", {"stops": stops})
    stops.append("#abcdef")
    assert descriptor.params == {"stops": ("#112233",)}

    with pytest.raises(ValueError, match="must be JSON-safe"):
        CustomWidgetDescriptor("example.gradient", {"minimum": float("nan")})


def test_schema_to_wire_names_unknown_widget_descriptor() -> None:
    descriptor = object()
    input_spec = InputSpec("value", STRING)
    object.__setattr__(input_spec, "widget", descriptor)
    schema = NodeSchema(
        "test.unknown_widget",
        inputs=(input_spec,),
        outputs=(OutputSpec("value", STRING),),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"node 'test\.unknown_widget', input 'value': unsupported widget descriptor "
            r"<object object"
        ),
    ):
        schema_to_wire(schema)


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


def test_false_storage_normalizes_only_at_input_field(schema: NodeSchema) -> None:
    schema = replace(schema, inputs=(replace(schema.inputs[0], default={"acceptsStorage": False}),))
    canonical: dict[str, Any] = schema_to_wire(schema)
    wire: dict[str, Any] = schema_to_wire(schema)
    wire["interface"][0]["acceptsStorage"] = False
    decoded = schema_from_wire(wire)
    assert decoded == schema
    assert schema_to_wire(decoded) == canonical
    assert decoded.inputs[0].default == {"acceptsStorage": False}


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_storage_requires_boolean(schema: NodeSchema, value: Any) -> None:
    with pytest.raises(ValueError, match="accepts_storage must be a bool"):
        replace(schema.inputs[0], accepts_storage=value)
    wire: dict[str, Any] = schema_to_wire(schema)
    wire["interface"][0]["acceptsStorage"] = value
    with pytest.raises(ValueError, match="input.acceptsStorage must be a bool"):
        schema_from_wire(wire)


@pytest.mark.parametrize("location", ["family", "combo", "slot", "variant"])
def test_recursive_storage_roundtrip(location: str) -> None:
    def nested(storage: bool) -> NodeSchema:
        leaf = InputSpec("data", INT, accepts_storage=storage)
        if location == "family":
            return NodeSchema("test.nested", input_families=(InputFamilySpec("group", (leaf,)),))
        if location == "combo":
            return NodeSchema(
                "test.nested",
                combos=(DynamicComboSpec("choice", (DynamicComboOption("a", (leaf,)),)),),
            )
        if location == "slot":
            slot = DynamicSlotSpec("slot", slot_type=INT, inputs=(leaf,))
        else:
            slot = DynamicSlotSpec("slot", variants=(SlotVariant("a", INT, (leaf,)),))
        return NodeSchema("test.nested", slots=(slot,))

    composed = nested(True)
    assert schema_from_wire(schema_to_wire(composed)) == composed
    assert schema_signature(composed) != schema_signature(nested(False))
    if location == "family":
        assert elaborate(composed, {"group.member.data": 1}).inputs[0].accepts_storage


def test_media_policies_compose_with_storage_and_descriptors(schema: NodeSchema) -> None:
    media_schema = replace(
        schema,
        inputs=(
            schema.inputs[0],
            InputSpec(
                "image",
                TypeExpr.concrete("comfy.IMAGE"),
                accepts_storage=True,
                alpha_policy="require",
            ),
        ),
        outputs=(OutputSpec("image", TypeExpr.concrete("comfy.IMAGE"), alpha_policy="drop"),),
        output_descriptors=OutputDescriptorsSpec(
            "entries",
            (
                OutputSpec(
                    "mask",
                    TypeExpr.concrete("comfy.MASK"),
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
            ),
            4,
        ),
    )
    wire: dict[str, Any] = schema_to_wire(media_schema)
    assert wire["interface"][1]["acceptsStorage"] is True
    assert wire["interface"][1]["alphaPolicy"] == "require"
    assert wire["interface"][2]["alphaPolicy"] == "drop"
    assert wire["interface"][3]["choices"][0]["maskPolarity"] == "transparency"
    assert wire["interface"][3]["choices"][0]["maskSemantic"] == "alpha"
    assert schema_from_wire(wire) == media_schema
    effective = elaborate(
        media_schema, {"entries": '{"entries":[{"id":"a","name":"Mask","type":"mask"}]}'}
    )
    assert effective.outputs[1].mask_polarity == "transparency"
    assert effective.outputs[1].mask_semantic == "alpha"
    assert schema_from_wire(schema_to_wire(effective)) == effective
