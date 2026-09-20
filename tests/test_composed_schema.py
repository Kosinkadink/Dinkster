from __future__ import annotations

from dataclasses import replace
from typing import Any

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

STRING = TypeExpr.concrete("core.string")
INT = TypeExpr.concrete("core.int")


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
