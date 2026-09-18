from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from dinkster_nodes_foundation import MathExpressions
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
@pytest.mark.parametrize("version", [39, 40])
def test_composed_wire39_roundtrip(
    schema: NodeSchema, storage: bool, descriptors: bool, version: int
) -> None:
    schema = replace(
        schema,
        inputs=(replace(schema.inputs[0], accepts_storage=storage),),
        output_descriptors=(
            OutputDescriptorsSpec("entries", (OutputSpec("int", INT),), 4) if descriptors else None
        ),
    )
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=version)
    assert ("acceptsStorage" in wire["interface"][0]) == storage
    assert sum(entry["role"] == "outputDescriptors" for entry in wire["interface"]) == descriptors
    assert schema_from_wire(wire) == schema
    assert schema_to_wire(schema_from_wire(wire), wire_version=version) == wire


@pytest.mark.parametrize("version", range(15, 39))
def test_legacy_downgrade(schema: NodeSchema, version: int) -> None:
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=version)
    assert "acceptsStorage" not in wire["interface"][0]
    assert all(entry["role"] != "outputDescriptors" for entry in wire["interface"])
    if version >= 21:
        assert schema_from_wire(wire) == schema
    with pytest.raises(ValueError, match="acceptsStorage.*39"):
        schema_to_wire(
            replace(schema, inputs=(replace(schema.inputs[0], accepts_storage=True),)),
            wire_version=version,
        )
    with pytest.raises(ValueError, match="output descriptors.*39"):
        schema_to_wire(
            replace(
                schema,
                output_descriptors=OutputDescriptorsSpec("entries", (OutputSpec("int", INT),), 4),
            ),
            wire_version=version,
        )


def test_legacy_signatures_are_byte_identical(schema: NodeSchema) -> None:
    assert schema_signature(schema) == "8ee629f737a8af74c962835ade7c4f28a82b111c"
    assert schema_signature(MathExpressions.schema()) == "4abd89eb944cbd9995ee4ee5527975619cb51252"
    explicit_default = replace(schema, inputs=(replace(schema.inputs[0], accepts_storage=False),))
    assert schema_signature(explicit_default) == schema_signature(schema)
    assert schema_signature(
        replace(schema, inputs=(replace(schema.inputs[0], accepts_storage=True),))
    ) != schema_signature(schema)


@pytest.mark.parametrize("version", [39, 40])
def test_false_storage_normalizes_only_at_input_field(schema: NodeSchema, version: int) -> None:
    schema = replace(schema, inputs=(replace(schema.inputs[0], default={"acceptsStorage": False}),))
    canonical: dict[str, Any] = schema_to_wire(schema, wire_version=version)
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=version)
    wire["interface"][0]["acceptsStorage"] = False
    decoded = schema_from_wire(wire)
    assert decoded == schema
    assert schema_to_wire(decoded, wire_version=version) == canonical
    assert decoded.inputs[0].default == {"acceptsStorage": False}
    assert schema_signature(decoded) == schema_signature(schema)
    assert schema_signature(decoded) != schema_signature(
        replace(schema, inputs=(replace(schema.inputs[0], default={}),))
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_storage_requires_boolean(schema: NodeSchema, value: Any) -> None:
    with pytest.raises(ValueError, match="accepts_storage must be a bool"):
        replace(schema.inputs[0], accepts_storage=value)
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=39)
    wire["interface"][0]["acceptsStorage"] = value
    with pytest.raises(ValueError, match="input.acceptsStorage must be a bool"):
        schema_from_wire(wire)


@pytest.mark.parametrize("version", range(21, 39))
@pytest.mark.parametrize("value", [False, True])
def test_storage_presence_rejected_on_older_wires(
    schema: NodeSchema, version: int, value: bool
) -> None:
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=version)
    wire["interface"][0]["acceptsStorage"] = value
    with pytest.raises(ValueError, match="acceptsStorage requires schema wire 39"):
        schema_from_wire(wire)


@pytest.mark.parametrize("location", ["family", "combo", "slot", "variant"])
def test_recursive_storage_roundtrip_identity_and_downgrade(location: str) -> None:
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

    schema = nested(True)
    assert schema_from_wire(schema_to_wire(schema, wire_version=39)) == schema
    assert schema_signature(schema) != schema_signature(nested(False))
    with pytest.raises(ValueError, match="acceptsStorage.*39"):
        schema_to_wire(schema, wire_version=38)
    if location == "family":
        assert elaborate(schema, {"group.member.data": 1}).inputs[0].accepts_storage


@pytest.fixture
def media_schema(schema: NodeSchema) -> NodeSchema:
    return replace(
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


def test_wire40_media_policies_compose_with_storage_and_descriptors(
    media_schema: NodeSchema,
) -> None:
    wire: dict[str, Any] = schema_to_wire(media_schema, wire_version=40)
    assert wire["interface"][1]["acceptsStorage"] is True
    assert wire["interface"][1]["alphaPolicy"] == "require"
    assert wire["interface"][2]["alphaPolicy"] == "drop"
    assert wire["interface"][3]["choices"][0]["maskPolarity"] == "transparency"
    assert wire["interface"][3]["choices"][0]["maskSemantic"] == "alpha"
    assert schema_from_wire(wire) == media_schema
    assert schema_to_wire(schema_from_wire(wire), wire_version=40) == wire
    effective = elaborate(
        media_schema, {"entries": '{"entries":[{"id":"a","name":"Mask","type":"mask"}]}'}
    )
    assert effective.outputs[1].mask_polarity == "transparency"
    assert effective.outputs[1].mask_semantic == "alpha"
    assert schema_from_wire(schema_to_wire(effective, wire_version=40)) == effective

    with pytest.raises(ValueError, match="media policies.*40"):
        schema_to_wire(media_schema, wire_version=39)
    with pytest.raises(ValueError, match="media policies.*40"):
        schema_to_wire(effective, wire_version=39)


@pytest.mark.parametrize("location", ["input", "output", "choice"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alphaPolicy", "preserve"),
        ("alphaPolicy", "require"),
        ("alphaPolicy", "create_if_missing"),
        ("alphaPolicy", "drop"),
        ("maskPolarity", "coverage"),
        ("maskPolarity", "transparency"),
        ("maskSemantic", "alpha"),
        ("maskSemantic", "selection"),
        ("maskSemantic", "other"),
    ],
)
def test_media_policy_identity_and_version_gate(
    media_schema: NodeSchema, location: str, field: str, value: str
) -> None:
    wire: dict[str, Any] = schema_to_wire(media_schema, wire_version=40)
    for item in wire["interface"] + wire["interface"][3]["choices"]:
        for policy in ("alphaPolicy", "maskPolarity", "maskSemantic"):
            item.pop(policy, None)
    wire["schemaVersion"] = 39
    baseline = schema_signature(schema_from_wire(wire))
    entry = {
        "input": wire["interface"][1],
        "output": wire["interface"][2],
        "choice": wire["interface"][3]["choices"][0],
    }[location]
    entry[field] = value
    with pytest.raises(ValueError, match="media policies require schema wire 40"):
        schema_from_wire(wire)
    wire["schemaVersion"] = 40
    parsed = schema_from_wire(wire)
    assert (schema_signature(parsed) == baseline) == (value == "preserve")
    encoded: dict[str, Any] = schema_to_wire(parsed, wire_version=40)
    encoded_entry = {
        "input": encoded["interface"][1],
        "output": encoded["interface"][2],
        "choice": encoded["interface"][3]["choices"][0],
    }[location]
    if value == "preserve":
        assert field not in encoded_entry
        assert (
            schema_signature(schema_from_wire(schema_to_wire(parsed, wire_version=39))) == baseline
        )
    else:
        assert encoded_entry[field] == value
        with pytest.raises(ValueError, match="media policies.*40"):
            schema_to_wire(parsed, wire_version=39)
    for malformed in (None, True, 1, "invalid", [], {}):
        entry[field] = malformed
        with pytest.raises(ValueError):
            schema_from_wire(wire)


@pytest.mark.parametrize("location", ["input", "output", "family", "combo", "slot", "variant"])
@pytest.mark.parametrize("version", [15, 38, 39])
@pytest.mark.parametrize(
    "policy",
    [
        {"alpha_policy": "preserve"},
        {"alpha_policy": "require"},
        {"alpha_policy": "create_if_missing"},
        {"alpha_policy": "drop"},
        {"mask_polarity": "coverage"},
        {"mask_polarity": "transparency"},
        {"mask_semantic": "alpha"},
        {"mask_semantic": "selection"},
        {"mask_semantic": "other"},
    ],
)
def test_recursive_media_policy_downgrade(
    location: str, version: int, policy: dict[str, Any]
) -> None:
    leaf = InputSpec("media", TypeExpr.concrete("comfy.IMAGE"), **policy)
    shapes: dict[str, dict[str, Any]] = {
        "input": {"inputs": (leaf,)},
        "output": {"outputs": (OutputSpec("media", leaf.type, **policy),)},
        "family": {"input_families": (InputFamilySpec("group", (leaf,)),)},
        "combo": {"combos": (DynamicComboSpec("choice", (DynamicComboOption("a", (leaf,)),)),)},
        "slot": {"slots": (DynamicSlotSpec("slot", slot_type=INT, inputs=(leaf,)),)},
        "variant": {
            "slots": (DynamicSlotSpec("slot", variants=(SlotVariant("a", INT, (leaf,)),)),)
        },
    }
    schema = NodeSchema("test.media-policy", **shapes[location])
    assert schema_from_wire(schema_to_wire(schema, wire_version=40)) == schema
    if policy == {"alpha_policy": "preserve"}:
        schema_to_wire(schema, wire_version=version)
    else:
        with pytest.raises(ValueError, match="media policies.*40"):
            schema_to_wire(schema, wire_version=version)
