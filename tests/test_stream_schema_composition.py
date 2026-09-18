from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from dinkster_protocol import Invocation
from dinkster_schema import (
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputFamilySpec,
    OutputSpec,
    SlotVariant,
    TypeExpr,
    combo_type_mismatch_is_error,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_schema.wire import SCHEMA_WIRE_SERVE_VERSIONS, SCHEMA_WIRE_VERSION
from dinkster_values import TypeRegistry
from dinkster_workers.boundary import ValueCodec, decode_invocation, encode_invocation
from dinkster_workers.catalog import worker_declarations

IMAGE = TypeExpr.concrete("comfy.IMAGE")
STRING = TypeExpr.concrete("core.string")
STREAM = TypeExpr.stream_of(IMAGE)


def test_worker_catalog_retains_streams_independently_of_public_default():
    schema = NodeSchema("test.source", outputs=(OutputSpec("image", STREAM),))
    worker = SimpleNamespace(
        schemas={schema.node_type: schema},
        body_arms={},
        combo_choices={},
        lazy_choice_ids=(),
        compat_skips={},
        extension_contributions=(),
    )
    declaration = worker_declarations(worker)["schemas"][schema.node_type]
    assert declaration["schemaVersion"] == 45
    assert schema_from_wire(declaration) == schema


def test_worker_invocation_retains_stream_schema_independently_of_public_default():
    schema = NodeSchema("test.stream-invocation", outputs=(OutputSpec("image", STREAM),))
    invocation = Invocation(
        invocation_id="inv-stream",
        node_id="source",
        node_type=schema.node_type,
        inputs={},
        effective_schema=schema,
        output_members=(("image", ()),),
    )
    header, blobs, segments, _stats = encode_invocation(ValueCodec(TypeRegistry()), invocation)
    try:
        effective = header["effectiveSchema"]
        assert isinstance(effective, dict)
        assert effective["schemaVersion"] == 45
        assert (
            decode_invocation(ValueCodec(TypeRegistry()), header, blobs, []).effective_schema
            == schema
        )
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


def test_stream_output_variable_is_bound_by_dynamic_slot():
    variable = TypeExpr.variable("T", ("comfy.IMAGE",))
    schema = NodeSchema(
        "test.stream-variable",
        slots=(
            DynamicSlotSpec(
                "source",
                variants=(SlotVariant("image", IMAGE),),
                type_template_id="T",
            ),
        ),
        outputs=(OutputSpec("stream", TypeExpr.stream_of(variable)),),
    )
    effective = elaborate(schema, (), slot_variants={"source": "image"})
    assert effective.outputs[0].type == STREAM


def test_stream_combo_mismatch_keeps_hard_combo_boundary():
    expected = TypeExpr.stream_of(TypeExpr.concrete("core.combo"))
    assert combo_type_mismatch_is_error("stream<core.string>", expected)


def test_storage_descriptors_policies_and_streams_coexist():
    schema = NodeSchema(
        "test.composed_stream",
        inputs=(
            InputSpec("entries", STRING),
            InputSpec("image", IMAGE, accepts_storage=True, alpha_policy="require"),
        ),
        outputs=(OutputSpec("image", STREAM, display_name="Image", alpha_policy="drop"),),
        output_descriptors=OutputDescriptorsSpec(
            "entries",
            (OutputSpec("frames", IMAGE, mask_polarity="transparency", mask_semantic="alpha"),),
            4,
        ),
        chunk_safe=(("image",), ("image",)),
    )
    assert SCHEMA_WIRE_VERSION == 40
    assert {39, 40, 41, 42} <= set(SCHEMA_WIRE_SERVE_VERSIONS)
    with pytest.raises(ValueError, match="stream type.*41"):
        schema_to_wire(schema)
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=41)
    assert schema_from_wire(wire) == schema
    assert wire["interface"][1]["acceptsStorage"] is True
    assert wire["interface"][1]["alphaPolicy"] == "require"
    assert wire["interface"][2]["displayName"] == "Image"
    assert wire["interface"][3]["choices"][0]["maskPolarity"] == "transparency"
    assert schema_signature(schema) != schema_signature(replace(schema, chunk_safe=None))
    for version in (39, 40):
        refusal = "media policies.*40" if version == 39 else "stream type.*41"
        with pytest.raises(ValueError, match=refusal):
            schema_to_wire(schema, wire_version=version)
        downgraded = {**wire, "schemaVersion": version}
        with pytest.raises(ValueError, match=refusal):
            schema_from_wire(downgraded)
    invalid = deepcopy(wire)
    invalid["interface"][3]["choices"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="output descriptor choice"):
        schema_from_wire(invalid)


def nested_schema(location, leaf):
    if location == "family":
        return NodeSchema("test.nested", input_families=(InputFamilySpec("group", (leaf,)),))
    if location == "combo":
        return NodeSchema(
            "test.nested",
            combos=(DynamicComboSpec("choice", (DynamicComboOption("a", (leaf,)),)),),
        )
    slot = (
        DynamicSlotSpec("slot", slot_type=IMAGE, inputs=(leaf,))
        if location == "slot"
        else DynamicSlotSpec("slot", variants=(SlotVariant("a", IMAGE, (leaf,)),))
    )
    return NodeSchema("test.nested", slots=(slot,))


@pytest.mark.parametrize("location", ["family", "combo", "slot", "variant"])
def test_recursive_stream_acceptance_retains_storage_and_policy(location):
    leaf = InputSpec(
        "image", IMAGE, accepts_storage=True, accepts_stream=True, alpha_policy="require"
    )
    schema = nested_schema(location, leaf)
    wire = schema_to_wire(schema, wire_version=41)
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) != schema_signature(
        nested_schema(location, replace(leaf, accepts_stream=False))
    )
    with pytest.raises(ValueError, match="acceptsStream.*41"):
        schema_to_wire(schema, wire_version=40)
    with pytest.raises(ValueError, match="acceptsStream.*41"):
        schema_from_wire({**wire, "schemaVersion": 40})
    if location == "family":
        effective = elaborate(schema, {"group.member.image": 1})
        assert effective.inputs[0].accepts_stream
        assert effective.inputs[0].accepts_storage
        assert effective.inputs[0].alpha_policy == "require"


def type_location_schema(location, expr):
    if location == "input":
        return NodeSchema("test.types", inputs=(InputSpec("image", expr),))
    if location in ("family", "combo", "slot", "variant"):
        return nested_schema(location, InputSpec("image", expr))
    if location == "slot_type":
        return NodeSchema("test.types", slots=(DynamicSlotSpec("slot", slot_type=expr),))
    if location == "variant_type":
        return NodeSchema(
            "test.types", slots=(DynamicSlotSpec("slot", variants=(SlotVariant("a", expr),)),)
        )
    if location == "output_family":
        return NodeSchema("test.types", output_families=(OutputFamilySpec("images", expr),))
    if location == "descriptor":
        return NodeSchema(
            "test.types",
            inputs=(InputSpec("entries", STRING),),
            output_descriptors=OutputDescriptorsSpec("entries", (OutputSpec("image", expr),), 4),
        )
    return NodeSchema("test.types", outputs=(OutputSpec("image", expr),))


@pytest.mark.parametrize(
    "location",
    [
        "input",
        "output",
        "family",
        "combo",
        "slot",
        "variant",
        "slot_type",
        "variant_type",
        "output_family",
    ],
)
@pytest.mark.parametrize("expr", [STREAM, TypeExpr.list_of(STREAM), TypeExpr.asset_of(STREAM)])
def test_every_type_path_negotiates_streams(location, expr):
    schema = type_location_schema(location, expr)
    wire = schema_to_wire(schema, wire_version=41)
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) != schema_signature(type_location_schema(location, IMAGE))
    for version in (39, 40):
        with pytest.raises(ValueError, match="stream type.*41"):
            schema_to_wire(schema, wire_version=version)
        with pytest.raises(ValueError, match="stream type.*41"):
            schema_from_wire({**wire, "schemaVersion": version})


@pytest.mark.parametrize("expr", [STREAM, TypeExpr.list_of(STREAM), TypeExpr.asset_of(STREAM)])
def test_descriptor_types_remain_concrete_and_refuse_stream_downgrade(expr):
    from dinkster_schema.wire import type_expr_to_wire

    with pytest.raises(ValueError, match="concrete runtime types"):
        type_location_schema("descriptor", expr)
    wire: dict[str, Any] = schema_to_wire(
        type_location_schema("descriptor", IMAGE), wire_version=41
    )
    wire["interface"][1]["choices"][0]["type"] = type_expr_to_wire(expr, wire_version=41)
    for version in (39, 40):
        with pytest.raises(ValueError, match="stream type.*41"):
            schema_from_wire({**wire, "schemaVersion": version})
    with pytest.raises(ValueError, match="concrete runtime types"):
        schema_from_wire(wire)


def test_scoped_chunk_contract_and_ordinary_unknown_metadata():
    schema = NodeSchema(
        "test.scoped",
        inputs=(InputSpec("image", IMAGE, accepts_stream=True, accepts_storage=True),),
        outputs=(OutputSpec("image", IMAGE),),
        combos=(
            DynamicComboSpec(
                "operation", (DynamicComboOption("map"), DynamicComboOption("reverse"))
            ),
        ),
        chunk_safe=(("image",), ("image",)),
        chunk_safe_applies={"operation": ("map",)},
    )
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=41)
    wire["interface"][0]["unknownMetadata"] = {"arbitrary": True}
    assert schema_from_wire(wire) == schema
    assert elaborate(schema, ("image",), slot_variants={"operation": "map"}).chunk_safe
    reverse = elaborate(schema, ("image",), slot_variants={"operation": "reverse"})
    assert reverse.chunk_safe is None
    assert not reverse.inputs[0].accepts_stream
    assert reverse.inputs[0].accepts_storage
    for key, value in (
        ("unknown", True),
        ("inputs", [1]),
        ("outputs", []),
        ("applies", {"missing": ["map"]}),
    ):
        malformed = deepcopy(wire)
        malformed["chunkSafe"][key] = value
        with pytest.raises(ValueError):
            schema_from_wire(malformed)


@pytest.mark.parametrize("field", ["acceptsStorage", "acceptsStream"])
def test_input_contracts_are_boolean_and_emit_only_true(field):
    schema = NodeSchema("test.flags", inputs=(InputSpec("image", IMAGE),))
    wire: dict[str, Any] = schema_to_wire(schema, wire_version=41)
    assert field not in wire["interface"][0]
    wire["interface"][0][field] = False
    assert schema_from_wire(wire) == schema
    normalized: dict[str, Any] = schema_to_wire(schema_from_wire(wire), wire_version=41)
    assert field not in normalized["interface"][0]
    for value in (None, 0, 1, "true", [], {}):
        wire["interface"][0][field] = value
        with pytest.raises(ValueError, match="must be a bool"):
            schema_from_wire(wire)
