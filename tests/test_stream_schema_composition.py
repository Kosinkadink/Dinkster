from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dinkster_protocol import Invocation
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import TypeRegistry
from dinkster_workers.boundary import ValueCodec, decode_invocation, encode_invocation
from dinkster_workers.catalog import worker_declarations

IMAGE = TypeExpr.concrete("comfy.IMAGE")
STRING = TypeExpr.concrete("core.string")
STREAM = TypeExpr.stream_of(IMAGE)


def test_worker_catalog_retains_stream_schema() -> None:
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
    assert declaration["schemaVersion"] == 1
    assert schema_from_wire(declaration) == schema


def test_worker_invocation_retains_stream_schema() -> None:
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
        assert effective["schemaVersion"] == 1
        assert (
            decode_invocation(ValueCodec(TypeRegistry()), header, blobs, []).effective_schema
            == schema
        )
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


def test_storage_descriptors_policies_and_streams_coexist() -> None:
    schema = NodeSchema(
        "test.composed-stream",
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
    wire: dict[str, Any] = schema_to_wire(schema)
    assert schema_from_wire(wire) == schema
    assert wire["interface"][1]["acceptsStorage"] is True
    assert wire["interface"][1]["alphaPolicy"] == "require"
    assert wire["interface"][2]["displayName"] == "Image"
    assert wire["interface"][3]["choices"][0]["maskPolarity"] == "transparency"
    assert schema_signature(schema) != schema_signature(replace(schema, chunk_safe=None))
    invalid = deepcopy(wire)
    invalid["interface"][3]["choices"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="output descriptor choice"):
        schema_from_wire(invalid)


@pytest.mark.parametrize("field", ["acceptsStorage", "acceptsStream"])
def test_input_contracts_are_boolean_and_emit_only_true(field: str) -> None:
    schema = NodeSchema("test.flags", inputs=(InputSpec("image", IMAGE),))
    wire: dict[str, Any] = schema_to_wire(schema)
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert field not in interface[0]
    interface[0][field] = False
    assert schema_from_wire(wire) == schema
    round_trip = cast("list[dict[str, Any]]", schema_to_wire(schema_from_wire(wire))["interface"])
    assert field not in round_trip[0]
    for value in (None, 0, 1, "true", [], {}):
        interface[0][field] = value
        with pytest.raises(ValueError, match="must be a bool"):
            schema_from_wire(wire)
