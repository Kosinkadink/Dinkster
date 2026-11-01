"""Structured boxes reject opaque payloads before lazy reconstruction."""

from __future__ import annotations

import copy
import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    ARRAY_TYPE,
    BOUNDING_BOX_TYPE,
    COMFY_ARRAY_TYPE,
    TypeRegistry,
    register_box_types,
    register_inference_types,
    validate_bounding_box_encoded,
    validate_placement_array_encoded,
)
from dinkster_assets import AssetError, AssetRef, digest_bytes
from dinkster_assets.value import value_asset_references
from dinkster_compat_comfy.translate import CompatTranslation, comfy_value_type_id
from dinkster_nodes_image import register_image_types
from dinkster_values import EncodedPayload, default_encode, register_core_types
from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

BOX = {"x": -3, "y": 2.5, "width": 0, "height": 7.0}
ELEMENT = {
    "bbox": [1200, 900.5, -200, -100],
    "type": "text",
    "text": "Title",
    "desc": "Foreground",
    "color_palette": ["#AABBCC", "#123456"],
}


@pytest.mark.parametrize(
    "type_id,value",
    [
        (BOUNDING_BOX_TYPE, BOX),
        (BOUNDING_BOX_TYPE, []),
        (BOUNDING_BOX_TYPE, [BOX]),
        (BOUNDING_BOX_TYPE, [[], [BOX], []]),
        (
            BOUNDING_BOX_TYPE,
            {
                **BOX,
                "label": "person",
                "score": 0.75,
                "metadata": {
                    "name": "Subject",
                    "desc": "First",
                    "z_index": -2,
                    "content_rect": [-1, 2.5, 0, 4],
                    "type": "obj",
                    "text": "",
                    "palette": ["#FFFFFF"],
                },
            },
        ),
        (COMFY_ARRAY_TYPE, []),
        (COMFY_ARRAY_TYPE, [ELEMENT, BOX]),
        (COMFY_ARRAY_TYPE, [[], [ELEMENT, BOX], []]),
    ],
)
def test_box_contract_preserves_values_nesting_numeric_kinds_and_canonical_identity(
    type_id: str, value: object
) -> None:
    before = copy.deepcopy(value)
    registry = TypeRegistry()
    register_box_types(registry)
    spec = registry.spec(type_id)
    assert spec.declared_codec and spec.coerce is not None
    encoded = spec.encode(value)
    assert encoded == default_encode(value)
    assert spec.decode(encoded) == before
    assert json.dumps(spec.decode(encoded), sort_keys=True) == json.dumps(before, sort_keys=True)
    assert spec.validate_encoded is not None
    noncanonical = b"json:" + json.dumps(value, indent=2).encode()
    spec.validate_encoded(noncanonical, {"asset_refs": []})
    assert spec.decode(noncanonical) == before
    assert (
        registry.wrap(type_id, spec.decode(encoded)).fingerprint
        == registry.wrap(type_id, value).fingerprint
    )
    assert (
        value_asset_references({"boxes": registry.wrap(type_id, value)}, complete_registry=registry)
        == {}
    )
    assert value == before


@pytest.mark.parametrize("type_id", (BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE))
@pytest.mark.parametrize(
    "invalid",
    [
        {**BOX, "source": {"digest": "blake3:undeclared"}},
        {**BOX, "metadata": {"source": {"digest": "blake3:undeclared"}}},
        {**BOX, "metadata": {"z_index": True}},
        {**BOX, "metadata": {"content_rect": [0, 0, -1, 2]}},
        {**BOX, "metadata": {"palette": [object()]}},
        {**BOX, "x": True},
        {**BOX, "y": "3"},
        {**BOX, "height": -1},
        {**BOX, "score": 1.01},
        {**BOX, "score": -0.01},
        {**BOX, "score": float("nan")},
        {**BOX, "x": float("inf")},
        {**BOX, "label": None},
        {"x": 0, "y": 0, "width": 1},
        {"boxes": [BOX], "labels": ["person"], "scores": [0.75]},
    ],
)
def test_box_contract_refuses_undeclared_and_non_json_fields(type_id: str, invalid: object) -> None:
    registry = TypeRegistry()
    register_box_types(registry)
    value = invalid if type_id == BOUNDING_BOX_TYPE else [invalid]
    with pytest.raises(ValueError):
        registry.wrap(type_id, value)


@pytest.mark.parametrize(
    "invalid",
    [
        ELEMENT,
        [{**ELEMENT, "x": 0}],
        [{**ELEMENT, "bbox": [0, 1, 2]}],
        [{**ELEMENT, "bbox": [0, 1, 2, False]}],
        [{**ELEMENT, "metadata": {"source": "asset"}}],
        [{**ELEMENT, "color_palette": [None]}],
        [{**ELEMENT, "type": "unknown"}],
        [[ELEMENT], BOX],
        [BOX, [ELEMENT]],
    ],
)
def test_placement_contract_refuses_mixed_or_malformed_elements(invalid: object) -> None:
    registry = TypeRegistry()
    register_box_types(registry)
    with pytest.raises(ValueError):
        registry.wrap(COMFY_ARRAY_TYPE, invalid)
    with pytest.raises(ValueError):
        registry.wrap(BOUNDING_BOX_TYPE, [ELEMENT])


@pytest.mark.parametrize("type_id", (BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE))
def test_box_reference_census_preserves_auxiliary_roots_without_inventing_payload_sources(
    type_id: str,
) -> None:
    registry = TypeRegistry()
    register_box_types(registry)
    first = AssetRef(digest_bytes(b"first source"), "first", 12)
    second = AssetRef(digest_bytes(b"second"), "second", 6)
    box = {**BOX, "metadata": {"desc": first.digest, "text": "/not/a/source.png"}}
    obj = [[], [box]] if type_id == BOUNDING_BOX_TYPE else [[ELEMENT, box], []]
    value = registry.wrap(type_id, obj)
    assert value_asset_references({"boxes": value}, complete_registry=registry) == {}
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    payload = wire["payload"]
    assert isinstance(payload, dict)
    payload_index = payload["blob"]
    metadata_index = wire["metaBlob"]
    assert isinstance(payload_index, int) and isinstance(metadata_index, int)
    noncanonical = b"json:" + json.dumps(obj, indent=2).encode()
    blobs[payload_index] = noncanonical
    metadata = {**value.meta.entries, "asset_refs": [second.to_wire(), first.to_wire()]}
    blobs[metadata_index] = default_encode(metadata)
    received, _ = codec.decode(wire, blobs, [])
    assert received.resolve() == obj
    assert received.fingerprint == value.fingerprint
    assert isinstance(received.payload, EncodedPayload)
    assert received.payload.data == noncanonical
    assert value_asset_references({"boxes": received}, complete_registry=registry) == {
        first.digest: first,
        second.digest: second,
    }
    relay: list[bytes] = []
    relayed, _ = codec.encode(received, relay, [])
    relayed_payload = relayed["payload"]
    assert isinstance(relayed_payload, dict)
    relayed_payload_index = relayed_payload["blob"]
    relayed_metadata_index = relayed["metaBlob"]
    assert isinstance(relayed_payload_index, int) and isinstance(relayed_metadata_index, int)
    assert relay[relayed_payload_index] == noncanonical
    assert relay[relayed_metadata_index] == blobs[metadata_index]
    metadata["asset_refs"] = [first.to_wire(), {**first.to_wire(), "size": first.size + 1}]
    blobs[metadata_index] = default_encode(metadata)
    conflicting, _ = codec.decode(wire, blobs, [])
    with pytest.raises(AssetError, match="disagree"):
        value_asset_references({"boxes": conflicting}, complete_registry=registry)
    blobs[payload_index] = default_encode([{**BOX, "metadata": {"source": first.to_wire()}}])
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


def _reconstruct(marker: str) -> list[object]:
    Path(marker).write_text("reconstructed", encoding="ascii")
    return []


class OpaqueBoxes:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _reconstruct, (str(self.marker),)


def _compat_register(registry: TypeRegistry) -> None:
    translation = CompatTranslation()
    translation.opaque_types.update((BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE))
    translation.register_types(registry)


@pytest.mark.parametrize("order", tuple(itertools.permutations(("native", "image", "compat"))))
@pytest.mark.parametrize("type_id", (BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE))
def test_all_registration_orders_refuse_pickle_at_real_boundary_admission(
    tmp_path: Path, order: tuple[str, ...], type_id: str
) -> None:
    registry = TypeRegistry()
    registrations = {
        "native": register_inference_types,
        "image": register_image_types,
        "compat": _compat_register,
    }
    first = None
    for name in order:
        registrations[name](registry)
        spec = registry.spec(type_id)
        if first is None:
            first = spec
        assert spec is first and spec.declared_codec and spec.validate_encoded is not None
    spec = registry.spec(type_id)
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(registry.wrap(type_id, [BOX]), blobs, [])
    restored, _ = codec.decode(wire, blobs, [])
    assert restored.resolve() == [BOX]
    assert value_asset_references({"boxes": restored}, complete_registry=registry) == {}
    payload = wire["payload"]
    assert isinstance(payload, dict)
    marker = tmp_path / "marker.txt"
    marker.write_text("unchanged", encoding="ascii")
    malicious = b"pickle:" + pickle.dumps(OpaqueBoxes(marker), protocol=5)
    blobs[payload["blob"]] = malicious
    with pytest.raises(EncodedPayloadValidationError, match="JSON framing"):
        codec.decode(wire, blobs, [])
    assert marker.read_text() == "unchanged"
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError, match="JSON framing"):
        spec.validate_encoded(malicious, {})
    with pytest.raises(ValueError, match="JSON framing"):
        spec.decode(malicious)
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize(
    "payload",
    [b"json:{", b"json:[] trailing", b"json:\xff", b'json:{"x":1,"x":2}', b"[]"],
)
def test_public_admission_validators_refuse_malformed_json(payload: bytes) -> None:
    for validate in (validate_bounding_box_encoded, validate_placement_array_encoded):
        with pytest.raises(ValueError):
            validate(payload, {})


def test_global_array_and_preexisting_type_authority_are_unchanged() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    generic = registry.spec(ARRAY_TYPE)
    foreign = registry.register(BOUNDING_BOX_TYPE)
    register_box_types(registry)
    assert registry.spec(BOUNDING_BOX_TYPE) is foreign
    assert registry.spec(ARRAY_TYPE) is generic
    assert not foreign.asset_references_complete
    values: list[Any] = ["not a box", {"arbitrary": True}]
    assert generic.decode(generic.encode(values)) == values
    assert comfy_value_type_id("ARRAY") == COMFY_ARRAY_TYPE != ARRAY_TYPE
