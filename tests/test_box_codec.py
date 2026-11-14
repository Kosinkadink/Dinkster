"""Structured boxes reject opaque payloads before lazy reconstruction."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    COMFY_ARRAY_TYPE,
    TypeRegistry,
    register_box_types,
)
from dinkster_assets.value import value_asset_references
from dinkster_values import default_encode
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


def _reconstruct(marker: str) -> list[object]:
    Path(marker).write_text("reconstructed", encoding="ascii")
    return []


class OpaqueBoxes:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _reconstruct, (str(self.marker),)


@pytest.mark.parametrize("type_id", (BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE))
def test_box_codec_refuses_pickle_at_real_boundary_admission(tmp_path: Path, type_id: str) -> None:
    registry = TypeRegistry()
    register_box_types(registry)
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
