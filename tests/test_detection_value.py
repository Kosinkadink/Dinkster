from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from dinkster_api.v1 import (
    DETECTION_TYPE,
    REGION_TYPE,
    Detection,
    Region,
    TypeRegistry,
    coerce_detection,
    coerce_region,
    decode_detection,
    decode_region,
    detection_meta,
    encode_detection,
    encode_region,
    region_meta,
    validate_detection_encoded,
    validate_region_encoded,
)


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    registry.register(
        REGION_TYPE,
        encode=encode_region,
        decode=decode_region,
        coerce=coerce_region,
        meta=region_meta,
        validate_encoded=validate_region_encoded,
        asset_references_complete=True,
    )
    registry.register(
        DETECTION_TYPE,
        encode=encode_detection,
        decode=decode_detection,
        coerce=coerce_detection,
        meta=detection_meta,
        validate_encoded=validate_detection_encoded,
        asset_references_complete=True,
    )
    return registry


def _mask(height: int = 3, width: int = 4) -> np.ndarray:
    values = np.arange(height * width, dtype=np.float32)
    return (values / max(1.0, float(values.size - 1))).reshape(height, width)


def test_spatial_codecs_survive_encoded_admission() -> None:
    from dinkster_assets import AssetRef, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_values import default_encode
    from dinkster_workers.boundary import EncodedPayloadValidationError, ValueCodec

    registry = _registry()
    reference = AssetRef(digest_bytes(b"source"), "source.bin", 6)
    for type_id, item in (
        ("REGION", Region(-2, 1.5, 0, 4)),
        (DETECTION_TYPE, Detection("cat", 0.625, Region(-2, 1, 0, 4))),
        (DETECTION_TYPE, Detection("cat", 0.625, Region(-2, 1, 0, 4), _mask())),
    ):
        wrapped = registry.wrap(type_id, item)
        assert value_asset_references({"item": wrapped}, complete_registry=registry) == {}
        codec = ValueCodec(registry)
        blobs, segments = [], []
        wire, _ = codec.encode(wrapped, blobs, segments)
        blobs[cast(int, wire["metaBlob"])] = default_encode(
            {**wrapped.meta.entries, "asset_refs": [reference.to_wire()]}
        )
        received, _ = codec.decode(wire, blobs, [])
        assert received.resolve() == item
        assert received.fingerprint == wrapped.fingerprint
        if type_id == "REGION":
            assert registry.spec(type_id).encode(received.resolve()) == (
                b'{"height":4,"width":0,"x":-2,"y":1.5}'
            )
        assert value_asset_references({"item": received}, complete_registry=registry) == {
            reference.digest: reference
        }
        payload = cast("dict[str, object]", wire["payload"])
        blobs[cast(int, payload["blob"])] = b"invalid spatial payload"
        with pytest.raises(EncodedPayloadValidationError):
            codec.decode(wire, blobs, [])
