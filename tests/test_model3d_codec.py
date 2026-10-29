"""Torch-free proofs for the dinkster.model3d GLB byte contract."""

from __future__ import annotations

import io
import struct

import pytest
from dinkster_nodes_media_io import MODEL3D_TYPE, register_media_types
from dinkster_values import (
    MODEL3D_FILE_DECODER_ID,
    TypeRegistry,
    decode_model3d,
    decode_model3d_file,
    encode_model3d,
    model3d_fingerprint,
    model3d_format,
    model3d_meta,
    register_model3d_type,
    render_model3d_original,
    validate_model3d_encoded,
)


def glb_bytes(document: bytes = b'{"asset":{"version":"2.0"}}') -> bytes:
    padded = document + b" " * (-len(document) % 4)
    chunk = struct.pack("<I4s", len(padded), b"JSON") + padded
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


class FakeAsset:
    def __init__(self, data: bytes, name: str = "model.glb") -> None:
        self._data = data
        self.name = name

    def open(self) -> io.BytesIO:
        return io.BytesIO(self._data)


def test_model3d_round_trip_preserves_raw_container_bytes() -> None:
    data = glb_bytes()
    value = {"format": "glb", "bytes": data}
    assert encode_model3d(value) is data
    assert decode_model3d(data) == value
    assert model3d_meta(value) == {"format": "glb", "byte_size": len(data)}


def test_model3d_fingerprint_is_wrapper_independent() -> None:
    value = {"format": "glb", "bytes": glb_bytes()}
    decoded = decode_model3d(encode_model3d(value))
    fingerprint = model3d_fingerprint(MODEL3D_TYPE)
    assert fingerprint(value) == fingerprint(decoded)
    assert fingerprint(value) != model3d_fingerprint("other.type")(value)


def test_model3d_refuses_unknown_formats_and_unrecognized_bytes() -> None:
    data = glb_bytes()
    with pytest.raises(ValueError, match="format must be one of"):
        encode_model3d({"format": "gltf", "bytes": data})
    with pytest.raises(ValueError, match="missing format"):
        encode_model3d({"bytes": data})
    with pytest.raises(TypeError, match="must be bytes"):
        encode_model3d({"format": "glb", "bytes": "not bytes"})
    with pytest.raises(TypeError, match="must be a mapping"):
        encode_model3d(data)
    for bad in (
        b"",
        b"not a model",
        b"glTF",  # truncated header
        struct.pack("<4sII", b"FTlg", 2, 12),  # wrong magic
        struct.pack("<4sII", b"glTF", 1, 12),  # wrong version
        struct.pack("<4sII", b"glTF", 2, 16),  # declared length mismatch
        data + b"trailing",  # declared length no longer matches
    ):
        with pytest.raises(ValueError, match="not a GLB"):
            model3d_format(bad)
        with pytest.raises(ValueError, match="not a GLB"):
            decode_model3d(bad)


def test_model3d_format_checks_the_header_only() -> None:
    # The codec check is deliberately shallow: a valid 12-byte header over
    # garbage chunks passes here. Deep chunk-level validation is the media
    # upload authority's job (dinkster-assets), which every upload already ran.
    body = b"\xff" * 8
    data = struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
    assert model3d_format(data) == "glb"


def test_model3d_original_rendition_preserves_validated_bytes_and_metadata() -> None:
    data = glb_bytes()
    value = {"format": "glb", "bytes": data}
    assert render_model3d_original(value) is data
    metadata = model3d_meta(value)
    validate_model3d_encoded(data, metadata)
    validate_model3d_encoded(memoryview(data), metadata)


def test_model3d_encoded_validator_refuses_metadata_that_misidentifies_bytes() -> None:
    data = glb_bytes()
    with pytest.raises(ValueError, match="metadata format"):
        validate_model3d_encoded(data, {"format": "gltf", "byte_size": len(data)})
    with pytest.raises(ValueError, match="byte_size must be an int"):
        validate_model3d_encoded(data, {"format": "glb", "byte_size": True})
    with pytest.raises(ValueError, match="byte_size does not match"):
        validate_model3d_encoded(data, {"format": "glb", "byte_size": len(data) + 1})
    with pytest.raises(ValueError, match="not a GLB"):
        validate_model3d_encoded(b"not a model", {"format": "glb", "byte_size": 11})


def test_model3d_file_decoder_reads_assets_and_names_failures() -> None:
    data = glb_bytes()
    assert decode_model3d_file(FakeAsset(data)) == {"format": "glb", "bytes": data}
    with pytest.raises(ValueError, match="cannot decode 'broken.glb' as a 3D model"):
        decode_model3d_file(FakeAsset(b"not a model", name="broken.glb"))
    with pytest.raises(ValueError, match="cannot decode 'asset' as a 3D model"):
        decode_model3d_file(FakeAsset(b"not a model", name=""))
    with pytest.raises(TypeError, match="expects an asset with open()"):
        decode_model3d_file(data)


def test_model3d_registration_wires_codec_rendition_and_asset_decoder() -> None:
    registry = TypeRegistry()
    register_media_types(registry)
    spec = registry.spec(MODEL3D_TYPE)
    assert spec.declared_codec
    assert spec.validate_encoded_buffer is validate_model3d_encoded
    data = glb_bytes()
    value = registry.wrap(MODEL3D_TYPE, {"format": "glb", "bytes": data})
    assert value.meta.entries == {"format": "glb", "byte_size": len(data)}
    assert [(s.kind, s.mime, s.default) for s in registry.renditions_of(MODEL3D_TYPE)] == [
        ("glb", "model/gltf-binary", True)
    ]
    rendition = registry.render(value)
    assert (rendition.kind, rendition.mime, rendition.data) == (
        "glb",
        "model/gltf-binary",
        data,
    )
    decoder = registry.asset_decoder_for(MODEL3D_TYPE)
    assert decoder is not None
    assert decoder.provider_id == MODEL3D_FILE_DECODER_ID
    assert decoder.decode(FakeAsset(data)) == {"format": "glb", "bytes": data}
    # Idempotent: a second registration pass must not conflict.
    register_media_types(registry)


def test_model3d_registration_helper_is_self_contained_and_idempotent() -> None:
    registry = TypeRegistry()
    register_model3d_type(registry, MODEL3D_TYPE)
    register_model3d_type(registry, MODEL3D_TYPE)
    assert registry.spec(MODEL3D_TYPE).declared_codec
    assert registry.renditions_of(MODEL3D_TYPE)[0].mime == "model/gltf-binary"
    assert registry.asset_decoder_for(MODEL3D_TYPE) is not None
