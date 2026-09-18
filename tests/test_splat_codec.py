"""Proofs for the dinkster.splat codec, PLY interchange, and registration."""

from __future__ import annotations

import io
import struct
from typing import Any, cast

import numpy
import pytest
from dinkster_nodes_media_io import SPLAT_TYPE, register_media_types
from dinkster_values import (
    SPLAT_CODEC_MAGIC,
    SPLAT_CODEC_VERSION,
    SPLAT_PLY_MIME,
    EncodedLatentTensor,
    TypeRegistry,
    decode_splat,
    decode_splat_file,
    encode_splat,
    parse_ply_splat,
    render_splat_ply,
    splat_fingerprint,
    splat_meta,
    validate_splat_encoded,
)

# 0.5 / C0 with C0 the degree-0 SH basis constant.
SH_C0 = 0.28209479177387814


def splat_value(
    batch: int = 1, gaussians: int = 5, coefficients: int = 4, counts: list[int] | None = None
) -> dict[str, Any]:
    rng = numpy.random.default_rng(7)
    rotations = rng.normal(0.0, 1.0, (batch, gaussians, 4)).astype(numpy.float32)
    rotations /= numpy.linalg.norm(rotations, axis=2, keepdims=True)
    value: dict[str, Any] = {
        "positions": rng.uniform(-0.5, 0.5, (batch, gaussians, 3)).astype(numpy.float32),
        "scales": rng.uniform(0.005, 0.05, (batch, gaussians, 3)).astype(numpy.float32),
        "rotations": rotations,
        "opacities": rng.uniform(0.05, 0.95, (batch, gaussians, 1)).astype(numpy.float32),
        "sh": rng.normal(0.0, 1.0, (batch, gaussians, coefficients, 3)).astype(numpy.float32),
    }
    if counts is not None:
        value["counts"] = numpy.asarray(counts, numpy.int64)
    return value


class FakeAsset:
    def __init__(self, data: bytes, name: str = "scene.ply") -> None:
        self._data = data
        self.name = name

    def open(self) -> io.BytesIO:
        return io.BytesIO(self._data)


def test_splat_round_trip_is_deterministic_and_lossless() -> None:
    value = splat_value(batch=2, counts=[5, 3])
    data = encode_splat(value)
    assert data == encode_splat(value)
    decoded = cast("dict[str, EncodedLatentTensor]", decode_splat(data))
    assert list(decoded) == ["positions", "scales", "rotations", "opacities", "sh", "counts"]
    for key, record in decoded.items():
        assert record.dtype == str(value[key].dtype)
        assert record.shape == value[key].shape
        assert record.data == value[key].tobytes()
    assert encode_splat(decoded) == data


def test_splat_decode_applies_the_tensor_decoder() -> None:
    value = splat_value()
    decoded = decode_splat(
        encode_splat(value),
        tensor_decoder=lambda record: numpy.frombuffer(record.data, record.dtype).reshape(
            record.shape
        ),
    )
    arrays = cast("dict[str, Any]", decoded)
    assert all(numpy.array_equal(arrays[key], value[key]) for key in value)


def test_splat_fingerprint_is_form_independent() -> None:
    value = splat_value()
    fingerprint = splat_fingerprint(SPLAT_TYPE)
    assert fingerprint(value) == fingerprint(decode_splat(encode_splat(value)))
    assert fingerprint(value) != splat_fingerprint("other.type")(value)


def test_splat_meta_and_encoded_validator_agree() -> None:
    value = splat_value(batch=2, gaussians=6, coefficients=3)
    metadata = splat_meta(value)
    assert metadata == {"batch": 2, "gaussians": 6, "sh_coefficients": 3}
    validate_splat_encoded(encode_splat(value), metadata)
    validate_splat_encoded(memoryview(encode_splat(value)), metadata)
    with pytest.raises(ValueError, match="metadata gaussians"):
        validate_splat_encoded(encode_splat(value), {**metadata, "gaussians": 7})
    with pytest.raises(ValueError, match="metadata batch"):
        validate_splat_encoded(encode_splat(value), {**metadata, "batch": True})


def test_splat_runtime_form_rejects_malformed_layouts() -> None:
    value = splat_value()
    with pytest.raises(TypeError, match="must be a mapping"):
        encode_splat(value["positions"])
    with pytest.raises(ValueError, match="exactly positions"):
        encode_splat({key: value[key] for key in value if key != "sh"})
    with pytest.raises(ValueError, match="exactly positions"):
        encode_splat({**value, "extras": value["positions"]})
    with pytest.raises(TypeError, match="positions must be a tensor-like"):
        encode_splat({**value, "positions": "vertices"})
    with pytest.raises(ValueError, match="rotations shape"):
        encode_splat({**value, "rotations": value["positions"]})
    with pytest.raises(ValueError, match="opacities shape"):
        encode_splat({**value, "opacities": value["opacities"][:, :3]})
    with pytest.raises(ValueError, match=r"shape \(batch, gaussians, coefficients, 3\)"):
        encode_splat({**value, "sh": value["sh"][..., :2]})
    with pytest.raises(ValueError, match="dtype must be one of"):
        encode_splat({**value, "scales": value["scales"].astype(numpy.float64)})
    with pytest.raises(ValueError, match="counts dtype"):
        encode_splat({**value, "counts": numpy.asarray([1.0], numpy.float32)})
    with pytest.raises(ValueError, match=r"counts must have shape \(batch,\)"):
        encode_splat({**value, "counts": numpy.asarray([1, 2], numpy.int64)})
    with pytest.raises(ValueError, match="between 0 and the gaussian"):
        encode_splat({**value, "counts": numpy.asarray([9], numpy.int64)})


def test_splat_decode_rejects_malformed_frames() -> None:
    data = encode_splat(splat_value())
    with pytest.raises(ValueError, match="invalid splat codec framing"):
        decode_splat(b"DINKSTER-OTHER\x00" + data[len(SPLAT_CODEC_MAGIC) :])
    with pytest.raises(ValueError, match="unsupported splat codec version"):
        decode_splat(
            SPLAT_CODEC_MAGIC
            + bytes((SPLAT_CODEC_VERSION + 1,))
            + data[len(SPLAT_CODEC_MAGIC) + 1 :]
        )
    with pytest.raises(ValueError, match="truncated"):
        decode_splat(data[:-4])
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_splat(data + b"\x00")
    reordered = dict(reversed(cast("dict[str, Any]", decode_splat(data)).items()))
    with pytest.raises(ValueError, match="canonical order"):
        decode_splat(_reorder_frame(data, reordered))
    offset = len(SPLAT_CODEC_MAGIC) + 1
    with pytest.raises(ValueError, match="size bound"):
        decode_splat(data[:offset] + (2**31).to_bytes(4, "little") + data[offset + 4 :])
    header_size = int.from_bytes(data[offset : offset + 4], "little")
    padded = data[offset + 4 : offset + 4 + header_size] + b" "
    with pytest.raises(ValueError, match="canonical form"):
        decode_splat(
            data[:offset]
            + len(padded).to_bytes(4, "little")
            + padded
            + data[offset + 4 + header_size :]
        )


def test_render_splat_ply_limit_refuses_before_serializing() -> None:
    value = splat_value(gaussians=8)
    with pytest.raises(ValueError, match="output limit"):
        render_splat_ply(value, limit=64)
    assert len(render_splat_ply(value, limit=1024 * 1024)) > 64


def _reorder_frame(data: bytes, records: dict[str, Any]) -> bytes:
    import json

    offset = len(SPLAT_CODEC_MAGIC)
    manifest = [[key, record.dtype, list(record.shape)] for key, record in records.items()]
    header = json.dumps({"arrays": manifest}, separators=(",", ":")).encode()
    frame = [data[: offset + 1], len(header).to_bytes(4, "little"), header]
    frame.extend(record.data for record in records.values())
    return b"".join(frame)


def test_ply_round_trip_recovers_activated_values() -> None:
    value = splat_value(gaussians=8, coefficients=4)
    data = render_splat_ply(value)
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 8\n")
    parsed = cast("dict[str, Any]", parse_ply_splat(data))
    assert parsed["positions"].shape == (1, 8, 3)
    for key in ("positions", "scales", "rotations", "opacities", "sh"):
        assert numpy.allclose(parsed[key], value[key], atol=1e-5), key


def test_ply_render_honors_counts_and_uses_batch_item_zero() -> None:
    value = splat_value(batch=2, gaussians=5, counts=[3, 5])
    data = render_splat_ply(value)
    assert b"element vertex 3\n" in data
    parsed = cast("dict[str, Any]", parse_ply_splat(data))
    assert numpy.allclose(parsed["positions"][0], value["positions"][0, :3], atol=1e-5)
    with pytest.raises(ValueError, match="no gaussians"):
        render_splat_ply(splat_value(counts=[0]))


def test_ply_render_accepts_encoded_records_and_half_precision() -> None:
    value = splat_value()
    records = decode_splat(encode_splat(value))
    assert render_splat_ply(records) == render_splat_ply(value)
    half = {key: array.astype(numpy.float16) for key, array in value.items()}
    parsed = cast("dict[str, Any]", parse_ply_splat(render_splat_ply(half)))
    assert numpy.allclose(parsed["positions"], value["positions"], atol=1e-2)


def test_ply_parse_defaults_missing_attributes() -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    body = numpy.arange(6, dtype="<f4").tobytes()
    parsed = cast("dict[str, Any]", parse_ply_splat(header + body))
    assert numpy.allclose(parsed["scales"], 0.01)
    assert numpy.allclose(parsed["rotations"], [[1, 0, 0, 0], [1, 0, 0, 0]])
    assert numpy.allclose(parsed["opacities"], 1.0)
    assert numpy.allclose(parsed["sh"], 0.0)


def test_ply_parse_maps_plain_colors_onto_the_sh_dc_band() -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    body = struct.pack("<fff3B", 0.0, 0.0, 0.0, 255, 0, 127)
    parsed = cast("dict[str, Any]", parse_ply_splat(header + body))
    expected = (numpy.array([255, 0, 127], numpy.float32) / 255.0 - 0.5) / SH_C0
    assert numpy.allclose(parsed["sh"][0, 0, 0], expected, atol=1e-5)


def test_ply_parse_rejects_malformed_containers() -> None:
    good = render_splat_ply(splat_value())
    with pytest.raises(ValueError, match="end_header"):
        parse_ply_splat(good[:20])
    with pytest.raises(ValueError, match="format 'ascii'"):
        parse_ply_splat(good.replace(b"binary_little_endian", b"ascii", 1))
    with pytest.raises(ValueError, match="vertex as its first element"):
        parse_ply_splat(good.replace(b"element vertex", b"element face", 1))
    with pytest.raises(ValueError, match="list properties"):
        parse_ply_splat(good.replace(b"property float x\n", b"property list uchar int i\n", 1))
    with pytest.raises(ValueError, match="truncated"):
        parse_ply_splat(good[:-8])
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float a\nproperty float b\nproperty float c\nend_header\n"
    )
    with pytest.raises(ValueError, match="x, y, and z"):
        parse_ply_splat(header + bytes(12))


def test_splat_file_decoder_reads_assets_and_names_failures() -> None:
    value = splat_value()
    data = render_splat_ply(value)
    decoded = cast("dict[str, Any]", decode_splat_file(FakeAsset(data)))
    assert numpy.allclose(decoded["positions"], value["positions"], atol=1e-5)
    with pytest.raises(ValueError, match="cannot decode 'broken.ply' as a gaussian splat"):
        decode_splat_file(FakeAsset(b"not a splat", name="broken.ply"))
    with pytest.raises(TypeError, match="expects an asset with open()"):
        decode_splat_file(data)


def test_splat_registration_wires_codec_rendition_and_asset_decoder() -> None:
    registry = TypeRegistry()
    register_media_types(registry)
    spec = registry.spec(SPLAT_TYPE)
    assert spec.declared_codec
    assert spec.validate_encoded_buffer is validate_splat_encoded
    value = registry.wrap(SPLAT_TYPE, splat_value(batch=1, gaussians=5, coefficients=4))
    assert value.meta.entries == {"batch": 1, "gaussians": 5, "sh_coefficients": 4}
    assert [(s.kind, s.mime, s.default) for s in registry.renditions_of(SPLAT_TYPE)] == [
        ("ply", SPLAT_PLY_MIME, True)
    ]
    rendition = registry.render(value)
    assert rendition.mime == SPLAT_PLY_MIME
    assert rendition.data.startswith(b"ply\n")
