"""Proofs for the SPLAT codec, PLY interchange, and registration."""

from __future__ import annotations

import io
import struct
from typing import Any, cast

import numpy
import pytest
from dinkster_values import (
    EncodedLatentTensor,
    decode_splat,
    encode_splat,
    parse_ply_splat,
    render_splat_ply,
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
