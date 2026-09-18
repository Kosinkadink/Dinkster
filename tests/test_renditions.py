"""dev.image PNG rendition (DESIGN 3.5): the envelope-first value model
never puts raw tensors on the client wire, so dev.image registers a
browser-renderable PNG form. These tests decode the actual bytes - the
contract is "a browser can show this", not "some bytes came back"."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest
from dinkster_nodes_dev import register_dev_types
from dinkster_values import TypeRegistry, register_core_types

DEV_IMAGE = "dev.image"


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_dev_types(registry)
    return registry


def decode_png(data: bytes) -> tuple[int, int, int, list[bytes]]:
    """Minimal PNG reader: returns (width, height, color_type, rows).

    Independent of the encoder's internals on purpose: it walks the chunk
    grammar, so a malformed length/CRC/filter byte fails the test."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = height = color_type = -1
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + body)
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", body[:10])
            assert depth == 8
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    assert data[-12:-8] == struct.pack(">I", 0)  # IEND is empty and last
    raw = zlib.decompress(idat)
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = 1 + width * channels
    assert len(raw) == height * stride
    rows = []
    for y in range(height):
        row = raw[y * stride : (y + 1) * stride]
        assert row[0] == 0  # filter type None
        rows.append(row[1:])
    return width, height, color_type, rows


def render(array: np.ndarray) -> bytes:
    registry = make_registry()
    rendition = registry.render(registry.wrap(DEV_IMAGE, array), "png")
    assert (rendition.kind, rendition.mime) == ("png", "image/png")
    return rendition.data


def test_png_registered_as_default_rendition() -> None:
    registry = make_registry()
    specs = registry.renditions_of(DEV_IMAGE)
    assert [(s.kind, s.mime, s.default) for s in specs] == [("png", "image/png", True)]


def test_grayscale_png_roundtrip() -> None:
    array = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    width, height, color_type, rows = decode_png(render(array))
    assert (width, height, color_type) == (2, 2, 0)
    assert rows == [bytes([0, 255]), bytes([128, 64])]


def test_single_channel_axis_is_squeezed() -> None:
    array = np.zeros((3, 4, 1), dtype=np.float32)
    width, height, color_type, _ = decode_png(render(array))
    assert (width, height, color_type) == (4, 3, 0)


def test_rgb_and_rgba_pngs() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    rgb[0, 0] = [1.0, 0.0, 0.5]
    width, height, color_type, rows = decode_png(render(rgb))
    assert (width, height, color_type) == (3, 2, 2)
    assert rows[0][:3] == bytes([255, 0, 128])

    rgba = np.ones((2, 2, 4), dtype=np.float32)
    _, _, color_type, rows = decode_png(render(rgba))
    assert color_type == 6
    assert rows[0] == bytes([255] * 8)


def test_out_of_range_values_clip() -> None:
    array = np.array([[-3.0, 7.5]], dtype=np.float32)
    _, _, _, rows = decode_png(render(array))
    assert rows == [bytes([0, 255])]


def test_unrenderable_shape_raises() -> None:
    registry = make_registry()
    bad = registry.wrap(DEV_IMAGE, np.zeros((2, 2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot render"):
        registry.render(bad, "png")
