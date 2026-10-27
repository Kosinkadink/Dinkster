"""Portable SVG values preserve opaque ordered source bytes."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest
from dinkster_api.v1 import SVG_TYPE, TypeRegistry, register_inference_types
from dinkster_values import (
    SVG_CODEC_MAGIC,
    SVG_CODEC_VERSION,
    decode_svg,
    encode_svg,
    svg_fingerprint,
    svg_meta,
    validate_svg_encoded,
)


def test_svg_round_trip_preserves_order_duplicates_empty_and_non_text_bytes() -> None:
    value = (b"<svg>one</svg>", b"", b"<svg>one</svg>", b"\xff\x00<svg/>")
    encoded = encode_svg(value)
    assert encoded == encode_svg(value)
    assert decode_svg(encoded) == value
    assert svg_meta(value) == {"streams": 4, "byte_size": sum(map(len, value))}
    fingerprint = svg_fingerprint(SVG_TYPE)
    assert fingerprint(value) == fingerprint(decode_svg(encoded))
    assert fingerprint(value) != fingerprint((value[2], value[1], value[0], value[3]))
    assert fingerprint(value) != fingerprint(value[:-1])


def test_svg_uses_complete_stream_bytes_without_changing_source_cursors() -> None:
    sources = [io.BytesIO(b"first"), io.BytesIO(b"second")]
    sources[0].seek(2)
    sources[1].seek(4)
    value = tuple(source.getvalue() for source in sources)
    assert encode_svg(value)
    assert value == (b"first", b"second")
    assert tuple(source.tell() for source in sources) == (2, 4)


@pytest.mark.parametrize(
    "value",
    (
        [b"<svg/>"],
        (bytearray(b"<svg/>"),),
        (memoryview(b"<svg/>"),),
        ("<svg/>",),
    ),
)
def test_svg_rejects_mutable_or_non_byte_runtime_values(value: object) -> None:
    with pytest.raises(TypeError):
        encode_svg(value)


def test_svg_rejects_malformed_frames_and_metadata() -> None:
    data = encode_svg((b"<svg/>", b""))
    with pytest.raises(ValueError, match="invalid SVG codec framing"):
        decode_svg(b"not-an-svg-frame")
    with pytest.raises(ValueError, match="unsupported SVG codec version"):
        decode_svg(
            SVG_CODEC_MAGIC + bytes((SVG_CODEC_VERSION + 1,)) + data[len(SVG_CODEC_MAGIC) + 1 :]
        )
    with pytest.raises(ValueError, match="stream length is truncated"):
        decode_svg(data[: -len(b"<svg/>") - 2])
    with pytest.raises(ValueError, match="stream is truncated"):
        decode_svg(data[:-1])
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_svg(data + b"\x00")
    with pytest.raises(ValueError, match="metadata streams"):
        validate_svg_encoded(data, {"streams": 3, "byte_size": 6})
    with pytest.raises(ValueError, match="metadata byte_size"):
        validate_svg_encoded(data, {"streams": 2, "byte_size": 7})


def test_svg_registration_preserves_the_first_authority() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SVG_TYPE)
    register_inference_types(registry)
    assert registry.spec(SVG_TYPE) is spec
    value = (b"<svg/>", b"", b"<svg/>")
    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, svg_meta(value))


def test_svg_public_type_registers_on_two_fresh_ordinary_hosts() -> None:
    script = """
import sys
from dinkster_api.v1 import SVG_TYPE, TypeRegistry, register_inference_types
from dinkster_values.builtin_types import SVG_TYPE as canonical

assert SVG_TYPE is canonical
registry = TypeRegistry()
register_inference_types(registry)
register_inference_types(registry)
spec = registry.spec(SVG_TYPE)
value = (b'<svg/>', b'', b'\\xff')
encoded = spec.encode(value)
assert spec.decode(encoded) == value
assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
