"""Portable SVG values as ordered, immutable raw byte streams."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .limits import ENCODED_MEDIA_LIMIT_BYTES
from .model import stable_hash

__all__ = [
    "SVG_CODEC_MAGIC",
    "SVG_CODEC_VERSION",
    "decode_svg",
    "encode_svg",
    "svg_fingerprint",
    "svg_meta",
    "validate_svg_encoded",
]

SVG_CODEC_MAGIC = b"DINKSTER-SVG\x00"
SVG_CODEC_VERSION = 1
_COUNT_BYTES = 4
_LENGTH_BYTES = 8
_HEADER_BYTES = len(SVG_CODEC_MAGIC) + 1 + _COUNT_BYTES
_MAX_STREAMS = 1_000_000


def _streams(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, tuple):
        raise TypeError("SVG value must be an immutable tuple of bytes")
    if len(value) > _MAX_STREAMS:
        raise ValueError("SVG value exceeds the stream count limit")
    total = 0
    for stream in value:
        if not isinstance(stream, bytes):
            raise TypeError("SVG value members must be bytes")
        total += len(stream)
        if total > ENCODED_MEDIA_LIMIT_BYTES:
            raise ValueError("SVG value exceeds the encoded media size limit")
    return value


def encode_svg(value: object) -> bytes:
    """Frame exact source bytes without interpreting SVG content."""
    streams = _streams(value)
    frame = [
        SVG_CODEC_MAGIC,
        bytes((SVG_CODEC_VERSION,)),
        len(streams).to_bytes(_COUNT_BYTES, "little"),
    ]
    for stream in streams:
        frame.extend((len(stream).to_bytes(_LENGTH_BYTES, "little"), stream))
    return b"".join(frame)


def decode_svg(data: bytes) -> tuple[bytes, ...]:
    """Decode one exact ordered SVG stream sequence."""
    data = bytes(data)
    if len(data) < _HEADER_BYTES or not data.startswith(SVG_CODEC_MAGIC):
        raise ValueError("invalid SVG codec framing")
    version_offset = len(SVG_CODEC_MAGIC)
    if data[version_offset] != SVG_CODEC_VERSION:
        raise ValueError("unsupported SVG codec version")
    count_offset = version_offset + 1
    count = int.from_bytes(data[count_offset : count_offset + _COUNT_BYTES], "little")
    if count > _MAX_STREAMS:
        raise ValueError("SVG codec stream count exceeds its limit")
    offset = _HEADER_BYTES
    total = 0
    streams: list[bytes] = []
    for _ in range(count):
        if len(data) - offset < _LENGTH_BYTES:
            raise ValueError("SVG codec stream length is truncated")
        size = int.from_bytes(data[offset : offset + _LENGTH_BYTES], "little")
        offset += _LENGTH_BYTES
        total += size
        if total > ENCODED_MEDIA_LIMIT_BYTES:
            raise ValueError("SVG codec streams exceed the encoded media size limit")
        if len(data) - offset < size:
            raise ValueError("SVG codec stream is truncated")
        streams.append(data[offset : offset + size])
        offset += size
    if offset != len(data):
        raise ValueError("SVG codec frame carries trailing bytes")
    return tuple(streams)


def svg_fingerprint(type_id: str) -> Callable[[object], str]:
    """Fingerprint the canonical framed byte sequence under its type id."""

    def fingerprint(value: object) -> str:
        return stable_hash([type_id.encode("utf-8"), encode_svg(value)])

    return fingerprint


def svg_meta(value: object) -> Mapping[str, object]:
    """Report the ordered member count and total unframed byte count."""
    streams = _streams(value)
    return {"streams": len(streams), "byte_size": sum(map(len, streams))}


def validate_svg_encoded(data: bytes | memoryview, metadata: Mapping[str, object]) -> None:
    """Require optional envelope metadata to agree with canonical framing."""
    streams = decode_svg(bytes(data))
    if not metadata:
        return
    if metadata.get("streams") != len(streams):
        raise ValueError("SVG metadata streams does not match encoded bytes")
    if metadata.get("byte_size") != sum(map(len, streams)):
        raise ValueError("SVG metadata byte_size does not match encoded bytes")
