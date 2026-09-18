"""Hand-built minimal PNG/WebP bytes for pack-icon tests.

The icon validator sniffs container structure (magic, IHDR/VP8X chunks,
animation markers), it does not decode pixels - so structurally valid
headers with dummy payload are exactly what the contract tests need,
with zero image-library dependencies.
"""

from __future__ import annotations

import zlib


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")


def png_bytes(width: int = 64, height: int = 64, *, animated: bool = False) -> bytes:
    """A minimal PNG: signature, IHDR, (acTL when animated), IDAT, IEND."""
    ihdr = _png_chunk(
        b"IHDR",
        width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0]),
    )
    actl = _png_chunk(b"acTL", (1).to_bytes(4, "big") + (0).to_bytes(4, "big")) if animated else b""
    idat = _png_chunk(b"IDAT", zlib.compress(b"\x00" * 16))
    return b"\x89PNG\r\n\x1a\n" + ihdr + actl + idat + _png_chunk(b"IEND", b"")


def _riff(chunks: bytes) -> bytes:
    payload = b"WEBP" + chunks
    return b"RIFF" + len(payload).to_bytes(4, "little") + payload


def webp_bytes(width: int = 64, height: int = 64) -> bytes:
    """A minimal static lossless WebP (VP8L form)."""
    bits = (width - 1) | ((height - 1) << 14)
    payload = b"\x2f" + bits.to_bytes(4, "little") + b"\x00" * 8
    chunk = b"VP8L" + len(payload).to_bytes(4, "little") + payload
    if len(payload) % 2:
        chunk += b"\x00"
    return _riff(chunk)


def animated_webp_bytes(width: int = 64, height: int = 64) -> bytes:
    """A minimal animated WebP (VP8X form with the animation flag set)."""
    payload = (
        bytes([0x02, 0, 0, 0])  # flags: animation bit
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    chunk = b"VP8X" + len(payload).to_bytes(4, "little") + payload
    return _riff(chunk)
