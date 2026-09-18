"""Dependency-free byte authority for the closed upload media set.

Classification uses only container bytes. Names, extensions, declared MIME,
asset metadata, and paths are deliberately absent from the API. The parsers
validate bounded container framing and the identifying structures needed to
assign a media kind; they do not claim that codec payloads fully decode.
"""

from __future__ import annotations

import json
import math
import mmap
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from dinkster_values.media_containers import bmff_boxes as _bmff_boxes
from dinkster_values.media_containers import ebml_elements as _ebml_elements

from .identity import AssetError
from .kind import KIND_MEDIA_AUDIO, KIND_MEDIA_IMAGE, KIND_MEDIA_MODEL3D, KIND_MEDIA_VIDEO

_MEDIA_ROWS = (
    (KIND_MEDIA_IMAGE, "image/png", "png"),
    (KIND_MEDIA_IMAGE, "image/jpeg", "jpg"),
    (KIND_MEDIA_IMAGE, "image/webp", "webp"),
    (KIND_MEDIA_AUDIO, "audio/wav", "wav"),
    (KIND_MEDIA_AUDIO, "audio/flac", "flac"),
    (KIND_MEDIA_AUDIO, "audio/mpeg", "mp3"),
    (KIND_MEDIA_AUDIO, "audio/ogg", "ogg"),
    (KIND_MEDIA_AUDIO, "audio/webm", "webm"),
    (KIND_MEDIA_AUDIO, "audio/mp4", "m4a"),
    (KIND_MEDIA_VIDEO, "video/mp4", "mp4"),
    (KIND_MEDIA_VIDEO, "video/webm", "webm"),
    (KIND_MEDIA_MODEL3D, "model/gltf-binary", "glb"),
    (KIND_MEDIA_MODEL3D, "model/ply", "ply"),
)
_STRUCTURE_ITEM_LIMIT = 100_000
_GLB_JSON_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True)
class MediaClassification:
    """Canonical immutable facts derived from supported media bytes."""

    kind: str
    media_type: str
    extension: str

    def __post_init__(self) -> None:
        if (self.kind, self.media_type, self.extension) not in _MEDIA_ROWS:
            raise AssetError("media classification facts are not canonical")


@dataclass(frozen=True)
class RasterImageFacts:
    media_type: str
    width: int
    height: int
    channel_depth: int
    alpha_mode: str


def raster_image_facts(data: Any) -> RasterImageFacts:
    """Return bounded, container-derived facts for a validated raster image."""
    classification = _classify_media_buffer(data)
    if classification.kind != KIND_MEDIA_IMAGE:
        raise AssetError("bytes are not a supported raster image")
    if classification.media_type == "image/png":
        width, height, depth, color = struct.unpack_from(">IIBB", data, 16)
        if depth == 16:
            raise AssetError("raster image channel depth is not 8-bit")
        depth = 8
        has_transparency = _png_has_chunk(data, b"tRNS")
        if _png_has_chunk(data, b"acTL"):
            raise AssetError("animated PNG is not a single raster image")
        alpha = "straight" if color in (4, 6) or has_transparency else "opaque"
    elif classification.media_type == "image/jpeg":
        width, height, depth = _jpeg_dimensions(data)
        if depth != 8:
            raise AssetError("raster image channel depth is not 8-bit")
        alpha = "opaque"
    else:
        width, height, alpha = _webp_dimensions(data)
        depth = 8
    return RasterImageFacts(classification.media_type, width, height, depth, alpha)


def _png_has_chunk(data: Any, expected: bytes) -> bool:
    offset = 8
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = bytes(data[offset + 4 : offset + 8])
        if kind == expected:
            return True
        offset += 12 + length
    return False


def _jpeg_dimensions(data: bytes | memoryview) -> tuple[int, int, int]:
    offset = 2
    while offset + 4 <= len(data):
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        marker = data[offset]
        offset += 1
        if marker in _JPEG_SOF:
            length = struct.unpack_from(">H", data, offset)[0]
            payload = data[offset + 2 : offset + length]
            return (
                int.from_bytes(payload[3:5], "big"),
                int.from_bytes(payload[1:3], "big"),
                payload[0],
            )
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        length = struct.unpack_from(">H", data, offset)[0]
        offset += length
    raise AssetError("JPEG has no frame dimensions")


def _webp_dimensions(data: Any) -> tuple[int, int, str]:
    chunks = _riff_chunks(data)
    for kind, payload in chunks:
        if kind == b"VP8X":
            if payload[0] & 0x02:
                raise AssetError("animated WebP is not a single raster image")
            return (
                int.from_bytes(payload[4:7], "little") + 1,
                int.from_bytes(payload[7:10], "little") + 1,
                "straight" if payload[0] & 0x10 else "opaque",
            )
        if kind == b"VP8 ":
            return (
                int.from_bytes(payload[6:8], "little") & 0x3FFF,
                int.from_bytes(payload[8:10], "little") & 0x3FFF,
                "opaque",
            )
        if kind == b"VP8L":
            packed = int.from_bytes(payload[1:5], "little")
            return (
                (packed & 0x3FFF) + 1,
                ((packed >> 14) & 0x3FFF) + 1,
                "straight" if packed & (1 << 28) else "opaque",
            )
    raise AssetError("WebP has no image dimensions")


_MEDIA_BY_TYPE = {
    media_type: MediaClassification(kind, media_type, extension)
    for kind, media_type, extension in _MEDIA_ROWS
}


def classify_media(data: object) -> MediaClassification:
    """Classify supported upload bytes or refuse malformed/unknown content."""
    if not isinstance(data, bytes):
        raise AssetError("media classification requires bytes")
    return _classify_media_buffer(data)


def classify_media_file(path: Path) -> MediaClassification:
    """Classify a held media file without copying it into process memory."""
    # The failing exception is reduced to a message before AssetError is
    # raised: a live exception's traceback would keep memoryview exports of
    # the mmap alive, making mmap close raise BufferError and, on Windows,
    # blocking deletion of the underlying file.
    try:
        with (
            path.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
        ):
            return _classify_media_buffer(data)
    except (OSError, ValueError) as error:
        failure = f"unsupported or malformed media bytes: {error}"
    raise AssetError(failure) from None


def classify_media_handle(handle: BinaryIO) -> MediaClassification:
    """Classify media through an already-authorized open descriptor."""
    try:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return _classify_media_buffer(data)
    except (OSError, ValueError) as error:
        failure = f"unsupported or malformed media bytes: {error}"
    raise AssetError(failure) from None


def _classify_media_buffer(data: Any) -> MediaClassification:
    try:
        prefix = data[:12]
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            _validate_png(data)
            media_type = "image/png"
        elif prefix.startswith(b"\xff\xd8"):
            _validate_jpeg(data)
            media_type = "image/jpeg"
        elif prefix.startswith(b"RIFF"):
            form = _validate_riff(data)
            if form == b"WEBP":
                _validate_webp(data)
                media_type = "image/webp"
            elif form == b"WAVE":
                _validate_wav(data)
                media_type = "audio/wav"
            else:
                raise ValueError("unsupported RIFF form")
        elif prefix.startswith(b"fLaC"):
            _validate_flac(data)
            media_type = "audio/flac"
        elif prefix.startswith((b"ID3", b"\xff")):
            _validate_mp3(data)
            media_type = "audio/mpeg"
        elif prefix.startswith(b"OggS"):
            _validate_ogg(data)
            media_type = "audio/ogg"
        elif len(data) >= 8 and data[4:8] == b"ftyp":
            media_type = _validate_mp4(data)
        elif prefix.startswith(b"\x1aE\xdf\xa3"):
            media_type = _validate_webm(data)
        elif prefix.startswith(b"glTF"):
            _validate_glb(data)
            media_type = "model/gltf-binary"
        elif prefix.startswith(b"ply"):
            _validate_ply(data)
            media_type = "model/ply"
        else:
            raise ValueError("unknown media signature")
    except (IndexError, OverflowError, struct.error, ValueError) as error:
        failure = f"unsupported or malformed media bytes: {error}"
    else:
        return _MEDIA_BY_TYPE[media_type]
    raise AssetError(failure) from None


def _validate_png(data: bytes) -> None:
    offset = 8
    seen_ihdr = False
    seen_idat = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk payload")
        kind = data[offset + 4 : offset + 8]
        payload = memoryview(data)[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(payload, zlib.crc32(kind)) != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if not seen_ihdr:
            if kind != b"IHDR" or length != 13:
                raise ValueError("PNG must start with a 13-byte IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                width == 0
                or height == 0
                or depth not in valid_depths.get(color, set())
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                raise ValueError("invalid PNG IHDR")
            seen_ihdr = True
        elif kind == b"IHDR":
            raise ValueError("duplicate PNG IHDR")
        if kind == b"IDAT":
            seen_idat = True
        if kind == b"IEND":
            if length != 0 or not seen_idat or end != len(data):
                raise ValueError("invalid or non-terminal PNG IEND")
            return
        offset = end
    raise ValueError("PNG has no IEND")


_JPEG_SOF = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _validate_jpeg(data: bytes) -> None:
    offset = 2
    saw_sof = False
    frame_components: set[int] = set()
    saw_scan = False
    entropy = False
    while offset < len(data):
        if entropy:
            marker_at = data.find(b"\xff", offset)
            if marker_at < 0 or marker_at + 1 >= len(data):
                raise ValueError("truncated JPEG scan")
            offset = marker_at
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                raise ValueError("truncated JPEG marker")
            marker = data[offset]
            offset += 1
            if marker == 0x00 or 0xD0 <= marker <= 0xD7:
                continue
            entropy = False
        else:
            if offset >= len(data) or data[offset] != 0xFF:
                raise ValueError("JPEG marker framing is invalid")
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                raise ValueError("truncated JPEG marker")
            marker = data[offset]
            offset += 1
        if marker == 0xD9:
            if not saw_sof or not saw_scan or offset != len(data):
                raise ValueError("invalid or non-terminal JPEG EOI")
            return
        if marker in (0xD8, 0x00) or 0xD0 <= marker <= 0xD7:
            raise ValueError("unexpected standalone JPEG marker")
        if marker == 0x01:
            continue
        if offset + 2 > len(data):
            raise ValueError("truncated JPEG segment length")
        length = struct.unpack_from(">H", data, offset)[0]
        if length < 2 or offset + length > len(data):
            raise ValueError("invalid JPEG segment length")
        payload = data[offset + 2 : offset + length]
        if marker in _JPEG_SOF:
            if saw_sof:
                raise ValueError("JPEG has multiple frame headers")
            component_count = payload[5] if len(payload) >= 6 else 0
            if (
                len(payload) < 6
                or int.from_bytes(payload[1:3], "big") == 0
                or int.from_bytes(payload[3:5], "big") == 0
                or not 1 <= component_count <= 4
                or len(payload) != 6 + 3 * component_count
            ):
                raise ValueError("invalid JPEG frame header")
            components = payload[6::3]
            sampling = payload[7::3]
            if len(set(components)) != component_count or any(
                value == 0 or value >> 4 == 0 or value & 0x0F == 0 for value in sampling
            ):
                raise ValueError("invalid JPEG frame components")
            frame_components = set(components)
            saw_sof = True
        if marker == 0xDA:
            scan_components = payload[0] if payload else 0
            scan_ids = payload[1:-3:2]
            if (
                not saw_sof
                or not 1 <= scan_components <= 4
                or len(payload) != 4 + 2 * scan_components
                or len(set(scan_ids)) != scan_components
                or not set(scan_ids) <= frame_components
                or any(value >> 4 > 3 or value & 0x0F > 3 for value in payload[2:-3:2])
            ):
                raise ValueError("JPEG scan precedes a valid frame")
            saw_scan = True
            entropy = True
        offset += length
    raise ValueError("JPEG has no EOI")


def _validate_riff(data: bytes) -> bytes:
    if len(data) < 12:
        raise ValueError("truncated RIFF header")
    declared = struct.unpack_from("<I", data, 4)[0]
    if declared != len(data) - 8:
        raise ValueError("RIFF declared size does not match bytes")
    return data[8:12]


def _riff_chunks(data: bytes) -> list[tuple[bytes, memoryview]]:
    chunks: list[tuple[bytes, memoryview]] = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated RIFF chunk")
        kind = data[offset : offset + 4]
        length = struct.unpack_from("<I", data, offset + 4)[0]
        payload_end = offset + 8 + length
        end = payload_end + (length & 1)
        if end > len(data):
            raise ValueError("truncated RIFF chunk payload")
        if length & 1 and data[payload_end] != 0:
            raise ValueError("invalid RIFF padding")
        chunks.append((kind, memoryview(data)[offset + 8 : payload_end]))
        if len(chunks) > _STRUCTURE_ITEM_LIMIT:
            raise ValueError("RIFF contains too many chunks")
        offset = end
    return chunks


def _validate_webp(data: bytes) -> None:
    chunks = _riff_chunks(data)
    headers = [payload for kind, payload in chunks if kind == b"VP8X"]
    images = [(kind, payload) for kind, payload in chunks if kind in (b"VP8 ", b"VP8L")]
    animations = [payload for kind, payload in chunks if kind == b"ANIM"]
    frames = [payload for kind, payload in chunks if kind == b"ANMF"]
    animated = bool(animations or frames)
    if len(headers) > 1 or len(images) > 1 or (not images and not animated):
        raise ValueError("WebP has no unambiguous image payload")
    if headers:
        header = headers[0]
        if len(header) != 10 or header[0] & 0xC1 or bytes(header[1:4]) != bytes(3):
            raise ValueError("invalid WebP VP8X header")
        if bool(header[0] & 0x02) != animated:
            raise ValueError("WebP animation flag does not match chunks")
        if chunks[0][0] != b"VP8X":
            raise ValueError("WebP VP8X must be the first chunk")
        canvas_width = int.from_bytes(header[4:7], "little") + 1
        canvas_height = int.from_bytes(header[7:10], "little") + 1
        if canvas_width * canvas_height > 0xFFFFFFFF:
            raise ValueError("WebP canvas exceeds format bound")
    elif animated:
        raise ValueError("animated WebP requires VP8X")
    else:
        canvas_width = canvas_height = 0
    if images:
        if animated or (not headers and chunks[0][0] != images[0][0]):
            raise ValueError("WebP top-level image chunk is misplaced")
        _validate_webp_image(*images[0])
    if animated:
        if len(animations) != 1 or len(animations[0]) != 6 or not frames:
            raise ValueError("invalid WebP animation framing")
        animation_index = next(index for index, chunk in enumerate(chunks) if chunk[0] == b"ANIM")
        frame_indexes = [index for index, chunk in enumerate(chunks) if chunk[0] == b"ANMF"]
        if animation_index == 0 or any(index <= animation_index for index in frame_indexes):
            raise ValueError("WebP animation chunks are out of order")
        for frame in frames:
            if len(frame) < 24:
                raise ValueError("truncated WebP animation frame")
            x = 2 * int.from_bytes(frame[0:3], "little")
            y = 2 * int.from_bytes(frame[3:6], "little")
            width = int.from_bytes(frame[6:9], "little") + 1
            height = int.from_bytes(frame[9:12], "little") + 1
            if frame[15] & 0xFC or x + width > canvas_width or y + height > canvas_height:
                raise ValueError("invalid WebP animation frame bounds or flags")
            nested = _riff_chunks(
                b"RIFF" + struct.pack("<I", len(frame) - 12) + b"WEBP" + bytes(frame[16:])
            )
            nested_images = [
                (kind, payload) for kind, payload in nested if kind in (b"VP8 ", b"VP8L")
            ]
            if len(nested_images) != 1:
                raise ValueError("WebP animation frame requires one image chunk")
            _validate_webp_image(*nested_images[0])


def _validate_webp_image(kind: bytes, payload: bytes | memoryview) -> None:
    if kind == b"VP8 ":
        if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
            raise ValueError("invalid WebP VP8 frame header")
        tag = int.from_bytes(payload[:3], "little")
        width = int.from_bytes(payload[6:8], "little") & 0x3FFF
        height = int.from_bytes(payload[8:10], "little") & 0x3FFF
        if (
            tag & 1
            or ((tag >> 1) & 0x07) > 3
            or not tag & 0x10
            or tag >> 5 > len(payload) - 10
            or not width
            or not height
        ):
            raise ValueError("invalid WebP VP8 key frame")
        return
    if kind == b"VP8L":
        if len(payload) < 5 or payload[0] != 0x2F:
            raise ValueError("invalid WebP VP8L frame header")
        if int.from_bytes(payload[1:5], "little") >> 29:
            raise ValueError("unsupported WebP VP8L version")
        return
    raise ValueError("unknown WebP image chunk")


def _validate_wav(data: bytes) -> None:
    chunks = _riff_chunks(data)
    formats = [payload for kind, payload in chunks if kind == b"fmt "]
    samples = [payload for kind, payload in chunks if kind == b"data"]
    if len(formats) != 1 or len(formats[0]) < 16 or not any(samples):
        raise ValueError("WAVE requires fmt and data chunks")
    audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", formats[0]
    )
    if (
        audio_format == 0
        or channels == 0
        or rate == 0
        or byte_rate == 0
        or block_align == 0
        or bits == 0
    ):
        raise ValueError("invalid WAVE fmt chunk")
    if audio_format in (1, 3, 0xFFFE):
        if bits % 8 or block_align != channels * (bits // 8) or byte_rate != rate * block_align:
            raise ValueError("inconsistent WAVE sample framing")
        if audio_format == 0xFFFE:
            extension_size = (
                struct.unpack_from("<H", formats[0], 16)[0] if len(formats[0]) >= 18 else 0
            )
            if len(formats[0]) < 40 or extension_size < 22 or 18 + extension_size > len(formats[0]):
                raise ValueError("truncated extensible WAVE fmt chunk")
            valid_bits = struct.unpack_from("<H", formats[0], 18)[0]
            subformat = bytes(formats[0][24:40])
            if (
                valid_bits == 0
                or valid_bits > bits
                or subformat
                not in (
                    b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq",
                    b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq",
                )
            ):
                raise ValueError("unsupported extensible WAVE subtype")
    elif len(formats[0]) < 18 or 18 + struct.unpack_from("<H", formats[0], 16)[0] > len(formats[0]):
        raise ValueError("truncated compressed WAVE fmt extension")
    if any(len(payload) % block_align for payload in samples):
        raise ValueError("WAVE data is not sample-block aligned")


def _validate_flac(data: bytes) -> None:
    offset = 4
    first = True
    while True:
        if offset + 4 > len(data):
            raise ValueError("truncated FLAC metadata")
        header = data[offset]
        kind = header & 0x7F
        length = int.from_bytes(data[offset + 1 : offset + 4], "big")
        offset += 4
        if offset + length > len(data):
            raise ValueError("truncated FLAC metadata block")
        if first:
            if kind != 0 or length != 34:
                raise ValueError("FLAC must start with 34-byte STREAMINFO")
            _validate_flac_streaminfo(data[offset : offset + length])
        first = False
        offset += length
        if header & 0x80:
            break
    _validate_flac_frame_header(data, offset)


def _validate_flac_streaminfo(streaminfo: bytes) -> None:
    if len(streaminfo) != 34:
        raise ValueError("FLAC STREAMINFO must be 34 bytes")
    minimum = int.from_bytes(streaminfo[:2], "big")
    maximum = int.from_bytes(streaminfo[2:4], "big")
    packed = int.from_bytes(streaminfo[10:18], "big")
    sample_rate = packed >> 44
    bits_per_sample = ((packed >> 36) & 0x1F) + 1
    if (
        minimum < 16
        or maximum < minimum
        or not 1 <= sample_rate <= 655_350
        or not 4 <= bits_per_sample <= 32
    ):
        raise ValueError("invalid FLAC STREAMINFO")


def _flac_crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _utf8_uint_end(data: bytes, offset: int) -> int:
    if offset >= len(data):
        raise ValueError("truncated FLAC coded frame number")
    first = data[offset]
    if first < 0x80:
        return offset + 1
    leading = 0
    marker = 0x80
    while first & marker:
        leading += 1
        marker >>= 1
    if not 2 <= leading <= 7 or offset + leading > len(data):
        raise ValueError("invalid FLAC coded frame number")
    for byte in data[offset + 1 : offset + leading]:
        if byte & 0xC0 != 0x80:
            raise ValueError("invalid FLAC coded frame number")
    return offset + leading


def _validate_flac_frame_header(data: bytes, offset: int) -> None:
    if offset + 7 > len(data) or data[offset] != 0xFF or data[offset + 1] & 0xFE != 0xF8:
        raise ValueError("FLAC has no complete audio frame header")
    block_code = data[offset + 2] >> 4
    rate_code = data[offset + 2] & 0x0F
    channel_code = data[offset + 3] >> 4
    sample_code = (data[offset + 3] >> 1) & 0x07
    if (
        block_code == 0
        or rate_code == 15
        or channel_code > 10
        or sample_code in (3, 7)
        or data[offset + 3] & 1
    ):
        raise ValueError("FLAC frame header uses a reserved field")
    end = _utf8_uint_end(data, offset + 4)
    end += 1 if block_code == 6 else 2 if block_code == 7 else 0
    end += 1 if rate_code == 12 else 2 if rate_code in (13, 14) else 0
    if end + 2 > len(data):
        raise ValueError("truncated FLAC frame header")
    if _flac_crc8(data[offset:end]) != data[end]:
        raise ValueError("FLAC frame header CRC mismatch")


_MP3_BITRATES = {
    (1, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (2, 3): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}


def _validate_mp3(data: bytes) -> None:
    offset = 0
    if data[:3] == b"ID3":
        major = data[3] if len(data) >= 4 else 0
        flags = data[5] if len(data) >= 6 else 0
        allowed_flags = {2: 0xC0, 3: 0xE0, 4: 0xF0}
        if (
            len(data) < 10
            or major not in allowed_flags
            or data[4] == 0xFF
            or flags & ~allowed_flags[major]
            or any(byte & 0x80 for byte in data[6:10])
        ):
            raise ValueError("invalid ID3 header")
        tag_size = sum(
            byte << shift for byte, shift in zip(data[6:10], (21, 14, 7, 0), strict=True)
        )
        offset = 10 + tag_size
        if offset > len(data):
            raise ValueError("truncated ID3 tag")
        if major == 4 and flags & 0x10:
            if (
                offset + 10 > len(data)
                or data[offset : offset + 3] != b"3DI"
                or data[offset + 3 : offset + 6] != data[3:6]
                or data[offset + 6 : offset + 10] != data[6:10]
            ):
                raise ValueError("invalid ID3v2.4 footer")
            offset += 10
    frames = 0
    while offset < len(data):
        if len(data) - offset == 128 and data[offset : offset + 3] == b"TAG":
            offset = len(data)
            break
        if offset + 4 > len(data):
            raise ValueError("truncated MPEG audio frame")
        header = int.from_bytes(data[offset : offset + 4], "big")
        if header >> 21 != 0x7FF:
            raise ValueError("invalid MPEG audio sync")
        version_bits = (header >> 19) & 0x3
        layer_bits = (header >> 17) & 0x3
        bitrate_index = (header >> 12) & 0xF
        rate_index = (header >> 10) & 0x3
        if version_bits == 1 or layer_bits != 1 or bitrate_index in (0, 15) or rate_index == 3:
            raise ValueError("invalid MPEG Audio Layer III header")
        version = 1 if version_bits == 3 else 2
        layer = 4 - layer_bits
        bitrate = _MP3_BITRATES[(version, layer)][bitrate_index] * 1000
        base_rate = (44_100, 48_000, 32_000)[rate_index]
        rate = base_rate if version_bits == 3 else base_rate // (2 if version_bits == 2 else 4)
        padding = (header >> 9) & 1
        if version != 1:
            frame_size = 72 * bitrate // rate + padding
        else:
            frame_size = 144 * bitrate // rate + padding
        if frame_size < 4 or offset + frame_size > len(data):
            raise ValueError("truncated MPEG audio frame payload")
        offset += frame_size
        frames += 1
    if frames == 0 or offset != len(data):
        raise ValueError("MPEG audio has no complete frame")


def _ogg_crc_table() -> tuple[int, ...]:
    values: list[int] = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
                if crc & 0x80000000
                else (crc << 1) & 0xFFFFFFFF
            )
        values.append(crc)
    return tuple(values)


_OGG_CRC_TABLE = _ogg_crc_table()


def _ogg_crc(page: bytes | bytearray) -> int:
    crc = 0
    for byte in page:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC_TABLE[((crc >> 24) ^ byte) & 0xFF]
    return crc


def _validate_ogg(data: bytes) -> None:
    offset = 0
    serial: bytes | None = None
    sequence = 0
    packet_open = False
    identification = bytearray()
    first_packet: bytes | None = None
    saw_eos = False
    while offset < len(data):
        if saw_eos:
            raise ValueError("Ogg contains pages after EOS")
        if offset + 27 > len(data) or data[offset : offset + 4] != b"OggS":
            raise ValueError("invalid or truncated Ogg page")
        if data[offset + 4] != 0:
            raise ValueError("unsupported Ogg version")
        flags = data[offset + 5]
        if flags & ~0x07 or (sequence > 0 and flags & 0x02):
            raise ValueError("invalid Ogg page flags")
        segments = data[offset + 26]
        header_end = offset + 27 + segments
        if header_end > len(data):
            raise ValueError("truncated Ogg segment table")
        lacing = data[offset + 27 : header_end]
        page_end = header_end + sum(lacing)
        if page_end > len(data):
            raise ValueError("truncated Ogg page payload")
        current_serial = data[offset + 14 : offset + 18]
        current_sequence = struct.unpack_from("<I", data, offset + 18)[0]
        if serial is None:
            if not flags & 0x02 or flags & 0x01 or current_sequence != 0:
                raise ValueError("Ogg must start with an independent BOS page")
            serial = current_serial
        elif current_serial != serial or current_sequence != sequence:
            raise ValueError("Ogg stream identity or sequence changed")
        expected_crc = struct.unpack_from("<I", data, offset + 22)[0]
        page = bytearray(data[offset:page_end])
        page[22:26] = bytes(4)
        if _ogg_crc(page) != expected_crc:
            raise ValueError("Ogg page checksum mismatch")
        body_offset = header_end
        if packet_open != bool(flags & 0x01):
            raise ValueError("Ogg continuation flag does not match packet framing")
        for size in lacing:
            if first_packet is None:
                identification.extend(data[body_offset : body_offset + size])
                if len(identification) > 65_536:
                    raise ValueError("Ogg identification packet exceeds bound")
            body_offset += size
            if size < 255:
                if first_packet is None:
                    first_packet = bytes(identification)
                    identification.clear()
                packet_open = False
            else:
                packet_open = True
        saw_eos = bool(flags & 0x04)
        sequence += 1
        offset = page_end
    if packet_open or not saw_eos or first_packet is None:
        raise ValueError("Ogg stream is incomplete")
    _validate_ogg_identification(first_packet)


def _validate_ogg_identification(packet: bytes) -> None:
    if packet.startswith(b"OpusHead"):
        if len(packet) < 19 or not 1 <= packet[8] <= 15 or packet[9] == 0:
            raise ValueError("truncated Opus identification packet")
        channels = packet[9]
        mapping = packet[18]
        if mapping == 0:
            valid_mapping = channels <= 2
        elif len(packet) >= 21 + channels:
            streams = packet[19]
            coupled = packet[20]
            valid_mapping = (
                streams > 0
                and coupled <= streams
                and streams + coupled <= 255
                and all(
                    value == 255 or value < streams + coupled
                    for value in packet[21 : 21 + channels]
                )
            )
        else:
            valid_mapping = False
        expected_size = 19 if mapping == 0 else 21 + channels
        if not valid_mapping or (packet[8] == 1 and len(packet) != expected_size):
            raise ValueError("invalid Opus channel mapping")
        return
    if packet.startswith(b"\x01vorbis"):
        if (
            len(packet) < 30
            or packet[7:11] != bytes(4)
            or packet[11] == 0
            or int.from_bytes(packet[12:16], "little") == 0
            or not 6 <= packet[28] & 0x0F <= packet[28] >> 4 <= 13
            or packet[29] != 1
        ):
            raise ValueError("invalid Vorbis identification packet")
        return
    if packet.startswith(b"Speex   "):
        if (
            len(packet) < 80
            or int.from_bytes(packet[32:36], "little") < 80
            or int.from_bytes(packet[32:36], "little") > len(packet)
            or int.from_bytes(packet[36:40], "little") == 0
            or int.from_bytes(packet[48:52], "little") == 0
            or int.from_bytes(packet[56:60], "little") == 0
            or int.from_bytes(packet[64:68], "little") == 0
            or packet[72:80] != bytes(8)
        ):
            raise ValueError("invalid Speex identification packet")
        return
    if packet.startswith(b"\x7fFLAC"):
        if (
            len(packet) < 51
            or packet[5:7] != b"\x01\x00"
            or int.from_bytes(packet[7:9], "big") == 0
            or packet[9:13] != b"fLaC"
            or packet[13] & 0x7F != 0
            or int.from_bytes(packet[14:17], "big") != 34
        ):
            raise ValueError("invalid Ogg FLAC identification packet")
        _validate_flac_streaminfo(packet[17:51])
        return
    raise ValueError("Ogg stream is not a recognized audio codec")


_MP4_BRANDS = frozenset(
    {
        b"avc1",
        b"dash",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"isom",
        b"M4A ",
        b"mp41",
        b"mp42",
    }
)


def _validate_mp4(data: bytes) -> str:
    boxes = _bmff_boxes(data, 0, len(data))
    if not boxes or boxes[0][0] != b"ftyp":
        raise ValueError("MP4 must start with ftyp")
    _, ftyp_start, ftyp_end = boxes[0]
    if ftyp_end - ftyp_start > 1024:
        raise ValueError("MP4 ftyp exceeds classification bound")
    ftyp = data[ftyp_start:ftyp_end]
    if len(ftyp) < 8 or (len(ftyp) - 8) % 4:
        raise ValueError("invalid MP4 ftyp")
    brands = {ftyp[:4], *(ftyp[index : index + 4] for index in range(8, len(ftyp), 4))}
    if not brands & _MP4_BRANDS:
        raise ValueError("unsupported MP4 brand")
    moov = [box for box in boxes if box[0] == b"moov"]
    if len(moov) != 1 or not any(kind == b"mdat" for kind, _, _ in boxes):
        raise ValueError("MP4 requires one moov and an mdat")
    movie = _bmff_boxes(data, moov[0][1], moov[0][2])
    movie_headers = [box for box in movie if box[0] == b"mvhd"]
    if len(movie_headers) != 1 or not _complete_full_box(data, movie_headers[0], 100, 112):
        raise ValueError("MP4 requires one complete mvhd")
    has_video = False
    has_audio = False
    for kind, trak_start, trak_end in movie:
        if kind != b"trak":
            continue
        track = _bmff_boxes(data, trak_start, trak_end)
        track_headers = [box for box in track if box[0] == b"tkhd"]
        media = [box for box in track if box[0] == b"mdia"]
        if len(track_headers) != 1 or not _complete_full_box(data, track_headers[0], 84, 96):
            raise ValueError("MP4 track requires one complete tkhd")
        if len(media) != 1:
            raise ValueError("MP4 track requires one mdia")
        media_boxes = _bmff_boxes(data, media[0][1], media[0][2])
        media_headers = [box for box in media_boxes if box[0] == b"mdhd"]
        handlers = [box for box in media_boxes if box[0] == b"hdlr"]
        media_info = [box for box in media_boxes if box[0] == b"minf"]
        if (
            len(media_headers) != 1
            or not _complete_full_box(data, media_headers[0], 24, 36)
            or len(handlers) != 1
            or handlers[0][2] - handlers[0][1] < 24
            or len(media_info) != 1
        ):
            raise ValueError("MP4 mdia hierarchy is incomplete or ambiguous")
        handler = data[handlers[0][1] + 8 : handlers[0][1] + 12]
        if handler == b"vide":
            has_video |= _mp4_has_visual_sample_entry(data, media_info[0][1], media_info[0][2])
        elif handler == b"soun":
            has_audio |= _mp4_has_audio_sample_entry(data, media_info[0][1], media_info[0][2])
    if has_video:
        return "video/mp4"
    if has_audio:
        return "audio/mp4"
    raise ValueError("MP4 has no supported audio or video track")


def _mp4_has_visual_sample_entry(data: bytes, start: int, end: int) -> bool:
    entries = _mp4_sample_entries(data, start, end, "video")
    visual_types = frozenset(
        {b"avc1", b"avc3", b"hvc1", b"hev1", b"vp08", b"vp09", b"av01", b"mp4v"}
    )
    return any(
        kind in visual_types
        and entry_end - entry_start >= 78
        and int.from_bytes(data[entry_start + 6 : entry_start + 8], "big") > 0
        and int.from_bytes(data[entry_start + 24 : entry_start + 26], "big") > 0
        and int.from_bytes(data[entry_start + 26 : entry_start + 28], "big") > 0
        for kind, entry_start, entry_end in entries
    )


def _mp4_has_audio_sample_entry(data: bytes, start: int, end: int) -> bool:
    entries = _mp4_sample_entries(data, start, end, "audio")
    return any(
        kind == b"mp4a"
        and entry_end - entry_start >= 28
        and int.from_bytes(data[entry_start + 6 : entry_start + 8], "big") > 0
        and int.from_bytes(data[entry_start + 16 : entry_start + 18], "big") > 0
        and int.from_bytes(data[entry_start + 18 : entry_start + 20], "big") > 0
        and int.from_bytes(data[entry_start + 24 : entry_start + 28], "big") > 0
        and _mp4_has_aac_descriptor(data, entry_start + 28, entry_end)
        for kind, entry_start, entry_end in entries
    )


def _mp4_has_aac_descriptor(data: bytes, start: int, end: int) -> bool:
    boxes = _bmff_boxes(data, start, end)
    descriptors = [box for box in boxes if box[0] == b"esds"]
    if len(descriptors) != 1:
        raise ValueError("MP4 AAC entry requires one esds")
    _, descriptor_start, descriptor_end = descriptors[0]
    if descriptor_end - descriptor_start < 4 or data[
        descriptor_start : descriptor_start + 4
    ] != bytes(4):
        raise ValueError("invalid MP4 esds full box")
    top_level = _mp4_descriptors(data, descriptor_start + 4, descriptor_end)
    if len(top_level) != 1 or top_level[0][0] != 0x03:
        raise ValueError("MP4 esds requires one elementary stream descriptor")
    _, stream_start, stream_end = top_level[0]
    if stream_end - stream_start < 3:
        raise ValueError("truncated MP4 elementary stream descriptor")
    flags = data[stream_start + 2]
    child_start = stream_start + 3
    if flags & 0x80:
        child_start += 2
    if flags & 0x40:
        if child_start >= stream_end:
            raise ValueError("truncated MP4 elementary stream URL")
        child_start += 1 + data[child_start]
    if flags & 0x20:
        child_start += 2
    stream_children = _mp4_descriptors(data, child_start, stream_end)
    decoder_configs = [item for item in stream_children if item[0] == 0x04]
    if len(decoder_configs) != 1:
        raise ValueError("MP4 esds requires one decoder config")
    _, config_start, config_end = decoder_configs[0]
    if (
        config_end - config_start < 13
        or data[config_start] != 0x40
        or data[config_start + 1] >> 2 != 0x05
    ):
        raise ValueError("MP4 decoder config is not MPEG-4 audio")
    config_children = _mp4_descriptors(data, config_start + 13, config_end)
    specific_configs = [item for item in config_children if item[0] == 0x05]
    if len(specific_configs) != 1:
        raise ValueError("MP4 decoder config requires one AudioSpecificConfig")
    _, specific_start, specific_end = specific_configs[0]
    _validate_aac_audio_specific_config(data[specific_start:specific_end])
    return True


def _mp4_descriptors(data: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    if start > end:
        raise ValueError("invalid MP4 descriptor bounds")
    descriptors: list[tuple[int, int, int]] = []
    offset = start
    while offset < end:
        kind = data[offset]
        offset += 1
        size = 0
        for _index in range(4):
            if offset >= end:
                raise ValueError("truncated MP4 descriptor length")
            part = data[offset]
            offset += 1
            size = (size << 7) | (part & 0x7F)
            if not part & 0x80:
                break
        else:
            raise ValueError("MP4 descriptor length exceeds four bytes")
        payload_end = offset + size
        if payload_end > end:
            raise ValueError("truncated MP4 descriptor")
        descriptors.append((kind, offset, payload_end))
        if len(descriptors) > 1_000:
            raise ValueError("MP4 contains too many descriptors")
        offset = payload_end
    return descriptors


def _validate_aac_audio_specific_config(config: bytes) -> None:
    if len(config) < 2 or len(config) > 64:
        raise ValueError("invalid AAC AudioSpecificConfig length")

    bits = int.from_bytes(config, "big")
    bit_count = len(config) * 8
    offset = 0

    def read(width: int) -> int:
        nonlocal offset
        if offset + width > bit_count:
            raise ValueError("truncated AAC AudioSpecificConfig")
        value = (bits >> (bit_count - offset - width)) & ((1 << width) - 1)
        offset += width
        return value

    def read_object_type() -> int:
        value = read(5)
        return 32 + read(6) if value == 31 else value

    def read_frequency() -> None:
        index = read(4)
        if index == 15:
            if read(24) == 0:
                raise ValueError("invalid AAC explicit sampling frequency")
        elif index > 12:
            raise ValueError("invalid AAC sampling frequency index")

    object_type = read_object_type()
    read_frequency()
    if read(4) == 0:
        raise ValueError("AAC program config elements are not supported")
    if object_type in {5, 29}:
        read_frequency()
        object_type = read_object_type()
    if object_type not in {1, 2, 3, 4, 6, 17, 19, 20, 23, 39}:
        raise ValueError("MP4 AudioSpecificConfig is not AAC")


def _mp4_sample_entries(
    data: bytes, start: int, end: int, track_kind: str
) -> list[tuple[bytes, int, int]]:
    minf = _bmff_boxes(data, start, end)
    sample_tables = [box for box in minf if box[0] == b"stbl"]
    if len(sample_tables) != 1:
        raise ValueError(f"MP4 {track_kind} minf requires one stbl")
    table = _bmff_boxes(data, sample_tables[0][1], sample_tables[0][2])
    descriptions = [box for box in table if box[0] == b"stsd"]
    if len(descriptions) != 1:
        raise ValueError(f"MP4 {track_kind} stbl requires one stsd")
    _, payload_start, payload_end = descriptions[0]
    if payload_end - payload_start < 8:
        raise ValueError("truncated MP4 stsd")
    count = struct.unpack_from(">I", data, payload_start + 4)[0]
    entries = _bmff_boxes(data, payload_start + 8, payload_end)
    if len(entries) != count:
        raise ValueError("MP4 stsd entry count does not match framing")
    return entries


def _complete_full_box(
    data: bytes,
    box: tuple[bytes, int, int],
    version_zero_size: int,
    version_one_size: int,
) -> bool:
    _, start, end = box
    if start >= end:
        return False
    required = (
        version_zero_size if data[start] == 0 else version_one_size if data[start] == 1 else 0
    )
    return required > 0 and end - start >= required


def _validate_webm(data: bytes) -> str:
    top = _ebml_elements(data, 0, len(data))
    if len(top) < 2 or top[0][0] != 0x1A45DFA3 or top[1][0] != 0x18538067:
        raise ValueError("WebM requires EBML header followed by Segment")
    if top[1][2] != len(data) or len(top) != 2:
        raise ValueError("WebM Segment must contain the remaining bytes")
    header = _ebml_elements(data, top[0][1], top[0][2])
    doc_types = [
        data[start:end] for kind, start, end in header if kind == 0x4282 and end - start <= 32
    ]
    if doc_types != [b"webm"]:
        raise ValueError("EBML DocType is not webm")
    segment = _ebml_elements(data, top[1][1], top[1][2])
    infos = [entry for entry in segment if entry[0] == 0x1549A966]
    tracks = [entry for entry in segment if entry[0] == 0x1654AE6B]
    clusters = [entry for entry in segment if entry[0] == 0x1F43B675]
    if len(infos) != 1 or len(tracks) != 1 or not clusters:
        raise ValueError("WebM requires one Info, one Tracks, and a Cluster")
    info = _ebml_elements(data, infos[0][1], infos[0][2])
    if not any(kind == 0x4D80 and end > start for kind, start, end in info) or not any(
        kind == 0x5741 and end > start for kind, start, end in info
    ):
        raise ValueError("WebM Info requires muxing and writing applications")
    for _, cluster_start, cluster_end in clusters:
        cluster = _ebml_elements(data, cluster_start, cluster_end)
        timestamps = [(start, end) for kind, start, end in cluster if kind == 0xE7]
        if len(timestamps) != 1 or not 1 <= timestamps[0][1] - timestamps[0][0] <= 8:
            raise ValueError("WebM Cluster requires a timestamp")
    has_audio = False
    has_video = False
    seen_numbers: set[int] = set()
    seen_uids: set[int] = set()
    for kind, entry_start, entry_end in _ebml_elements(data, tracks[0][1], tracks[0][2]):
        if kind != 0xAE:
            continue
        fields = _ebml_elements(data, entry_start, entry_end)
        numbers = [
            int.from_bytes(data[start:end], "big") for field, start, end in fields if field == 0xD7
        ]
        uids = [
            int.from_bytes(data[start:end], "big")
            for field, start, end in fields
            if field == 0x73C5
        ]
        types = [
            int.from_bytes(data[start:end], "big") for field, start, end in fields if field == 0x83
        ]
        codecs = [
            data[start:end] for field, start, end in fields if field == 0x86 and end - start <= 64
        ]
        videos = [(start, end) for field, start, end in fields if field == 0xE0]
        audios = [(start, end) for field, start, end in fields if field == 0xE1]
        if not (
            len(numbers) == len(uids) == len(types) == len(codecs) == 1
            and numbers[0] > 0
            and uids[0] > 0
        ):
            raise ValueError("WebM TrackEntry identity is incomplete or ambiguous")
        if numbers[0] in seen_numbers or uids[0] in seen_uids:
            raise ValueError("WebM track identity is duplicated")
        seen_numbers.add(numbers[0])
        seen_uids.add(uids[0])
        if types[0] == 2:
            if codecs[0] not in (b"A_OPUS", b"A_VORBIS") or len(audios) != 1:
                raise ValueError("WebM audio TrackEntry has no supported codec or Audio")
            audio = _ebml_elements(data, audios[0][0], audios[0][1])
            rates = [(start, end) for field, start, end in audio if field == 0xB5]
            channels = [
                int.from_bytes(data[start:end], "big")
                for field, start, end in audio
                if field == 0x9F
            ]
            if len(rates) != 1 or rates[0][1] - rates[0][0] not in (4, 8):
                raise ValueError("WebM audio sampling frequency is incomplete or invalid")
            sampling_frequency = struct.unpack(
                ">f" if rates[0][1] - rates[0][0] == 4 else ">d",
                data[rates[0][0] : rates[0][1]],
            )[0]
            if (
                not (sampling_frequency > 0 and math.isfinite(sampling_frequency))
                or len(channels) != 1
                or channels[0] == 0
            ):
                raise ValueError("WebM audio properties are incomplete or invalid")
            has_audio = True
            continue
        if types[0] != 1:
            continue
        if (
            codecs[0]
            not in (
                b"V_VP8",
                b"V_VP9",
                b"V_AV1",
            )
            or len(videos) != 1
        ):
            raise ValueError("WebM video TrackEntry has no supported codec or Video")
        video = _ebml_elements(data, videos[0][0], videos[0][1])
        widths = [
            int.from_bytes(data[start:end], "big") for field, start, end in video if field == 0xB0
        ]
        heights = [
            int.from_bytes(data[start:end], "big") for field, start, end in video if field == 0xBA
        ]
        if len(widths) != 1 or len(heights) != 1 or widths[0] == 0 or heights[0] == 0:
            raise ValueError("WebM video dimensions are incomplete or invalid")
        has_video = True
    if has_video:
        return "video/webm"
    if has_audio:
        return "audio/webm"
    raise ValueError("WebM has no declared audio or video track")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"GLB JSON chunk holds non-JSON constant {value}")


def _validate_glb(data: bytes) -> None:
    if len(data) < 12:
        raise ValueError("truncated GLB header")
    magic, container_version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("invalid GLB magic")
    if container_version != 2:
        raise ValueError("unsupported GLB container version")
    if declared_length != len(data):
        raise ValueError("GLB declared length does not match bytes")
    chunks: list[tuple[bytes, int, int]] = []
    offset = 12
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        start = offset + 8
        end = start + chunk_length
        if chunk_length % 4:
            raise ValueError("GLB chunk length is not 4-byte aligned")
        if end > len(data):
            raise ValueError("truncated GLB chunk payload")
        chunks.append((chunk_type, start, end))
        if len(chunks) > _STRUCTURE_ITEM_LIMIT:
            raise ValueError("GLB contains too many chunks")
        offset = end
    types = [chunk_type for chunk_type, _, _ in chunks]
    if not chunks or types[0] != b"JSON" or chunks[0][1] == chunks[0][2]:
        raise ValueError("GLB must start with a non-empty JSON chunk")
    if types.count(b"JSON") != 1:
        raise ValueError("GLB JSON chunk must not repeat")
    bin_indexes = [index for index, kind in enumerate(types) if kind == b"BIN\x00"]
    if len(bin_indexes) > 1 or (bin_indexes and bin_indexes[0] != 1):
        raise ValueError("GLB BIN chunk must directly follow the JSON chunk")
    json_start, json_end = chunks[0][1], chunks[0][2]
    if json_end - json_start > _GLB_JSON_LIMIT:
        raise ValueError("GLB JSON chunk exceeds classification bound")
    try:
        loaded: object = json.loads(
            bytes(data[json_start:json_end]).decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except RecursionError:
        raise ValueError("GLB JSON chunk exceeds nesting bound") from None
    if not isinstance(loaded, dict):
        raise ValueError("GLB JSON chunk must hold an object")
    document = cast("dict[str, object]", loaded)
    asset = document.get("asset")
    version = cast("dict[str, object]", asset).get("version") if isinstance(asset, dict) else None
    parts = version.split(".") if isinstance(version, str) else []
    if len(parts) != 2 or parts[0] != "2" or not (parts[1].isascii() and parts[1].isdigit()):
        raise ValueError("GLB asset version is not glTF 2")
    buffers = document.get("buffers", [])
    if not isinstance(buffers, list):
        raise ValueError("GLB buffers must be a list")
    first_buffer: object = cast("list[object]", buffers)[0] if buffers else None
    embedded: dict[str, object] | None = None
    if isinstance(first_buffer, dict):
        first = cast("dict[str, object]", first_buffer)
        if "uri" not in first:
            embedded = first
    if bin_indexes:
        bin_start, bin_end = chunks[1][1], chunks[1][2]
        bin_length = bin_end - bin_start
        raw_declared = embedded.get("byteLength") if embedded is not None else None
        if isinstance(raw_declared, bool) or not isinstance(raw_declared, int | float):
            raise ValueError("GLB buffer byteLength must be a number")
        if isinstance(raw_declared, float) and not raw_declared.is_integer():
            raise ValueError("GLB buffer byteLength must be integral")
        declared = int(raw_declared)
        if declared < 1:
            raise ValueError("GLB buffer byteLength must be at least one")
        if not 0 <= bin_length - declared < 4:
            raise ValueError("GLB BIN chunk does not match the declared buffer")
        if bytes(data[bin_start + declared : bin_end]) != bytes(bin_length - declared):
            raise ValueError("GLB BIN chunk padding must be zero")
    elif embedded is not None:
        raise ValueError("GLB declares an embedded buffer but has no BIN chunk")


_PLY_HEADER_LIMIT = 64 * 1024
_PLY_SCALAR_SIZES = {
    "char": 1,
    "uchar": 1,
    "short": 2,
    "ushort": 2,
    "int": 4,
    "uint": 4,
    "float": 4,
    "double": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "float64": 8,
}


def _validate_ply(data: Any) -> None:
    """Validate a binary little-endian PLY point/splat container.

    The accepted shape is the 3D gaussian splat interchange layout: scalar
    vertex properties only (no list properties, so no mesh face elements),
    an x/y/z float position, and element data sized exactly by the header."""
    terminator = bytes(data[:_PLY_HEADER_LIMIT]).find(b"end_header")
    if terminator < 0:
        raise ValueError("PLY header has no end_header terminator")
    body = terminator + len(b"end_header")
    if data[body : body + 2] == b"\r\n":
        body += 2
    elif data[body : body + 1] == b"\n":
        body += 1
    else:
        raise ValueError("PLY end_header must terminate with a newline")
    try:
        header = bytes(data[:terminator]).decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("PLY header must be ASCII") from None
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if not lines or lines[0] != "ply":
        raise ValueError("invalid PLY magic line")
    if "format binary_little_endian 1.0" not in lines[1:]:
        raise ValueError("PLY format must be binary_little_endian 1.0")
    # (name, count, stride, has x/y/z floats) per declared element, in order.
    elements: list[list[object]] = []
    items = 0
    for line in lines[1:]:
        parts = line.split()
        items += 1
        if items > _STRUCTURE_ITEM_LIMIT:
            raise ValueError("PLY header exceeds classification bound")
        if parts[0] in ("comment", "obj_info", "format"):
            continue
        if parts[0] == "element":
            if len(parts) != 3 or not (parts[2].isascii() and parts[2].isdigit()):
                raise ValueError("invalid PLY element declaration")
            elements.append([parts[1], int(parts[2]), 0, set()])
        elif parts[0] == "property":
            if not elements:
                raise ValueError("PLY property declared before any element")
            if len(parts) != 3:
                raise ValueError("PLY vertex properties must be scalar (no lists)")
            size = _PLY_SCALAR_SIZES.get(parts[1])
            if size is None:
                raise ValueError(f"unsupported PLY property type {parts[1]}")
            elements[-1][2] = cast(int, elements[-1][2]) + size
            if parts[1] in ("float", "float32") and parts[2] in ("x", "y", "z"):
                cast("set[str]", elements[-1][3]).add(parts[2])
        else:
            raise ValueError(f"unsupported PLY header keyword {parts[0]}")
    if not elements:
        raise ValueError("PLY declares no elements")
    name, count, stride, axes = elements[0]
    if name != "vertex" or cast(int, count) < 1:
        raise ValueError("PLY must declare a non-empty vertex element first")
    if cast("set[str]", axes) != {"x", "y", "z"}:
        raise ValueError("PLY vertex element must carry float x, y, and z")
    total = 0
    for _, count, stride, _ in elements:
        if cast(int, stride) == 0 and cast(int, count) > 0:
            raise ValueError("PLY element declares no properties")
        total += cast(int, count) * cast(int, stride)
    if total != len(data) - body:
        raise ValueError("PLY element data does not match the declared header")
