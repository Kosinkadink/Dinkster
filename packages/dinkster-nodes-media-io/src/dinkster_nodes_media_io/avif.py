"""Bounded AVIF encoding with ComfyUI-compatible EXIF metadata."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from fractions import Fraction
from typing import Any, BinaryIO

import av
import numpy as np
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
from PIL.Image import Exif

_COLOR_PROPERTIES = {
    "sRGB": (ColorPrimaries.BT709, ColorTrc.IEC61966_2_1, 1),
    "HDR": (ColorPrimaries.BT2020, ColorTrc.ARIB_STD_B67, 9),
    "HDR PQ": (ColorPrimaries.BT2020, ColorTrc.SMPTE2084, 9),
}


def _box(box_type: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    if size > 0xFFFFFFFF:
        raise ValueError("AVIF metadata box is too large")
    return struct.pack(">I4s", size, box_type) + payload


def _boxes(data: bytes | bytearray, start: int, end: int) -> list[tuple[int, int, bytes, int]]:
    boxes: list[tuple[int, int, bytes, int]] = []
    position = start
    while position < end:
        if position + 8 > end:
            raise ValueError("invalid AVIF box structure")
        size, box_type = struct.unpack_from(">I4s", data, position)
        header_size = 8
        if size == 1:
            if position + 16 > end:
                raise ValueError("invalid AVIF extended-size box")
            size = struct.unpack_from(">Q", data, position + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise ValueError("invalid AVIF box size")
        boxes.append((position, size, box_type, header_size))
        position += size
    return boxes


def _exif(metadata: dict[str, object]) -> bytes:
    exif = Exif()
    if "prompt" in metadata:
        exif[0x0110] = f"prompt:{json.dumps(metadata['prompt'])}"
    next_tag = 0x010F
    for key, value in metadata.items():
        if key == "prompt":
            continue
        exif[next_tag] = f"{key}:{json.dumps(value)}"
        next_tag -= 1
    return b"\x00\x00\x00\x00" + exif.tobytes()[6:]


def _add_exif_item(meta: bytes, exif_offset: int, exif_length: int, offset_delta: int) -> bytes:
    if meta[4:8] != b"meta" or len(meta) < 12:
        raise ValueError("AVIF metadata requires a valid meta box")
    children = _boxes(meta, 12, len(meta))
    child_by_type = {
        box_type: (position, size, header_size)
        for position, size, box_type, header_size in children
    }
    if not all(box_type in child_by_type for box_type in (b"pitm", b"iloc", b"iinf")):
        raise ValueError("AVIF metadata boxes are incomplete")

    pitm_position, pitm_size, _ = child_by_type[b"pitm"]
    pitm = meta[pitm_position : pitm_position + pitm_size]
    if pitm[8] != 0:
        raise ValueError("unsupported AVIF primary-item format")
    primary_item_id = struct.unpack_from(">H", pitm, 12)[0]

    iloc_position, iloc_size, _ = child_by_type[b"iloc"]
    iloc = bytearray(meta[iloc_position : iloc_position + iloc_size])
    if iloc[8] != 0 or iloc[12] != 0x44 or iloc[13] != 0:
        raise ValueError("unsupported AVIF item-location format")
    item_count = struct.unpack_from(">H", iloc, 14)[0]
    cursor = 16
    item_ids: list[int] = []
    for _ in range(item_count):
        item_id, _, extent_count = struct.unpack_from(">HHH", iloc, cursor)
        cursor += 6
        item_ids.append(item_id)
        for _ in range(extent_count):
            extent_offset = struct.unpack_from(">I", iloc, cursor)[0]
            struct.pack_into(">I", iloc, cursor, extent_offset + offset_delta)
            cursor += 8
    if cursor != len(iloc):
        raise ValueError("unsupported AVIF item-location entries")
    exif_item_id = max(item_ids) + 1
    if exif_item_id > 0xFFFF or exif_offset > 0xFFFFFFFF or exif_length > 0xFFFFFFFF:
        raise ValueError("AVIF metadata exceeds 32-bit item limits")
    struct.pack_into(">H", iloc, 14, item_count + 1)
    iloc.extend(struct.pack(">HHHII", exif_item_id, 0, 1, exif_offset, exif_length))
    struct.pack_into(">I", iloc, 0, len(iloc))

    iinf_position, iinf_size, _ = child_by_type[b"iinf"]
    iinf = bytearray(meta[iinf_position : iinf_position + iinf_size])
    if iinf[8] != 0:
        raise ValueError("unsupported AVIF item-information format")
    iinf_count = struct.unpack_from(">H", iinf, 12)[0]
    struct.pack_into(">H", iinf, 12, iinf_count + 1)
    infe_payload = b"\x02\x00\x00\x00" + struct.pack(">HH4s", exif_item_id, 0, b"Exif") + b"\x00"
    iinf.extend(_box(b"infe", infe_payload))
    struct.pack_into(">I", iinf, 0, len(iinf))

    cdsc = _box(b"cdsc", struct.pack(">HHH", exif_item_id, 1, primary_item_id))
    if b"iref" in child_by_type:
        iref_position, iref_size, _ = child_by_type[b"iref"]
        iref = bytearray(meta[iref_position : iref_position + iref_size])
        if iref[8] != 0:
            raise ValueError("unsupported AVIF item-reference format")
        iref.extend(cdsc)
        struct.pack_into(">I", iref, 0, len(iref))
    else:
        iref = _box(b"iref", b"\x00\x00\x00\x00" + cdsc)

    output = bytearray(meta[:12])
    for position, size, box_type, _ in children:
        if box_type == b"iloc":
            output.extend(iloc)
        elif box_type == b"iinf":
            output.extend(iinf)
            if b"iref" not in child_by_type:
                output.extend(iref)
        elif box_type == b"iref":
            output.extend(iref)
        else:
            output.extend(meta[position : position + size])
    struct.pack_into(">I", output, 0, len(output))
    return bytes(output)


def _adjust_chunk_offsets(moov: bytes, offset_delta: int) -> bytes:
    output = bytearray(moov)
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}

    def adjust(start: int, end: int) -> None:
        for position, size, box_type, header_size in _boxes(output, start, end):
            if box_type in containers:
                adjust(position + header_size, position + size)
            elif box_type in (b"stco", b"co64"):
                entry_size = 4 if box_type == b"stco" else 8
                entry_count = struct.unpack_from(">I", output, position + header_size + 4)[0]
                cursor = position + header_size + 8
                if cursor + entry_count * entry_size != position + size:
                    raise ValueError("invalid AVIF chunk-offset table")
                value_format = ">I" if entry_size == 4 else ">Q"
                for _ in range(entry_count):
                    value = struct.unpack_from(value_format, output, cursor)[0]
                    struct.pack_into(value_format, output, cursor, value + offset_delta)
                    cursor += entry_size

    adjust(0, len(output))
    return bytes(output)


def _copy(source: BinaryIO, destination: BinaryIO, size: int) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("unexpected end of AVIF file")
        destination.write(chunk)
        remaining -= len(chunk)


def inject_metadata(path: str, metadata: dict[str, object]) -> None:
    if not metadata:
        return
    exif = _exif(metadata)
    file_size = os.path.getsize(path)
    top_level_boxes: list[tuple[int, int, bytes, int, int]] = []
    with open(path, "rb") as source:
        position = 0
        while position < file_size:
            source.seek(position)
            header = source.read(8)
            if len(header) != 8:
                raise ValueError("invalid AVIF box header")
            size, box_type = struct.unpack(">I4s", header)
            header_size = 8
            size_field = size
            if size == 1:
                extended_size = source.read(8)
                if len(extended_size) != 8:
                    raise ValueError("invalid AVIF extended-size box")
                size = struct.unpack(">Q", extended_size)[0]
                header_size = 16
            elif size == 0:
                size = file_size - position
            if size < header_size or position + size > file_size:
                raise ValueError("invalid AVIF top-level box size")
            top_level_boxes.append((position, size, box_type, header_size, size_field))
            position += size

        meta_box = next((box for box in top_level_boxes if box[2] == b"meta"), None)
        mdat_box = next((box for box in top_level_boxes if box[2] == b"mdat"), None)
        if (
            meta_box is None
            or mdat_box is None
            or mdat_box[0] + mdat_box[1] != file_size
            or meta_box[0] > mdat_box[0]
        ):
            raise ValueError("unsupported AVIF file layout for metadata")
        source.seek(meta_box[0])
        meta = source.read(meta_box[1])

    provisional_meta = _add_exif_item(meta, 0, len(exif), 0)
    offset_delta = len(provisional_meta) - len(meta)
    exif_offset = mdat_box[0] + mdat_box[1] + offset_delta
    updated_meta = _add_exif_item(meta, exif_offset, len(exif), offset_delta)

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path) or "."
    )
    try:
        with os.fdopen(descriptor, "wb") as destination, open(path, "rb") as source:
            for position, size, box_type, header_size, size_field in top_level_boxes:
                source.seek(position)
                if box_type == b"ftyp":
                    ftyp = bytearray(source.read(size))
                    if bytes(ftyp[8:12]) == b"avis" and b"avif" in ftyp[16:]:
                        ftyp[8:12] = b"avif"
                    destination.write(ftyp)
                elif box_type == b"meta":
                    destination.write(updated_meta)
                elif box_type == b"moov":
                    destination.write(_adjust_chunk_offsets(source.read(size), offset_delta))
                elif box_type == b"mdat":
                    new_size = size + len(exif)
                    if size_field == 1:
                        destination.write(struct.pack(">I4sQ", 1, b"mdat", new_size))
                    elif size_field == 0:
                        destination.write(struct.pack(">I4s", 0, b"mdat"))
                    elif new_size <= 0xFFFFFFFF:
                        destination.write(struct.pack(">I4s", new_size, b"mdat"))
                    else:
                        raise ValueError("AVIF media-data box exceeds its 32-bit size field")
                    source.seek(position + header_size)
                    _copy(source, destination, size - header_size)
                    destination.write(exif)
                else:
                    _copy(source, destination, size)
        os.chmod(temporary, os.stat(path).st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _set_color(target: Any, color_space: str) -> None:
    primaries, transfer, matrix = _COLOR_PROPERTIES[color_space]
    target.color_primaries = primaries
    target.color_trc = transfer
    target.colorspace = matrix
    target.color_range = ColorRange.MPEG


def _frame(image: np.ndarray, bit_depth: str, color_space: str, pixel_format: str) -> av.VideoFrame:
    channels = image.shape[-1]
    if channels not in (1, 3):
        raise ValueError(
            "AVIF supports grayscale and RGB images; the SVT-AV1 encoder does not support alpha"
        )
    if bit_depth == "10-bit YUV420":
        values = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
        source_format = "gray16le" if channels == 1 else "rgb48le"
    else:
        values = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        source_format = "gray" if channels == 1 else "rgb24"
    if channels == 1:
        values = values[..., 0]
    frame = av.VideoFrame.from_ndarray(values, format=source_format)
    frame = frame.reformat(format=pixel_format, dst_colorspace=_COLOR_PROPERTIES[color_space][2])
    _set_color(frame, color_space)
    return frame


def encode_avif(
    images: np.ndarray,
    *,
    bit_depth: str,
    color_space: str,
    crf: int,
    fps: float = 1.0,
    loop: int | None = None,
    metadata: dict[str, object] | None = None,
    limit: int,
) -> bytes:
    if bit_depth == "auto":
        bit_depth = "10-bit YUV420" if color_space in ("HDR", "HDR PQ") else "8-bit YUV420"
    if bit_depth not in ("8-bit YUV420", "10-bit YUV420"):
        raise ValueError("invalid AVIF bit depth")
    if color_space not in _COLOR_PROPERTIES:
        raise ValueError("invalid AVIF input color space")
    if type(crf) is not int or not 1 <= crf <= 63:
        raise ValueError("AVIF crf must be an integer in 1..63")
    rate = float(fps)
    if not np.isfinite(rate) or not 0.01 <= rate <= 1000.0:
        raise ValueError("AVIF fps must be finite and in 0.01..1000")
    if loop is not None and (type(loop) is not int or not 0 <= loop <= 1000):
        raise ValueError("AVIF loop must be an integer in 0..1000")
    pixel_format = "yuv420p10le" if bit_depth == "10-bit YUV420" else "yuv420p"
    options = {"loop": str(loop)} if loop is not None else None
    descriptor, path = tempfile.mkstemp(suffix=".avif")
    os.close(descriptor)
    try:
        with av.open(path, mode="w", format="avif", options=options) as container:
            output_container: Any = container
            stream = output_container.add_stream(
                "libsvtav1", rate=Fraction(round(rate * 1000), 1000)
            )
            stream.width = int(images.shape[2])
            stream.height = int(images.shape[1])
            stream.pix_fmt = pixel_format
            stream.options = {
                "crf": str(crf),
                "preset": "8",
            }
            _set_color(stream.codec_context, color_space)
            for image in images:
                frame = _frame(image, bit_depth, color_space, pixel_format)
                for packet in stream.encode(frame):
                    output_container.mux(packet)
            for packet in stream.encode(None):
                output_container.mux(packet)
        if metadata:
            inject_metadata(path, metadata)
        size = os.path.getsize(path)
        if size > limit:
            raise ValueError(f"encoded AVIF exceeds the {limit}-byte output limit")
        with open(path, "rb") as source:
            return source.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


__all__ = ["encode_avif", "inject_metadata"]
