"""Bounded ISO BMFF and EBML framing shared by media admission and probing."""

from __future__ import annotations

import mmap
import struct

MediaBuffer = bytes | mmap.mmap
_STRUCTURE_ITEM_LIMIT = 100_000


def bmff_boxes(data: MediaBuffer, start: int, end: int) -> list[tuple[bytes, int, int]]:
    boxes: list[tuple[bytes, int, int]] = []
    offset = start
    while offset < end:
        if offset + 8 > end:
            raise ValueError("truncated ISO BMFF box")
        size = struct.unpack_from(">I", data, offset)[0]
        header = 8
        if size == 1:
            if offset + 16 > end:
                raise ValueError("truncated ISO BMFF large box")
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            raise ValueError("invalid ISO BMFF box length")
        boxes.append((data[offset + 4 : offset + 8], offset + header, offset + size))
        if len(boxes) > _STRUCTURE_ITEM_LIMIT:
            raise ValueError("ISO BMFF contains too many boxes")
        offset += size
    return boxes


def _ebml_vint(data: MediaBuffer, offset: int, *, keep_marker: bool) -> tuple[int | None, int]:
    if offset >= len(data) or data[offset] == 0:
        raise ValueError("invalid EBML variable integer")
    width = 1
    marker = 0x80
    while not data[offset] & marker:
        marker >>= 1
        width += 1
        if width > 8:
            raise ValueError("EBML variable integer is too wide")
    if offset + width > len(data):
        raise ValueError("truncated EBML variable integer")
    value = int.from_bytes(data[offset : offset + width], "big")
    if keep_marker:
        return value, width
    value &= (1 << (7 * width)) - 1
    return (None if value == (1 << (7 * width)) - 1 else value), width


def ebml_elements(data: MediaBuffer, start: int, end: int) -> list[tuple[int, int, int]]:
    elements: list[tuple[int, int, int]] = []
    offset = start
    while offset < end:
        element_id, id_width = _ebml_vint(data, offset, keep_marker=True)
        size, size_width = _ebml_vint(data, offset + id_width, keep_marker=False)
        payload_start = offset + id_width + size_width
        payload_end = end if size is None else payload_start + size
        if payload_start > end or payload_end > end:
            raise ValueError("truncated EBML element")
        assert element_id is not None
        elements.append((element_id, payload_start, payload_end))
        if len(elements) > _STRUCTURE_ITEM_LIMIT:
            raise ValueError("EBML contains too many elements")
        offset = payload_end
    return elements
