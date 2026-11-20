"""Bounded pipe frames for library documents larger than one protocol line."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from typing import cast

CHUNK_BYTES = 1024 * 1024


def message_frames(payload: bytes) -> Iterator[bytes]:
    for offset in range(0, len(payload), CHUNK_BYTES):
        chunk = payload[offset : offset + CHUNK_BYTES]
        yield (
            json.dumps(
                {
                    "chunk": base64.b64encode(chunk).decode("ascii"),
                    "offset": offset,
                    "complete": offset + len(chunk) == len(payload),
                }
            ).encode("ascii")
            + b"\n"
        )


class MessageReader:
    def __init__(self, line_limit: int, *, chunked: bool = False) -> None:
        self.line_limit = line_limit
        self.chunked = chunked
        self.pending = bytearray()
        self.discarding = False

    def accept(self, line: bytes) -> dict[str, object] | None:
        if not line:
            raise ValueError("library pipe closed")
        document: dict[str, object] | None = None
        try:
            if len(line) > self.line_limit or not line.endswith(b"\n"):
                raise ValueError("library frame is too large or incomplete")
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("library message must be an object")
            document = cast("dict[str, object]", value)
            if self.discarding:
                if "chunk" in document:
                    if document.get("complete") is True:
                        self.discarding = False
                    return None
                self.discarding = False
            if "chunk" not in document:
                if self.pending:
                    raise ValueError("library message interrupted an incomplete transfer")
                return document
            if not self.chunked:
                raise ValueError("library chunks were not negotiated")
            if set(document) != {"chunk", "offset", "complete"}:
                raise ValueError("unknown library frame field")
            offset, encoded, complete = document["offset"], document["chunk"], document["complete"]
            if (
                type(offset) is not int
                or offset != len(self.pending)
                or not isinstance(encoded, str)
                or type(complete) is not bool
            ):
                raise ValueError("library frame has invalid offset, data or completion")
            chunk = base64.b64decode(encoded, validate=True)
            if not 1 <= len(chunk) <= CHUNK_BYTES:
                raise ValueError("library chunk must contain 1 byte to 1 MiB")
            self.pending.extend(chunk)
            if not complete:
                return None
            value = json.loads(self.pending)
            self.pending.clear()
            if not isinstance(value, dict):
                raise ValueError("library message must be an object")
            return cast("dict[str, object]", value)
        except ValueError:
            if self.discarding:
                return None
            self.discarding = (
                document.get("complete") is not True
                if document is not None and "chunk" in document
                else bool(self.pending)
            )
            self.pending.clear()
            raise
