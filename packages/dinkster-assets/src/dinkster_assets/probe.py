"""Probing bytes for digest-keyed metadata (DESIGN 3.12 + roadmap
"templates/asset distribution").

The substitution UX ("this model is missing, but it was an F16 UNet with
N parameters - here are compatible alternatives") needs facts ABOUT held
bytes, keyed by digest so they are immutable and cacheable forever. This
module extracts those facts cheaply: header-only reads, no torch import,
no tensor bodies through memory - probing a 20 GB checkpoint costs one
small read.

First format: safetensors, whose design makes this trivial (an 8-byte
little-endian length followed by a JSON header describing every tensor's
dtype/shape). The result reports what the header SAYS - dtype histogram,
tensor count, parameter count, the author's ``__metadata__`` strings -
and claims nothing it cannot see: architecture/family identification is
a future, separate concern (a catalog service keyed by these same
digests), never an inference smuggled in here.

Probing is conservative: bytes that are not a recognized format yield
``{"format": "unknown"}``, never an exception - unknown is a valid,
cacheable answer.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

# The reference implementation caps safetensors headers at 100 MB; a
# larger claimed length is malformed (or malicious), not a big header.
_MAX_HEADER_BYTES = 100_000_000


def probe_file(path: Path) -> dict[str, object]:
    """Extract metadata from held bytes; the digest the caller resolved
    ``path`` from is the cache key. Never raises on content: unrecognized
    or malformed bytes report ``{"format": "unknown"}``."""
    try:
        with path.open("rb") as handle:
            return probe_handle(handle)
    except OSError:
        return {"format": "unknown"}


def probe_handle(handle: BinaryIO) -> dict[str, object]:
    """Probe an already-open, seekable handle. The point of taking a handle:
    callers that verified a descriptor against a digest (integrity.py) can
    probe the PROVEN bytes instead of reopening a reboundable path. Same
    conservatism as :func:`probe_file`; the handle stays open."""
    safetensors = _probe_safetensors(handle)
    if safetensors is not None:
        return safetensors
    return {"format": "unknown"}


def _probe_safetensors(handle: BinaryIO) -> dict[str, object] | None:
    try:
        size = handle.seek(0, 2)
        handle.seek(0)
        prefix = handle.read(8)
        if len(prefix) != 8:
            return None
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len == 0 or header_len > min(_MAX_HEADER_BYTES, size - 8):
            return None
        header_raw: object = json.loads(handle.read(header_len))
    except (OSError, ValueError):
        return None
    if not isinstance(header_raw, dict):
        return None
    header = cast("dict[str, object]", header_raw)

    dtypes: dict[str, int] = {}
    tensor_count = 0
    parameter_count = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, Mapping):
            return None  # not a safetensors header after all
        tensor = cast("Mapping[str, object]", entry)
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            return None
        dims = cast("list[object]", shape)
        if not all(isinstance(dim, int) and dim >= 0 for dim in dims):
            return None
        tensor_count += 1
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
        elements = 1
        for dim in cast("list[int]", dims):
            elements *= dim
        parameter_count += elements
    if tensor_count == 0:
        return None

    result: dict[str, object] = {
        "format": "safetensors",
        "tensorCount": tensor_count,
        "parameterCount": parameter_count,
        # Sorted for a stable wire shape; counts make mixed-dtype files
        # (quantized weights + F32 norms) visible at a glance.
        "dtypes": dict(sorted(dtypes.items())),
    }
    metadata = header.get("__metadata__")
    if isinstance(metadata, Mapping):
        strings = {
            str(k): v
            for k, v in cast("Mapping[object, object]", metadata).items()
            if isinstance(v, str)
        }
        if strings:
            result["extra"] = strings
    return result
