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
digests), never an inference smuggled in here. Media files additionally carry
decoded formula facts; these can require a full streaming decode and are
retained by digest with :func:`probe_cached`.

Probing is conservative: bytes that are not a recognized format yield
``{"format": "unknown"}``, never an exception - unknown is a valid,
cacheable answer.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, cast

from filelock import FileLock

from .identity import AssetError, require_digest
from .integrity import open_verified
from .latent_format import parse_latent_asset, valid_vae_hint
from .media import classify_media_handle

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
        try:
            latent = parse_latent_asset(handle)
        except AssetError:
            return safetensors
        safetensors["latent"] = {
            "profile": latent.profile,
            "streams": [
                {
                    "name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    **({} if tensor.role is None else {"role": tensor.role}),
                }
                for tensor in latent.tensors
            ],
            "vaeHint": valid_vae_hint(latent),
        }
        return safetensors
    handle.seek(0)
    prefix = handle.read(6)
    if prefix in (b"GIF87a", b"GIF89a"):
        media_type = "image/gif"
    else:
        try:
            media_type = classify_media_handle(handle).media_type
        except AssetError:
            return {"format": "unknown"}
    from dinkster_values.file_facts import decoded_file_facts

    result: dict[str, object] = {"format": "media", "mediaType": media_type}
    try:
        facts, reason = decoded_file_facts(handle, media_type)
        result["facts"] = facts
        if reason is not None:
            result["factsError"] = reason
    except Exception as error:  # noqa: BLE001 - codec failures are author diagnostics, not metadata failures
        result["facts"] = {}
        result["factsError"] = f"Cannot read file facts: {error}"
    return result


def probe_cached(path: Path, digest: str, cache_root: Path) -> dict[str, object]:
    """Retain verified source facts across concurrent reads and engine restarts.

    This is not an authorization boundary. Callers must authorize the exact
    scope and confirm local presence before reading even an existing answer.
    """
    hex_digest = require_digest(digest).split(":", 1)[1]
    root = cache_root / "v2" / hex_digest[:2]
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{hex_digest}.json"
    with FileLock(root / f"{hex_digest}.lock"):
        try:
            stored: object = json.loads(target.read_text("utf-8"))
        except FileNotFoundError:
            pass
        else:
            if not isinstance(stored, dict):
                raise AssetError("invalid cached asset metadata")
            return cast(dict[str, object], stored)
        with open_verified(path, digest) as handle:
            metadata = probe_handle(handle)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as pending:
            temporary = Path(pending.name)
            try:
                json.dump(metadata, pending, allow_nan=False)
                pending.flush()
                os.fsync(pending.fileno())
                pending.close()
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return metadata


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
