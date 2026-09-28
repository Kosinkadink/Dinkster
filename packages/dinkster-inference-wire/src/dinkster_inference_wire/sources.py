"""Safetensors header inspection: the strict, loading-path WeightSource.

A safetensors file is an 8-byte little-endian header length, a JSON
header mapping tensor keys to ``{dtype, shape, data_offsets}``, then the
packed payload. Everything detection (stage 2) and load planning
(stage 4) need is in the header; this module reads it WITHOUT touching
a single payload byte - inspecting a 20 GB checkpoint costs one small
read.

This is deliberately NOT dinkster-assets' ``probe_file``. Probing serves
digest-keyed metadata where "unknown" is a valid cacheable answer, so it
is lenient by design. This is the loading path: a file that claims to be
safetensors but violates the format is a loud :class:`MalformedSafetensors`
naming exactly what is wrong - never a guess, never a partial read that
stage 4 later trips over. (ComfyUI defers most of this to the
safetensors library inside a whole-dict load, comfy/utils.py
load_torch_file @ b78cec87; here validation happens up front, once,
against the header alone.)

Strictness rules (matching the reference safetensors validation,
huggingface/safetensors safetensors/src/tensor.rs):

- header length must be positive, fit inside the file, and stay under
  the 100 MB cap the reference implementation enforces;
- duplicate tensor keys are an error (last-one-wins JSON parsing is how
  two tensors silently become one);
- every entry carries ``dtype``/``shape``/``data_offsets`` (unknown
  extra fields are ignored, as the reference deserializer does), the
  dtype is one of the reference's codes, dims are non-negative ints;
- each byte range must lie inside the payload and match the tensor's
  exact packed size - for sub-byte dtypes (F4, F6) the total bit count
  must be byte-aligned, the reference's MisalignedSlice rule, never
  rounded up;
- the ranges together must tile the payload exactly - no gaps, no
  overlaps;
- ``__metadata__`` values must all be strings (the spec's type).
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, cast

from .devices import (
    BFLOAT16,
    BOOL,
    COMPLEX64,
    FLOAT4,
    FLOAT6_E2M3,
    FLOAT6_E3M2,
    FLOAT8_E4M3,
    FLOAT8_E4M3FNUZ,
    FLOAT8_E5M2,
    FLOAT8_E5M2FNUZ,
    FLOAT8_E8M0,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    DType,
)
from .weights import TensorGeometry, WeightEntry

# The reference implementation refuses headers above 100 MB; a larger
# claimed length is malformed (or malicious), not a big header.
_MAX_HEADER_BYTES = 100_000_000

# The COMPLETE reference dtype table (huggingface/safetensors Dtype
# enum) -> Dinkster dtypes. A code outside this table is malformed, not
# merely unsupported - the reference deserializer would refuse it too.
_SAFETENSORS_DTYPES: Mapping[str, DType] = MappingProxyType(
    {
        "BOOL": BOOL,
        "F4": FLOAT4,
        "F6_E2M3": FLOAT6_E2M3,
        "F6_E3M2": FLOAT6_E3M2,
        "U8": UINT8,
        "I8": INT8,
        "F8_E5M2": FLOAT8_E5M2,
        "F8_E4M3": FLOAT8_E4M3,
        "F8_E8M0": FLOAT8_E8M0,
        "F8_E4M3FNUZ": FLOAT8_E4M3FNUZ,
        "F8_E5M2FNUZ": FLOAT8_E5M2FNUZ,
        "I16": INT16,
        "U16": UINT16,
        "F16": FLOAT16,
        "BF16": BFLOAT16,
        "I32": INT32,
        "U32": UINT32,
        "F32": FLOAT32,
        "C64": COMPLEX64,
        "F64": FLOAT64,
        "I64": INT64,
        "U64": UINT64,
    }
)


class MalformedSafetensors(ValueError):
    """The file violates the safetensors format; the message names the
    file and the exact violation."""


@dataclass(frozen=True)
class SafetensorsSource:
    """A parsed safetensors header implementing the WeightSource
    contract. Construct via :func:`load_safetensors_header`.

    ``WeightEntry.offset`` values are ABSOLUTE file offsets (prefix +
    header + payload-relative begin), so stage-4 slice reads seek
    directly. Key order preserves header order.

    ``asset_digest``/``asset_size`` optionally bind the source to the
    immutable asset it was opened from (the AssetIdentifiedSource seam);
    both are set together or not at all.
    ``configuration_file`` pins auxiliary reads to a caller-owned open descriptor.
    """

    path: Path
    entries: Mapping[str, WeightEntry]
    extra: Mapping[str, str]
    asset_digest: str | None = None
    asset_size: int | None = None
    configuration_file: BinaryIO | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))
        if (self.asset_digest is None) != (self.asset_size is None):
            raise ValueError("asset_digest and asset_size must be set together")
        if self.asset_digest is not None:
            digest = self.asset_digest
            if (
                type(digest) is not str
                or not digest.startswith("blake3:")
                or len(digest) != 71
                or any(char not in "0123456789abcdef" for char in digest[7:])
            ):
                raise ValueError("asset_digest must be a canonical blake3 digest")
        if self.asset_size is not None and (
            type(self.asset_size) is not int or self.asset_size < 0
        ):
            raise ValueError("asset_size must be a nonnegative integer")

    def keys(self) -> Sequence[str]:
        return tuple(self.entries)

    def entry(self, key: str) -> WeightEntry:
        return self.entries[key]

    def metadata(self) -> Mapping[str, str]:
        return self.extra

    def read_float_scalar(self, key: str) -> float:
        """Read one floating-point scalar used as model configuration.

        Architecture detection remains header-only. Assembly uses this
        narrow payload seam only for checkpoint-provided scalar settings
        whose numeric value affects execution and structural identity.
        """
        entry = self.entry(key)
        geometry = entry.geometry
        if geometry.numel != 1:
            raise ValueError(f"{key!r} must contain exactly one value")
        with (
            self.path.open("rb")
            if self.configuration_file is None
            else nullcontext(self.configuration_file)
        ) as handle:
            handle.seek(entry.offset)
            data = handle.read(entry.nbytes)
        if len(data) != entry.nbytes:
            raise ValueError(f"{self.path}: tensor {key!r} payload is truncated")
        dtype = geometry.dtype.name
        if dtype == "float16":
            value = struct.unpack("<e", data)[0]
        elif dtype == "bfloat16":
            (bits,) = struct.unpack("<H", data)
            value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        elif dtype == "float32":
            value = struct.unpack("<f", data)[0]
        elif dtype == "float64":
            value = struct.unpack("<d", data)[0]
        else:
            raise ValueError(f"{key!r} must be a floating-point scalar, got {dtype}")
        if not math.isfinite(value):
            raise ValueError(f"{key!r} must be finite, got {value}")
        return value

    def read_uint8_configuration(self, key: str, *, limit: int = 65_536) -> bytes:
        """Read one small, explicit uint8 configuration payload."""
        if self.configuration_file is not None:
            return self.read_uint8_configuration_from_file(
                self.configuration_file, key, limit=limit
            )
        with self.path.open("rb") as handle:
            return self.read_uint8_configuration_from_file(handle, key, limit=limit)

    def read_uint8_configuration_from_file(
        self, file: BinaryIO, key: str, *, limit: int = 65_536
    ) -> bytes:
        """Read configuration from a caller-owned source descriptor.

        ``limit`` guards against reading model weights as configuration;
        callers with a known-large payload (an embedded tokenizer model)
        pass their own explicit byte cap.
        """
        entry = self.entry(key)
        geometry = entry.geometry
        if geometry.dtype.name != "uint8" or len(geometry.shape) != 1:
            raise ValueError(f"{key!r} must be a rank-1 uint8 configuration tensor")
        if entry.nbytes > limit:
            raise ValueError(f"{key!r} exceeds the {limit}-byte configuration cap")
        file.seek(entry.offset)
        data = file.read(entry.nbytes)
        if len(data) != entry.nbytes:
            raise ValueError(f"{self.path}: tensor {key!r} payload is truncated")
        return data


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r} in header")
        result[key] = value
    return result


def load_safetensors_header(
    path: Path,
    *,
    asset_digest: str | None = None,
    asset_size: int | None = None,
) -> SafetensorsSource:
    """Read and validate a safetensors header; never reads payload bytes.

    Raises :class:`MalformedSafetensors` on any format violation and
    ``OSError`` on plain IO failure (missing file, permissions) - IO
    trouble is not malformation.
    """

    with path.open("rb") as handle:
        return load_safetensors_header_from_file(
            handle, path=path, asset_digest=asset_digest, asset_size=asset_size
        )


def load_safetensors_header_from_file(
    handle: BinaryIO,
    *,
    path: Path,
    asset_digest: str | None = None,
    asset_size: int | None = None,
) -> SafetensorsSource:
    """Validate a safetensors header from one caller-owned open file."""

    def bad(reason: str) -> MalformedSafetensors:
        return MalformedSafetensors(f"{path}: {reason}")

    size = os.fstat(handle.fileno()).st_size
    handle.seek(0)
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise bad(f"file is {size} bytes; the 8-byte header length is missing")
    (header_len,) = struct.unpack("<Q", prefix)
    if header_len == 0:
        raise bad("header length is 0")
    if header_len > _MAX_HEADER_BYTES:
        raise bad(f"header length {header_len} exceeds the {_MAX_HEADER_BYTES} byte cap")
    if 8 + header_len > size:
        raise bad(f"header length {header_len} overruns the {size}-byte file")
    header_bytes = handle.read(header_len)
    if len(header_bytes) != header_len:
        raise bad("file shrank while reading the header")

    try:
        header_raw: object = json.loads(
            header_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise bad(f"header is not valid JSON: {error}") from error
    if not isinstance(header_raw, dict):
        raise bad("header must be a JSON object")
    header = cast("dict[str, object]", header_raw)

    payload_size = size - 8 - header_len
    payload_start = 8 + header_len
    entries: dict[str, WeightEntry] = {}
    ranges: list[tuple[int, int, str]] = []
    extra: dict[str, str] = {}

    for key, raw in header.items():
        if key == "__metadata__":
            if not isinstance(raw, dict):
                raise bad("__metadata__ must be a JSON object")
            for meta_key, meta_value in cast("dict[str, object]", raw).items():
                if not isinstance(meta_value, str):
                    raise bad(f"__metadata__[{meta_key!r}] must be a string")
                extra[meta_key] = meta_value
            continue
        if not key:
            raise bad("tensor key must not be empty")
        if not isinstance(raw, dict):
            raise bad(f"tensor {key!r}: entry must be a JSON object")
        entry = cast("dict[str, object]", raw)
        # Unknown extra fields are ignored, matching the reference
        # deserializer (serde's default for TensorInfo).

        dtype_code = entry.get("dtype")
        if not isinstance(dtype_code, str):
            raise bad(f"tensor {key!r}: dtype must be a string")
        dtype = _SAFETENSORS_DTYPES.get(dtype_code)
        if dtype is None:
            raise bad(f"tensor {key!r}: unknown dtype code {dtype_code!r}")

        shape_raw = entry.get("shape")
        if not isinstance(shape_raw, list):
            raise bad(f"tensor {key!r}: shape must be a list of non-negative integers")
        dims = cast("list[object]", shape_raw)
        if not all(type(dim) is int and dim >= 0 for dim in dims):
            raise bad(f"tensor {key!r}: shape must be a list of non-negative integers")
        geometry = TensorGeometry(tuple(cast("list[int]", dims)), dtype)

        offsets_raw = entry.get("data_offsets")
        if not isinstance(offsets_raw, list):
            raise bad(f"tensor {key!r}: data_offsets must be a pair of integers")
        edges = cast("list[object]", offsets_raw)
        if len(edges) != 2 or not all(type(edge) is int for edge in edges):
            raise bad(f"tensor {key!r}: data_offsets must be a pair of integers")
        begin, end = cast("list[int]", edges)
        if not 0 <= begin <= end <= payload_size:
            raise bad(
                f"tensor {key!r}: data_offsets [{begin}, {end}] fall outside "
                f"the {payload_size}-byte payload"
            )
        # Exact packed size, the reference rule: the total bit count
        # must be byte-aligned (sub-byte dtypes cannot leave a partially
        # filled trailing byte), then bytes = bits // 8. Deliberately
        # NOT TensorGeometry.nbytes, which rounds up for planning.
        nbits = geometry.numel * dtype.bits
        if nbits % 8 != 0:
            raise bad(
                f"tensor {key!r}: {geometry.numel} x {dtype.bits}-bit elements "
                f"({dtype_code}) do not fill whole bytes"
            )
        if end - begin != nbits // 8:
            raise bad(
                f"tensor {key!r}: byte range is {end - begin} bytes but the "
                f"geometry needs {nbits // 8}"
            )
        entries[key] = WeightEntry(
            key=key,
            geometry=geometry,
            offset=payload_start + begin,
            nbytes=end - begin,
        )
        ranges.append((begin, end, key))

    ranges.sort()
    cursor = 0
    for begin, end, key in ranges:
        if begin != cursor:
            gap = "overlaps the previous tensor" if begin < cursor else "leaves a gap"
            raise bad(f"tensor {key!r}: byte range [{begin}, {end}] {gap}")
        cursor = end
    if cursor != payload_size:
        raise bad(f"tensors cover {cursor} of {payload_size} payload bytes")

    return SafetensorsSource(
        path=path,
        entries=entries,
        extra=extra,
        asset_digest=asset_digest,
        asset_size=asset_size,
    )


__all__ = [
    "MalformedSafetensors",
    "SafetensorsSource",
    "load_safetensors_header",
    "load_safetensors_header_from_file",
]
