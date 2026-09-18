"""Strict, allocation-free validation of supported latent safetensors files."""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import BinaryIO, Literal, cast

from .identity import AssetError, require_digest

LATENT_SUFFIX = ".latent"
LATENT_MEDIA_TYPE = "application/x-comfy-latent"
LATENT_ASSET_KIND = "data/latent"
LATENT_SCHEMA_KEY = "dinkster_latent_schema"
MAX_LATENT_HEADER_BYTES = 4 * 1024 * 1024
MAX_LATENT_SCHEMA_BYTES = 64 * 1024
MAX_LATENT_DATA_BYTES = 1024 * 1024 * 1024
MAX_LATENT_STREAMS = 64
MAX_VAE_HINT_BYTES = 8 * 1024

_DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_LATENT_SPACE_RE = re.compile(r"^dinkster\.[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_VAE_HINT_FIELDS = frozenset({"sourceName", "sourceLogicalId", "latentSpace"})


@dataclass(frozen=True, slots=True)
class LatentTensorDescriptor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    byte_length: int
    role: str | None = None


@dataclass(frozen=True, slots=True)
class LatentAssetDescriptor:
    profile: Literal["dinkster-v1", "comfyui-single"]
    tensors: tuple[LatentTensorDescriptor, ...]
    header_bytes: int
    file_bytes: int
    legacy_scale: bool = False
    metadata: Mapping[str, str] = MappingProxyType({})


def validate_vae_hint_field(name: str, value: object) -> str:
    """Validate one path-free VAE hint field shared by producers and readers."""
    if name not in _VAE_HINT_FIELDS:
        raise AssetError(f"unknown VAE hint field {name!r}")
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 1024
        or value in {".", ".."}
        or value.startswith("~")
        or "/" in value
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AssetError(f"VAE {name} must be bounded path-free text")
    if name == "latentSpace" and _LATENT_SPACE_RE.fullmatch(value) is None:
        raise AssetError("VAE latentSpace must be a registered dinkster family or codec ID")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssetError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _object(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssetError(message)
    return cast("dict[str, object]", value)


def parse_latent_asset(source: BinaryIO) -> LatentAssetDescriptor:
    """Validate framing and schema without reading any tensor body."""
    try:
        source.seek(0, 2)
        file_bytes = source.tell()
        source.seek(0)
        framing = source.read(8)
    except (OSError, AttributeError) as exc:
        raise AssetError("latent source must be seekable binary input") from exc
    if len(framing) != 8:
        raise AssetError("truncated safetensors framing")
    header_bytes = struct.unpack("<Q", framing)[0]
    if header_bytes == 0 or header_bytes % 8 or header_bytes > MAX_LATENT_HEADER_BYTES:
        raise AssetError("safetensors header exceeds the supported bound")
    data_start = 8 + header_bytes
    if data_start > file_bytes:
        raise AssetError("safetensors header extends beyond the file")
    raw = source.read(header_bytes)
    if len(raw) != header_bytes:
        raise AssetError("truncated safetensors header")
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise AssetError("invalid safetensors header JSON") from exc
    table = _object(header, "safetensors header must be an object")
    metadata_value = table.get("__metadata__", {})
    metadata = _object(metadata_value, "safetensors metadata must be an object")
    if any(not isinstance(value, str) for value in metadata.values()):
        raise AssetError("safetensors metadata values must be strings")
    string_metadata = cast("dict[str, str]", metadata)

    parsed: dict[str, LatentTensorDescriptor] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, value in table.items():
        if name == "__metadata__":
            continue
        row = _object(value, f"tensor {name!r} descriptor must be an object")
        if set(row) != {"dtype", "shape", "data_offsets"}:
            raise AssetError(f"tensor {name!r} has invalid descriptor fields")
        dtype = row["dtype"]
        shape = row["shape"]
        offsets = row["data_offsets"]
        if dtype not in _DTYPE_BYTES:
            raise AssetError(f"tensor {name!r} has unsupported dtype")
        if not isinstance(shape, list):
            raise AssetError(f"tensor {name!r} must have rank 1 through 8")
        shape_values = cast("list[object]", shape)
        if not 1 <= len(shape_values) <= 8:
            raise AssetError(f"tensor {name!r} must have rank 1 through 8")
        if any(type(dim) is not int or dim <= 0 for dim in shape_values):
            # ComfyUI's empty marker is the sole supported zero-sized tensor.
            marker = name == "latent_format_version_0" and shape == [0]
            if not marker:
                raise AssetError(f"tensor {name!r} has invalid shape")
        if not isinstance(offsets, list):
            raise AssetError(f"tensor {name!r} has invalid offsets")
        offset_values = cast("list[object]", offsets)
        if len(offset_values) != 2 or any(type(x) is not int for x in offset_values):
            raise AssetError(f"tensor {name!r} has invalid offsets")
        start, end = cast("list[int]", offsets)
        shape_ints = cast("list[int]", shape)
        elements = 1
        for dim in shape_ints:
            elements *= dim
            if elements > MAX_LATENT_DATA_BYTES:
                raise AssetError(f"tensor {name!r} shape is too large")
        expected = elements * _DTYPE_BYTES[cast(str, dtype)]
        if start < 0 or end < start or end - start != expected:
            raise AssetError(f"tensor {name!r} extent does not match shape")
        parsed[name] = LatentTensorDescriptor(
            name, cast(str, dtype), tuple(shape_ints), data_start + start, expected
        )
        intervals.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise AssetError(f"tensor {name!r} offsets are not exactly contiguous")
        cursor = end
    if cursor > MAX_LATENT_DATA_BYTES or data_start + cursor != file_bytes:
        raise AssetError("tensor data extent does not exactly match the file")

    schema_raw = metadata.get(LATENT_SCHEMA_KEY)
    if schema_raw is not None:
        if not isinstance(schema_raw, str):
            raise AssetError("Dinkster latent schema metadata must be a string")
        if len(schema_raw.encode("utf-8")) > MAX_LATENT_SCHEMA_BYTES:
            raise AssetError("Dinkster latent schema exceeds the supported bound")
        try:
            schema = json.loads(schema_raw, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError) as exc:
            raise AssetError("invalid Dinkster latent schema JSON") from exc
        return _parse_native(
            _object(schema, "Dinkster latent schema must be an object"),
            parsed,
            header_bytes,
            file_bytes,
            string_metadata,
        )
    tensor_names = set(parsed)
    if tensor_names not in (
        {"latent_tensor"},
        {"latent_tensor", "latent_format_version_0"},
    ):
        raise AssetError("unsupported schema-less latent tensor profile")
    tensor = parsed["latent_tensor"]
    marker = parsed.get("latent_format_version_0")
    # Safetensors orders by dtype then name, so the empty F32 marker can
    # precede or follow the payload. Contiguity was validated above.
    if marker is not None and (
        marker.dtype != "F32" or marker.shape != (0,) or marker.byte_length != 0
    ):
        raise AssetError("invalid ComfyUI latent format marker")
    return LatentAssetDescriptor(
        "comfyui-single",
        (tensor,),
        header_bytes,
        file_bytes,
        marker is None,
        MappingProxyType(dict(string_metadata)),
    )


def _parse_native(
    schema: dict[str, object],
    tensors: Mapping[str, LatentTensorDescriptor],
    header_bytes: int,
    file_bytes: int,
    metadata: Mapping[str, str],
) -> LatentAssetDescriptor:
    if set(schema) != {"format", "version", "structure", "streams"}:
        raise AssetError("Dinkster latent schema has invalid fields")
    version = schema["version"]
    if schema["format"] != "dinkster.latent" or type(version) is not int or version != 1:
        raise AssetError("unknown Dinkster latent schema")
    structure = schema["structure"]
    streams = schema["streams"]
    if not isinstance(streams, list):
        raise AssetError("Dinkster latent streams must contain 1 through 64 descriptors")
    stream_values = cast("list[object]", streams)
    if not 1 <= len(stream_values) <= MAX_LATENT_STREAMS:
        raise AssetError("Dinkster latent streams must contain 1 through 64 descriptors")
    result: list[LatentTensorDescriptor] = []
    roles: set[str] = set()
    for index, value in enumerate(stream_values):
        row = _object(value, "Dinkster stream descriptor must be an object")
        expected_name = (
            "dinkster_samples" if structure == "single" else f"dinkster_stream_{index:04d}"
        )
        expected_keys = (
            {"tensor", "dtype", "shape"}
            if structure == "single"
            else {"order", "role", "tensor", "dtype", "shape"}
        )
        if set(row) != expected_keys or row.get("tensor") != expected_name:
            raise AssetError("Dinkster stream descriptor fields or name are invalid")
        if structure == "single" and len(stream_values) != 1:
            raise AssetError("single Dinkster latent requires exactly one stream")
        role: str | None = None
        if structure == "multi":
            role_value = row.get("role")
            order = row.get("order")
            if (
                type(order) is not int
                or order != index
                or type(role_value) is not str
                or not role_value
                or len(role_value.encode("utf-8")) > 128
                or role_value in roles
            ):
                raise AssetError("Dinkster stream order or role is invalid")
            role = role_value
            roles.add(role)
        elif structure != "single":
            raise AssetError("unknown Dinkster latent structure")
        tensor = tensors.get(expected_name)
        schema_shape = row["shape"]
        schema_shape_values = (
            cast("list[object]", schema_shape) if isinstance(schema_shape, list) else None
        )
        if (
            tensor is None
            or row["dtype"] != tensor.dtype
            or schema_shape_values is None
            or any(type(dim) is not int for dim in schema_shape_values)
            or schema_shape_values != list(tensor.shape)
        ):
            raise AssetError("Dinkster stream descriptor does not equal its tensor header")
        result.append(
            LatentTensorDescriptor(
                tensor.name,
                tensor.dtype,
                tensor.shape,
                tensor.data_offset,
                tensor.byte_length,
                role,
            )
        )
    if set(tensors) != {item.name for item in result}:
        raise AssetError("native Dinkster latent contains an unreferenced tensor")
    return LatentAssetDescriptor(
        "dinkster-v1",
        tuple(result),
        header_bytes,
        file_bytes,
        metadata=MappingProxyType(dict(metadata)),
    )


def valid_vae_hint(descriptor: LatentAssetDescriptor) -> str:
    """Return a bounded canonical VAE hint, or empty text when unusable."""
    raw = descriptor.metadata.get("dinkster_vae_hint")
    if raw is None or len(raw.encode("utf-8")) > MAX_VAE_HINT_BYTES:
        return ""
    try:
        value: object = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError, AssetError):
        return ""
    if not isinstance(value, dict):
        return ""
    fields = cast("dict[str, object]", value)
    if not set(fields).issubset(
        {"version", "sourceDigest", "sourceName", "sourceLogicalId", "latentSpace"}
    ):
        return ""
    digest = fields.get("sourceDigest")
    version = fields.get("version")
    if type(version) is not int or version != 1 or not isinstance(digest, str):
        return ""
    try:
        require_digest(digest)
    except AssetError:
        return ""
    for name in _VAE_HINT_FIELDS:
        field = fields.get(name)
        if field is not None:
            try:
                validate_vae_hint_field(name, field)
            except AssetError:
                return ""
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return raw if raw == canonical else ""
