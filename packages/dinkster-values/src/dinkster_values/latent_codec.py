"""Deterministic structural codec for ordinary and multi-stream latents."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import math
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from .model import stable_hash

LATENT_CODEC_MAGIC = b"DINKSTER-LATENT\x00"
LATENT_CODEC_VERSION = 1
_DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


@dataclass(frozen=True)
class EncodedLatentTensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True)
class EncodedMultiStreamLatent:
    streams: tuple[tuple[str, object], ...]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role for role, _ in self.streams)


@dataclass(frozen=True)
class EncodedSparseSupport:
    coordinates: object
    batch_counts: tuple[int, ...]
    resolution: int
    origin: tuple[float, float, float]
    voxel_size: tuple[float, float, float]
    support_id: str


@dataclass(frozen=True)
class EncodedSparseLatent:
    support: object
    features: object


TensorDecoder = Callable[[EncodedLatentTensor], object]
MultiStreamFactory = Callable[[Sequence[tuple[str, object]]], object]
SparseSupportFactory = Callable[[EncodedSparseSupport], object]
SparseLatentFactory = Callable[[EncodedSparseLatent], object]

_SPARSE_SUPPORT_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SPARSE_DIGEST_DOMAIN = b"dinkster.sparse-support.v1\0"
_INTEGER_FORMATS = {
    "uint8": "B",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
}


def _copy_address(address: int, size: int) -> bytes:
    return bytes((ctypes.c_ubyte * size).from_address(address))


def tensor_record(value: object) -> EncodedLatentTensor | None:
    if isinstance(value, EncodedLatentTensor):
        return value
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    if hasattr(value, "detach"):
        tensor = cast("Any", value).detach().cpu().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        data = _copy_address(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        return EncodedLatentTensor(dtype, tuple(int(n) for n in tensor.shape), data)
    try:
        numpy = __import__("numpy")
        array = numpy.ascontiguousarray(value)
    except (ImportError, TypeError, ValueError):
        return None
    return EncodedLatentTensor(
        str(array.dtype), tuple(int(n) for n in array.shape), array.tobytes()
    )


def _multi_stream_pairs(value: object) -> tuple[tuple[str, object], ...] | None:
    if isinstance(value, EncodedMultiStreamLatent):
        return value.streams
    if type(value).__name__ != "MultiStreamLatent" or not hasattr(value, "streams"):
        return None
    streams = cast("Any", value).streams
    return tuple((stream.role, stream.payload) for stream in streams)


def _is_sparse_value(value: object, name: str) -> bool:
    value_type = type(value)
    return value_type.__module__ == "dinkster_inference.sparse" and value_type.__name__ == name


def _validate_encoded_sparse_support(value: EncodedSparseSupport) -> None:
    if (
        type(value.batch_counts) is not tuple
        or not value.batch_counts
        or any(type(count) is not int or count < 1 for count in value.batch_counts)
        or type(value.resolution) is not int
        or value.resolution < 1
        or type(value.origin) is not tuple
        or len(value.origin) != 3
        or any(type(item) is not float or not math.isfinite(item) for item in value.origin)
        or type(value.voxel_size) is not tuple
        or len(value.voxel_size) != 3
        or any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in value.voxel_size
        )
        or type(value.support_id) is not str
        or _SPARSE_SUPPORT_ID.fullmatch(value.support_id) is None
    ):
        raise ValueError("latent frame contains invalid sparse support metadata")
    coordinates = value.coordinates
    if not isinstance(coordinates, EncodedLatentTensor):
        return
    if coordinates.dtype not in _INTEGER_FORMATS or coordinates.shape != (
        sum(value.batch_counts),
        4,
    ):
        raise ValueError("sparse support coordinates must be an integer (points, 4) tensor")
    rows = tuple(
        item[0]
        for item in struct.iter_unpack("=" + _INTEGER_FORMATS[coordinates.dtype], coordinates.data)
    )
    digest = hashlib.sha256()
    digest.update(_SPARSE_DIGEST_DOMAIN)
    digest.update(struct.pack("<Q", coordinates.shape[0]))
    for coordinate in rows:
        digest.update(struct.pack("<q", coordinate))
    if value.support_id != "sha256:" + digest.hexdigest():
        raise ValueError("sparse support coordinate digest does not match its authenticated id")
    offset = 0
    for batch, count in enumerate(value.batch_counts):
        for index in range(offset, offset + count):
            row = rows[index * 4 : index * 4 + 4]
            if row[0] != batch:
                raise ValueError("sparse coordinate rows must be contiguous and match batch counts")
            if any(item < 0 or item >= value.resolution for item in row[1:]):
                raise ValueError(
                    "sparse spatial coordinates must be inside the declared resolution"
                )
        offset += count


def _validate_encoded_sparse_latent(value: EncodedSparseLatent) -> None:
    if not isinstance(value.support, EncodedSparseSupport):
        return
    _validate_encoded_sparse_support(value.support)
    features = value.features
    if not isinstance(features, EncodedLatentTensor):
        return
    if (
        features.dtype not in {"float16", "bfloat16", "float32", "float64"}
        or len(features.shape) != 2
        or features.shape[0] != sum(value.support.batch_counts)
        or features.shape[1] < 1
    ):
        raise ValueError("sparse latent features must be floating (points, channels) tensor")


def _sparse_support_record(value: object) -> EncodedSparseSupport | None:
    if isinstance(value, EncodedSparseSupport):
        return value
    if not _is_sparse_value(value, "SparseSupport"):
        return None
    support = cast("Any", value)
    record = EncodedSparseSupport(
        support.coordinates,
        tuple(support.batch_counts),
        support.resolution,
        tuple(support.origin),
        tuple(support.voxel_size),
        support.support_id,
    )
    return record


def _sparse_latent_record(value: object) -> EncodedSparseLatent | None:
    if isinstance(value, EncodedSparseLatent):
        return value
    if not _is_sparse_value(value, "SparseLatent"):
        return None
    latent = cast("Any", value)
    return EncodedSparseLatent(latent.support, latent.features)


def _encode_tree(value: object) -> object:
    sparse_latent = _sparse_latent_record(value)
    if sparse_latent is not None:
        return {
            "$": "sparse-latent",
            "features": _encode_tree(sparse_latent.features),
            "support": _encode_tree(sparse_latent.support),
        }
    sparse_support = _sparse_support_record(value)
    if sparse_support is not None:
        coordinates = tensor_record(sparse_support.coordinates)
        if coordinates is None:
            raise TypeError("sparse support coordinates must be tensor-like")
        authenticated = EncodedSparseSupport(
            coordinates,
            sparse_support.batch_counts,
            sparse_support.resolution,
            sparse_support.origin,
            sparse_support.voxel_size,
            sparse_support.support_id,
        )
        _validate_encoded_sparse_support(authenticated)
        return {
            "$": "sparse-support",
            "batch_counts": _encode_tree(authenticated.batch_counts),
            "coordinates": _encode_tree(authenticated.coordinates),
            "origin": _encode_tree(authenticated.origin),
            "resolution": authenticated.resolution,
            "support_id": authenticated.support_id,
            "voxel_size": _encode_tree(authenticated.voxel_size),
        }
    tensor = tensor_record(value)
    if tensor is not None:
        return {
            "$": "tensor",
            "data": base64.b64encode(tensor.data).decode("ascii"),
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
        }
    pairs = _multi_stream_pairs(value)
    if pairs is not None:
        roles = tuple(role for role, _ in pairs)
        if not roles or len(set(roles)) != len(roles) or any(not role for role in roles):
            raise ValueError("multi-stream latent roles must be unique nonempty strings")
        return {
            "$": "multi",
            "streams": [[role, _encode_tree(payload)] for role, payload in pairs],
        }
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, tuple):
        return {
            "$": "tuple",
            "items": [_encode_tree(item) for item in cast("tuple[object, ...]", value)],
        }
    if isinstance(value, list):
        return [_encode_tree(item) for item in cast("list[object]", value)]
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        if any(type(key) is not str for key in mapping):
            raise TypeError("latent mapping keys must be strings")
        return {
            "$": "map",
            "items": {
                key: _encode_tree(mapping[key])
                for key in sorted(cast("Sequence[str]", tuple(mapping)))
            },
        }
    raise TypeError(f"unsupported latent metadata value: {type(value).__name__}")


def encode_latent(value: object) -> bytes:
    body = json.dumps(
        _encode_tree(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return LATENT_CODEC_MAGIC + bytes((LATENT_CODEC_VERSION,)) + body


def _decode_tree(
    value: object,
    tensor_decoder: TensorDecoder | None,
    multi_stream_factory: MultiStreamFactory | None,
    sparse_support_factory: SparseSupportFactory | None,
    sparse_latent_factory: SparseLatentFactory | None,
) -> object:
    def decode(item: object) -> object:
        return _decode_tree(
            item,
            tensor_decoder,
            multi_stream_factory,
            sparse_support_factory,
            sparse_latent_factory,
        )

    if isinstance(value, list):
        return [decode(item) for item in cast("list[object]", value)]
    if not isinstance(value, Mapping):
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise ValueError("latent frame contains an unsupported primitive")
    mapping = cast("Mapping[str, object]", value)
    tag = mapping.get("$")
    if tag == "tensor":
        if set(mapping) != {"$", "data", "dtype", "shape"}:
            raise ValueError("latent frame contains an invalid tensor record")
        dtype = mapping.get("dtype")
        shape = mapping.get("shape")
        data = mapping.get("data")
        if type(dtype) is not str or not isinstance(shape, list) or type(data) is not str:
            raise ValueError("latent frame contains an invalid tensor record")
        shape_items = cast("list[object]", shape)
        if any(type(n) is not int or n < 0 for n in shape_items):
            raise ValueError("latent frame contains an invalid tensor shape")
        shape_values = cast("list[int]", shape_items)
        try:
            raw = base64.b64decode(data, validate=True)
        except ValueError as exc:
            raise ValueError("latent frame contains invalid tensor bytes") from exc
        item_size = _DTYPE_SIZES.get(dtype)
        expected_size = 1
        for dimension in shape_values:
            expected_size *= dimension
        if item_size is None or len(raw) != expected_size * item_size:
            raise ValueError("latent frame tensor byte length does not match dtype and shape")
        record = EncodedLatentTensor(dtype, tuple(shape_values), raw)
        return record if tensor_decoder is None else tensor_decoder(record)
    if tag == "sparse-support":
        expected = {
            "$",
            "batch_counts",
            "coordinates",
            "origin",
            "resolution",
            "support_id",
            "voxel_size",
        }
        if set(mapping) != expected:
            raise ValueError("latent frame contains an invalid sparse support")
        coordinates = decode(mapping["coordinates"])
        batch_counts = decode(mapping["batch_counts"])
        origin = decode(mapping["origin"])
        voxel_size = decode(mapping["voxel_size"])
        resolution = mapping["resolution"]
        support_id = mapping["support_id"]
        raw_batch_counts = (
            cast("tuple[object, ...]", batch_counts) if type(batch_counts) is tuple else ()
        )
        raw_origin = cast("tuple[object, ...]", origin) if type(origin) is tuple else ()
        raw_voxel_size = cast("tuple[object, ...]", voxel_size) if type(voxel_size) is tuple else ()
        if (
            not raw_batch_counts
            or any(type(count) is not int or count < 1 for count in raw_batch_counts)
            or type(resolution) is not int
            or resolution < 1
            or len(raw_origin) != 3
            or any(type(item) is not float for item in raw_origin)
            or len(raw_voxel_size) != 3
            or any(type(item) is not float or item <= 0.0 for item in raw_voxel_size)
            or type(support_id) is not str
        ):
            raise ValueError("latent frame contains invalid sparse support metadata")
        record = EncodedSparseSupport(
            coordinates,
            cast("tuple[int, ...]", raw_batch_counts),
            resolution,
            cast("tuple[float, float, float]", raw_origin),
            cast("tuple[float, float, float]", raw_voxel_size),
            support_id,
        )
        _validate_encoded_sparse_support(record)
        return record if sparse_support_factory is None else sparse_support_factory(record)
    if tag == "sparse-latent":
        if set(mapping) != {"$", "features", "support"}:
            raise ValueError("latent frame contains an invalid sparse latent")
        record = EncodedSparseLatent(
            decode(mapping["support"]),
            decode(mapping["features"]),
        )
        _validate_encoded_sparse_latent(record)
        return record if sparse_latent_factory is None else sparse_latent_factory(record)
    if tag == "multi":
        if set(mapping) != {"$", "streams"}:
            raise ValueError("latent frame contains an invalid multi-stream record")
        streams = mapping.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ValueError("latent frame contains invalid multi-stream topology")
        pairs: list[tuple[str, object]] = []
        for item in cast("list[object]", streams):
            if not isinstance(item, list):
                raise ValueError("latent frame contains an invalid stream record")
            stream_item = cast("list[object]", item)
            if len(stream_item) != 2 or type(stream_item[0]) is not str:
                raise ValueError("latent frame contains an invalid stream record")
            pairs.append(
                (
                    stream_item[0],
                    decode(stream_item[1]),
                )
            )
        roles = tuple(role for role, _ in pairs)
        if len(set(roles)) != len(roles) or any(not role for role in roles):
            raise ValueError("latent frame contains invalid stream roles")
        return (
            EncodedMultiStreamLatent(tuple(pairs))
            if multi_stream_factory is None
            else multi_stream_factory(pairs)
        )
    if tag == "tuple":
        if set(mapping) != {"$", "items"}:
            raise ValueError("latent frame contains an invalid tuple")
        items = mapping.get("items")
        if not isinstance(items, list):
            raise ValueError("latent frame contains an invalid tuple")
        return tuple(decode(item) for item in cast("list[object]", items))
    if tag == "map":
        if set(mapping) != {"$", "items"} or not isinstance(mapping.get("items"), Mapping):
            raise ValueError("latent frame contains an invalid mapping")
        items = cast("Mapping[object, object]", mapping["items"])
        if any(type(key) is not str for key in items):
            raise ValueError("latent frame contains an invalid mapping key")
        return {cast("str", key): decode(item) for key, item in items.items()}
    raise ValueError(f"latent frame contains unknown tag {tag!r}")


def decode_latent(
    data: bytes,
    *,
    tensor_decoder: TensorDecoder | None = None,
    multi_stream_factory: MultiStreamFactory | None = None,
    sparse_support_factory: SparseSupportFactory | None = None,
    sparse_latent_factory: SparseLatentFactory | None = None,
) -> object:
    prefix_length = len(LATENT_CODEC_MAGIC)
    if not data.startswith(LATENT_CODEC_MAGIC) or len(data) <= prefix_length:
        raise ValueError("invalid latent codec framing")
    if data[prefix_length] != LATENT_CODEC_VERSION:
        raise ValueError("unsupported latent codec version")
    try:
        tree = json.loads(data[prefix_length + 1 :])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid latent codec payload") from exc
    value = _decode_tree(
        tree,
        tensor_decoder,
        multi_stream_factory,
        sparse_support_factory,
        sparse_latent_factory,
    )
    if not isinstance(value, Mapping) or "samples" not in value:
        raise ValueError("latent frame must contain a samples mapping entry")
    return cast("object", value)


def validate_latent_encoded(data: bytes, meta: Mapping[str, object]) -> None:
    del meta
    decode_latent(data)


def latent_fingerprint(type_id: str) -> Callable[[object], str]:
    def fingerprint(value: object) -> str:
        return stable_hash([type_id.encode("utf-8"), encode_latent(value)])

    return fingerprint


__all__ = [
    "EncodedLatentTensor",
    "EncodedMultiStreamLatent",
    "EncodedSparseLatent",
    "EncodedSparseSupport",
    "LATENT_CODEC_MAGIC",
    "LATENT_CODEC_VERSION",
    "decode_latent",
    "encode_latent",
    "latent_fingerprint",
    "validate_latent_encoded",
]
