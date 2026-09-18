"""Registration of the structural ``dinkster.latent`` boundary type.

The shared codec carries ordinary tensors and ordered, role-labeled
multi-stream tensors under one versioned envelope. Registration and encode
are torch-free; only decode materializes tensors, importing torch lazily in
the process that consumes the value.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_values import (
    EncodedLatentTensor,
    EncodedSparseLatent,
    EncodedSparseSupport,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    latent_fingerprint,
    validate_latent_encoded,
)

from .latents import MultiStreamLatent
from .sparse import SparseSupport

LATENT_TYPE_ID = "dinkster.latent"


def _decode_multi_stream(pairs: Sequence[tuple[str, object]]) -> object:
    return MultiStreamLatent[object].from_pairs(pairs)


def _refuse_multi_stream(pairs: Sequence[tuple[str, object]]) -> object:
    del pairs
    raise ValueError(
        "dinkster.latent carries single-stream latents only; multistream "
        "latents cross the boundary under their own type id"
    )


def _decode_torch_tensor(record: EncodedLatentTensor) -> object:
    torch = importlib.import_module("torch")
    dtype = getattr(torch, record.dtype, None)
    if dtype is None:
        raise ValueError(f"unsupported latent tensor dtype {record.dtype!r}")
    if not record.data:
        return torch.empty(record.shape, dtype=dtype)
    storage = torch.frombuffer(bytearray(record.data), dtype=torch.uint8)
    return storage.view(dtype).reshape(record.shape).clone()


def _decode_sparse_support(record: EncodedSparseSupport) -> object:
    sparse = importlib.import_module("dinkster_inference_torch.sparse")
    support = sparse.make_sparse_support(
        record.coordinates,
        record.batch_counts,
        record.resolution,
        record.origin,
        record.voxel_size,
    )
    if support.support_id != record.support_id:
        raise ValueError("sparse support coordinate digest changed during decode")
    return support


def _decode_sparse_latent(record: EncodedSparseLatent) -> object:
    if type(record.support) is not SparseSupport:
        raise TypeError("decoded sparse latent support has the wrong type")
    sparse = importlib.import_module("dinkster_inference_torch.sparse")
    return cast(
        "object",
        cast("Any", sparse).pack_sparse_latent(
            cast("Any", record).support,
            record.features,
        ),
    )


def encode_single_stream_latent(value: object) -> bytes:
    encoded = encode_latent(value)
    validate_single_stream_latent_encoded(encoded, {})
    return encoded


def decode_single_stream_latent(data: bytes) -> object:
    return decode_latent(
        data,
        tensor_decoder=_decode_torch_tensor,
        multi_stream_factory=_refuse_multi_stream,
        sparse_support_factory=_decode_sparse_support,
        sparse_latent_factory=_decode_sparse_latent,
    )


def validate_single_stream_latent_encoded(data: bytes, meta: Mapping[str, object]) -> None:
    del meta
    decode_latent(data, tensor_decoder=None, multi_stream_factory=_refuse_multi_stream)


def _decode_latent_wire(data: bytes) -> object:
    return decode_latent(
        data,
        tensor_decoder=_decode_torch_tensor,
        multi_stream_factory=_decode_multi_stream,
        sparse_support_factory=_decode_sparse_support,
        sparse_latent_factory=_decode_sparse_latent,
    )


def _encode_latent_wire(value: object) -> bytes:
    encoded = encode_latent(value)
    validate_latent_encoded(encoded, {})
    return encoded


def register_latent_type(registry: TypeRegistry) -> TypeSpec:
    return registry.register(
        LATENT_TYPE_ID,
        encode=_encode_latent_wire,
        decode=_decode_latent_wire,
        fingerprint=latent_fingerprint(LATENT_TYPE_ID),
        validate_encoded=validate_latent_encoded,
    )


__all__ = [
    "LATENT_TYPE_ID",
    "decode_single_stream_latent",
    "encode_single_stream_latent",
    "register_latent_type",
    "validate_single_stream_latent_encoded",
]
