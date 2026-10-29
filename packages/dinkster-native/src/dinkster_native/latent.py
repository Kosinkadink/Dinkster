"""Torch runtime half of the shared comfy.LATENT structural codec."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

from dinkster_values import (
    EncodedLatentTensor,
    TypeRegistry,
    decode_latent,
    encode_latent,
    latent_fingerprint,
    validate_latent_encoded,
)


def _decode_tensor(record: EncodedLatentTensor) -> object:
    torch = importlib.import_module("torch")
    dtype = getattr(torch, record.dtype, None)
    if dtype is None:
        raise ValueError(f"unsupported latent tensor dtype {record.dtype!r}")
    if not record.data:
        return torch.empty(record.shape, dtype=dtype)
    storage = torch.frombuffer(bytearray(record.data), dtype=torch.uint8)
    return storage.view(dtype).reshape(record.shape).clone()


def _multi_stream(pairs: Sequence[tuple[str, object]]) -> object:
    multi_stream = importlib.import_module("dinkster_inference").MultiStreamLatent
    return multi_stream.from_pairs(pairs)


def decode_torch_latent(data: bytes) -> object:
    return decode_latent(
        data,
        tensor_decoder=_decode_tensor,
        multi_stream_factory=_multi_stream,
    )


def register_latent_type(registry: TypeRegistry, type_id: str) -> None:
    registry.register(
        type_id,
        encode=encode_latent,
        decode=decode_torch_latent,
        fingerprint=latent_fingerprint(type_id),
        validate_encoded=validate_latent_encoded,
    )


__all__ = ["decode_torch_latent", "register_latent_type"]
