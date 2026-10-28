"""Tensor-worker codecs for the shared IMAGE and MASK values.

Workers decode the shared image-array bytes into Torch tensors. Native
media host registration decodes the same bytes into NumPy arrays and
provides PNG renditions without importing Torch.

Torch imports lazily inside the decoder so schema tooling stays Torch-free.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, cast

from dinkster_values import TypeRegistry
from dinkster_values.image_codec import (
    IMAGE_BATCH_MERGER_ID,
    IMAGE_FILE_DECODER_ID,
    copy_media_semantics,
    decode_image_array,
    decode_image_file,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    mask_array_meta,
    merge_image_batches,
    prepare_image_array_encoding,
    validate_image_encoded,
)

IMAGE_V1_NAME = "IMAGE"
"""The exact value type string that gets this codec."""

MASK_V1_NAME = "MASK"
"""MASK shares IMAGE's byte contract but has mask metadata and rendition."""


def _decode_torch(data: bytes) -> object:
    """npy bytes -> torch tensor, the runtime form v1 nodes expect."""
    return _to_torch(decode_image_array(data))


def register_image_type(registry: TypeRegistry, type_id: str) -> None:
    """Register an image type with the tensor decoder; callers guard duplicates."""
    registry.register(
        type_id,
        encode=encode_image_array,
        decode=_decode_torch,
        prepare_buffer_encoding=prepare_image_array_encoding,
        fingerprint=image_array_fingerprint(type_id),
        meta=mask_array_meta if type_id == MASK_V1_NAME else image_array_meta,
        validate_encoded=validate_image_encoded,
        validate_encoded_buffer=validate_image_encoded,
    )


def _to_torch(array: object) -> object:
    numpy = cast("Any", importlib.import_module("numpy"))
    torch = cast("Any", importlib.import_module("torch"))
    return copy_media_semantics(array, torch.from_numpy(numpy.ascontiguousarray(array)))


def register_image_asset_providers(registry: TypeRegistry, type_id: str) -> None:
    """Decode and merge image assets into tensors using the native host's
    provider identities. Cache identity follows semantics, not representation."""

    def decode(asset: object) -> object:
        return _to_torch(decode_image_file(asset))

    def merge(batches: Sequence[object]) -> object:
        return _to_torch(merge_image_batches(batches))

    registry.register_asset_decoder(type_id, provider_id=IMAGE_FILE_DECODER_ID, decode=decode)
    registry.register_batch_merge(type_id, provider_id=IMAGE_BATCH_MERGER_ID, merge=merge)
