"""The worker half of the comfy.IMAGE byte contract.

comfy.IMAGE payloads must be loadable on BOTH sides of the compat
boundary: v1 nodes consume torch tensors in this worker, while the engine
process - torchless by design - renders previews from the same value
(/api/values PNG renditions). The default codec cannot serve that split
(pickle only unpickles where torch lives), so comfy.IMAGE registers the
shared image-array codec (dinkster_values.image_codec): npy bytes on the
wire, decoded HERE back into torch tensors, decoded host-side into numpy.
The host half - numpy decode plus the PNG rendition - registers through
the compat PackSpecs' host_types hook (dinkster.comfy_compose).

torch imports lazily inside the decoder: this module stays importable by
tooling and tests that have numpy but no torch.
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
"""The v1 type string that gets this codec (translate.py special-cases it;
the host mirrors the derived ``comfy.IMAGE`` id in comfy_compose)."""

MASK_V1_NAME = "MASK"
"""MASK shares the image-array byte contract (``[H, W]`` / ``[B, H, W]``
per the codec's layout contract), so ``comfy.MASK`` and ``dinkster.mask``
are one value type at the boundary; only the host rendition differs
(render_mask_png)."""

IMAGE_TYPE_EQUIVALENCE_PROVIDER = "dinkster.image-array-type-equivalence@1"
"""Stable identity for the shared Comfy/native image and mask byte contract."""


def _decode_torch(data: bytes) -> object:
    """npy bytes -> torch tensor, the runtime form v1 nodes expect."""
    return _to_torch(decode_image_array(data))


def register_image_type(registry: TypeRegistry, type_id: str) -> None:
    """Register ``type_id`` (comfy.IMAGE, comfy.MASK, or a dinkster spelling)
    with the shared image codec in a compat worker. Callers guard against
    double registration - translation registers each opaque type exactly
    once; register_native_types checks ``type_id in registry`` first."""
    registry.register(
        type_id,
        encode=encode_image_array,
        decode=_decode_torch,
        prepare_buffer_encoding=prepare_image_array_encoding,
        fingerprint=image_array_fingerprint(type_id),
        meta=mask_array_meta if type_id in {"comfy.MASK", "dinkster.mask"} else image_array_meta,
        validate_encoded=validate_image_encoded,
        validate_encoded_buffer=validate_image_encoded,
    )


def register_image_type_equivalences(registry: TypeRegistry) -> None:
    """Bind Comfy and native spellings after their shared codecs exist."""
    for compat_type, native_type in (
        ("comfy.IMAGE", "dinkster.image"),
        ("comfy.MASK", "dinkster.mask"),
    ):
        if compat_type in registry and native_type in registry:
            registry.register_type_equivalence(
                compat_type,
                native_type,
                provider_id=IMAGE_TYPE_EQUIVALENCE_PROVIDER,
            )


def _to_torch(array: object) -> object:
    numpy = cast("Any", importlib.import_module("numpy"))
    torch = cast("Any", importlib.import_module("torch"))
    return copy_media_semantics(array, torch.from_numpy(numpy.ascontiguousarray(array)))


def register_image_asset_providers(registry: TypeRegistry, type_id: str) -> None:
    """The typed-asset providers for ``type_id`` (comfy.IMAGE) in a compat
    worker: asset<comfy.IMAGE> decode and the comfy.IMAGE batch merge.

    Same provider identities as the host's numpy registrations
    (comfy_compose.register_comfy_host_types) - identity is the decode
    SEMANTICS the cache key must capture, not the runtime representation -
    but the results convert to torch, the form v1 node code consumes."""

    def decode(asset: object) -> object:
        return _to_torch(decode_image_file(asset))

    def merge(batches: Sequence[object]) -> object:
        return _to_torch(merge_image_batches(batches))

    registry.register_asset_decoder(type_id, provider_id=IMAGE_FILE_DECODER_ID, decode=decode)
    registry.register_batch_merge(type_id, provider_id=IMAGE_BATCH_MERGER_ID, merge=merge)
