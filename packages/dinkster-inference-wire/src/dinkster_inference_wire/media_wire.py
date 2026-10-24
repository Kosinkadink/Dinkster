"""Portable media registrations and lazy latent tensor materialization."""

from __future__ import annotations

import importlib
from typing import Any, cast

from dinkster_values import (
    IMAGE_BATCH_MERGER_ID,
    IMAGE_FILE_DECODER_ID,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    PNG_CONTAINER_VERSION,
    EncodedLatentTensor,
    EncodedMultiStreamLatent,
    TypeRegistry,
    TypeSpec,
    decode_image_array,
    decode_image_file,
    decode_latent,
    encode_image_array,
    encode_latent,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    latent_fingerprint,
    mask_array_meta,
    merge_image_batches,
    prepare_image_array_encoding,
    render_image_png,
    render_mask_png,
    validate_image_encoded,
    validate_latent_encoded,
)

from .latents import MultiStreamLatent


def materialize_image(value: object) -> Any:
    """Load portable pixels at a Torch consumer without moving live tensors."""
    if hasattr(value, "detach"):
        return value
    torch = importlib.import_module("torch")
    return torch.as_tensor(value)


def materialize_latent(value: object) -> Any:
    """Reconstruct portable tensor records only where a node needs Torch.

    Live tensors retain their device, dtype and storage. The host can inspect
    and forward the structural wire value without importing Torch.
    """
    if isinstance(value, EncodedLatentTensor):
        torch = importlib.import_module("torch")
        dtype = getattr(torch, value.dtype)
        if not value.data:
            return torch.empty(value.shape, dtype=dtype)
        storage = torch.frombuffer(bytearray(value.data), dtype=torch.uint8)
        return storage.view(dtype).reshape(value.shape).clone()
    if isinstance(value, EncodedMultiStreamLatent):
        return MultiStreamLatent[Any].from_pairs(
            tuple((role, materialize_latent(payload)) for role, payload in value.streams)
        )
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        converted = {key: materialize_latent(item) for key, item in mapping.items()}
        return (
            mapping if all(converted[key] is item for key, item in mapping.items()) else converted
        )
    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        converted = [materialize_latent(item) for item in sequence]
        if all(left is right for left, right in zip(sequence, converted, strict=True)):
            return sequence
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


def register_media_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Register exact media atoms with process-independent byte contracts."""
    registered: list[TypeSpec] = []
    for type_id, meta, render in (
        (IMAGE_TYPE, image_array_meta, render_image_png),
        (MASK_TYPE, mask_array_meta, render_mask_png),
    ):
        if type_id not in registry:
            registry.register(
                type_id,
                encode=encode_image_array,
                decode=decode_image_array,
                prepare_buffer_encoding=prepare_image_array_encoding,
                fingerprint=image_array_fingerprint(type_id),
                meta=meta,
                input_convert=image_input,
                validate_encoded=validate_image_encoded,
                validate_encoded_buffer=validate_image_encoded,
            )
        registered.append(registry.spec(type_id))
        if not any(item.kind == "png" for item in registry.renditions_of(type_id)):
            registry.register_rendition(
                type_id, "png", mime="image/png", render=render, version=PNG_CONTAINER_VERSION
            )
    if registry.asset_decoder_for(IMAGE_TYPE) is None:
        registry.register_asset_decoder(
            IMAGE_TYPE, provider_id=IMAGE_FILE_DECODER_ID, decode=decode_image_file
        )
    if registry.batch_merge_for(IMAGE_TYPE) is None:
        registry.register_batch_merge(
            IMAGE_TYPE, provider_id=IMAGE_BATCH_MERGER_ID, merge=merge_image_batches
        )
    if LATENT_TYPE not in registry:
        registry.register(
            LATENT_TYPE,
            encode=encode_latent,
            decode=decode_latent,
            fingerprint=latent_fingerprint(LATENT_TYPE),
            validate_encoded=validate_latent_encoded,
        )
    registered.append(registry.spec(LATENT_TYPE))
    return tuple(registered)


__all__ = ["materialize_image", "materialize_latent", "register_media_types"]
