"""Portable media registrations and lazy latent tensor materialization."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_values import (
    IMAGE_BATCH_MERGER_ID,
    IMAGE_FILE_DECODER_ID,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    PNG_CONTAINER_VERSION,
    SVG_TYPE,
    EncodedLatentTensor,
    EncodedMultiStreamLatent,
    TypeRegistry,
    TypeSpec,
    audio_input,
    decode_image_array,
    decode_image_file,
    decode_latent,
    decode_svg,
    encode_image_array,
    encode_latent,
    encode_svg,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    latent_fingerprint,
    mask_array_meta,
    merge_image_batches,
    prepare_image_array_encoding,
    render_image_png,
    render_mask_png,
    svg_fingerprint,
    svg_meta,
    validate_image_encoded,
    validate_latent_encoded,
    validate_svg_encoded,
)
from dinkster_values.audio_codec import audio_window, effective_audio_facts
from dinkster_values.audio_lazy import AUDIO_WINDOW_LIMIT

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


def materialize_audio(value: object) -> Mapping[str, object]:
    """Materialize a complete bounded AUDIO waveform for a Torch consumer."""
    if not isinstance(value, Mapping):
        raise TypeError("audio value must be a mapping")
    audio = cast("Mapping[str, object]", value)
    if "waveform" in audio:
        normalized = cast("Mapping[str, object]", audio_input(audio))
        waveform = normalized["waveform"]
        if hasattr(waveform, "detach"):
            return normalized
        torch = importlib.import_module("torch")
        return {**normalized, "waveform": torch.as_tensor(waveform)}

    facts = effective_audio_facts(audio)
    frames = facts["frames"]
    if frames is None:
        raise ValueError("AUDIO has unknown extent and cannot be fully materialized")
    if frames * facts["batch"] * facts["channels"] * 4 > AUDIO_WINDOW_LIMIT:
        raise ValueError("AUDIO whole waveform exceeds the 32 MiB allocation budget")
    window = audio_window(audio, 0, frames)
    torch = importlib.import_module("torch")
    return {**window, "waveform": torch.as_tensor(window["waveform"])}


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
    if SVG_TYPE not in registry:
        registry.register(
            SVG_TYPE,
            encode=encode_svg,
            decode=decode_svg,
            fingerprint=svg_fingerprint(SVG_TYPE),
            meta=svg_meta,
            validate_encoded=validate_svg_encoded,
            validate_encoded_buffer=validate_svg_encoded,
        )
    registered.append(registry.spec(SVG_TYPE))
    return tuple(registered)


__all__ = ["materialize_audio", "materialize_image", "materialize_latent", "register_media_types"]
