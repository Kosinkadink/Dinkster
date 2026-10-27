"""Portable camera, track and audio-encoder values using the latent tensor envelope."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import cast

from dinkster_values import (
    TRACKS_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

_AUDIO_ENCODER_LEGACY_FIELDS = frozenset(
    {"encoded_audio", "encoded_audio_all_layers", "audio_samples"}
)


def _decode(data: bytes, *, type_id: str) -> object:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError(f"{type_id} frame must contain only samples")
    value = frame["samples"]
    if type_id == WAN_CAMERA_EMBEDDING_TYPE:
        if not isinstance(value, EncodedLatentTensor):
            raise ValueError("WAN_CAMERA_EMBEDDING must contain a tensor")
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{type_id} must contain a mapping")
    mapping = cast("dict[str, object]", value)
    if type_id == TRACKS_TYPE:
        if not isinstance(mapping.get("track_path"), EncodedLatentTensor):
            raise ValueError("TRACKS track_path must be a tensor")
        if "track_visibility" in mapping and not isinstance(
            mapping["track_visibility"], EncodedLatentTensor
        ):
            raise ValueError("TRACKS track_visibility must be a tensor when present")
    else:
        if _AUDIO_ENCODER_LEGACY_FIELDS.intersection(mapping):
            if not isinstance(mapping.get("encoded_audio"), EncodedLatentTensor):
                raise ValueError("AUDIO_ENCODER_OUTPUT encoded_audio must be a tensor")
            layers = mapping.get("encoded_audio_all_layers")
            if not isinstance(layers, (tuple, list)) or any(
                not isinstance(layer, EncodedLatentTensor)
                for layer in cast("list[object] | tuple[object, ...]", layers)
            ):
                raise ValueError("AUDIO_ENCODER_OUTPUT layers must be an ordered tensor sequence")
            if type(mapping.get("audio_samples")) is not int:
                raise ValueError("AUDIO_ENCODER_OUTPUT audio_samples must be an integer")
        else:
            if not isinstance(mapping.get("audio_feature"), EncodedLatentTensor):
                raise ValueError("AUDIO_ENCODER_OUTPUT audio_feature must be a tensor")
            if type(mapping.get("fps")) is not float:
                raise ValueError("AUDIO_ENCODER_OUTPUT fps must be a float")
            if "audio_inject_scale" in mapping and type(mapping["audio_inject_scale"]) is not float:
                raise ValueError("AUDIO_ENCODER_OUTPUT audio_inject_scale must be a float")
    return mapping


def _encode(value: object, *, type_id: str) -> bytes:
    data = encode_latent({"samples": value})
    _decode(data, type_id=type_id)
    return data


def _fingerprint(value: object, *, type_id: str) -> str:
    return stable_hash([type_id.encode("utf-8"), _encode(value, type_id=type_id)])


def _validate(data: bytes, _metadata: Mapping[str, object], *, type_id: str) -> None:
    _decode(data, type_id=type_id)


def register_wan_type(registry: TypeRegistry, type_id: str) -> TypeSpec:
    return registry.register(
        type_id,
        encode=partial(_encode, type_id=type_id),
        decode=partial(_decode, type_id=type_id),
        fingerprint=partial(_fingerprint, type_id=type_id),
        validate_encoded=partial(_validate, type_id=type_id),
    )
