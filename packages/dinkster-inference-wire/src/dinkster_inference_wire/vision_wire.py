"""Portable CLIP vision attributes without importing the model runtime."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_values import CLIP_VISION_OUTPUT_TYPE, decode_latent, encode_latent, stable_hash

from .media_wire import materialize_latent


def _attributes(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        attributes = cast("Mapping[object, object]", value)
    else:
        attributes = cast("Mapping[object, object]", vars(value))
    if any(type(key) is not str for key in attributes):
        raise TypeError("CLIP vision output attribute names must be strings")
    return cast("Mapping[str, object]", attributes)


def encode_clip_vision_output(value: object) -> bytes:
    """Reuse the structural tensor envelope, never pickle the runtime Output class."""
    return encode_latent({"samples": _attributes(value)})


def decode_clip_vision_output(data: bytes) -> dict[str, object]:
    """Keep every attribute as a host-readable scalar, tree or encoded tensor."""
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"} or not isinstance(frame["samples"], dict):
        raise ValueError("CLIP vision output frame must contain an attribute mapping")
    return cast("dict[str, object]", frame["samples"])


def validate_clip_vision_output(data: bytes, _metadata: Mapping[str, object]) -> None:
    decode_clip_vision_output(data)


def clip_vision_output_fingerprint(value: object) -> str:
    return stable_hash([CLIP_VISION_OUTPUT_TYPE.encode("utf-8"), encode_clip_vision_output(value)])


def materialize_clip_vision_output(value: object) -> Any:
    """Reconstruct the reset runtime Output only at an inference consumer."""
    process = importlib.import_module("dinkster_inference_wire.process")
    process.initialize_inference_process()
    vision = importlib.import_module("dinkster_inference.clip_vision")
    result = vision.Output()
    for name, item in _attributes(value).items():
        setattr(result, name, materialize_latent(item))
    return result
