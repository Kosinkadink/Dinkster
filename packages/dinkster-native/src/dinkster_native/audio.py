"""Lazy AUDIO transport with torch conversion only at legacy waveform access."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_assets import resolver_from_env
from dinkster_assets.audio import bind_audio_value
from dinkster_values import TypeRegistry
from dinkster_values.audio_codec import (
    audio_fingerprint,
    audio_meta,
    decode_audio,
    encode_audio,
    validate_audio_encoded,
)
from dinkster_values.audio_lazy import LazyAudio

AUDIO_V1_NAME = "AUDIO"


class TorchAudio(LazyAudio):
    def __missing__(self, key: str) -> object:
        source = self.get("source")
        if key == "waveform" and not self.get("edits") and isinstance(source, Mapping):
            pcm = cast("Mapping[str, object]", source).get("pcm")
            if hasattr(pcm, "detach"):
                return pcm
            if pcm is not None:
                torch = cast(Any, importlib.import_module("torch"))
                return torch.from_numpy(pcm)
        result = super().__missing__(key)
        if key == "waveform":
            if hasattr(result, "detach"):
                return result
            torch = cast(Any, importlib.import_module("torch"))
            return torch.from_numpy(result)
        return result


def _decode_torch(data: bytes) -> object:
    return TorchAudio(bind_audio_value(decode_audio(data), resolver_from_env()))


def register_audio_type(registry: TypeRegistry, type_id: str) -> None:
    """Register comfy.AUDIO with the upstream torch runtime form."""
    registry.register(
        type_id,
        encode=encode_audio,
        decode=_decode_torch,
        coerce=lambda obj: TorchAudio(bind_audio_value(obj, resolver_from_env())),
        fingerprint=audio_fingerprint(type_id),
        meta=audio_meta,
        validate_encoded_buffer=validate_audio_encoded,
    )
