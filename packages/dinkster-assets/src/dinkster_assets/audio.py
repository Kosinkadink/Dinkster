"""Bind portable AUDIO sources and publish large compatibility PCM without copying it."""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import os
import struct
import tempfile
from collections.abc import Mapping
from typing import Any, cast

from dinkster_values import TypeRegistry
from dinkster_values.audio_codec import (
    AUDIO_INLINE_LIMIT,
    AUDIO_WAVEFORM_LIMITS,
    AUDIO_WAVEFORM_VERSION,
    AUDIO_WINDOW_LIMITS,
    AUDIO_WINDOW_VERSION,
    audio_fingerprint,
    audio_from_source,
    audio_meta,
    bind_audio_sources,
    coerce_audio,
    decode_audio,
    encode_audio,
    normalize_audio_waveform_request,
    normalize_audio_window_request,
    render_audio_wav,
    render_audio_waveform,
    render_audio_window,
    validate_audio_encoded,
)
from dinkster_values.audio_lazy import LazyAudio
from dinkster_values.storage import audio_input

from .identity import AssetError, new_hasher
from .model import AssetRef, AssetResolver
from .vault import AssetVault


def bind_audio_value(obj: object, resolver: AssetResolver | None = None) -> LazyAudio:
    value = coerce_audio(obj)
    source = value["source"]
    if isinstance(source, Mapping) and "pcm" in source:
        pcm = cast(Any, source["pcm"])
        if pcm.nbytes > AUDIO_INLINE_LIMIT:
            import numpy as np

            root = os.environ.get("DINKSTER_ASSET_VAULT")
            if not root:
                raise AssetError("large AUDIO PCM requires a configured DINKSTER_ASSET_VAULT")
            vault = AssetVault(root)
            with tempfile.TemporaryFile() as file:
                file.write(struct.pack("<Q", cast(int, source["sample_rate"])))
                # numpy streams contiguous arrays directly to disk, rather than a BytesIO copy.
                np.save(file, np.ascontiguousarray(pcm), allow_pickle=False)
                size = file.tell()
                file.seek(0)
                hasher = new_hasher()
                while chunk := file.read(MEBIBYTE):
                    hasher.update(chunk)
                digest = "blake3:" + hasher.hexdigest()
                file.seek(0)
                with vault.writer(digest) as writer:
                    while chunk := file.read(MEBIBYTE):
                        writer.write(chunk)
                    writer.commit()
            published = audio_from_source(AssetRef(digest, "audio.pcm", size, resolver=vault))
            cast(dict[str, object], published["probe"])["layout"] = cast(
                Mapping[str, object], value["probe"]
            )["layout"]
            published["edits"] = value["edits"]
            value = published
    elif isinstance(source, AssetRef) and source.resolver is None and resolver is not None:
        value["source"] = AssetRef.from_wire(source.to_wire(), resolver)
    value["edits"] = [
        {"concat": [bind_audio_value(child, resolver) for child in edit["concat"]]}
        if "concat" in edit
        else edit
        for edit in cast("list[dict[str, Any]]", value["edits"])
    ]
    return bind_audio_sources(value, lambda wire: AssetRef.from_wire(wire, resolver))


def register_audio_value_type(
    registry: TypeRegistry, type_id: str, resolver: AssetResolver | None = None
) -> None:
    registry.register(
        type_id,
        encode=encode_audio,
        decode=lambda data: bind_audio_value(decode_audio(data), resolver),
        coerce=lambda obj: bind_audio_value(obj, resolver),
        fingerprint=audio_fingerprint(type_id),
        meta=audio_meta,
        input_convert=audio_input,
        validate_encoded_buffer=validate_audio_encoded,
    )
    registry.register_rendition(type_id, "wav", mime="audio/wav", render=render_audio_wav)
    registry.register_rendition(
        type_id,
        "waveform",
        mime="image/png",
        render=render_audio_waveform,
        version=AUDIO_WAVEFORM_VERSION,
        parameters=("batch", "waveform"),
        defaults={"batch": "0"},
        limits=AUDIO_WAVEFORM_LIMITS,
        normalize=normalize_audio_waveform_request,
    )
    registry.register_rendition(
        type_id,
        "window",
        mime="audio/wav",
        render=render_audio_window,
        version=AUDIO_WINDOW_VERSION,
        parameters=("batch", "window"),
        defaults={"batch": "0"},
        limits=AUDIO_WINDOW_LIMITS,
        normalize=normalize_audio_window_request,
    )
