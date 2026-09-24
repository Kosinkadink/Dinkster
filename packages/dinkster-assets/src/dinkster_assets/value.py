"""The ``dinkster.asset`` value type: assets as envelopes (DESIGN 3.12 + 3.2).

The registration makes three deliberate choices:

- **The fingerprint is the content digest.** A model input's cache identity
  is its bytes, wherever they live - location-independent cache keys for
  model-dependent nodes fall out with zero special-casing (hazard H4).
- **The codec carries identity + metadata only, never bytes.** An AssetRef
  crossing a process/machine boundary is a couple hundred bytes of JSON;
  content moves only when a worker actually materializes it, from its own
  configured stores.
- **Coercion accepts wire-shaped mappings.** Graph literals and frontend
  submissions carry ``{"digest": ..., "name": ...}``; the engine's wrap
  turns them into resolver-bound AssetRefs before any node sees them.
"""

from __future__ import annotations
from dinkster_values import GIBIBYTE, MEBIBYTE

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dinkster_values import (
    ASSET_BASE_TYPE,
    TypeRegistry,
    bind_video_sources,
    decode_video,
    effective_video_facts,
    encode_video,
    render_video_original,
    validate_video_encoded,
    video_fingerprint,
    video_meta,
    video_rendition_mime,
    video_source,
)
from dinkster_values.video_codec import VIDEO_INLINE_LIMIT, coerce_video, video_from_source
from dinkster_values.video_probe import probe_video

from .audio import bind_audio_value
from .declared import produced_asset_vault
from .identity import AssetError, digest_bytes, digest_file
from .model import AssetRef, AssetResolver

ASSET_TYPE = ASSET_BASE_TYPE
"""Aliases dinkster_values.ASSET_BASE_TYPE (typed assets pinned the constant
down there so TypeRegistry can resolve ``asset<...>`` ids without importing
upward); this package keeps its historical name."""


def register_asset_type(registry: TypeRegistry, resolver: AssetResolver | None = None) -> None:
    """Register ``dinkster.asset`` bound to this process's resolver (or none:
    refs still flow, cache, and interrogate; only local_path() requires a
    resolver where bytes are actually read)."""

    def coerce(obj: object) -> object:
        if isinstance(obj, AssetRef):
            if obj.resolver is None and resolver is not None:
                return AssetRef(
                    digest=obj.digest,
                    name=obj.name,
                    size=obj.size,
                    media_type=obj.media_type,
                    virtual_path=obj.virtual_path,
                    resolver=resolver,
                )
            return obj
        if isinstance(obj, Mapping):
            return AssetRef.from_wire(cast("Mapping[str, object]", obj), resolver=resolver)
        raise AssetError(
            "asset inputs must be an AssetRef or a mapping with a 'digest' - "
            f"got {type(obj).__name__} (asset literals carry identity, never "
            "filesystem paths)"
        )

    def encode(obj: object) -> bytes:
        if not isinstance(obj, AssetRef):
            raise AssetError(f"cannot encode {type(obj).__name__} as {ASSET_TYPE}")
        return json.dumps(obj.to_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(data: bytes) -> object:
        wire: object = json.loads(data)
        if not isinstance(wire, Mapping):
            raise AssetError("asset wire form must be a JSON object")
        return AssetRef.from_wire(cast("Mapping[str, object]", wire), resolver=resolver)

    def fingerprint(obj: object) -> str:
        assert isinstance(obj, AssetRef)  # coerce ran first
        return obj.digest

    def meta(obj: object) -> Mapping[str, object]:
        assert isinstance(obj, AssetRef)  # coerce ran first
        return obj.to_wire()

    registry.register(
        ASSET_TYPE,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=meta,
        coerce=coerce,
    )


def _publish_video_source(source: bytes | Path) -> AssetRef:
    size = len(source) if isinstance(source, bytes) else source.stat().st_size
    if size > GIBIBYTE:
        raise AssetError("VIDEO source exceeds 1 GiB")
    vault = produced_asset_vault()
    if vault is None:
        raise AssetError(
            "publishing VIDEO sources requires DINKSTER_PACK_SCRATCH or DINKSTER_ASSET_VAULT"
        )
    digest = digest_bytes(source) if isinstance(source, bytes) else digest_file(source)
    with vault.writer(digest) as writer:
        if isinstance(source, bytes):
            for offset in range(0, size, MEBIBYTE):
                writer.write(source[offset : offset + MEBIBYTE])
        else:
            with source.open("rb") as handle:
                while chunk := handle.read(MEBIBYTE):
                    writer.write(chunk)
        writer.commit()
    return AssetRef(
        digest, source.name if isinstance(source, Path) else "video", size, resolver=vault
    )


def admit_video_source(source: bytes | Path) -> dict[str, object]:
    """Probe before publishing local media as a portable VIDEO value."""
    size = len(source) if isinstance(source, bytes) else source.stat().st_size
    if size <= VIDEO_INLINE_LIMIT:
        return video_from_source(source if isinstance(source, bytes) else source.read_bytes())
    probe = probe_video(source)
    effective_video_facts({"probe": probe, "edits": []})
    return video_from_source(_publish_video_source(source))


def bind_video_value(
    obj: object,
    resolver: AssetResolver | None = None,
    *,
    for_audio_extraction: bool = False,
) -> dict[str, object]:
    """Bind clips and publish large sources, or only audio-bearing sources for extraction."""
    value = coerce_video(obj)
    if "timeline" in value:
        return bind_video_sources(value, lambda wire: AssetRef.from_wire(wire, resolver))

    def publish(clip: dict[str, object]) -> None:
        source = clip.get("source")
        if isinstance(source, bytes) and (
            cast("Mapping[str, object]", clip["probe"])["audio"]
            if for_audio_extraction
            else len(source) > VIDEO_INLINE_LIMIT
        ):
            clip["source"] = _publish_video_source(source)
        if isinstance(source, AssetRef) and source.resolver is None:
            clip["source"] = AssetRef.from_wire(source.to_wire(), resolver)
        if "components" in clip:
            components = dict(cast("Mapping[str, object]", clip["components"]))
            if components["audio"] is not None:
                components["audio"] = bind_audio_value(components["audio"], resolver)
            clip["components"] = components
        for edit in cast("list[dict[str, object]]", clip["edits"]):
            for child in cast("list[dict[str, object]]", edit.get("concat", [])):
                publish(child)

    publish(value)
    value = bind_video_sources(value, lambda wire: AssetRef.from_wire(wire, resolver))

    def verify(clip: dict[str, object]) -> None:
        source = clip.get("source")
        if isinstance(source, AssetRef) and source.resolver is not None:
            video_source(clip)
        for edit in cast("list[dict[str, object]]", clip["edits"]):
            for child in cast("list[dict[str, object]]", edit.get("concat", [])):
                verify(child)

    verify(value)
    return value


def register_video_value_type(
    registry: TypeRegistry, type_id: str, resolver: AssetResolver | None = None
) -> None:
    """Register portable VIDEO with source references bound to the host's asset chain."""
    registry.register(
        type_id,
        encode=encode_video,
        decode=lambda data: bind_video_value(decode_video(data), resolver),
        coerce=lambda obj: bind_video_value(obj, resolver),
        fingerprint=video_fingerprint(type_id),
        meta=video_meta,
        validate_encoded_buffer=validate_video_encoded,
    )
    registry.register_rendition(
        type_id, "original", mime=video_rendition_mime, render=render_video_original
    )
