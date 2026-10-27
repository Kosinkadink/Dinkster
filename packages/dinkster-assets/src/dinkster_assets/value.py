"""The ``dinkster.asset`` value type: assets as envelopes (DESIGN 3.12 + 3.2).

The registration makes three deliberate choices:

- **Verified fingerprints are content digests.** Unverified local files
  instead use opaque, revocable binding tokens with an explicit prefix;
  those tokens never claim content identity or cross-host portability.
- **The codec carries identity + metadata only, never bytes.** An AssetRef
  crossing a process/machine boundary is a couple hundred bytes of JSON;
  content moves only when a worker actually materializes it, from its own
  configured stores.
- **Coercion accepts wire-shaped mappings.** Graph literals and frontend
  submissions carry ``{"digest": ..., "name": ...}``; the engine's wrap
  turns them into resolver-bound AssetRefs before any node sees them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from dinkster_values import (
    ASSET_BASE_TYPE,
    GIBIBYTE,
    MEBIBYTE,
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
from .declared import MOUNTS_ENV, produced_output_scratch
from .identity import AssetError, digest_bytes, digest_file
from .model import (
    AssetRef,
    AssetResolver,
    LocalAssetRef,
    LocalAssetResolver,
    MountAssetRef,
    MountAssetResolver,
)
from .mounts import MountSnapshotResolver, MountTable

ASSET_TYPE = ASSET_BASE_TYPE
"""Aliases dinkster_values.ASSET_BASE_TYPE (typed assets pinned the constant
down there so TypeRegistry can resolve ``asset<...>`` ids without importing
upward); this package keeps its historical name."""


def register_asset_type(registry: TypeRegistry, resolver: AssetResolver | None = None) -> None:
    """Register ``dinkster.asset`` bound to this process's resolver (or none:
    refs still flow, cache, and interrogate; only local_path() requires a
    resolver where bytes are actually read)."""

    def coerce(obj: object) -> object:
        if isinstance(obj, LocalAssetRef):
            if obj.resolver is None and isinstance(resolver, LocalAssetResolver):
                obj = replace(obj, resolver=resolver)
            obj.local_path()
            return obj.verified() or obj
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
            if cast("Mapping[str, object]", obj).get("verification") == "unverified":
                return coerce(
                    LocalAssetRef.from_wire(
                        cast("Mapping[str, object]", obj),
                        resolver=resolver if isinstance(resolver, LocalAssetResolver) else None,
                    )
                )
            return AssetRef.from_wire(cast("Mapping[str, object]", obj), resolver=resolver)
        raise AssetError(
            "asset inputs must be an AssetRef or a mapping with a 'digest' - "
            f"got {type(obj).__name__} (asset literals carry identity, never "
            "filesystem paths)"
        )

    def encode(obj: object) -> bytes:
        if not isinstance(obj, AssetRef | LocalAssetRef):
            raise AssetError(f"cannot encode {type(obj).__name__} as {ASSET_TYPE}")
        return json.dumps(obj.to_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(data: bytes) -> object:
        wire: object = json.loads(data)
        if not isinstance(wire, Mapping):
            raise AssetError("asset wire form must be a JSON object")
        if cast("Mapping[str, object]", wire).get("verification") == "unverified":
            return LocalAssetRef.from_wire(
                cast("Mapping[str, object]", wire),
                resolver=resolver if isinstance(resolver, LocalAssetResolver) else None,
            )
        return AssetRef.from_wire(cast("Mapping[str, object]", wire), resolver=resolver)

    def fingerprint(obj: object) -> str:
        if isinstance(obj, LocalAssetRef):
            return "unverified-local:" + obj.binding
        assert isinstance(obj, AssetRef)  # coerce ran first
        return obj.digest

    def meta(obj: object) -> Mapping[str, object]:
        assert isinstance(obj, AssetRef | LocalAssetRef)  # coerce ran first
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
    vault = produced_output_scratch()
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


def admit_video_source(
    source: bytes | Path | AssetRef | LocalAssetRef | MountAssetRef,
    *,
    publish: bool = False,
) -> dict[str, object]:
    """Reference granted files; bound copied or explicitly published sources to 1 GiB."""
    if isinstance(source, MountAssetRef):
        path = source.local_path()
        if not publish:
            return video_from_source(source)
        source = path
    if isinstance(source, AssetRef):
        if not publish and isinstance(source.resolver, MountAssetResolver):
            mounted = source.resolver.reference_path(source.local_path())
            if mounted is not None:
                return admit_video_source(mounted)
        with source.open() as handle:
            handle.seek(0, 2)
            if handle.tell() > GIBIBYTE:
                raise AssetError("VIDEO source exceeds 1 GiB")
        return video_from_source(source)
    if isinstance(source, LocalAssetRef):
        path = source.local_path()
        if publish:
            return admit_video_source(path, publish=True)
        if isinstance(source.resolver, (MountTable, MountSnapshotResolver)):
            return admit_video_source(source.resolver.reference_local(source.binding))
        with TemporaryDirectory(prefix="dinkster-video-") as directory:
            path = Path(directory) / "video"
            with source.open() as handle, path.open("wb") as output:
                size = 0
                while chunk := handle.read(MEBIBYTE):
                    size += len(chunk)
                    if size > GIBIBYTE:
                        raise AssetError("VIDEO source exceeds 1 GiB")
                    output.write(chunk)
            return admit_video_source(path, publish=True)
    if isinstance(source, Path) and not publish:
        import os

        snapshot = os.environ.get(MOUNTS_ENV)
        if snapshot:
            mounted = MountSnapshotResolver(snapshot).reference_path(source)
            if mounted is not None:
                return admit_video_source(mounted)
    size = len(source) if isinstance(source, bytes) else source.stat().st_size
    if size > GIBIBYTE:
        raise AssetError("VIDEO source exceeds 1 GiB")
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

    def reference(wire: Mapping[str, object]) -> AssetRef | MountAssetRef:
        if "mount" in wire:
            return MountAssetRef.from_wire(
                wire, resolver if isinstance(resolver, MountAssetResolver) else None
            )
        return AssetRef.from_wire(wire, resolver)

    if "timeline" in value:
        return bind_video_sources(value, reference)

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
        if isinstance(source, MountAssetRef) and source.resolver is None:
            clip["source"] = reference(source.to_wire())
        if "components" in clip:
            components = dict(cast("Mapping[str, object]", clip["components"]))
            if components["audio"] is not None:
                components["audio"] = bind_audio_value(components["audio"], resolver)
            clip["components"] = components
        for edit in cast("list[dict[str, object]]", clip["edits"]):
            for child in cast("list[dict[str, object]]", edit.get("concat", [])):
                publish(child)

    publish(value)
    value = bind_video_sources(value, reference)

    def verify(clip: dict[str, object]) -> None:
        source = clip.get("source")
        if isinstance(source, (AssetRef, MountAssetRef)) and source.resolver is not None:
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
