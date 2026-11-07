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
import os
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_values import (
    ASSET_BASE_TYPE,
    GIBIBYTE,
    MEBIBYTE,
    TypeRegistry,
    Value,
    bind_video_sources,
    decode_video,
    effective_video_facts,
    encode_video,
    iter_value_tree,
    parse_asset_type_id,
    render_video_original,
    validate_video_encoded,
    value_children,
    video_fingerprint,
    video_meta,
    video_rendition_mime,
    video_source,
)
from dinkster_values.video_codec import VIDEO_INLINE_LIMIT, coerce_video, video_from_source
from dinkster_values.video_probe import probe_video

from .audio import bind_audio_value
from .declared import MOUNTS_ENV, produced_output_scratch, resolver_from_env
from .identity import AssetError, digest_bytes, digest_file
from .model import AssetRef, AssetResolver, LocalAssetRef, LocalAssetResolver
from .mounts import MountSnapshotResolver

ASSET_TYPE = ASSET_BASE_TYPE
"""Aliases dinkster_values.ASSET_BASE_TYPE (typed assets pinned the constant
down there so TypeRegistry can resolve ``asset<...>`` ids without importing
upward); this package keeps its historical name."""


def metadata_asset_references(
    type_id: str, metadata: Mapping[str, object], *, include_asset_value: bool = True
) -> Iterator[AssetRef]:
    """Read declared external identities, never payload fingerprints or blob digests."""
    raw = metadata.get("asset_refs", [])
    if not isinstance(raw, list):
        raise AssetError("asset_refs must be a list")
    items = list(cast("list[object]", raw))
    if include_asset_value and (
        type_id == ASSET_BASE_TYPE or parse_asset_type_id(type_id) is not None
    ):
        items.append(metadata)
    for item in items:
        if not isinstance(item, Mapping):
            raise AssetError("asset reference must be a mapping")
        wire = cast("Mapping[str, object]", item)
        if wire.get("verification") == "unverified":
            LocalAssetRef.from_wire(wire)
        else:
            yield AssetRef.from_wire(wire)


def value_asset_references(
    values: Mapping[str, Value],
    *,
    include_asset_values: bool = True,
    complete_registry: TypeRegistry | None = None,
) -> dict[str, AssetRef]:
    """Collect identities through recursive value containers without decoding payloads.

    Supplying complete_registry requires an exhaustive declaration for every
    atom. A declared codec or absent asset_refs alone never establishes it.
    """
    if complete_registry is not None and not include_asset_values:
        raise AssetError("complete asset references cannot exclude asset values")
    references: dict[str, AssetRef] = {}
    for value in values.values():
        for child in iter_value_tree(value):
            if complete_registry is not None and value_children(child) is None:
                try:
                    complete = complete_registry.spec(child.type_id).asset_references_complete
                except KeyError:
                    complete = False
                if not complete:
                    raise AssetError(f"asset references are incomplete for {child.type_id}")
            for ref in metadata_asset_references(
                child.type_id, child.meta.entries, include_asset_value=include_asset_values
            ):
                previous = references.setdefault(ref.digest, ref)
                if previous.size != ref.size:
                    raise AssetError("source references disagree about byte size")
    return references


def register_asset_type(
    registry: TypeRegistry,
    resolver: AssetResolver | None = None,
    *,
    type_id: str = ASSET_TYPE,
) -> None:
    """Register a bound asset identity to this process's resolver (or none:
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
            raise AssetError(f"cannot encode {type(obj).__name__} as {type_id}")
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
        type_id,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=meta,
        coerce=coerce,
        asset_references_complete=type_id == ASSET_TYPE,
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
        digest,
        source.name if isinstance(source, Path) else "video",
        size,
        resolver=resolver_from_env(),
    )


def admit_video_source(source: bytes | Path | AssetRef | LocalAssetRef) -> dict[str, object]:
    """Admit indexed files by reference; publish only produced byte streams."""
    if isinstance(source, AssetRef):
        with source.open():
            pass
        return video_from_source(source)
    if isinstance(source, LocalAssetRef):
        source.local_path()
        verified = source.verified()
        if verified is None:
            raise AssetError("VIDEO file selection must be indexed before admission")
        return admit_video_source(verified)
    if isinstance(source, Path):
        snapshot = os.environ.get(MOUNTS_ENV)
        if snapshot:
            resolver = MountSnapshotResolver(snapshot)
            path = source.resolve(strict=True)
            for kind in ("media/video", "directory/input"):
                for root in resolver.roots_for_kind(kind):
                    root = root.resolve()
                    if path.is_relative_to(root):
                        return admit_video_source(
                            resolver.bind_relative_file(
                                kind, root, path.relative_to(root).as_posix()
                            )
                        )
        raise AssetError("VIDEO file selection requires an indexed asset reference")
    size = len(source)
    if size > GIBIBYTE:
        raise AssetError("VIDEO source exceeds 1 GiB")
    if size <= VIDEO_INLINE_LIMIT:
        return video_from_source(source)
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
        if "complete_audio" in clip:
            clip["complete_audio"] = bind_audio_value(clip["complete_audio"], resolver)
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
        asset_references_complete=True,
    )
    registry.register_rendition(
        type_id, "original", mime=video_rendition_mime, render=render_video_original
    )
