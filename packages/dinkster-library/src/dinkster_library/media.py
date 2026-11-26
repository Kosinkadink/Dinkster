"""Canonical media facts and helper-owned scoped authority records."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from ._legacy_records import legacy_store_exists

if TYPE_CHECKING:
    from ._records import Records


MEDIA_FORMATS = (
    ("media/image", "image/png", "png"),
    ("media/image", "image/jpeg", "jpg"),
    ("media/image", "image/webp", "webp"),
    ("media/audio", "audio/wav", "wav"),
    ("media/audio", "audio/flac", "flac"),
    ("media/audio", "audio/mpeg", "mp3"),
    ("media/audio", "audio/ogg", "ogg"),
    ("media/audio", "audio/webm", "webm"),
    ("media/audio", "audio/mp4", "m4a"),
    ("media/video", "video/mp4", "mp4"),
    ("media/video", "video/webm", "webm"),
    ("media/model3d", "model/gltf-binary", "glb"),
    ("media/model3d", "model/ply", "ply"),
)


def grants(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    """Persist facts only after the owning engine has verified the supplied bytes."""
    from ._manifests import asset_digest
    from ._records import object_value
    from ._workflows import text
    from .client import LibraryError

    if legacy_store_exists(records.root):
        raise LibraryError("migration", "Finish the complete library migration first")
    writing = operation == "grants.put"
    allowed = {"scope", "grant"} if writing else {"scope", "digest", "kind"}
    if set(arguments) != allowed:
        raise LibraryError("invalid", "invalid media grant fields")
    scope = text(arguments["scope"], "grant scope")
    document = dict(object_value(arguments["grant"], "media grant")) if writing else arguments
    digest = asset_digest(document.get("digest"))
    kind = text(document.get("kind"), "grant kind")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)+", kind) is None:
        raise LibraryError("invalid", "grant kind must be a lowercase namespaced kind")
    key = (scope, digest, kind)
    if not writing:
        row = records.db.execute(
            "SELECT document FROM media_grants WHERE scope=? AND digest=? AND kind=?", key
        ).fetchone()
        return {"grant": json.loads(row[0]) if row else None}
    if document.get("scope", scope) != scope:
        raise LibraryError("invalid", "grant scope does not match its owner")
    document["scope"] = scope
    facts = (kind, document.get("mediaType"), document.get("extension"))
    if facts not in MEDIA_FORMATS and facts not in (
        ("data/latent", "application/x-comfy-latent", "latent"),
        ("asset/file", "application/octet-stream", "bin"),
    ):
        raise LibraryError("invalid", "media grant facts are not canonical")
    size = document.get("byteSize")
    minimum = 0 if kind == "asset/file" else 1
    if type(size) is not int or not minimum <= size <= 9223372036854775807:
        raise LibraryError(
            "invalid", f"media grant byte size must be a SQLite integer >= {minimum}"
        )
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        row = records.db.execute(
            "SELECT document FROM media_grants WHERE scope=? AND digest=? AND kind=?", key
        ).fetchone()
        if row is not None:
            if row[0] != encoded:
                raise LibraryError(
                    "conflict", "media authority is immutable for this scope, digest and kind"
                )
            return {"grant": json.loads(row[0]), "created": False}
        records.db.execute("INSERT INTO media_grants VALUES (?,?,?,?)", (*key, encoded))
        records.db.execute(
            "INSERT INTO changes(kind,id,revision) VALUES ('grant',?,1)", (json.dumps(key),)
        )
    return {"grant": document, "created": True}
