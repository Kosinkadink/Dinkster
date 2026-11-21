"""Immutable content dependencies and derivations owned by the helper."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

from ._records import object_value
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def asset_digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"blake3:[0-9a-f]{64}", value) is None:
        raise LibraryError("invalid", "asset identity must be a canonical blake3 digest")
    return value


def asset_digests(value: object) -> list[str]:
    if not isinstance(value, list):
        raise LibraryError("invalid", "asset references must be a list")
    return sorted({asset_digest(item) for item in cast("list[object]", value)})


def dependencies(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise LibraryError("invalid", "dependencies must be a list")
    documents: list[dict[str, object]] = []
    for item in cast("list[object]", value):
        document = object_value(item, "dependency")
        asset_digest(document.get("digest"))
        documents.append(document)
    return documents


def manifests(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if operation not in (
        "dependencies.get",
        "dependencies.put",
        "derivations.get",
        "derivations.put",
    ):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    if (records.root / "library.sqlite").exists():
        raise LibraryError("migration", "Finish the complete library migration first")
    derived = operation.startswith("derivations.")
    writing = operation.endswith(".put")
    allowed = {"digest"}
    if writing:
        allowed |= {"manifest", "digests"} if derived else {"dependencies"}
    if set(arguments) - allowed:
        raise LibraryError("invalid", "unknown manifest field")
    identifier = asset_digest(arguments.get("digest"))
    table = "derivation_manifests" if derived else "dependency_manifests"
    key = "manifest" if derived else "dependencies"
    if not writing:
        row = records.db.execute(
            f"SELECT document FROM {table} WHERE digest=?", (identifier,)
        ).fetchone()
        return {key: json.loads(row[0]) if row else None}
    document = (
        object_value(arguments.get("manifest"), "derivation manifest")
        if derived
        else dependencies(arguments.get("dependencies"))
    )
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    roots = asset_digests(arguments.get("digests")) if derived else []
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        if derived:
            inserted = records.db.execute(
                "INSERT INTO derivation_manifests VALUES (?,?,?) ON CONFLICT(digest) DO NOTHING",
                (identifier, encoded, json.dumps(roots)),
            ).rowcount
            row = records.db.execute(
                "SELECT document,asset_digests FROM derivation_manifests WHERE digest=?",
                (identifier,),
            ).fetchone()
            assert row is not None
            if row[1] is None:
                # Older derivations can lack roots. Only an identical immutable
                # document may fill that gap; a disagreement rolls back the write.
                records.db.execute(
                    "UPDATE derivation_manifests SET asset_digests=? WHERE digest=?",
                    (json.dumps(roots), identifier),
                )
            elif asset_digests(json.loads(row[1])) != roots:
                raise LibraryError("conflict", "derivation asset references are immutable")
        else:
            inserted = records.db.execute(
                "INSERT INTO dependency_manifests VALUES (?,?) ON CONFLICT(digest) DO NOTHING",
                (identifier, encoded),
            ).rowcount
            row = records.db.execute(
                "SELECT document FROM dependency_manifests WHERE digest=?", (identifier,)
            ).fetchone()
            assert row is not None
        if json.dumps(json.loads(row[0]), sort_keys=True, separators=(",", ":")) != encoded:
            raise LibraryError("conflict", "content manifest is immutable")
        if inserted or (derived and row[1] is None):
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES (?,?,1)",
                ("derivation" if derived else "dependencies", identifier),
            )
    return {"created": bool(inserted)}
