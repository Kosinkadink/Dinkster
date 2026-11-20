"""Additive acquisition leads and independently replaceable source layers."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

from ._browse import record_cursor, record_page
from ._manifests import asset_digest
from ._records import object_value
from ._workflows import text
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def provenance_document(value: object) -> dict[str, object]:
    document = dict(object_value(value, "provenance record"))
    document["digest"] = asset_digest(document.get("digest"))
    urls = document.get("sources", [])
    if not isinstance(urls, list) or any(
        not isinstance(url, str) for url in cast("list[object]", urls)
    ):
        raise LibraryError("invalid", "provenance sources must be a list of strings")
    document["sources"] = list(dict.fromkeys(cast("list[str]", urls)))
    for field in ("license", "note"):
        scalar = document.get(field, "")
        if not isinstance(scalar, str):
            raise LibraryError("invalid", f"provenance {field} must be a string")
        document[field] = scalar
    document["metadata"] = object_value(document.get("metadata", {}), "provenance metadata")
    _encoded(document)
    return document


def _encoded(document: dict[str, object]) -> str:
    return json.dumps(document, allow_nan=False, sort_keys=True)


def merge_provenance(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    merged = {**first, **second}
    merged["sources"] = list(
        dict.fromkeys(cast("list[str]", first["sources"]) + cast("list[str]", second["sources"]))
    )
    for field in ("license", "note"):
        merged[field] = second[field] or first[field]
    merged["metadata"] = {
        **object_value(first["metadata"], "metadata"),
        **object_value(second["metadata"], "metadata"),
    }
    return merged


def _get(records: Records, digest: str, source: str | None) -> dict[str, object] | None:
    rows = records.db.execute(
        "SELECT p.document FROM provenance_records p "
        "LEFT JOIN provenance_layers l ON l.source=p.source "
        "WHERE p.digest=? AND (? IS NULL OR p.source=?) "
        "ORDER BY COALESCE(l.position,0)",
        (digest, source, source),
    ).fetchall()
    result: dict[str, object] | None = None
    for row in rows:
        document = object_value(json.loads(row[0]), "stored provenance")
        result = document if result is None else merge_provenance(result, document)
    return result


def replace_layer(records: Records, source: str, incoming: dict[str, dict[str, object]]) -> bool:
    """Replace one layer inside the caller's transaction, preserving layer order."""
    previous = records.db.execute(
        "SELECT 1 FROM provenance_records WHERE source=? LIMIT 1", (source,)
    ).fetchone()
    records.db.execute("DELETE FROM provenance_records WHERE source=?", (source,))
    if incoming:
        records.db.execute(
            "INSERT INTO provenance_layers(source) VALUES (?) ON CONFLICT DO NOTHING", (source,)
        )
    else:
        records.db.execute("DELETE FROM provenance_layers WHERE source=?", (source,))
    for digest, document in incoming.items():
        records.db.execute(
            "INSERT INTO provenance_records VALUES (?,?,?)", (source, digest, _encoded(document))
        )
    return bool(previous)


def provenance(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    if any(
        (records.root / name).exists()
        for name in ("library.sqlite", "provenance.json", "resolver-indexes.json")
    ):
        raise LibraryError("migration", "Finish the complete library migration first")
    allowed = {
        "sources.provenance.add": {"record"},
        "sources.provenance.replace": {"source", "records"},
        "sources.provenance.remove": {"source"},
        "sources.provenance.get": {"digest", "source"},
        "sources.provenance.list": {"source", "limit", "cursor"},
        "sources.provenance.layers": {"limit", "cursor"},
    }
    if operation not in allowed:
        raise LibraryError("operation", f"unknown library operation: {operation}")
    if set(arguments) - allowed[operation]:
        raise LibraryError("invalid", "unknown provenance field")
    source: str | None = None
    if "source" in arguments:
        raw_source = arguments["source"]
        if not isinstance(raw_source, str) or len(raw_source) > 512:
            raise LibraryError(
                "invalid", "provenance source must be a string of at most 512 characters"
            )
        source = raw_source
        if source:
            text(source, "provenance source")
    replacing = operation == "sources.provenance.replace"
    removing = operation == "sources.provenance.remove"
    adding = operation == "sources.provenance.add"
    if (replacing or removing) and not source:
        raise LibraryError("invalid", "named provenance source must be nonempty")
    incoming: dict[str, dict[str, object]] = {}
    if adding:
        document = provenance_document(arguments.get("record"))
        incoming[cast(str, document["digest"])] = document
        source = ""
    if replacing:
        raw = arguments.get("records")
        if not isinstance(raw, list):
            raise LibraryError("invalid", "provenance records must be a list")
        for value in cast("list[object]", raw):
            document = provenance_document(value)
            digest = cast(str, document["digest"])
            incoming[digest] = (
                merge_provenance(incoming[digest], document) if digest in incoming else document
            )
    with records.db:
        # A read snapshot keeps a paged merge consistent across all selected digests.
        records.db.execute("BEGIN IMMEDIATE" if adding or replacing or removing else "BEGIN")
        if adding or replacing or removing:
            assert source is not None
            if source.startswith("resolver-index:"):
                raise LibraryError("invalid", "resolver layers require atomic resolver publication")
            removed = False
            if adding:
                digest = next(iter(incoming))
                previous = _get(records, digest, "")
                document = (
                    merge_provenance(previous, incoming[digest]) if previous else incoming[digest]
                )
                records.db.execute(
                    "INSERT INTO provenance_records VALUES (?,?,?) "
                    "ON CONFLICT(source,digest) DO UPDATE SET document=excluded.document",
                    (source, digest, _encoded(document)),
                )
            else:
                removed = replace_layer(records, source, incoming)
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('provenance',?,1)", (source,)
            )
            if adding:
                return {"record": _get(records, next(iter(incoming)), None)}
            return {"removed": removed} if removing else {}
        if operation == "sources.provenance.get":
            return {"record": _get(records, asset_digest(arguments.get("digest")), source)}
        layers = operation == "sources.provenance.layers"
        # Hash escaped source names so the cursor remains small at the source-name bound.
        bound: dict[str, object] = {
            "provenance": "layers" if layers else "records",
            "source": hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
        }
        limit, after = record_page(arguments, bound)
        if limit > 100:
            raise LibraryError("invalid", "provenance page size must be from 1 to 100")
        position = after[1] if after is not None else ""
        if layers:
            rows = records.db.execute(
                "SELECT source FROM provenance_layers WHERE source>? ORDER BY source LIMIT ?",
                (position, limit + 1),
            ).fetchall()
        else:
            rows = records.db.execute(
                "SELECT DISTINCT digest FROM provenance_records "
                "WHERE digest>? AND (? IS NULL OR source=?) ORDER BY digest LIMIT ?",
                (position, source, source, limit + 1),
            ).fetchall()
        return {
            "items": [row[0] if layers else _get(records, row[0], source) for row in rows[:limit]],
            "cursor": record_cursor(bound, (0, rows[limit - 1][0])) if len(rows) > limit else None,
        }
