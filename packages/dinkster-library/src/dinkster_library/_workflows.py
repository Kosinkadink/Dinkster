"""Project ownership and engine-supplied facts for saved workflows."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING, cast

from ._records import object_value, project_id
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


WORKFLOW_MEDIA_TYPE = "application/x-dinkster-workflow+json"


def text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LibraryError("invalid", f"{name} must be a nonempty string")
    return value


def workflow_records(records: Records) -> list[dict[str, object]]:
    path = records.root / "library.sqlite"
    if not path.exists():
        return [
            object_value(json.loads(row[0]), "saved record")
            for row in records.db.execute("SELECT document FROM saved_records WHERE deleted=0")
        ]
    # This bridge reads the stable direct-engine table without opening it for
    # migration. It must also see writes by engines that know nothing of facts.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        documents: list[dict[str, object]] = []
        for row in connection.execute("SELECT * FROM records"):
            document = dict(row)
            document["mediaType"] = document.pop("media_type")
            document["labels"] = json.loads(document["labels"])
            documents.append(document)
        return documents
    finally:
        connection.close()


def workflow(records: Records, operation: str, arguments: dict[str, object]) -> dict[str, object]:
    scope = text(arguments.get("scope"), "workflow scope")
    identifier = text(arguments.get("id"), "workflow id")
    current = next(
        (
            row
            for row in workflow_records(records)
            if row["scope"] == scope and row["id"] == identifier
        ),
        None,
    )
    if operation == "workflows.read":
        if current is None:
            return {"record": None}
        return {"record": current}
    if operation not in ("workflows.get", "workflows.record"):
        raise LibraryError("operation", f"unknown library operation: {operation}")
    with records.db:
        records.db.execute("BEGIN IMMEDIATE")
        if not (records.root / "library.sqlite").exists():
            stored = records.db.execute(
                "SELECT document FROM saved_records WHERE scope=? AND id=? AND deleted=0",
                (scope, identifier),
            ).fetchone()
            current = object_value(json.loads(stored[0]), "saved record") if stored else None
        row = records.db.execute(
            "SELECT project,document FROM workflow_facts WHERE scope=? AND id=?",
            (scope, identifier),
        ).fetchone()
        project = current.get("project", "default") if current else "default"
        if row:
            project = row[0]
        document: dict[str, object] = (
            object_value(json.loads(row[1]), "workflow facts") if row else {"revision": 0}
        )
        if operation == "workflows.record":
            if current is None or current["mediaType"] != WORKFLOW_MEDIA_TYPE:
                raise LibraryError("missing", "workflow record is unavailable")
            revision = arguments.get("revision")
            if type(revision) is not int or revision != document["revision"]:
                raise LibraryError("stale", "workflow facts changed; read them again before saving")
            requested = project_id(arguments.get("project", project))
            owner = records.get_project(requested)
            if owner is None or owner["deleted"]:
                raise LibraryError("missing", "project is unavailable")
            if (row or "project" in current) and requested != project:
                raise LibraryError("invalid", "copy the workflow to put it in another project")
            project = requested
            if "facts" in arguments:
                facts = object_value(arguments["facts"], "workflow facts")
                validate_facts(facts)
                if facts["digest"] != current["digest"]:
                    raise LibraryError("stale", "workflow document changed; read it again")
                document["facts"] = facts
            document["revision"] = revision + 1
            records.db.execute(
                "INSERT INTO workflow_facts VALUES (?,?,?,?) ON CONFLICT(scope,id) "
                "DO UPDATE SET document=excluded.document",
                (scope, identifier, project, json.dumps(document, allow_nan=False)),
            )
            records.db.execute(
                "INSERT INTO changes(kind,id,revision) VALUES ('workflow',?,?)",
                (json.dumps([scope, identifier]), document["revision"]),
            )
        facts = document.get("facts")
        known = (
            isinstance(facts, dict)
            and current is not None
            and cast("dict[str, object]", facts).get("digest") == current["digest"]
        )
        return {
            "project": project,
            "revision": document["revision"],
            "facts": facts if known else None,
        }


def validate_facts(facts: dict[str, object]) -> None:
    digest = facts.get("digest")
    if not isinstance(digest, str) or re.fullmatch(r"blake3:[0-9a-f]{64}", digest) is None:
        raise LibraryError("invalid", "facts require a blake3 document digest")
    packs, models = facts.get("packs"), facts.get("models")
    if not isinstance(packs, list) or not isinstance(models, list):
        raise LibraryError("invalid", "facts require packs and models lists")
    if type(facts.get("machinePaths")) is not bool:
        raise LibraryError("invalid", "facts must state whether the document has machine paths")
    for raw in cast("list[object]", packs):
        pack = object_value(raw, "pack facts")
        text(pack.get("pack"), "pack id")
        for key in ("version", "artifactDigest", "source", "publisher"):
            if pack.get(key) is not None:
                text(pack[key], f"pack {key}")
    for raw in cast("list[object]", models):
        model = object_value(raw, "model reference")
        for key in ("location", "name", "kind"):
            text(model.get(key), f"model {key}")
        digest = model.get("digest")
        if digest is not None and (
            not isinstance(digest, str) or re.fullmatch(r"blake3:[0-9a-f]{64}", digest) is None
        ):
            raise LibraryError("invalid", "model digest must be blake3 or null")


def needs(
    records: Records, identifier: str, *, include_workflows: object = False
) -> dict[str, object]:
    if type(include_workflows) is not bool:
        raise LibraryError("invalid", "includeWorkflows must be a boolean")
    project = records.get_project(identifier)
    if project is None or project["deleted"]:
        raise LibraryError("missing", "project is unavailable")
    query = "SELECT scope,id,project,document FROM workflow_facts"
    parameters: tuple[str, ...] = ()
    if identifier != "default":
        query += " WHERE project=?"
        parameters = (identifier,)
    ownership = {
        (row[0], row[1]): (row[2], object_value(json.loads(row[3]), "workflow facts"))
        for row in records.db.execute(query, parameters)
    }
    packs: dict[str, dict[str, object]] = {}
    unknown: list[dict[str, object]] = []
    workflows: list[dict[str, object]] = []
    for row in workflow_records(records):
        if row["mediaType"] != WORKFLOW_MEDIA_TYPE:
            continue
        owner, document = ownership.get(
            (row["scope"], row["id"]), (row.get("project", "default"), {})
        )
        if owner != identifier:
            continue
        workflows.append({key: row[key] for key in ("scope", "id", "digest", "name")})
        facts = document.get("facts")
        if (
            not isinstance(facts, dict)
            or cast("dict[str, object]", facts).get("digest") != row["digest"]
        ):
            unknown.append({"scope": row["scope"], "id": row["id"]})
            continue
        for pack in cast("list[dict[str, object]]", facts["packs"]):
            packs[json.dumps(pack, sort_keys=True)] = pack
    return {
        "packs": [packs[key] for key in sorted(packs)],
        "extra": object_value(project["packs"], "packs")["extra"],
        "unknown": unknown,
        **({"workflows": workflows} if include_workflows else {}),
    }
