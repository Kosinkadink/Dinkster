"""Create a project and its saved starting contents in one library transaction."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._legacy_records import legacy_store_exists
from ._records import make_directory, object_value, project_id
from ._saved import saved
from ._settings import preference
from ._workflows import WORKFLOW_MEDIA_TYPE, text
from .asset_identity import digest_bytes, digest_file
from .client import LibraryError

if TYPE_CHECKING:
    from ._records import Records


def create_project(records: Records, arguments: dict[str, object]) -> dict[str, object]:
    identifier = project_id(arguments.get("id"))
    created: list[Path] = []
    published: list[Path] = []
    try:
        with records.db:
            records.db.execute("BEGIN IMMEDIATE")
            return _create_project(records, arguments, identifier, created, published)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for path in reversed(created):
            try:
                path.rmdir()
            except OSError:
                # A concurrent writer's data is never removed during rollback.
                pass
        raise


def _create_project(
    records: Records,
    arguments: dict[str, object],
    identifier: str,
    created: list[Path],
    published: list[Path],
) -> dict[str, object]:
    existing = records.get_project(identifier)
    if existing is not None:
        return {"project": existing}
    start = object_value(arguments.get("startWith", {"kind": "empty"}), "startWith")
    kind = start.get("kind")
    if kind != "empty" and legacy_store_exists(records.root):
        raise LibraryError("migration", "Finish the complete library migration first")
    source: dict[str, object] | None = None
    contents: list[tuple[str, dict[str, object]]] = []
    workflow_data: bytes | None = None
    if kind == "project":
        if set(start) != {"kind", "project"}:
            raise LibraryError("invalid", "project startWith requires a project id")
        source_id = project_id(start.get("project"))
        source = records.get_project(source_id)
        if source is None or source["deleted"]:
            raise LibraryError("missing", "source project is unavailable")
        for scope, raw in records.db.execute(
            "SELECT scope,document FROM saved_records WHERE project=? AND deleted=0",
            (source_id,),
        ):
            original = object_value(json.loads(raw), "saved record")
            contents.append(
                (
                    scope,
                    {
                        key: original[key]
                        for key in ("name", "digest", "mediaType", "labels", "folder")
                        if key in original
                    },
                )
            )
    elif kind == "workflow":
        if set(start) != {"kind", "name", "content"}:
            raise LibraryError("invalid", "workflow startWith requires name and content")
        content = text(start.get("content"), "workflow content")
        try:
            parsed = json.loads(content)
            object_value(parsed, "workflow document")
            json.dumps(parsed, allow_nan=False)
            workflow_data = content.encode("utf-8")
        except (ValueError, UnicodeError) as error:
            raise LibraryError("invalid", f"invalid workflow JSON: {error}") from error
        contents.append(
            (
                "local",
                {
                    "name": start.get("name"),
                    "digest": digest_bytes(workflow_data),
                    "mediaType": WORKFLOW_MEDIA_TYPE,
                },
            )
        )
    elif kind != "empty" or set(start) != {"kind"}:
        raise LibraryError("invalid", "unknown startWith choice")
    if kind != "empty" and arguments.get("filesFolder") is None:
        raise LibraryError("invalid", "starting contents require a Files folder")
    document = records.register_project(
        identifier,
        arguments.get("name", identifier),
        files_folder=arguments.get("filesFolder"),
        created_folders=created,
    )
    if source is not None:
        document["packs"] = {"extra": object_value(source["packs"], "packs")["extra"]}
        folders = object_value(
            preference(records, "folders.projects.get", {})["folders"], "folders"
        )
        values = object_value(folders["values"], "folder preferences")
        memberships = object_value(values.get("projects", {}), "project memberships")
        selected = cast("list[str]", memberships.get(identifier, []))
        source_id = cast(str, source["id"])
        original = cast("list[str]", memberships.get(source_id, []))
        patch: dict[str, object] = {
            "projects": {identifier: list(dict.fromkeys((*original, *selected)))}
        }
        for field in ("excluded", "hiddenModels"):
            choices = object_value(values.get(field, {}), field)
            patch[field] = {identifier: choices.get(source_id, [])}
        preference(
            records,
            "folders.projects.patch",
            {"revision": folders["revision"], "patch": patch},
        )
        records.save_project(document)
    for scope, content_record in contents:
        saved(
            records,
            "records.create",
            {"scope": scope, "project": identifier, "record": content_record},
        )
    if workflow_data is not None:
        digest = digest_bytes(workflow_data)
        target = records.root / "vault" / digest[7:9] / digest[7:]
        make_directory(target.parent, created)
        with tempfile.NamedTemporaryFile(dir=target.parent) as staging:
            staging.write(workflow_data)
            staging.flush()
            os.fsync(staging.fileno())
            try:
                os.link(staging.name, target)
            except FileExistsError:
                if digest_file(target) != digest:
                    raise LibraryError("invalid", "stored workflow content is corrupt") from None
            else:
                published.append(target)
    return {"project": document}
