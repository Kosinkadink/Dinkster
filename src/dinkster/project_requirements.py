"""Read environment-owned project references without changing library state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from dinkster_assets import AssetRef, AssetVault, LibraryRecord, LibraryStore

from .workflow_requirements import (
    PackRequirement,
    RequirementError,
    WorkflowRequirements,
    _object,
    project_requirements,
)


def load_project_requirements(
    root: Path,
) -> tuple[WorkflowRequirements, tuple[dict[str, object], ...]]:
    """Snapshot explicitly registered scoped records and verified local bytes.

    The local environment operator owns project.json; a saved workflow cannot
    add references or authorize another scope. Activation must compare both
    requirements and input revisions/digests before replacing current.
    """
    project = _object(json.loads((root / "project.json").read_text("utf-8")), "project")
    if project.get("format") != "dinkster.project/1":
        raise RequirementError("expected a dinkster.project/1 project")
    additions = _object(project.get("additions", {}), "project additions")
    if set(additions) - {"packs", "assets"}:
        raise RequirementError("unknown project addition fields")
    packs: list[PackRequirement] = []
    raw_packs = additions.get("packs", [])
    raw_assets = additions.get("assets", [])
    references = project.get("workflows", [])
    if not all(isinstance(value, list) for value in (raw_packs, raw_assets, references)):
        raise RequirementError("project workflows, added packs and added assets must be lists")
    for raw in cast("list[object]", raw_packs):
        pack = _object(raw, "added pack")
        if set(pack) != {"pack", "version"} or any(
            not isinstance(value, str) or not value.strip() for value in pack.values()
        ):
            raise RequirementError("added pack requires exact nonempty pack and version strings")
        packs.append(PackRequirement(cast(str, pack["pack"]), cast(str, pack["version"])))
    explicit = WorkflowRequirements(
        tuple(packs),
        tuple(
            AssetRef.from_wire(_object(raw, "added asset"))
            for raw in cast("list[object]", raw_assets)
        ),
    )
    selected: set[tuple[str, str]] = set()
    for raw in cast("list[object]", references):
        ref = _object(raw, "workflow reference")
        if set(ref) != {"scope", "id"} or any(
            not isinstance(value, str) or not value.strip() for value in ref.values()
        ):
            raise RequirementError("workflow reference requires nonempty scope and id strings")
        selected.add((cast(str, ref["scope"]), cast(str, ref["id"])))
    if not selected:
        return project_requirements((), additions=explicit), ()
    library = root / "library"
    if not (library / "vault").is_dir():
        raise RequirementError("project workflow vault is unavailable")
    vault = AssetVault(library / "vault")
    store = LibraryStore(library / "library.sqlite", read_only=True)
    try:
        records: list[LibraryRecord] = []
        documents: list[object] = []
        for scope, record_id in sorted(selected):
            record = store.get(scope, record_id)
            if record is None:
                raise RequirementError(f"workflow record {scope}/{record_id} is unavailable")
            if record.media_type != "application/x-dinkster-workflow+json":
                raise RequirementError(f"record {scope}/{record_id} is not a workflow")
            data = AssetRef(record.digest, record.name, 0, resolver=vault).read_bytes()
            documents.append(json.loads(data))
            records.append(record)
        requirements = project_requirements(documents, additions=explicit)
        for record in records:
            current = store.get(record.scope, record.id)
            if current is None or (current.revision, current.digest) != (
                record.revision,
                record.digest,
            ):
                raise RequirementError("project workflow inputs changed; request a new plan")
        return requirements, tuple(
            {
                "scope": record.scope,
                "id": record.id,
                "revision": record.revision,
                "digest": record.digest,
            }
            for record in records
        )
    finally:
        store.close()
