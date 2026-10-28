"""Read environment-owned project references without changing library state."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetSource,
    AssetVault,
    LibraryRecord,
    LibraryStore,
    PackAssetCatalog,
    ProvenanceStore,
    RemoteSource,
)
from dinkster_registry.install import Lockfile
from dinkster_schema import canonical_name
from dinkster_workers import PackManifest, load_manifest

from .workflow_requirements import (
    PackRequirement,
    RequirementError,
    WorkflowRequirements,
    _object,
    manifest_unavailability,
    plan_workflow_requirements,
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


def plan_project_requirements(
    root: Path,
    *,
    variant: str,
    source_generation: Path | None = None,
    catalog: Path | None = None,
) -> dict[str, object]:
    """Plan from operator-supplied, already admitted release metadata.

    Environment acquisition verifies the catalog and staged source artifacts.
    This boundary reads them, never downloads or grants trust to workflow leads.
    Only this project's verified vault bytes count as locally materialized.
    """
    if variant not in ("linux-cpu", "win-cpu", "win-nvidia", "win-nvidia-cu126"):
        raise RequirementError(f"unknown engine variant: {variant}")
    requirements, inputs = load_project_requirements(root)
    installed = (
        Lockfile.from_record_json((source_generation / "dinkster.lock").read_text("utf-8"))
        if source_generation is not None
        else Lockfile()
    )
    if requirements.packs and catalog is None:
        raise RequirementError("trusted target pack catalog is unavailable; supply --catalog")
    target = (
        Lockfile.from_record_json(catalog.read_text("utf-8")) if catalog is not None else Lockfile()
    )
    manifests: dict[str, PackManifest] = {}
    if catalog is not None:
        sources = catalog.parent / "pack-sources"
        for pack in target.packs:
            path = sources / pack.pack / "dinkster-pack.toml"
            manifest = load_manifest(path)
            if canonical_name(manifest.name) != pack.pack:
                raise RequirementError(f"target manifest identity does not match pack {pack.pack}")
            manifests[pack.pack] = manifest
    declared = PackAssetCatalog()
    declared.replace_all((pack, None, manifest.assets) for pack, manifest in manifests.items())
    library = root / "library"
    vault = AssetVault(library / "vault") if (library / "vault").is_dir() else None
    provenance = ProvenanceStore(library / "provenance.json")

    def available(asset: AssetRef) -> bool:
        if vault is None:
            return False
        try:
            with replace(asset, resolver=vault).open():
                return True
        except (AssetError, OSError):
            return False

    def leads(digest: str) -> list[AssetSource]:
        sources: list[AssetSource] = [
            RemoteSource(url)
            for url in provenance.sources(digest)
            if url.startswith(("http://", "https://"))
        ]
        need = declared.need_for(digest)
        if need is not None:
            sources.extend(need.sources)
        return sources

    def unavailable(pack: PackRequirement) -> str | None:
        entry = target.get(pack.pack)
        if entry is None or entry.version != pack.version:
            return "saved version is unavailable in the selected target release"
        return manifest_unavailability(pack, manifests[entry.pack], variant)

    plan = plan_workflow_requirements(
        requirements,
        installed_packs={pack.pack: pack.version for pack in installed.packs},
        asset_available=available,
        sources_for=leads,
        pack_unavailable=unavailable,
    )
    return {"requirements": requirements.to_wire(), "plan": plan.to_wire(), "inputs": list(inputs)}
