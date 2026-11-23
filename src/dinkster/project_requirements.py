"""Plan and materialize environment-owned requirements without changing library records."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetRef,
    AssetSource,
    AssetVault,
    PackAssetCatalog,
    ProvenanceStore,
    RemoteSource,
    acquire_need,
)
from dinkster_library import LibraryClient, LibraryError
from dinkster_registry.artifact import unpack_artifact
from dinkster_registry.install import Lockfile
from dinkster_schema import canonical_name
from dinkster_store import pack_archive_path
from dinkster_workers import PackManifest, load_manifest

from .workflow_requirements import (
    VARIANTS,
    PackRequirement,
    RequirementError,
    WorkflowRequirements,
    _object,
    generation_device,
    manifest_unavailability,
    plan_workflow_requirements,
    project_requirements,
)


def load_project_requirements(
    root: Path,
) -> tuple[WorkflowRequirements, tuple[dict[str, object], ...]]:
    """Snapshot explicitly registered scoped records and verified local bytes.

    The local environment operator owns environment.json; a saved workflow cannot
    add references or authorize another scope. Activation must compare both
    requirements and input revisions/digests before replacing current.
    """
    project = _object(json.loads((root / "environment.json").read_text("utf-8")), "project")
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
    library = project_library(root)
    if not (library / "vault").is_dir():
        raise RequirementError("project workflow vault is unavailable")
    vault = AssetVault(library / "vault")
    try:
        with LibraryClient(library) as helper:
            records: list[dict[str, object]] = []
            documents: list[object] = []
            for scope, record_id in sorted(selected):
                raw = helper.call("workflows.read", scope=scope, id=record_id)["record"]
                if raw is None:
                    raise RequirementError(f"workflow record {scope}/{record_id} is unavailable")
                record = dict(_object(raw, "workflow record"))
                if record["mediaType"] != "application/x-dinkster-workflow+json":
                    raise RequirementError(f"record {scope}/{record_id} is not a workflow")
                data = AssetRef(
                    cast(str, record["digest"]), cast(str, record["name"]), 0, resolver=vault
                ).read_bytes()
                documents.append(json.loads(data))
                records.append(record)
            requirements = project_requirements(documents, additions=explicit)
            for record in records:
                raw = helper.call("workflows.read", scope=record["scope"], id=record["id"])[
                    "record"
                ]
                if raw is None or any(
                    _object(raw, "workflow record")[key] != record[key]
                    for key in ("revision", "digest")
                ):
                    raise RequirementError("project workflow inputs changed; request a new plan")
            return requirements, tuple(
                {key: record[key] for key in ("scope", "id", "revision", "digest")}
                for record in records
            )
    except LibraryError as exc:
        raise RequirementError(
            f"cannot read project workflows through library helper: {exc}"
        ) from exc


def project_library(root: Path) -> Path:
    return Path(os.environ.get("DINKSTER_LIBRARY_ROOT", str(root / "dinkster-library")))


def plan_project_requirements(
    root: Path,
    *,
    variant: str,
    source_generation: Path | None = None,
    target_generation: Path | None = None,
    catalog_lock: Path | None = None,
    artifacts: Path | None = None,
    target_device: str | None = None,
) -> dict[str, object]:
    """Plan from operator-supplied, already admitted release metadata.

    Environment acquisition admits the catalog's exact artifact identities.
    This boundary verifies cached archives, never downloads or trusts workflow leads.
    Only this project's verified vault bytes count as locally materialized.
    """
    device = target_device
    if target_device is not None:
        from dinkster_workers.manifest import KNOWN_ACCELERATORS

        if target_device not in KNOWN_ACCELERATORS:
            raise RequirementError("unsupported acquired target device")
        if target_generation is None or catalog_lock is None or artifacts is None:
            raise RequirementError("acquired target device requires target, catalog and artifacts")
    elif variant not in VARIANTS:
        selected = target_generation if target_generation is not None else source_generation
        if selected is None:
            raise RequirementError("custom workflow target requires a snapshot")
        device = generation_device(selected, variant)
    requirements, inputs = load_project_requirements(root)
    installed = (
        Lockfile.from_record_json((source_generation / "dinkster.lock").read_text("utf-8"))
        if source_generation is not None
        else Lockfile()
    )
    if requirements.packs and catalog_lock is None:
        raise RequirementError("trusted target pack catalog is unavailable; supply --catalog-lock")
    target = (
        Lockfile.from_record_json(catalog_lock.read_text("utf-8"))
        if catalog_lock is not None
        else Lockfile()
    )
    if target.packs and artifacts is None:
        raise RequirementError("trusted target artifact cache is unavailable; supply --artifacts")
    manifests: dict[str, PackManifest] = {}
    if artifacts is not None:
        with TemporaryDirectory(prefix="dinkster-workflow-plan-") as scratch:
            for pack in target.packs:
                archive = pack_archive_path(artifacts, pack.artifact_digest)
                extracted = unpack_artifact(
                    archive, Path(scratch) / pack.pack, pack.artifact_digest
                )
                manifest = load_manifest(extracted / "dinkster-pack.toml")
                if canonical_name(manifest.name) != pack.pack:
                    raise RequirementError(
                        f"target manifest identity does not match pack {pack.pack}"
                    )
                manifests[pack.pack] = manifest
    declared = PackAssetCatalog()
    declared.replace_all((pack, None, manifest.assets) for pack, manifest in manifests.items())
    library = project_library(root)
    vault = AssetVault(library / "vault") if (library / "vault").is_dir() else None

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
            for url in (provenance.sources(digest) if provenance is not None else ())
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
        return manifest_unavailability(pack, manifests[entry.pack], variant, device=device)

    with ExitStack() as stack:
        provenance = (
            ProvenanceStore(stack.enter_context(LibraryClient(library)))
            if library.is_dir()
            else None
        )
        plan = plan_workflow_requirements(
            requirements,
            installed_packs={pack.pack: pack.version for pack in installed.packs},
            asset_available=available,
            sources_for=leads,
            pack_unavailable=unavailable,
        )
    return {"requirements": requirements.to_wire(), "plan": plan.to_wire(), "inputs": list(inputs)}


def apply_project_requirements(
    root: Path,
    *,
    approved: dict[str, object],
    generation: Path,
    variant: str,
    source_generation: Path | None = None,
    catalog_lock: Path | None = None,
    artifacts: Path | None = None,
) -> dict[str, object]:
    """Materialize only an unchanged approved plan over an admitted target snapshot."""
    from .generation_packs import load_generation_packs

    current = plan_project_requirements(
        root,
        variant=variant,
        source_generation=source_generation,
        target_generation=generation,
        catalog_lock=catalog_lock,
        artifacts=artifacts,
    )
    if current != approved:
        raise RequirementError("project requirement plan changed; request a new plan")
    actions = _object(current["plan"], "project plan")
    if actions["errors"]:
        raise RequirementError("project requirement plan has blocking errors")
    target_variant, specs = load_generation_packs(generation)
    if target_variant != variant:
        raise RequirementError("target snapshot variant differs from approved plan")
    target = Lockfile.from_record_json((generation / "dinkster.lock").read_text("utf-8"))
    selected = (
        Lockfile.from_record_json(catalog_lock.read_text("utf-8"))
        if catalog_lock is not None
        else Lockfile()
    )
    required = _object(current["requirements"], "project requirements")
    for raw in cast("list[object]", required["packs"]):
        pack = _object(raw, "required pack")
        installed = target.get(cast(str, pack["pack"]))
        if installed is None or installed.version != pack["version"]:
            raise RequirementError(f"target snapshot lacks approved pack {pack['pack']}")
        if installed != selected.get(installed.pack):
            raise RequirementError(f"target pack {installed.pack} differs from selected catalog")
    for raw in cast("list[object]", actions["assets"]):
        action = _object(raw, "asset action")
        for name in cast("list[str]", action.get("packagedFrom", [])):
            if target.get(name) is None or target.get(name) != selected.get(name):
                raise RequirementError(f"target pack {name} differs from selected catalog")
    declared = PackAssetCatalog()
    declared.replace_all(
        (canonical_name(manifest.name), manifest.path.parent, manifest.assets)
        for spec in specs
        for manifest in (load_manifest(spec.manifest),)
    )

    def revalidate(_source: AssetSource | None = None, _size: int = 0) -> None:
        requirements, inputs = load_project_requirements(root)
        if requirements.to_wire() != current["requirements"] or list(inputs) != current["inputs"]:
            raise RequirementError("project workflow inputs changed; request a new plan")

    vault = AssetVault(project_library(root) / "vault")
    for raw in cast("list[object]", actions["assets"]):
        revalidate()
        action = _object(raw, "asset action")
        ref = AssetRef.from_wire(action)
        sources: list[AssetSource] = [
            RemoteSource(url) for url in cast("list[str]", action["sources"])
        ]
        need = declared.need_for(ref.digest)
        if need is not None:
            sources.extend(
                source
                for source in need.sources
                if not isinstance(source, RemoteSource)
                and source.pack in cast("list[str]", action.get("packagedFrom", []))
            )
        result = acquire_need(
            AssetNeed(
                name=ref.name,
                digest=ref.digest,
                size=ref.size,
                media_type=ref.media_type,
                sources=tuple(sources),
            ),
            vault,
            pack_roots=declared.pack_roots(),
            before_publish=revalidate,
        )
        if not result.ok:
            raise RequirementError(f"asset {ref.digest}: {result.detail}")
    after = plan_project_requirements(
        root,
        variant=variant,
        source_generation=generation,
        catalog_lock=catalog_lock,
        artifacts=artifacts,
    )
    if any(after[key] != current[key] for key in ("requirements", "inputs")):
        raise RequirementError("project workflow inputs changed; request a new plan")
    if any(_object(after["plan"], "project plan").values()):
        raise RequirementError("target snapshot does not satisfy the approved requirements")
    return after
