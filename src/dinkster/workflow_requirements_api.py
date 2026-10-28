"""Read-only workflow plans over host-owned manifests and local asset grants."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError, AssetRef, AssetSource, MountsError, RemoteSource
from dinkster_server import LIBRARY_KEY, STATE_KEY, ServerLibrary, principal_for
from dinkster_server.preflight import parse_asset_consent
from dinkster_workers import PackManifest

from .mounts_api import MOUNTS_KEY, MountService
from .project_folders import ProjectFolders
from .workflow_requirements import (
    PackRequirement,
    RequirementError,
    manifest_unavailability,
    plan_workflow_requirements,
    project_requirements,
)

VARIANTS = ("linux-cpu", "win-cpu", "win-nvidia", "win-nvidia-cu126")


def _asset_present(
    asset: AssetRef, library: ServerLibrary, mounts: MountService | None, project: str | None
) -> bool:
    try:
        with replace(asset, resolver=library.vault).open():
            return True
    except (AssetError, OSError):
        pass
    if mounts is None:
        return False
    selected = mounts.project_context(project).mounts if project is not None else mounts.mount_ids()
    ready = {str(row["id"]) for row in mounts.table.descriptors() if row["state"] == "ready"}
    for mount_id in selected:
        grant = mounts.table.get(mount_id)
        if grant is None or mount_id not in ready:
            continue
        try:
            entries = mounts.table.entries(mount_id)
        except AssetError:
            continue
        for entry in entries:
            if entry.digest != asset.digest:
                continue
            try:
                ref = mounts.table.ref(entry.virtual_path)
                if not ref.local_path().resolve(strict=True).is_relative_to(grant.path.resolve()):
                    continue
                # Bind to this selected mount, not the table's global digest lookup.
                with ref.open():
                    if mounts.table.get(mount_id) == grant:
                        return True
            except (AssetError, OSError):
                continue
    return False


def add_workflow_requirement_routes(
    app: web.Application,
    *,
    manifests: Callable[[], Mapping[str, PackManifest]],
    engine_variant: str | None = None,
) -> None:
    async def handle_plan(request: web.Request) -> web.Response:
        principal = principal_for(request)
        if not principal.local or principal.kind != "human" or principal.delegation_id is not None:
            raise web.HTTPForbidden(
                text='{"error":"local-operator-required"}', content_type="application/json"
            )
        library = request.app.get(LIBRARY_KEY)
        if library is None:
            raise web.HTTPServiceUnavailable(
                text='{"error":"library-required"}', content_type="application/json"
            )
        try:
            body: object = await request.json()
            if not isinstance(body, dict):
                raise RequirementError("request body must be an object")
            raw = cast("dict[str, object]", body)
            if set(raw) - {"workflows", "variant", "project", "assetSources"}:
                raise RequirementError("unknown workflow plan fields")
            workflows = raw.get("workflows")
            if not isinstance(workflows, list) or not workflows:
                raise RequirementError("workflows must be a non-empty list")
            variant = raw.get("variant", engine_variant)
            if variant is None:
                raise RequirementError(
                    "engine variant is undeclared; supply variant or launch with --engine-variant"
                )
            if not isinstance(variant, str) or variant not in VARIANTS:
                raise RequirementError(f"variant must be one of {', '.join(VARIANTS)}")
            project = raw.get("project")
            if project is not None:
                ProjectFolders.validate_project(cast(str, project))
            consent = parse_asset_consent({"assetSources": raw.get("assetSources", {})})
            if isinstance(consent, str):
                raise RequirementError(consent)
            _, hinted = consent
            requirements = project_requirements(cast("list[object]", workflows))
        except (json.JSONDecodeError, RequirementError, AssetError, MountsError) as exc:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": str(exc)}), content_type="application/json"
            ) from exc
        state = request.app[STATE_KEY]
        installed = {pack: info.version or None for pack, info in state.packs.items()}
        declared = manifests()

        def unavailable(pack: PackRequirement) -> str | None:
            manifest = declared.get(pack.pack)
            if manifest is None or installed.get(pack.pack) != pack.version:
                return "saved-version manifest is unavailable for compatibility planning"
            return manifest_unavailability(pack, manifest, variant)

        def sources(digest: str) -> list[AssetSource]:
            leads: list[AssetSource] = [RemoteSource(url) for url in hinted.get(digest, ())]
            if library.provenance is not None:
                leads.extend(
                    RemoteSource(url)
                    for url in library.provenance.sources(digest)
                    if url.startswith(("http://", "https://"))
                )
            if library.pack_assets is not None:
                need = library.pack_assets.need_for(digest)
                if need is not None:
                    leads.extend(need.sources)
            return leads

        mounts = request.app.get(MOUNTS_KEY)
        plan = await asyncio.to_thread(
            plan_workflow_requirements,
            requirements,
            installed_packs=installed,
            asset_available=lambda asset: _asset_present(
                asset, library, mounts, cast("str | None", project)
            ),
            sources_for=sources,
            pack_unavailable=unavailable,
        )
        return web.json_response(plan.to_wire(), headers={"Cache-Control": "no-store"})

    app.router.add_post("/api/workflows/plan", handle_plan)
