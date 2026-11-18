"""Read-only workflow plans over host-owned manifests and local asset grants."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetSource,
    LocalAssetRef,
    MountsError,
    RemoteSource,
)
from dinkster_graph import Graph, GraphNode
from dinkster_schema import AssetWidget, WidgetRepresentations
from dinkster_server import LIBRARY_KEY, STATE_KEY, ServerLibrary, principal_for
from dinkster_server.auth import resolve_scope
from dinkster_server.preflight import bind_model_names, parse_asset_consent, stamp_model_provenance
from dinkster_values.asset_folders import use_asset_folders
from dinkster_workers import PackManifest

from .mounts_api import MOUNTS_KEY, MountService
from .project_folders import ProjectFolders
from .workflow_requirements import (
    VARIANTS,
    PackRequirement,
    RequirementError,
    generation_device,
    manifest_unavailability,
    plan_workflow_requirements,
    project_requirements,
)


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
    generation: Path | None = None,
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
            if not isinstance(variant, str):
                raise RequirementError("variant must be a string")
            device = None
            if variant not in VARIANTS:
                if generation is None or variant != engine_variant:
                    raise RequirementError(
                        f"variant must be one of {', '.join(VARIANTS)} "
                        "or the selected generation's variant"
                    )
                device = await asyncio.to_thread(generation_device, generation, variant)
            project = raw.get("project")
            if project is not None:
                ProjectFolders.validate_project(cast(str, project))
            consent = parse_asset_consent({"assetSources": raw.get("assetSources", {})})
            if isinstance(consent, str):
                raise RequirementError(consent)
            _, hinted = consent
            requirements = project_requirements(cast("list[object]", workflows))
        except (ValueError, OSError, AssetError, MountsError) as exc:
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
            return manifest_unavailability(pack, manifest, cast("str", variant), device=device)

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

    async def handle_model_provenance(request: web.Request) -> web.Response:
        principal = principal_for(request)
        if not principal.local or principal.kind != "human" or principal.delegation_id is not None:
            raise web.HTTPForbidden(
                text='{"error":"local-operator-required"}', content_type="application/json"
            )
        resolve_scope(principal, "assets:read", "local")
        library = request.app.get(LIBRARY_KEY)
        mounts = request.app.get(MOUNTS_KEY)
        if library is None or mounts is None:
            return web.json_response({"error": "library-required"}, status=503)
        try:
            raw = await request.json()
            if not isinstance(raw, dict) or set(raw) != {"document", "project"}:
                raise ValueError("expected document and project")
            project = raw["project"]
            if not isinstance(project, str):
                raise ValueError("expected project")
            ProjectFolders.validate_project(project)
            document = raw["document"]
            if (
                not isinstance(document, dict)
                or document.get("format") != "dinkster-workflow"
                or document.get("formatVersion") != 1
            ):
                raise ValueError("expected a workflow document")
            graphs = document.get("graphs")
            if not isinstance(graphs, dict):
                raise ValueError("expected workflow graphs")
            nodes: dict[str, GraphNode] = {}
            paths: dict[str, list[str]] = {}
            for graph_id, graph in graphs.items():
                if (
                    not isinstance(graph_id, str)
                    or not isinstance(graph, dict)
                    or not isinstance(graph.get("nodes"), dict)
                ):
                    raise ValueError("expected graph nodes")
                for node_id, node in graph["nodes"].items():
                    if (
                        not isinstance(node_id, str)
                        or not isinstance(node, dict)
                        or not isinstance(node.get("type"), str)
                        or not isinstance(node.get("values", {}), dict)
                    ):
                        raise ValueError("expected node type and values")
                    key = str(len(nodes))
                    paths[key] = [graph_id, node_id]
                    nodes[key] = GraphNode(node["type"], node.get("values", {}))
            context = mounts.project_context(project)
        except (ValueError, MountsError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        schemas = {
            node_type: replace(
                schema,
                inputs=tuple(
                    spec
                    for spec in schema.inputs
                    if isinstance(
                        widget := (
                            spec.widget.representations[0].widget
                            if isinstance(spec.widget, WidgetRepresentations)
                            else spec.widget
                        ),
                        AssetWidget,
                    )
                    and widget.kind.startswith("model/")
                ),
            )
            for node_type, schema in request.app[STATE_KEY].schemas.items()
        }

        def inspect() -> dict[str, object]:
            with use_asset_folders(context):
                admitted = {}
                for node_id, node in nodes.items():
                    inputs = {}
                    schema = schemas.get(node.node_type)
                    if schema is None:
                        continue
                    for spec in schema.inputs:
                        value = node.inputs.get(spec.id)
                        if isinstance(value, str):
                            inputs[spec.id] = value
                            continue
                        if not isinstance(value, dict):
                            continue
                        widget = spec.widget
                        if isinstance(widget, WidgetRepresentations):
                            widget = widget.representations[0].widget
                        assert isinstance(widget, AssetWidget)
                        try:
                            wire = cast("dict[str, object]", value)
                            selected: AssetRef | LocalAssetRef
                            location = None
                            if wire.get("verification") == "unverified":
                                selected = LocalAssetRef.from_wire(wire, mounts.table)
                                location = mounts.table.resolve_local(selected.binding)
                                path = location[0] if location else None
                            else:
                                selected = AssetRef.from_wire(wire)
                                locations = (
                                    [selected.virtual_path]
                                    if selected.virtual_path
                                    else [
                                        entry.virtual_path
                                        for mount_id in context.mounts
                                        if mount_id in mounts.table.mounts_for_kind(widget.kind)
                                        for entry in mounts.table.entries(mount_id)
                                        if entry.digest == selected.digest
                                    ]
                                )
                                path = next(
                                    (
                                        resolved
                                        for virtual_path in locations
                                        if (
                                            resolved := mounts.table.resolve_location(
                                                virtual_path, selected.digest
                                            )
                                        )
                                        is not None
                                    ),
                                    None,
                                )
                            if path is None:
                                continue
                            for root in mounts.table.roots_for_kind(widget.kind):
                                if not path.is_relative_to(root):
                                    continue
                                candidate = mounts.table.bind_relative_file(
                                    widget.kind, root, path.relative_to(root).as_posix()
                                )
                                if isinstance(selected, AssetRef):
                                    if (
                                        not isinstance(candidate, AssetRef)
                                        or candidate.digest != selected.digest
                                    ):
                                        continue
                                elif mounts.table.resolve_local(selected.binding) != location:
                                    continue
                                inputs[spec.id] = candidate.to_wire()
                                break
                        except (AssetError, OSError, ValueError):
                            continue
                    admitted[node_id] = replace(node, inputs=inputs)
                bound, missing = bind_model_names(Graph(admitted), schemas, library)
                visible_nodes = dict(bound.nodes)
                for unavailable in missing:
                    node_id, input_id = str(unavailable["nodeId"]), str(unavailable["inputId"])
                    node = visible_nodes[node_id]
                    visible_nodes[node_id] = replace(
                        node,
                        inputs={
                            key: value for key, value in node.inputs.items() if key != input_id
                        },
                    )
                snapshot = stamp_model_provenance(
                    Graph(visible_nodes), schemas, library, None, targets=list(visible_nodes)
                )
            if snapshot is None:
                return {"version": 1, "models": []}
            assert snapshot.extra_pnginfo is not None
            facts = cast("dict[str, object]", snapshot.extra_pnginfo["dinkster.modelProvenance"])
            for row in cast("list[dict[str, object]]", facts["models"]):
                row["nodePath"] = paths[cast("list[str]", row["nodePath"])[0]]
            return facts

        try:
            facts = await asyncio.to_thread(inspect)
        except (AssetError, OSError, ValueError):
            return web.json_response({"error": "model-provenance-unavailable"}, status=409)
        return web.json_response(facts, headers={"Cache-Control": "no-store"})

    app.router.add_post("/api/workflows/plan", handle_plan)
    app.router.add_post("/api/workflows/model-provenance", handle_model_provenance)
