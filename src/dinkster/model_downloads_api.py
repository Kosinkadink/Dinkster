"""Operator provider metadata plans, separate from content-addressed asset contracts."""

from __future__ import annotations

import asyncio
from http.client import HTTPException
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError
from dinkster_assets.huggingface import select_huggingface_file
from dinkster_server.auth import principal_for, resolve_scope
from dinkster_values.asset_folders import use_asset_folders

from .model_downloads import ModelDownloads
from .model_settings import MODEL_FOLDERS
from .project_folders import ProjectFolders
from .template_model_sources import TemplateModelSources


def add_model_download_routes(app: web.Application, downloads: ModelDownloads) -> None:
    sources = TemplateModelSources()
    planning = asyncio.Semaphore(4)

    def plan(name: str, kind: str, project: str) -> dict[str, object]:
        mounts = downloads.mounts
        with use_asset_folders(mounts.project_context(project)):
            if mounts.table.bind_name(name, kind) is not None:
                return {"status": "local", "name": name, "kind": kind}
            disabled = downloads.disabled_name_locations(name, kind)
            if disabled:
                return {
                    "status": "folder-disabled",
                    "disabledLocations": list(disabled),
                    "detail": "Folder not enabled for this project",
                }
        candidates = sources.for_name(name, kind)
        if len(candidates) != 1:
            return {"status": "picker-required", "sources": list(candidates)}
        target = downloads.target(kind, project)
        if target is None:
            return {"status": "destination-required"}
        selected = select_huggingface_file(candidates[0])
        return {
            "status": "consent-required",
            "name": Path(selected.file).name,
            "kind": kind,
            "provider": "huggingface",
            "source": selected.url,
            "repo": selected.repo,
            "revision": selected.revision,
            "file": selected.file,
            "sha256": selected.sha256,
            "size": selected.size,
            "license": selected.license,
            "destination": str(target.path / Path(selected.file).name),
            "mount": target.id,
        }

    async def handle_plan(request: web.Request) -> web.Response:
        principal = principal_for(request)
        if not principal.is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}', content_type="application/json"
            )
        try:
            raw: object = await request.json()
            if not isinstance(raw, dict):
                raise ValueError("expected name, kind, project and scope")
            body = cast("dict[str, object]", raw)
            if set(body) != {"name", "kind", "project", "scope"}:
                raise ValueError("expected name, kind, project and scope")
            name, kind, project = body["name"], body["kind"], body["project"]
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 255
                or name in {".", ".."}
                or any(ord(char) < 32 or ord(char) == 127 or char in "/\\" for char in name)
                or not isinstance(kind, str)
                or kind not in MODEL_FOLDERS
                or not isinstance(project, str)
            ):
                raise ValueError("expected a bare model name, supported kind and project")
            ProjectFolders.validate_project(project)
            resolve_scope(principal, "assets:read", body["scope"])
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if planning.locked():
            return web.json_response({"error": "model-planning-busy"}, status=429)
        async with planning:
            try:
                task = asyncio.create_task(asyncio.to_thread(plan, name, kind, project))
                try:
                    result = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            except (AssetError, OSError, HTTPException):
                return web.json_response(
                    {"error": "model-source-unavailable"},
                    status=409,
                    headers={"Cache-Control": "no-store"},
                )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    app.router.add_post("/api/models/downloads/plan", handle_plan)
