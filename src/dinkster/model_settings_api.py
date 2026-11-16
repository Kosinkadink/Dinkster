"""Operator model locations and project switches over shared filesystem grants."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from aiohttp import web
from dinkster_assets import MountsError
from dinkster_server.auth import principal_for

from .model_settings import ModelSettings, ModelSettingsConflict
from .project_folders import ProjectFolders
from .storelock import StoreLockTimeout

if TYPE_CHECKING:
    from .mounts_api import MountService


def add_model_settings_routes(app: web.Application, service: MountService) -> None:
    def require_operator(request: web.Request, *, write: bool = False) -> None:
        if not principal_for(request).is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}', content_type="application/json"
            )
        if not service.shared_models or write and not service.allow_changes:
            raise web.HTTPForbidden(
                text='{"error":"model-settings-disabled"}', content_type="application/json"
            )

    def response(project: str) -> web.Response:
        ProjectFolders.validate_project(project)
        service.refresh_models()
        for mount_id in service.table.pending():
            if service.table.source(mount_id) == "shared-models":
                service.scan_soon(app, mount_id)
        settings = ModelSettings.load()
        selected = service.projects.ordered(
            project, service.mount_ids(settings=True), modes=service.model_modes(settings=True)
        )
        document = settings.to_wire()
        live_modes = service.model_modes()
        saved_modes = service.model_modes(settings=True)
        for row, location in zip(
            cast("list[dict[str, object]]", document["locations"]), settings.locations, strict=True
        ):
            mounts = location.mounts(0)
            row["mounts"] = [mount.id for mount in mounts]
            row["enabled"] = all(mount.id in selected for mount in mounts)
            row["active"] = all(
                service.table.get(mount.id) is not None
                and service.table.get(mount.id) == service.settings_table.get(mount.id)
                and live_modes.get(mount.id) == saved_modes.get(mount.id)
                for mount in mounts
            )
        return web.json_response(
            {
                "settings": settings.to_wire(),
                "locations": document["locations"],
                "revision": settings.revision,
                "project": project,
                "changesAllowed": service.allow_changes,
                **service.pending_changes(),
            },
            headers={"Cache-Control": "no-store"},
        )

    async def handle_get(request: web.Request) -> web.Response:
        require_operator(request)
        try:
            return response(request.query.get("project", "default"))
        except MountsError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def handle_put(request: web.Request) -> web.Response:
        require_operator(request, write=True)
        try:
            project = request.query.get("project", "default")
            ProjectFolders.validate_project(project)
            body = await request.json()
            if (
                not isinstance(body, dict)
                or set(body) != {"settings", "revision"}
                or not isinstance(body["revision"], str)
            ):
                raise MountsError("expected model settings and revision")
            settings = ModelSettings.from_wire(body["settings"])
            desired = {mount.id: mount for mount in settings.mounts()}
            for mount_id in desired:
                if service.settings_table.source(mount_id) not in {None, "shared-models"}:
                    raise MountsError("model location conflicts with an existing filesystem grant")
            for row in service.settings_table.descriptors():
                mount_id = str(row["id"])
                if row["source"] != "shared-models" or service.settings_table.get(
                    mount_id
                ) == desired.get(mount_id):
                    continue
                if (
                    mount_id == service.settings_table.output_mount
                    or service.projects.output_projects(mount_id)
                ):
                    raise MountsError(
                        "select replacement output folders before changing this location"
                    )
            await asyncio.to_thread(settings.save, body["revision"])
            result = response(project)
            service.publish(app)
            return result
        except ModelSettingsConflict as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except StoreLockTimeout:
            return web.json_response(
                {"error": "model settings are being edited; retry"}, status=409
            )
        except (MountsError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError:
            return web.json_response({"error": "model settings could not be saved"}, status=500)

    async def handle_project(request: web.Request) -> web.Response:
        require_operator(request, write=True)
        try:
            project = request.match_info["project"]
            ProjectFolders.validate_project(project)
            body = await request.json()
            if (
                not isinstance(body, dict)
                or set(body) != {"enabled"}
                or type(body["enabled"]) is not bool
            ):
                raise MountsError("expected enabled boolean")
            for mount_id in service.refresh_models():
                service.scan_soon(app, mount_id)
            settings = ModelSettings.load()
            location = next(
                (row for row in settings.locations if row.id == request.match_info["location"]),
                None,
            )
            if location is None:
                return web.json_response({"error": "unknown model location"}, status=404)
            selected = list(
                service.projects.ordered(
                    project,
                    service.mount_ids(settings=True),
                    modes=service.model_modes(settings=True),
                )
            )
            ids = {mount.id for mount in location.mounts(0)}
            selected = [mount_id for mount_id in selected if mount_id not in ids]
            if body["enabled"]:
                selected.extend(mount.id for mount in location.mounts(0))
            output = service.projects.output(project, service.settings_table.output_mount)
            service.update_project(project, selected, output)
            service.publish(app)
            return response(project)
        except ModelSettingsConflict as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (MountsError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    app.router.add_get("/api/models/settings", handle_get)
    app.router.add_put("/api/models/settings", handle_put)
    app.router.add_put("/api/models/locations/{location}/projects/{project}", handle_project)
