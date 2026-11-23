"""Operator model locations and project switches over shared filesystem grants."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from aiohttp import web
from dinkster_assets import AssetError, AssetRef, LocalAssetRef, MountsError
from dinkster_assets.model import model_visibility_ids
from dinkster_server.auth import principal_for
from dinkster_server.projects import validate_project
from dinkster_values.asset_folders import use_asset_folders

from .model_settings import ModelSettings, ModelSettingsConflict
from .project_folders import HiddenModelPreference, ProjectFolders
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
                "hiddenModels": service.projects.hidden_model_preferences(project),
                **service.pending_changes(),
            },
            headers={"Cache-Control": "no-store"},
        )

    async def handle_get(request: web.Request) -> web.Response:
        require_operator(request)
        project = request.query.get("project", "default")
        await validate_project(request, project)
        try:
            return response(project)
        except MountsError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def handle_put(request: web.Request) -> web.Response:
        require_operator(request, write=True)
        try:
            project = request.query.get("project", "default")
            ProjectFolders.validate_project(project)
            await validate_project(request, project)
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
            service.publish(app, mount_ids=service.mount_ids())
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
            await validate_project(request, project)
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
            service.publish(app, mount_ids=service.mount_ids())
            return response(project)
        except ModelSettingsConflict as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (MountsError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def handle_hidden(request: web.Request) -> web.Response:
        require_operator(request, write=True)
        try:
            project = request.match_info["project"]
            await validate_project(request, project)
            context = service.project_context(project)
            body = await request.json()
            if request.method == "DELETE":
                if not isinstance(body, dict) or set(body) != {"identity"}:
                    raise ValueError("expected hidden model identity")
                preference = next(
                    (
                        row
                        for row in service.projects.hidden_model_preferences(project)
                        if body["identity"] in cast("list[str]", row["identities"])
                    ),
                    None,
                )
                if preference is None:
                    return web.json_response({"error": "unknown hidden model"}, status=404)
                selected = HiddenModelPreference(
                    str(preference["name"]),
                    str(preference["kind"]),
                    tuple(cast("list[str]", preference["identities"])),
                )
            else:
                if (
                    not isinstance(body, dict)
                    or set(body) != {"asset", "kind"}
                    or not isinstance(body["kind"], str)
                    or not body["kind"].startswith("model/")
                    or not isinstance(body["asset"], dict)
                ):
                    raise ValueError("expected model asset and kind")
                wire = cast("dict[str, object]", body["asset"])
                virtual = wire.get("virtualPath")
                if not isinstance(virtual, str):
                    raise ValueError("model asset requires virtualPath")
                parts = virtual.split("/", 2)
                if (
                    len(parts) != 3
                    or parts[1] not in context.mounts
                    or service.table.kind(parts[1]) != body["kind"]
                ):
                    raise ValueError("model is outside project folders")
                with use_asset_folders(replace(context, hidden_models=())):
                    if parts[0] == "model-files":
                        ref = service.table.ref(virtual)
                    elif parts[0] == "mounts":
                        mount = service.table.get(parts[1])
                        if mount is None:
                            raise ValueError("model folder is unavailable")
                        ref = service.table.bind_relative_file(body["kind"], mount.path, parts[2])
                    else:
                        raise ValueError("model is outside project folders")
                    identity = "digest" if isinstance(ref, AssetRef) else "localBinding"
                    if ref.to_wire()[identity] != wire.get(identity):
                        raise ValueError("model changed; select it again")
                    path = (
                        ref.local_path()
                        if isinstance(ref, LocalAssetRef)
                        else service.table.resolve_location(ref.virtual_path, ref.digest)
                    )
                    if path is None:
                        raise ValueError("model is unavailable")
                    selected = HiddenModelPreference(
                        ref.name,
                        body["kind"],
                        model_visibility_ids(path, ref.digest if isinstance(ref, AssetRef) else ""),
                    )
            service.projects.set_model_hidden(project, selected, request.method == "PUT")
            service.publish(app, kinds=(selected.kind,))
            return response(project)
        except (AssetError, MountsError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError:
            return web.json_response({"error": "model preference could not be saved"}, status=500)

    app.router.add_put("/api/models/hidden/{project}", handle_hidden)
    app.router.add_delete("/api/models/hidden/{project}", handle_hidden)
    app.router.add_get("/api/models/settings", handle_get)
    app.router.add_put("/api/models/settings", handle_put)
    app.router.add_put("/api/models/locations/{location}/projects/{project}", handle_project)
