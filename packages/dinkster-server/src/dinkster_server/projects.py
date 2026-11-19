"""Project requests use the tools-selected library helper, not a record file."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError, AssetRef, LibraryRecord
from dinkster_library import AsyncLibraryClient, LibraryError

from .library import json_body

PROJECTS_KEY: web.AppKey[AsyncLibraryClient] = web.AppKey("library_helper", AsyncLibraryClient)


async def validate_project(request: web.Request, project: object) -> None:
    client = request.app.get(PROJECTS_KEY)
    if client is None:
        return
    try:
        result = await client.call("projects.get", id=project)
        selected = result["project"]
        if selected is None or cast("dict[str, object]", selected)["deleted"]:
            raise web.HTTPBadRequest(text="project is unavailable")
    except LibraryError as error:
        if error.code == "invalid":
            raise web.HTTPBadRequest(text=str(error)) from error
        raise web.HTTPServiceUnavailable(text=str(error)) from error


async def workflow_project(
    request: web.Request, body: Mapping[str, object], *, record_id: str | None = None
) -> str | None:
    client = request.app.get(PROJECTS_KEY)
    if client is None:
        return None
    try:
        owner = (
            await client.call("workflows.get", scope=body["scope"], id=record_id)
            if record_id is not None
            else None
        )
    except LibraryError as error:
        raise web.HTTPServiceUnavailable(text=str(error)) from error
    project = body.get("project", owner["project"] if owner is not None else "default")
    await validate_project(request, project)
    if owner is not None and owner["project"] != project:
        raise web.HTTPBadRequest(text="copy the workflow to put it in another project")
    return cast(str, project)


async def workflow_record_wire(
    request: web.Request, record: LibraryRecord, *, project: str | None = None
) -> dict[str, object]:
    wire = record.to_wire()
    client = request.app.get(PROJECTS_KEY)
    if client is None or record.media_type != "application/x-dinkster-workflow+json":
        return wire
    from .app import STATE_KEY
    from .library import LIBRARY_KEY
    from .workflow_facts import workflow_facts

    library = request.app[LIBRARY_KEY]

    def read_facts() -> dict[str, object] | None:
        data = AssetRef(record.digest, record.name, 0, resolver=library.vault).read_bytes()
        state = request.app[STATE_KEY]
        return workflow_facts(
            data, record.digest, state.schemas, library, node_packs=state.node_packs
        )

    try:
        current = await client.call("workflows.get", scope=record.scope, id=record.id)
        wire["project"] = current["project"] if project is None else project
        if project is None and current["facts"] is not None:
            return wire
        if project is not None:
            current = await client.call(
                "workflows.record",
                scope=record.scope,
                id=record.id,
                revision=current["revision"],
                project=project,
            )
    except LibraryError as error:
        raise web.HTTPServiceUnavailable(text=str(error)) from error
    facts = await asyncio.to_thread(read_facts)
    if facts is None:
        return wire
    try:
        await client.call(
            "workflows.record",
            scope=record.scope,
            id=record.id,
            revision=current["revision"],
            facts=facts,
        )
    except LibraryError as error:
        if error.code in {"stale", "missing"}:
            # A concurrent save or deletion wins; facts stay digest-checked.
            return wire
        raise web.HTTPServiceUnavailable(text=str(error)) from error
    return wire


async def project_needs(request: web.Request, identifier: str) -> dict[str, object]:
    from .app import STATE_KEY
    from .library import LIBRARY_KEY
    from .workflow_facts import workflow_facts

    selected = await request.app[PROJECTS_KEY].call(
        "projects.needs", id=identifier, includeWorkflows=True
    )
    state = request.app[STATE_KEY]
    library = request.app[LIBRARY_KEY]
    schemas, node_packs = state.schemas, state.node_packs
    installed = [
        {"pack": pack, "version": info.version or None}
        for pack, info in sorted(state.packs.items())
    ]

    def compute() -> dict[str, object]:
        packs: dict[str, dict[str, object]] = {}
        unknown: list[dict[str, object]] = []
        for record in cast("list[dict[str, str]]", selected["workflows"]):
            try:
                data = AssetRef(
                    record["digest"], record["name"], 0, resolver=library.vault
                ).read_bytes()
                facts = workflow_facts(
                    data,
                    record["digest"],
                    schemas,
                    library,
                    node_packs=node_packs,
                    include_models=False,
                )
            except (AssetError, OSError):
                facts = None
            if facts is None:
                unknown.append({"scope": record["scope"], "id": record["id"]})
            else:
                for pack in cast("list[dict[str, object]]", facts["packs"]):
                    packs[json.dumps(pack, sort_keys=True)] = pack
        return {
            "packs": [packs[key] for key in sorted(packs)],
            "extra": selected["extra"],
            "unknown": unknown,
            "installed": installed,
        }

    return await asyncio.to_thread(compute)


async def project_request(request: web.Request) -> web.Response:
    client = request.app[PROJECTS_KEY]
    identifier = request.match_info.get("project")
    arguments: dict[str, object] = {}
    if identifier is not None:
        arguments["id"] = identifier
    try:
        if request.method == "GET":
            if request.path.endswith("/needs"):
                assert identifier is not None
                return web.json_response(
                    await project_needs(request, identifier), headers={"Cache-Control": "no-store"}
                )
            elif identifier is not None:
                operation = "projects.get"
            else:
                operation = "projects.list"
                arguments.update(request.query)
                if "limit" in arguments:
                    arguments["limit"] = int(cast(str, arguments["limit"]))
                if "includeDeleted" in arguments:
                    if arguments["includeDeleted"] not in ("true", "false"):
                        raise ValueError("includeDeleted must be true or false")
                    arguments["includeDeleted"] = arguments["includeDeleted"] == "true"
        else:
            body = await json_body(request)
            if request.method == "POST":
                operation = "projects.register"
                arguments.update(body)
            else:
                operation = "projects.delete" if request.method == "DELETE" else "projects.patch"
                arguments.update(body)
                arguments["id"] = identifier
        result = await client.call(operation, **arguments)
        return web.json_response(result)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    except LibraryError as error:
        status = {"stale": 409, "missing": 404, "invalid": 400}.get(error.code, 503)
        return web.json_response({"error": str(error)}, status=status)


def add_project_routes(app: web.Application, root: Path, *, engine_version: str) -> None:
    async def helper_lifetime(app: web.Application) -> AsyncIterator[None]:
        client = await AsyncLibraryClient.start(root)
        try:
            identity = os.environ.get("DINKSTER_ENGINE_IDENTITY")
            if identity is not None:
                value: object = json.loads(identity)
                if not isinstance(value, dict):
                    raise ValueError("engine identity must be an object")
                await client.call(
                    "engines.seen",
                    **cast("dict[str, object]", value),
                    engineVersion=engine_version,
                    apiVersion=1,
                )
            app[PROJECTS_KEY] = client
            yield
        finally:
            await client.close()

    app.cleanup_ctx.append(helper_lifetime)
    app.router.add_get("/api/projects", project_request)
    app.router.add_post("/api/projects", project_request)
    app.router.add_get("/api/projects/{project}", project_request)
    app.router.add_patch("/api/projects/{project}", project_request)
    app.router.add_delete("/api/projects/{project}", project_request)
    app.router.add_get("/api/projects/{project}/needs", project_request)
