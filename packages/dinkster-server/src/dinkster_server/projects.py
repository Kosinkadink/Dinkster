"""Project requests use the tools-selected library helper, not a record file."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError, AssetRef, LibraryRecord
from dinkster_library import AsyncLibraryClient, LibraryClient, LibraryError

from .library import json_body

PROJECTS_KEY: web.AppKey[AsyncLibraryClient] = web.AppKey("library_helper", AsyncLibraryClient)
ACTIVE_ENVIRONMENT_KEY: web.AppKey[dict[str, str] | None] = web.AppKey("active_environment")


async def project_for_run(app: web.Application, identifier: str) -> dict[str, object] | None:
    client = app.get(PROJECTS_KEY)
    if client is None:
        return None
    result = await client.call("projects.get", id=identifier)
    project = cast("dict[str, object] | None", result["project"])
    if project is None or project["deleted"]:
        raise ValueError("project is unavailable")
    previous = cast("dict[str, object] | None", project["environment"])
    environment = app.get(ACTIVE_ENVIRONMENT_KEY)
    if environment is None:
        if previous is not None:
            raise ValueError("this engine cannot identify the project's environment")
        return project
    if previous is not None and previous["id"] != environment["id"]:
        raise ValueError("open this project's own environment before running it")
    return project


async def record_project_run(app: web.Application, identifier: str, started_at: float) -> None:
    while (project := await project_for_run(app, identifier)) is not None:
        environment = app[ACTIVE_ENVIRONMENT_KEY]
        if environment is None:
            return
        previous = cast("dict[str, object] | None", project["environment"])
        last_run = datetime.fromtimestamp(started_at, UTC)
        if previous is not None and previous["lastRunAt"] is not None:
            try:
                previous_run = datetime.fromisoformat(cast(str, previous["lastRunAt"]))
            except ValueError:
                previous_run = None
            if (
                previous_run is not None
                and previous_run.tzinfo is not None
                and previous_run >= last_run
            ):
                return
        try:
            await app[PROJECTS_KEY].call(
                "projects.patch",
                id=identifier,
                revision=project["revision"],
                patch={
                    "environment": {
                        "id": environment["id"],
                        "lastSnapshot": environment["snapshot"],
                        "lastRunAt": last_run.isoformat(),
                    }
                },
            )
            return
        except LibraryError as error:
            if error.code != "stale":
                raise


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
    from dinkster_library.http_routes import (
        ERROR_STATUS,
        outputs_call,
        project_call,
        restore_call,
    )

    client = request.app[PROJECTS_KEY]
    identifier = request.match_info.get("project")
    # The matched route, not the path, names the action: a project may be called "restore".
    resource = request.match_info.route.resource
    route = resource.canonical if resource is not None else ""
    try:
        if route.endswith("/needs"):
            assert identifier is not None
            return web.json_response(
                await project_needs(request, identifier), headers={"Cache-Control": "no-store"}
            )
        if route.endswith("/outputs"):
            assert identifier is not None
            result = await client.call("assets.list", **outputs_call(identifier, request.query))
            return web.json_response(result)
        body = {} if request.method == "GET" else await json_body(request)
        if route.endswith("/restore"):
            assert identifier is not None
            operation, arguments = restore_call(identifier, body)
        else:
            operation, arguments = project_call(request.method, identifier, request.query, body)
        result = await client.call(operation, **arguments)
        return web.json_response(result)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    except LibraryError as error:
        status = ERROR_STATUS.get(error.code, 503)
        return web.json_response({"error": str(error)}, status=status)


def add_project_routes(
    app: web.Application, root: Path | LibraryClient, *, engine_version: str
) -> None:
    environment: dict[str, str] | None = None

    async def active_environment(request: web.Request) -> web.Response:
        return web.json_response({"environment": environment})

    async def relay_changes(client: AsyncLibraryClient, cursor: int) -> None:
        from dinkster_schema import core_logger

        from .app import STATE_KEY

        unavailable = False
        while True:
            try:
                page = await client.call("changes", cursor=cursor, limit=1000)
            except LibraryError as error:
                if not unavailable:
                    core_logger("server").warning("Library change relay unavailable: %s", error)
                unavailable = True
                await asyncio.sleep(0.5)
                continue
            unavailable = False
            cursor = cast(int, page["cursor"])
            items = cast("list[dict[str, object]]", page["items"])
            # Working-reference publications do not change a library view. No
            # record identities or scope details cross the broadcast; subscribers
            # refetch through their authorized HTTP routes.
            if any(item["kind"] != "engine-references" for item in items):
                app[STATE_KEY].hub.publish(
                    {"type": "library_changed"}, client_id=None, droppable=False
                )
            if len(items) < 1000:
                await asyncio.sleep(0.5)

    async def helper_lifetime(app: web.Application) -> AsyncIterator[None]:
        nonlocal environment
        client = (
            AsyncLibraryClient(root)
            if isinstance(root, LibraryClient)
            else await AsyncLibraryClient.start(root)
        )
        relay: asyncio.Task[None] | None = None
        try:
            if "changes.head" not in cast("list[str]", client.sync.hello["capabilities"]):
                raise LibraryError("capability", "Library change relay requires updated tools.")
            identity = os.environ.get("DINKSTER_ENGINE_IDENTITY")
            if identity is not None:
                value: object = json.loads(identity)
                if not isinstance(value, dict):
                    raise ValueError("engine identity must be an object")
                supplied = cast("dict[str, object]", value)
                if supplied.get("identityVersion") == 1:
                    identifier, snapshot = supplied.get("environment"), supplied.get("snapshot")
                    if (
                        not isinstance(identifier, str)
                        or re.fullmatch(r"[0-9a-f]{32}", identifier) is None
                        or not isinstance(snapshot, str)
                        or re.fullmatch(r"[1-9][0-9]*", snapshot) is None
                    ):
                        raise ValueError("invalid managed environment identity")
                    environment = {"id": identifier, "snapshot": snapshot}
                await client.call(
                    "engines.seen",
                    environment=supplied.get("environment"),
                    snapshot=supplied.get("snapshot"),
                    engineVersion=engine_version,
                    apiVersion=1,
                )
            app[PROJECTS_KEY] = client
            app[ACTIVE_ENVIRONMENT_KEY] = environment
            cursor = cast(int, (await client.call("changes.head"))["cursor"])
            relay = asyncio.create_task(relay_changes(client, cursor))
            yield
        finally:
            from .app import STATE_KEY

            try:
                try:
                    if relay is not None:
                        relay.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await relay
                finally:
                    state = app.get(STATE_KEY)
                    if state is not None:
                        await state.queue.close()
            finally:
                if not isinstance(root, LibraryClient):
                    await client.close()

    app.cleanup_ctx.append(helper_lifetime)
    app.router.add_get("/api/environment", active_environment)
    app.router.add_get("/api/projects", project_request)
    app.router.add_post("/api/projects", project_request)
    app.router.add_get("/api/projects/{project}", project_request)
    app.router.add_patch("/api/projects/{project}", project_request)
    app.router.add_delete("/api/projects/{project}", project_request)
    app.router.add_post("/api/projects/{project}/restore", project_request)
    app.router.add_get("/api/projects/{project}/needs", project_request)
    app.router.add_get("/api/projects/{project}/outputs", project_request)
