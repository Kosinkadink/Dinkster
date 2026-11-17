"""Project requests use the tools-selected library helper, not a record file."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_library import AsyncLibraryClient, LibraryError

from .library import json_body

PROJECTS_KEY: web.AppKey[AsyncLibraryClient] = web.AppKey("library_helper", AsyncLibraryClient)


async def project_request(request: web.Request) -> web.Response:
    client = request.app[PROJECTS_KEY]
    identifier = request.match_info.get("project")
    arguments: dict[str, object] = {}
    if identifier is not None:
        arguments["id"] = identifier
    try:
        if request.method == "GET":
            if request.path.endswith("/needs"):
                operation = "projects.needs"
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
