"""Server-admitted client generations and their durable holder census."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from typing import cast

from aiohttp import web
from aiohttp.web_urldispatcher import ResourceRoute
from dinkster_assets import AssetError, StaleRevision
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    claim_inactive_produced_scratch,
    hold_produced_vault,
    is_link_or_junction,
)
from dinkster_protocol.frontend_modules import FRONTEND_ASSET_PATH

from .auth import principal_for
from .library import LIBRARY_KEY, json_body, run_owned_thread

CLIENT_HEADER = "X-Dinkster-Retention-Generation"
HOLDER_HEADER = "X-Dinkster-Retention-Holder"
CLIENT_PATH = "/api/library/retention-clients"


def retention_owner_client(request: web.Request) -> tuple[str, str, tuple[int, ...]] | None:
    generation = request.headers.get(CLIENT_HEADER)
    holder = request.headers.get(HOLDER_HEADER)
    if generation is None and holder is None:
        return None
    classes = [part.strip() for part in (holder or "").split(",")]
    if (
        generation is None
        or len(set(classes)) != len(classes)
        or any(part not in {str(n) for n in range(1, 9)} for part in classes)
    ):
        raise AssetError(
            "retained owners require a generation and distinct holder classes 1 through 8"
        )
    return principal_for(request).principal_id, generation, tuple(sorted(int(n) for n in classes))


async def handle_retention_client(request: web.Request) -> web.Response:
    principal = principal_for(request)
    if not principal.allows("assets:read") or (
        request.method not in {"GET", "HEAD"} and not principal.allows("assets:write")
    ):
        raise web.HTTPForbidden(text="client census requires asset authority")
    if request.query:
        raise web.HTTPBadRequest(text="client census accepts no query parameters")
    store = request.app[LIBRARY_KEY].store
    generation = request.headers.get(CLIENT_HEADER)
    try:
        if request.method == "POST":
            body = await json_body(request)
            if set(body) != {"revision"} or type(body["revision"]) is not int:
                raise AssetError("opening a client requires only an integer revision")
            result = await run_owned_thread(
                lambda: store.open_retention_client(
                    principal.principal_id,
                    generation=generation,
                    revision=cast(int, body["revision"]),
                )
            )
        else:
            if generation is None:
                raise AssetError("a client generation is required")
            if request.method in {"GET", "HEAD"}:
                result = await run_owned_thread(
                    lambda: store.retention_client_status(principal.principal_id, generation)
                )
            elif request.method == "PUT":
                current = await run_owned_thread(
                    lambda: store.retention_client_status(principal.principal_id, generation)
                )
                census_bytes = len(
                    json.dumps(
                        {"revision": current["revision"], "holders": current["holders"]}
                    ).encode()
                )
                # The acknowledgment echoes a server-enumerated census, which can
                # exceed a job submission. Keep the larger bound local to this
                # request; the atomic store check still rejects a changed census.
                body = await json_body(
                    request.clone(client_max_size=max(request.client_max_size, census_bytes + 1))
                )
                if set(body) != {"revision", "holders"} or type(body["revision"]) is not int:
                    raise AssetError("client acknowledgment requires revision and holders")
                result = await run_owned_thread(
                    lambda: store.acknowledge_retention_client(
                        principal.principal_id,
                        generation,
                        revision=cast(int, body["revision"]),
                        holders=body["holders"],
                    )
                )
            else:
                if await json_body(request):
                    raise AssetError("disconnect requires an empty object")
                await run_owned_thread(
                    lambda: store.disconnect_retention_client(principal.principal_id, generation)
                )
                result = await run_owned_thread(
                    lambda: store.retention_client_status(principal.principal_id, generation)
                )
    except StaleRevision as exc:
        raise web.HTTPConflict(text=str(exc)) from None
    except AssetError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from None
    except sqlite3.Error:
        raise web.HTTPServiceUnavailable(text="client census is unavailable") from None
    return web.json_response(result, headers={"Cache-Control": "no-store"})


def add_retention_clients(app: web.Application) -> None:
    for method in ("GET", "HEAD", "POST", "PUT", "DELETE"):
        app.router.add_route(method, CLIENT_PATH, handle_retention_client)

    async def initialize(app: web.Application) -> None:
        library = app[LIBRARY_KEY]

        def establish() -> None:
            root = library.vault.root
            with closing(hold_produced_vault(root)):
                pass
            with claim_inactive_produced_scratch(root / ".produced-collection") as inactive:
                contents = list(root.iterdir())
                pristine = inactive and all(
                    path.name == ".leases"
                    or (
                        path.name
                        in {
                            ".produced-assets.sqlite",
                            ".produced-assets.sqlite-wal",
                            ".produced-assets.sqlite-shm",
                        }
                        and path.is_file()
                        and not is_link_or_junction(path)
                    )
                    for path in contents
                )
                if pristine and any(path.name != ".leases" for path in contents):
                    try:
                        pristine = not ProducedAssetInventory(root).entries()
                    except (AssetError, OSError):
                        pristine = False
                if library.scratch_root is not None and library.scratch_root.exists():
                    pristine = pristine and not any(library.scratch_root.iterdir())
                library.store.initialize_retention_clients(pristine_vault=pristine)

        await run_owned_thread(establish)

    app.on_startup.append(initialize)

    @web.middleware
    async def admit(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        route = request.match_info.route
        # These reads expose neither vault values nor job publications.
        if (
            request.method in {"GET", "HEAD"}
            and isinstance(route, ResourceRoute)
            and route.resource is not None
            and route.resource.canonical in {FRONTEND_ASSET_PATH, "/api/queue/idle"}
        ):
            return await handler(request)
        if (
            request.path == CLIENT_PATH
            or request.path == "/api/health"
            or request.path.startswith("/api/auth/")
            or request.path
            in {"/api/library/storage", "/api/library/storage/policy", "/api/library/storage/clear"}
        ):
            return await handler(request)
        store = request.app[LIBRARY_KEY].store
        principal = principal_for(request).principal_id
        websocket = request.headers.get("Upgrade", "").lower() == "websocket"
        generation = request.headers.get(CLIENT_HEADER)
        if websocket:
            query_generation = request.query.get("retentionGeneration")
            if generation is not None and query_generation not in (None, generation):
                raise web.HTTPBadRequest(text="conflicting retention generations")
            generation = generation or query_generation
        try:
            await run_owned_thread(
                lambda: store.admit_retention_client(principal, generation, pending=not websocket)
            )
        except StaleRevision as exc:
            raise web.HTTPConflict(text=str(exc)) from None
        except (AssetError, sqlite3.Error):
            raise web.HTTPServiceUnavailable(text="client census is unavailable") from None
        try:
            return await handler(request)
        finally:
            if generation is not None:
                if websocket:
                    await run_owned_thread(
                        lambda: store.settle_retention_stream(principal, generation)
                    )
                else:
                    await run_owned_thread(
                        lambda: store.settle_retention_request(principal, generation)
                    )

    app.middlewares.append(admit)
