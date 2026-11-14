"""Server-admitted client generations and their durable holder census."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from typing import cast

from aiohttp import web
from dinkster_assets import AssetError, StaleRevision
from dinkster_assets.scratch import claim_inactive_produced_scratch, hold_produced_vault

from .auth import principal_for
from .library import LIBRARY_KEY, json_body, run_owned_thread

CLIENT_HEADER = "X-Dinkster-Retention-Generation"
HOLDER_HEADER = "X-Dinkster-Retention-Holder"
CLIENT_PATH = "/api/library/retention-clients"


def retention_owner_client(request: web.Request) -> tuple[str, str, int] | None:
    generation = request.headers.get(CLIENT_HEADER)
    holder = request.headers.get(HOLDER_HEADER)
    if generation is None and holder is None:
        return None
    if generation is None or holder not in {str(n) for n in range(1, 9)}:
        raise AssetError("retained owners require a generation and holder class 1 through 8")
    return principal_for(request).principal_id, generation, int(holder)


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
                body = await json_body(request)
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
                pristine = inactive and all(path.name == ".leases" for path in root.iterdir())
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
        generation = request.headers.get(CLIENT_HEADER)
        websocket = request.headers.get("Upgrade", "").lower() == "websocket"
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
                        lambda: store.disconnect_retention_client(principal, generation)
                    )
                else:
                    await run_owned_thread(
                        lambda: store.settle_retention_request(principal, generation)
                    )

    app.middlewares.append(admit)
