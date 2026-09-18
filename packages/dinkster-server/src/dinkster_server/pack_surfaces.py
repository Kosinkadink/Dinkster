"""Host-owned HTTP adaptation for immutable pack extension declarations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping

from aiohttp import web
from dinkster_protocol import ExtensionSnapshot, extension_behavior_hash
from dinkster_protocol.frontend_modules import FRONTEND_ASSET_PATH, FrontendModule
from dinkster_protocol.pack_surfaces import (
    PACK_JSON_MAX_BYTES,
    PACK_ROUTE_PATH,
    PACK_ROUTE_TIMEOUT,
    PackRoute,
)

PackRouteDispatch = Callable[
    [str, PackRoute, Mapping[str, object], str], Awaitable[dict[str, object]]
]
FrontendModuleRead = Callable[[str, FrontendModule], Awaitable[bytes]]


def install_pack_surfaces(
    app: web.Application,
    snapshot: Callable[[], ExtensionSnapshot],
    dispatch: PackRouteDispatch | None,
    read_module: FrontendModuleRead | None,
) -> dict[tuple[str, str], str]:
    """Static host paths resolve only entries selected by the current snapshot."""

    async def route(request: web.Request) -> web.Response:
        pack = request.match_info["pack_id"]
        route_id = request.match_info["route_id"]
        selected = snapshot()
        digest = "sha256:" + extension_behavior_hash(selected)
        expected = request.headers.get("If-Match")
        if expected is not None and expected != digest:
            raise web.HTTPPreconditionFailed()
        declaration = next(
            (
                item
                for extension in selected.extensions
                if extension.id == pack
                for item in extension.routes
                if item.id == route_id
            ),
            None,
        )
        if declaration is None or dispatch is None:
            raise web.HTTPNotFound()
        if request.method != declaration.method:
            raise web.HTTPMethodNotAllowed(request.method, [declaration.method])
        if request.query:
            return web.json_response({"error": "pack-route-invalid-request"}, status=400)
        try:
            body: object = {}
            if request.method == "POST":
                if request.content_type != "application/json":
                    raise web.HTTPUnsupportedMediaType()
                data = bytearray()
                async for chunk in request.content.iter_chunked(PACK_JSON_MAX_BYTES + 1):
                    data.extend(chunk)
                    if len(data) > PACK_JSON_MAX_BYTES:
                        raise web.HTTPRequestEntityTooLarge(
                            max_size=PACK_JSON_MAX_BYTES, actual_size=len(data)
                        )
                body = json.loads(data)
            elif request.can_read_body:
                raise ValueError("GET request bodies are not supported")
            payload = declaration.request.validate(body)
        except (ValueError, UnicodeError):
            return web.json_response({"error": "pack-route-invalid-request"}, status=400)
        try:
            async with asyncio.timeout(PACK_ROUTE_TIMEOUT):
                result = await dispatch(pack, declaration, payload, digest)
            return web.json_response(
                declaration.response.validate(result),
                headers={"X-Dinkster-Extension-Snapshot": digest},
            )
        except TimeoutError:
            return web.json_response({"error": "pack-route-timeout"}, status=504)
        except KeyError:
            if "sha256:" + extension_behavior_hash(snapshot()) != digest:
                raise web.HTTPPreconditionFailed() from None
            raise web.HTTPNotFound() from None
        except Exception:
            return web.json_response({"error": "pack-route-failed"}, status=502)

    async def module(request: web.Request) -> web.Response:
        pack, entry, digest = (request.match_info[key] for key in ("pack_id", "entry_id", "digest"))
        declaration = next(
            (
                item
                for extension in snapshot().extensions
                if extension.id == pack
                for item in extension.frontend_modules
                if item.id == entry and item.module_digest == digest
            ),
            None,
        )
        if declaration is None or read_module is None:
            raise web.HTTPNotFound()
        try:
            body = await read_module(pack, declaration)
        except (OSError, ValueError, KeyError):
            raise web.HTTPNotFound() from None
        return web.Response(
            body=body,
            content_type="text/javascript",
            charset="utf-8",
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "ETag": f'"{digest}"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.router.add_get(PACK_ROUTE_PATH, route, allow_head=False)
    app.router.add_post(PACK_ROUTE_PATH, route)
    app.router.add_get(FRONTEND_ASSET_PATH, module)
    return {
        ("GET", PACK_ROUTE_PATH): "jobs:read",
        ("POST", PACK_ROUTE_PATH): "jobs:submit",
        ("GET", FRONTEND_ASSET_PATH): "jobs:read",
        ("HEAD", FRONTEND_ASSET_PATH): "jobs:read",
    }
