"""Wheel-owned import metadata, independent of installed node packs."""

from __future__ import annotations

import hashlib
from importlib.resources import files

from aiohttp import web


def add_comfy_translation_routes(app: web.Application) -> None:
    data = files("dinkster").joinpath("comfy-translations.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    headers = {"ETag": f'"{digest}"', "Cache-Control": "no-cache"}

    async def index(request: web.Request) -> web.Response:
        if any(tag.value in (digest, "*") for tag in (request.if_none_match or ())):
            return web.Response(status=304, headers=headers)
        return web.Response(body=data, content_type="application/json", headers=headers)

    app.router.add_get("/api/comfy-translations", index)
