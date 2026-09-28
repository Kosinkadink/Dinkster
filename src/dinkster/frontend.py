"""Serve the browser application from the engine's public origin."""

from __future__ import annotations

import asyncio
import importlib.resources
import mimetypes
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import cast
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import WSMsgType, web

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def discover_frontend_bundle() -> Path | None:
    """Find the installed release bundle or a sibling development build."""
    try:
        packaged = importlib.resources.files("dinkster_frontend").joinpath("dist")
        packaged_path = Path(str(packaged))
        if (packaged_path / "index.html").is_file():
            return packaged_path
    except ModuleNotFoundError:
        pass

    repository = Path(__file__).resolve().parents[2]
    sibling = repository.parent / "Dinkster-Frontend" / "packages" / "app" / "dist"
    return sibling if (sibling / "index.html").is_file() else None


def _safe_asset(root: Path, request_path: str) -> Path:
    candidate = (root / request_path.lstrip("/")).resolve()
    if not candidate.is_relative_to(root):
        raise web.HTTPForbidden()
    return candidate if candidate.is_file() else root / "index.html"


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP and name.lower() not in {"host", "content-length"}
    }


async def _proxy_websocket(
    request: web.Request, session: aiohttp.ClientSession, target: str
) -> web.WebSocketResponse:
    protocols = tuple(
        value.strip()
        for value in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
        if value.strip()
    )
    browser = web.WebSocketResponse(protocols=protocols)
    await browser.prepare(request)
    async with session.ws_connect(
        target,
        protocols=protocols,
        headers=_forward_headers(request.headers),
    ) as vite:

        async def browser_to_vite() -> None:
            async for message in browser:
                if message.type == WSMsgType.TEXT:
                    await vite.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await vite.send_bytes(message.data)
                elif message.type == WSMsgType.CLOSE:
                    await vite.close()

        async def vite_to_browser() -> None:
            async for message in vite:
                if message.type == WSMsgType.TEXT:
                    await browser.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await browser.send_bytes(message.data)
                elif message.type == WSMsgType.CLOSE:
                    await browser.close()

        await asyncio.gather(browser_to_vite(), vite_to_browser())
    return browser


def install_frontend(
    app: web.Application,
    *,
    bundle: Path | None = None,
    development_url: str | None = None,
) -> None:
    """Install the final catch-all route for a static bundle or Vite proxy."""
    if (bundle is None) == (development_url is None):
        raise ValueError("configure exactly one frontend source")

    if development_url is not None:
        parsed = urlparse(development_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("frontend development URL must be an http(s) origin")
        base = development_url.rstrip("/") + "/"
        session_key = web.AppKey("frontend_proxy_session", aiohttp.ClientSession)

        async def proxy_context(_: web.Application) -> AsyncIterator[None]:
            session = aiohttp.ClientSession()
            app[session_key] = session
            try:
                yield
            finally:
                await session.close()

        async def proxy(request: web.Request) -> web.StreamResponse:
            target = urljoin(base, request.path_qs.lstrip("/"))
            if request.headers.get("Upgrade", "").lower() == "websocket":
                return await _proxy_websocket(request, app[session_key], target)
            body = await request.read()
            async with app[session_key].request(
                request.method,
                target,
                headers=_forward_headers(request.headers),
                data=body,
                allow_redirects=False,
            ) as response:
                return web.Response(
                    status=response.status,
                    headers=_forward_headers(response.headers),
                    body=await response.read(),
                )

        app.cleanup_ctx.append(proxy_context)
        app.router.add_route("*", "/{path:.*}", proxy)
        return

    root = cast("Path", bundle).resolve()
    if not (root / "index.html").is_file():
        raise ValueError(f"frontend bundle has no index.html: {root}")

    async def static(request: web.Request) -> web.StreamResponse:
        if request.method not in {"GET", "HEAD"}:
            raise web.HTTPNotFound()
        asset = _safe_asset(root, request.match_info["path"])
        content_type = (
            "text/javascript"
            if asset.suffix.lower() in {".js", ".mjs"}
            else mimetypes.guess_type(asset.name)[0]
        )
        return web.FileResponse(
            asset, headers={"Content-Type": content_type or "application/octet-stream"}
        )

    app.router.add_route("*", "/{path:.*}", static)
