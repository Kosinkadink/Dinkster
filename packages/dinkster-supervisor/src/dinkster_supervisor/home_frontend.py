"""The frontends the always-on process serves.

The shared frontend (an installed ``dinkster_frontend``, a folder, or a
development server) is served at the process's address, for the projects
page and every editor that can use it. An installation's own frontend is
what its engine would serve: its development folder or server, else the
frontend installed in its snapshot. The process serves it at the
installation frontend address while that installation is open, a second
origin so an older frontend's browser storage stays apart from the shared
one's.

The page decides which one opens a project from each installation's facts:
the schema wire version its engine speaks, and whether it selected a
frontend of its own.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import aiohttp
from aiohttp import WSMsgType, web
from aiohttp.abc import AbstractStreamWriter

FrontendKind = Literal["installed", "folder", "dev"]
Selection = Literal["folder", "dev", "git", "version"]

ENGINE_PATHS = ("/api", "/memory", "/cache")
"""Paths a development server leaves to the engine; the frontend's own dev proxy names the same."""

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

_PROBE = """
import json
found = {"wire": None, "bundle": None}
try:
    from dinkster_schema.wire import SCHEMA_WIRE_VERSION
    found["wire"] = SCHEMA_WIRE_VERSION if type(SCHEMA_WIRE_VERSION) is int else None
except Exception:
    pass
try:
    from dinkster.frontend import discover_frontend_bundle
    bundle = discover_frontend_bundle()
    found["bundle"] = None if bundle is None else str(bundle)
except Exception:
    pass
print(json.dumps(found))
"""


@dataclass(frozen=True)
class FrontendSource:
    kind: FrontendKind
    location: str
    """A folder holding ``index.html``, or a development server's origin."""

    def wire(self) -> dict[str, str]:
        return {"kind": self.kind, "location": self.location}


@dataclass(frozen=True)
class InstallationFrontend:
    wire: int | None
    """The schema wire version the installation's engine speaks; None when unknown."""
    selected: Selection | None
    """How the installation chose a frontend of its own, or None for its release's."""
    source: FrontendSource | None
    """What the installation frontend address serves for it; None when it has no frontend."""

    def facts(self) -> dict[str, object]:
        return {"wire": self.wire, "selected": self.selected, "available": self.source is not None}


_probed: dict[Path, dict[str, object]] = {}


def _probe(generation: Path) -> dict[str, object]:
    """What the snapshot's own interpreter reports; a sealed snapshot never changes."""
    if generation in _probed:
        return _probed[generation]
    from .environment_state import interpreter

    found: dict[str, object] = {"wire": None, "bundle": None}
    try:
        result = subprocess.run(
            [str(interpreter(generation)), "-I", "-B", "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        found = cast("dict[str, object]", json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError):
        return found
    _probed[generation] = found
    return found


def installation_frontend(root: Path, engine: str = "main") -> InstallationFrontend:
    """Read an installation's frontend facts from disk; no engine needs to run."""
    from .environment_runtime import development_frontend
    from .environment_state import read_current, read_document

    current = read_current(root)
    if current is None:
        return InstallationFrontend(None, None, None)
    generation = root / "builds" / str(current)
    probe = _probe(generation)
    wire = probe["wire"] if type(probe["wire"]) is int else None
    folder, dev = development_frontend(root, current, engine)
    if folder is not None:
        return InstallationFrontend(wire, "folder", FrontendSource("folder", folder))
    if dev is not None:
        return InstallationFrontend(wire, "dev", FrontendSource("dev", dev))
    selected: Selection | None = None
    manifest = generation / "engine-manifest.json"
    provenance = generation / "artifact-sources.json"
    if manifest.is_file() and "frontend" in cast(
        "dict[str, object]", read_document(manifest).get("sources", {})
    ):
        selected = "git"
    elif provenance.is_file() and "frontend" in cast(
        "dict[str, object]", read_document(provenance).get("channel", {})
    ):
        selected = "version"
    bundle = probe["bundle"]
    source = FrontendSource("installed", bundle) if isinstance(bundle, str) else None
    return InstallationFrontend(wire, selected, source)


class _FileResponse(web.FileResponse):
    async def prepare(self, request: web.BaseRequest) -> AbstractStreamWriter | None:
        # Serve the file as built; FileResponse would otherwise pick compressed siblings.
        headers = request.headers.copy()
        headers.popall("Accept-Encoding", None)
        return await super().prepare(request.clone(headers=headers))


def frontend_file(root: Path, path: str) -> web.FileResponse | None:
    """The frontend folder's file for a GET of ``path``, or None when it has none."""
    if path in {"", "/"}:
        asset = root / "index.html"
    else:
        asset = (root / path.lstrip("/")).resolve()
        if not asset.is_relative_to(root) or not asset.is_file():
            return None
    content_type = (
        "text/javascript"
        if asset.suffix.lower() in {".js", ".mjs"}
        else mimetypes.guess_type(asset.name)[0]
    )
    headers = {"Content-Type": content_type or "application/octet-stream"}
    if asset.name == "index.html":
        headers["Cache-Control"] = "no-cache"
    return _FileResponse(asset, headers=headers)


def is_engine_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in ENGINE_PATHS)


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP and name.lower() not in {"host", "content-length"}
    }


def dev_session() -> aiohttp.ClientSession:
    """A session for development servers; their redirects go back to the browser unfollowed."""
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=10),
        auto_decompress=False,
        skip_auto_headers={"Accept-Encoding"},
    )


async def proxy_dev(
    request: web.Request, session: aiohttp.ClientSession, origin: str
) -> web.StreamResponse:
    """Pass a request, or its live-reload socket, to a frontend development server."""
    target = origin.rstrip("/") + "/" + request.path_qs.lstrip("/")
    try:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await _proxy_dev_socket(request, session, target)
        async with session.request(
            request.method,
            target,
            headers=_forward_headers(request.headers),
            data=await request.read(),
            allow_redirects=False,
        ) as response:
            return web.Response(
                status=response.status,
                headers=_forward_headers(response.headers),
                body=await response.read(),
            )
    except (aiohttp.ClientError, OSError) as error:
        return web.json_response(
            {
                "error": "frontend-unavailable",
                "message": f"the frontend development server at {origin} does not answer: {error}",
            },
            status=502,
        )


async def _proxy_dev_socket(
    request: web.Request, session: aiohttp.ClientSession, target: str
) -> web.WebSocketResponse:
    protocols = tuple(
        value.strip()
        for value in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
        if value.strip()
    )
    async with session.ws_connect(
        target, protocols=protocols, headers=_forward_headers(request.headers)
    ) as server:
        browser = web.WebSocketResponse(protocols=(server.protocol,) if server.protocol else ())
        await browser.prepare(request)

        async def pump(
            source: aiohttp.ClientWebSocketResponse | web.WebSocketResponse,
            sink: aiohttp.ClientWebSocketResponse | web.WebSocketResponse,
        ) -> None:
            async for message in source:
                if message.type == WSMsgType.TEXT:
                    await sink.send_str(cast(str, message.data))
                elif message.type == WSMsgType.BINARY:
                    await sink.send_bytes(cast(bytes, message.data))
            await sink.close()

        pumps = (
            asyncio.create_task(pump(browser, server)),
            asyncio.create_task(pump(server, browser)),
        )
        try:
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in pumps:
                task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
    await browser.close()
    return browser
