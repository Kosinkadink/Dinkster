"""dinkster-supervisor: the layer-0 process above an engine host.

ComfyUI's startup delta - the port only opens after every package, custom
node, and file is ready, so nothing can even show progress - is a direct
consequence of the port being owned by the thing that loads everything.
The supervisor inverts that: it owns the public port and almost nothing
else, binds it in milliseconds, and spawns the engine host (dinkster-serve)
behind it. Clients connect immediately and get an honest status surface
while the engine composes; once the engine is ready every other route is
a transparent proxy (HTTP + the /api/events WebSocket).

The supervisor NEVER imports engine code - it spawns processes and speaks
HTTP, which is exactly what keeps it engine-version-agnostic: the layer
that will later manage multiple installations (different code versions,
different venvs - the Comfy Desktop role) cannot share a Python
environment with any one of them. For the same reason this module uses
stdlib logging directly instead of dinkster_schema's core_logger: the one
deliberate exception to the shared-logger rule, because the supervisor
must not depend on any dinkster package.

Engine protocol v1 (versioned; grows additively):
- the supervisor appends ``--host <internal-host> --port <internal-port>``
  to the engine command; the engine must bind exactly there;
- readiness is ``GET /api/health`` on that address answering 200;
- process exit before readiness is startup failure (state "failed");
- the health body MAY carry in-flight composition narration:
  ``{"ok": true, "composition": {"done": int, "total": int,
  "phase"?: str}}``. While present, the supervisor keeps polling health
  and mirrors it onto /supervisor/status as "progress"; once absent
  (or never present - an engine composed before binding) polling stops.
  Advisory narration only: the engine is READY the moment health answers
  200 - with progressive announcement the surface keeps growing behind
  the proxy via /api/nodes epochs and schema_changed events, which are
  the client's contract, not the supervisor's.

Public surface:
- GET  /supervisor/status          {"protocol": 1, "state": ...} (below)
- POST /supervisor/engine/restart  restart the engine process
- /supervisor/<anything else>      local 404 {"error":
                                   "unknown-supervisor-path"} - the whole
                                   /supervisor/ prefix is RESERVED and
                                   never proxied, so future management
                                   routes can be added without ever
                                   colliding with (or leaking to) engine
                                   routes, and clients probing a newer
                                   supervisor API get a crisp local answer
- everything else                  proxied when ready, else 503 with
                                   {"error": "engine-not-ready",
                                    "state": ..., "status":
                                    "/supervisor/status"}

CORS is owned by whoever owns the public port (here: the supervisor), so
UIs - a browser tab, a desktop shell - can talk to /supervisor/* and the
proxied engine surface through one origin policy. Opt-in via repeatable
--allow-origin (exact origin or '*'); default is NO CORS headers, so a
random website cannot script a local supervisor. When enabled the
supervisor answers preflights itself and stamps both local and proxied
responses; upstream Access-Control-* headers are stripped from proxied
responses so a CORS-configured child can never double-stamp. The child
engine needs no CORS configuration behind a supervisor.

Status wire: {"protocol": 1, "state": "starting"|"ready"|"failed"|
"stopped", "detail"?: str, "engine"?: {"pid"?: int, "exitCode"?: int},
"progress"?: {"done": int, "total": int, "phase"?: str}, "instance"?: str}.
Optional fields are omitted when unknown, never null. "progress" is the
settled frontend contract: emitted exactly while the engine's health body
reports in-flight composition, absent otherwise; protocol stays 1 because
the growth is additive. Launchers may supply an opaque instance identifier
and require the same value in status responses to bind readiness to the
process they started.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import socket
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

import aiohttp
from aiohttp import WSMsgType, web

__all__ = [
    "PROTOCOL_VERSION",
    "STATUS_PATH",
    "EngineLink",
    "EngineProcess",
    "create_supervisor_app",
    "main",
]

log = logging.getLogger("dinkster.supervisor")

PROTOCOL_VERSION = 1
STATUS_PATH = "/supervisor/status"
HEALTH_PATH = "/api/health"

# Standard hop-by-hop headers (RFC 9110 7.6.1) plus Host: everything else
# crosses the proxy verbatim in both directions.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)
# Content-Length is recomputed by the proxy's own streaming writer.
_RESPONSE_DROP = _HOP_BY_HOP | {"content-length"}


def _cors_origin(request: web.Request, allowed: frozenset[str]) -> str | None:
    """The Access-Control-Allow-Origin value for this request, or None for
    no CORS response headers (no Origin, or an origin not allowed).

    Duplicated (not shared) with dinkster_server on purpose: the supervisor
    imports no dinkster packages, and ~40 lines of CORS is far cheaper than
    breaking that rule or minting a shared package for it.
    """
    origin = request.headers.get("Origin")
    if origin is None or not allowed:
        return None
    if "*" in allowed:
        return "*"
    return origin if origin in allowed else None


def install_cors(app: web.Application, allowed: frozenset[str]) -> None:
    """Opt-in CORS for the public port. Preflights are answered before
    routing (they never reach the engine proxy); actual responses - local
    and proxied alike - are stamped in on_response_prepare, which fires
    for streamed proxy responses too."""
    if not allowed:
        return

    @web.middleware
    async def preflight(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if request.method == "OPTIONS" and "Access-Control-Request-Method" in request.headers:
            origin = _cors_origin(request, allowed)
            if origin is not None:
                headers = {
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": request.headers[
                        "Access-Control-Request-Method"
                    ],
                    "Access-Control-Max-Age": "600",
                }
                requested = request.headers.get("Access-Control-Request-Headers")
                if requested:
                    headers["Access-Control-Allow-Headers"] = requested
                if origin != "*":
                    headers["Vary"] = "Origin"
                return web.Response(status=204, headers=headers)
        return await handler(request)

    async def stamp(request: web.Request, response: web.StreamResponse) -> None:
        origin = _cors_origin(request, allowed)
        if origin is None:
            return
        response.headers["Access-Control-Allow-Origin"] = origin
        # ETag drives the immutable icon/blueprint/value caching contract;
        # cross-origin scripts must be able to read it through the proxy.
        response.headers["Access-Control-Expose-Headers"] = "ETag"
        if origin != "*" and not any(
            "origin" in value.lower() for value in response.headers.getall("Vary", [])
        ):
            response.headers.add("Vary", "Origin")

    app.middlewares.append(preflight)
    app.on_response_prepare.append(stamp)


@dataclass
class EngineLink:
    """The supervisor's view of its engine: a tiny mutable state machine
    plus the proxy target. Deliberately separate from EngineProcess so the
    HTTP surface is testable without spawning processes - and so a future
    multi-install supervisor can hold several links."""

    state: str = "starting"
    detail: str = ""
    base_url: str = ""
    pid: int | None = None
    exit_code: int | None = None
    progress: dict[str, object] | None = None
    """Mirrored composition progress from the engine's health body
    ({"done": int, "total": int, "phase"?: str}); None once (or while)
    the engine reports no in-flight composition."""

    def status_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"protocol": PROTOCOL_VERSION, "state": self.state}
        if self.detail:
            wire["detail"] = self.detail
        engine: dict[str, object] = {}
        if self.pid is not None:
            engine["pid"] = self.pid
        if self.exit_code is not None:
            engine["exitCode"] = self.exit_code
        if engine:
            wire["engine"] = engine
        if self.progress is not None:
            wire["progress"] = dict(self.progress)
        return wire


LINK_KEY: web.AppKey[EngineLink] = web.AppKey("supervisor_link")
SESSION_KEY: web.AppKey[aiohttp.ClientSession] = web.AppKey("supervisor_session")
RESTART_KEY: web.AppKey[Callable[[], Awaitable[None]] | None] = web.AppKey("supervisor_restart")
CORS_KEY: web.AppKey[frozenset[str]] = web.AppKey("supervisor_cors")
INSTANCE_KEY: web.AppKey[str | None] = web.AppKey("supervisor_instance")


def _progress_from_health(body: object) -> dict[str, object] | None:
    """The status-wire "progress" value carried by an engine health body,
    or None when the engine reports no in-flight composition.

    Deliberately strict about shape (int done/total, optional non-empty
    str phase): the supervisor mirrors this to clients verbatim, so a
    malformed narration degrades to "no progress", never to a malformed
    status wire. Copies only the contracted keys - engine-side additions
    never leak through the supervisor unreviewed."""
    if not isinstance(body, dict):
        return None
    composition = cast("dict[object, object]", body).get("composition")
    if not isinstance(composition, dict):
        return None
    fields = cast("dict[object, object]", composition)
    done = fields.get("done")
    total = fields.get("total")
    if isinstance(done, bool) or isinstance(total, bool):
        return None
    if not isinstance(done, int) or not isinstance(total, int):
        return None
    wire: dict[str, object] = {"done": done, "total": total}
    phase = fields.get("phase")
    if isinstance(phase, str) and phase:
        wire["phase"] = phase
    return wire


def _free_port(host: str) -> int:
    """Ask the OS for an ephemeral port. The classic bind-close-reuse race
    is acceptable here: the engine binds moments later on a loopback host,
    and a lost race fails loudly at engine bind time."""
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class EngineProcess:
    """Spawns and watches one engine host process (protocol v1).

    start() injects ``--host/--port`` and probes ``/api/health`` until the
    engine answers 200 (link -> "ready") or the process exits (link ->
    "failed"). close() terminates the child and marks the link "stopped".
    """

    def __init__(
        self,
        command: Sequence[str],
        link: EngineLink,
        *,
        host: str = "127.0.0.1",
        probe_interval: float = 0.25,
        startup_timeout: float = 600.0,
    ) -> None:
        self.command = list(command)
        self.link = link
        self.host = host
        self.probe_interval = probe_interval
        self.startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._closing = False
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            await self._start()

    async def _start(self) -> None:
        port = _free_port(self.host)
        self.link.state = "starting"
        self.link.detail = ""
        self.link.exit_code = None
        self.link.progress = None
        self.link.base_url = f"http://{self.host}:{port}"
        command = [*self.command, "--host", self.host, "--port", str(port)]
        log.info("starting engine: %s", " ".join(command))
        self._process = await asyncio.create_subprocess_exec(*command)
        self.link.pid = self._process.pid
        self._watch_task = asyncio.create_task(self._watch(self._process))

    async def _watch(self, process: asyncio.subprocess.Process) -> None:
        """Probe health until ready, watching for early exit; keep probing
        while the health body narrates in-flight composition (mirrored to
        the link as progress); then wait for exit so a dying engine flips
        the link to "failed" immediately.

        The startup deadline applies only until the first healthy answer:
        once the engine serves, composition may legitimately take as long
        as its packs take to import - the port is already honest about it.
        """
        deadline = time.monotonic() + self.startup_timeout
        probe_url = self.link.base_url + HEALTH_PATH
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                if process.returncode is not None:
                    label = (
                        "engine exited"
                        if self.link.state == "ready"
                        else "engine exited during startup"
                    )
                    self._mark_dead(process.returncode, label)
                    return
                if self.link.state != "ready" and time.monotonic() > deadline:
                    process.terminate()
                    try:
                        exit_code = await asyncio.wait_for(process.wait(), timeout=10)
                    except TimeoutError:
                        process.kill()
                        exit_code = await process.wait()
                    self._mark_dead(
                        exit_code,
                        f"engine did not become healthy within {self.startup_timeout}s",
                    )
                    return
                healthy = False
                progress: dict[str, object] | None = None
                try:
                    async with session.get(probe_url) as resp:
                        if resp.status == 200:
                            healthy = True
                            progress = _progress_from_health(await resp.json())
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    # Connection refused (still binding), or a health body
                    # that is not JSON (foreign engine): the former retries,
                    # the latter is simply an engine with no narration.
                    pass
                if healthy:
                    if self.link.state != "ready":
                        self.link.state = "ready"
                        self.link.detail = ""
                        log.info("engine ready at %s", self.link.base_url)
                    self.link.progress = progress
                    if progress is None:
                        break  # fully composed (or no narration): stop polling
                await asyncio.sleep(self.probe_interval)
        exit_code = await process.wait()
        if not self._closing:
            self._mark_dead(exit_code, "engine exited")

    def _mark_dead(self, exit_code: int, detail: str) -> None:
        self.link.state = "failed"
        self.link.exit_code = exit_code
        self.link.detail = f"{detail} (exit code {exit_code})"
        self.link.progress = None
        log.error("%s (exit code %d)", detail, exit_code)

    async def restart(self) -> None:
        async with self._lifecycle_lock:
            await self._close()
            self._closing = False
            await self._start()

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await self._close()

    async def _close(self) -> None:
        self._closing = True
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._process = None
        self.link.state = "stopped"
        self.link.pid = None
        self.link.progress = None


async def _handle_status(request: web.Request) -> web.Response:
    status = request.app[LINK_KEY].status_wire()
    instance = request.app[INSTANCE_KEY]
    if instance is not None:
        status["instance"] = instance
    return web.json_response(status)


async def _handle_supervisor_unknown(request: web.Request) -> web.Response:
    """The /supervisor/ prefix is reserved: unknown management paths answer
    locally and NEVER fall through to the engine proxy. This keeps the
    supervisor free to grow its API without engine-route collisions, and
    makes 'this supervisor doesn't know that route' distinguishable from
    an engine 404."""
    return web.json_response(
        {"error": "unknown-supervisor-path", "status": STATUS_PATH}, status=404
    )


async def _handle_restart(request: web.Request) -> web.Response:
    restart = request.app[RESTART_KEY]
    if restart is None:
        return web.json_response({"error": "restart-unavailable"}, status=409)
    await restart()
    return web.json_response(request.app[LINK_KEY].status_wire())


def _not_ready(link: EngineLink) -> web.Response:
    return web.json_response(
        {"error": "engine-not-ready", "state": link.state, "status": STATUS_PATH},
        status=503,
    )


async def _proxy(request: web.Request) -> web.StreamResponse:
    link = request.app[LINK_KEY]
    if link.state != "ready":
        return _not_ready(link)
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await _proxy_ws(request, link)

    session = request.app[SESSION_KEY]
    url = link.base_url + str(request.rel_url)
    headers = {
        name: value for name, value in request.headers.items() if name.lower() not in _HOP_BY_HOP
    }
    data = request.content if request.body_exists else None
    try:
        async with session.request(
            request.method, url, headers=headers, data=data, allow_redirects=False
        ) as upstream:
            response = web.StreamResponse(status=upstream.status)
            # When the supervisor owns CORS, upstream CORS headers are
            # stripped: the stamp hook is the single writer, so a child
            # that happens to be CORS-configured can never double-stamp.
            strip_cors = bool(request.app[CORS_KEY])
            for name, value in upstream.headers.items():
                lowered = name.lower()
                if lowered in _RESPONSE_DROP:
                    continue
                if strip_cors and lowered.startswith("access-control-"):
                    continue
                response.headers.add(name, value)
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except aiohttp.ClientError as exc:
        log.warning("proxy error for %s %s: %s", request.method, url, exc)
        return web.json_response({"error": "engine-unreachable", "status": STATUS_PATH}, status=502)


async def _proxy_ws(request: web.Request, link: EngineLink) -> web.WebSocketResponse:
    """Bidirectional WebSocket pump (the /api/events channel). Text and
    binary frames cross verbatim; either side closing closes both."""
    server_ws = web.WebSocketResponse(heartbeat=30)
    await server_ws.prepare(request)
    session = request.app[SESSION_KEY]
    url = link.base_url + str(request.rel_url)

    async def pump(
        source: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
        sink: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
    ) -> None:
        async for message in source:
            if message.type == WSMsgType.TEXT:
                await sink.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await sink.send_bytes(message.data)
            elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                break

    try:
        async with session.ws_connect(url) as client_ws:
            up = asyncio.create_task(pump(server_ws, client_ws))
            down = asyncio.create_task(pump(client_ws, server_ws))
            _, pending = await asyncio.wait((up, down), return_when=asyncio.FIRST_COMPLETED)
            for task in (up, down):
                if task.done():
                    try:
                        task.result()
                    except Exception:  # noqa: BLE001 - log pump failures, then close both
                        log.exception("websocket proxy pump failed for %s", url)
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    except aiohttp.ClientError as exc:
        log.warning("websocket proxy error for %s: %s", url, exc)
    await server_ws.close()
    return server_ws


def create_supervisor_app(
    link: EngineLink,
    *,
    restart: Callable[[], Awaitable[None]] | None = None,
    allow_origins: Sequence[str] = (),
    instance: str | None = None,
) -> web.Application:
    """The supervisor's HTTP surface over an EngineLink. Process management
    is the caller's (main() wires an EngineProcess; tests drive the link
    directly and point base_url at an in-process engine)."""
    app = web.Application()
    app[LINK_KEY] = link
    app[RESTART_KEY] = restart
    app[CORS_KEY] = frozenset(allow_origins)
    app[INSTANCE_KEY] = instance
    install_cors(app, app[CORS_KEY])

    async def manage_session(app: web.Application) -> None:
        # No total timeout: proxied requests may legitimately stream for a
        # long time (events, large values); connect failures still surface.
        app[SESSION_KEY] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10)
        )

    async def close_session(app: web.Application) -> None:
        await app[SESSION_KEY].close()

    app.on_startup.append(manage_session)
    app.on_cleanup.append(close_session)
    app.router.add_get(STATUS_PATH, _handle_status)
    app.router.add_post("/supervisor/engine/restart", _handle_restart)
    # Reserved namespace: registered before the proxy catch-all, so any
    # /supervisor/ path not claimed above answers locally, never proxies.
    app.router.add_route("*", "/supervisor/{tail:.*}", _handle_supervisor_unknown)
    app.router.add_route("*", "/{tail:.*}", _proxy)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Dinkster supervisor: bind the public port "
        "instantly, start the engine behind it, proxy when ready",
        epilog="Everything after '--' is the engine command (default: "
        "'<this python> -m dinkster.serve'); the supervisor appends "
        "--host/--port for the internal address the engine must bind.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3639)
    parser.add_argument(
        "--instance",
        help="opaque launcher identifier echoed by /supervisor/status",
    )
    parser.add_argument(
        "--engine-host",
        default="127.0.0.1",
        metavar="HOST",
        help="internal address the engine binds (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="enable CORS for this browser origin (repeatable; '*' allows "
        "any); applies to supervisor routes AND proxied engine routes - "
        "the engine behind a supervisor needs no CORS of its own "
        "(default: no CORS headers)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="how long the engine may take to become healthy (default: 600)",
    )
    parser.add_argument(
        "engine",
        nargs=argparse.REMAINDER,
        help="engine command after '--' (default: '<this python> -m dinkster.serve')",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    command = list(args.engine)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [sys.executable, "-m", "dinkster.serve"]

    link = EngineLink()
    engine = EngineProcess(
        command, link, host=args.engine_host, startup_timeout=args.startup_timeout
    )
    app = create_supervisor_app(
        link,
        restart=engine.restart,
        allow_origins=args.allow_origin,
        instance=args.instance,
    )

    async def start_engine(app: web.Application) -> None:
        await engine.start()

    async def stop_engine(app: web.Application) -> None:
        await engine.close()

    app.on_startup.append(start_engine)
    app.on_cleanup.append(stop_engine)
    print(f"dinkster supervisor on http://{args.host}:{args.port} (status: {STATUS_PATH})")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
