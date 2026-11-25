"""dinkster-home: the always-on Dinkster process.

One per user. It owns the public address and serves the frontend itself, so
the browser and Desktop show a page before any engine runs, and it passes
every other path, ``/api`` and the events socket included, to the engine of
the installation that is open. Engines keep running under their own managed
supervisors (``environment_runtime.start_engine``); this process starts them
through ``dinkster-env start`` and proxies to the supervisor's address.

Routing, in order:
- ``/supervisor/*`` answers here (status, open, restart);
- ``/`` and any path naming a file of the frontend are the frontend;
- everything else passes to the open engine when it is ready, else 503
  ``engine-not-ready``. The events socket is refused the same way, and every
  proxied socket closes with 1012 when the open engine changes or stops, so
  an editor's next events connection is always a new engine run.

Status wire: the supervisor's protocol 1 (see ``supervisor.py``) plus
``"role": "home"``, ``"installations"`` (how many the store registry lists)
and ``"open": {"root", "engine"}`` while an installation is open. Top-level
``state`` is the open engine's, ``stopped`` when none is. Restart with nothing
open opens the default installation, or refuses with a ``message`` saying why.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import logging
import mimetypes
import os
import signal
import sys
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import aiohttp
from aiohttp import web
from aiohttp.abc import AbstractStreamWriter
from yarl import URL

from .supervisor import (
    CORS_KEY,
    HTTP_TASKS_KEY,
    INSTANCE_KEY,
    LINK_KEY,
    SESSION_KEY,
    STATUS_PATH,
    WEBSOCKETS_KEY,
    EngineLink,
    install_cors,
    install_host_validation,
    progress_from_health,
    proxy_to_engine,
)

__all__ = ["DEFAULT_PORT", "Home", "OpenTarget", "create_home_app", "discover_frontend", "main"]

log = logging.getLogger("dinkster.home")

DEFAULT_PORT = 3638
ENGINE_CHANGED = 1012
_WATCH_INTERVAL = 1.0


def discover_frontend() -> Path | None:
    """The frontend installed in this interpreter, found without importing it."""
    for package, parts in (
        ("dinkster_frontend", ("static",)),
        ("dinkster", ("frontend_bundle", "static")),
    ):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        for location in (spec.submodule_search_locations or []) if spec else []:
            candidate = Path(location).joinpath(*parts)
            if (candidate / "index.html").is_file():
                return candidate
    return None


def dinkster_home() -> Path:
    return Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()


def known_installations() -> frozenset[Path]:
    """Every managed installation Dinkster's store registry lists."""
    from .environment_stores import registered_stores, store_projects

    roots: set[Path] = set()
    for store in registered_stores().values():
        if Path(store).is_dir():
            roots.update(Path(project).resolve() for project in store_projects(Path(store)))
    return frozenset(root for root in roots if (root / "environment.json").is_file())


@dataclass(frozen=True)
class OpenTarget:
    root: Path
    engine: str

    def wire(self) -> dict[str, str]:
        return {"root": str(self.root), "engine": self.engine}


StartEngine = Callable[[OpenTarget, Callable[[str], None]], Coroutine[object, object, str]]


async def start_managed_engine(target: OpenTarget, found: Callable[[str], None]) -> str:
    """Start (or adopt) the installation's engine; return its supervisor address.

    ``found`` receives the supervisor address as soon as the engine's record
    names it, so startup progress can be read while the start is running.
    """
    from .environment_runtime import engine_status

    def address() -> str | None:
        for row in engine_status(target.root):
            if row["name"] == target.engine and row["state"] in {"starting", "ready"}:
                return cast(str, row["url"])
        return None

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(target.root),
        "start",
        "--engine",
        target.engine,
        "--yes",
        "--json",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
    )
    output = asyncio.ensure_future(process.communicate())
    while not output.done():
        with contextlib.suppress(OSError, ValueError):
            url = await asyncio.to_thread(address)
            if url is not None:
                found(url)
        await asyncio.wait((output,), timeout=0.25)
    stdout, _ = output.result()
    try:
        result = cast("dict[str, object]", json.loads(stdout))
    except ValueError:
        result = {}
    if process.returncode != 0 or not isinstance(result.get("url"), str):
        raise RuntimeError(str(result.get("error") or f"engine start exited {process.returncode}"))
    return cast(str, result["url"])


class Home:
    """Which installation is open, and the link the proxy follows to its engine."""

    def __init__(
        self,
        *,
        installations: Callable[[], frozenset[Path]] = known_installations,
        start: StartEngine = start_managed_engine,
        watch_interval: float = _WATCH_INTERVAL,
        default_root: Path | None = None,
    ) -> None:
        self.link = EngineLink(state="stopped")
        self.target: OpenTarget | None = None
        self._installations = installations
        self._default_root = (default_root or dinkster_home() / "project").resolve()
        self._start = start
        self._watch_interval = watch_interval
        self._task: asyncio.Task[None] | None = None
        self._on_change: list[Callable[[], Awaitable[None]]] = []
        self._changing = asyncio.Lock()

    def on_change(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._on_change.append(callback)

    async def open(self, root: Path, engine: str) -> None:
        resolved = root.resolve()
        if not root.is_absolute() or resolved not in await asyncio.to_thread(self._installations):
            raise LookupError("not an installation Dinkster knows")
        await self._replace(OpenTarget(resolved, engine))

    async def installation_count(self) -> int | None:
        with contextlib.suppress(ValueError):
            return len(await asyncio.to_thread(self._installations))
        return None

    async def restart(self) -> None:
        """Restart the open engine under its own supervisor, or open it again without one.

        With nothing open, open the default installation: the one ``dinkster``
        opens when the registry lists it, else the only one listed.
        """
        if self.target is None:
            known = await asyncio.to_thread(self._installations)
            if self._default_root in known:
                root = self._default_root
            elif len(known) == 1:
                (root,) = known
            elif not known:
                raise LookupError("No installation is set up on this machine. Run dinkster setup.")
            else:
                raise LookupError(
                    "More than one installation exists and none is the default. "
                    "Open one with dinkster --root."
                )
            await self._replace(OpenTarget(root, "main"))
            return
        if self._task is not None and not self._task.done() and self.link.state != "starting":
            timeout = aiohttp.ClientTimeout(total=10)
            with contextlib.suppress(aiohttp.ClientError, TimeoutError):
                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.post(self.link.base_url + "/supervisor/engine/restart") as response,
                ):
                    if response.status == 200:
                        self._set(state="starting")
                        return
        await self._replace(self.target)

    async def close(self) -> None:
        await self._cancel()

    async def _cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _replace(self, target: OpenTarget) -> None:
        async with self._changing:
            await self._cancel()
            self.target = target
            self._set(state="starting", base_url="")
            for callback in self._on_change:
                await callback()
            self._task = asyncio.create_task(self._run(target))

    def _set(self, *, state: str, base_url: str | None = None, detail: str = "") -> None:
        self.link.state = state
        self.link.detail = detail
        if base_url is not None:
            self.link.base_url = base_url
        if state != "ready":
            self.link.progress = None

    async def _run(self, target: OpenTarget) -> None:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def engine_status() -> dict[str, object] | None:
                """The engine supervisor's status, or None once it does not answer."""
                try:
                    async with session.get(self.link.base_url + STATUS_PATH) as response:
                        status = await response.json()
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    return None
                if not isinstance(status, dict):
                    return None
                body = cast("dict[str, object]", status)
                return body if body.get("state") in {"starting", "ready", "failed"} else None

            starting = asyncio.create_task(self._start(target, self._found))
            try:
                while not starting.done():
                    if self.link.base_url and (status := await engine_status()) is not None:
                        # Composition narration while the start is still running.
                        self.link.progress = progress_from_health(
                            {"composition": status.get("progress")}
                        )
                    await asyncio.wait((starting,), timeout=self._watch_interval)
                url = starting.result()
            except Exception as error:  # noqa: BLE001 - every start failure is reported in status
                log.warning("opening %s failed: %s", target.root, error)
                self._set(state="failed", detail=str(error))
                return
            finally:
                starting.cancel()
            self.link.base_url = url.rstrip("/")
            self._set(state="ready")
            log.info("engine %s of %s is open at %s", target.engine, target.root, url)
            while (status := await engine_status()) is not None:
                self.link.state = cast(str, status["state"])
                self.link.detail = str(status.get("detail", ""))
                self.link.progress = progress_from_health({"composition": status.get("progress")})
                await asyncio.sleep(self._watch_interval)
            self._set(state="failed", detail="the engine stopped answering")
            for callback in self._on_change:
                await callback()

    def _found(self, url: str) -> None:
        if not self.link.base_url:
            self.link.base_url = url.rstrip("/")


HOME_KEY: web.AppKey[Home] = web.AppKey("home")
FRONTEND_KEY: web.AppKey[Path | None] = web.AppKey("home_frontend")


class _FrontendFileResponse(web.FileResponse):
    async def prepare(self, request: web.BaseRequest) -> AbstractStreamWriter | None:
        # Serve the file as built; FileResponse would otherwise pick compressed siblings.
        headers = request.headers.copy()
        headers.popall("Accept-Encoding", None)
        return await super().prepare(request.clone(headers=headers))


def _frontend_file(root: Path, path: str) -> Path | None:
    if path in {"", "/"}:
        return root / "index.html"
    candidate = (root / path.lstrip("/")).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


async def _route(request: web.Request) -> web.StreamResponse:
    frontend = request.app[FRONTEND_KEY]
    if (
        frontend is not None
        and request.method in {"GET", "HEAD"}
        and request.headers.get("Upgrade", "").lower() != "websocket"
    ):
        asset = _frontend_file(frontend, request.path)
        if asset is not None:
            content_type = (
                "text/javascript"
                if asset.suffix.lower() in {".js", ".mjs"}
                else mimetypes.guess_type(asset.name)[0]
            )
            headers = {"Content-Type": content_type or "application/octet-stream"}
            if asset.name == "index.html":
                headers["Cache-Control"] = "no-cache"
            return _FrontendFileResponse(asset, headers=headers)
    if frontend is None and request.path == "/":
        return web.json_response({"error": "no-frontend", "status": STATUS_PATH}, status=503)
    home = request.app[HOME_KEY]
    origin = request.headers.get("Origin")
    if origin is not None and home.link.base_url:
        # The origin check already passed; the engine sees the request as its own page's.
        headers = request.headers.copy()
        headers["Origin"] = str(URL(home.link.base_url).origin())
        request = request.clone(headers=headers)
    return await proxy_to_engine(request)


async def _handle_status(request: web.Request) -> web.Response:
    home = request.app[HOME_KEY]
    status = home.link.status_wire()
    status["role"] = "home"
    status["supervisor"] = {"pid": os.getpid()}
    if home.target is not None:
        status["open"] = home.target.wire()
    count = await home.installation_count()
    if count is not None:
        status["installations"] = count
    instance = request.app[INSTANCE_KEY]
    if instance is not None:
        status["instance"] = instance
    return web.json_response(status)


async def _handle_open(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except ValueError:
        body = None
    fields = cast("dict[str, object]", body) if isinstance(body, dict) else {}
    root, engine = fields.get("root"), fields.get("engine", "main")
    if not isinstance(root, str) or not root or not isinstance(engine, str) or not engine:
        return web.json_response(
            {"error": "invalid-request", "message": "root and engine must be strings"}, status=400
        )
    try:
        await request.app[HOME_KEY].open(Path(root), engine)
    except LookupError as error:
        return web.json_response(
            {"error": "unknown-installation", "message": str(error)}, status=404
        )
    except ValueError as error:
        # The store registry could not be read, so no installation can be checked.
        return web.json_response(
            {"error": "installations-unavailable", "message": str(error)}, status=503
        )
    return await _handle_status(request)


async def _handle_restart(request: web.Request) -> web.Response:
    try:
        await request.app[HOME_KEY].restart()
    except LookupError as error:
        return web.json_response(
            {"error": "restart-unavailable", "message": str(error)}, status=409
        )
    except ValueError as error:
        return web.json_response(
            {"error": "installations-unavailable", "message": str(error)}, status=503
        )
    return await _handle_status(request)


async def _handle_unknown(request: web.Request) -> web.Response:
    return web.json_response(
        {"error": "unknown-supervisor-path", "status": STATUS_PATH}, status=404
    )


def create_home_app(
    home: Home,
    *,
    frontend: Path | None,
    host: str = "127.0.0.1",
    allow_hosts: Sequence[str] = (),
    allow_origins: Sequence[str] = (),
    instance: str | None = None,
) -> web.Application:
    app = web.Application()
    app[HOME_KEY] = home
    app[FRONTEND_KEY] = frontend.resolve() if frontend is not None else None
    app[LINK_KEY] = home.link
    app[WEBSOCKETS_KEY] = set()
    app[HTTP_TASKS_KEY] = set()
    app[CORS_KEY] = frozenset(allow_origins)
    app[INSTANCE_KEY] = instance
    install_host_validation(app, (host, *allow_hosts))

    @web.middleware
    async def browser_boundary(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        origins = request.headers.getall("Origin", [])
        if origins and (
            len(origins) != 1
            or origins[0] not in {*app[CORS_KEY], f"{request.scheme}://{request.host}"}
        ):
            return web.json_response({"error": "origin-not-allowed"}, status=403)
        return await handler(request)

    app.middlewares.append(browser_boundary)
    install_cors(app, app[CORS_KEY])

    async def disconnect_engine_clients() -> None:
        # Clients of the previous engine reconnect to whichever engine is open now.
        await asyncio.gather(
            *(
                ws.close(code=ENGINE_CHANGED, message=b"engine changed")
                for ws in tuple(app[WEBSOCKETS_KEY])
            )
        )
        for task in tuple(app[HTTP_TASKS_KEY]):
            task.cancel()

    home.on_change(disconnect_engine_clients)

    async def manage_session(app: web.Application) -> None:
        app[SESSION_KEY] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10),
            auto_decompress=False,
            skip_auto_headers={"Accept-Encoding"},
        )

    async def close(app: web.Application) -> None:
        await home.close()
        await asyncio.gather(
            *(ws.close(code=aiohttp.WSCloseCode.GOING_AWAY) for ws in tuple(app[WEBSOCKETS_KEY]))
        )
        tasks = tuple(app[HTTP_TASKS_KEY])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close_session(app: web.Application) -> None:
        await app[SESSION_KEY].close()

    app.on_startup.append(manage_session)
    app.on_shutdown.append(close)
    app.on_cleanup.append(close_session)
    app.router.add_get(STATUS_PATH, _handle_status)
    app.router.add_post("/supervisor/engine", _handle_open)
    app.router.add_post("/supervisor/engine/restart", _handle_restart)
    app.router.add_route("*", "/supervisor/{tail:.*}", _handle_unknown)
    app.router.add_route("*", "/{tail:.*}", _route)
    return app


def _request(url: str, body: dict[str, object] | None = None) -> dict[str, object] | None:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    data = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:  # noqa: S310 - loopback address we built
            value: object = json.loads(response.read())
    except HTTPError as error:
        value = json.loads(error.read() or b"null")
    except (URLError, OSError, ValueError):
        return None
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def ensure_running(port: int = DEFAULT_PORT, *, frontend_root: Path | None = None) -> str:
    """Start the process detached unless it already answers on ``port``; return its address."""
    import subprocess
    import time

    url = f"http://127.0.0.1:{port}"
    status = _request(url + STATUS_PATH)
    if status is None:
        home = dinkster_home()
        home.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "dinkster_supervisor.home", "--port", str(port)]
        if frontend_root is not None:
            command += ["--frontend-root", str(frontend_root)]
        with (home / "home.log").open("ab") as output:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=os.name != "nt",
            )
        deadline = time.monotonic() + 15
        while (status := _request(url + STATUS_PATH)) is None and time.monotonic() < deadline:
            time.sleep(0.05)
    if status is None or status.get("role") != "home":
        raise RuntimeError(f"port {port} does not answer as the Dinkster home process")
    return url


def open_installation(url: str, root: Path, engine: str = "main") -> str | None:
    """Ask the running process to open an installation; return the refusal, if any."""
    answer = _request(url + "/supervisor/engine", {"root": str(root), "engine": engine})
    if answer is None:
        return "the Dinkster home process did not answer"
    return cast("str | None", answer.get("message")) if "error" in answer else None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the always-on Dinkster process: serve the frontend and "
        "pass the API to the open installation's engine"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--frontend-root",
        type=Path,
        help="folder holding the frontend's index.html (default: the installed frontend)",
    )
    parser.add_argument("--allow-host", action="append", default=[], metavar="HOST")
    parser.add_argument("--allow-origin", action="append", default=[], metavar="ORIGIN")
    parser.add_argument("--instance", help="opaque launcher identifier echoed by status")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    frontend = args.frontend_root or discover_frontend()
    if frontend is not None and not (frontend / "index.html").is_file():
        parser.error(f"frontend folder has no index.html: {frontend}")
    if frontend is None:
        log.warning("no frontend is installed; only the API passes through")

    async def serve() -> None:
        stopped = asyncio.Event()
        if os.name == "posix":
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(signum, stopped.set)
        home = Home()
        app = create_home_app(
            home,
            frontend=frontend,
            host=args.host,
            allow_hosts=args.allow_host,
            allow_origins=args.allow_origin,
            instance=args.instance,
        )
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, args.host, args.port).start()
            log.info("serving on http://%s:%d", args.host, args.port)
            await stopped.wait()
        finally:
            await runner.cleanup()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
