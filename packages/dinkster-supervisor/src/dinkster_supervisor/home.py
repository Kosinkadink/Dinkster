"""dinkster-home: the always-on Dinkster process.

One per user. It owns the public address and serves the frontend itself, so
the browser and Desktop show a page before any engine runs, and it passes
every other path, ``/api`` and the events socket included, to the engine of
the installation that is open. Engines keep running under their own managed
supervisors (``environment_runtime.start_engine``); this process starts them
through ``dinkster-env start`` and proxies to the supervisor's address.

Routing, in order:
- ``/supervisor/*`` answers here (status, open, restart, stop, shutdown);
- ``GET /api/nodes/provisional`` answers here, whatever the engine's state:
  the node table the engine recorded at its last run, marked
  ``"provisional": true`` and carrying no epoch, so an editor can draw a
  project before its engine is up (404 when there is no record);
- ``/`` and any path naming a file of the frontend are the frontend;
- everything else passes to the open engine when it is ready, else 503
  ``engine-not-ready``. The events socket is refused the same way, and every
  proxied socket closes with 1012 when the open engine changes or stops, so
  an editor's next events connection is always a new engine run.

Status wire: the supervisor's protocol 1 (see ``supervisor.py``) plus
``"role": "home"``, ``"installations"`` (how many the store registry lists)
``"frontend"`` (the folder it serves, so a client can tell whether it is the
one it would have served) and ``"open": {"root", "engine"}`` while an
installation is open. Top-level ``state`` is the open engine's, ``stopped``
when none is open or the open engine was stopped on purpose. Restart with
nothing open opens the default installation, or refuses with a ``message``
saying why.

``POST /supervisor/engine/stop`` stops the open engine and keeps its
installation open. ``POST /supervisor/shutdown`` stops every engine this process opened, answers
with what it stopped, and then the process exits. While it runs, the process
records its address and pid in ``$DINKSTER_HOME/home.json`` so ``dinkster
stop`` finds it without probing a port.
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
_STOP_GRACE = 30.0
"""How long a stopped engine's record may lag its listener: the supervisor's stop timeout."""


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
StopEngine = Callable[[OpenTarget], None]
EngineStopped = Callable[[OpenTarget], bool]


def recorded_node_table(target: OpenTarget) -> dict[str, object] | None:
    """The engine's last recorded node table, marked provisional; None without a record.

    A record from another snapshot than the installation's current one keeps
    only what that snapshot's sealed catalogs cannot replace (see
    ``supersede_from_catalogs``).
    """
    from .environment_runtime import node_table_record
    from .environment_state import EnvironmentError, read_current

    try:
        record: object = json.loads(
            node_table_record(target.root, target.engine).read_text(encoding="utf-8")
        )
        current = read_current(target.root)
    except (OSError, ValueError, EnvironmentError):
        return None
    if not isinstance(record, dict):
        return None
    table = cast("dict[str, object]", record)
    if current is not None and table.get("snapshot") != str(current):
        table = supersede_from_catalogs(table, target.root / "builds" / str(current))
    return {**table, "provisional": True}


def snapshot_catalog_nodes(generation: Path) -> dict[str, object] | None:
    """The node entries a sealed snapshot's catalogs supply, read once per snapshot.

    Reading them needs the snapshot's own packages, so it runs as a short
    command in the snapshot's interpreter; nothing is composed.
    """
    import subprocess

    from .environment_state import EnvironmentError, interpreter

    kept = generation.parent.parent / "node-tables" / f"catalogs-{generation.name}.json"
    with contextlib.suppress(OSError, ValueError):
        return cast("dict[str, object]", json.loads(kept.read_text(encoding="utf-8")))
    try:
        result = subprocess.run(
            [
                str(interpreter(generation)),
                "-I",
                "-B",
                "-m",
                "dinkster.manager",
                "--root",
                "",
                "catalog-nodes",
                "--generation",
                str(generation),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.SubprocessError, EnvironmentError) as error:
        log.warning("reading the catalogs of %s failed: %s", generation, error)
        return None
    if result.returncode:
        log.warning("reading the catalogs of %s failed: %s", generation, result.stderr.strip())
        return None
    try:
        catalogs = cast("dict[str, object]", json.loads(result.stdout))
    except ValueError:
        return None
    with contextlib.suppress(OSError):
        kept.parent.mkdir(parents=True, exist_ok=True)
        staged = kept.with_name(f".{kept.name}.pending")
        staged.write_text(result.stdout, encoding="utf-8")
        os.replace(staged, kept)
    return catalogs


def supersede_from_catalogs(table: dict[str, object], generation: Path) -> dict[str, object]:
    """Bring a record from an older snapshot up to ``generation`` where its catalogs can.

    Nodes of every pack with a sealed catalog come from that catalog; nodes of
    an artifact pack the snapshot no longer installs are dropped; nodes the
    catalogs cannot cover (core, development packs, remote workers) stay.
    """
    catalogs = snapshot_catalog_nodes(generation)
    if catalogs is None:
        return table
    installed = set(cast("list[str]", catalogs["packs"]))
    replacements = cast("dict[str, dict[str, object]]", catalogs["nodes"])
    replaced = {str(entry["pack"]) for entry in replacements.values()}
    packs = cast("dict[str, dict[str, object]]", table.get("packs", {}))
    gone = {
        pack for pack, info in packs.items() if "artifactDigest" in info and pack not in installed
    }
    nodes = {
        node_type: entry
        for node_type, entry in cast("dict[str, dict[str, object]]", table["nodes"]).items()
        if entry.get("pack") not in replaced | gone
    }
    return {
        **table,
        "packs": {pack: info for pack, info in packs.items() if pack not in gone}
        | {pack: packs.get(pack, {"displayName": pack}) for pack in replaced},
        "nodes": nodes | replacements,
    }


def project_installation(project: str, known: frozenset[Path]) -> Path | None:
    """The installation whose environment the library names for ``project``."""
    from dinkster_library import LibraryClient, LibraryError

    from .environment_state import EnvironmentError, library_root, read_environment_identity

    by_identity: dict[str, Path] = {}
    libraries: set[str] = set()
    for root in sorted(known):
        with contextlib.suppress(OSError, EnvironmentError):
            if (identity := read_environment_identity(root)) is not None:
                by_identity[identity] = root
            libraries.add(library_root(root))
    for library in sorted(libraries):
        with contextlib.suppress(LibraryError, OSError, ValueError, KeyError, TypeError):
            with LibraryClient(Path(library)) as helper:
                selected = cast(
                    "dict[str, object] | None", helper.call("projects.get", id=project)["project"]
                )
            environment = selected.get("environment") if selected is not None else None
            if isinstance(environment, dict):
                found = by_identity.get(str(cast("dict[str, object]", environment).get("id")))
                if found is not None:
                    return found
    return None


def stop_managed_engine(target: OpenTarget) -> None:
    from .environment_runtime import stop_engine

    stop_engine(target.root, target.engine)


def managed_engine_stopped(target: OpenTarget) -> bool:
    """True when the installation's record says its engine is stopped, as after a stop."""
    from .environment_runtime import engine_status

    try:
        rows = engine_status(target.root)
    except (OSError, ValueError):
        return False
    return any(row["name"] == target.engine and row["state"] == "stopped" for row in rows)


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
        stop: StopEngine = stop_managed_engine,
        stopped: EngineStopped = managed_engine_stopped,
        watch_interval: float = _WATCH_INTERVAL,
        default_root: Path | None = None,
    ) -> None:
        self.link = EngineLink(state="stopped")
        self.target: OpenTarget | None = None
        self.opened: list[OpenTarget] = []
        """Every engine this process opened, in order; shutdown stops them all."""
        self._installations = installations
        self._default_root = (default_root or dinkster_home() / "project").resolve()
        self._start = start
        self._stop = stop
        self._stopped = stopped
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
            await self._replace(OpenTarget(self._default_installation(known), "main"))
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

    def _default_installation(self, known: frozenset[Path]) -> Path:
        if self._default_root in known:
            return self._default_root
        if len(known) == 1:
            (root,) = known
            return root
        if not known:
            raise LookupError("No installation is set up on this machine.")
        raise LookupError(
            "More than one installation exists and none is the default. "
            "Open one with dinkster --root."
        )

    async def provisional_nodes(self, project: str | None) -> dict[str, object] | None:
        """The node table recorded by the engine an editor of ``project`` would reach.

        That is the open engine; with none open, the main engine of the
        project's installation, else of the installation restart would open.
        """
        target = self.target
        if target is None:
            try:
                known = await asyncio.to_thread(self._installations)
                root = (
                    await asyncio.to_thread(project_installation, project, known)
                    if project is not None
                    else None
                ) or self._default_installation(known)
            except (LookupError, ValueError):
                return None
            target = OpenTarget(root, "main")
        return await asyncio.to_thread(recorded_node_table, target)

    async def stop_open(self) -> None:
        """Stop the open engine and leave its installation open, so restart starts it again."""
        async with self._changing:
            target = self.target
            if target is None:
                raise LookupError("No installation is open.")
            await self._cancel()
            self._set(state="stopped", base_url="")
            for callback in self._on_change:
                await callback()
            await asyncio.to_thread(self._stop, target)

    async def shutdown(self) -> tuple[list[OpenTarget], dict[OpenTarget, str]]:
        """Stop every engine this process opened; return those stopped and those that failed."""
        async with self._changing:
            await self._cancel()
            self._set(state="stopped", base_url="")
            for callback in self._on_change:
                await callback()
            stopped: list[OpenTarget] = []
            failed: dict[OpenTarget, str] = {}
            for target in self.opened:
                try:
                    await asyncio.to_thread(self._stop, target)
                except Exception as error:  # noqa: BLE001 - each failure is reported to the caller
                    failed[target] = str(error) or type(error).__name__
                else:
                    stopped.append(target)
            return stopped, failed

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
            if target not in self.opened:
                self.opened.append(target)
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
            # A deliberate stop closes the engine's listener before its record
            # says stopped; reread the record for as long as such a stop may take.
            deadline = asyncio.get_running_loop().time() + _STOP_GRACE
            while asyncio.get_running_loop().time() < deadline:
                if await asyncio.to_thread(self._stopped, target):
                    self._set(state="stopped", base_url="")
                    return
                await asyncio.sleep(self._watch_interval)

    def _found(self, url: str) -> None:
        if not self.link.base_url:
            self.link.base_url = url.rstrip("/")


HOME_KEY: web.AppKey[Home] = web.AppKey("home")
FRONTEND_KEY: web.AppKey[Path | None] = web.AppKey("home_frontend")
EXIT_KEY: web.AppKey[Callable[[], None]] = web.AppKey("home_exit")


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
    frontend = request.app[FRONTEND_KEY]
    if frontend is not None:
        status["frontend"] = str(frontend)
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


async def _handle_stop_engine(request: web.Request) -> web.Response:
    try:
        await request.app[HOME_KEY].stop_open()
    except LookupError as error:
        return web.json_response({"error": "nothing-open", "message": str(error)}, status=409)
    except Exception as error:  # noqa: BLE001 - the refusal is reported to the caller
        return web.json_response(
            {"error": "stop-failed", "message": str(error) or type(error).__name__}, status=500
        )
    return await _handle_status(request)


async def _handle_shutdown(request: web.Request) -> web.Response:
    stopped, failed = await request.app[HOME_KEY].shutdown()
    # Exit after this answer is sent; the runner's cleanup waits for it.
    asyncio.get_running_loop().call_soon(request.app[EXIT_KEY])
    return web.json_response(
        {
            "status": "stopped" if not failed else "partial",
            "stopped": [target.wire() for target in stopped],
            "failed": [target.wire() | {"message": message} for target, message in failed.items()],
        }
    )


async def _handle_provisional_nodes(request: web.Request) -> web.Response:
    table = await request.app[HOME_KEY].provisional_nodes(request.query.get("project"))
    if table is None:
        return web.json_response({"error": "no-node-table"}, status=404)
    return web.json_response(table)


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
    exit: Callable[[], None] = lambda: None,
) -> web.Application:
    """The process's web app. ``exit`` is called once a shutdown request has been answered."""
    app = web.Application()
    app[HOME_KEY] = home
    app[EXIT_KEY] = exit
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
    app.router.add_post("/supervisor/engine/stop", _handle_stop_engine)
    app.router.add_post("/supervisor/shutdown", _handle_shutdown)
    app.router.add_route("*", "/supervisor/{tail:.*}", _handle_unknown)
    app.router.add_get("/api/nodes/provisional", _handle_provisional_nodes)
    app.router.add_route("*", "/{tail:.*}", _route)
    return app


def _request(
    url: str, body: dict[str, object] | None = None, *, timeout: float = 5
) -> dict[str, object] | None:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    data = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback address we built
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
            # Detached from the terminal that ran ``dinkster``, so closing it leaves Dinkster up.
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        deadline = time.monotonic() + 15
        while (status := _request(url + STATUS_PATH)) is None and time.monotonic() < deadline:
            time.sleep(0.05)
    if status is None or status.get("role") != "home":
        raise RuntimeError(f"port {port} does not answer as the Dinkster home process")
    return url


def running_status() -> tuple[str, dict[str, object]] | None:
    """The address and status of the process this Dinkster home folder's ``home.json``
    names, when that process still answers there."""
    try:
        recorded = json.loads((dinkster_home() / "home.json").read_text(encoding="utf-8"))
        url, pid = str(recorded["url"]), int(recorded["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    status = _request(url + STATUS_PATH)
    supervisor = status.get("supervisor") if status is not None else None
    if (
        status is None
        or status.get("role") != "home"
        or not isinstance(supervisor, dict)
        or cast("dict[str, object]", supervisor).get("pid") != pid
    ):
        return None
    return url, status


def stop_running() -> dict[str, object] | None:
    """Shut down the process ``running_status`` finds; None when none is running.

    Returns its answer: what it stopped.
    """
    running = running_status()
    if running is None:
        return None
    # Each engine may take its supervisor's whole stop timeout to stop.
    return _request(running[0] + "/supervisor/shutdown", {}, timeout=300) or {
        "status": "error",
        "message": "the Dinkster home process did not answer",
    }


def stop_open_engine(root: Path, engine: str) -> dict[str, object] | None:
    """Ask the running process to stop its open engine when that is ``engine`` of ``root``.

    Returns its status afterwards, or None when no process has that engine open.
    """
    running = running_status()
    if running is None or running[1].get("open") != {"root": str(root), "engine": engine}:
        return None
    return _request(running[0] + "/supervisor/engine/stop", {}, timeout=300) or {
        "error": "no-answer",
        "message": "the Dinkster home process did not answer",
    }


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
            exit=stopped.set,
        )
        runner = web.AppRunner(app)
        await runner.setup()
        record = dinkster_home() / "home.json"
        connect = "127.0.0.1" if args.host in {"", "0.0.0.0", "::"} else args.host
        try:
            await web.TCPSite(runner, args.host, args.port).start()
            log.info("serving on http://%s:%d", args.host, args.port)
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(
                json.dumps({"url": f"http://{connect}:{args.port}", "pid": os.getpid()}),
                encoding="utf-8",
            )
            await stopped.wait()
        finally:
            await runner.cleanup()
            with contextlib.suppress(OSError, ValueError):
                if json.loads(record.read_text(encoding="utf-8")).get("pid") == os.getpid():
                    record.unlink()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
