"""The station: multi-install management over the supervisor machinery.

The Comfy Desktop role, absorbed onto what already exists: every
configured install (see :mod:`.installs`) gets its OWN public port
running the exact single-engine supervisor surface - status, restart,
transparent proxy - bound the moment the station starts, so a stopped
install's port answers an honest 503 instead of a connection refusal,
and a frontend pointed at an install port needs zero changes. One
management port on top narrates and drives the fleet:

- GET  /supervisor/status                    {"protocol": 1, "state":
                                             "ready", "role": "station",
                                             "installs": <count>} - the
                                             "role" key is what tells a
                                             client this port manages
                                             engines rather than fronting
                                             one (install ports omit it)
- GET  /supervisor/installs                  every install's config facts
                                             (root, port, autostart) plus
                                             its live engine status wire
- POST /supervisor/installs/{name}/start     409 {"error":
                                             "already-running"} when up
- POST /supervisor/installs/{name}/stop      409 {"error": "not-running"}
                                             when already stopped
- POST /supervisor/installs/{name}/restart   any state; stopped = start
- unknown /supervisor/*                      local 404, reserved namespace
- anything else                              404 {"error":
                                             "not-an-engine-port", ...} -
                                             the management port proxies
                                             NOTHING; engines live on
                                             their install ports

Engine command per install: the config's ``engine`` list (or the
station's default, ``<this python> -m dinkster.serve``) plus
``--install-root <root>``; the supervisor protocol appends
``--host/--port``. Different installs may name different interpreters -
different Dinkster versions in different venvs - which is the whole point:
this layer never imports engine code, so it can manage engines it could
not import.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

from .ingress import IngressMember, create_ingress_app
from .installs import (
    DEFAULT_INSTALLS_FILE,
    IngressDef,
    InstallDef,
    InstallsError,
    load_station_config,
)
from .leases import LeaseStore
from .supervisor import (
    PROTOCOL_VERSION,
    STATUS_PATH,
    EngineLink,
    EngineProcess,
    create_supervisor_app,
    install_cors,
)

__all__ = ["ManagedInstall", "Station", "create_station_app", "engine_command", "start_station"]

log = logging.getLogger("dinkster.station")


def engine_command(install: InstallDef, default: list[str] | None = None) -> list[str]:
    """The full engine argv for an install: its configured command (or the
    default engine) plus the install root. ``--host/--port`` ride the
    supervisor protocol, appended at spawn."""
    base = list(install.engine) if install.engine else (default or _default_engine())
    return [*base, "--install-root", str(install.root)]


def _default_engine() -> list[str]:
    return [sys.executable, "-m", "dinkster.serve"]


@dataclass
class ManagedInstall:
    """One install's runtime: config + link + process + a mutation lock
    (concurrent start/stop requests on one engine serialize, never
    interleave a terminate with a spawn)."""

    install: InstallDef
    link: EngineLink
    engine: EngineProcess
    lock: asyncio.Lock

    def wire(self) -> dict[str, object]:
        return {
            "root": str(self.install.root),
            "port": self.install.port,
            "autostart": self.install.autostart,
            "status": self.link.status_wire(),
        }


MANAGED_KEY: web.AppKey[dict[str, ManagedInstall]] = web.AppKey("station_managed")

RUNNING_STATES = frozenset({"starting", "ready"})


async def _handle_station_status(request: web.Request) -> web.Response:
    managed = request.app[MANAGED_KEY]
    return web.json_response(
        {
            "protocol": PROTOCOL_VERSION,
            "state": "ready",
            "role": "station",
            "installs": len(managed),
        }
    )


async def _handle_installs(request: web.Request) -> web.Response:
    managed = request.app[MANAGED_KEY]
    return web.json_response(
        {
            "protocol": PROTOCOL_VERSION,
            "installs": {name: entry.wire() for name, entry in managed.items()},
        }
    )


def _managed_or_none(request: web.Request) -> ManagedInstall | None:
    return request.app[MANAGED_KEY].get(request.match_info["name"])


async def _handle_start(request: web.Request) -> web.Response:
    entry = _managed_or_none(request)
    if entry is None:
        return web.json_response({"error": "unknown-install"}, status=404)
    async with entry.lock:
        if entry.link.state in RUNNING_STATES:
            return web.json_response(
                {"error": "already-running", "state": entry.link.state}, status=409
            )
        # restart(), not start(): it resets the process wrapper's closing
        # flag, so an engine stopped earlier starts cleanly again.
        await entry.engine.restart()
    return web.json_response({"ok": True, "state": entry.link.state})


async def _handle_stop(request: web.Request) -> web.Response:
    entry = _managed_or_none(request)
    if entry is None:
        return web.json_response({"error": "unknown-install"}, status=404)
    async with entry.lock:
        if entry.link.state == "stopped":
            return web.json_response({"error": "not-running", "state": "stopped"}, status=409)
        await entry.engine.close()
    return web.json_response({"ok": True, "state": entry.link.state})


async def _handle_restart_install(request: web.Request) -> web.Response:
    entry = _managed_or_none(request)
    if entry is None:
        return web.json_response({"error": "unknown-install"}, status=404)
    async with entry.lock:
        await entry.engine.restart()
    return web.json_response({"ok": True, "state": entry.link.state})


async def _handle_station_unknown(request: web.Request) -> web.Response:
    return web.json_response({"error": "unknown-supervisor-path"}, status=404)


async def _handle_not_engine(request: web.Request) -> web.Response:
    return web.json_response(
        {"error": "not-an-engine-port", "installs": "/supervisor/installs"},
        status=404,
    )


def create_station_app(
    managed: dict[str, ManagedInstall], *, allow_origins: tuple[str, ...] = ()
) -> web.Application:
    """The management surface. Proxies nothing - install engines answer on
    their own ports through the ordinary supervisor app."""
    app = web.Application()
    app[MANAGED_KEY] = managed
    install_cors(app, frozenset(allow_origins))
    app.router.add_get(STATUS_PATH, _handle_station_status)
    app.router.add_get("/supervisor/installs", _handle_installs)
    app.router.add_post("/supervisor/installs/{name}/start", _handle_start)
    app.router.add_post("/supervisor/installs/{name}/stop", _handle_stop)
    app.router.add_post("/supervisor/installs/{name}/restart", _handle_restart_install)
    app.router.add_route("*", "/supervisor/{tail:.*}", _handle_station_unknown)
    app.router.add_route("*", "/{tail:.*}", _handle_not_engine)
    return app


@dataclass
class Station:
    """A running station: the managed fleet plus every bound port."""

    managed: dict[str, ManagedInstall]
    runners: list[web.AppRunner]

    async def close(self) -> None:
        for entry in self.managed.values():
            await entry.engine.close()
        for runner in self.runners:
            await runner.cleanup()


async def start_station(
    installs: tuple[InstallDef, ...],
    *,
    host: str = "127.0.0.1",
    port: int = 3649,
    allow_origins: tuple[str, ...] = (),
    startup_timeout: float = 600.0,
    default_engine: list[str] | None = None,
    ingress: IngressDef | None = None,
) -> Station:
    """Bind the management port and every install's port, then autostart
    the installs that ask for it. Every port answers immediately; a
    stopped install's port 503s honestly. An autostart engine that fails
    leaves its link "failed" and the station serving - a broken install
    is a queryable fact, never a dead station."""
    taken = {entry.port: entry.name for entry in installs}
    if port in taken:
        raise InstallsError(f"management port {port} collides with [installs.{taken[port]}]")
    if ingress is not None and ingress.port == port:
        raise InstallsError(f"ingress port {ingress.port} collides with management port")
    if ingress is not None and ingress.port in taken:
        raise InstallsError(
            f"ingress port {ingress.port} collides with [installs.{taken[ingress.port]}]"
        )
    names = set(taken.values())
    if ingress is not None and (
        not ingress.members
        or len(set(ingress.members)) != len(ingress.members)
        or not set(ingress.members) <= names
        or ingress.primary not in ingress.members
    ):
        raise InstallsError("ingress members and primary must name configured installs")
    managed: dict[str, ManagedInstall] = {}
    for install in installs:
        link = EngineLink(state="stopped")
        engine = EngineProcess(
            engine_command(install, default_engine), link, startup_timeout=startup_timeout
        )
        managed[install.name] = ManagedInstall(install, link, engine, asyncio.Lock())

    runners: list[web.AppRunner] = []

    async def bind(app: web.Application, bind_port: int) -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        runners.append(runner)
        await web.TCPSite(runner, host, bind_port).start()

    try:
        await bind(create_station_app(managed, allow_origins=allow_origins), port)
        for entry in managed.values():
            await bind(
                create_supervisor_app(
                    entry.link, restart=entry.engine.restart, allow_origins=allow_origins
                ),
                entry.install.port,
            )
        if ingress is not None:
            store = LeaseStore(
                ingress.state_path or DEFAULT_INSTALLS_FILE.with_name("ingress.sqlite3")
            )
            await store.initialize()
            members = {name: IngressMember(name, managed[name].link) for name in ingress.members}
            await bind(
                create_ingress_app(members, ingress.primary, store, allow_origins=allow_origins),
                ingress.port,
            )
        for entry in managed.values():
            if entry.install.autostart:
                await entry.engine.start()
    except BaseException:
        for entry in managed.values():
            await entry.engine.close()
        for runner in runners:
            await runner.cleanup()
        raise
    return Station(managed=managed, runners=runners)


async def _run_forever(
    installs: tuple[InstallDef, ...],
    *,
    host: str,
    port: int,
    allow_origins: tuple[str, ...],
    startup_timeout: float,
    ingress: IngressDef | None,
) -> None:
    station = await start_station(
        installs,
        host=host,
        port=port,
        allow_origins=allow_origins,
        startup_timeout=startup_timeout,
        ingress=ingress,
    )
    names = ", ".join(sorted(station.managed)) or "none configured"
    print(f"dinkster station on http://{host}:{port} (installs: {names})")
    try:
        await asyncio.Event().wait()
    finally:
        await station.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Dinkster station: one management port, one "
        "supervised engine port per configured install"
    )
    parser.add_argument(
        "--installs",
        default=os.environ.get("DINKSTER_INSTALLS", ""),
        metavar="FILE",
        help=f"installs config (default: $DINKSTER_INSTALLS, else {DEFAULT_INSTALLS_FILE})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3649)
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="enable CORS for this browser origin (repeatable; '*' allows "
        "any) on the management port AND every install port",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="how long each engine may take to become healthy (default: 600)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    config = Path(args.installs) if args.installs else DEFAULT_INSTALLS_FILE
    try:
        station_config = load_station_config(config)
    except InstallsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            _run_forever(
                station_config.installs,
                host=args.host,
                port=args.port,
                allow_origins=tuple(args.allow_origin),
                startup_timeout=args.startup_timeout,
                ingress=station_config.ingress,
            )
        )


if __name__ == "__main__":
    main()
