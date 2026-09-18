"""Run a Dinkster instance: bind the app, announce it, heartbeat, withdraw.

`create_app` builds the protocol; this module gives it a life cycle. The
piece that matters is discovery integration (DESIGN 3.10): an instance
announces its *actual* endpoint only after the socket is bound - never a
guess - and re-announces on an interval so peers reading the registry see
a fresh heartbeat. Shutdown withdraws the entry first, then stops serving:
peers stop finding an instance before it stops answering, never after.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Callable, Mapping

from aiohttp import web
from dinkster_engine import Engine, EventListener
from dinkster_memory import MemoryGovernor
from dinkster_schema import NodeSchema

from .app import create_app
from .discovery import InstanceRegistry


class RunningServer:
    """One live Dinkster instance: its endpoint, its heartbeat, its shutdown."""

    def __init__(
        self,
        *,
        runner: web.AppRunner,
        endpoint: str,
        instance_id: str,
        registry: InstanceRegistry | None,
        heartbeat: asyncio.Task[None] | None,
    ) -> None:
        self._runner = runner
        self.endpoint = endpoint
        self.instance_id = instance_id
        self._registry = registry
        self._heartbeat = heartbeat
        self._closed = False

    async def close(self) -> None:
        """Withdraw from discovery, then stop serving; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            try:
                await self._heartbeat
            except asyncio.CancelledError:
                pass
        if self._registry is not None:
            self._registry.close()
        await self._runner.cleanup()

    async def __aenter__(self) -> RunningServer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


async def start_server(
    make_engine: Callable[[EventListener], Engine],
    schemas: Mapping[str, NodeSchema],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    instance_id: str | None = None,
    registry: InstanceRegistry | None = None,
    heartbeat_interval: float = 5.0,
    max_running_jobs: int = 1,
    governor: MemoryGovernor | None = None,
) -> RunningServer:
    """Bind the app and (when a registry is given) join per-machine discovery.

    ``port=0`` binds an ephemeral port; the advertised endpoint always uses
    the port actually bound. ``heartbeat_interval`` must stay comfortably
    below the registry's ``stale_after`` or peers will flap between seeing
    and losing this instance.
    """
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be > 0")
    app = create_app(make_engine, schemas, max_running_jobs=max_running_jobs, governor=governor)
    # Upgrade tickets travel in the query string; default aiohttp access
    # logging includes it verbatim in the request target.
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
        addresses = runner.addresses
        assert addresses, "TCPSite.start() bound no address"
        bound_port = int(addresses[0][1])
        host_part = f"[{host}]" if ":" in host else host
        endpoint = f"http://{host_part}:{bound_port}"
        resolved_id = instance_id or f"dinkster-{uuid.uuid4().hex[:12]}"

        heartbeat: asyncio.Task[None] | None = None
        if registry is not None:
            registry.announce(resolved_id, endpoint)

            async def beat() -> None:
                while True:
                    await asyncio.sleep(heartbeat_interval)
                    # A failed write is a missed beat, not a dead server:
                    # the entry merely ages until the next one lands.
                    with contextlib.suppress(OSError):
                        registry.announce(resolved_id, endpoint)

            heartbeat = asyncio.create_task(beat())
    except BaseException:
        await runner.cleanup()
        raise

    return RunningServer(
        runner=runner,
        endpoint=endpoint,
        instance_id=resolved_id,
        registry=registry,
        heartbeat=heartbeat,
    )
