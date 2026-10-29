"""Asset staging on remote worker daemons: a daemon whose pack declares
``[[pack.assets]]`` resolves those assets before dispatch by pulling
digest-verified bytes over HTTP from the engine's advertised endpoint (or
declared provenance URLs) - never over the worker socket itself. What this
proves: an empty daemon vault fills and the node executes; a held digest
short-circuits without a transfer; a source serving wrong bytes rolls back
to nothing and fails loudly naming digest and worker; an interrupted
transfer leaves no partial state and the daemon keeps serving; cancellation
aborts the download and rolls back; a daemon that predates staging is
refused clearly when staging is needed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp import web
from dinkster_assets import AssetVault
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_workers import RemoteWorker, TransportError
from dinkster_workers.staging import StageAsset, StagingSource
from test_declared_assets import (
    MODEL_BYTES,
    MODEL_DIGEST,
    core_registry,
    read_graph,
)
from test_relay import eventually
from test_remote import TOKEN, start_service, stop_service

ENDPOINT_TOKEN = "staging-bearer-secret-0123456789"


def write_staging_manifest(tmp_path: Path) -> Path:
    """declpack with the declaration tied to decl.read (the staging
    trigger) and a packaged source only - no external URLs, so the sole
    remote lead a daemon ever gets is the engine's advertised endpoint."""
    (tmp_path / "aux.bin").write_bytes(MODEL_BYTES)
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        "[pack]\n"
        'name = "declpack"\n'
        'namespaces = ["decl"]\n'
        "[pack.entry]\n"
        'nodes = "declpack_nodes:NODES"\n'
        "[[pack.assets]]\n"
        f'id = "aux-model"\nname = "Aux Model"\ndigest = "{MODEL_DIGEST}"\n'
        'file = "aux.bin"\n'
        'nodes = ["decl.read"]\n'
    )
    return manifest


class AssetHost:
    """Loopback HTTP asset source with scriptable behavior per test:
    serve the real bytes, serve corrupt bytes, cut the transfer short,
    or stream forever (for cancellation). Requires the bearer token,
    proving the engine's credential rides staging requests."""

    def __init__(self, mode: str = "serve") -> None:
        self.mode = mode
        self.requests = 0
        self.request_started = asyncio.Event()
        self._runner: web.AppRunner | None = None
        self.endpoint = ""

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.requests += 1
        self.request_started.set()
        if request.headers.get("Authorization") != f"Bearer {ENDPOINT_TOKEN}":
            return web.json_response({"error": "missing bearer"}, status=401)
        if request.match_info["digest"] != MODEL_DIGEST:
            return web.json_response({"error": "not held"}, status=404)
        if self.mode == "serve":
            return web.Response(body=MODEL_BYTES)
        if self.mode == "corrupt":
            return web.Response(body=b"not the declared bytes at all" * 8)
        if self.mode == "truncate":
            # Promise the full asset, send half, cut the connection: the
            # partial bytes cannot verify as the digest, so the daemon's
            # writer must roll back to nothing.
            response = web.StreamResponse()
            response.content_length = len(MODEL_BYTES)
            await response.prepare(request)
            await response.write(MODEL_BYTES[: len(MODEL_BYTES) // 2])
            assert request.transport is not None
            request.transport.abort()
            return response
        assert self.mode == "stream-forever"
        response = web.StreamResponse()
        response.enable_chunked_encoding()
        await response.prepare(request)
        block = b"\0" * (1024 * 1024)
        try:
            while True:
                await response.write(block)
                await asyncio.sleep(0.01)
        except (ConnectionError, asyncio.CancelledError):
            pass  # the daemon aborted the pull; that is the test
        return response

    async def __aenter__(self) -> AssetHost:
        app = web.Application()
        app.router.add_get("/assets/{digest}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = self._runner.addresses[0]
        self.endpoint = f"http://{server[0]}:{server[1]}"
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


def staging_worker(host: str, port: int, endpoint: str) -> RemoteWorker:
    return RemoteWorker(
        host,
        port,
        TOKEN,
        core_registry(),
        name="box1",
        connect_timeout=10.0,
        asset_endpoint=endpoint,
        asset_endpoint_token=ENDPOINT_TOKEN,
    )


def vault_files(vault_root: Path) -> list[Path]:
    """Every file under the daemon's vault: committed assets AND leftover
    ingest temporaries - rollback-to-nothing means this list is empty."""
    if not vault_root.exists():
        return []
    return [path for path in vault_root.rglob("*") if path.is_file()]


def test_empty_daemon_stages_from_engine_then_executes(tmp_path: Path) -> None:
    """The acceptance path: the daemon holds nothing, the engine advertises
    its asset endpoint, and dispatching the declared-asset node stages the
    verified bytes into the daemon's vault before the node reads them."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_staging_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost() as asset_host:
                worker = staging_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    assert worker.asset_staging
                    declared = {asset.id for asset in worker.declared_assets}
                    assert declared == {"aux-model"}
                    engine = Engine(
                        schemas=dict(worker.schemas),
                        registry=core_registry(),
                        worker=worker,
                        cache=MemoryLRUCache(),
                    )
                    result = await engine.run(read_graph(), ["r"])
                    assert result.outputs["r"]["text"].resolve() == MODEL_BYTES.decode()
                    assert asset_host.requests == 1
                    assert AssetVault(daemon_vault).has(MODEL_DIGEST)
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_held_asset_short_circuits_without_transfer(tmp_path: Path) -> None:
    """A digest the daemon's store already resolves costs one assetQuery
    round trip and zero HTTP requests."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        seeded = AssetVault(daemon_vault)
        with seeded.writer(MODEL_DIGEST) as writer:
            writer.write(MODEL_BYTES)
            writer.commit()
        proc, host, port = await start_service(
            write_staging_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost() as asset_host:
                worker = staging_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    engine = Engine(
                        schemas=dict(worker.schemas),
                        registry=core_registry(),
                        worker=worker,
                        cache=MemoryLRUCache(),
                    )
                    result = await engine.run(read_graph(), ["r"])
                    assert result.outputs["r"]["text"].resolve() == MODEL_BYTES.decode()
                    assert asset_host.requests == 0
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_digest_mismatch_rolls_back_and_names_digest_and_worker(tmp_path: Path) -> None:
    """Bytes that do not verify never publish: the daemon's vault stays
    empty (no committed asset, no ingest leftovers) and the engine-side
    error names the worker and the digest."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_staging_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost(mode="corrupt") as asset_host:
                worker = staging_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    with pytest.raises(TransportError) as failure:
                        await worker.prepare(("decl.read",))
                    assert "box1" in str(failure.value)
                    assert MODEL_DIGEST in str(failure.value)
                    assert asset_host.requests == 1
                    assert vault_files(daemon_vault) == []
                    # The failure poisons nothing: staging the real bytes
                    # afterwards succeeds on the same conversation.
                    asset_host.mode = "serve"
                    await worker.prepare(("decl.read",))
                    assert AssetVault(daemon_vault).has(MODEL_DIGEST)
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_interrupted_transfer_cleans_up(tmp_path: Path) -> None:
    """A source that cuts the connection mid-body leaves no partial state
    daemon-side, fails loudly engine-side, and the daemon keeps serving."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_staging_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost(mode="truncate") as asset_host:
                worker = staging_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    with pytest.raises(TransportError, match="box1"):
                        await worker.prepare(("decl.read",))
                    assert vault_files(daemon_vault) == []
                    asset_host.mode = "serve"
                    engine = Engine(
                        schemas=dict(worker.schemas),
                        registry=core_registry(),
                        worker=worker,
                        cache=MemoryLRUCache(),
                    )
                    result = await engine.run(read_graph(), ["r"])
                    assert result.outputs["r"]["text"].resolve() == MODEL_BYTES.decode()
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_cancellation_aborts_download_and_rolls_back(tmp_path: Path) -> None:
    """Cancelling an in-flight stage sends cancelStage: the daemon aborts
    the pull at a chunk boundary, rolls the partial file back to nothing,
    and the conversation stays usable."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_staging_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost(mode="stream-forever") as asset_host:
                worker = staging_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    session = worker._session
                    events: list[dict[str, object]] = []
                    stage = asyncio.create_task(
                        session.stage_assets(
                            [
                                StageAsset(
                                    digest=MODEL_DIGEST,
                                    name="Aux Model",
                                    sources=(
                                        StagingSource(
                                            url=(f"{asset_host.endpoint}/assets/{MODEL_DIGEST}"),
                                            headers={"Authorization": f"Bearer {ENDPOINT_TOKEN}"},
                                        ),
                                    ),
                                )
                            ],
                            on_event=events.append,
                        )
                    )
                    assert await eventually(
                        lambda: any(e.get("event") == "progress" for e in events)
                    ), "the daemon must report pull progress before the cancel"
                    stage.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await stage
                    assert await eventually(lambda: vault_files(daemon_vault) == [])
                    # The daemon is still serving this conversation: a held
                    # query answers, and a real stage lands afterwards.
                    asset_host.mode = "serve"
                    await worker.prepare(("decl.read",))
                    assert AssetVault(daemon_vault).has(MODEL_DIGEST)
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_old_daemon_without_staging_refused_when_needed(tmp_path: Path) -> None:
    """A service that never announced assetStaging (the flag defaults to
    False and only an explicit hello field sets it) gets a clear refusal
    the moment a dispatched type needs declared assets - never a silent
    skip that would surface later as an execution-time read failure."""

    async def scenario() -> None:
        proc, host, port = await start_service(write_staging_manifest(tmp_path), tmp_path)
        try:
            async with AssetHost() as asset_host:
                worker = staging_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    # Simulate the pre-staging hello: the negotiation flag
                    # is exactly what an old service never sends.
                    worker._session._asset_staging = False
                    with pytest.raises(TransportError, match="predates asset staging"):
                        await worker.prepare(("decl.read",))
                    assert asset_host.requests == 0
                    # Types that need no declared assets still dispatch.
                    await worker.prepare(())
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())
