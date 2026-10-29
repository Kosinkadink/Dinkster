"""Job-referenced asset staging on remote worker daemons: an asset that
reaches a remote arm as an input value (engine-library/mount assets such as
checkpoints) crosses the boundary as a digest envelope, never bytes, so
dispatch stages the missing content into the daemon's vault from the
engine's advertised endpoint - no operator pre-seeding. What this proves:
an empty daemon vault fills before the node reads; a held digest costs zero
transfers; no advertised endpoint refuses loudly at dispatch (naming
worker, digest, and remedy) instead of a resolver error inside the node;
a daemon that predates staging is refused the same way."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from dinkster_assets import AssetVault, register_asset_type
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import Invocation
from dinkster_values import (
    CORE_STRING,
    TypeRegistry,
    asset_type_id,
    make_list_value,
    register_core_types,
)
from dinkster_workers import RemoteWorker
from test_declared_assets import MODEL_BYTES, MODEL_DIGEST
from test_remote import TOKEN, start_service, stop_service
from test_remote_asset_staging import ENDPOINT_TOKEN, AssetHost, vault_files


def write_assetpack_manifest(tmp_path: Path) -> Path:
    """A pack with an asset-input node and NO [[pack.assets]] declarations:
    the only way its digest can reach the daemon is job-asset staging."""
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        "[pack]\n"
        'name = "assetpack"\n'
        'namespaces = ["apack"]\n'
        "[pack.entry]\n"
        'nodes = "assetpack_nodes:NODES"\n'
        'types = "assetpack_nodes:register_types"\n'
    )
    return manifest


def engine_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    # Engine-side wrap only: coerces the graph's wire mapping into an
    # AssetRef envelope. No resolver - the engine never reads the bytes.
    register_asset_type(registry)
    return registry


def asset_graph() -> Graph:
    return Graph(
        nodes={
            "r": GraphNode(
                "apack.read",
                {
                    "data": {
                        "digest": MODEL_DIGEST,
                        "name": "Aux Model",
                        "size": len(MODEL_BYTES),
                    }
                },
            )
        }
    )


def job_worker(host: str, port: int, endpoint: str | None) -> RemoteWorker:
    return RemoteWorker(
        host,
        port,
        TOKEN,
        engine_registry(),
        name="box1",
        connect_timeout=10.0,
        asset_endpoint=endpoint,
        asset_endpoint_token=ENDPOINT_TOKEN if endpoint is not None else None,
    )


def build_engine(worker: RemoteWorker) -> Engine:
    return Engine(
        schemas=dict(worker.schemas),
        registry=engine_registry(),
        worker=worker,
        cache=MemoryLRUCache(),
    )


def test_empty_daemon_stages_job_asset_then_executes(tmp_path: Path) -> None:
    """The acceptance path: the daemon holds nothing, no declaration names
    the digest, and dispatching the asset-input node stages the verified
    bytes into the daemon's vault before the node reads them."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_assetpack_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost() as asset_host:
                worker = job_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    assert worker.asset_staging
                    assert worker.declared_assets == ()
                    result = await build_engine(worker).run(asset_graph(), ["r"])
                    assert result.outputs["r"]["text"].resolve() == MODEL_BYTES.decode()
                    assert result.outputs["r"]["size"].resolve() == len(MODEL_BYTES)
                    assert asset_host.requests == 1
                    assert AssetVault(daemon_vault).has(MODEL_DIGEST)
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def spy_on_asset_queries(worker: RemoteWorker, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every assetQuery the worker sends, pass-through otherwise."""
    queries: list[list[str]] = []
    real_query = worker._session.query_assets

    async def spying_query(digests: Sequence[str]) -> dict[str, Any]:
        queries.append(list(digests))
        return await real_query(digests)

    monkeypatch.setattr(worker._session, "query_assets", spying_query)
    return queries


def test_held_job_asset_short_circuits_without_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest the daemon's store already resolves costs exactly one
    assetQuery round trip and zero HTTP requests - the query proves job
    staging ran rather than being skipped."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        seeded = AssetVault(daemon_vault)
        with seeded.writer(MODEL_DIGEST) as writer:
            writer.write(MODEL_BYTES)
            writer.commit()
        proc, host, port = await start_service(
            write_assetpack_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with AssetHost() as asset_host:
                worker = job_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    queries = spy_on_asset_queries(worker, monkeypatch)
                    result = await build_engine(worker).run(asset_graph(), ["r"])
                    assert result.outputs["r"]["text"].resolve() == MODEL_BYTES.decode()
                    assert queries == [[MODEL_DIGEST]]
                    assert asset_host.requests == 0
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


@pytest.mark.parametrize("media_type", [None, "comfy.AUDIO", "dinkster.layers", "custom.media"])
def test_assets_query_once_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media_type: str | None
) -> None:
    """One deduplicated assetQuery per dispatch: duplicate typed-asset
    digests nested in a list input collapse to a single queried digest,
    and an invocation with no asset inputs never touches the asset
    protocol."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        seeded = AssetVault(daemon_vault)
        with seeded.writer(MODEL_DIGEST) as writer:
            writer.write(MODEL_BYTES)
            writer.commit()
        proc, host, port = await start_service(
            write_assetpack_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            worker = job_worker(host, port, None)
            await worker.start()
            try:
                queries = spy_on_asset_queries(worker, monkeypatch)
                registry = engine_registry()
                typed = asset_type_id(CORE_STRING)
                reference = {
                    "digest": MODEL_DIGEST,
                    "name": "Aux Model",
                    "size": len(MODEL_BYTES),
                }
                if media_type is not None:
                    typed = media_type
                    registry.register(typed, meta=lambda _: {"asset_refs": [reference]})
                envelope = registry.wrap(
                    typed,
                    reference,
                )
                schema = worker.schemas["apack.read"]
                await worker._stage_job_assets(
                    Invocation(
                        invocation_id="dup",
                        node_id="r",
                        node_type="apack.read",
                        inputs={"data": make_list_value(typed, [envelope, envelope])},
                        effective_schema=schema,
                    )
                )
                assert queries == [[MODEL_DIGEST]]
                await worker._stage_job_assets(
                    Invocation(
                        invocation_id="plain",
                        node_id="r",
                        node_type="apack.read",
                        inputs={"data": registry.wrap(CORE_STRING, "no assets here")},
                        effective_schema=schema,
                    )
                )
                assert queries == [[MODEL_DIGEST]]
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_missing_endpoint_refuses_at_dispatch(tmp_path: Path) -> None:
    """No advertised endpoint means no source for a missing job asset: the
    failure is a dispatch-time node error naming worker, digest, and the
    remedy - never a resolver error from inside the executing node - and
    the daemon's vault stays empty."""

    async def scenario() -> None:
        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_assetpack_manifest(tmp_path), tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            worker = job_worker(host, port, None)
            await worker.start()
            try:
                with pytest.raises(ExecutionError) as failure:
                    await build_engine(worker).run(asset_graph(), ["r"])
                message = str(failure.value)
                assert "box1" in message
                assert MODEL_DIGEST in message
                assert "asset_endpoint" in message
                assert vault_files(daemon_vault) == []
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_pre_staging_daemon_refused_when_job_assets_needed(tmp_path: Path) -> None:
    """A daemon that never negotiated staging cannot receive job assets:
    the dispatch refusal says so instead of failing inside the node."""

    async def scenario() -> None:
        proc, host, port = await start_service(
            write_assetpack_manifest(tmp_path), tmp_path, "--asset-vault", str(tmp_path / "v")
        )
        try:
            async with AssetHost() as asset_host:
                worker = job_worker(host, port, asset_host.endpoint)
                await worker.start()
                try:
                    # Simulate the pre-staging hello: the negotiation flag
                    # is exactly what an old service never sends.
                    worker._session._asset_staging = False
                    with pytest.raises(ExecutionError, match="predates asset staging"):
                        await build_engine(worker).run(asset_graph(), ["r"])
                    assert asset_host.requests == 0
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())
