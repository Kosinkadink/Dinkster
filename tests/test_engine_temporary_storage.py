"""Temporary cleanup is engine-local and never depends on a workflow census."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes, register_asset_type
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_schema import (
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import STATE_KEY, ServerLibrary, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario


class EchoAsset(Node):
    gate: asyncio.Event | None = None
    entered: asyncio.Event

    @classmethod
    def define_schema(cls) -> NodeSchema:
        asset = TypeExpr.concrete("ASSET")
        return NodeSchema(
            node_type="test.echo_asset",
            body=(Input("asset", asset), Output("asset", asset)),
        )

    @classmethod
    async def execute(cls, *, asset: AssetRef) -> Mapping[str, object]:
        if cls.gate is not None:
            cls.entered.set()
            await cls.gate.wait()
        return cls.outputs(asset=asset)


def write(vault: AssetVault, payload: bytes, *, temporary: bool = True) -> str:
    digest = digest_bytes(payload)
    inventory = ProducedAssetInventory(vault.root)
    if temporary:
        inventory.pending(digest, len(payload))
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
        if temporary:
            inventory.adopted(digest, len(payload))
    return digest


def make_client(library: ServerLibrary) -> TestClient:
    schemas = build_schemas((EchoAsset,))

    def engine(listener: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, library.stored_assets)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types((EchoAsset,)), registry),
            cache=MemoryLRUCache(),
            on_event=listener,
        )

    return TestClient(TestServer(create_app(engine, schemas, library=library)))


def make_library(root: Path, temporary: Path, *, keep: bool = True) -> ServerLibrary:
    ProducedAssetInventory.initialize_vault(temporary)
    vault = AssetVault(temporary)
    return ServerLibrary(
        AssetVault(root / "vault"),
        LibraryStore(root / "library.sqlite"),
        preview_vaults=(vault,),
        temporary_vaults=(vault,),
        keep_temporary_files=keep,
    )


@pytest.mark.parametrize("keep", (False, True))
@pytest.mark.parametrize("inventory", ("intact", "missing", "corrupt"))
def test_restart_and_clean_stop_only_clear_this_engines_temporary_bytes(
    tmp_path: Path, keep: bool, inventory: str
) -> None:
    async def scenario() -> None:
        library = make_library(tmp_path / "library", tmp_path / "engine" / "temporary", keep=keep)
        temporary = library.temporary_vaults[0]
        before_start = write(temporary, b"old preview")
        interrupted = write(temporary, b"interrupted preview", temporary=False)
        ProducedAssetInventory(temporary.root).pending(interrupted, len(b"interrupted preview"))
        unrecorded = write(temporary, b"unrecorded preview", temporary=False)
        staged = temporary.root / ".ingest-interrupted"
        staged.write_bytes(b"partial write")
        durable = write(library.vault, b"user saved document", temporary=False)
        inventory_path = temporary.root / ".produced-assets.sqlite"
        if inventory == "missing":
            inventory_path.unlink()
        elif inventory == "corrupt":
            inventory_path.write_bytes(b"not a database")
        client = make_client(library)
        await client.start_server()
        try:
            assert temporary.has(before_start) is keep
            assert temporary.has(interrupted) is keep
            assert temporary.has(unrecorded) is keep
            assert staged.exists() is keep
            before_stop = write(
                temporary, b"new preview after startup", temporary=inventory == "intact"
            )
            staged.write_bytes(b"interrupted before stop")
        finally:
            await client.close()
        assert staged.exists() is keep
        assert temporary.has(before_stop) is keep
        assert library.vault.has(durable)
        restarted = make_library(tmp_path / "library", temporary.root, keep=keep)
        client = make_client(restarted)
        await client.start_server()
        try:
            assert temporary.has(before_start) is keep
            assert temporary.has(before_stop) is keep
            assert restarted.vault.has(durable)
        finally:
            await client.close()

    run_scenario(scenario())


@pytest.mark.parametrize("running", (False, True))
def test_clear_keeps_active_inputs_and_other_engines_on_one_library(
    tmp_path: Path, running: bool
) -> None:
    async def scenario() -> None:
        EchoAsset.gate = asyncio.Event() if running else None
        EchoAsset.entered = asyncio.Event()
        shared = tmp_path / "library"
        first = make_library(shared, tmp_path / "engine-one" / "temporary")
        second = make_library(shared, tmp_path / "engine-two" / "temporary")
        one, two = first.temporary_vaults[0], second.temporary_vaults[0]
        queued = write(one, b"queued input")
        unrelated = write(one, b"unrelated preview")
        other = write(two, b"another engine preview")
        uploaded = write(first.vault, b"uploaded source", temporary=False)
        left, right = make_client(first), make_client(second)
        await left.start_server()
        await right.start_server()
        try:
            left.app[STATE_KEY].queue.pause()
            response = await left.post(
                "/api/jobs",
                json={
                    "clientId": "cleanup",
                    "jobId": "queued",
                    "graph": {
                        "nodes": {
                            "echo": {
                                "nodeType": "test.echo_asset",
                                "inputs": {
                                    "asset": {
                                        "digest": queued,
                                        "name": "input.bin",
                                        "size": len(b"queued input"),
                                    }
                                },
                            }
                        }
                    },
                    "targets": ["echo"],
                },
            )
            assert response.status == 202, await response.text()
            if running:
                left.app[STATE_KEY].queue.resume()
                await asyncio.wait_for(EchoAsset.entered.wait(), timeout=10)
            response = await left.post("/api/library/storage/clear", json={})
            assert response.status == 200, await response.text()
            assert one.has(queued)
            assert not one.has(unrelated)
            assert two.has(other)
            assert first.vault.has(uploaded)
            assert (await left.delete("/api/jobs/cleanup/queued")).status == 200
            if running:
                async with asyncio.timeout(10):
                    while (await (await left.get("/api/jobs/cleanup/queued")).json())[
                        "state"
                    ] != "cancelled":
                        await asyncio.sleep(0)
            assert (await left.post("/api/library/storage/clear", json={})).status == 200
            assert not one.has(queued)
            assert two.has(other)
            assert first.vault.has(uploaded)
        finally:
            await left.close()
            await right.close()

    run_scenario(scenario())
