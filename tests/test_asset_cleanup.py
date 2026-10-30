"""Operator accounting never promotes incomplete reference state into eligibility."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_server import HistoryStore, Principal, ServerLibrary, create_app
from dinkster_server.auth import StaticBearerAuthenticator
from dinkster_server.cleanup import inspect_cleanup
from dinkster_values import EncodedPayload, TypeRegistry, Value, ValueMeta
from test_history import make_record
from test_server import SCHEMAS, make_engine


def test_storage_accounting_combines_held_owners_without_enabling_collection(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        store = LibraryStore(tmp_path / "library.sqlite")
        history = HistoryStore(tmp_path / "history.sqlite")
        inventory = ProducedAssetInventory(vault.root)
        payloads = [b"ram", b"disk bytes", b"private", b"terminal", b"child"]
        digests = [digest_bytes(data) for data in payloads]
        for index, (digest, data) in enumerate(zip(digests, payloads, strict=True)):
            if index < 2:
                inventory.pending(digest, len(data))
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
                if index < 2:
                    inventory.adopted(digest, len(data))
        memory = MemoryLRUCache()
        disk = DiskCacheStore(tmp_path / "cache", TypeRegistry())
        for cache, digest, data in zip((memory, disk), digests[:2], payloads[:2], strict=True):
            ref = AssetRef(digest, "source", len(data))
            value = Value(
                "asset<file>",
                digest,
                ValueMeta(ref.to_wire()),
                EncodedPayload("asset<file>", b"opaque", None),
            )
            await cache.put("key", {"out": value})
        # A second store sharing the disk root must not acquire the same file lock twice.
        cache = LayeredCache(memory, disk, DiskCacheStore(disk.root, TypeRegistry()))
        store.create("private", "kept", digests[2], "application/octet-stream")
        history.put(make_record("run", scope="another", asset_digests=(digests[3],)))
        store.put_dependencies(digests[0], [{"digest": digests[4]}])
        store.put_dependencies(digest_bytes(b"orphan"), [{"digest": digest_bytes(b"not a root")}])
        try:
            status = await inspect_cleanup(vault, store, history, cache)
            assert status == {
                "temporaryBytes": None,
                "vaultBytes": 33,
                "adoptedBytes": 13,
                "reclaimableBytes": None,
                "knownRetainedAssets": 5,
                "blocked": ["owner_completeness_unknown", "temporary_accounting_unavailable"],
                "collectionEnabled": False,
            }
            store.retain_assets("unrelated", "preview", "unknown", [], revision=0)
            blocked = await inspect_cleanup(vault, store, history, cache)
            assert blocked["knownRetainedAssets"] is None
            assert "reference_state_unavailable" in blocked["blocked"]
            assert not blocked["collectionEnabled"]
            for digest, data in zip(digests, payloads, strict=True):
                path = vault.resolve(digest)
                assert path is not None and path.read_bytes() == data
        finally:
            history.close()
            store.close()

    asyncio.run(scenario())


def test_cancelled_accounting_keeps_all_reference_owners_until_inspection_settles(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    finish = threading.Event()

    class PausedLibrary(LibraryStore):
        @contextmanager
        def hold_asset_references(
            self, additional_roots: Sequence[str] = ()
        ) -> Generator[frozenset[str]]:
            with super().hold_asset_references(additional_roots) as roots:
                entered.set()
                if not finish.wait(10):
                    raise AssertionError("inspection was not released")
                yield roots

    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        store = PausedLibrary(tmp_path / "library")
        history = HistoryStore(tmp_path / "history")
        memory = MemoryLRUCache()
        disk = DiskCacheStore(tmp_path / "cache", TypeRegistry())
        inspection = asyncio.create_task(
            inspect_cleanup(vault, store, history, LayeredCache(memory, disk))
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            inspection.cancel()
            # A scheduled callback observes delivery without timing assumptions.
            checkpoint = asyncio.Event()
            asyncio.get_running_loop().call_soon(checkpoint.set)
            await checkpoint.wait()
            inspection.cancel()
            checkpoint.clear()
            asyncio.get_running_loop().call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not inspection.done()
            for path in (tmp_path / "library", tmp_path / "history"):
                with sqlite3.connect(path, timeout=0) as connection:
                    with pytest.raises(sqlite3.OperationalError, match="locked"):
                        connection.execute("BEGIN IMMEDIATE")
            mutation_started = asyncio.Event()

            async def replace_memory() -> None:
                mutation_started.set()
                await memory.put("replacement", {})

            mutation = asyncio.create_task(replace_memory())
            await mutation_started.wait()
            assert not mutation.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await inspection
            await mutation
            for path in (tmp_path / "library", tmp_path / "history"):
                with sqlite3.connect(path, timeout=0) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.rollback()
        finally:
            finish.set()
            await asyncio.gather(inspection, return_exceptions=True)
            history.close()
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["operator", "reader", "agent", "delegated"])
def test_storage_accounting_http_is_operator_only_and_accepts_no_paths(
    tmp_path: Path, kind: str
) -> None:
    async def scenario() -> None:
        grants = {"local": frozenset({"assets:read", "assets:write", "settings:write"})}
        principal = Principal(
            "caller",
            grants if kind != "reader" else {"local": frozenset({"assets:read"})},
            kind="agent" if kind == "agent" else "human",
            delegation_id="delegation" if kind == "delegated" else None,
        )
        library = ServerLibrary(AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library"))
        client = TestClient(
            TestServer(
                create_app(
                    make_engine,
                    SCHEMAS,
                    library=library,
                    authenticator=StaticBearerAuthenticator({"token": principal}),
                )
            )
        )
        await client.start_server()
        try:
            url = "/api/library/storage"
            headers = {"Authorization": "Bearer token"}
            assert (await client.get(url)).status == 401
            response = await client.get(url, headers=headers)
            assert response.status == (200 if kind == "operator" else 403)
            if kind == "operator":
                status = await response.json()
                assert response.headers["Cache-Control"] == "no-store"
                assert status["vaultBytes"] == 0
                assert status["adoptedBytes"] is None
                assert status["reclaimableBytes"] is None
                assert "adoption_state_unavailable" in status["blocked"]
                assert "history_reference_state_unavailable" in status["blocked"]
                assert not status["collectionEnabled"]
                assert (await client.get(url + "?path=/", headers=headers)).status == 400
                assert (await client.post(url, json={}, headers=headers)).status == 405
        finally:
            await client.close()

    asyncio.run(scenario())


def test_storage_accounting_refuses_escaping_vault_shard(tmp_path: Path) -> None:
    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        store = LibraryStore(tmp_path / "library")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / ("ab" * 32)).write_bytes(b"untouched external bytes")
        (vault.root / "ab").symlink_to(outside, target_is_directory=True)
        try:
            status = await inspect_cleanup(vault, store, None, MemoryLRUCache())
            assert status["vaultBytes"] is None
            assert "vault_accounting_unavailable" in status["blocked"]
            assert (outside / ("ab" * 32)).read_bytes() == b"untouched external bytes"
            assert not (vault.root / ".produced-assets.sqlite").exists()
        finally:
            store.close()

    asyncio.run(scenario())
