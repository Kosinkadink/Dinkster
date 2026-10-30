"""Operator accounting never promotes incomplete reference state into eligibility."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
import threading
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes, open_verified
from dinkster_assets.declared import produced_output_scratch, use_produced_output_job
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchState,
    claim_inactive_produced_vault,
    hold_produced_job,
    hold_produced_vault,
)
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_server import (
    STATE_KEY,
    HistoryStore,
    Job,
    JobQueue,
    Principal,
    ServerLibrary,
    create_app,
)
from dinkster_server.auth import StaticBearerAuthenticator
from dinkster_server.cleanup import inspect_cleanup
from dinkster_server.library import run_media_publication, run_owned_thread
from dinkster_values import EncodedPayload, TypeRegistry, Value, ValueMeta
from test_history import make_record
from test_server import SCHEMAS, Sleeper, echo_graph, make_engine, reset_sleeper, wait_for_state


def test_cancelled_publication_retains_vault_until_reference_commit(tmp_path: Path) -> None:
    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        store = LibraryStore(tmp_path / "library")
        library = ServerLibrary(vault, store)
        entered = threading.Event()
        release = threading.Event()
        data = b"published before reference commit"
        digest = digest_bytes(data)
        ProducedAssetInventory(vault.root).pending(digest, len(data))

        def publish() -> None:
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
            entered.set()
            assert release.wait(10)
            store.create("private", "kept", digest, "application/octet-stream")

        task = asyncio.create_task(run_media_publication(library, publish))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert not inactive
            assert store.query("private") == []
            release.set()
            async with asyncio.timeout(10), library.media_ingest_lock:
                with claim_inactive_produced_vault(vault.root) as inactive:
                    assert inactive
                assert [record.digest for record in store.query("private")] == [digest]
                path = vault.resolve(digest)
                assert path is not None and path.read_bytes() == data
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            async with library.media_ingest_lock:
                store.close()

    asyncio.run(scenario())


def test_job_settlement_holds_dispatch_and_identity_through_repeated_cancellation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        entered = threading.Event()
        release = threading.Event()

        def work() -> None:
            entered.set()
            assert release.wait(10)

        async def settle(job: Job) -> None:
            await run_owned_thread(work)

        vault = AssetVault(tmp_path / "vault")
        ProducedAssetInventory(vault.root).pending(digest_bytes(b"pending"), 7)
        terminal_claims: list[bool] = []

        def terminal(job: Job) -> None:
            if job.state in {"completed", "failed", "cancelled"}:
                with claim_inactive_produced_vault(vault.root) as inactive:
                    terminal_claims.append(inactive)

        queue = JobQueue(
            make_engine(),
            settle_job=settle,
            job_context=lambda job: closing(hold_produced_vault(vault.root)),
            on_job_event=terminal,
        )
        queue.start()
        try:
            job = queue.submit("client", "job", echo_graph(), ["s"])
            assert await asyncio.to_thread(entered.wait, 10)
            assert job.state == "running"
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert not inactive
            assert queue.submit("client", "job", echo_graph(), ["s"]) is job
            with pytest.raises(ValueError, match="in use"):
                queue.submit("client", "job", echo_graph("different"), ["s"])
            following = queue.submit("client", "next", echo_graph("next"), ["s"])
            assert job.task is not None
            for _ in range(2):
                job.task.cancel()
                await asyncio.sleep(0)
                assert not job.task.done()
                assert following.state == "queued"
                assert queue.running() == [job]
                with claim_inactive_produced_vault(vault.root) as inactive:
                    assert not inactive
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await job.task
            assert job.state == "completed"
            assert job.result is not None
            await wait_for_state(following, "completed")
            assert queue.running() == []
            assert terminal_claims == [False, False]
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert inactive
        finally:
            release.set()
            await queue.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_terminal_jobs_recover_only_their_offered_publications(
    tmp_path: Path, outcome: str, corrupt: bool
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        vault = AssetVault(tmp_path / "vault")
        data = b"terminal offered publication"
        digest = digest_bytes(data)
        ProducedAssetInventory(vault.root).pending(digest, len(data))
        sources: list[Path] = []

        @contextmanager
        def running(job: str) -> Generator[None]:
            host = hold_produced_job(scratch, job)
            try:
                with claim_inactive_produced_vault(vault.root) as inactive:
                    assert not inactive
                with use_produced_output_job(job, 1):
                    source = produced_output_scratch(
                        {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
                    )
                    assert source is not None
                    with source.writer(digest) as writer:
                        writer.write(data)
                        path = writer.commit()
                        sources.append(path)
                    ProducedScratchState(source.root).offer([digest])
                    if corrupt:
                        path.write_bytes(b"x" * len(data))
                yield
            finally:
                host.close()

        def factory(on_event: EventListener) -> Engine:
            base = make_engine(on_event)
            runtime = replace(base.pin_execution(), run_context=running)
            return Engine(
                schemas=SCHEMAS,
                registry=base.registry,
                worker=runtime.worker,
                cache=MemoryLRUCache(),
                on_event=on_event,
                pin_execution=lambda: runtime,
            )

        library = ServerLibrary(vault, LibraryStore(tmp_path / "library"), scratch_root=scratch)
        app = create_app(factory, SCHEMAS, library=library)
        client = TestClient(TestServer(app))
        await client.start_server()
        other = hold_produced_job(scratch, "unrelated job")
        unrelated = b"unrelated offered publication"
        unrelated_digest = digest_bytes(unrelated)
        with use_produced_output_job("unrelated job", 1):
            other_source = produced_output_scratch(
                {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "other")}
            )
            assert other_source is not None
            with other_source.writer(unrelated_digest) as writer:
                writer.write(unrelated)
                other_path = writer.commit()
            ProducedScratchState(other_source.root).offer([unrelated_digest])
        if outcome == "failed":
            other.close()
        try:
            reset_sleeper()
            node = {
                "completed": {"nodeType": "test.echo", "inputs": {"text": "done"}},
                "failed": {"nodeType": "test.boom", "inputs": {"tag": "failed"}},
                "cancelled": {"nodeType": "test.sleeper", "inputs": {"tag": "cancel"}},
            }[outcome]
            response = await client.post(
                "/api/jobs",
                json={
                    "clientId": "recovery",
                    "jobId": outcome,
                    "graph": {"nodes": {"n": node}},
                    "targets": ["n"],
                },
            )
            assert response.status == 202, await response.text()
            job = app[STATE_KEY].queue.get("recovery", outcome)
            assert job is not None
            async with asyncio.timeout(10):
                if outcome == "cancelled":
                    await Sleeper.entered.wait()
                    cancelled = await client.delete(f"/api/jobs/recovery/{outcome}")
                    assert cancelled.status == 200
                while job.task is None:
                    await asyncio.sleep(0)
                try:
                    await job.task
                except asyncio.CancelledError:
                    assert outcome == "cancelled"
            assert job.state == outcome
            status = await (await client.get("/api/library/storage")).json()
            assert status["lastPublicationRecovery"] == {
                "verifiedPublications": 0 if corrupt else 1,
                "blockedInspections": 1 if corrupt else 0,
            }
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert inactive
            response = await client.get(f"/api/assets/{digest}")
            if corrupt:
                assert response.status == 404
                assert sources[0].read_bytes() == b"x" * len(data)
            else:
                assert response.status == 200 and await response.read() == data
                assert sources[0].read_bytes() == data
            assert other_path.read_bytes() == unrelated
            assert vault.resolve(unrelated_digest) is None
            assert not status["collectionEnabled"]
        finally:
            other.close()
            await client.close()

    asyncio.run(scenario())


def _crash_after_offering(channel: Connection, scratch: str) -> None:
    root = Path(scratch)
    host = hold_produced_job(root, "crashed job")
    with use_produced_output_job("crashed job", 1):
        source = produced_output_scratch({"DINKSTER_PACK_SCRATCH": str(root / "packs" / "media")})
        assert source is not None
        data = b"offered before process exit"
        digest = digest_bytes(data)
        with source.writer(digest) as writer:
            writer.write(data)
            path = writer.commit()
        ProducedScratchState(source.root).offer([digest])
        channel.send(str(path))
        assert channel.recv() == "crash"
        assert host is not None
        os._exit(0)


def test_startup_recovers_publication_after_producer_process_crash(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_crash_after_offering, args=(child, str(scratch)))
    process.start()
    child.close()
    try:
        assert parent.poll(15)
        source = Path(parent.recv())
        parent.send("crash")
        process.join(15)
        assert process.exitcode == 0

        async def scenario() -> None:
            vault = AssetVault(tmp_path / "vault")
            library = ServerLibrary(vault, LibraryStore(tmp_path / "library"), scratch_root=scratch)
            client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
            await client.start_server()
            try:
                status = await (await client.get("/api/library/storage")).json()
                assert status["lastPublicationRecovery"] == {
                    "verifiedPublications": 1,
                    "blockedInspections": 0,
                }
                data = b"offered before process exit"
                digest = digest_bytes(data)
                response = await client.get(f"/api/assets/{digest}")
                assert response.status == 200 and await response.read() == data
                assert source.read_bytes() == data
            finally:
                await client.close()

        asyncio.run(scenario())
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(15)


@pytest.mark.parametrize("destination", ["absent", "pending", "untracked"])
def test_startup_recovers_offered_publications_without_collecting_scratch(
    tmp_path: Path, destination: str
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        vault = AssetVault(tmp_path / "vault")
        host = hold_produced_job(scratch, "interrupted job")
        data = b"recover offered bytes"
        digest = digest_bytes(data)
        unused = b"not offered for handoff"
        with use_produced_output_job("interrupted job", 3):
            source = produced_output_scratch(
                {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
            )
            assert source is not None
            for payload in (data, unused):
                with source.writer(digest_bytes(payload)) as writer:
                    writer.write(payload)
                    writer.commit()
            ProducedScratchState(source.root).offer([digest])
        host.close()
        inventory = ProducedAssetInventory(vault.root)
        if destination != "absent":
            if destination == "pending":
                inventory.pending(digest, len(data))
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
        before = {
            p.relative_to(source.root): p.read_bytes()
            for p in source.root.rglob("*")
            if p.is_file()
        }
        timestamp: int | None = None
        for _ in range(2):
            library = ServerLibrary(vault, LibraryStore(tmp_path / "library"), scratch_root=scratch)
            client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
            await client.start_server()
            try:
                response = await client.get("/api/library/storage")
                assert response.status == 200
                status = await response.json()
                assert status["lastPublicationRecovery"] == {
                    "verifiedPublications": 1,
                    "blockedInspections": 0,
                }
                assert not status["collectionEnabled"]
                assert (await client.get(f"/api/assets/{digest}")).status == 200
                path = vault.resolve(digest)
                assert path is not None and path.read_bytes() == data
                assert vault.resolve(digest_bytes(unused)) is None
                assert {
                    p.relative_to(source.root): p.read_bytes()
                    for p in source.root.rglob("*")
                    if p.is_file()
                } == before
                assert library.store.query("local") == []
                if destination == "untracked":
                    assert not (vault.root / ".produced-assets.sqlite").exists()
                else:
                    (entry,) = inventory.entries()
                    assert entry.digest == digest and entry.size == len(data)
                    assert entry.adopted_at_ns is not None
                    if timestamp is not None:
                        assert entry.adopted_at_ns == timestamp
                    timestamp = entry.adopted_at_ns
            finally:
                await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault", ["host", "writer", "source", "destination", "symlink", "size", "journal"]
)
def test_startup_blocks_unsafe_recovery_without_blocking_other_jobs(
    tmp_path: Path, fault: str
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        vault = AssetVault(tmp_path / "vault")
        host = hold_produced_job(scratch, "blocked job")
        data = b"recoverable source"
        digest = digest_bytes(data)
        with use_produced_output_job("blocked job", 1):
            source = produced_output_scratch(
                {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
            )
            assert source is not None
            with source.writer(digest) as writer:
                writer.write(data)
                path = writer.commit()
            ProducedScratchState(source.root).offer([digest])
            live_writer = source.writer(digest_bytes(b"unfinished")) if fault == "writer" else None
        if fault != "host":
            host.close()
        destination = vault.root / digest[7:9] / digest[7:]
        if fault == "source":
            path.write_bytes(b"x" * len(data))
        elif fault == "destination":
            destination.parent.mkdir()
            destination.write_bytes(b"x" * len(data))
        elif fault == "symlink":
            outside = tmp_path / "outside"
            outside.write_bytes(data)
            path.unlink()
            path.symlink_to(outside)
        elif fault == "size":
            path.write_bytes(data + b"extra")
        elif fault == "journal":
            (source.root / ".produced-scratch.sqlite").unlink()
        before = path.read_bytes()
        healthy = b"another job can recover"
        healthy_digest = digest_bytes(healthy)
        other = hold_produced_job(scratch, "healthy job")
        with use_produced_output_job("healthy job", 2):
            healthy_source = produced_output_scratch(
                {"DINKSTER_PACK_SCRATCH": str(scratch / "groups" / "healthy")}
            )
            assert healthy_source is not None
            with healthy_source.writer(healthy_digest) as writer:
                writer.write(healthy)
                healthy_path = writer.commit()
            ProducedScratchState(healthy_source.root).offer([healthy_digest])
        other.close()
        library = ServerLibrary(vault, LibraryStore(tmp_path / "library"), scratch_root=scratch)
        client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
        try:
            await client.start_server()
            response = await client.get("/api/library/storage")
            assert response.status == 200
            status = await response.json()
            assert status["lastPublicationRecovery"] == {
                "verifiedPublications": 1,
                "blockedInspections": 1,
            }
            assert not status["collectionEnabled"]
            assert path.read_bytes() == before
            if fault == "destination":
                assert destination.read_bytes() == b"x" * len(data)
            else:
                assert not destination.exists()
            adopted = vault.resolve(healthy_digest)
            assert adopted is not None and adopted.read_bytes() == healthy
            assert healthy_path.read_bytes() == healthy
        finally:
            await client.close()
            host.close()
            if live_writer is not None:
                live_writer.__exit__(None, None, None)

    asyncio.run(scenario())


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
        cache = LayeredCache(
            memory, disk, DiskCacheStore(disk.root / ".." / "cache", TypeRegistry())
        )
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
                "vaultOwnershipAvailable": True,
                "blocked": ["owner_completeness_unknown", "temporary_accounting_unavailable"],
                "collectionEnabled": False,
            }
            held_path = vault.resolve(digests[0])
            assert held_path is not None
            with open_verified(held_path, digests[0]):
                reading = await inspect_cleanup(vault, store, history, cache)
                assert reading["vaultOwnershipAvailable"] is False
                assert "vault_in_use" in cast(list[str], reading["blocked"])
            store.retain_assets("unrelated", "preview", "unknown", [], revision=0)
            blocked = await inspect_cleanup(vault, store, history, cache)
            assert blocked["knownRetainedAssets"] is None
            assert "reference_state_unavailable" in cast(list[str], blocked["blocked"])
            assert not blocked["collectionEnabled"]
            for digest, data in zip(digests, payloads, strict=True):
                path = vault.resolve(digest)
                assert path is not None and path.read_bytes() == data
        finally:
            history.close()
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["packs", "groups"])
def test_scratch_accounting_waits_for_host_and_store_owners_without_deleting(
    tmp_path: Path, scope: str
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        pack = scratch / scope / "media"
        env = {"DINKSTER_PACK_SCRATCH": str(pack)}
        store = LibraryStore(tmp_path / "library")
        vault = AssetVault(tmp_path / "vault")
        memory = MemoryLRUCache()

        async def status() -> dict[str, object]:
            return await inspect_cleanup(vault, store, None, memory, scratch_root=scratch)

        host = hold_produced_job(scratch, "../job with a path")
        try:
            with use_produced_output_job("../job with a path", 2):
                produced = produced_output_scratch(env)
                assert produced is not None
                data = b"publication whose adoption is not established"
                with produced.writer(digest_bytes(data)) as writer:
                    writer.write(data)
                    path = writer.commit()
                assert (await status())["temporaryBytes"] is None
                host.close()
                assert (await status())["temporaryBytes"] is None
                host = hold_produced_job(scratch, "../job with a path")
            # The invocation is gone, but the host still protects invocation gaps.
            assert (await status())["temporaryBytes"] is None
            host.close()
            observed = await status()
            size = observed["temporaryBytes"]
            assert isinstance(size, int) and size >= len(data)
            assert "temporary_accounting_unavailable" not in cast(list[str], observed["blocked"])
            assert not observed["collectionEnabled"]
            assert path.read_bytes() == data
            (produced.root / ".ingest-interrupted").write_bytes(b"unfinished seven")
            (pack / "unrelated-user-file").write_bytes(b"outside" * 1000)
            assert (await status())["temporaryBytes"] == size + 16
            # A new host can reuse the job only after inspection releases its claim.
            next_host = hold_produced_job(scratch, "../job with a path")
            try:
                assert (await status())["temporaryBytes"] is None
            finally:
                next_host.close()
            outside = tmp_path / "outside"
            outside.write_bytes(b"not managed")
            escape = produced.root / "escape"
            escape.symlink_to(outside)
            assert (await status())["temporaryBytes"] is None
            assert outside.read_bytes() == b"not managed"
            escape.unlink()
            moved = produced.root.with_name("wrong-owner-location")
            produced.root.rename(moved)
            assert (await status())["temporaryBytes"] is None
            moved.rename(produced.root)
            (produced.root / ".produced-scratch.sqlite").unlink()
            assert (await status())["temporaryBytes"] is None
            assert path.read_bytes() == data
        finally:
            host.close()
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
        library = ServerLibrary(
            AssetVault(tmp_path / "vault"),
            LibraryStore(tmp_path / "library"),
            scratch_root=tmp_path / "scratch",
        )
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
                assert status["temporaryBytes"] == 0
                assert not (tmp_path / "scratch").exists()
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
            assert "vault_accounting_unavailable" in cast(list[str], status["blocked"])
            assert (outside / ("ab" * 32)).read_bytes() == b"untouched external bytes"
            assert not (vault.root / ".produced-assets.sqlite").exists()
        finally:
            store.close()

    asyncio.run(scenario())
