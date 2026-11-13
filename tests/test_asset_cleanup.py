"""Operator accounting never promotes incomplete reference state into eligibility."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import closing, contextmanager
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetRef,
    AssetVault,
    LibraryStore,
    digest_bytes,
    open_verified,
    register_asset_type,
)
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
from dinkster_image_document import IMAGE_DOCUMENT_MEDIA_TYPE, ImageDocument
from dinkster_image_document.document import empty_document
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
from dinkster_server.queue import JobState
from dinkster_values import (
    EncodedPayload,
    Rendition,
    TypeRegistry,
    Value,
    ValueMeta,
)
from test_history import make_record
from test_image_document_rendering import _png, _resource
from test_server import (
    SCHEMAS,
    Sleeper,
    echo_graph,
    finish_job,
    make_engine,
    reset_sleeper,
    values_url,
    wait_for_state,
)

from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip


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

    run_scenario(scenario())


@pytest.mark.parametrize("failure", ["", "missing", "corrupt", "size"])
def test_native_saved_document_retains_all_resources_before_catalog_acknowledgment(
    tmp_path: Path, failure: str
) -> None:
    from dinkster_assets import AssetError, AssetWriter, MountDef, MountTable
    from dinkster_engine import RunResult
    from dinkster_graph import Graph
    from dinkster_protocol import AssetFolderContext, SavedArtifact
    from dinkster_server.library import recover_asset_observations
    from dinkster_server.queue import JobKey

    output = tmp_path / "output"
    output.mkdir()
    mounts = MountTable()
    mounts.add(MountDef("out", output, "readwrite"))
    mounts.scan("out")
    vault = AssetVault(tmp_path / "vault")
    data = _png([[(73, 21, 190, 255)]])
    unused = _png([[(21, 73, 190, 255)]])
    for body in (data, unused):
        with vault.writer(digest_bytes(body)) as writer:
            writer.write(body)
            writer.commit()
    record = empty_document(1, 1).to_record()
    record["allocation"]["nextOrdinal"] = 2
    record["resources"] = {
        "r0": _resource("r0", data, 1, 1),
        "r1": _resource("r1", unused, 1, 1),
    }
    if failure == "size":
        record["resources"]["r1"]["byteSize"] = len(unused) + 1
    document = ImageDocument.from_record(record)
    saved = AssetWriter(mounts).save_bytes(
        {"mount": "out", "prefix": "kept"},
        document.data,
        suffix=".json",
        media_type=IMAGE_DOCUMENT_MEDIA_TYPE,
    )
    resource = vault.resolve(digest_bytes(unused))
    assert resource is not None
    if failure == "missing":
        resource.unlink()
    elif failure == "corrupt":
        resource.write_bytes(b"corrupt")
    database = tmp_path / "library.sqlite"
    store = LibraryStore(database)
    library = ServerLibrary(
        vault,
        store,
        upload_writer=AssetWriter(mounts),
        local_resolver=mounts,
        project_folders_for=lambda _project: AssetFolderContext(("out",), "out"),
    )
    job = Job(JobKey("client", "save"), Graph(nodes={}), (), 0, "saved", "fingerprint")
    artifact = SavedArtifact(
        "save", saved.digest, saved.name, saved.size, saved.media_type, saved.virtual_path
    )
    result = RunResult("saved", {}, (), (), (), artifacts=(artifact,))
    try:
        if failure:
            with pytest.raises(AssetError):
                library.record_saved_outputs(job, result)
            assert store.query_assets("local") == []
            assert store.get_dependencies(saved.digest) is None
            assert list(vault.root.glob(".upload-observations/*.json"))
            if failure == "size":
                return
            resource.write_bytes(unused)
            recover_asset_observations(library)
        else:
            assert library.record_saved_outputs(job, result) == result
        assert store.query_assets("local")[0][0].asset.digest == saved.digest
        assert not list(vault.root.glob(".upload-observations/*.json"))
    finally:
        store.close()
    with closing(LibraryStore(database)) as restarted:
        with restarted.hold_asset_references() as roots:
            assert roots == {saved.digest, digest_bytes(data), digest_bytes(unused)}
        assert restarted.media_grant("local", digest_bytes(data), "media/image") is None


@pytest.mark.parametrize("cancelled", [False, True])
def test_document_rendition_commits_source_roots_before_response_or_cancellation(
    tmp_path: Path, cancelled: bool
) -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        data = _png([[(73, 21, 190, 255)]])
        digest = digest_bytes(data)

        class DelayedStore(LibraryStore):
            def record_result_rendition(
                self,
                scope: str,
                token: str,
                cache_key: str,
                document_digest: str,
                digests: list[str],
            ) -> bool:
                entered.set()
                assert release.wait(10)
                return super().record_result_rendition(
                    scope, token, cache_key, document_digest, digests
                )

        vault = AssetVault(tmp_path / "vault")
        store = DelayedStore(tmp_path / "library")
        library = ServerLibrary(vault, store)
        record = empty_document(1, 1).to_record()
        # Unused resources still belong to the returned document.
        record["allocation"]["nextOrdinal"] = 1
        record["resources"] = {"r0": _resource("r0", data, 1, 1)}
        document = ImageDocument.from_record(record).data
        record["lineage"] = "other rendition parameters"
        other_document = ImageDocument.from_record(record).data
        handlers: list[asyncio.Task[None]] = []

        async def mime(_metadata: Mapping[str, object]) -> str:
            return IMAGE_DOCUMENT_MEDIA_TYPE

        async def resolve(
            _metadata: Mapping[str, object], parameters: Mapping[str, str]
        ) -> tuple[str, Mapping[str, str]]:
            return IMAGE_DOCUMENT_MEDIA_TYPE, parameters

        async def render(_value: Value, parameters: Mapping[str, str]) -> Rendition:
            inventory = ProducedAssetInventory(vault.root)
            inventory.pending(digest, len(data))
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
            inventory.adopted(digest, len(data))
            return Rendition(
                "document",
                IMAGE_DOCUMENT_MEDIA_TYPE,
                other_document if parameters["variant"] == "two" else document,
            )

        def factory(on_event: EventListener) -> Engine:
            engine = make_engine(on_event)
            engine.registry.register_relayed_rendition(
                "STRING",
                "document",
                mime=IMAGE_DOCUMENT_MEDIA_TYPE,
                default=False,
                version="1",
                parameters=("variant",),
                defaults={"variant": "one"},
                limits=None,
                owner="worker",
                resolve_mime=mime,
                resolve=resolve,
                render=render,
            )
            return engine

        @web.middleware
        async def observe(
            request: web.Request,
            handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
        ) -> web.StreamResponse:
            if request.path == "/api/values":
                assert request.task is not None
                handlers.append(request.task)
            return await handler(request)

        app = create_app(factory, SCHEMAS, library=library)
        app.middlewares.append(observe)
        client = TestClient(TestServer(app))
        await client.start_server()
        pending: asyncio.Future[ClientResponse] | None = None
        try:
            status = await finish_job(client, echo_graph(), ["s"])
            source = status["resultOffers"]["s"]["out"]
            assert source["digests"] == [] and source["complete"]
            pending = asyncio.ensure_future(
                client.get(values_url("s", rendition="document"), headers={"If-None-Match": "*"})
            )
            assert await asyncio.to_thread(entered.wait, 10)
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert not inactive
            if cancelled:
                for _ in range(2):
                    handlers[-1].cancel()
                    await asyncio.sleep(0)
                assert not handlers[-1].done()
                with claim_inactive_produced_vault(vault.root) as inactive:
                    assert not inactive
            release.set()
            if cancelled:
                result = await asyncio.gather(handlers[-1], return_exceptions=True)
                assert isinstance(result[0], asyncio.CancelledError)
            else:
                assert (await pending).status == 304
            with store.hold_asset_references() as roots:
                assert roots == {digest}
            original = store.result_asset_offer("local", source["token"])
            assert original is not None and original.to_wire() == source
            response = await client.get(values_url("s", rendition="document"))
            assert response.status == 200 and await response.read() == document
            response = await client.get(values_url("s", rendition="document", variant="two"))
            assert response.status == 200 and await response.read() == other_document
            asset = await client.get(f"/api/assets/{digest}")
            assert asset.status == 200 and await asset.read() == data
            assert store.claim_result_assets("local", source["token"], "shown-result", "view")
            store.release_assets("local", "shown-result", "view", revision=1)
            response = await client.get(
                values_url("s", rendition="document"), headers={"If-None-Match": "*"}
            )
            assert response.status == 404
            with store.hold_asset_references() as roots:
                assert roots == set()
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await client.close()

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

        async def settle(job: Job, terminal_state: JobState) -> None:
            assert terminal_state == "completed"
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

    run_scenario(scenario())


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_terminal_jobs_recover_only_their_offered_publications(
    tmp_path: Path, outcome: str, corrupt: bool, collect: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dinkster_server.cleanup.FRONTEND_RETENTION_COMPLETE", collect)

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
            recovery = status["lastPublicationRecovery"]
            assert {
                key: recovery[key] for key in ("verifiedPublications", "blockedInspections")
            } == {
                "verifiedPublications": 0 if corrupt else 1,
                "blockedInspections": 1 if corrupt else 0,
            }
            if collect:
                assert recovery["reclaimedFiles"] == (0 if corrupt else 2)
                assert (recovery["reclaimedBytes"] > len(data)) == (not corrupt)
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert inactive
            response = await client.get(f"/api/assets/{digest}")
            if corrupt:
                assert response.status == 404
                assert sources[0].read_bytes() == b"x" * len(data)
            else:
                assert response.status == 200 and await response.read() == data
                if collect:
                    assert not sources[0].exists()
                else:
                    assert sources[0].read_bytes() == data
            assert other_path.read_bytes() == unrelated
            assert vault.resolve(unrelated_digest) is None
            assert not status["collectionEnabled"]
        finally:
            other.close()
            await client.close()

    run_scenario(scenario())


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

        run_scenario(scenario())
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(15)


@pytest.mark.parametrize("destination", ["absent", "pending", "untracked"])
@pytest.mark.parametrize("collect", [False, True])
def test_startup_recovers_offered_publications_with_gated_scratch_removal(
    tmp_path: Path, destination: str, collect: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dinkster_server.cleanup.FRONTEND_RETENTION_COMPLETE", collect)

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
        if not collect:
            # Read-only WAL inspection creates SQLite sidecars in this writable
            # directory. Include them in the unchanged recovery-state snapshot.
            assert len(ProducedScratchState(source.root).publications()) == 2
        before = {
            p.relative_to(source.root): p.read_bytes()
            for p in source.root.rglob("*")
            if p.is_file()
        }
        timestamp: int | None = None
        for cycle in range(2):
            library = ServerLibrary(vault, LibraryStore(tmp_path / "library"), scratch_root=scratch)
            client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
            await client.start_server()
            try:
                response = await client.get("/api/library/storage")
                assert response.status == 200
                status = await response.json()
                expected = {
                    "verifiedPublications": 0 if collect and cycle else 1,
                    "blockedInspections": 0,
                }
                if collect:
                    removed = [
                        value
                        for path, value in before.items()
                        if path.name != ".produced-scratch.sqlite"
                    ]
                    expected.update(
                        reclaimedBytes=sum(map(len, removed)) if cycle == 0 else 0,
                        reclaimedFiles=len(removed) if cycle == 0 else 0,
                    )
                assert status["lastPublicationRecovery"] == expected
                assert not status["collectionEnabled"]
                assert (await client.get(f"/api/assets/{digest}")).status == 200
                path = vault.resolve(digest)
                assert path is not None and path.read_bytes() == data
                assert vault.resolve(digest_bytes(unused)) is None
                remaining = {
                    p.relative_to(source.root): p.read_bytes()
                    for p in source.root.rglob("*")
                    if p.is_file()
                }
                if collect:
                    assert set(remaining) == {
                        Path(".produced-scratch.sqlite" + suffix) for suffix in ("", "-wal", "-shm")
                    }
                    state = ProducedScratchState(source.root)
                    assert state.owner() == ("interrupted job", 3)
                    assert state.publications() == ()
                else:
                    assert remaining == before
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

    run_scenario(scenario())


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
            symlink_or_skip(path, outside)
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

    run_scenario(scenario())


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
        registry = TypeRegistry()
        register_asset_type(registry)
        registry.register("file", asset_references_complete=True)
        memory = MemoryLRUCache()
        disk = DiskCacheStore(tmp_path / "cache", registry)
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
        cache = LayeredCache(memory, disk, DiskCacheStore(disk.root / ".." / "cache", registry))
        store.create("private", "kept", digests[2], "application/octet-stream")
        history.put(make_record("run", scope="another", asset_digests=(digests[3],)))
        store.put_dependencies(digests[0], [{"digest": digests[4]}])
        store.put_dependencies(digest_bytes(b"orphan"), [{"digest": digest_bytes(b"not a root")}])
        try:
            status = await inspect_cleanup(vault, store, history, cache, registry=registry)
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
                reading = await inspect_cleanup(vault, store, history, cache, registry=registry)
                assert reading["vaultOwnershipAvailable"] is False
                assert "vault_in_use" in cast(list[str], reading["blocked"])
            store.retain_assets("unrelated", "preview", "unknown", [], revision=0)
            blocked = await inspect_cleanup(vault, store, history, cache, registry=registry)
            assert blocked["knownRetainedAssets"] is None
            assert "reference_state_unavailable" in cast(list[str], blocked["blocked"])
            assert not blocked["collectionEnabled"]
            for digest, data in zip(digests, payloads, strict=True):
                path = vault.resolve(digest)
                assert path is not None and path.read_bytes() == data
        finally:
            history.close()
            store.close()

    run_scenario(scenario())


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
            symlink_or_skip(escape, outside)
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

    run_scenario(scenario())


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
            for method, suffix, body in (
                (client.put, "/policy", {"limitBytes": None}),
                (client.post, "/clear", {}),
            ):
                assert (await method(url + suffix, json=body)).status == 401
                response = await method(url + suffix, json=body, headers=headers)
                expected = 200 if suffix == "/policy" else 409
                assert response.status == (expected if kind == "operator" else 403)
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

    run_scenario(scenario())


@pytest.mark.parametrize("limit", [None, 37, 2**53 - 1])
def test_cleanup_policy_persists_without_authorizing_collection(
    tmp_path: Path, limit: int | None
) -> None:
    async def scenario() -> None:
        data = b"kept untracked vault bytes"
        digest = digest_bytes(data)
        for cycle in range(2):
            library = ServerLibrary(
                AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library")
            )
            with library.vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
            if cycle == 0:
                library.store.retain_assets(
                    "local", "document", "empty-complete", [], revision=0, complete=True
                )
            client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
            await client.start_server()
            try:
                url = "/api/library/storage"
                status = await (await client.get(url)).json()
                assert status["policyAvailable"]
                assert status["limitBytes"] == (21474836480 if cycle == 0 else limit)
                response = await client.put(url + "/policy", json={"limitBytes": limit})
                assert response.status == 200
                assert await response.json() == {"limitBytes": limit, "collectionEnabled": False}
                for invalid in (True, False, 0, -1, 1.5, "37", 2**53):
                    response = await client.put(url + "/policy", json={"limitBytes": invalid})
                    assert response.status == 400
                for body in ({}, {"limitBytes": None, "path": "/"}):
                    assert (await client.put(url + "/policy", json=body)).status == 400
                assert (
                    await client.put(url + "/policy?path=/", json={"limitBytes": None})
                ).status == 400
                for _ in range(2):
                    response = await client.post(url + "/clear", json={})
                    assert response.status == 409
                    assert await response.json() == {
                        "collectionEnabled": False,
                        "blocked": ["owner_completeness_unknown"],
                        "reclaimedBytes": 0,
                        "reclaimedFiles": 0,
                    }
                    path = library.vault.resolve(digest)
                    assert path is not None and path.read_bytes() == data
                assert (await client.post(url + "/clear", json={"path": "/"})).status == 400
                assert (await client.post(url + "/clear?path=/", json={})).status == 400
                status = await (await client.get(url)).json()
                assert status["limitBytes"] == limit
                assert not status["collectionEnabled"]
                if cycle == 1:
                    with closing(sqlite3.connect(tmp_path / "library")) as db, db:
                        db.execute("DELETE FROM asset_cleanup_policy")
                    status = await (await client.get(url)).json()
                    assert status["policyAvailable"] is False
                    assert status["limitBytes"] is None
                    response = await client.put(url + "/policy", json={"limitBytes": None})
                    assert response.status == 503
            finally:
                await client.close()

    run_scenario(scenario())


def test_storage_accounting_refuses_escaping_vault_shard(tmp_path: Path) -> None:
    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        store = LibraryStore(tmp_path / "library")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / ("ab" * 32)).write_bytes(b"untouched external bytes")
        symlink_or_skip(vault.root / "ab", outside, target_is_directory=True)
        try:
            status = await inspect_cleanup(vault, store, None, MemoryLRUCache())
            assert status["vaultBytes"] is None
            assert "vault_accounting_unavailable" in cast(list[str], status["blocked"])
            assert (outside / ("ab" * 32)).read_bytes() == b"untouched external bytes"
            assert not (vault.root / ".produced-assets.sqlite").exists()
        finally:
            store.close()

    run_scenario(scenario())
