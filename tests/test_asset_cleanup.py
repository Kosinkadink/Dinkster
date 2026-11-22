"""Operator accounting never promotes incomplete reference state into eligibility."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
import sys
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
    AssetError,
    AssetRef,
    AssetResolver,
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
from test_history import make_job, make_record
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


def test_engine_reference_publication_tracks_local_roots_and_invalidates_admission(
    tmp_path: Path,
) -> None:
    from dinkster_library import LibraryClient, LibraryError
    from dinkster_server.cleanup import EngineReferences

    async def scenario() -> None:
        with (
            LibraryClient(tmp_path / "shared") as helper,
            LibraryClient(tmp_path / "shared") as peer,
        ):
            history = HistoryStore(helper, accepted_path=tmp_path / "state" / "accepted.sqlite")
            registry = TypeRegistry()
            register_asset_type(registry)
            registry.register("file", asset_references_complete=True)
            memory = MemoryLRUCache()
            disk = DiskCacheStore(tmp_path / "cache", registry)
            alias = DiskCacheStore(disk.root, registry)
            cache = LayeredCache(memory, disk, alias)
            scratch = tmp_path / "scratch"
            publisher = EngineReferences(helper, "21" * 16, cache, history, registry, scratch)
            other_history = HistoryStore(
                peer, accepted_path=tmp_path / "other-state" / "accepted.sqlite"
            )
            other_memory = MemoryLRUCache()
            other_publisher = EngineReferences(
                peer, "22" * 16, other_memory, other_history, registry, tmp_path / "other-scratch"
            )
            other_digest = digest_bytes(b"another engine")
            other_ref = AssetRef(other_digest, "peer", len(b"another engine"))
            await other_memory.put(
                "peer",
                {
                    "out": Value(
                        "asset<file>",
                        other_digest,
                        ValueMeta(other_ref.to_wire()),
                        EncodedPayload("asset<file>", b"opaque", None),
                    )
                },
            )
            assert (await other_publisher.publish())["complete"] is True
            payloads = [b"in ram", b"on disk", b"old scratch", b"accepted document", b"new root"]
            digests = [digest_bytes(data) for data in payloads]

            def outputs(index: int) -> dict[str, Value]:
                ref = AssetRef(digests[index], "source", len(payloads[index]))
                return {
                    "out": Value(
                        "asset<file>",
                        ref.digest,
                        ValueMeta(ref.to_wire()),
                        EncodedPayload("asset<file>", b"opaque", None),
                    )
                }

            async def reserve(expected: set[str]) -> None:
                reservation = await asyncio.to_thread(peer.call, "references.acquire")
                assert set(cast(list[str], reservation["digests"])) == expected | {other_digest}
                await asyncio.to_thread(peer.call, "references.release", lease=reservation["lease"])

            async def incomplete() -> None:
                with pytest.raises(LibraryError) as blocked:
                    await asyncio.to_thread(peer.call, "references.acquire")
                assert blocked.value.code == "references"

            try:
                await memory.put("ram", outputs(0))
                await disk.put("disk", outputs(1))
                with (
                    closing(hold_produced_job(scratch, "old job")),
                    use_produced_output_job("old job", 1),
                ):
                    produced = produced_output_scratch(
                        {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
                    )
                    assert produced is not None
                    with produced.writer(digests[2]) as writer:
                        writer.write(payloads[2])
                        writer.commit()
                    ProducedScratchState(produced.root).offer([digests[2]])
                    with pytest.raises(AssetError):
                        await publisher.publish()
                await incomplete()
                manifest = await publisher.publish()
                assert manifest["complete"] is True
                assert set(cast(list[str], manifest["digests"])) == set(digests[:3])
                await reserve(set(digests[:3]))

                job = make_job("pending", scope="private", source_document=digests[3])
                history.job_accepted(job)
                await incomplete()
                manifest = await publisher.publish()
                assert manifest["complete"] is False
                assert set(cast(list[str], manifest["digests"])) == set(digests[:4])
                await incomplete()
                job.state = "completed"
                history.job_terminal(job)
                assert (await publisher.publish())["complete"] is True
                # Accepted roots move to shared immutable history without disappearing.
                await reserve(set(digests[:4]))

                for write in (memory.put, memory.promote, disk.put, alias.put):
                    await write("new", outputs(4))
                    await incomplete()
                    assert (await publisher.publish())["complete"] is True
                    await reserve(set(digests))
                memory.clear()
                disk.clear()
                assert (await publisher.publish())["complete"] is True
                await reserve({digests[2], digests[3]})

                await memory.put(
                    "unknown",
                    {
                        "out": Value(
                            "unregistered",
                            "1",
                            ValueMeta(),
                            EncodedPayload("unregistered", b"opaque", None),
                        )
                    },
                )
                with pytest.raises(AssetError):
                    await publisher.publish()
                await incomplete()
                memory.clear()
                # Layered spill also persists the opaque entry; clearing RAM alone
                # must not certify the remaining disk manifest.
                with pytest.raises(AssetError):
                    await publisher.publish()
                disk.clear()
                assert (await publisher.publish())["complete"] is True
                await reserve({digests[2], digests[3]})

                # Losing the helper cannot admit a new, unreported root.
                helper.close()
                for write in (memory.put, memory.promote, disk.put, alias.put):
                    with pytest.raises(LibraryError):
                        await write("refused", outputs(4))
                with pytest.raises(LibraryError):
                    history.job_accepted(make_job("refused", source_document=digests[4]))
                assert await memory.get("refused") is None
                assert await disk.get("refused") is None
                with closing(sqlite3.connect(tmp_path / "state" / "accepted.sqlite")) as db:
                    assert db.execute("SELECT count(*) FROM accepted").fetchone() == (0,)
            finally:
                history.close()
                other_history.close()

    run_scenario(scenario())


def test_engine_cleanup_uses_bound_vaults_and_shared_references(tmp_path: Path) -> None:
    from dinkster_library import LibraryClient, LibraryError
    from dinkster_server.cleanup import EngineReferences

    class LostReplyClient(LibraryClient):
        lose_reply = False

        def call(self, operation: str, **arguments: object) -> dict[str, object]:
            result = super().call(operation, **arguments)
            if self.lose_reply and operation == "references.collect":
                raise LibraryError("transport", "injected lost reply after collection")
            return result

    async def scenario() -> None:
        with (
            LostReplyClient(tmp_path / "shared") as helper,
            LibraryClient(tmp_path / "shared") as peer,
        ):
            vault, preview, foreign = (
                AssetVault(tmp_path / name) for name in ("vault", "preview", "foreign")
            )
            for item in (vault, preview, foreign):
                ProducedAssetInventory.initialize_vault(item.root, existing_empty=True)

            def adopt(target: AssetVault, payload: bytes) -> str:
                digest = digest_bytes(payload)
                inventory = ProducedAssetInventory(target.root)
                inventory.pending(digest, len(payload))
                with target.writer(digest) as writer:
                    writer.write(payload)
                    writer.commit()
                inventory.adopted(digest, len(payload))
                return digest

            cached = adopt(vault, b"cache survives")
            shared = adopt(vault, b"another engine's cache root")
            durable = adopt(vault, b"finished history and offered scratch survive")
            disposable = adopt(vault, b"unreferenced main")
            transient = adopt(preview, b"unreferenced preview")
            outside = adopt(foreign, b"another engine's configured preview")
            registry = TypeRegistry()
            register_asset_type(registry)
            registry.register("file", asset_references_complete=True)
            memory = MemoryLRUCache()
            history = HistoryStore(helper, accepted_path=tmp_path / "state" / "accepted.sqlite")
            other_history = HistoryStore(peer, accepted_path=tmp_path / "other" / "accepted.sqlite")
            scratch = tmp_path / "scratch"
            scratch.mkdir()
            publisher = EngineReferences(
                helper, "51" * 16, memory, history, registry, scratch, vaults=(vault, preview)
            )
            other_memory = MemoryLRUCache()
            other = EngineReferences(
                peer,
                "52" * 16,
                other_memory,
                other_history,
                registry,
                tmp_path / "other-scratch",
            )
            try:
                ref = AssetRef(cached, "cached", len(b"cache survives"))
                await memory.put(
                    "cached",
                    {
                        "out": Value(
                            "asset<file>",
                            cached,
                            ValueMeta(ref.to_wire()),
                            EncodedPayload("asset<file>", b"opaque", None),
                        )
                    },
                )
                history.put(make_record("finished", source_document=durable, asset_digests=()))
                with (
                    closing(hold_produced_job(scratch, "finished")),
                    use_produced_output_job("finished", 1),
                ):
                    produced = produced_output_scratch(
                        {"DINKSTER_PACK_SCRATCH": str(scratch / "groups" / "media")}
                    )
                    assert produced is not None
                    with produced.writer(durable) as writer:
                        writer.write(b"finished history and offered scratch survive")
                        temporary = writer.commit()
                    ProducedScratchState(produced.root).offer([durable])
                temporary_files = tuple(
                    (path, path.stat().st_size) for path in temporary.parent.iterdir()
                )
                refused = await publisher.inspect(collect=True)
                assert not refused["collectionEnabled"]
                assert refused["reclaimedBytes"] == refused["reclaimedFiles"] == 0
                assert vault.resolve(disposable) is not None
                assert preview.resolve(transient) is not None
                assert temporary.exists()
                unconfigured = await other.inspect(collect=True)
                assert not unconfigured["collectionEnabled"]
                assert unconfigured["reclaimedBytes"] == unconfigured["reclaimedFiles"] == 0
                peer_ref = AssetRef(shared, "peer cache", len(b"another engine's cache root"))
                await other_memory.put(
                    "peer-cache",
                    {
                        "out": Value(
                            "asset<file>",
                            shared,
                            ValueMeta(peer_ref.to_wire()),
                            EncodedPayload("asset<file>", b"opaque", None),
                        )
                    },
                )
                await other.publish()
                with pytest.raises(ValueError, match="nonnegative"):
                    await publisher.inspect(collect=True, limit_bytes=-1)
                assert temporary.exists()
                measured = await publisher.inspect()
                assert measured["blocked"] == []
                assert measured["collectionEnabled"]
                assert measured["knownRetainedAssets"] == 3
                assert measured["quarantine"] == {"departedClients": 0, "owners": 0, "bytes": 0}
                assert cast(int, measured["reclaimableBytes"]) >= len(b"unreferenced main") + len(
                    b"unreferenced preview"
                )
                assert temporary.exists()
                bounded = await publisher.inspect(collect=True, limit_bytes=10**6)
                assert bounded["collectionEnabled"]
                assert bounded["reclaimedFiles"] == len(temporary_files)
                assert bounded["reclaimedBytes"] == sum(size for _, size in temporary_files)
                assert all(not path.exists() for path, _ in temporary_files)
                assert not temporary.exists()
                assert ProducedScratchState(produced.root).publications() == ()
                assert vault.resolve(disposable) is not None
                assert preview.resolve(transient) is not None
                cleared = await publisher.inspect(collect=True)
                assert cleared["blocked"] == []
                assert cleared["collectionEnabled"]
                assert cast(int, cleared["reclaimedFiles"]) >= 2
                assert cleared["knownRetainedAssets"] == 3
                assert vault.resolve(shared) is not None
                assert vault.resolve(disposable) is None
                assert preview.resolve(transient) is None
                assert vault.resolve(cached) is not None
                assert vault.resolve(durable) is not None
                assert foreign.resolve(outside) is not None
                assert cleared["vaultBytes"] == sum(
                    len(payload)
                    for payload in (
                        b"cache survives",
                        b"another engine's cache root",
                        b"finished history and offered scratch survive",
                    )
                )
                await memory.put(
                    "unknown",
                    {
                        "out": Value(
                            "unknown", "1", ValueMeta(), EncodedPayload("unknown", b"opaque", None)
                        )
                    },
                )
                blocked = await publisher.inspect(collect=True)
                assert not blocked["collectionEnabled"]
                assert blocked["reclaimableBytes"] is None
                assert blocked["reclaimedBytes"] == blocked["reclaimedFiles"] == 0
                assert vault.resolve(cached) is not None
                assert foreign.resolve(outside) is not None
                memory.clear()
                uncertain = adopt(vault, b"collection completed before its reply was lost")
                helper.lose_reply = True
                unknown = await publisher.inspect(collect=True)
                assert not unknown["collectionEnabled"]
                assert unknown["reclaimableBytes"] is None
                assert unknown["reclaimedBytes"] is unknown["reclaimedFiles"] is None
                assert vault.resolve(uncertain) is None
                assert foreign.resolve(outside) is not None
                reservation = peer.call("references.acquire")
                peer.call("references.release", lease=reservation["lease"])
                helper.lose_reply = False
                assert (await publisher.inspect())["collectionEnabled"]
            finally:
                history.close()
                other_history.close()

    run_scenario(scenario())




@pytest.mark.parametrize("collect", [False, True])
def test_engine_cleanup_cancellation_waits_for_the_helper_reservation(
    tmp_path: Path, collect: bool
) -> None:
    from dinkster_library import LibraryClient, LibraryError
    from dinkster_server.cleanup import EngineReferences

    entered, release = tmp_path / "hashing", tmp_path / "release"
    command = [
        sys.executable,
        "-c",
        f"""
from pathlib import Path
import time
from dinkster_library import produced_collection
from dinkster_library.__main__ import main
original = produced_collection.digest_file
def paused(path, *, progress=None):
    Path({str(entered)!r}).write_text("hashing")
    while not Path({str(release)!r}).exists():
        time.sleep(0.01)
    return original(path, progress=progress)
produced_collection.digest_file = paused
main()
""",
    ]

    async def scenario() -> None:
        with (
            LibraryClient(tmp_path / "shared", command=command) as helper,
            LibraryClient(tmp_path / "shared") as peer,
        ):
            vault = AssetVault(tmp_path / "vault")
            ProducedAssetInventory.initialize_vault(vault.root, existing_empty=True)
            payload = b"only the helper removes these bytes"
            digest = digest_bytes(payload)
            inventory = ProducedAssetInventory(vault.root)
            inventory.pending(digest, len(payload))
            with vault.writer(digest) as writer:
                writer.write(payload)
                writer.commit()
            inventory.adopted(digest, len(payload))
            engine = make_engine()
            memory = cast(MemoryLRUCache, engine.cache)
            registry = engine.registry
            register_asset_type(registry)
            registry.register("file", asset_references_complete=True)
            disk = DiskCacheStore(tmp_path / "cache", registry)
            history = HistoryStore(helper, accepted_path=tmp_path / "accepted.sqlite")
            queue = JobQueue(engine, store=history)
            publisher = EngineReferences(
                helper,
                "53" * 16,
                LayeredCache(memory, disk),
                history,
                registry,
                tmp_path / "scratch",
                vaults=(vault,),
            )
            inspection = asyncio.create_task(publisher.inspect(collect=collect))
            admitted, stalled = threading.Event(), threading.Event()

            def unblock_stalled_submission() -> None:
                if not admitted.wait(2):
                    stalled.set()
                    release.write_text("release stalled helper")

            watchdog = threading.Thread(target=unblock_stalled_submission)
            try:
                for _ in range(1000):
                    if entered.exists():
                        break
                    await asyncio.sleep(0.01)
                assert entered.exists()
                inspection.cancel()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(inspection), timeout=0.1)
                watchdog.start()
                try:
                    with pytest.raises(LibraryError) as refused_job:
                        queue.submit("client", "racing", echo_graph(), ["s"])
                finally:
                    admitted.set()
                assert not stalled.is_set(), "job submission waited for the helper's pipe lock"
                assert refused_job.value.code == "busy"
                with history.hold_engine_references() as (roots, idle):
                    assert roots == frozenset() and idle
                ref = AssetRef(digest, "must not be admitted", len(payload))
                output = {
                    "out": Value(
                        "asset<file>",
                        digest,
                        ValueMeta(ref.to_wire()),
                        EncodedPayload("asset<file>", b"opaque", None),
                    )
                }
                for cache in (memory, disk):
                    with pytest.raises(LibraryError) as refused:
                        await cache.put("racing", output)
                    assert refused.value.code == "busy"
                    assert await cache.get("racing") is None
                assert vault.resolve(digest) is not None
                project = await asyncio.to_thread(
                    peer.call, "projects.register", id="while-collecting", name="Concurrent project"
                )
                assert cast("dict[str, object]", project["project"])["name"] == "Concurrent project"
                assert not release.exists()
                release.write_text("finish hashing")
                with pytest.raises(asyncio.CancelledError):
                    await inspection
                assert (vault.resolve(digest) is None) is collect
                reservation = await asyncio.to_thread(peer.call, "references.acquire")
                assert reservation["digests"] == []
                await asyncio.to_thread(peer.call, "references.release", lease=reservation["lease"])
                assert (
                    helper.call("projects.get", id="while-collecting")["project"]
                    == project["project"]
                )
                await memory.put("after", output)
                await disk.put("after", output)
                assert (await publisher.publish())["digests"] == [digest]
            finally:
                admitted.set()
                release.write_text("cleanup")
                if watchdog.ident is not None:
                    watchdog.join()
                await asyncio.gather(inspection, return_exceptions=True)
                history.close()

    run_scenario(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_engine_reference_publication_holds_admission_through_settlement(
    tmp_path: Path, cancelled: bool
) -> None:
    from dinkster_library import LibraryClient, LibraryError
    from dinkster_server.cleanup import EngineReferences

    entered, release = threading.Event(), threading.Event()

    class PausingClient(LibraryClient):
        paused = False

        def call(self, operation: str, **arguments: object) -> dict[str, object]:
            if (
                self.paused
                and operation == "references.engine.patch"
                and cast("dict[str, object]", arguments["patch"]).get("complete") is True
            ):
                entered.set()
                assert release.wait(10)
            return super().call(operation, **arguments)

    async def scenario() -> None:
        with (
            PausingClient(tmp_path / "shared") as helper,
            LibraryClient(tmp_path / "shared") as peer,
        ):
            history = HistoryStore(helper, accepted_path=tmp_path / "accepted.sqlite")
            memory = MemoryLRUCache()
            registry = TypeRegistry()
            register_asset_type(registry)
            registry.register("file", asset_references_complete=True)
            publisher = EngineReferences(
                helper, "31" * 16, memory, history, registry, tmp_path / "scratch"
            )
            digest = digest_bytes(b"admitted after the snapshot")
            ref = AssetRef(digest, "source", len(b"admitted after the snapshot"))
            helper.paused = True
            publication = asyncio.create_task(publisher.publish())
            admission: asyncio.Task[None] | None = None
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                if cancelled:
                    publication.cancel()
                admission = asyncio.create_task(
                    memory.put(
                        "new",
                        {
                            "out": Value(
                                "asset<file>",
                                digest,
                                ValueMeta(ref.to_wire()),
                                EncodedPayload("asset<file>", b"opaque", None),
                            )
                        },
                    )
                )
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(admission), timeout=0.1)
                assert not publication.done()
                with pytest.raises(LibraryError) as incomplete:
                    await asyncio.to_thread(peer.call, "references.acquire")
                assert incomplete.value.code == "references"
                release.set()
                outcome = (await asyncio.gather(publication, return_exceptions=True))[0]
                if cancelled:
                    assert isinstance(outcome, asyncio.CancelledError)
                else:
                    assert isinstance(outcome, dict) and outcome["complete"] is True
                await admission
                with pytest.raises(LibraryError) as changed:
                    await asyncio.to_thread(peer.call, "references.acquire")
                assert changed.value.code == "references"
                helper.paused = False
                assert (await publisher.publish())["digests"] == [digest]
                reservation = peer.call("references.acquire")
                assert reservation["digests"] == [digest]
                peer.call("references.release", lease=reservation["lease"])
            finally:
                release.set()
                await asyncio.gather(
                    publication, *([admission] if admission else []), return_exceptions=True
                )
                history.close()

    run_scenario(scenario())


def test_engine_reference_publication_lost_reply_cannot_admit_unreported_roots(
    tmp_path: Path,
) -> None:
    from dinkster_library import LibraryClient, LibraryError
    from dinkster_server.cleanup import EngineReferences

    class LostReplyClient(LibraryClient):
        def call(self, operation: str, **arguments: object) -> dict[str, object]:
            result = super().call(operation, **arguments)
            if (
                operation == "references.engine.patch"
                and cast("dict[str, object]", arguments["patch"]).get("complete") is True
            ):
                raise LibraryError("transport", "injected lost reply after commit")
            return result

    async def scenario() -> None:
        with (
            LostReplyClient(tmp_path / "shared") as helper,
            LibraryClient(tmp_path / "shared") as peer,
        ):
            history = HistoryStore(helper, accepted_path=tmp_path / "accepted.sqlite")
            memory = MemoryLRUCache()
            registry = TypeRegistry()
            registry.register("unknown")
            publisher = EngineReferences(
                helper, "41" * 16, memory, history, registry, tmp_path / "scratch"
            )
            try:
                with pytest.raises(LibraryError) as lost:
                    await publisher.publish()
                assert lost.value.code == "transport"
                with pytest.raises(LibraryError) as blocked:
                    await memory.put(
                        "refused",
                        {
                            "out": Value(
                                "unknown",
                                "1",
                                ValueMeta(),
                                EncodedPayload("unknown", b"payload", None),
                            )
                        },
                    )
                assert blocked.value.code == "stale"
                assert await memory.get("refused") is None
                reservation = peer.call("references.acquire")
                assert reservation["digests"] == []
                peer.call("references.release", lease=reservation["lease"])
            finally:
                history.close()

    run_scenario(scenario())


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
                *,
                vault: AssetVault | None = None,
                resolver: AssetResolver | None = None,
            ) -> bool:
                entered.set()
                assert release.wait(10)
                return super().record_result_rendition(
                    scope,
                    token,
                    cache_key,
                    document_digest,
                    digests,
                    vault=vault,
                    resolver=resolver,
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
            response = await client.get(
                values_url("s", rendition="document"), headers={"If-None-Match": "*"}
            )
            assert response.status == 304
            with store.hold_asset_references() as roots:
                assert roots == {digest}
            assert vault.delete(digest)
            response = await client.get(
                values_url("s", rendition="document"), headers={"If-None-Match": "*"}
            )
            assert response.status == 404
            assert not vault.has(digest)
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


def _exit_after_result_publication(root: str, complete: bool | None, attempt: int) -> None:
    path = Path(root)
    store = LibraryStore(path / "library.sqlite")
    history = HistoryStore(path / "history.sqlite")
    job = make_job("published-run", scope="private", attempt=attempt)
    history.job_accepted(job)
    job.started_at = 9.0
    history.job_started(job)
    payloads = (b"published output", b"saved file", b"preview", b"unreferenced bytes")
    digests = [digest_bytes(data) for data in payloads]
    ProducedAssetInventory.initialize_vault(path / "vault")
    vault = AssetVault(path / "vault")
    inventory = ProducedAssetInventory(vault.root)
    for digest, data in zip(digests, payloads, strict=True):
        inventory.pending(digest, len(data))
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        inventory.adopted(digest, len(data))
    output = store.offer_result_assets(
        "private", job.run_id, 1, "output", "image", 1, [digests[0]], complete=True
    )
    artifact = store.offer_result_assets(
        "private",
        job.run_id,
        1,
        "save",
        "",
        1,
        digests[1:3],
        complete=True,
        artifact={
            "nodeId": "save",
            "digest": digests[1],
            "name": "saved.bin",
            "size": 10,
            "mediaType": "application/octet-stream",
            "virtualPath": "output/saved.bin",
            "index": 0,
            "preview": {
                "digest": digests[2],
                "name": "preview.bin",
                "size": 7,
                "mediaType": "application/octet-stream",
                "virtualPath": f"previews/{digests[2][7:]}/preview.bin",
            },
        },
    )
    if complete is not None:
        store.publish_terminal_result(
            "private",
            job.run_id,
            1,
            [output.token, artifact.token] if complete else [output.token],
            state="completed",
            identity={"clientId": job.key.client_id, "jobId": job.key.job_id},
            artifacts_complete=complete,
        )
    os._exit(0)


@pytest.mark.parametrize("complete,attempt", [(None, 1), (False, 1), (True, 2), (True, 1)])
def test_restart_after_result_publication_keeps_known_roots_without_blocking_cleanup(
    tmp_path: Path, complete: bool | None, attempt: int
) -> None:
    known = complete is True and attempt == 1
    process = multiprocessing.get_context("spawn").Process(
        target=_exit_after_result_publication, args=(str(tmp_path), complete, attempt)
    )
    process.start()
    process.join(20)
    assert not process.is_alive()
    assert process.exitcode == 0

    async def scenario() -> None:
        digests = tuple(
            sorted(digest_bytes(data) for data in (b"published output", b"saved file", b"preview"))
        )
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        for _ in range(2):
            history = HistoryStore(tmp_path / "history.sqlite")
            library = ServerLibrary(
                AssetVault(tmp_path / "vault"),
                LibraryStore(tmp_path / "library.sqlite"),
                scratch_root=scratch,
            )
            client = TestClient(
                TestServer(create_app(make_engine, SCHEMAS, library=library, history=history))
            )
            await client.start_server()
            try:
                record = history.get("private", "published-run")
                assert record is not None and record.state == "interrupted"
                assert record.error is not None and record.error["phase"] == "running"
                assert history.get("other", "published-run") is None
                status = await (await client.get("/api/library/storage")).json()
                assert status["collectionEnabled"] is known, status
                candidate = library.vault.resolve(digest_bytes(b"unreferenced bytes"))
                expected_removed = (
                    [candidate, *candidate.parent.glob(candidate.name + ".*")]
                    if candidate is not None
                    else []
                )
                expected_bytes = sum(path.stat().st_size for path in expected_removed)
                response = await client.post("/api/library/storage/clear", json={})
                if known:
                    assert record.asset_digests == digests
                    with history.hold_asset_references() as roots:
                        assert roots == set(digests)
                    assert response.status == 200, await response.text()
                    collected = await response.json()
                    assert collected["reclaimedFiles"] == len(expected_removed)
                    assert collected["reclaimedBytes"] == expected_bytes
                    assert all(not path.exists() for path in expected_removed)
                    assert set(library.vault.digests()) == set(digests)
                    assert library.vault.resolve(digest_bytes(b"unreferenced bytes")) is None
                    for data in (b"published output", b"saved file", b"preview"):
                        path = library.vault.resolve(digest_bytes(data))
                        assert path is not None and path.read_bytes() == data
                else:
                    assert "reference_state_unavailable" in status["blocked"]
                    assert response.status == 409
                    assert library.vault.resolve(digest_bytes(b"unreferenced bytes")) is not None
            finally:
                await client.close()

    run_scenario(scenario())


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
            recovery = status["lastPublicationRecovery"]
            assert {
                key: recovery[key] for key in ("verifiedPublications", "blockedInspections")
            } == {
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
def test_startup_recovers_offered_publications_without_certifying_historical_clients(
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
        # Read-only WAL inspection creates SQLite sidecars in this writable
        # directory. Include them in the unchanged recovery-state snapshot.
        assert len(ProducedScratchState(source.root).publications()) == 2
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
                remaining = {
                    p.relative_to(source.root): p.read_bytes()
                    for p in source.root.rglob("*")
                    if p.is_file()
                }
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




def test_preview_vaults_share_retention_budget_and_recover_without_original_copy(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        previews = (AssetVault(tmp_path / "previews"), AssetVault(tmp_path / "previous"))
        store = LibraryStore(tmp_path / "library")
        history = HistoryStore(tmp_path / "history")
        scratch = tmp_path / "scratch"
        cache = MemoryLRUCache()
        library = ServerLibrary(vault, store, preview_vaults=previews, scratch_root=scratch)
        app = create_app(make_engine, SCHEMAS, library=library, history=history)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            payloads = (b"keep", b"discard preview", b"older preview data")
            digests = tuple(digest_bytes(data) for data in payloads)
            for target, data, digest in zip((vault, *previews), payloads, digests, strict=True):
                inventory = ProducedAssetInventory(target.root)
                inventory.pending(digest, len(data))
                with target.writer(digest) as writer:
                    writer.write(data)
                    writer.commit()
                    inventory.adopted(digest, len(data))
            store.create("local", "original", digests[0], "application/octet-stream")
            with closing(hold_produced_job(scratch, "saved preview")):
                with use_produced_output_job("saved preview", 1):
                    source = produced_output_scratch(
                        {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
                    )
                    assert source is not None
                    with source.writer(digests[1]) as writer:
                        writer.write(payloads[1])
                        staged = writer.commit()
                    ProducedScratchState(source.root).offer([digests[1]])
            from dinkster_server.cleanup import recover_produced_publications

            recovered = recover_produced_publications(
                vault, scratch, store=store, preview_vaults=previews
            )
            assert recovered["verifiedPublications"] == 1
            assert recovered["blockedInspections"] == 0
            assert not staged.exists()
            assert vault.resolve(digests[1]) is None
            assert previews[0].resolve(digests[1]) is not None
            preview_sizes = []
            for target, digest in zip(previews, digests[1:], strict=True):
                path = target.resolve(digest)
                assert path is not None
                preview_sizes.append(
                    path.stat().st_size
                    + sum(sidecar.stat().st_size for sidecar in path.parent.glob(path.name + ".*"))
                )
            before = await inspect_cleanup(
                vault, store, history, cache, scratch_root=scratch, preview_vaults=previews
            )
            assert before["collectionEnabled"]
            assert before["vaultBytes"] == 37
            assert before["adoptedBytes"] == 37
            assert before["reclaimableBytes"] == sum(preview_sizes)
            held = previews[1].resolve(digests[2])
            assert held is not None
            with open_verified(held, digests[2]):
                blocked = await inspect_cleanup(
                    vault,
                    store,
                    history,
                    cache,
                    scratch_root=scratch,
                    preview_vaults=previews,
                    collect=True,
                )
                assert not blocked["collectionEnabled"]
                assert "vault_in_use" in cast(list[str], blocked["blocked"])
                assert previews[0].resolve(digests[1]) is not None
            limited = await inspect_cleanup(
                vault,
                store,
                history,
                cache,
                scratch_root=scratch,
                preview_vaults=previews,
                collect=True,
                limit_bytes=22,
            )
            assert limited["collectionEnabled"]
            assert limited["reclaimedBytes"] == preview_sizes[0]
            assert limited["vaultBytes"] == 22
            assert previews[0].resolve(digests[1]) is None
            assert previews[1].resolve(digests[2]) is not None
            cleared = await inspect_cleanup(
                vault,
                store,
                history,
                cache,
                scratch_root=scratch,
                preview_vaults=previews,
                collect=True,
            )
            assert cleared["reclaimedBytes"] == preview_sizes[1]
            assert cleared["vaultBytes"] == 4
            original = vault.resolve(digests[0])
            assert original is not None and original.read_bytes() == b"keep"
        finally:
            await client.close()

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
                "quarantine": {"departedClients": 0, "owners": 0, "bytes": 0},
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


@pytest.mark.parametrize("helper_storage", [False, True])
@pytest.mark.parametrize("limit", [None, 37, 2**53 - 1])
def test_cleanup_policy_persists_without_authorizing_collection(
    tmp_path: Path, limit: int | None, helper_storage: bool
) -> None:
    from dinkster_library import LibraryClient
    from dinkster_server.cleanup import CleanupPolicy

    async def scenario() -> None:
        data = b"kept untracked vault bytes"
        digest = digest_bytes(data)
        for cycle in range(2):
            helper = LibraryClient(tmp_path / "shared") if helper_storage else None
            library = ServerLibrary(
                AssetVault(tmp_path / "vault"),
                LibraryStore(tmp_path / "library"),
                cleanup_policy=CleanupPolicy(helper) if helper is not None else None,
            )
            with library.vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
            ProducedAssetInventory.initialize_vault(library.vault.root, existing_empty=True)
            assert not (library.vault.root / ".produced-assets.sqlite").exists()
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
                    result = await response.json()
                    assert result["collectionEnabled"] is False
                    assert "owner_completeness_unknown" in result["blocked"]
                    assert result["reclaimedBytes"] == result["reclaimedFiles"] == 0
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
                    if helper is not None:
                        status = await (await client.get(url)).json()
                        assert status["policyAvailable"] is True
                        assert status["limitBytes"] == limit
                        helper.close()
                    status = await (await client.get(url)).json()
                    assert status["policyAvailable"] is False
                    assert status["limitBytes"] is None
                    response = await client.put(url + "/policy", json={"limitBytes": None})
                    assert response.status == 503
            finally:
                await client.close()
                if helper is not None:
                    if cycle == 0:
                        assert "cleanup.policy" in cast(
                            "list[str]", helper.call("hello")["capabilities"]
                        )
                    helper.close()

    run_scenario(scenario())




def test_helper_cleanup_policy_refuses_stale_edits_and_legacy_collection(tmp_path: Path) -> None:
    from dinkster_library import LibraryClient, LibraryError
    from dinkster_server.cleanup import POLICY_KEY, POLICY_WAKE_KEY, CleanupPolicy

    async def scenario() -> None:
        root = tmp_path / "shared"
        with LibraryClient(root) as helper, LibraryClient(root) as peer:
            policy = CleanupPolicy(helper)
            other = CleanupPolicy(peer)
            other.set_cleanup_limit(91)
            vault = AssetVault(tmp_path / "vault")
            ProducedAssetInventory.initialize_vault(vault.root, existing_empty=True)
            store = LibraryStore(tmp_path / "legacy.sqlite")
            history = HistoryStore(tmp_path / "history.sqlite")
            scratch = tmp_path / "scratch"
            scratch.mkdir()
            payload = b"unreferenced produced bytes must not be collected by an engine"
            digest = digest_bytes(payload)
            inventory = ProducedAssetInventory(vault.root)
            inventory.pending(digest, len(payload))
            with vault.writer(digest) as writer:
                writer.write(payload)
                writer.commit()
            inventory.adopted(digest, len(payload))
            eligible = await inspect_cleanup(
                vault, store, history, MemoryLRUCache(), scratch_root=scratch
            )
            assert eligible["collectionEnabled"]
            shard = vault.root / digest.split(":")[1][:2]
            assert eligible["reclaimableBytes"] == sum(
                path.stat().st_size for path in shard.iterdir()
            )
            with (
                closing(hold_produced_job(scratch, "retained scratch")),
                use_produced_output_job("retained scratch", 1),
            ):
                temporary = produced_output_scratch(
                    {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
                )
                assert temporary is not None
                with temporary.writer(digest) as writer:
                    writer.write(payload)
                    scratch_file = writer.commit()
            library = ServerLibrary(vault, store, cleanup_policy=policy, scratch_root=scratch)
            app = create_app(make_engine, SCHEMAS, library=library, history=history)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                url = "/api/library/storage"
                assert scratch_file.read_bytes() == payload
                response = await client.put(url + "/policy", json={"limitBytes": 1})
                assert response.status == 409
                assert other.cleanup_limit() == 91
                status = await (await client.get(url)).json()
                assert status["limitBytes"] == 91
                assert status["blocked"] == ["helper_collection_unavailable"]
                assert (await client.put(url + "/policy", json={"limitBytes": 1})).status == 200
                for _ in range(200):
                    if app[POLICY_KEY]:
                        break
                    await asyncio.sleep(0.01)
                assert app[POLICY_KEY]["blocked"] == ["helper_collection_unavailable"]
                assert not app[POLICY_KEY]["collectionEnabled"]
                response = await client.post(url + "/clear", json={})
                assert response.status == 409
                result = await response.json()
                assert result["blocked"] == ["helper_collection_unavailable"]
                assert result["reclaimedBytes"] == result["reclaimedFiles"] == 0
                assert not result["collectionEnabled"]
                assert vault.resolve(digest) is not None
                assert scratch_file.read_bytes() == payload
                assert (await client.get(f"/api/assets/{digest}")).status == 200
                assert store.cleanup_limit() == 21474836480
                assert other.cleanup_limit() == 1
                other.set_cleanup_limit(71)
                app[POLICY_KEY].clear()
                app[POLICY_WAKE_KEY].set()
                for _ in range(200):
                    if app[POLICY_KEY]:
                        break
                    await asyncio.sleep(0.01)
                assert app[POLICY_KEY]["limitBytes"] == 71
                assert (await client.put(url + "/policy", json={"limitBytes": 23})).status == 409
                assert other.cleanup_limit() == 71
                (root / "library.sqlite").write_bytes(b"unmigrated")
                assert (await client.put(url + "/policy", json={"limitBytes": 2})).status == 503
                status = await (await client.get(url)).json()
                assert not status["policyAvailable"]
                assert status["limitBytes"] is None
                with pytest.raises(LibraryError) as refused:
                    CleanupPolicy(peer)
                assert refused.value.code == "migration"
            finally:
                await client.close()
                history.close()
                store.close()
            assert "cleanup.policy" in cast("list[str]", helper.call("hello")["capabilities"])

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
