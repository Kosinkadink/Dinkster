"""The persistentCas transport: bulk boundary values land once in a
persistent store on each side and every later crossing is a digest
reference. Negotiated in the hellos, so either side without a store falls
back to the conversation-scoped cas transport unchanged. Accounting is
observable: EdgeCost.network_bytes counts exactly the payload bytes that
streamed, zero on a store hit - across fresh conversations, reconnects,
and daemon restarts. Failures are conservative: a truncated or corrupt
blob rolls back to nothing and the failure names the digest and the edge.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import pytest
from dinkster_assets import digest_bytes
from dinkster_caches import BudgetedDiskCAS
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_workers import BoundaryDiagnostic, BoundaryError, EdgeCost
from dinkster_workers import blobs as blobs_module
from dinkster_workers.blobs import BlobTransfer
from test_isolated import DEV_MANIFEST, core_registry
from test_remote import big_image_graph, make_engine, remote_worker, start_service, stop_service

BIG = 256 * 1024


def big_edges(diagnostics: Sequence[BoundaryDiagnostic]) -> list[EdgeCost]:
    return [
        edge
        for diag in diagnostics
        for edge in (*diag.inputs, *diag.outputs)
        if edge.size_bytes >= BIG
    ]


def store_blob_files(root: Path) -> list[Path]:
    return [
        blob
        for shard in root.iterdir()
        if shard.is_dir() and len(shard.name) == 2
        for blob in shard.iterdir()
        if blob.is_file()
    ]


def test_value_crosses_once_then_reference_only(tmp_path: Path) -> None:
    """First run: every big value streams exactly once (its first crossing),
    and the accounting matches the bytes that physically landed. Second run
    over a fresh engine: same values, zero payload bytes moved - every big
    edge is a persistent-store hit."""

    async def scenario() -> None:
        daemon_store = tmp_path / "daemon-store"
        engine_store = BudgetedDiskCAS(tmp_path / "engine-store")
        proc, host, port = await start_service(
            DEV_MANIFEST, tmp_path, "--value-store", str(daemon_store)
        )
        diagnostics: list[BoundaryDiagnostic] = []
        try:
            registry = core_registry()
            worker = remote_worker(
                host,
                port,
                registry,
                on_diagnostic=diagnostics.append,
                value_store=engine_store,
            )
            await worker.start()
            try:
                first = await make_engine(registry, worker).run(big_image_graph(), ["s"])
                first_prints = {o: v.fingerprint for o, v in first.outputs["s"].items()}
                first_big = big_edges(diagnostics)
                assert first_big, "the gradient must be big enough to matter"
                assert all(edge.transport == "persistentCas" for edge in first_big)
                # Outputs stream on their first daemon-to-engine crossing;
                # inputs reference bytes the daemon itself produced, so they
                # never move.
                for diag in diagnostics:
                    for edge in diag.outputs:
                        if edge.size_bytes >= BIG:
                            assert edge.network_bytes == edge.size_bytes
                            assert edge.transfer_ms > 0.0
                    for edge in diag.inputs:
                        if edge.size_bytes >= BIG:
                            assert edge.network_bytes == 0
                            assert edge.transfer_ms == 0.0
                # The claimed movement equals what physically landed.
                moved = sum(edge.network_bytes for edge in first_big)
                assert moved == engine_store.total_bytes()
                assert moved == BudgetedDiskCAS(daemon_store).total_bytes()

                diagnostics.clear()
                second = await make_engine(registry, worker).run(big_image_graph(), ["s"])
                assert {o: v.fingerprint for o, v in second.outputs["s"].items()} == first_prints
                second_big = big_edges(diagnostics)
                assert second_big
                assert all(edge.transport == "persistentCas" for edge in second_big)
                assert all(edge.network_bytes == 0 for edge in second_big)
                assert all(edge.transfer_ms == 0.0 for edge in second_big)
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_reconnect_after_socket_death_reuses_stores(tmp_path: Path) -> None:
    """The daemon's store outlives the conversation: after the socket dies
    mid-session, a new connection re-runs the graph without a single
    payload byte crossing again."""

    async def scenario() -> None:
        daemon_store = tmp_path / "daemon-store"
        engine_store = BudgetedDiskCAS(tmp_path / "engine-store")
        proc, host, port = await start_service(
            DEV_MANIFEST, tmp_path, "--value-store", str(daemon_store)
        )
        diagnostics: list[BoundaryDiagnostic] = []
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry, value_store=engine_store)
            await worker.start()
            try:
                first = await make_engine(registry, worker).run(big_image_graph(), ["s"])
                first_prints = {o: v.fingerprint for o, v in first.outputs["s"].items()}
                writer = worker._session._writer
                assert writer is not None
                writer.transport.abort()  # socket death, not a goodbye
            finally:
                await worker.close()

            # The daemon frees its one-conversation slot when it notices the
            # dead socket - eventually, not instantly. Poll until it does.
            deadline = asyncio.get_running_loop().time() + 10.0
            while True:
                reconnected = remote_worker(
                    host,
                    port,
                    registry,
                    on_diagnostic=diagnostics.append,
                    value_store=engine_store,
                )
                try:
                    await reconnected.start()
                    break
                except RuntimeError as exc:
                    if "busy" not in str(exc) or asyncio.get_running_loop().time() > deadline:
                        raise
                    await reconnected.close()
                    await asyncio.sleep(0.05)
            try:
                again = await make_engine(registry, reconnected).run(big_image_graph(), ["s"])
                assert {o: v.fingerprint for o, v in again.outputs["s"].items()} == first_prints
                moved_edges = big_edges(diagnostics)
                assert moved_edges
                assert all(edge.transport == "persistentCas" for edge in moved_edges)
                assert all(edge.network_bytes == 0 for edge in moved_edges)
            finally:
                await reconnected.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_unnegotiated_peers_fall_back_to_conversation_cas(tmp_path: Path) -> None:
    """A store on only one side never produces a persistentCas descriptor:
    an old daemon (no --value-store) serves a store-holding client via the
    conversation cas transport, and a store-holding daemon does the same
    for a storeless client."""

    async def scenario() -> None:
        registry = core_registry()

        async def run_once(
            daemon_args: tuple[str, ...], **worker_kwargs: object
        ) -> list[BoundaryDiagnostic]:
            proc, host, port = await start_service(DEV_MANIFEST, tmp_path, *daemon_args)
            diagnostics: list[BoundaryDiagnostic] = []
            try:
                worker = remote_worker(
                    host, port, registry, on_diagnostic=diagnostics.append, **worker_kwargs
                )
                await worker.start()
                try:
                    result = await make_engine(registry, worker).run(big_image_graph(), ["s"])
                    assert result.outputs["s"]
                finally:
                    await worker.close()
            finally:
                await stop_service(proc)
            return diagnostics

        old_daemon = await run_once((), value_store=BudgetedDiskCAS(tmp_path / "engine-store"))
        storeless_client = await run_once(
            ("--value-store", str(tmp_path / "daemon-store")),
        )
        for diagnostics in (old_daemon, storeless_client):
            edges = big_edges(diagnostics)
            assert edges
            assert all(edge.transport == "cas" for edge in edges)
            assert all(edge.network_bytes == 0 for edge in edges)

    asyncio.run(scenario())


def test_corrupt_daemon_blob_fails_loudly_then_heals(tmp_path: Path) -> None:
    """A corrupt stored blob is caught at decode: the run fails naming the
    digest and the edge, the corrupt file is deleted (rollback to absence),
    and the deleted blob re-streams on the next run - corruption costs
    retries, never wrong bytes."""

    async def scenario() -> None:
        daemon_store = tmp_path / "daemon-store"
        engine_store = BudgetedDiskCAS(tmp_path / "engine-store")
        proc, host, port = await start_service(
            DEV_MANIFEST, tmp_path, "--value-store", str(daemon_store)
        )
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry, value_store=engine_store)
            await worker.start()
            try:
                first = await make_engine(registry, worker).run(big_image_graph(), ["s"])
                first_prints = {o: v.fingerprint for o, v in first.outputs["s"].items()}

                blob_files = store_blob_files(daemon_store)
                assert len(blob_files) == 2  # the two big intermediates
                for blob in blob_files:
                    blob.write_bytes(b"x" * blob.stat().st_size)

                # Each corrupt blob survives existence checks until a decode
                # reads it, which deletes it and fails that node's edge. The
                # linear graph touches one corrupt blob per run, so exactly
                # two runs fail, each naming the transport, digest, and edge.
                for _ in range(2):
                    with pytest.raises(ExecutionError) as failure:
                        await make_engine(registry, worker).run(big_image_graph(), ["s"])
                    message = str(failure.value)
                    assert "persistentCas blob blake3:" in message
                    assert "input 'image'" in message

                healed = await make_engine(registry, worker).run(big_image_graph(), ["s"])
                assert {o: v.fingerprint for o, v in healed.outputs["s"].items()} == first_prints
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def linked_pair(
    sender_store: BudgetedDiskCAS, receiver_store: BudgetedDiskCAS
) -> tuple[BlobTransfer, BlobTransfer]:
    """Two BlobTransfer ends wired directly together: what one sends, the
    other handles - the frame plumbing of a conversation without a socket."""
    ends: dict[str, BlobTransfer] = {}

    def dispatch_to(name: str) -> blobs_module.SendFrame:
        async def send(header: dict[str, object], blobs: Sequence[bytes]) -> None:
            peer = ends[name]
            kind = header["type"]
            if kind == "blobQuery":
                await peer.answer_query(header)
            elif kind == "blobQueryResult":
                peer.resolve_query(header)
            else:
                await peer.accept_chunk(header, blobs)

        return send

    sender = BlobTransfer(
        sender_store, dispatch_to("receiver"), closed_exc=lambda: ConnectionError("closed")
    )
    receiver = BlobTransfer(
        receiver_store, dispatch_to("sender"), closed_exc=lambda: ConnectionError("closed")
    )
    ends["sender"] = sender
    ends["receiver"] = receiver
    return sender, receiver


def transfer_pair(
    tmp_path: Path, *, receiver_budget: int
) -> tuple[BlobTransfer, BlobTransfer, BudgetedDiskCAS, BudgetedDiskCAS]:
    sender_store = BudgetedDiskCAS(tmp_path / "sender")
    receiver_store = BudgetedDiskCAS(tmp_path / "receiver", max_bytes=receiver_budget)
    sender, receiver = linked_pair(sender_store, receiver_store)
    return sender, receiver, sender_store, receiver_store


def test_budget_eviction_refetches_evicted_blobs(tmp_path: Path) -> None:
    """An evicted blob is a conservative miss: the peer that still holds the
    bytes streams them again, verified, and pays for them again in the
    accounting - never an error."""

    async def scenario() -> None:
        sender, _, _, receiver_store = transfer_pair(tmp_path, receiver_budget=1000)
        blob_x = b"x" * 600
        blob_y = b"y" * 700
        digest_x = digest_bytes(blob_x)
        digest_y = digest_bytes(blob_y)

        moved = await sender.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x][0] == 600
        assert receiver_store.get(digest_x) == blob_x

        # Landing y busts the budget; x is the LRU victim.
        moved = await sender.ensure_peer_holds([(digest_y, blob_y)])
        assert moved[digest_y][0] == 700
        assert not receiver_store.has(digest_x)
        assert receiver_store.get(digest_y) == blob_y

        moved = await sender.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x][0] == 600
        assert receiver_store.get(digest_x) == blob_x

        # A held blob costs a query, not a transfer.
        moved = await sender.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x] == (0, 0.0)

    asyncio.run(scenario())


def test_working_set_larger_than_budget_lands_whole(tmp_path: Path) -> None:
    """One frame can reference more bytes than the receiver's budget: every
    blob of one query is pinned until the next query, so landing the later
    blobs never evicts the earlier ones out from under the frame."""

    async def scenario() -> None:
        sender, _, _, receiver_store = transfer_pair(tmp_path, receiver_budget=1000)
        blob_x = b"x" * 600
        blob_y = b"y" * 700
        digest_x = digest_bytes(blob_x)
        digest_y = digest_bytes(blob_y)

        moved = await sender.ensure_peer_holds([(digest_x, blob_x), (digest_y, blob_y)])
        assert moved[digest_x][0] == 600
        assert moved[digest_y][0] == 700
        assert receiver_store.get(digest_x) == blob_x
        assert receiver_store.get(digest_y) == blob_y

    asyncio.run(scenario())


def test_landing_missing_blob_never_evicts_held_blob_of_same_query(tmp_path: Path) -> None:
    """A held digest in a query is part of the referencing frame's working
    set too: landing the query's missing blobs must not evict it."""

    async def scenario() -> None:
        sender, _, _, receiver_store = transfer_pair(tmp_path, receiver_budget=1000)
        blob_x = b"x" * 600
        blob_y = b"y" * 700
        digest_x = digest_bytes(blob_x)
        digest_y = digest_bytes(blob_y)

        moved = await sender.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x][0] == 600

        moved = await sender.ensure_peer_holds([(digest_x, blob_x), (digest_y, blob_y)])
        assert moved[digest_x] == (0, 0.0)
        assert moved[digest_y][0] == 700
        assert receiver_store.get(digest_x) == blob_x
        assert receiver_store.get(digest_y) == blob_y

    asyncio.run(scenario())


def test_outbound_persist_never_evicts_pinned_inbound_working_set(tmp_path: Path) -> None:
    """One store serves both directions on a side: while an inbound frame's
    working set is pinned, this side's own outbound persist must not trim
    it out of the store, or the inbound frame could never decode."""

    async def scenario() -> None:
        sender, receiver, _, receiver_store = transfer_pair(tmp_path, receiver_budget=1000)

        blob_x = b"x" * 600
        digest_x = digest_bytes(blob_x)
        moved = await sender.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x][0] == 600
        # The receiver answered the query, so x is its pinned working set.

        blob_z = b"z" * 700
        digest_z = digest_bytes(blob_z)
        moved = await receiver.ensure_peer_holds([(digest_z, blob_z)])
        assert moved[digest_z][0] == 700

        assert receiver_store.get(digest_x) == blob_x
        assert receiver_store.get(digest_z) == blob_z

    asyncio.run(scenario())


def test_shared_store_respects_other_conversations_pins(tmp_path: Path) -> None:
    """dinkster-serve hands one engine store to every composed RemoteWorker,
    so pins are owner-scoped in the store itself: one conversation's store
    mutations never evict another conversation's pinned working set."""

    async def scenario() -> None:
        shared = BudgetedDiskCAS(tmp_path / "shared", max_bytes=1000)
        sender_a, _ = linked_pair(BudgetedDiskCAS(tmp_path / "peer-a"), shared)
        transfer_b, _ = linked_pair(shared, BudgetedDiskCAS(tmp_path / "peer-b"))

        blob_x = b"x" * 600
        digest_x = digest_bytes(blob_x)
        moved = await sender_a.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x][0] == 600
        # Conversation A's receiving end pinned {x} in the shared store.

        blob_z = b"z" * 700
        digest_z = digest_bytes(blob_z)
        moved = await transfer_b.ensure_peer_holds([(digest_z, blob_z)])
        assert moved[digest_z][0] == 700

        assert shared.get(digest_x) == blob_x
        assert shared.get(digest_z) == blob_z

    asyncio.run(scenario())


def test_trim_respects_pin_registered_after_it_started(tmp_path: Path) -> None:
    """Cancelling ensure_peer_holds abandons its persist in the executor.
    The orphaned put's trim reads the pin registry as it picks victims, so
    it respects a pin registered after the put started."""

    async def scenario() -> None:
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        blob_slow = b"s" * 700
        digest_slow = digest_bytes(blob_slow)

        class SlowPut(BudgetedDiskCAS):
            def put(self, data: bytes, *, protect: Collection[str] = ()) -> str:
                if digest_bytes(data) == digest_slow:
                    started.set()
                    assert release.wait(5.0)
                    stored = super().put(data, protect=protect)
                    finished.set()
                    return stored
                return super().put(data, protect=protect)

        store = SlowPut(tmp_path / "store", max_bytes=1000)
        blob_x = b"x" * 600
        digest_x = digest_bytes(blob_x)
        store.put(blob_x)

        async def no_send(header: dict[str, object], data: Sequence[bytes]) -> None:
            raise AssertionError("cancelled before any frame is sent")

        transfer = BlobTransfer(store, no_send, closed_exc=lambda: ConnectionError("closed"))
        task = asyncio.create_task(transfer.ensure_peer_holds([(digest_slow, blob_slow)]))
        assert await asyncio.to_thread(started.wait, 5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The pin lands while the orphaned put is still mid-flight.
        pin_owner = object()
        store.pin(pin_owner, [digest_x])
        release.set()
        assert await asyncio.to_thread(finished.wait, 5.0)

        assert store.get(digest_x) == blob_x
        assert store.get(digest_slow) == blob_slow

    asyncio.run(scenario())


def test_close_releases_pins(tmp_path: Path) -> None:
    """A dead conversation's pins never outlive it: after close(), a put
    may evict the digests its last query pinned."""

    async def scenario() -> None:
        sender, receiver, _, receiver_store = transfer_pair(tmp_path, receiver_budget=1000)
        blob_x = b"x" * 600
        digest_x = digest_bytes(blob_x)
        moved = await sender.ensure_peer_holds([(digest_x, blob_x)])
        assert moved[digest_x][0] == 600

        receiver.close()
        blob_z = b"z" * 700
        digest_z = receiver_store.put(blob_z)
        assert not receiver_store.has(digest_x)
        assert receiver_store.get(digest_z) == blob_z

    asyncio.run(scenario())


def test_cancelled_transfer_does_not_poison_the_next_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling ensure_peer_holds between chunks strands a partial landing
    on the receiver. The next query naming that digest discards it, so the
    re-send starts clean instead of overflowing the stale partial."""

    async def scenario() -> None:
        monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", 100)
        sender, _, _, receiver_store = transfer_pair(tmp_path, receiver_budget=10_000)
        blob = bytes(range(256))
        digest = digest_bytes(blob)
        real_send = sender._send
        chunks_sent = 0

        async def cancel_after_first_chunk(
            header: dict[str, object], data: Sequence[bytes]
        ) -> None:
            nonlocal chunks_sent
            await real_send(header, data)
            if header.get("type") == "blobData":
                chunks_sent += 1
                if chunks_sent == 1:
                    raise asyncio.CancelledError

        sender._send = cancel_after_first_chunk
        with pytest.raises(asyncio.CancelledError):
            await sender.ensure_peer_holds([(digest, blob)])
        assert not receiver_store.has(digest)

        sender._send = real_send
        moved = await sender.ensure_peer_holds([(digest, blob)])
        assert moved[digest][0] == len(blob)
        assert receiver_store.get(digest) == blob
        assert not list(receiver_store.root.glob("incoming-*"))

    asyncio.run(scenario())


def test_closed_transfer_raises_instead_of_sending(tmp_path: Path) -> None:
    async def scenario() -> None:
        sender, _, _, _ = transfer_pair(tmp_path, receiver_budget=10_000)
        sender.close()
        blob = b"x" * 64
        with pytest.raises(ConnectionError):
            await sender.ensure_peer_holds([(digest_bytes(blob), blob)])

    asyncio.run(scenario())


def test_close_during_persist_raises_instead_of_hanging(tmp_path: Path) -> None:
    """close() landing while the local store write runs off-loop must not
    let the resumed sender query a peer that will never answer (half-close:
    the send still succeeds, the reply never comes)."""

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        closed = threading.Event()
        transfers: list[BlobTransfer] = []

        def close_transfer() -> None:
            transfers[0].close()
            closed.set()

        class ClosesDuringPersist(BudgetedDiskCAS):
            def has(self, digest: str) -> bool:
                loop.call_soon_threadsafe(close_transfer)
                assert closed.wait(5.0)
                return super().has(digest)

        async def send_into_the_void(header: dict[str, object], data: Sequence[bytes]) -> None:
            return None

        transfer = BlobTransfer(
            ClosesDuringPersist(tmp_path / "store"),
            send_into_the_void,
            closed_exc=lambda: ConnectionError("closed"),
        )
        transfers.append(transfer)
        blob = b"x" * 64
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(
                transfer.ensure_peer_holds([(digest_bytes(blob), blob)]), timeout=5.0
            )

    asyncio.run(scenario())


def test_multi_chunk_blob_lands_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", 100)
        sender, _, _, receiver_store = transfer_pair(tmp_path, receiver_budget=10_000)
        blob = bytes(range(256)) * 2  # 512 bytes, six chunks
        digest = digest_bytes(blob)
        moved = await sender.ensure_peer_holds([(digest, blob)])
        assert moved[digest][0] == len(blob)
        assert receiver_store.get(digest) == blob

    asyncio.run(scenario())


def test_truncated_and_corrupt_transfers_roll_back_to_nothing(tmp_path: Path) -> None:
    """A landing that cannot verify leaves no partial state: not in the
    store, no temp file, and the failure names the digest."""

    async def scenario() -> None:
        store = BudgetedDiskCAS(tmp_path / "store")

        async def no_send(header: dict[str, object], blobs: Sequence[bytes]) -> None:
            raise AssertionError("landing a blob must not send frames")

        transfer = BlobTransfer(store, no_send, closed_exc=lambda: ConnectionError("closed"))
        data = b"payload-bytes" * 10
        digest = digest_bytes(data)

        def frame(size: int, *, last: bool) -> dict[str, Any]:
            return {"type": "blobData", "digest": digest, "size": size, "last": last}

        with pytest.raises(BoundaryError, match=f"{digest}.*truncated"):
            await transfer.accept_chunk(frame(len(data), last=True), [data[: len(data) // 2]])
        assert store.total_bytes() == 0
        assert store_blob_files(store.root) == []
        assert not list(store.root.glob("incoming-*"))

        with pytest.raises(BoundaryError, match=f"{digest}.*failed verification"):
            await transfer.accept_chunk(frame(len(data), last=True), [b"z" * len(data)])
        assert store.total_bytes() == 0
        assert not list(store.root.glob("incoming-*"))

        # close() discards a landing abandoned mid-stream.
        await transfer.accept_chunk(frame(len(data), last=False), [data[:10]])
        transfer.close()
        assert not list(store.root.glob("incoming-*"))

    asyncio.run(scenario())


def test_dedup_graph_streams_shared_payload_once(tmp_path: Path) -> None:
    """One payload feeding two invocations crosses once even on a cold
    store, and the movement is charged to the first referencing edge."""

    async def scenario() -> None:
        daemon_store = tmp_path / "daemon-store"
        engine_store = BudgetedDiskCAS(tmp_path / "engine-store")
        proc, host, port = await start_service(
            DEV_MANIFEST, tmp_path, "--value-store", str(daemon_store)
        )
        diagnostics: list[BoundaryDiagnostic] = []
        try:
            registry = core_registry()
            worker = remote_worker(
                host,
                port,
                registry,
                on_diagnostic=diagnostics.append,
                value_store=engine_store,
            )
            await worker.start()
            try:
                graph = Graph(
                    nodes={
                        "g": GraphNode("dev.image.gradient", {"width": 512, "height": 256}),
                        "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
                        "b": GraphNode(
                            "dev.image.blend",
                            {"a": Link("g", "image"), "b": Link("i", "image"), "ratio": 0.5},
                        ),
                        "s": GraphNode("dev.image.stats", {"image": Link("b", "image")}),
                    }
                )
                result = await make_engine(registry, worker).run(graph, ["s"])
                assert result.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
                edges = big_edges(diagnostics)
                assert edges
                assert all(edge.transport == "persistentCas" for edge in edges)
                moved = sum(edge.network_bytes for edge in edges)
                assert moved == engine_store.total_bytes()
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())
