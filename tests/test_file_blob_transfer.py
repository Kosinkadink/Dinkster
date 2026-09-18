from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest
from dinkster_assets import digest_file
from dinkster_caches import BudgetedDiskCAS, CASError, DiskCAS
from dinkster_values import TypeRegistry
from dinkster_workers import BoundaryError
from dinkster_workers import blobs as blobs_module
from dinkster_workers.blobs import BlobTransfer
from dinkster_workers.boundary import ValueCodec
from dinkster_workers.session import BoundarySession


def linked_pair(
    sender_store: DiskCAS, receiver_store: DiskCAS
) -> tuple[BlobTransfer, BlobTransfer]:
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
    ends.update(sender=sender, receiver=receiver)
    return sender, receiver


def large_source(path: Path) -> bytes:
    data = bytes(range(251)) * (9 * 1024 * 1024 // 251 + 1)
    path.write_bytes(data)
    return data


def test_file_payload_streams_once_and_persists_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "producer-video.bin"
        expected = large_source(source)
        digest = digest_file(source)
        sender_store = DiskCAS(tmp_path / "sender")
        receiver_store = BudgetedDiskCAS(tmp_path / "receiver")
        sender, receiver = linked_pair(sender_store, receiver_store)

        real_open = Path.open
        source_reads: list[int] = []
        source_handles: list[BinaryIO] = []

        class TrackedReader:
            def __init__(self, handle: BinaryIO) -> None:
                self.handle = handle
                source_handles.append(handle)

            def __enter__(self) -> TrackedReader:
                return self

            def __exit__(self, *args: object) -> None:
                self.handle.close()

            def fileno(self) -> int:
                return self.handle.fileno()

            def read(self, size: int = -1) -> bytes:
                source_reads.append(size)
                return self.handle.read(size)

        def tracked_open(path: Path, *args: Any, **kwargs: Any) -> BinaryIO:
            handle = real_open(path, *args, **kwargs)
            if path == source and args and args[0] == "rb":
                return cast(BinaryIO, TrackedReader(cast(BinaryIO, handle)))
            return cast(BinaryIO, handle)

        real_read_bytes = Path.read_bytes

        def reject_source_read_bytes(path: Path) -> bytes:
            if path == source:
                raise AssertionError("source.read_bytes() must not be used")
            return real_read_bytes(path)

        monkeypatch.setattr(Path, "open", tracked_open)
        monkeypatch.setattr(Path, "read_bytes", reject_source_read_bytes)

        first = await sender.ensure_peer_holds([(digest, source)])
        assert first[digest][0] == len(expected)
        assert first[digest][1] > 0.0
        assert receiver_store.get(digest) == expected
        assert sender_store.get(digest) == expected
        assert source.exists()
        assert source.stat().st_size == len(expected)
        assert source_reads
        assert all(0 <= size <= blobs_module._CHUNK_BYTES for size in source_reads)
        assert all(handle.closed for handle in source_handles)

        second = await sender.ensure_peer_holds([(digest, source)])
        assert second[digest] == (0, 0.0)
        sender.close()
        receiver.close()

    asyncio.run(scenario())


def test_wrong_file_digest_never_reaches_receiver(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "producer-video.bin"
        large_source(source)
        sender_store = DiskCAS(tmp_path / "sender")
        receiver_store = DiskCAS(tmp_path / "receiver")
        sender, receiver = linked_pair(sender_store, receiver_store)
        wrong_digest = "blake3:" + "0" * 64

        with pytest.raises(CASError, match="does not hash"):
            await sender.ensure_peer_holds([(wrong_digest, source)])

        assert source.exists()
        assert not sender_store.has(wrong_digest)
        assert not receiver_store.has(wrong_digest)
        assert not list(sender_store.root.glob("outgoing-*"))
        assert not list(receiver_store.root.glob("incoming-*"))
        sender.close()
        receiver.close()

    asyncio.run(scenario())


def test_file_changed_after_local_adoption_fails_receiver_verification(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "producer-video.bin"
        expected = large_source(source)
        digest = digest_file(source)
        sender_store = DiskCAS(tmp_path / "sender")
        receiver_store = DiskCAS(tmp_path / "receiver")
        sender, receiver = linked_pair(sender_store, receiver_store)
        real_send = sender._send

        async def change_after_persist(header: dict[str, object], blobs: Sequence[bytes]) -> None:
            if header.get("type") == "blobQuery":
                source.write_bytes(b"z" * len(expected))
            await real_send(header, blobs)

        sender._send = change_after_persist
        with pytest.raises(BoundaryError, match=f"{digest}.*failed verification"):
            await sender.ensure_peer_holds([(digest, source)])

        assert sender_store.get(digest) == expected
        assert not receiver_store.has(digest)
        assert not list(receiver_store.root.glob("incoming-*"))
        sender.close()
        receiver.close()

    asyncio.run(scenario())


def test_interrupted_file_stream_closes_source_and_receiver_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "producer-video.bin"
        large_source(source)
        digest = digest_file(source)
        sender_store = DiskCAS(tmp_path / "sender")
        receiver_store = DiskCAS(tmp_path / "receiver")
        sender, receiver = linked_pair(sender_store, receiver_store)
        real_send = sender._send
        source_handle: BinaryIO | None = None
        chunks = 0

        async def interrupt(header: dict[str, object], blobs: Sequence[bytes]) -> None:
            nonlocal chunks
            await real_send(header, blobs)
            if header.get("type") == "blobData":
                chunks += 1
                if chunks == 1:
                    raise asyncio.CancelledError

        real_open = Path.open
        opens = 0

        def capture_transfer_open(path: Path, *args: Any, **kwargs: Any) -> BinaryIO:
            nonlocal opens, source_handle
            handle = cast(BinaryIO, real_open(path, *args, **kwargs))
            if path == source and args and args[0] == "rb":
                opens += 1
                if opens == 2:
                    source_handle = handle
            return handle

        monkeypatch.setattr(Path, "open", capture_transfer_open)
        sender._send = interrupt
        with pytest.raises(asyncio.CancelledError):
            await sender.ensure_peer_holds([(digest, source)])

        assert source_handle is not None and source_handle.closed
        assert source.exists()
        assert not receiver_store.has(digest)
        assert list(receiver_store.root.glob("incoming-*"))
        receiver.close()
        assert not list(receiver_store.root.glob("incoming-*"))
        sender.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("blocked_open", [1, 2], ids=["publication", "read"])
def test_file_io_finishes_before_repeated_cancellation_releases_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_open: int
) -> None:
    async def scenario() -> None:
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        source_handle: BinaryIO | None = None
        opens = 0
        real_open = Path.open

        class SlowReader:
            def __init__(self, handle: BinaryIO) -> None:
                self.handle = handle

            def __enter__(self) -> SlowReader:
                return self

            def __exit__(self, *args: object) -> None:
                self.handle.close()

            def fileno(self) -> int:
                return self.handle.fileno()

            def read(self, size: int = -1) -> bytes:
                entered.set()
                assert release.wait(5)
                chunk = self.handle.read(size)
                finished.set()
                return chunk

        def blocked_source_open(path: Path, *args: Any, **kwargs: Any) -> BinaryIO:
            nonlocal opens, source_handle
            handle = cast(BinaryIO, real_open(path, *args, **kwargs))
            if path == source and args and args[0] == "rb":
                opens += 1
                if opens == blocked_open:
                    source_handle = handle
                    return cast(BinaryIO, SlowReader(handle))
            return handle

        producer = BudgetedDiskCAS(tmp_path / "producer", max_bytes=1)
        digest = producer.put(b"encoded")
        source = producer.resolve(digest)
        assert source is not None
        sender, receiver = linked_pair(DiskCAS(tmp_path / "sender"), DiskCAS(tmp_path / "peer"))
        monkeypatch.setattr(Path, "open", blocked_source_open)

        async def transfer() -> None:
            producer.pin(sender, [digest])
            try:
                await sender.ensure_peer_holds([(digest, source)])
            finally:
                producer.unpin(sender)
                sender.close()

        task = asyncio.create_task(transfer())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
            assert not sender._closed
            assert source_handle is not None and not source_handle.closed
            producer.put(b"cache pressure during file operation")
            assert producer.has(digest)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            assert source_handle.closed
            assert sender._closed
            producer.put(b"cache pressure after file operation")
            assert not producer.has(digest)
            assert not list(sender._store.root.glob("outgoing-*"))
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            sender.close()
            receiver.close()

    asyncio.run(scenario())


def test_received_source_stays_pinned_until_result_replay_after_disconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = BudgetedDiskCAS(tmp_path / "receiver", max_bytes=1)
        registry = TypeRegistry()
        session = BoundarySession(
            registry, role="remote worker", pack="pack", codec=ValueCodec(registry), resumable=True
        )

        async def send(
            header: Mapping[str, object],
            blobs: Sequence[bytes],
            segments: Sequence[SharedMemory] = (),
        ) -> None:
            pass

        session.send = send
        session.enable_blob_transfer(store)
        try:
            source = store.put(b"produced source")
            transfer = session._blob_transfer
            assert transfer is not None
            await transfer.answer_query({"requestId": "source", "digests": [source]})
            await session._detach_for_resume()
            store.put(b"another session sharing this cache")
            assert store.has(source), "replayed result must still resolve its transferred source"
            replacement = session._blob_transfer
            assert replacement is not None and replacement is not transfer
            await replacement.answer_query({"requestId": "next", "digests": [source]})
        finally:
            await session.close()
        store.put(b"cache pressure after final close")
        assert not store.has(source), "final close must release the retained query pins"

    asyncio.run(scenario())
