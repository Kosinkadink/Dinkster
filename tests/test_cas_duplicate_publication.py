from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from dinkster_assets import digest_file
from dinkster_caches import BudgetedDiskCAS, DiskCAS
from dinkster_workers import blobs as blobs_module
from dinkster_workers.blobs import BlobTransfer


def test_duplicate_path_transfers_publish_with_bounded_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        source = tmp_path / "source.bin"
        source.write_bytes(bytes(range(251)) * (9 * 1024 * 1024 // 251 + 1))
        digest = digest_file(source)
        sender_stores = [DiskCAS(tmp_path / f"sender-{index}") for index in range(2)]
        receiver_store = BudgetedDiskCAS(tmp_path / "receiver", max_bytes=1)
        senders: list[BlobTransfer] = []
        receivers: list[BlobTransfer] = []
        replies_ready = asyncio.Event()
        reply_count = 0

        def sender_send(index: int) -> blobs_module.SendFrame:
            async def send(header: dict[str, object], blobs: Sequence[bytes]) -> None:
                if header["type"] == "blobQuery":
                    await receivers[index].answer_query(header)
                else:
                    await receivers[index].accept_chunk(header, blobs)

            return send

        def receiver_send(index: int) -> blobs_module.SendFrame:
            async def send(header: dict[str, object], _blobs: Sequence[bytes]) -> None:
                nonlocal reply_count
                assert header["type"] == "blobQueryResult"
                assert header["missing"] == [digest]
                reply_count += 1
                if reply_count == 2:
                    replies_ready.set()
                await replies_ready.wait()
                senders[index].resolve_query(header)

            return send

        for index, sender_store in enumerate(sender_stores):
            senders.append(
                BlobTransfer(
                    sender_store,
                    sender_send(index),
                    closed_exc=lambda: ConnectionError("closed"),
                )
            )
            receivers.append(
                BlobTransfer(
                    receiver_store,
                    receiver_send(index),
                    closed_exc=lambda: ConnectionError("closed"),
                )
            )

        cas_roots = {store.root for store in [*sender_stores, receiver_store]}
        real_read_bytes = Path.read_bytes

        def reject_whole_file_read(path: Path) -> bytes:
            if path == source or any(path.is_relative_to(root) for root in cas_roots):
                raise AssertionError(f"landing must stream file hashing: {path}")
            return real_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)
        try:
            moved = await asyncio.gather(
                *(sender.ensure_peer_holds([(digest, source)]) for sender in senders)
            )
            assert reply_count == 2
            assert all(stats[digest][0] == source.stat().st_size for stats in moved)
            received = receiver_store.resolve(digest)
            assert received is not None and digest_file(received) == digest

            receiver_store.put(b"pressure while both conversations pin")
            assert receiver_store.has(digest)
            receivers[0].close()
            receiver_store.put(b"pressure while one conversation pins")
            assert receiver_store.has(digest)
        finally:
            for transfer in [*senders, *receivers]:
                transfer.close()

    asyncio.run(scenario())


def test_publication_fallback_and_existing_put_hash_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = DiskCAS(tmp_path / "cas")
    data = bytes(range(251)) * (9 * 1024 * 1024 // 251 + 1)
    digest = cas.put(data)
    target = cas.resolve(digest)
    assert target is not None
    target.unlink()
    candidate = tmp_path / "candidate"
    candidate.write_bytes(data)

    real_replace = os.replace
    real_read_bytes = Path.read_bytes

    def lose_replace_race(source: Path, destination: Path) -> None:
        winner = tmp_path / "winner"
        winner.write_bytes(data)
        real_replace(winner, destination)
        raise PermissionError("concurrent publisher won")

    def reject_target_read_bytes(path: Path) -> bytes:
        if path == target:
            raise AssertionError("published CAS blobs must be hashed incrementally")
        return real_read_bytes(path)

    monkeypatch.setattr(os, "replace", lose_replace_race)
    monkeypatch.setattr(Path, "read_bytes", reject_target_read_bytes)

    cas.adopt_file(candidate, digest)
    assert not candidate.exists()
    assert cas.put(data) == digest
    assert digest_file(target) == digest
