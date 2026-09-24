"""Persistent-CAS blob transport: land payload bytes in the peer's value
store before the frame that references them crosses.

The frames are symmetric - either side can be sender or receiver:

- ``blobQuery``: the sender names the digests its next frame references;
  ``blobQueryResult`` answers with the ones the receiver's store is
  missing. A held digest costs one round trip, never a transfer, and the
  stores are persistent, so a blob crosses at most once across
  conversations, reconnects, and re-runs.
- ``blobData``: one chunk of one missing blob. Chunked so a multi-GB value
  never monopolizes the socket - the send lock is taken per frame, so
  frames from other tasks interleave at chunk boundaries. The receiver
  streams chunks to a temp file and, on the last chunk, verifies size and
  digest before the blob becomes visible (adopt_file). A failed transfer
  rolls back to nothing: no partial blob ever resolves.
"""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import asyncio
import contextlib
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, TypeVar, cast

from .boundary import BoundaryError, TransferStat, ValueStore
from .produced_assets import result_asset_digests

_CHUNK_BYTES = 8 * MEBIBYTE

SendFrame = Callable[[dict[str, object], Sequence[bytes]], Awaitable[None]]
_T = TypeVar("_T")


async def await_file_operation(task: asyncio.Future[_T]) -> _T:
    """Keep file lifetimes owned until executor work stops, even after repeated cancellation."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(task)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()
        raise


@dataclass
class _Incoming:
    path: Path
    handle: BinaryIO
    expected: int
    received: int = 0


class BlobTransfer:
    """One side's blob-transfer state for one conversation.

    ``ensure_peer_holds`` serializes senders (one query/stream cycle at a
    time) but never holds the frame send lock across a wait, so the read
    loop stays free to answer the peer's own queries - both sides can be
    mid-transfer without deadlock."""

    def __init__(
        self,
        store: ValueStore,
        send: SendFrame,
        *,
        closed_exc: Callable[[], Exception],
        pin_owner: object | None = None,
    ) -> None:
        self._store = store
        self._send = send
        self._closed_exc = closed_exc
        self._pin_owner = self if pin_owner is None else pin_owner
        self._lock = asyncio.Lock()
        self._closed = False
        self._request_index = 0
        self._queries: dict[str, asyncio.Future[Mapping[str, Any]]] = {}
        self._incoming: dict[str, _Incoming] = {}

    async def ensure_peer_holds(
        self, pending: Sequence[tuple[str, bytes | Path]]
    ) -> dict[str, tuple[int, float]]:
        """Make every (digest, payload) pair resolvable in the peer's store.

        Also lands the payload in the local store, so this side answers the
        peer's future queries for them. Returns, per digest, the payload
        bytes actually streamed and the milliseconds spent streaming them -
        (0, 0.0) for digests the peer already held."""
        unique: dict[str, bytes | Path] = {}
        for digest, data in pending:
            unique.setdefault(digest, data)
        if not unique:
            return {}
        async with self._lock:
            if self._closed:
                raise self._closed_exc()

            def persist() -> None:
                # put's trim reads the store's pin registry as it picks
                # victims, so this can never evict a pinned working set -
                # not even when cancellation orphans it in the executor
                # and the pin registers after it started.
                for digest, data in unique.items():
                    if self._store.has(digest):
                        continue
                    if isinstance(data, Path):
                        candidate = self._store.root / f"outgoing-{uuid.uuid4().hex}"
                        try:
                            with data.open("rb") as source, candidate.open("xb") as target:
                                while chunk := source.read(_CHUNK_BYTES):
                                    target.write(chunk)
                            self._store.adopt_file(candidate, digest)
                        except BaseException:
                            with contextlib.suppress(OSError):
                                candidate.unlink()
                            raise
                    else:
                        self._store.put(data)

            if any(isinstance(data, Path) for data in unique.values()):
                publication = asyncio.create_task(asyncio.to_thread(persist))
                await await_file_operation(publication)
            else:
                await asyncio.to_thread(persist)
            if self._closed:
                # close() ran during persist: it already failed every
                # registered query, so a future registered now would never
                # resolve and the caller would wait forever.
                raise self._closed_exc()
            request_id = f"blobs-{self._request_index}-{uuid.uuid4().hex}"
            self._request_index += 1
            future: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
            self._queries[request_id] = future
            try:
                await self._send(
                    {"type": "blobQuery", "requestId": request_id, "digests": list(unique)},
                    (),
                )
                reply = await future
            finally:
                self._queries.pop(request_id, None)
            missing_raw = reply.get("missing")
            if not isinstance(missing_raw, list):
                raise BoundaryError("blobQueryResult must carry a 'missing' list")
            missing = [str(digest) for digest in cast("list[object]", missing_raw)]
            unknown = sorted(set(missing) - set(unique))
            if unknown:
                raise BoundaryError(
                    f"peer reported digests it was never offered as missing: {', '.join(unknown)}"
                )
            moved = {digest: (0, 0.0) for digest in unique}
            for digest in missing:
                data = unique[digest]
                started = time.perf_counter()
                if isinstance(data, Path):
                    with data.open("rb") as source:
                        size = os.fstat(source.fileno()).st_size
                        offset = 0
                        while True:
                            if offset >= size:
                                await self._send(
                                    {
                                        "type": "blobData",
                                        "digest": digest,
                                        "size": size,
                                        "last": True,
                                    },
                                    (b"",),
                                )
                                break
                            read = asyncio.create_task(
                                asyncio.to_thread(source.read, min(_CHUNK_BYTES, size - offset))
                            )
                            chunk = await await_file_operation(read)
                            if not chunk:
                                raise BoundaryError(
                                    f"blob {digest}: source ended after {offset} of {size} bytes"
                                )
                            offset += len(chunk)
                            last = offset >= size
                            await self._send(
                                {
                                    "type": "blobData",
                                    "digest": digest,
                                    "size": size,
                                    "last": last,
                                },
                                (chunk,),
                            )
                            if last:
                                break
                else:
                    size = len(data)
                    offset = 0
                    while True:
                        chunk = data[offset : offset + _CHUNK_BYTES]
                        offset += len(chunk)
                        last = offset >= size
                        await self._send(
                            {
                                "type": "blobData",
                                "digest": digest,
                                "size": size,
                                "last": last,
                            },
                            (chunk,),
                        )
                        if last:
                            break
                moved[digest] = (size, (time.perf_counter() - started) * 1000.0)
            return moved

    async def answer_query(self, header: Mapping[str, Any]) -> None:
        request_id = str(header.get("requestId", ""))
        digests_raw = header.get("digests")
        digests = (
            [str(digest) for digest in cast("list[object]", digests_raw)]
            if isinstance(digests_raw, list)
            else []
        )
        # A query naming a digest proves no stream of it is in flight from
        # this peer (senders serialize query+stream cycles), so any landing
        # state still here is stale - an earlier transfer the sender
        # abandoned mid-stream. Discard it or the re-send would append to
        # the partial and overflow.
        for digest in digests:
            self._discard(digest)
        # Pin the frame's whole working set - held digests too - until the
        # next query replaces it: no store mutation (an inbound landing,
        # this side's own outbound persist, or another conversation sharing
        # the store) may evict a blob the frame references, or a working
        # set larger than the budget could never decode. Pinned before the
        # probe so a digest reported held cannot be evicted after the
        # report.
        self._store.pin(self._pin_owner, digests)

        def probe() -> list[str]:
            return [digest for digest in digests if not self._store.has(digest)]

        missing = await asyncio.to_thread(probe)
        await self._send(
            {"type": "blobQueryResult", "requestId": request_id, "missing": missing}, ()
        )

    def resolve_query(self, header: Mapping[str, Any]) -> None:
        future = self._queries.get(str(header.get("requestId", "")))
        if future is not None and not future.done():
            future.set_result(header)

    async def accept_chunk(self, header: Mapping[str, Any], blobs: Sequence[bytes]) -> None:
        digest = str(header.get("digest", ""))
        expected = header.get("size")
        if not digest or type(expected) is not int or expected < 0 or len(blobs) != 1:
            raise BoundaryError("malformed blobData frame")
        chunk = blobs[0]
        state = self._incoming.get(digest)
        if state is None:
            path = self._store.root / f"incoming-{uuid.uuid4().hex}"
            handle = cast(BinaryIO, await asyncio.to_thread(path.open, "wb"))
            state = _Incoming(path=path, handle=handle, expected=expected)
            self._incoming[digest] = state
        if state.expected != expected:
            self._discard(digest)
            raise BoundaryError(f"blob {digest}: chunks disagree about the blob size")
        state.received += len(chunk)
        if state.received > state.expected:
            self._discard(digest)
            raise BoundaryError(
                f"blob {digest} transfer overflowed: got {state.received} of {state.expected} bytes"
            )
        await asyncio.to_thread(state.handle.write, chunk)
        if not bool(header.get("last")):
            return
        del self._incoming[digest]
        await asyncio.to_thread(state.handle.close)
        if state.received != state.expected:
            with contextlib.suppress(OSError):
                state.path.unlink()
            raise BoundaryError(
                f"blob {digest} transfer truncated: got {state.received} of {state.expected} bytes"
            )
        try:
            # adopt_file verifies the digest and consumes the temp file
            # either way, so a corrupt transfer leaves nothing behind. The
            # rest of the frame's working set is pinned, so the landing's
            # trim cannot evict it.
            await asyncio.to_thread(self._store.adopt_file, state.path, digest)
        except Exception as exc:
            raise BoundaryError(f"blob {digest} failed verification on landing: {exc}") from exc

    def _discard(self, digest: str) -> None:
        state = self._incoming.pop(digest, None)
        if state is None:
            return
        with contextlib.suppress(Exception):
            state.handle.close()
        with contextlib.suppress(OSError):
            state.path.unlink()

    def close(self, *, preserve_pins: bool = False) -> None:
        """Abort transfers; retain source pins only for a resumable transport replacement."""
        self._closed = True
        if not preserve_pins:
            self._store.unpin(self._pin_owner)
        for future in self._queries.values():
            if not future.done():
                future.set_exception(self._closed_exc())
        self._queries.clear()
        for digest in list(self._incoming):
            self._discard(digest)


def _walk_digests(wire: Mapping[str, Any], out: list[str]) -> None:
    elements = wire.get("elements")
    if isinstance(elements, list):
        for child in cast("list[object]", elements):
            if isinstance(child, Mapping):
                _walk_digests(cast("Mapping[str, Any]", child), out)
        return
    payload = wire.get("payload")
    if not isinstance(payload, Mapping):
        return
    payload_wire = cast("Mapping[str, Any]", payload)
    if payload_wire.get("transport") == "persistentCas":
        out.append(str(payload_wire.get("digest", "")))


def _moved_for_edge(
    wire: object, remaining: dict[str, tuple[int, float]]
) -> tuple[bool, int, float]:
    """Bytes/time this edge's persistentCas digests actually moved.

    Each streamed blob is charged to the first edge that references its
    digest; later edges naming the same digest moved nothing extra."""
    digests: list[str] = []
    if isinstance(wire, Mapping):
        _walk_digests(cast("Mapping[str, Any]", wire), digests)
    network = 0
    transfer = 0.0
    for digest in digests:
        entry = remaining.pop(digest, None)
        if entry is not None:
            network += entry[0]
            transfer += entry[1]
    return bool(digests), network, transfer


def attribute_moved(
    stats: Mapping[str, TransferStat],
    values_wire: Mapping[str, Any],
    moved: Mapping[str, tuple[int, float]],
) -> dict[str, TransferStat]:
    """Fold actually-moved byte counts into per-edge transfer stats."""
    remaining = dict(moved)
    attributed: dict[str, TransferStat] = {}
    for edge_id, stat in stats.items():
        referenced, network, transfer = _moved_for_edge(values_wire.get(edge_id), remaining)
        attributed[edge_id] = (
            replace(stat, network_bytes=network, transfer_ms=transfer) if referenced else stat
        )
    return attributed


def attribute_moved_wire(
    header: Mapping[str, Any],
    moved: Mapping[str, tuple[int, float]],
    blobs: Sequence[bytes] = (),
) -> None:
    """attribute_moved for an already-encoded result frame: fold moved
    bytes into its ``outputStats`` wire entries in place."""
    outputs = header.get("outputs")
    stats = header.get("outputStats")
    if not isinstance(outputs, Mapping) or not isinstance(stats, Mapping):
        return
    remaining = dict(moved)
    for edge_id, wire in cast("Mapping[str, Any]", outputs).items():
        referenced, network, transfer = _moved_for_edge(wire, remaining)
        stat_wire = cast("Mapping[str, Any]", stats).get(edge_id)
        if not isinstance(stat_wire, dict):
            continue
        stat_wire = cast("dict[str, Any]", stat_wire)
        if referenced:
            stat_wire["networkBytes"] = network
            stat_wire["transferMs"] = transfer
        if blobs:
            for digest in result_asset_digests({"outputs": {edge_id: wire}}, blobs):
                source_bytes, source_ms = remaining.pop(digest, (0, 0.0))
                stat_wire["networkBytes"] = stat_wire.get("networkBytes", 0) + source_bytes
                stat_wire["transferMs"] = stat_wire.get("transferMs", 0.0) + source_ms
