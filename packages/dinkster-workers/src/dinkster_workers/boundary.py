"""The process boundary: framed messages and the value wire codec.

Everything that crosses between an engine process and an isolated worker
process goes through here, in both directions:

- Frames: ``uint32 header length | JSON header | binary blobs``. The header
  carries structure (message type, ids, value descriptors, schema wire); the
  blobs carry payload and meta bytes so binary data never rides base64.
- Values cross as envelopes (hazard H2): type id, fingerprint, and meta in
  the header; payload bytes produced by the type's registered codec. The
  fingerprint travels verbatim, so cache keys stay location-independent
  (hazard H4) - a value fingerprinted in a worker keys the same cache entry
  everywhere.
- Payload transports (DESIGN 3.2): ``inline`` (bytes in the frame) and
  ``shm`` (bytes in a POSIX shared-memory segment, for encoded payloads at
  or above a size threshold). Which transport carries a value is a placement
  decision made here; node code never sees it (hazard H9).

shm segments are single-hop handoffs: the sender creates and fills a
segment (keeping its handle open - see ``release_segment``), the receiver
retains a read-only mapping and acknowledges (``shmAck``), and the sender
unlinks its name. Anything unacknowledged when a side shuts down is released
by its creator (hazard H14).
"""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import mmap
import os
import sys
import time
import uuid
from collections.abc import Callable, Collection, Generator, Mapping, Sequence, Set
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Protocol, cast

from dinkster_assets import AssetError, digest_bytes
from dinkster_protocol import (
    ErrorHint,
    ExportSnapshot,
    Invocation,
    InvocationOutcome,
    InvocationResult,
    MediaSourceAuthority,
    NodeError,
    SavedArtifactCandidate,
    is_extension_snapshot_digest,
    validate_preview_animation,
    validate_preview_mode,
)
from dinkster_protocol.result_algebra import JsonLiteral
from dinkster_schema import SCHEMA_WIRE_SERVE_VERSIONS, schema_from_wire, schema_to_wire
from dinkster_values import (
    BufferEncoding,
    EncodedPayload,
    ListPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    default_decode,
    default_encode,
    list_children,
    parse_list_type_id,
    stable_hash,
)

#: Windows and macOS report shared-memory mappings rounded up to a page
#: multiple (macOS ftruncates POSIX shm to 16 KiB pages on Apple Silicon);
#: Linux reports the logical size exactly.
_SHM_MAPPING_ROUNDS = os.name == "nt" or sys.platform == "darwin"

DEFAULT_SHM_THRESHOLD = 256 * 1024
"""Encoded payloads at or above this many bytes ride shared memory instead
of the socket frame."""

DEFAULT_CAS_THRESHOLD = 256 * 1024
"""Encoded payloads at or above this many bytes are content-addressed over
a network boundary: repeat crossings of the same bytes send a digest-only
descriptor instead of the payload."""

DEFAULT_CAS_BUDGET = 512 * 1024 * 1024
"""Cap on payload bytes each side retains for conversation-scoped dedup.
Once a codec's retained bytes reach this budget, further first-crossings
still work - they just carry their bytes every time instead of becoming
digest-only. Bounds the RAM a long-lived connection can pin."""

PROTOCOL_VERSION = 9
"""The boundary protocol version. Carried in the remote hello exchange
(clientHello/hello) so peers built against different wire contracts refuse
each other explicitly instead of corrupting frames. Local parent/child pairs
ship together and do not negotiate.

v9: schema wire v41 in worker declarations and invocation effective schemas;
v8: stable engine, daemon, job-attempt, and invocation identities with
same-process rebind, owner fencing, and acknowledged result replay;
v7: session leases (``engine`` label in clientHello, ``leaseTtl`` in the
service hello, ``heartbeat``/``heartbeatAck`` frames);
v6: resident type ids ``dinkster.clip``/``dinkster.vae`` (renamed from
``dinkster.text_encoder``/``dinkster.codec``);
v5: role-labeled multi-stream payloads in ``dinkster.latent``;
v4: saved output artifacts in invocation results;
v3: lazy-status hook frames and schema wire v16;
v2: recursive list value descriptors (``elements``) and schema wire v2
(recursive ``element`` type expressions)."""

_MAX_HEADER_BYTES = 64 * 1024 * 1024


class BoundaryError(Exception):
    """A frame or value could not cross the process boundary."""


class EncodedPayloadValidationError(BoundaryError):
    """A registered type refused encoded bytes before their fingerprint was trusted."""


class PersistentBlobMissing(BoundaryError):
    """A persistentCas descriptor named a digest the local value store cannot resolve."""


class ValueStore(Protocol):
    """The persistent blob store behind the ``persistentCas`` transport,
    on either side of the boundary. dinkster-caches' BudgetedDiskCAS (and
    DiskCAS) satisfy it structurally. Synchronous like DiskCAS: async
    callers wrap calls in ``asyncio.to_thread``."""

    @property
    def root(self) -> Path: ...

    def has(self, digest: str) -> bool: ...

    def resolve(self, digest: str) -> Path | None: ...

    def get(self, digest: str) -> bytes | None: ...

    def put(self, data: bytes, *, protect: Collection[str] = ()) -> str: ...

    def adopt_file(self, path: Path, digest: str, *, protect: Collection[str] = ()) -> None: ...

    def pin(self, owner: object, digests: Collection[str]) -> None: ...

    def unpin(self, owner: object) -> None: ...


@dataclass(frozen=True)
class TransferStat:
    """What one value's boundary crossing cost on this side.

    ``codec_ms`` is time spent encoding (sender) or attaching shared memory
    (receiver) at transfer time; lazy ``load()`` decode is not included.
    ``reused`` means encoded bytes were relayed without re-encoding.
    ``network_bytes``/``transfer_ms`` are payload bytes the persistentCas
    transport actually streamed for this edge and the time spent streaming
    them - zero on a store hit, and zero for the other transports (their
    movement is implied: inline and first-crossing cas bytes ride the
    frame itself)."""

    transport: str
    size_bytes: int
    codec_ms: float
    declared_codec: bool
    reused: bool
    network_bytes: int = 0
    transfer_ms: float = 0.0


async def read_frame(
    reader: asyncio.StreamReader,
) -> tuple[dict[str, Any], list[bytes]] | None:
    """Read one frame. Returns None on clean or broken EOF."""
    try:
        raw = await reader.readexactly(4)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    size = int.from_bytes(raw, "big")
    if size > _MAX_HEADER_BYTES:
        raise BoundaryError(f"frame header too large: {size} bytes")
    try:
        header_obj = cast(
            object,
            json.loads(
                await reader.readexactly(size),
                object_pairs_hook=_unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            ),
        )
        if not isinstance(header_obj, dict):
            raise BoundaryError("frame header must be a JSON object")
        header = cast("dict[str, Any]", header_obj)
        blobs: list[bytes] = []
        for length in header.get("blobs", ()):
            if type(length) is not int or length < 0:
                raise BoundaryError("frame blob lengths must be non-negative integers")
            blobs.append(await reader.readexactly(length))
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    except (ValueError, TypeError, json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise BoundaryError("frame header is not strict JSON") from exc
    return header, blobs


async def write_frame(
    writer: asyncio.StreamWriter, header: Mapping[str, object], blobs: Sequence[bytes]
) -> None:
    """Write one frame. Callers serialize access (frames must not interleave)."""
    full = dict(header)
    full["blobs"] = [len(blob) for blob in blobs]
    try:
        data = json.dumps(full, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise BoundaryError("frame header is not strict JSON") from exc
    if len(data) > _MAX_HEADER_BYTES:
        raise BoundaryError(f"frame header too large: {len(data)} bytes")
    writer.write(len(data).to_bytes(4, "big"))
    writer.write(data)
    for blob in blobs:
        writer.write(blob)
    await writer.drain()


def _create_empty_segment(size: int) -> SharedMemory:
    return SharedMemory(name="dinkster" + uuid.uuid4().hex[:16], create=True, size=size)


def _create_segment(data: bytes | memoryview) -> SharedMemory:
    segment = _create_empty_segment(len(data))
    buffer = segment.buf
    assert buffer is not None  # always set on a live segment
    try:
        buffer[: len(data)] = data
    except Exception:
        buffer.release()
        release_segment(segment)
        raise
    return segment


def _create_segment_from_encoding(encoding: BufferEncoding) -> SharedMemory:
    segment = _create_empty_segment(encoding.size)
    buffer = segment.buf
    assert buffer is not None  # always set on a live segment
    target = buffer[: encoding.size]
    try:
        written = encoding.write(target)
        if written != encoding.size:
            raise ValueError(f"buffer encoder wrote {written} bytes, expected {encoding.size}")
    except Exception:
        with contextlib.suppress(Exception):
            target.release()
        with contextlib.suppress(Exception):
            buffer.release()
        release_segment(segment)
        raise
    target.release()
    return segment


def _attach_segment(name: str) -> SharedMemory:
    if sys.version_info >= (3, 13):
        return SharedMemory(name=name, track=False)
    segment = SharedMemory(name=name)
    if os.name == "posix":
        # Pre-3.13 CPython registers even attach-only handles with the
        # resource tracker, which would unlink the segment when *this*
        # process exits - but the creator owns unlinking (hazard H14).
        from multiprocessing import resource_tracker

        resource_tracker.unregister(f"/{name}", "shared_memory")
    return segment


def release_segment(segment: SharedMemory) -> None:
    """Drop a sent segment after acknowledgement (or on shutdown): close our
    handle, then unlink the name.

    The sender must hold its handle open from send until ack - on Windows a
    segment is freed when its last handle closes, so closing at send time
    would destroy it before the receiver attaches. Close-then-unlink is the
    portable release order (unlink is a no-op on Windows)."""
    with contextlib.suppress(Exception):
        segment.close()
    with contextlib.suppress(Exception):
        segment.unlink()


def _open_segment(name: str, size: int) -> tuple[SharedMemory, memoryview]:
    if size <= 0:
        raise BoundaryError("shared-memory payload size must be positive")
    try:
        segment = _attach_segment(name)
    except FileNotFoundError as exc:
        raise BoundaryError(f"shared-memory segment '{name}' has vanished") from exc
    buffer = segment.buf
    assert buffer is not None  # always set on a live segment
    actual_size = len(buffer)
    expected_size = (
        ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        if _SHM_MAPPING_ROUNDS
        else size
    )
    if actual_size != expected_size:
        segment.close()
        raise BoundaryError(
            f"shared-memory segment '{name}' has size {actual_size}, expected {expected_size}"
        )
    return segment, buffer


class ValueCodec:
    """Encodes/decodes Values for one side of a boundary.

    Encoding uses the type's registered codec; values that arrived encoded
    (EncodedPayload) relay their bytes as-is, so unknown types can flow
    *through* a process that cannot decode them. The transport choice
    (inline vs shm vs cas) is made per value from the encoded size.

    The ``cas`` transport is conversation-scoped content addressing for
    network boundaries (where shm is structurally refused): the first
    crossing of a payload carries digest + bytes, and every later crossing
    of the same bytes - in either direction - carries the digest alone.
    Both sides retain each crossed payload in ``_conversation`` (keyed by
    digest), so a digest-only descriptor always resolves locally; the store
    lives and dies with the codec, i.e. with the conversation (hazard H14:
    nothing outlives the conversation). Digests here are ``stable_hash``
    (stdlib blake2b) - conversation-local dedup identity, deliberately not
    the canonical ``blake3:`` CAS namespace, so isolated pack venvs never
    need the blake3 wheel. Enabled only after hello negotiation
    (``enable_cas``); never assumed.

    Retention is bounded (``cas_budget``): the *sender* decides whether a
    first-crossing payload is retained (fits its remaining budget) and says
    so in the descriptor (``retained``); the receiver mirrors that decision
    verbatim. Sender-decides-receiver-mirrors keeps the two stores
    consistent by construction - a digest-only descriptor only ever names a
    digest both sides kept - so an over-budget conversation degrades to
    resending bytes, never to a protocol error. Nothing is evicted:
    coordinated eviction would need protocol round-trips for RAM that the
    budget already bounds.
    """

    def __init__(
        self,
        registry: TypeRegistry,
        *,
        shm_threshold: int = DEFAULT_SHM_THRESHOLD,
        use_shm: bool = True,
        accept_shm: bool = True,
        cas_threshold: int = DEFAULT_CAS_THRESHOLD,
        cas_budget: int = DEFAULT_CAS_BUDGET,
        value_store: ValueStore | None = None,
    ) -> None:
        if shm_threshold < 1:
            raise ValueError("shm_threshold must be >= 1")
        if cas_threshold < 1:
            raise ValueError("cas_threshold must be >= 1")
        if cas_budget < 0:
            raise ValueError("cas_budget must be >= 0")
        self._registry = registry
        self._shm_threshold = shm_threshold
        self._use_shm = use_shm
        # A remote peer's segment *name* would attach to whatever local
        # memory happens to share it - never the peer's bytes. A codec for a
        # network boundary refuses shm descriptors outright (accept_shm
        # False) instead of reading garbage that parses.
        self._accept_shm = accept_shm
        self._cas_threshold = cas_threshold
        self._cas_budget = cas_budget
        self._cas = False
        self._conversation: dict[str, bytes] = {}
        self._conversation_bytes = 0
        self._value_store = value_store
        self._persistent = False
        self._pending_store: list[tuple[str, bytes]] = []

    def _retain(self, digest: str, data: bytes) -> None:
        if digest not in self._conversation:
            self._conversation[digest] = data
            self._conversation_bytes += len(data)

    def enable_cas(self) -> None:
        """Turn on the cas transport - called only after both hellos listed
        it, so a descriptor never reaches a peer that cannot resolve it."""
        self._cas = True

    def enable_persistent_cas(self) -> None:
        """Turn on the persistentCas transport - called only after both
        hellos listed it AND this side holds a value store. Takes priority
        over the conversation cas transport at encode time, so a persistent
        conversation never populates the conversation-scoped index."""
        if self._value_store is None:
            raise ValueError("persistent cas requires a value store")
        self._persistent = True

    @contextlib.contextmanager
    def suspend_persistent_cas(self) -> Generator[None]:
        """Encode without persistentCas descriptors. Result algebra
        documents must be self-contained: they have no blob-transfer
        companion frames, so their values fall back to inline/cas."""
        previous, self._persistent = self._persistent, False
        try:
            yield
        finally:
            self._persistent = previous

    def take_pending_store_blobs(self) -> list[tuple[str, bytes]]:
        """Drain the (digest, bytes) pairs the encodes since the last drain
        promised as persistentCas references. The caller must make them
        resolvable in the peer's store (BlobTransfer.ensure_peer_holds)
        before the frame that references them is sent, or drop them when
        the frame is abandoned. Callers drain synchronously after encoding
        (no await in between), so concurrent encoders on one event loop
        never see each other's pending blobs."""
        pending, self._pending_store = self._pending_store, []
        return pending

    def conversation_checkpoint(self) -> tuple[dict[str, bytes], int]:
        """Snapshot CAS retention so a multi-value frame can encode transactionally."""
        return dict(self._conversation), self._conversation_bytes

    def restore_conversation(self, checkpoint: tuple[dict[str, bytes], int]) -> None:
        """Roll back CAS retention after a frame that was never sent."""
        self._conversation, self._conversation_bytes = checkpoint

    def encode(
        self, value: Value, blobs: list[bytes], segments: list[SharedMemory]
    ) -> tuple[dict[str, object], TransferStat]:
        children = list_children(value)
        if children is not None:
            # Lists cross as recursive descriptors: each child is a full
            # value descriptor with its own payload transport, so child
            # envelopes (resource stubs, residency, cost) survive the
            # boundary exactly like top-level values (DESIGN 3.13).
            element_wires: list[dict[str, object]] = []
            total_size = 0
            total_ms = 0.0
            all_declared = True
            all_reused = bool(children)
            for child in children:
                child_wire, child_stat = self.encode(child, blobs, segments)
                element_wires.append(child_wire)
                total_size += child_stat.size_bytes
                total_ms += child_stat.codec_ms
                all_declared = all_declared and child_stat.declared_codec
                all_reused = all_reused and child_stat.reused
            wire: dict[str, object] = {
                "typeId": value.type_id,
                "fingerprint": value.fingerprint,
                "metaBlob": len(blobs),
                "elements": element_wires,
            }
            blobs.append(default_encode(dict(value.meta.entries)))
            return wire, TransferStat("list", total_size, total_ms, all_declared, all_reused)
        started = time.perf_counter()
        payload = value.payload
        reused = isinstance(payload, EncodedPayload) and payload.type_id == value.type_id
        if reused:
            encoded = cast(EncodedPayload, payload)
            if self._use_shm and self._shm_threshold <= encoded.size:
                codec_ms = (time.perf_counter() - started) * 1000.0
                meta_data = default_encode(dict(value.meta.entries))
                with encoded.borrow_data() as data:
                    segment = _create_segment(data)
                segments.append(segment)
                declared = (
                    self._registry.spec(value.type_id).declared_codec
                    if value.type_id in self._registry
                    else False
                )
                wire = {
                    "typeId": value.type_id,
                    "fingerprint": value.fingerprint,
                    "metaBlob": len(blobs),
                    "payload": {
                        "transport": "shm",
                        "segment": segment.name,
                        "size": encoded.size,
                    },
                }
                blobs.append(meta_data)
                return wire, TransferStat("shm", encoded.size, codec_ms, declared, True)
            data = encoded.data
        else:
            try:
                spec = self._registry.spec(value.type_id)
            except KeyError as exc:
                raise BoundaryError(
                    f"cannot send value of type '{value.type_id}': the type is not "
                    "registered in this process and the payload is not already encoded"
                ) from exc
            obj = payload.load()
            if self._use_shm and spec.prepare_buffer_encoding is not None:
                encoding = spec.prepare_buffer_encoding(obj)
                if self._shm_threshold <= encoding.size:
                    meta_data = default_encode(dict(value.meta.entries))
                    segment = _create_segment_from_encoding(encoding)
                    segments.append(segment)
                    codec_ms = (time.perf_counter() - started) * 1000.0
                    wire = {
                        "typeId": value.type_id,
                        "fingerprint": value.fingerprint,
                        "metaBlob": len(blobs),
                        "payload": {
                            "transport": "shm",
                            "segment": segment.name,
                            "size": encoding.size,
                        },
                    }
                    blobs.append(meta_data)
                    return wire, TransferStat(
                        "shm", encoding.size, codec_ms, spec.declared_codec, False
                    )
            data = spec.encode(obj)
        codec_ms = (time.perf_counter() - started) * 1000.0
        declared = (
            self._registry.spec(value.type_id).declared_codec
            if value.type_id in self._registry
            else False
        )
        wire: dict[str, object] = {
            "typeId": value.type_id,
            "fingerprint": value.fingerprint,
            "metaBlob": len(blobs),
        }
        blobs.append(default_encode(dict(value.meta.entries)))
        if self._use_shm and 0 < self._shm_threshold <= len(data):
            segment = _create_segment(data)
            segments.append(segment)
            wire["payload"] = {"transport": "shm", "segment": segment.name, "size": len(data)}
            transport = "shm"
        elif self._persistent and self._cas_threshold <= len(data):
            # The frame carries only the digest; the actual bytes are the
            # frame sender's obligation (take_pending_store_blobs) to land
            # in the peer's store before this frame crosses.
            digest = digest_bytes(data)
            self._pending_store.append((digest, data))
            wire["payload"] = {
                "transport": "persistentCas",
                "digest": digest,
                "size": len(data),
            }
            transport = "persistentCas"
        elif self._cas and self._cas_threshold <= len(data):
            digest = stable_hash([data])
            payload_wire: dict[str, object] = {
                "transport": "cas",
                "digest": digest,
                "size": len(data),
            }
            if digest not in self._conversation:
                # First crossing: the bytes ride along. The sender decides
                # whether both sides retain them (fits the budget) and the
                # receiver mirrors that decision, so the stores never
                # disagree about which digests resolve.
                retained = self._conversation_bytes + len(data) <= self._cas_budget
                payload_wire["blob"] = len(blobs)
                payload_wire["retained"] = retained
                blobs.append(data)
                if retained:
                    self._retain(digest, data)
            wire["payload"] = payload_wire
            transport = "cas"
        else:
            wire["payload"] = {"transport": "inline", "blob": len(blobs)}
            blobs.append(data)
            transport = "inline"
        return wire, TransferStat(transport, len(data), codec_ms, declared, reused)

    def decode(
        self,
        wire: Mapping[str, Any],
        blobs: Sequence[bytes],
        consumed_segments: list[str],
    ) -> tuple[Value, TransferStat]:
        """Decode one value descriptor. Shared-memory payloads retain a
        read-only receiver mapping while their segment names are acknowledged;
        object decode and owned-byte materialization stay lazy."""
        type_id = str(wire["typeId"])
        fingerprint = str(wire["fingerprint"])
        meta_raw = default_decode(blobs[int(wire["metaBlob"])])
        if not isinstance(meta_raw, Mapping):
            raise BoundaryError(f"value meta must decode to a mapping, got {type(meta_raw)}")
        meta_entries = {
            str(key): val for key, val in cast("Mapping[object, object]", meta_raw).items()
        }
        meta = ValueMeta(meta_entries)
        if "elements" in wire:
            # Recursive list descriptor (see encode): rebuild the child
            # envelopes, then the list envelope around them. Identity crosses
            # verbatim like every fingerprint does (hazard H4).
            if parse_list_type_id(type_id) is None:
                raise BoundaryError(f"elements on a non-list type id: {type_id!r}")
            child_values: list[Value] = []
            total_size = 0
            total_ms = 0.0
            all_declared = True
            elements = cast("Sequence[Mapping[str, Any]]", wire["elements"])
            for child_wire in elements:
                child, child_stat = self.decode(child_wire, blobs, consumed_segments)
                child_values.append(child)
                total_size += child_stat.size_bytes
                total_ms += child_stat.codec_ms
                all_declared = all_declared and child_stat.declared_codec
            list_value = Value(
                type_id=type_id,
                fingerprint=fingerprint,
                meta=meta,
                payload=ListPayload(tuple(child_values)),
            )
            return list_value, TransferStat("list", total_size, total_ms, all_declared, False)
        payload_wire = cast("Mapping[str, Any]", wire["payload"])
        transport = str(payload_wire["transport"])
        started = time.perf_counter()
        data = b""
        segment: SharedMemory | None = None
        segment_view: memoryview | None = None
        size_bytes: int
        if transport == "shm":
            if not self._accept_shm:
                raise BoundaryError(
                    "shm payloads are not accepted over this boundary; "
                    "the peers do not share memory"
                )
            name = str(payload_wire["segment"])
            raw_size = payload_wire["size"]
            if type(raw_size) is not int:
                raise BoundaryError("shared-memory payload size must be an integer")
            size_bytes = raw_size
            segment, segment_view = _open_segment(name, size_bytes)
            consumed_segments.append(name)
        elif transport == "cas":
            if not self._cas:
                raise BoundaryError("cas payloads were not negotiated over this boundary")
            digest = str(payload_wire["digest"])
            raw_size = payload_wire["size"]
            if type(raw_size) is not int or raw_size < 0:
                raise BoundaryError("cas payload size must be a non-negative integer")
            size_bytes = raw_size
            if "blob" in payload_wire:
                data = blobs[int(payload_wire["blob"])]
                if len(data) != size_bytes or stable_hash([data]) != digest:
                    raise BoundaryError(f"cas payload does not match its digest ({digest})")
                # Mirror the sender's retention decision verbatim (missing
                # means retained, the pre-budget wire shape).
                if bool(payload_wire.get("retained", True)):
                    self._retain(digest, data)
            else:
                cached = self._conversation.get(digest)
                if cached is None:
                    raise BoundaryError(
                        "peer referenced a cas digest that never crossed "
                        f"this conversation: {digest}"
                    )
                if len(cached) != size_bytes:
                    raise BoundaryError(
                        f"cas payload has size {len(cached)}, expected {size_bytes}"
                    )
                data = cached
        elif transport == "persistentCas":
            if not self._persistent:
                raise BoundaryError("persistentCas payloads were not negotiated over this boundary")
            store = self._value_store
            assert store is not None  # enable_persistent_cas required one
            digest = str(payload_wire["digest"])
            raw_size = payload_wire["size"]
            if type(raw_size) is not int or raw_size < 0:
                raise BoundaryError("persistentCas payload size must be a non-negative integer")
            size_bytes = raw_size
            stored = store.get(digest)
            if stored is None:
                raise PersistentBlobMissing(
                    f"persistentCas blob {digest} is not in the local value store"
                )
            if len(stored) != size_bytes:
                raise BoundaryError(
                    f"persistentCas blob {digest} has size {len(stored)}, expected {size_bytes}"
                )
            data = stored
        elif transport == "inline":
            data = blobs[int(payload_wire["blob"])]
            size_bytes = len(data)
        else:
            raise BoundaryError(f"unknown payload transport: {transport!r}")
        spec = self._registry.spec(type_id) if type_id in self._registry else None
        if spec is not None and (
            spec.validate_encoded is not None or spec.validate_encoded_buffer is not None
        ):
            if segment_view is not None:
                validation_view = segment_view.toreadonly()
                if validation_view.nbytes != size_bytes:
                    sized_view = validation_view[:size_bytes]
                    validation_view.release()
                    validation_view = sized_view
                validation_data: bytes | memoryview = validation_view
            else:
                validation_view = None
                validation_data = data
            try:
                if spec.validate_encoded_buffer is not None:
                    if isinstance(validation_data, memoryview):
                        spec.validate_encoded_buffer(validation_data, meta.entries)
                    else:
                        with memoryview(validation_data) as inline_view:
                            spec.validate_encoded_buffer(inline_view, meta.entries)
                else:
                    assert spec.validate_encoded is not None
                    spec.validate_encoded(bytes(validation_data), meta.entries)
            except Exception as exc:
                if validation_view is not None:
                    validation_view.release()
                    validation_view = None
                if segment_view is not None:
                    segment_view.release()
                    segment_view = None
                    assert segment is not None
                    segment.close()
                    segment = None
                raise EncodedPayloadValidationError(
                    f"encoded payload validation failed for type '{type_id}': {exc}"
                ) from exc
            if validation_view is not None:
                validation_view.release()
        codec_ms = (time.perf_counter() - started) * 1000.0
        decoder: Callable[[bytes], object] | None = None
        declared = False
        if spec is not None:
            decoder = spec.decode
            declared = spec.declared_codec
        value = Value(
            type_id=type_id,
            fingerprint=fingerprint,
            meta=meta,
            payload=(
                EncodedPayload.from_buffer(
                    type_id,
                    segment_view,
                    decoder,
                    transport,
                    cast(SharedMemory, segment).close,
                    size_bytes,
                )
                if segment_view is not None
                else EncodedPayload(type_id, data, decoder, transport)
            ),
        )
        return value, TransferStat(transport, size_bytes, codec_ms, declared, False)


def _stat_to_wire(stat: TransferStat) -> dict[str, object]:
    wire: dict[str, object] = {
        "transport": stat.transport,
        "sizeBytes": stat.size_bytes,
        "codecMs": stat.codec_ms,
        "declaredCodec": stat.declared_codec,
        "reused": stat.reused,
    }
    if stat.network_bytes:
        wire["networkBytes"] = stat.network_bytes
        wire["transferMs"] = stat.transfer_ms
    return wire


def stat_from_wire(wire: Mapping[str, Any]) -> TransferStat:
    return TransferStat(
        transport=str(wire["transport"]),
        size_bytes=int(wire["sizeBytes"]),
        codec_ms=float(wire["codecMs"]),
        declared_codec=bool(wire["declaredCodec"]),
        reused=bool(wire["reused"]),
        network_bytes=int(wire.get("networkBytes", 0)),
        transfer_ms=float(wire.get("transferMs", 0.0)),
    )


def encode_invocation(
    codec: ValueCodec, invocation: Invocation
) -> tuple[dict[str, object], list[bytes], list[SharedMemory], dict[str, TransferStat]]:
    """Build an ``invoke`` frame. The effective schema crosses in the schema
    wire format - the same format the hello handshake and the server API use
    (hazard H1: one interface description everywhere)."""
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    stats: dict[str, TransferStat] = {}
    inputs_wire: dict[str, object] = {}
    header: dict[str, object] = {
        "type": "invoke",
        "invocationId": invocation.invocation_id,
        "jobRef": invocation.job_ref or invocation.invocation_id,
        "attemptId": invocation.attempt_id,
        "nodeId": invocation.node_id,
        "nodeType": invocation.node_type,
        "effectiveSchema": schema_to_wire(
            invocation.effective_schema, wire_version=max(SCHEMA_WIRE_SERVE_VERSIONS)
        ),
        "outputMembers": [
            [family_id, list(suffixes)] for family_id, suffixes in invocation.output_members
        ],
        "inputs": inputs_wire,
    }
    if invocation.arm is not None:
        header["arm"] = invocation.arm
    if invocation.expected_execution_identity is not None:
        header["expectedExecutionIdentity"] = invocation.expected_execution_identity
    if invocation.extension_snapshot_digest is not None:
        header["extensionSnapshotDigest"] = invocation.extension_snapshot_digest
    if invocation.connected_undemanded_inputs:
        header["connectedUndemandedInputs"] = list(invocation.connected_undemanded_inputs)
    if invocation.export_snapshot is not None:
        header["exportSnapshot"] = {
            "prompt": dict(invocation.export_snapshot.prompt),
            "extraPnginfo": (
                None
                if invocation.export_snapshot.extra_pnginfo is None
                else dict(invocation.export_snapshot.extra_pnginfo)
            ),
        }
    header["fp8Matmul"] = invocation.fp8_matmul
    if invocation.diffusion_dtype is not None:
        header["componentDtypes"] = {
            "diffusion": invocation.diffusion_dtype,
            "textEncoder": invocation.text_dtype,
            "vae": invocation.vae_dtype,
        }
    header["attentionPolicy"] = invocation.attention_policy
    if invocation.preview_mode != "off":
        header["previewMode"] = invocation.preview_mode
        if invocation.preview_animation != "ring":
            header["previewAnimation"] = invocation.preview_animation
    if invocation.attention_route_token is not None:
        from dinkster_protocol import attention_route_token_to_wire

        header["attentionRouteToken"] = attention_route_token_to_wire(
            invocation.attention_route_token
        )
    if invocation.media_sources:
        header["mediaSources"] = [
            {
                "digest": source.digest,
                "kind": source.kind,
                "mediaType": source.media_type,
                "extension": source.extension,
                "byteSize": source.byte_size,
            }
            for source in invocation.media_sources
        ]
    conversation_checkpoint = codec.conversation_checkpoint()
    try:
        for input_id, value in invocation.inputs.items():
            wire, stat = codec.encode(value, blobs, segments)
            inputs_wire[input_id] = wire
            stats[input_id] = stat
    except Exception:
        for segment in segments:
            release_segment(segment)
        codec.restore_conversation(conversation_checkpoint)
        codec.take_pending_store_blobs()
        raise
    return header, blobs, segments, stats


def decode_invocation(
    codec: ValueCodec, header: Mapping[str, Any], blobs: Sequence[bytes], consumed: list[str]
) -> Invocation:
    from dinkster_protocol import (
        attention_route_token_from_wire,
        resolve_attention_runtime_status,
        validate_attention_policy,
    )

    job_ref_raw = header.get("jobRef")
    if type(job_ref_raw) is not str or not job_ref_raw:
        raise BoundaryError("invocation jobRef must be a non-empty string")
    attempt_id_raw = header.get("attemptId")
    if type(attempt_id_raw) is not int or attempt_id_raw < 1:
        raise BoundaryError("invocation attemptId must be a positive integer")
    inputs: dict[str, Value] = {}
    for input_id, wire in cast("Mapping[str, Any]", header["inputs"]).items():
        try:
            value, _ = codec.decode(cast("Mapping[str, Any]", wire), blobs, consumed)
        except PersistentBlobMissing as exc:
            raise PersistentBlobMissing(
                f"input '{input_id}' of node '{header.get('nodeId')}': {exc}"
            ) from exc
        inputs[str(input_id)] = value
    output_members = tuple(
        (str(family_id), tuple(str(suffix) for suffix in suffixes))
        for family_id, suffixes in header.get("outputMembers", ())
    )
    arm_raw = header.get("arm")
    if "arm" in header and (not isinstance(arm_raw, str) or not arm_raw):
        raise BoundaryError("invocation arm must be a non-empty string when present")
    identity_raw = header.get("expectedExecutionIdentity")
    if "expectedExecutionIdentity" in header and (
        not isinstance(identity_raw, str) or not identity_raw
    ):
        raise BoundaryError(
            "invocation expectedExecutionIdentity must be a non-empty string when present"
        )
    extension_digest_raw = header.get("extensionSnapshotDigest")
    if "extensionSnapshotDigest" in header and (
        not is_extension_snapshot_digest(extension_digest_raw)
    ):
        raise BoundaryError(
            "invocation extensionSnapshotDigest must be a sha256 digest when present"
        )
    export_snapshot_raw = header.get("exportSnapshot")
    export_snapshot = None
    if "exportSnapshot" in header:
        if not isinstance(export_snapshot_raw, Mapping):
            raise BoundaryError("invocation exportSnapshot must contain prompt and extraPnginfo")
        export_snapshot_wire = cast("Mapping[str, Any]", export_snapshot_raw)
        if set(export_snapshot_wire) != {
            "prompt",
            "extraPnginfo",
        }:
            raise BoundaryError("invocation exportSnapshot must contain prompt and extraPnginfo")
        try:
            export_snapshot = ExportSnapshot(
                prompt=cast("Mapping[str, object]", export_snapshot_wire["prompt"]),
                extra_pnginfo=cast(
                    "Mapping[str, object] | None", export_snapshot_wire["extraPnginfo"]
                ),
            )
        except ValueError as exc:
            raise BoundaryError(f"invalid invocation exportSnapshot: {exc}") from exc
    fp8_matmul_raw = header.get("fp8Matmul", False)
    if not isinstance(fp8_matmul_raw, bool):
        raise BoundaryError("invocation fp8Matmul must be a boolean")
    if fp8_matmul_raw and identity_raw is None:
        raise BoundaryError("invocation fp8Matmul requires expectedExecutionIdentity")
    component_dtypes_raw = header.get("componentDtypes")
    component_dtypes: tuple[str | None, str | None, str | None] = (None, None, None)
    if component_dtypes_raw is not None:
        if not isinstance(component_dtypes_raw, Mapping) or set(component_dtypes_raw) != {
            "diffusion",
            "textEncoder",
            "vae",
        }:
            raise BoundaryError("invocation componentDtypes must contain all components")
        values = cast("Mapping[str, object]", component_dtypes_raw)
        component_dtypes = (
            cast("str", values["diffusion"]),
            cast("str", values["textEncoder"]),
            cast("str", values["vae"]),
        )
        if identity_raw is None or not all(
            isinstance(dtype, str) and dtype for dtype in component_dtypes
        ):
            raise BoundaryError(
                "invocation componentDtypes must be non-empty strings and require identity"
            )
    try:
        attention_policy = validate_attention_policy(header.get("attentionPolicy", "auto"))
        attention_token = (
            attention_route_token_from_wire(header["attentionRouteToken"])
            if "attentionRouteToken" in header
            else None
        )
        resolve_attention_runtime_status(attention_policy, attention_token)
    except (TypeError, ValueError) as exc:
        raise BoundaryError(f"malformed invocation attention routing: {exc}") from exc
    try:
        preview_mode = validate_preview_mode(header.get("previewMode", "off"))
    except ValueError as exc:
        raise BoundaryError(f"malformed invocation preview mode: {exc}") from exc
    try:
        preview_animation = validate_preview_animation(header.get("previewAnimation", "ring"))
    except ValueError as exc:
        raise BoundaryError(f"malformed invocation preview animation: {exc}") from exc
    undemanded_raw = cast("object", header.get("connectedUndemandedInputs", ()))
    if (
        not isinstance(undemanded_raw, (list, tuple))
        or not all(isinstance(item, str) for item in cast("Sequence[object]", undemanded_raw))
        or len(set(cast("Sequence[object]", undemanded_raw)))
        != len(cast("Sequence[object]", undemanded_raw))
    ):
        raise BoundaryError("invocation connectedUndemandedInputs must be unique strings")
    media_sources_raw = cast("object", header.get("mediaSources", ()))
    if not isinstance(media_sources_raw, (list, tuple)):
        raise BoundaryError("invocation mediaSources must be a list")
    media_sources: list[MediaSourceAuthority] = []
    for raw_object in cast("Sequence[object]", media_sources_raw):
        if not isinstance(raw_object, Mapping) or set(
            cast("Mapping[object, object]", raw_object)
        ) != {
            "digest",
            "kind",
            "mediaType",
            "extension",
            "byteSize",
        }:
            raise BoundaryError("invocation mediaSources entries have invalid fields")
        raw = cast("Mapping[str, object]", raw_object)
        try:
            media_sources.append(
                MediaSourceAuthority(
                    digest=cast("str", raw["digest"]),
                    kind=cast("str", raw["kind"]),
                    media_type=cast("str", raw["mediaType"]),
                    extension=cast("str", raw["extension"]),
                    byte_size=cast("int", raw["byteSize"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise BoundaryError(f"invalid invocation mediaSources entry: {exc}") from exc
    try:
        return Invocation(
            invocation_id=str(header["invocationId"]),
            node_id=str(header["nodeId"]),
            node_type=str(header["nodeType"]),
            inputs=inputs,
            effective_schema=schema_from_wire(
                dict(cast("Mapping[str, Any]", header["effectiveSchema"]))
            ),
            job_ref=job_ref_raw,
            attempt_id=attempt_id_raw,
            output_members=output_members,
            connected_undemanded_inputs=tuple(cast("Sequence[str]", undemanded_raw)),
            arm=arm_raw,
            expected_execution_identity=identity_raw,
            extension_snapshot_digest=extension_digest_raw,
            export_snapshot=export_snapshot,
            fp8_matmul=fp8_matmul_raw,
            diffusion_dtype=component_dtypes[0],
            text_dtype=component_dtypes[1],
            vae_dtype=component_dtypes[2],
            attention_policy=attention_policy,
            attention_route_token=attention_token,
            media_sources=tuple(media_sources),
            preview_mode=preview_mode,
            preview_animation=preview_animation,
        )
    except ValueError as exc:
        raise BoundaryError(f"invalid invocation: {exc}") from exc


def encode_result(
    codec: ValueCodec, result: InvocationResult, invocation_id: str, execute_ms: float
) -> tuple[dict[str, object], list[bytes], list[SharedMemory]]:
    from .saved_artifacts import MAX_SAVED_ARTIFACTS, validate_candidate

    if len(result.artifact_candidates) > MAX_SAVED_ARTIFACTS:
        raise BoundaryError(f"result artifactCandidates exceeds {MAX_SAVED_ARTIFACTS} entries")
    try:
        for candidate in result.artifact_candidates:
            validate_candidate(candidate)
    except AssetError as exc:
        raise BoundaryError(f"invalid result artifact candidate: {exc}") from exc
    header: dict[str, object] = {
        "type": "result",
        "invocationId": invocation_id,
        "executeMs": execute_ms,
    }
    if result.artifact_candidates:
        header["artifactCandidates"] = [
            {
                "nodeId": candidate.node_id,
                "filename": candidate.filename,
                "subfolder": candidate.subfolder,
                "type": candidate.folder_type,
            }
            for candidate in result.artifact_candidates
        ]
    if result.artifacts:
        raise BoundaryError("workers cannot send authoritative saved artifact descriptors")
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    if result.error is not None:
        header["error"] = {
            "nodeId": result.error.node_id,
            "nodeType": result.error.node_type,
            "message": result.error.message,
            "traceback": result.error.traceback,
            "hints": [
                {
                    "code": hint.code,
                    "message": hint.message,
                    **({"suggestion": hint.suggestion} if hint.suggestion is not None else {}),
                }
                for hint in result.error.hints
            ],
        }
        return header, blobs, segments
    outputs_wire: dict[str, object] = {}
    stats_wire: dict[str, object] = {}
    conversation_checkpoint = codec.conversation_checkpoint()
    try:
        for output_id, value in (result.outputs or {}).items():
            wire, stat = codec.encode(value, blobs, segments)
            outputs_wire[output_id] = wire
            stats_wire[output_id] = _stat_to_wire(stat)
    except Exception:
        for segment in segments:
            release_segment(segment)
        codec.restore_conversation(conversation_checkpoint)
        codec.take_pending_store_blobs()
        raise
    header["outputs"] = outputs_wire
    header["outputStats"] = stats_wire
    return header, blobs, segments


def decode_result_artifact_candidates(
    header: Mapping[str, Any],
) -> tuple[SavedArtifactCandidate, ...]:
    """Bound raw worker reports before host filesystem validation."""
    if "artifacts" in header:
        raise BoundaryError("worker sent a forged authoritative artifact descriptor")
    raw_candidates = header.get("artifactCandidates", [])
    if type(raw_candidates) is not list:
        raise BoundaryError("result artifactCandidates must be a list")
    from .saved_artifacts import (
        MAX_SAVED_ARTIFACTS,
        validate_candidate,
    )

    if len(raw_candidates) > MAX_SAVED_ARTIFACTS:
        raise BoundaryError(f"result artifactCandidates exceeds {MAX_SAVED_ARTIFACTS} entries")
    candidates: list[SavedArtifactCandidate] = []
    for item in cast("list[object]", raw_candidates):
        if not isinstance(item, Mapping) or set(item) != {
            "nodeId",
            "filename",
            "subfolder",
            "type",
        }:
            raise BoundaryError("result artifact candidate has invalid fields")
        wire = cast("Mapping[str, object]", item)
        try:
            candidate = SavedArtifactCandidate(
                node_id=cast("str", wire["nodeId"]),
                filename=cast("str", wire["filename"]),
                subfolder=cast("str", wire["subfolder"]),
                folder_type=cast("str", wire["type"]),
            )
            validate_candidate(candidate)
            candidates.append(candidate)
        except (AssetError, TypeError, ValueError) as exc:
            raise BoundaryError(f"invalid result artifact candidate: {exc}") from exc
    return tuple(candidates)


def encode_invocation_outcome(
    codec: ValueCodec,
    outcome: InvocationOutcome,
    invocation_id: str,
    execute_ms: float,
    *,
    negotiated_capabilities: Set[str],
) -> tuple[dict[str, object], list[bytes], list[SharedMemory]]:
    """Encode the dormant result algebra without changing legacy results."""
    from dinkster_protocol import (
        RESULT_ALGEBRA_CAPABILITY,
        RESULT_ALGEBRA_MAX_COUNT,
        RESULT_ALGEBRA_MAX_DOCUMENT_BYTES,
        RESULT_ALGEBRA_VERSION,
    )

    if RESULT_ALGEBRA_CAPABILITY not in negotiated_capabilities:
        raise BoundaryError("result algebra capability was not negotiated")
    if type(invocation_id) is not str:
        raise BoundaryError("typed result invocation id must be a string")
    _finite_number(execute_ms, "typed result executeMs")
    values: list[object] = []
    stats: list[object] = []
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    value_budget = [RESULT_ALGEBRA_MAX_COUNT]
    conversation_checkpoint = codec.conversation_checkpoint()
    try:
        with codec.suspend_persistent_cas():
            document = _outcome_to_document(
                outcome, codec, values, stats, blobs, segments, value_budget
            )
        _validate_json_shape(document)
        document_json = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if len(document_json.encode("ascii")) > RESULT_ALGEBRA_MAX_DOCUMENT_BYTES:
            raise BoundaryError("result algebra document is excessive")
        header: dict[str, object] = {
            "type": "result",
            "invocationId": invocation_id,
            "executeMs": execute_ms,
            "resultAlgebra": {
                "capability": RESULT_ALGEBRA_CAPABILITY,
                "version": RESULT_ALGEBRA_VERSION,
                "document": document_json,
                "values": values,
                "valueStats": stats,
            },
        }
        framed = dict(header)
        framed["blobs"] = [len(blob) for blob in blobs]
        if len(json.dumps(framed, separators=(",", ":")).encode("utf-8")) > _MAX_HEADER_BYTES:
            raise BoundaryError("result algebra frame header is excessive")
    except Exception as exc:
        for segment in segments:
            release_segment(segment)
        codec.restore_conversation(conversation_checkpoint)
        raise BoundaryError("result algebra document cannot be encoded") from exc
    return header, blobs, segments


def _ref_to_document(value: object) -> dict[str, object]:
    from dinkster_protocol import CurrentOutputRef, LocalOutputRef

    if isinstance(value, CurrentOutputRef):
        return {"kind": "outputRef", "scope": "current", "outputId": value.output_id}
    if isinstance(value, LocalOutputRef):
        return {
            "kind": "outputRef",
            "scope": "local",
            "localNodeId": value.local_node_id,
            "outputId": value.output_id,
        }
    raise BoundaryError("invalid output reference")


def _metadata_ref_to_document(value: object) -> dict[str, object]:
    from dinkster_protocol import CurrentNodeRef, LocalNodeRef

    if isinstance(value, CurrentNodeRef):
        return {"kind": "nodeRef", "scope": "current"}
    if isinstance(value, LocalNodeRef):
        return {"kind": "nodeRef", "scope": "local", "localNodeId": value.local_node_id}
    raise BoundaryError("invalid node metadata reference")


def _outcome_to_document(
    outcome: object,
    codec: ValueCodec,
    values: list[object],
    stats: list[object],
    blobs: list[bytes],
    segments: list[SharedMemory],
    value_budget: list[int],
) -> dict[str, object]:
    from dinkster_protocol import (
        BlockedOutput,
        DirectReturn,
        ExpandedReturn,
        InvocationFailure,
        InvocationReturn,
        LiteralInput,
        PresentOutput,
    )

    if isinstance(outcome, InvocationFailure):
        error = outcome.error
        return {
            "kind": "error",
            "error": {
                "hints": [
                    {"code": h.code, "message": h.message, "suggestion": h.suggestion}
                    for h in error.hints
                ],
                "message": error.message,
                "nodeId": error.node_id,
                "nodeType": error.node_type,
                "traceback": error.traceback,
            },
        }
    if not isinstance(outcome, InvocationReturn):
        raise BoundaryError("invalid invocation outcome")

    def binding(item: object) -> dict[str, object]:
        if isinstance(item, PresentOutput):
            _preflight_value_tree(item.value, value_budget)
            wire, stat = codec.encode(item.value, blobs, segments)
            index = len(values)
            values.append(wire)
            stats.append(_stat_to_wire(stat))
            return {"kind": "present", "valueIndex": index}
        if isinstance(item, BlockedOutput):
            return {"kind": "blocked", "message": item.message}
        return _ref_to_document(item)

    def inputs(item: object) -> dict[str, object]:
        if isinstance(item, LiteralInput):
            return {"kind": "literal", "value": _literal_to_json(item.value)}
        return _ref_to_document(item)

    units = []
    for unit in outcome.batch.units:
        result: dict[str, object] = {
            "bindings": [{"id": key, "value": binding(value)} for key, value in unit.bindings],
            "kind": "direct",
        }
        if isinstance(unit, ExpandedReturn):
            result["kind"] = "expanded"
            expansion = unit.expansion
            result["expansion"] = {
                "nodes": [
                    {
                        "display": _metadata_ref_to_document(node.display)
                        if node.display
                        else None,
                        "inputs": [
                            {"id": key, "value": inputs(value)} for key, value in node.inputs
                        ],
                        "localId": node.local_id,
                        "nodeType": node.node_type,
                        "parent": _metadata_ref_to_document(node.parent) if node.parent else None,
                    }
                    for node in expansion.nodes
                ]
            }
        elif not isinstance(unit, DirectReturn):
            raise BoundaryError("invalid return unit")
        units.append(result)
    return {"batch": {"mode": outcome.batch.mode, "units": units}, "kind": "return"}


def _literal_to_json(value: object) -> object:
    from dinkster_protocol import JsonObject

    if isinstance(value, JsonObject):
        return {key: _literal_to_json(item) for key, item in value.fields}
    if isinstance(value, tuple):
        return [_literal_to_json(item) for item in value]
    return value


def _preflight_value_tree(root: Value, budget: list[int]) -> None:
    """Bound recursive ValueCodec work and prove canonical list envelopes."""
    from dinkster_protocol import RESULT_ALGEBRA_MAX_NESTING

    pending: list[tuple[Value, int]] = [(root, 0)]
    while pending:
        value, depth = pending.pop()
        budget[0] -= 1
        if budget[0] < 0 or depth > RESULT_ALGEBRA_MAX_NESTING:
            raise BoundaryError("result algebra value tree is excessive")
        children = list_children(value)
        element_type = parse_list_type_id(value.type_id)
        if (children is None) != (element_type is None):
            raise BoundaryError("result algebra value has a noncanonical list envelope")
        if children is not None:
            if any(child.type_id != element_type for child in children):
                raise BoundaryError("result algebra list child type does not match its envelope")
            pending.extend((child, depth + 1) for child in children)


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise BoundaryError(f"{name} must be a finite number")
    if type(value) is int and not -(2**63) <= value < 2**63:
        raise BoundaryError(f"{name} is outside signed 64-bit range")
    try:
        converted = float(cast("int | float", value))
    except (OverflowError, TypeError, ValueError) as exc:
        raise BoundaryError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted) or (converted == 0.0 and math.copysign(1.0, converted) < 0):
        raise BoundaryError(f"{name} must be finite and not negative zero")
    return converted


def _json_to_literal(value: object) -> JsonLiteral:
    from dinkster_protocol import JsonObject

    if isinstance(value, list):
        return tuple(_json_to_literal(item) for item in value)
    if isinstance(value, dict):
        object_value = cast("dict[object, object]", value)
        if not all(isinstance(key, str) for key in object_value):
            raise ValueError("literal object keys must be strings")
        return JsonObject(
            tuple(
                (cast(str, key), _json_to_literal(item))
                for key, item in sorted(object_value.items())
            )
        )
    return cast("JsonLiteral", value)


def decode_invocation_outcome(
    codec: ValueCodec,
    header: Mapping[str, Any],
    blobs: Sequence[bytes],
    consumed: list[str],
    *,
    negotiated_capabilities: Set[str],
) -> tuple[InvocationOutcome, dict[tuple[int, str], TransferStat]]:
    """Strictly decode a negotiated typed result frame."""
    from dinkster_protocol import (
        RESULT_ALGEBRA_CAPABILITY,
        RESULT_ALGEBRA_MAX_COUNT,
        RESULT_ALGEBRA_MAX_DOCUMENT_BYTES,
        RESULT_ALGEBRA_VERSION,
    )

    expected_header = {"type", "invocationId", "executeMs", "resultAlgebra"}
    if "blobs" in header:
        expected_header.add("blobs")
    execute_ms = header.get("executeMs")
    if (
        set(header) != expected_header
        or header.get("type") != "result"
        or type(header.get("invocationId")) is not str
    ):
        raise BoundaryError("invalid typed result frame")
    _finite_number(execute_ms, "typed result executeMs")
    if "blobs" in header:
        lengths = header["blobs"]
        if (
            type(lengths) is not list
            or len(lengths) != len(blobs)
            or any(type(length) is not int or length < 0 for length in lengths)
            or list(lengths) != [len(blob) for blob in blobs]
        ):
            raise BoundaryError("typed result frame has invalid blob lengths")
    if RESULT_ALGEBRA_CAPABILITY not in negotiated_capabilities:
        raise BoundaryError("result algebra capability was not negotiated")
    algebra = header["resultAlgebra"]
    if not isinstance(algebra, Mapping) or set(algebra) != {
        "capability",
        "version",
        "document",
        "values",
        "valueStats",
    }:
        raise BoundaryError("invalid resultAlgebra fields")
    if (
        type(algebra["capability"]) is not str
        or algebra["capability"] != RESULT_ALGEBRA_CAPABILITY
        or type(algebra["version"]) is not int
        or algebra["version"] != RESULT_ALGEBRA_VERSION
    ):
        raise BoundaryError("unsupported result algebra capability or version")
    text = algebra["document"]
    try:
        if type(text) is not str or len(text.encode("ascii")) > RESULT_ALGEBRA_MAX_DOCUMENT_BYTES:
            raise BoundaryError("invalid result algebra document")
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        _validate_json_shape(document)
    except (
        ValueError,
        TypeError,
        json.JSONDecodeError,
        RecursionError,
        OverflowError,
        UnicodeError,
    ) as exc:
        raise BoundaryError("invalid result algebra JSON") from exc
    try:
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise BoundaryError("invalid result algebra JSON") from exc
    if canonical != text:
        raise BoundaryError("result algebra document is not canonical")
    values = algebra["values"]
    stats = algebra["valueStats"]
    if (
        type(values) is not list
        or type(stats) is not list
        or len(values) != len(stats)
        or len(values) > RESULT_ALGEBRA_MAX_COUNT
    ):
        raise BoundaryError("invalid result algebra value descriptors")
    descriptor_budget = [RESULT_ALGEBRA_MAX_COUNT]
    used_blobs: set[int] = set()
    for index, wire in enumerate(values):
        if not isinstance(wire, Mapping) or not isinstance(stats[index], Mapping):
            raise BoundaryError("invalid result algebra value descriptor")
        _validate_value_descriptor(wire, blobs, used_blobs, budget=descriptor_budget)
        _typed_stat_from_wire(cast("Mapping[str, Any]", stats[index]))
    if used_blobs != set(range(len(blobs))):
        raise BoundaryError("result algebra has unused or duplicate blobs")
    placeholder = Value(
        type_id="result.preflight",
        fingerprint="result.preflight",
        meta=ValueMeta(),
        payload=EncodedPayload("result.preflight", b"", None, "inline"),
    )
    try:
        _, preflight_used, _ = _document_to_outcome(document, [placeholder] * len(values))
    except (KeyError, TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise BoundaryError("malformed result algebra document") from exc
    if preflight_used != set(range(len(values))):
        raise BoundaryError("result algebra has unused or duplicate value descriptors")
    decoded: list[Value] = []
    for wire in values:
        try:
            value, _ = codec.decode(cast("Mapping[str, Any]", wire), blobs, consumed)
        except (BoundaryError, KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise BoundaryError("invalid result algebra value descriptor") from exc
        decoded.append(value)
    try:
        outcome, used, locations = _document_to_outcome(document, decoded)
    except (KeyError, TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise BoundaryError("malformed result algebra document") from exc
    if used != set(range(len(decoded))):
        raise BoundaryError("result algebra has unused or duplicate value descriptors")
    return outcome, {
        location: _typed_stat_from_wire(cast("Mapping[str, Any]", stats[index]))
        for index, location in locations.items()
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _validate_json_shape(value: object) -> None:
    """Enforce cumulative canonical-document limits before typed decoding."""
    from dinkster_protocol import (
        RESULT_ALGEBRA_MAX_COUNT,
        RESULT_ALGEBRA_MAX_NESTING,
    )

    count = 0
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > RESULT_ALGEBRA_MAX_COUNT * 16:
            raise ValueError("result algebra document has excessive items")
        if depth > RESULT_ALGEBRA_MAX_NESTING:
            raise ValueError("result algebra document has excessive nesting")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            item.encode("utf-8")
        elif type(item) is int:
            if item < -(2**63) or item > 2**63 - 1:
                raise ValueError("JSON integer is outside signed 64-bit range")
        elif type(item) is float:
            if not math.isfinite(item) or (item == 0.0 and math.copysign(1.0, item) < 0):
                raise ValueError("JSON float is not canonical and finite")
        elif item is not None and type(item) is not bool:
            raise ValueError("unsupported JSON value")


def _blob_index(value: object, blobs: Sequence[bytes], used: set[int]) -> None:
    if type(value) is not int or value < 0 or value >= len(blobs) or value in used:
        raise BoundaryError("result algebra has invalid or duplicate blob index")
    used.add(value)


def _validate_value_descriptor(
    wire: Mapping[str, Any],
    blobs: Sequence[bytes],
    used: set[int],
    *,
    budget: list[int],
    depth: int = 0,
    expected_type: str | None = None,
) -> None:
    """Validate the exact ValueCodec wire shape before invoking its permissive decoder."""
    from dinkster_protocol import RESULT_ALGEBRA_MAX_COUNT, RESULT_ALGEBRA_MAX_NESTING

    budget[0] -= 1
    if budget[0] < 0 or depth > RESULT_ALGEBRA_MAX_NESTING:
        raise BoundaryError("result algebra value descriptor nesting is excessive")
    if (
        type(wire) is not dict
        or type(wire.get("typeId")) is not str
        or type(wire.get("fingerprint")) is not str
    ):
        raise BoundaryError("invalid result algebra value descriptor")
    if expected_type is not None and wire["typeId"] != expected_type:
        raise BoundaryError("result algebra list child type does not match its envelope")
    _blob_index(wire.get("metaBlob"), blobs, used)
    element_type = parse_list_type_id(wire["typeId"])
    if "elements" in wire:
        if element_type is None or set(wire) != {
            "typeId",
            "fingerprint",
            "metaBlob",
            "elements",
        }:
            raise BoundaryError("invalid list value descriptor fields")
        elements = wire["elements"]
        if type(elements) is not list or len(elements) > RESULT_ALGEBRA_MAX_COUNT:
            raise BoundaryError("invalid list value descriptor elements")
        for child in elements:
            if type(child) is not dict:
                raise BoundaryError("invalid list child value descriptor")
            _validate_value_descriptor(
                child,
                blobs,
                used,
                budget=budget,
                depth=depth + 1,
                expected_type=element_type,
            )
        return
    if element_type is not None or set(wire) != {"typeId", "fingerprint", "metaBlob", "payload"}:
        raise BoundaryError("invalid value descriptor fields")
    payload = wire["payload"]
    if type(payload) is not dict or type(payload.get("transport")) is not str:
        raise BoundaryError("invalid value payload descriptor")
    transport = payload["transport"]
    if transport == "inline":
        if set(payload) != {"transport", "blob"}:
            raise BoundaryError("invalid inline payload descriptor")
        _blob_index(payload["blob"], blobs, used)
    elif transport == "shm":
        if (
            set(payload) != {"transport", "segment", "size"}
            or type(payload["segment"]) is not str
            or type(payload["size"]) is not int
            or payload["size"] < 0
        ):
            raise BoundaryError("invalid shared-memory payload descriptor")
    elif transport == "cas":
        expected = {"transport", "digest", "size"}
        if "blob" in payload:
            expected |= {"blob", "retained"}
        if (
            set(payload) != expected
            or type(payload["digest"]) is not str
            or type(payload["size"]) is not int
            or payload["size"] < 0
            or ("retained" in payload and type(payload["retained"]) is not bool)
        ):
            raise BoundaryError("invalid CAS payload descriptor")
        if "blob" in payload:
            _blob_index(payload["blob"], blobs, used)
    else:
        raise BoundaryError("unknown value payload transport")


def _typed_stat_from_wire(wire: Mapping[str, Any]) -> TransferStat:
    if (
        type(wire) is not dict
        or set(wire) != {"transport", "sizeBytes", "codecMs", "declaredCodec", "reused"}
        or type(wire["transport"]) is not str
        or type(wire["sizeBytes"]) is not int
        or wire["sizeBytes"] < 0
        or type(wire["codecMs"]) not in (int, float)
        or type(wire["declaredCodec"]) is not bool
        or type(wire["reused"]) is not bool
    ):
        raise BoundaryError("invalid result algebra transfer stat")
    try:
        codec_ms = _finite_number(wire["codecMs"], "result algebra codecMs")
    except BoundaryError as exc:
        raise BoundaryError("invalid result algebra transfer stat") from exc
    if codec_ms < 0:
        raise BoundaryError("invalid result algebra transfer stat")
    return stat_from_wire(wire)


def _exact(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("unexpected fields")
    return cast("dict[str, Any]", value)


def _document_to_outcome(
    document: object, values: list[Value]
) -> tuple[InvocationOutcome, set[int], dict[int, tuple[int, str]]]:
    from dinkster_protocol import (
        BlockedOutput,
        CurrentNodeRef,
        CurrentOutputRef,
        DirectReturn,
        ErrorHint,
        ExpandedReturn,
        InvocationFailure,
        InvocationReturn,
        LiteralInput,
        LocalExpansion,
        LocalNode,
        LocalNodeRef,
        LocalOutputRef,
        NodeError,
        PresentOutput,
        ReturnBatch,
    )
    from dinkster_protocol.result_algebra import (
        InputBinding,
        NodeMetadataRef,
        OutputBinding,
        ReturnUnit,
    )

    if not isinstance(document, dict):
        raise ValueError("outcome document must be an object")
    root = cast("dict[str, Any]", document)
    if root.get("kind") == "error":
        root = _exact(root, {"kind", "error"})
        error = _exact(root["error"], {"hints", "message", "nodeId", "nodeType", "traceback"})
        if (
            not all(
                type(error[key]) is str for key in ("message", "nodeId", "nodeType", "traceback")
            )
            or type(error["hints"]) is not list
        ):
            raise ValueError("invalid error primitives")
        hints = tuple(
            ErrorHint(**_exact(item, {"code", "message", "suggestion"})) for item in error["hints"]
        )
        if not all(
            type(h.code) is str
            and type(h.message) is str
            and (h.suggestion is None or type(h.suggestion) is str)
            for h in hints
        ):
            raise ValueError("invalid hints")
        return (
            InvocationFailure(
                NodeError(
                    error["nodeId"], error["nodeType"], error["message"], error["traceback"], hints
                )
            ),
            set(),
            {},
        )
    root = _exact(root, {"kind", "batch"})
    if root["kind"] != "return":
        raise ValueError("unknown outcome kind")
    batch = _exact(root["batch"], {"mode", "units"})
    used: set[int] = set()
    locations: dict[int, tuple[int, str]] = {}

    def ref(raw: object) -> CurrentOutputRef | LocalOutputRef:
        if not isinstance(raw, dict):
            raise ValueError("reference must be an object")
        item = cast("dict[str, Any]", raw)
        kind = item.get("kind")
        if kind != "outputRef":
            raise ValueError("unknown reference")
        if item.get("scope") == "current":
            item = _exact(item, {"kind", "scope", "outputId"})
            return CurrentOutputRef(item["outputId"])
        if item.get("scope") == "local":
            item = _exact(item, {"kind", "scope", "localNodeId", "outputId"})
            return LocalOutputRef(item["localNodeId"], item["outputId"])
        raise ValueError("unknown reference")

    def metadata_ref(raw: object) -> NodeMetadataRef:
        if not isinstance(raw, dict):
            raise ValueError("metadata reference must be an object")
        item = cast("dict[str, Any]", raw)
        if item.get("kind") != "nodeRef":
            raise ValueError("unknown metadata reference")
        if item.get("scope") == "current":
            _exact(item, {"kind", "scope"})
            return CurrentNodeRef()
        if item.get("scope") == "local":
            item = _exact(item, {"kind", "scope", "localNodeId"})
            return LocalNodeRef(item["localNodeId"])
        raise ValueError("unknown metadata scope")

    def binding(raw: object, unit_index: int, output_id: str) -> OutputBinding:
        if not isinstance(raw, dict):
            raise ValueError("binding must be an object")
        item = cast("dict[str, Any]", raw)
        kind = item.get("kind")
        if kind == "present":
            item = _exact(item, {"kind", "valueIndex"})
            index = item["valueIndex"]
            if type(index) is not int or index < 0 or index >= len(values) or index in used:
                raise ValueError("invalid descriptor index")
            used.add(index)
            locations[index] = (unit_index, output_id)
            return PresentOutput(values[index])
        if kind == "blocked":
            item = _exact(item, {"kind", "message"})
            return BlockedOutput(item["message"])
        return ref(item)

    units: list[ReturnUnit] = []
    if type(batch["units"]) is not list:
        raise ValueError("units must be an array")
    for unit_index, raw_unit in enumerate(batch["units"]):
        if not isinstance(raw_unit, dict):
            raise ValueError("return unit must be an object")
        unit = cast("dict[str, Any]", raw_unit)
        kind = unit.get("kind")
        expected = {"kind", "bindings"} if kind == "direct" else {"kind", "bindings", "expansion"}
        unit = _exact(unit, expected)
        bindings: tuple[tuple[str, OutputBinding], ...] = tuple(
            (
                cast(str, entry["id"]),
                binding(entry["value"], unit_index, cast(str, entry["id"])),
            )
            for entry in (_exact(item, {"id", "value"}) for item in unit["bindings"])
        )
        if kind == "direct":
            units.append(DirectReturn(bindings))
            continue
        if kind != "expanded":
            raise ValueError("unknown return kind")
        expansion = _exact(unit["expansion"], {"nodes"})
        nodes: list[LocalNode] = []
        for raw_node in expansion["nodes"]:
            node = _exact(raw_node, {"display", "inputs", "localId", "nodeType", "parent"})
            inputs_list: list[tuple[str, InputBinding]] = []
            for entry in (_exact(item, {"id", "value"}) for item in node["inputs"]):
                input_value = entry["value"]
                if isinstance(input_value, dict) and input_value.get("kind") == "literal":
                    literal = _exact(input_value, {"kind", "value"})
                    decoded_input = LiteralInput(_json_to_literal(literal["value"]))
                else:
                    decoded_input = ref(input_value)
                inputs_list.append((cast(str, entry["id"]), decoded_input))
            inputs = tuple(inputs_list)
            nodes.append(
                LocalNode(
                    node["localId"],
                    node["nodeType"],
                    inputs,
                    metadata_ref(node["parent"]) if node["parent"] is not None else None,
                    metadata_ref(node["display"]) if node["display"] is not None else None,
                )
            )
        local = LocalExpansion(tuple(nodes))
        units.append(ExpandedReturn(local, bindings))
    return InvocationReturn(ReturnBatch(batch["mode"], tuple(units))), used, locations


def decode_result_outputs(
    codec: ValueCodec, header: Mapping[str, Any], blobs: Sequence[bytes], consumed: list[str]
) -> tuple[dict[str, Value], dict[str, TransferStat]]:
    """Decode a successful result's outputs. Returned stats merge the
    sender's encode cost with this side's transfer cost per output."""
    outputs: dict[str, Value] = {}
    stats: dict[str, TransferStat] = {}
    sender_stats = cast("Mapping[str, Any]", header.get("outputStats", {}))
    for output_id, wire in cast("Mapping[str, Any]", header["outputs"]).items():
        try:
            value, local = codec.decode(cast("Mapping[str, Any]", wire), blobs, consumed)
        except PersistentBlobMissing as exc:
            raise PersistentBlobMissing(f"output '{output_id}': {exc}") from exc
        outputs[str(output_id)] = value
        remote_wire = sender_stats.get(output_id)
        if remote_wire is not None:
            remote = stat_from_wire(cast("Mapping[str, Any]", remote_wire))
            stats[str(output_id)] = TransferStat(
                transport=remote.transport,
                size_bytes=remote.size_bytes,
                codec_ms=remote.codec_ms + local.codec_ms,
                declared_codec=remote.declared_codec,
                reused=remote.reused,
                network_bytes=remote.network_bytes,
                transfer_ms=remote.transfer_ms,
            )
        else:
            stats[str(output_id)] = local
    return outputs, stats


def error_from_wire(wire: Mapping[str, Any]) -> NodeError:
    hints = tuple(
        ErrorHint(
            code=str(hint.get("code", "")),
            message=str(hint.get("message", "")),
            suggestion=(str(hint["suggestion"]) if hint.get("suggestion") is not None else None),
        )
        for hint in cast("Sequence[Mapping[str, Any]]", wire.get("hints", ()))
    )
    return NodeError(
        node_id=str(wire.get("nodeId", "")),
        node_type=str(wire.get("nodeType", "")),
        message=str(wire.get("message", "")),
        traceback=str(wire.get("traceback", "")),
        hints=hints,
    )
