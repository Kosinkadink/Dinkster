"""The CAS payload store and the caches built on it.

DiskCAS is verified bytes by digest - atomic, idempotent, corruption is
absence. DiskCacheStore makes cache entries survive the process: a manifest
written by one store instance is a hit in a fresh one (restart persistence),
resource stubs are structurally refused (they reference live process state),
and unknown types round-trip as relayable encoded envelopes. LayeredCache
composes stores fastest-first with promotion. The boundary's cas transport
sends each payload's bytes once per conversation - repeat crossings, either
direction, are digest-only.
"""

from __future__ import annotations

import asyncio
import mmap
import multiprocessing
import os
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import dinkster_workers.boundary as boundary_module
import pytest
from cas_lock_holder import hold_disk_cache_process_lock
from dinkster_assets import AssetError, digest_bytes
from dinkster_caches import (
    BudgetedDiskCAS,
    CASError,
    DiskCacheStore,
    DiskCAS,
    LayeredCache,
    MemoryLRUCache,
)
from dinkster_engine import (
    Engine,
    EngineEvent,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
)
from dinkster_schema import build_schemas
from dinkster_values import (
    CORE_COMBO,
    RESOURCE_ID_META_KEY,
    BufferEncoding,
    EncodedPayload,
    TypeRegistry,
    UnresolvablePayload,
    Value,
    ValueMeta,
    default_encode,
    register_core_types,
)
from dinkster_workers import BoundaryError, ValueCodec
from dinkster_workers.boundary import EncodedPayloadValidationError, encode_result, release_segment
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types
from test_engine import InProcessWorkerFactory, image_graph


class ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("metadata exploded")

    def __len__(self) -> int:
        return 1


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def test_shm_payload_survives_sender_ack_without_eager_copy() -> None:
    registry = make_registry()
    registry.register("test.large")
    value = registry.wrap("test.large", "x" * 1024)
    sender = ValueCodec(registry, shm_threshold=1)
    receiver = ValueCodec(registry, shm_threshold=1)
    blobs: list[bytes] = []
    segments = []
    wire, sent = sender.encode(value, blobs, segments)
    assert sent.transport == "shm"

    consumed: list[str] = []
    received, received_stat = receiver.decode(wire, blobs, consumed)
    assert received_stat.transport == "shm"
    assert consumed == [segments[0].name]
    for segment in segments:
        release_segment(segment)

    payload = received.payload
    assert isinstance(payload, EncodedPayload)
    assert not isinstance(object.__getattribute__(payload, "_encoded"), bytes)
    with payload.borrow_data() as view:
        assert view.readonly
        assert bytes(view) == registry.spec("test.large").encode("x" * 1024)
    assert received.resolve() == "x" * 1024


def test_shm_validator_borrows_only_declared_bytes_from_padded_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = make_registry()
    validated: list[tuple[bool, bool, bytes]] = []

    def validate(data: bytes | memoryview, _metadata: object) -> None:
        is_view = isinstance(data, memoryview)
        validated.append((is_view, data.readonly if is_view else False, bytes(data)))

    registry.register("test.validated-large", validate_encoded_buffer=validate)
    value = registry.wrap("test.validated-large", "mapped payload")
    sender = ValueCodec(registry, shm_threshold=1)
    receiver = ValueCodec(registry, shm_threshold=1)
    blobs: list[bytes] = []
    original_segments = []
    wire, _ = sender.encode(value, blobs, original_segments)
    declared_size = _payload_wire(wire)["size"]
    assert isinstance(declared_size, int)
    encoded_size = declared_size
    mapping_size = ((encoded_size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
    padded = SharedMemory(create=True, size=mapping_size)
    assert padded.buf is not None and original_segments[0].buf is not None
    padded.buf[:encoded_size] = original_segments[0].buf[:encoded_size]
    release_segment(original_segments[0])
    wire["payload"] = {
        "transport": "shm",
        "segment": padded.name,
        "size": encoded_size,
    }
    if not boundary_module._SHM_MAPPING_ROUNDS:  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(BoundaryError, match="has size"):
            receiver.decode(wire, blobs, [])
    monkeypatch.setattr(boundary_module, "_SHM_MAPPING_ROUNDS", True)

    consumed: list[str] = []
    received, _ = receiver.decode(wire, blobs, consumed)
    expected = registry.spec("test.validated-large").encode("mapped payload")
    assert validated == [(True, True, expected)]
    assert not isinstance(object.__getattribute__(received.payload, "_encoded"), bytes)
    release_segment(padded)

    relay_blobs: list[bytes] = []
    relay_segments = []
    _, relayed = receiver.encode(received, relay_blobs, relay_segments)
    assert relayed.reused is True
    assert received.resolve() == "mapped payload"
    for segment in relay_segments:
        release_segment(segment)


def test_shm_preserves_bytes_validator_contract() -> None:
    registry = make_registry()
    validated: list[bytes] = []

    def validate(data: bytes, _metadata: object) -> None:
        assert data.startswith(b"json:")
        validated.append(data)

    registry.register("test.legacy-validator", validate_encoded=validate)
    codec = ValueCodec(registry, shm_threshold=1)
    value = registry.wrap("test.legacy-validator", "payload")
    blobs: list[bytes] = []
    segments = []
    wire, _ = codec.encode(value, blobs, segments)

    received, _ = codec.decode(wire, blobs, [])
    assert validated == [registry.spec("test.legacy-validator").encode("payload")]
    assert isinstance(received.resolve(), str)
    for segment in segments:
        release_segment(segment)


def test_shm_validates_metadata_before_attaching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = make_registry()
    registry.register("test.large")
    codec = ValueCodec(registry, shm_threshold=1)
    value = registry.wrap("test.large", "payload")
    blobs: list[bytes] = []
    segments = []
    wire, _ = codec.encode(value, blobs, segments)
    meta_blob = wire["metaBlob"]
    assert isinstance(meta_blob, int)
    blobs[meta_blob] = default_encode({"resources": ExplodingMapping()})
    monkeypatch.setattr(
        boundary_module,
        "_attach_segment",
        lambda _name: pytest.fail("attached before metadata validation"),
    )

    with pytest.raises(RuntimeError, match="metadata exploded"):
        codec.decode(wire, blobs, [])
    for segment in segments:
        release_segment(segment)


@pytest.mark.parametrize("invalid_size", (-1, True, 1.5, "1"))
def test_shm_rejects_non_positive_size_before_attaching(
    monkeypatch: pytest.MonkeyPatch,
    invalid_size: object,
) -> None:
    registry = make_registry()
    codec = ValueCodec(registry, shm_threshold=1)
    value = registry.wrap(CORE_COMBO, "euler")
    blobs: list[bytes] = []
    segments = []
    wire, _ = codec.encode(value, blobs, segments)
    release_segment(segments[0])
    wire["payload"] = {
        "transport": "shm",
        "segment": "must-not-attach",
        "size": invalid_size,
    }
    monkeypatch.setattr(
        boundary_module,
        "_attach_segment",
        lambda _name: pytest.fail("invalid size attempted shared-memory attach"),
    )

    with pytest.raises(BoundaryError, match="size must be (positive|an integer)"):
        codec.decode(wire, blobs, [])


def test_relay_encodes_metadata_before_creating_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Unencodable:
        def __reduce__(self) -> tuple[object, ...]:
            raise RuntimeError("metadata cannot be encoded")

    registry = make_registry()
    registry.register("test.large")
    payload = EncodedPayload.from_buffer(
        "test.large",
        memoryview(b"mapped payload"),
        None,
        "shm",
        lambda: None,
    )
    value = Value(
        "test.large",
        "fingerprint",
        ValueMeta({"unencodable": Unencodable()}),
        payload,
    )
    codec = ValueCodec(registry, shm_threshold=1)
    monkeypatch.setattr(
        boundary_module,
        "_create_segment",
        lambda _data: pytest.fail("segment created before metadata encoded"),
    )

    with pytest.raises(RuntimeError, match="metadata cannot be encoded"):
        codec.encode(value, [], [])


def test_direct_shm_writer_failure_releases_its_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = make_registry()

    def fail(_buffer: memoryview) -> int:
        raise RuntimeError("direct write failed")

    registry.register(
        "test.direct",
        encode=lambda obj: bytes(str(obj), "utf-8"),
        decode=lambda data: data.decode(),
        prepare_buffer_encoding=lambda _obj: BufferEncoding(32, fail),
    )
    codec = ValueCodec(registry, shm_threshold=1)
    names: list[str] = []
    create_empty_segment = boundary_module._create_empty_segment

    def record_segment(size: int) -> SharedMemory:
        segment = create_empty_segment(size)
        names.append(segment.name)
        return segment

    monkeypatch.setattr(boundary_module, "_create_empty_segment", record_segment)
    blobs: list[bytes] = []
    segments = []
    with pytest.raises(RuntimeError, match="direct write failed"):
        codec.encode(registry.wrap("test.direct", "payload"), blobs, segments)

    assert len(names) == 1
    assert blobs == []
    assert segments == []
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=names[0])


def test_buffer_encoding_falls_back_to_bytes_when_shm_is_disabled() -> None:
    registry = make_registry()
    prepared: list[object] = []
    encoded: list[object] = []

    def encode(obj: object) -> bytes:
        encoded.append(obj)
        return f"encoded:{obj}".encode()

    def prepare(obj: object) -> BufferEncoding:
        prepared.append(obj)
        return BufferEncoding(100, lambda _buffer: 100)

    registry.register(
        "test.buffered",
        encode=encode,
        decode=lambda data: data.decode(),
        prepare_buffer_encoding=prepare,
        fingerprint=lambda _obj: "fingerprint",
    )
    blobs: list[bytes] = []
    wire, stat = ValueCodec(registry, use_shm=False).encode(
        registry.wrap("test.buffered", "payload"), blobs, []
    )

    assert prepared == []
    assert encoded == ["payload"]
    assert stat.transport == "inline"
    payload = wire["payload"]
    assert isinstance(payload, dict)
    assert blobs[payload["blob"]] == b"encoded:payload"


def test_result_encoding_releases_prior_segments_on_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Unencodable:
        def __reduce__(self) -> tuple[object, ...]:
            raise RuntimeError("metadata cannot be encoded")

    registry = make_registry()

    def prepare(obj: object) -> BufferEncoding:
        data = str(obj).encode()

        def write(buffer: memoryview) -> int:
            buffer[:] = data
            return len(data)

        return BufferEncoding(len(data), write)

    registry.register(
        "test.large",
        encode=lambda obj: str(obj).encode(),
        decode=lambda data: data.decode(),
        prepare_buffer_encoding=prepare,
    )
    first = registry.wrap("test.large", "first payload")
    second = Value(
        "test.large",
        "fingerprint",
        ValueMeta({"unencodable": Unencodable()}),
        EncodedPayload("test.large", b"second payload", None, "inline"),
    )
    codec = ValueCodec(registry, shm_threshold=1)
    names: list[str] = []
    create_segment = boundary_module._create_empty_segment

    def record_segment(size: int) -> SharedMemory:
        segment = create_segment(size)
        names.append(segment.name)
        return segment

    monkeypatch.setattr(boundary_module, "_create_empty_segment", record_segment)
    with pytest.raises(RuntimeError, match="metadata cannot be encoded"):
        encode_result(
            codec,
            InvocationResult(outputs={"first": first, "second": second}),
            "invocation",
            1.0,
        )

    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=names[0])


# -- DiskCAS ---------------------------------------------------------------


def test_cas_round_trip(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path)
    digest = cas.put(b"payload bytes")
    assert digest.startswith("blake3:")
    assert cas.has(digest)
    assert cas.get(digest) == b"payload bytes"
    assert cas.put(b"payload bytes") == digest  # idempotent
    assert cas.total_bytes() == len(b"payload bytes")
    assert cas.digests() == [digest]


def test_cas_rejects_malformed_digest(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path)
    with pytest.raises(AssetError, match="not a canonical"):
        cas.get("blake3:../../../etc/passwd")
    with pytest.raises(AssetError, match="not a canonical"):
        cas.get("sha256:" + "0" * 64)


def test_cas_corruption_is_absence(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path)
    digest = cas.put(b"good bytes")
    hexpart = digest.split(":", 1)[1]
    blob_path = tmp_path / hexpart[:2] / hexpart
    blob_path.write_bytes(b"rotted")
    assert cas.get(digest) is None  # conservative miss, never bad bytes
    assert not blob_path.exists()  # and the corpse is gone


def test_cas_adopt_file_lands_verified_and_consumes(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path / "cas")
    data = b"adopted bytes" * 3
    digest = digest_bytes(data)
    candidate = tmp_path / "candidate"
    candidate.write_bytes(data)
    cas.adopt_file(candidate, digest)
    assert not candidate.exists()
    assert cas.get(digest) == data


def test_cas_adopt_file_rejects_mismatched_bytes(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path / "cas")
    digest = digest_bytes(b"expected bytes")
    candidate = tmp_path / "candidate"
    candidate.write_bytes(b"different bytes")
    with pytest.raises(CASError, match="does not hash"):
        cas.adopt_file(candidate, digest)
    assert not candidate.exists()  # consumed on failure too
    assert cas.get(digest) is None
    assert cas.total_bytes() == 0


def test_budgeted_cas_evicts_least_recently_used(tmp_path: Path) -> None:
    cas = BudgetedDiskCAS(tmp_path / "cas", max_bytes=1000)
    first = cas.put(b"a" * 400)
    time.sleep(0.02)
    second = cas.put(b"b" * 400)
    time.sleep(0.02)
    assert cas.has(first)  # a hit refreshes recency
    time.sleep(0.02)
    third = cas.put(b"c" * 400)  # busts the budget; the LRU blob goes
    assert not cas.has(second)
    assert cas.has(first)
    assert cas.has(third)
    assert cas.total_bytes() <= 1000


def test_budgeted_cas_never_evicts_the_just_stored_blob(tmp_path: Path) -> None:
    cas = BudgetedDiskCAS(tmp_path / "cas", max_bytes=10)
    data = b"bigger than the whole budget"
    digest = cas.put(data)
    assert cas.get(digest) == data


def test_cas_concurrent_identical_puts(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path)
    data = b"x" * 65536
    with ThreadPoolExecutor(max_workers=8) as executor:
        digests = list(executor.map(cas.put, [data] * 8))
    assert set(digests) == {digest_bytes(data)}
    assert cas.get(digests[0]) == data


def test_cas_concurrent_identical_adoptions(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path / "cas")
    data = b"x" * 65536
    digest = digest_bytes(data)
    candidates = [tmp_path / f"candidate-{index}" for index in range(8)]
    for candidate in candidates:
        candidate.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda path: cas.adopt_file(path, digest), candidates))

    assert all(not candidate.exists() for candidate in candidates)
    assert cas.get(digest) == data


def test_cas_concurrent_puts_and_adoptions_share_publication_lock(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path / "cas")
    data = b"x" * 65536
    digest = digest_bytes(data)
    candidates = [tmp_path / f"candidate-{index}" for index in range(4)]
    for candidate in candidates:
        candidate.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        put_futures = [executor.submit(cas.put, data) for _ in range(4)]
        adopt_futures = [
            executor.submit(cas.adopt_file, candidate, digest) for candidate in candidates
        ]
        put_results = [future.result() for future in put_futures]
        adopt_results = [future.result() for future in adopt_futures]

    assert put_results == [digest] * 4
    assert adopt_results == [None] * 4
    assert all(not candidate.exists() for candidate in candidates)
    assert cas.get(digest) == data


def test_cas_retries_verified_winner_after_transient_access_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = DiskCAS(tmp_path)
    data = b"winner"
    digest = digest_bytes(data)
    hexpart = digest.split(":", 1)[1]
    path = tmp_path / hexpart[:2] / hexpart
    real_open = Path.open
    denied_opens = 0

    def lose_replace_race(_source: Path, target: Path) -> None:
        target.write_bytes(data)
        raise PermissionError("target is settling")

    def open_after_winner(file: Path, mode: str = "r"):
        nonlocal denied_opens
        if file == path and mode == "rb" and denied_opens < 2:
            denied_opens += 1
            raise PermissionError("target is settling")
        return real_open(file, mode)

    monkeypatch.setattr(os, "replace", lose_replace_race)
    monkeypatch.setattr(Path, "open", open_after_winner)

    assert cas.put(data) == digest
    assert denied_opens == 2


# -- DiskCacheStore ----------------------------------------------------------


def test_disk_cache_survives_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        first = DiskCacheStore(tmp_path, registry)
        value = registry.wrap("core.int", 42)
        await first.put("key-1", {"answer": value})

        # A fresh store over the same root - "the process restarted".
        second = DiskCacheStore(tmp_path, registry)
        hit = await second.get("key-1")
        assert hit is not None
        assert hit["answer"].fingerprint == value.fingerprint  # H4: unchanged
        assert hit["answer"].type_id == "core.int"
        assert hit["answer"].resolve() == 42
        assert second.hits == 1

        assert await second.get("missing") is None
        assert second.misses == 1

    asyncio.run(scenario())


def test_disk_cache_unknown_type_relays(tmp_path: Path) -> None:
    """A type registered elsewhere still persists and rehydrates - as an
    encoded envelope that carries but does not load (hazard H2)."""

    async def scenario() -> None:
        registry = make_registry()
        foreign = Value(
            type_id="otherpack.mystery",
            fingerprint="fp-mystery",
            meta=ValueMeta({"shape": [2, 2]}),
            payload=EncodedPayload("otherpack.mystery", b"opaque-bytes", None),
        )
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-foreign", {"out": foreign})

        hit = await DiskCacheStore(tmp_path, registry).get("key-foreign")
        assert hit is not None
        rehydrated = hit["out"]
        assert rehydrated.fingerprint == "fp-mystery"
        assert rehydrated.meta.get("shape") == [2, 2]
        payload = rehydrated.payload
        assert isinstance(payload, EncodedPayload)
        assert payload.data == b"opaque-bytes"  # relayable as-is
        with pytest.raises(UnresolvablePayload, match="not registered"):
            rehydrated.resolve()

    asyncio.run(scenario())


def test_encoded_validator_failure_is_counted_as_disk_cache_miss(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()

        def refuse(data: bytes, metadata: object) -> None:
            raise ValueError(f"refused {len(data)} bytes with {metadata!r}")

        registry.register("test.validated", validate_encoded=refuse)
        value = Value(
            type_id="test.validated",
            fingerprint="untrusted-fingerprint",
            meta=ValueMeta({"format": "test"}),
            payload=EncodedPayload("test.validated", b"malformed", None),
        )
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-invalid", {"out": value})
        assert await store.get("key-invalid") is None
        assert store.hits == 0
        assert store.misses == 1

    asyncio.run(scenario())


def test_core_combo_encoded_validation_is_enforced_by_disk_cache(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        value = Value(
            type_id=CORE_COMBO,
            fingerprint="untrusted-fingerprint",
            meta=ValueMeta(),
            payload=EncodedPayload(CORE_COMBO, b"json:1", None),
        )
        store = DiskCacheStore(tmp_path, registry)
        await store.put("invalid-combo", {"out": value})
        assert await store.get("invalid-combo") is None
        assert store.hits == 0
        assert store.misses == 1

    asyncio.run(scenario())


def test_core_combo_valid_boundary_and_disk_cache_round_trips(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        value = registry.wrap(CORE_COMBO, "euler")
        expected_fingerprint = "adfaa12b6da4ac3f28ffc6a3ec6712f845573052"
        assert value.fingerprint == expected_fingerprint

        codec = ValueCodec(registry, use_shm=False)
        blobs: list[bytes] = []
        wire, _ = codec.encode(value, blobs, [])
        decoded, _ = codec.decode(wire, blobs, [])
        assert decoded.resolve() == "euler"
        assert decoded.fingerprint == expected_fingerprint

        store = DiskCacheStore(tmp_path, registry)
        await store.put("valid-combo", {"out": decoded})
        hit = await store.get("valid-combo")
        assert hit is not None
        assert hit["out"].resolve() == "euler"
        assert hit["out"].fingerprint == expected_fingerprint
        assert store.hits == 1
        assert store.misses == 0

    asyncio.run(scenario())


def test_disk_cache_refuses_resource_stubs(tmp_path: Path) -> None:
    """A value referencing live process state must never outlive the
    process: the whole entry is refused, silently."""

    async def scenario() -> None:
        registry = make_registry()
        stub = Value(
            type_id="core.int",
            fingerprint="resident:abc",
            meta=ValueMeta({RESOURCE_ID_META_KEY: "res-1"}),
            payload=EncodedPayload("core.int", b"json:1", None),
        )
        plain = registry.wrap("core.int", 7)
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-stub", {"stub": stub, "plain": plain})
        assert await store.get("key-stub") is None
        assert store.drop_referencing("res-1") == 0  # nothing persisted

    asyncio.run(scenario())


def test_disk_cache_missing_blob_is_miss(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-1", {"answer": registry.wrap("core.int", 42)})
        for digest in DiskCAS(tmp_path / "cas").digests():
            DiskCAS(tmp_path / "cas").delete(digest)
        assert await store.get("key-1") is None
        # The unservable manifest was dropped, not left advertising a lie.
        assert list((tmp_path / "entries").glob("*.json")) == []

    asyncio.run(scenario())


def test_disk_cache_trims_lru_and_gcs_blobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        registry.register("test.bulk")  # default codec: pickle fallback
        store = DiskCacheStore(tmp_path, registry, max_bytes=64 * 1024)
        for index in range(6):
            await store.put(
                f"key-{index}",
                {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))},
            )
        # Budget holds: evictions kept total under max_bytes.
        cas = DiskCAS(tmp_path / "cas")
        manifests = list((tmp_path / "entries").glob("*.json"))
        assert cas.total_bytes() + sum(p.stat().st_size for p in manifests) <= 64 * 1024
        assert 0 < len(manifests) < 6
        # The newest entry survived (LRU evicts oldest first)...
        assert await store.get("key-5") is not None
        # ...an evicted one is a miss, and no orphan blobs linger.
        assert await store.get("key-0") is None
        referenced = len(list((tmp_path / "entries").glob("*.json")))
        assert len(cas.digests()) == referenced

    asyncio.run(scenario())


def test_disk_cache_hit_refreshes_lru_across_store_instances(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        registry.register("test.bulk")
        store = DiskCacheStore(tmp_path, registry, max_bytes=40 * 1024)
        await store.put("key-a", {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))})
        time.sleep(0.02)
        await store.put("key-b", {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))})
        time.sleep(0.02)
        assert await DiskCacheStore(tmp_path, registry, max_bytes=40 * 1024).get("key-a")
        time.sleep(0.02)
        await store.put("key-c", {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))})

        reopened = DiskCacheStore(tmp_path, registry, max_bytes=40 * 1024)
        assert await reopened.get("key-a") is not None
        assert await reopened.get("key-b") is None
        assert await reopened.get("key-c") is not None

    asyncio.run(scenario())


def test_disk_cache_shared_root_concurrent_writes_are_complete(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        registry.register("test.bulk")
        stores = (DiskCacheStore(tmp_path, registry), DiskCacheStore(tmp_path, registry))
        values = [os.urandom(4096) for _ in range(12)]
        await asyncio.gather(
            *(
                stores[index % 2].put(f"key-{index}", {"out": registry.wrap("test.bulk", value)})
                for index, value in enumerate(values)
            )
        )

        reopened = DiskCacheStore(tmp_path, registry)
        for index, value in enumerate(values):
            hit = await reopened.get(f"key-{index}")
            assert hit is not None
            assert hit["out"].resolve() == value
        assert not list((tmp_path / "entries").glob("*.tmp"))

    asyncio.run(scenario())


def test_disk_cache_process_lock_waits_for_another_instance(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=hold_disk_cache_process_lock,
        args=(str(tmp_path), ready, release),
    )
    holder.start()
    try:
        assert ready.wait(15)
        registry = make_registry()
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(DiskCacheStore, tmp_path, registry)
            time.sleep(10.5 if os.name == "nt" else 0.2)
            assert not waiting.done()
            release.set()
            store = waiting.result(timeout=15)
        asyncio.run(store.put("after-lock", {"out": registry.wrap("core.int", 7)}))
        assert asyncio.run(store.get("after-lock")) is not None
    finally:
        release.set()
        holder.join(timeout=15)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_disk_cache_startup_recovers_orphans_and_interrupted_manifests(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        await store.put("keeper", {"out": registry.wrap("core.int", 7)})
        cas = DiskCAS(tmp_path / "cas")
        orphan = cas.put(b"unreferenced payload")
        interrupted = tmp_path / "entries" / ".interrupted.tmp"
        interrupted.write_text("partial", encoding="utf-8")
        malformed = tmp_path / "entries" / "malformed.json"
        malformed.write_text("{", encoding="utf-8")

        reopened = DiskCacheStore(tmp_path, registry)
        assert await reopened.get("keeper") is not None
        assert not cas.has(orphan)
        assert not interrupted.exists()
        assert not malformed.exists()

    asyncio.run(scenario())


def test_disk_cache_shared_blob_survives_one_eviction(tmp_path: Path) -> None:
    """Two entries carrying identical bytes share one blob; evicting one
    entry must not take the other's payload with it."""

    async def scenario() -> None:
        from dinkster_values import stable_hash

        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        value = registry.wrap("core.string", "shared payload " * 100)
        await store.put("key-a", {"out": value})
        await store.put("key-b", {"out": value})
        cas = DiskCAS(tmp_path / "cas")
        assert len(cas.digests()) == 1  # deduplicated

        # Make key-a the clear LRU victim, then budget so the next put
        # forces exactly one eviction.
        entries = tmp_path / "entries"
        key_a = entries / (stable_hash([b"key-a"]) + ".json")
        old = key_a.stat().st_mtime - 1000
        os.utime(key_a, (old, old))
        total = cas.total_bytes() + sum(p.stat().st_size for p in entries.glob("*.json"))
        probe_root = tmp_path / "probe"
        probe = DiskCacheStore(probe_root, registry)
        await probe.put("key-c", {"out": registry.wrap("core.string", "more")})
        added = DiskCAS(probe_root / "cas").total_bytes() + sum(
            p.stat().st_size for p in (probe_root / "entries").glob("*.json")
        )
        # Exactly the post-write usage minus key-a's manifest: removing the
        # first shared reference is sufficient, but must not credit its blob.
        tight = DiskCacheStore(tmp_path, registry, max_bytes=total + added - key_a.stat().st_size)
        await tight.put("key-c", {"out": registry.wrap("core.string", "more")})

        assert await tight.get("key-a") is None  # evicted (was LRU)
        hit = await tight.get("key-b")  # its shared blob survived
        assert hit is not None
        assert hit["out"].resolve() == "shared payload " * 100

    asyncio.run(scenario())


# -- LayeredCache ------------------------------------------------------------


def test_layered_cache_promotes_hits(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        memory = MemoryLRUCache()
        disk = DiskCacheStore(tmp_path, registry)
        layered = LayeredCache(memory, disk)

        # Seed disk only (as if a previous process wrote it).
        await disk.put("key-1", {"out": registry.wrap("core.int", 9)})
        assert len(memory) == 0

        hit = await layered.get("key-1")
        assert hit is not None and hit["out"].resolve() == 9
        assert len(memory) == 1  # promoted
        assert layered.take_hit_layer() == "disk"
        assert await memory.get("key-1") is not None

        assert await layered.get("key-1") is not None
        assert layered.take_hit_layer() == "memory"
        assert layered.hits_by_layer == {"disk": 1, "memory": 1}

        # Write-through: a put lands everywhere.
        await layered.put("key-2", {"out": registry.wrap("core.int", 10)})
        assert await memory.get("key-2") is not None
        assert await DiskCacheStore(tmp_path, registry).get("key-2") is not None

        assert layered.clear() >= 2
        assert await layered.get("key-1") is None
        assert await layered.get("key-2") is None

    asyncio.run(scenario())


def test_layered_cache_invalidation_fans_out(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        memory = MemoryLRUCache()
        layered = LayeredCache(memory, DiskCacheStore(tmp_path, registry))
        stub = Value(
            type_id="core.int",
            fingerprint="resident:abc",
            meta=ValueMeta({RESOURCE_ID_META_KEY: "res-1"}),
            payload=EncodedPayload("core.int", b"json:1", None),
        )
        await layered.put("key-stub", {"out": stub})
        assert await memory.get("key-stub") is not None  # memory took it
        assert await DiskCacheStore(tmp_path, registry).get("key-stub") is None
        assert layered.drop_referencing("res-1") == 1  # disk refused it
        assert await layered.get("key-stub") is None

    asyncio.run(scenario())


def test_layered_cache_keeps_unencodable_values_in_memory(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()

        def reject_encode(_value: object) -> bytes:
            raise ValueError("not persistable")

        registry.register("test.unpersistable", encode=reject_encode, decode=lambda data: data)
        payload = registry.wrap("core.int", 5).payload
        value = Value("test.unpersistable", "fp-unpersistable", ValueMeta(), payload)
        memory = MemoryLRUCache()
        layered = LayeredCache(memory, DiskCacheStore(tmp_path, registry))

        await layered.put("key-unpersistable", {"out": value})
        memory_hit = await layered.get("key-unpersistable")
        assert memory_hit is not None
        assert memory_hit["out"].resolve() == 5
        assert await DiskCacheStore(tmp_path, registry).get("key-unpersistable") is None

    asyncio.run(scenario())


# -- boundary cas transport ---------------------------------------------------


def _payload_wire(wire: dict[str, object]) -> dict[str, object]:
    from typing import cast

    return cast("dict[str, object]", wire["payload"])


def _paired_cas_codecs() -> tuple[ValueCodec, ValueCodec, TypeRegistry]:
    registry = make_registry()
    left = ValueCodec(registry, use_shm=False, accept_shm=False, cas_threshold=16)
    right = ValueCodec(registry, use_shm=False, accept_shm=False, cas_threshold=16)
    left.enable_cas()
    right.enable_cas()
    return left, right, registry


def test_cas_transport_sends_bytes_once() -> None:
    """First crossing carries digest + bytes; every repeat crossing of the
    same payload is digest-only and still decodes to the same bytes."""
    left, right, registry = _paired_cas_codecs()
    value = registry.wrap("core.string", "x" * 100)

    blobs: list[bytes] = []
    wire, stat = left.encode(value, blobs, [])
    assert stat.transport == "cas"
    assert "blob" in _payload_wire(wire)
    assert len(blobs) == 2  # meta blob + payload bytes
    decoded, _ = right.decode(wire, blobs, [])
    assert decoded.resolve() == "x" * 100

    blobs2: list[bytes] = []
    wire2, stat2 = left.encode(value, blobs2, [])
    assert stat2.transport == "cas"
    assert "blob" not in _payload_wire(wire2)  # digest-only
    assert len(blobs2) == 1  # only the (small) meta blob rides the frame
    decoded2, _ = right.decode(wire2, blobs2, [])
    assert decoded2.resolve() == "x" * 100
    assert decoded2.fingerprint == value.fingerprint


def test_cas_transport_dedups_echo_back() -> None:
    """Bytes that crossed one way never cross again the other way: the
    receiver can echo the value back digest-only, because both sides retain
    every crossed payload for the conversation."""
    left, right, registry = _paired_cas_codecs()
    value = registry.wrap("core.string", "y" * 100)

    blobs: list[bytes] = []
    wire, _ = left.encode(value, blobs, [])
    received, _ = right.decode(wire, blobs, [])

    back_blobs: list[bytes] = []
    back_wire, _ = right.encode(received, back_blobs, [])
    assert "blob" not in _payload_wire(back_wire)  # right knows left has it
    assert len(back_blobs) == 1  # meta only
    round_tripped, _ = left.decode(back_wire, back_blobs, [])
    assert round_tripped.resolve() == "y" * 100


def test_cas_transport_budget_bounds_retention() -> None:
    """Past the retention budget the transport degrades, never breaks:
    over-budget payloads carry their bytes on every crossing (marked
    retained=false, mirrored by the receiver) instead of pinning RAM."""
    registry = make_registry()
    left = ValueCodec(registry, use_shm=False, accept_shm=False, cas_threshold=16, cas_budget=150)
    right = ValueCodec(registry, use_shm=False, accept_shm=False, cas_threshold=16, cas_budget=150)
    left.enable_cas()
    right.enable_cas()

    fits = registry.wrap("core.string", "a" * 100)
    over = registry.wrap("core.string", "b" * 100)  # second 100B blows the 150B budget

    blobs: list[bytes] = []
    wire, _ = left.encode(fits, blobs, [])
    assert _payload_wire(wire)["retained"] is True
    right.decode(wire, blobs, [])

    for _ in range(2):  # every crossing re-sends bytes, both stay consistent
        blobs2: list[bytes] = []
        wire2, stat2 = left.encode(over, blobs2, [])
        assert stat2.transport == "cas"
        assert "blob" in _payload_wire(wire2)  # bytes ride along again
        assert _payload_wire(wire2)["retained"] is False
        decoded, _ = right.decode(wire2, blobs2, [])
        assert decoded.resolve() == "b" * 100

    # The retained value still dedups; the echo back is digest-only.
    back_blobs: list[bytes] = []
    back_wire, _ = right.encode(fits, back_blobs, [])
    assert "blob" not in _payload_wire(back_wire)
    round_tripped, _ = left.decode(back_wire, back_blobs, [])
    assert round_tripped.resolve() == "a" * 100


def test_cas_transport_requires_negotiation() -> None:
    left, _, registry = _paired_cas_codecs()
    value = registry.wrap("core.string", "z" * 100)
    blobs: list[bytes] = []
    wire, _ = left.encode(value, blobs, [])

    unnegotiated = ValueCodec(registry, use_shm=False, accept_shm=False)
    with pytest.raises(BoundaryError, match="not negotiated"):
        unnegotiated.decode(wire, blobs, [])
    # And an unnegotiated sender never produces cas descriptors.
    plain_wire, plain_stat = unnegotiated.encode(value, [], [])
    assert plain_stat.transport == "inline"
    assert _payload_wire(plain_wire)["transport"] == "inline"


def test_cas_transport_conservative_failures() -> None:
    left, right, registry = _paired_cas_codecs()
    value = registry.wrap("core.string", "w" * 100)
    blobs: list[bytes] = []
    wire, _ = left.encode(value, blobs, [])

    # A digest the receiver never saw is a protocol violation, not a guess.
    orphan = dict(wire)
    orphan["payload"] = {
        "transport": "cas",
        "digest": "0" * 40,
        "size": 100,
    }
    with pytest.raises(BoundaryError, match="never crossed"):
        right.decode(orphan, [blobs[0]], [])  # meta blob still present

    # A retained digest cannot be reused with a false declared size.
    right.decode(wire, blobs, [])
    false_size = dict(wire)
    false_size_payload = dict(_payload_wire(wire))
    false_size_payload.pop("blob")
    declared_size = false_size_payload["size"]
    assert isinstance(declared_size, int)
    false_size_payload["size"] = declared_size + 1
    false_size["payload"] = false_size_payload
    with pytest.raises(BoundaryError, match="has size"):
        right.decode(false_size, [blobs[0]], [])

    # Bytes that do not hash to their claimed digest are refused.
    tampered_blobs = [blobs[0], b"not the payload" + b"!" * 85]
    with pytest.raises(BoundaryError, match="does not match"):
        right.decode(wire, tampered_blobs, [])


def test_encoded_validator_boundary_failure_is_named_and_precedes_value_creation() -> None:
    registry = make_registry()

    def refuse(data: bytes, metadata: object) -> None:
        raise ValueError(f"bad payload {data!r} with {metadata!r}")

    registry.register("test.validated", validate_encoded=refuse)
    value = Value(
        type_id="test.validated",
        fingerprint="untrusted-fingerprint",
        meta=ValueMeta({"format": "test"}),
        payload=EncodedPayload("test.validated", b"malformed", None),
    )
    codec = ValueCodec(registry, use_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    with pytest.raises(
        EncodedPayloadValidationError,
        match="encoded payload validation failed for type 'test.validated'",
    ):
        codec.decode(wire, blobs, [])


def test_core_combo_encoded_validation_is_enforced_at_worker_boundary() -> None:
    registry = make_registry()
    value = Value(
        type_id=CORE_COMBO,
        fingerprint="untrusted-fingerprint",
        meta=ValueMeta(),
        payload=EncodedPayload(CORE_COMBO, b'json:["not-a-scalar"]', None),
    )
    codec = ValueCodec(registry, use_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    with pytest.raises(
        EncodedPayloadValidationError,
        match="encoded payload validation failed for type 'core.combo'",
    ):
        codec.decode(wire, blobs, [])


def test_types_without_encoded_validator_keep_boundary_and_cache_behavior(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        registry = make_registry()
        registry.register("test.plain")
        value = registry.wrap("test.plain", {"answer": 42})
        codec = ValueCodec(registry, use_shm=False)
        blobs: list[bytes] = []
        wire, _ = codec.encode(value, blobs, [])
        decoded, _ = codec.decode(wire, blobs, [])
        assert decoded.resolve() == {"answer": 42}

        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-plain", {"out": value})
        hit = await store.get("key-plain")
        assert hit is not None
        assert hit["out"].resolve() == {"answer": 42}
        assert store.hits == 1
        assert store.misses == 0

    asyncio.run(scenario())


# -- engine restart persistence ----------------------------------------------


class PoisonWorker:
    """Proof that nothing executed: any invocation is a hard failure."""

    async def prepare(self, node_types: object) -> None:
        pass

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        raise AssertionError(f"node {invocation.node_id} should have been cached")


def test_engine_cache_survives_restart(tmp_path: Path) -> None:
    """The demo that matters: run a graph, 'restart', run it again on a
    worker that cannot execute anything - every node is a disk hit."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        schemas = build_schemas(SCAFFOLD_NODES)

        first = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorkerFactory(registry),
            cache=DiskCacheStore(tmp_path, registry),
        )
        result = await first.run(image_graph(), ["s"])
        assert set(result.executed) == {"g", "i", "b", "s"}

        events: list[EngineEvent] = []
        second = Engine(
            schemas=schemas,
            registry=registry,
            worker=PoisonWorker(),
            cache=DiskCacheStore(tmp_path, registry),
            on_event=events.append,
        )
        rerun = await second.run(image_graph(), ["s"])
        assert rerun.executed == ()
        assert set(rerun.cached) == {"g", "i", "b", "s"}
        assert rerun.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
        assert all(e.kind != "node_started" for e in events)

    asyncio.run(scenario())
