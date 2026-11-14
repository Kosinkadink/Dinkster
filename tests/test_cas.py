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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
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
    RESOURCE_ID_META_KEY,
    EncodedPayload,
    TypeRegistry,
    UnresolvablePayload,
    Value,
    ValueMeta,
    register_core_types,
)
from dinkster_workers import ValueCodec
from dinkster_workers.boundary import release_segment
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types
from test_engine import InProcessWorkerFactory, image_graph

from tests.async_scenarios import run_scenario


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
    second = cas.put(b"b" * 400)
    first_path = cas.resolve(first)
    second_path = cas.resolve(second)
    assert first_path is not None and second_path is not None
    os.utime(first_path, (10, 10))
    os.utime(second_path, (20, 20))
    assert cas.has(first)  # a hit refreshes recency
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


# -- DiskCacheStore ----------------------------------------------------------


def test_disk_cache_survives_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        first = DiskCacheStore(tmp_path, registry)
        value = registry.wrap("INT", 42)
        await first.put("key-1", {"answer": value})

        # A fresh store over the same root - "the process restarted".
        second = DiskCacheStore(tmp_path, registry)
        hit = await second.get("key-1")
        assert hit is not None
        assert hit["answer"].fingerprint == value.fingerprint  # H4: unchanged
        assert hit["answer"].type_id == "INT"
        assert hit["answer"].resolve() == 42
        assert second.hits == 1

        assert await second.get("missing") is None
        assert second.misses == 1

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_disk_cache_refuses_resource_stubs(tmp_path: Path) -> None:
    """A value referencing live process state must never outlive the
    process: the whole entry is refused, silently."""

    async def scenario() -> None:
        registry = make_registry()
        stub = Value(
            type_id="INT",
            fingerprint="resident:abc",
            meta=ValueMeta({RESOURCE_ID_META_KEY: "res-1"}),
            payload=EncodedPayload("INT", b"json:1", None),
        )
        plain = registry.wrap("INT", 7)
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-stub", {"stub": stub, "plain": plain})
        assert await store.get("key-stub") is None
        assert store.drop_referencing("res-1") == 0  # nothing persisted

    run_scenario(scenario())


def test_disk_cache_missing_blob_is_miss(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        await store.put("key-1", {"answer": registry.wrap("INT", 42)})
        for digest in DiskCAS(tmp_path / "cas").digests():
            DiskCAS(tmp_path / "cas").delete(digest)
        assert await store.get("key-1") is None
        # The unservable manifest was dropped, not left advertising a lie.
        assert list((tmp_path / "entries").glob("*.json")) == []

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_disk_cache_hit_refreshes_lru_across_store_instances(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        registry.register("test.bulk")
        store = DiskCacheStore(tmp_path, registry, max_bytes=40 * 1024)
        await store.put("key-a", {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))})
        first_manifest = next((tmp_path / "entries").glob("*.json"))
        await store.put("key-b", {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))})
        second_manifest = next(
            path for path in (tmp_path / "entries").glob("*.json") if path != first_manifest
        )
        os.utime(first_manifest, (10, 10))
        os.utime(second_manifest, (20, 20))
        assert await DiskCacheStore(tmp_path, registry, max_bytes=40 * 1024).get("key-a")
        await store.put("key-c", {"out": registry.wrap("test.bulk", os.urandom(16 * 1024))})

        reopened = DiskCacheStore(tmp_path, registry, max_bytes=40 * 1024)
        assert await reopened.get("key-a") is not None
        assert await reopened.get("key-b") is None
        assert await reopened.get("key-c") is not None

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_disk_cache_startup_recovers_orphans_and_interrupted_manifests(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        await store.put("keeper", {"out": registry.wrap("INT", 7)})
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

    run_scenario(scenario())


def test_disk_cache_shared_blob_survives_one_eviction(tmp_path: Path) -> None:
    """Two entries carrying identical bytes share one blob; evicting one
    entry must not take the other's payload with it."""

    async def scenario() -> None:
        from dinkster_values import stable_hash

        registry = make_registry()
        store = DiskCacheStore(tmp_path, registry)
        value = registry.wrap("STRING", "shared payload " * 100)
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
        await probe.put("key-c", {"out": registry.wrap("STRING", "more")})
        added = DiskCAS(probe_root / "cas").total_bytes() + sum(
            p.stat().st_size for p in (probe_root / "entries").glob("*.json")
        )
        # Exactly the post-write usage minus key-a's manifest: removing the
        # first shared reference is sufficient, but must not credit its blob.
        tight = DiskCacheStore(tmp_path, registry, max_bytes=total + added - key_a.stat().st_size)
        await tight.put("key-c", {"out": registry.wrap("STRING", "more")})

        assert await tight.get("key-a") is None  # evicted (was LRU)
        hit = await tight.get("key-b")  # its shared blob survived
        assert hit is not None
        assert hit["out"].resolve() == "shared payload " * 100

    run_scenario(scenario())


# -- LayeredCache ------------------------------------------------------------


def test_layered_cache_keeps_unencodable_values_in_memory(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = make_registry()

        def reject_encode(_value: object) -> bytes:
            raise ValueError("not persistable")

        registry.register("test.unpersistable", encode=reject_encode, decode=lambda data: data)
        payload = registry.wrap("INT", 5).payload
        value = Value("test.unpersistable", "fp-unpersistable", ValueMeta(), payload)
        memory = MemoryLRUCache()
        layered = LayeredCache(memory, DiskCacheStore(tmp_path, registry))

        await layered.put("key-unpersistable", {"out": value})
        memory_hit = await layered.get("key-unpersistable")
        assert memory_hit is not None
        assert memory_hit["out"].resolve() == 5
        assert await DiskCacheStore(tmp_path, registry).get("key-unpersistable") is None

    run_scenario(scenario())


# -- boundary cas transport ---------------------------------------------------


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

    run_scenario(scenario())
