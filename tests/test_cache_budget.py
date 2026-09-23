"""Byte-bounded RAM admission and lazy content-addressed disk replay."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_caches.layered import DEFAULT_DISK_SPILL_MIN_BYTES
from dinkster_caches.memory import entry_cost
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import PressureSignal, system_memory_snapshot
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    COST_META_KEY,
    RESOURCE_ID_META_KEY,
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    default_encode,
    iter_value_tree,
    make_list_value,
    register_core_types,
)
from dinkster_values.image_codec import image_array_fingerprint, image_array_meta
from dinkster_values.model import PyObjPayload
from dinkster_values.storage import array_storage_meta, image_input
from dinkster_workers import InProcessWorker


def entry(nbytes: int, *, device: str = "ram", text: str = "payload") -> dict[str, Value]:
    return {
        "out": Value(
            "core.string",
            text,
            ValueMeta({COST_META_KEY: {device: nbytes}}),
            EncodedPayload("core.string", default_encode(text), None),
        )
    }


def disk_store(root: Path) -> DiskCacheStore:
    registry = TypeRegistry()
    register_core_types(registry)
    return DiskCacheStore(root, registry)


def test_default_ram_budget_uses_effective_host_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = system_memory_snapshot(
        "linux", host_query=lambda: (8192, 4096), linux_cgroup_query=lambda: (2048, 1024)
    )
    calls: list[None] = []

    def probe():
        calls.append(None)
        return snapshot

    monkeypatch.setattr("dinkster_caches.memory.system_memory_snapshot", probe)
    assert MemoryLRUCache().max_bytes == 256
    assert len(calls) == 1
    assert MemoryLRUCache(max_bytes=0).max_bytes == 0
    assert len(calls) == 1


@pytest.mark.parametrize("options", [{"max_entries": 0}, {"max_bytes": -1}])
def test_invalid_ram_budgets(options: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        MemoryLRUCache(**options)


def test_ram_budget_evicts_lru_without_dropping_other_devices() -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache(max_bytes=100)
        await cache.put("gpu", entry(500, device="vram:cuda:0"))
        await cache.put("a", entry(40))
        await cache.put("b", entry(40))
        await cache.get("a")
        await cache.put("c", entry(50))
        assert await cache.get("b") is None
        assert await cache.get("a") is not None
        assert await cache.get("gpu") is not None
        assert cache.footprint("ram") == 90
        assert cache.footprint("vram:cuda:0") == 500
        assert await cache.shed(PressureSignal("vram:cuda:1", 50)) == 0
        assert await cache.shed(PressureSignal("vram:cuda:0", 50)) == 500
        assert cache.footprint("ram") == 90
        assert cache.footprint("vram:cuda:0") == 0

    asyncio.run(scenario())


def test_budgeted_batch_workflow_matches_unbounded_output_and_peak_stays_bounded() -> None:
    image_type = TypeExpr.concrete("test.image")
    shape = (4, 8, 8, 3)
    batch_bytes = int(np.prod(shape))

    class Source(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(node_type="test.source", outputs=(OutputSpec("image", image_type),))

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(image=np.arange(batch_bytes, dtype=np.uint8).reshape(shape))

    class Xor(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.xor",
                inputs=(
                    InputSpec("image", image_type, accepts_storage=True),
                    InputSpec("mask", TypeExpr.concrete("core.int")),
                ),
                outputs=(OutputSpec("image", image_type),),
            )

        @classmethod
        def execute(cls, image: object, mask: int) -> Mapping[str, object]:
            return cls.outputs(image=np.bitwise_xor(np.asarray(image), np.uint8(mask)))

    class TrackingCache(MemoryLRUCache):
        def __init__(self, max_bytes: int) -> None:
            super().__init__(max_bytes=max_bytes)
            self.peak_ram = 0

        async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
            await super().put(key, outputs)
            self.peak_ram = max(self.peak_ram, self.footprint("ram"))

    nodes = [Source, Xor]
    schemas = build_schemas(nodes)
    graph = Graph(
        nodes={
            "source": GraphNode("test.source", {}),
            "first": GraphNode("test.xor", {"image": Link("source", "image"), "mask": 0xA5}),
            "second": GraphNode("test.xor", {"image": Link("first", "image"), "mask": 0x3C}),
        }
    )

    async def run(max_bytes: int) -> tuple[bytes, TrackingCache]:
        registry = TypeRegistry()
        register_core_types(registry)
        registry.register(
            "test.image",
            fingerprint=image_array_fingerprint("test.image"),
            meta=image_array_meta,
            input_convert=image_input,
        )
        cache = TrackingCache(max_bytes)
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=cache,
        )
        result = await engine.run(graph, ["second"])
        output = np.asarray(result.outputs["second"]["image"].resolve())
        return output.tobytes(), cache

    async def scenario() -> None:
        expected, _ = await run(batch_bytes * 4)
        actual, constrained = await run(batch_bytes)
        assert actual == expected
        assert constrained.peak_ram <= constrained.max_bytes
        assert constrained.peak_ram == batch_bytes

    asyncio.run(scenario())


def test_entry_bound_retains_uncosted_and_unknown_types() -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache(max_entries=2, max_bytes=0)
        value = Value(
            "unknown.type", "opaque", ValueMeta(), EncodedPayload("unknown.type", b"x", None)
        )
        await cache.put("a", {"out": value})
        await cache.put("b", {})
        await cache.put("c", {"out": value})
        assert len(cache) == 2
        assert await cache.get("a") is None
        assert (await cache.get("c")) == {"out": value}
        await cache.put("oversized", entry(1))
        assert await cache.get("oversized") is None
        assert cache.footprint("ram") == 0
        assert len(cache) == 2
        assert await cache.get("b") == {}

    asyncio.run(scenario())


def test_replacement_and_invalidation_account_all_residencies() -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache(max_bytes=100)
        await cache.put("key", entry(80))
        await cache.put("key", entry(60, device="vram:cuda:0"))
        await cache.put("other", entry(90))
        assert len(cache) == 2
        assert cache.footprint("ram") == 90
        assert cache.footprint("vram:cuda:0") == 60
        reference = Value(
            "unknown.type",
            "reference",
            ValueMeta({RESOURCE_ID_META_KEY: "resource", COST_META_KEY: {"ram": 1000}}),
            EncodedPayload("unknown.type", b"stub", None),
        )
        await cache.put("stub", {"reference": reference, **entry(10)})
        assert cache.footprint("ram") == 100
        assert cache.drop_referencing("resource") == 1
        assert cache.footprint("ram") == 90
        assert cache.clear() == 2
        assert cache.footprint("ram") == cache.footprint("vram:cuda:0") == 0

    asyncio.run(scenario())


def test_lazy_spill_threshold_replay_and_clean_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=50)
        await cache.put("small", entry(49))
        await cache.put("large", entry(50))
        assert await disk_store(tmp_path).get("small") is not None
        assert await disk.get("large") is None
        assert len(list((tmp_path / "entries").glob("*.json"))) == 1
        assert await memory.shed(PressureSignal("ram", 99)) == 99
        assert await disk.get("small") is not None
        assert await disk_store(tmp_path).get("large") is not None
        manifest = disk._entry_path("large")
        content = manifest.read_bytes()
        assert await cache.get("large") is not None
        assert cache.take_hit_layer() == "disk"

        # A clean promoted entry is evicted without even invoking a writer.
        def reject_write(*_args: object) -> None:
            raise AssertionError("redundant disk write")

        monkeypatch.setattr(disk, "_put", reject_write)
        assert await memory.shed(PressureSignal("ram", 50)) == 50
        assert manifest.read_bytes() == content
        assert await disk_store(tmp_path).get("large") is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("threshold", [0, 50])
def test_uncosted_unknown_outputs_write_through_and_evict_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: int
) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_entries=1, max_bytes=0)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=threshold)
        value = Value(
            "unknown.type", "opaque", ValueMeta(), EncodedPayload("unknown.type", b"x", None)
        )
        await cache.put("key", {"out": value})
        replay = await disk_store(tmp_path).get("key")
        assert replay is not None and replay["out"].fingerprint == value.fingerprint
        original = disk._put

        def reject_rewrite(key, outputs) -> None:
            assert key != "key", "redundant disk write"
            original(key, outputs)

        monkeypatch.setattr(disk, "_put", reject_rewrite)
        await cache.put("other", {})
        assert await memory.get("key") is None
        assert await disk_store(tmp_path).get("key") is not None

    asyncio.run(scenario())


def test_failed_eager_replacement_keeps_previous_ram_copy_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=50)
        await cache.put("key", entry(10, text="old"))
        assert await disk.get("key") is not None
        original = disk._write_entry

        def fail(*_args: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(disk, "_write_entry", fail)
        with pytest.raises(OSError, match="disk full"):
            await cache.put("key", entry(20, text="new"))
        assert await memory.get("key") == entry(10, text="old")
        with pytest.raises(OSError, match="disk full"):
            await memory.shed(PressureSignal("ram", 10))
        assert memory.footprint("ram") == 10
        monkeypatch.setattr(disk, "_write_entry", original)
        assert await memory.shed(PressureSignal("ram", 10)) == 10
        replay = await disk_store(tmp_path).get("key")
        assert replay is not None and replay["out"].fingerprint == "old"

    asyncio.run(scenario())


def test_pressure_cannot_overwrite_an_eager_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=50)
        await cache.put("key", entry(10, text="old"))
        published = asyncio.Event()
        resume = asyncio.Event()
        original = disk.put

        async def pause_after_publish(key: str, outputs: Mapping[str, Value]) -> None:
            await original(key, outputs)
            if outputs["out"].fingerprint == "new":
                published.set()
                await resume.wait()

        monkeypatch.setattr(disk, "put", pause_after_publish)
        replacing = asyncio.create_task(cache.put("key", entry(20, text="new")))
        await published.wait()
        shedding = asyncio.create_task(memory.shed(PressureSignal("ram", 10)))
        try:
            await asyncio.sleep(0)
        finally:
            resume.set()
        await replacing
        assert await shedding == 20
        assert len(memory) == 0
        replay = await disk_store(tmp_path).get("key")
        assert replay is not None and replay["out"].fingerprint == "new"

    asyncio.run(scenario())


def test_default_spill_threshold_and_oversized_admission(tmp_path: Path) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=1)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk)
        await cache.put("small", entry(DEFAULT_DISK_SPILL_MIN_BYTES - 1))
        await cache.put("large", entry(DEFAULT_DISK_SPILL_MIN_BYTES))
        assert len(memory) == 0
        assert await disk.get("small") is not None
        assert await disk_store(tmp_path).get("large") is not None
        assert await cache.get("large") is not None
        assert len(memory) == 0

    asyncio.run(scenario())


def test_lazy_spill_keeps_content_addressed_deduplication(tmp_path: Path) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=1)
        await cache.put("a", entry(50))
        await cache.put("b", entry(50))
        await memory.shed(PressureSignal("ram", 100))
        first = await disk.entry_wire("a")
        second = await disk.entry_wire("b")
        assert first is not None and second is not None
        assert first["outputs"]["out"]["digest"] == second["outputs"]["out"]["digest"]
        assert len(list(disk._cas.digests())) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["shed", "admit", "replace"])
def test_failed_spill_retains_dirty_victim_and_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=50)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=1)
        await cache.put("dirty", entry(50))
        original = disk._write_entry

        def fail(*_args: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(disk, "_write_entry", fail)
        with pytest.raises(OSError, match="disk full"):
            if operation == "shed":
                await memory.shed(PressureSignal("ram", 50))
            elif operation == "replace":
                await cache.put("dirty", entry(100, text="replacement"))
            else:
                await cache.put("new", entry(50, text="new"))
        assert memory.footprint("ram") == 50
        assert await memory.get("dirty") == entry(50)
        assert await memory.get("new") is None
        monkeypatch.setattr(disk, "_write_entry", original)
        assert await memory.shed(PressureSignal("ram", 50)) == 50
        assert await disk_store(tmp_path).get("dirty") is not None

    asyncio.run(scenario())


def test_replacing_persisted_value_cannot_resurrect_old_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=50)
        await cache.put("key", entry(50, text="old"))
        await memory.shed(PressureSignal("ram", 50))
        assert await disk.get("key") is not None
        await cache.put("key", entry(1, text="new"))
        persisted = await disk.get("key")
        assert persisted is not None and persisted["out"].fingerprint == "new"
        await memory.shed(PressureSignal("ram", 1))
        replay = await cache.get("key")
        assert replay is not None and replay["out"].fingerprint == "new"

    asyncio.run(scenario())


def test_clear_discards_dirty_values_without_persisting(tmp_path: Path) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=1)
        await cache.put("dirty", entry(50))
        assert cache.clear() == 1
        assert await memory.shed(PressureSignal("ram", 100)) == 0
        assert await disk_store(tmp_path).get("dirty") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("invalidate", ["clear", "replace"])
def test_invalidation_cancels_queued_spill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalidate: str
) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=50)
        await cache.put("key", entry(50))
        started = threading.Event()
        resume = threading.Event()
        original = disk._with_lock

        def gated(operation, *args):
            if operation == disk._put_pending:
                started.set()
                assert resume.wait(5)
            return original(operation, *args)

        monkeypatch.setattr(disk, "_with_lock", gated)
        shedding = asyncio.create_task(memory.shed(PressureSignal("ram", 50)))
        assert await asyncio.to_thread(started.wait, 5)
        replacement = None
        try:
            if invalidate == "clear":
                cache.clear()
            else:
                replacement = asyncio.create_task(cache.put("key", entry(1, text="new")))
                await asyncio.sleep(0)
        finally:
            resume.set()
        await shedding
        if replacement is not None:
            await replacement
            await memory.shed(PressureSignal("ram", 1))
            persisted = await disk_store(tmp_path).get("key")
            replay = await cache.get("key")
            assert persisted is not None and persisted["out"].fingerprint == "new"
            assert replay is not None and replay["out"].fingerprint == "new"
        else:
            assert await disk_store(tmp_path).get("key") is None
            assert await cache.get("key") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("other_process", [False, True])
@pytest.mark.parametrize("invalidate", ["clear", "same-key", "other-key"])
def test_shared_root_invalidation_serializes_without_cancelling_other_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalidate: str, other_process: bool
) -> None:
    async def scenario() -> None:
        writer = disk_store(tmp_path)
        other = disk_store(tmp_path)
        started = threading.Event()
        resume = threading.Event()
        original = writer._with_lock

        def gated(operation, *args):
            if operation == writer._put_pending:
                started.set()
                assert resume.wait(5)
            return original(operation, *args)

        monkeypatch.setattr(writer, "_with_lock", gated)
        pending = asyncio.create_task(writer.put("key", entry(50)))
        assert await asyncio.to_thread(started.wait, 5)
        key = "other" if invalidate == "other-key" else "key"
        try:
            if other_process:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        sys.executable,
                        "-c",
                        "import sys; from pathlib import Path; "
                        "from dinkster_caches import DiskCacheStore; "
                        "from dinkster_values import TypeRegistry, register_core_types; "
                        "r = TypeRegistry(); register_core_types(r); "
                        "s = DiskCacheStore(Path(sys.argv[1]), r); "
                        "s.clear() if sys.argv[2] == 'clear' else s.discard(sys.argv[3])",
                        str(tmp_path),
                        invalidate,
                        key,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                assert result.stderr == ""
            elif invalidate == "clear":
                other.clear()
            else:
                other.discard(key)
            assert await other.get("key") is None
            await other.put("other", entry(1, text="independent"))
        finally:
            resume.set()
        await pending
        fresh = disk_store(tmp_path)
        assert await fresh.get("key") is not None
        assert await fresh.get("other") is not None
        writer.clear()
        assert await fresh.get("key") is None

    asyncio.run(scenario())


def test_clear_during_slow_hit_does_not_promote_stale_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        resume = asyncio.Event()

        class SlowStore:
            async def get(self, key: str) -> Mapping[str, Value] | None:
                started.set()
                await resume.wait()
                return entry(50)

            async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
                pass

        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, SlowStore(), disk_spill_min_bytes=1)
        getting = asyncio.create_task(cache.get("key"))
        await started.wait()
        cache.clear()
        resume.set()
        assert await getting is None
        assert len(memory) == 0
        assert await disk.get("key") is None

    asyncio.run(scenario())


def test_slow_miss_does_not_block_other_keys_or_duplicate_promotion() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        resume = asyncio.Event()
        requests: list[str] = []

        class SlowStore:
            async def get(self, key: str) -> Mapping[str, Value] | None:
                requests.append(key)
                started.set()
                await resume.wait()
                return entry(50)

            async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
                pass

        memory = MemoryLRUCache(max_bytes=100)
        await memory.put("hot", entry(50))
        cache = LayeredCache(memory, SlowStore())
        first = asyncio.create_task(cache.get("cold"))
        await started.wait()
        second = asyncio.create_task(cache.get("cold"))
        try:
            assert await asyncio.wait_for(cache.get("hot"), 5) == entry(50)
        finally:
            resume.set()
        assert await first == await second == entry(50)
        assert requests == ["cold"]
        assert not cache._key_locks

    asyncio.run(scenario())


@pytest.mark.parametrize("device", ["ram", "vram:cuda:0"])
@pytest.mark.parametrize("new_envelope", [False, True])
def test_shared_payload_shedding_credits_only_actual_footprint_delta(
    device: str, new_envelope: bool
) -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache(max_bytes=70)
        shared = Value(
            "unknown.buffer",
            "shared",
            ValueMeta({COST_META_KEY: {device: 50}}),
            PyObjPayload(bytearray(50)),
        )
        alias = (
            Value(shared.type_id, shared.fingerprint, shared.meta, shared.payload)
            if new_envelope
            else shared
        )
        await cache.put("old-alias", {"out": shared})
        await cache.put("independent", entry(20, device=device))
        await cache.put("new-alias", {"out": alias})
        assert len(cache) == 3
        assert cache.footprint(device) == 70
        assert await cache.shed(PressureSignal(device, 10)) == 20
        assert await cache.get("old-alias") is None
        assert await cache.get("independent") is None
        assert await cache.get("new-alias") == {"out": alias}
        assert cache.footprint(device) == 50
        assert await cache.shed(PressureSignal(device, 1)) == 50
        assert cache.footprint(device) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("new_envelope", [False, True])
def test_shared_view_charges_full_backing_without_duplicate_shedding(new_envelope: bool) -> None:
    async def scenario() -> None:
        backing = np.zeros((32, 32), dtype=np.float32)
        view = backing[:1, :1]
        assert view.nbytes == 4 and backing.nbytes == 4096
        shared = Value(
            "unknown.media", "view", ValueMeta(array_storage_meta(view)), PyObjPayload(view)
        )
        assert shared.meta.get(COST_META_KEY) == {"ram": backing.nbytes}
        alias = (
            Value(shared.type_id, shared.fingerprint, shared.meta, shared.payload)
            if new_envelope
            else shared
        )
        cache = MemoryLRUCache(max_bytes=backing.nbytes + 16)
        await cache.put("old-alias", {"out": shared})
        await cache.put("independent", entry(16))
        await cache.put("new-alias", {"out": alias})
        assert len(cache) == 3
        assert cache.footprint("ram") == backing.nbytes + 16
        assert await cache.shed(PressureSignal("ram", 1)) == 16
        assert await cache.get("old-alias") is None
        assert await cache.get("new-alias") == {"out": alias}
        assert cache.footprint("ram") == backing.nbytes
        assert await cache.shed(PressureSignal("ram", 1)) == backing.nbytes
        assert cache.footprint("ram") == 0

    asyncio.run(scenario())


def test_list_aliases_and_replacements_count_shared_payload_once() -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache(max_bytes=200)
        original = entry(40)["out"]
        alias = Value(original.type_id, original.fingerprint, original.meta, original.payload)
        repeated = make_list_value("core.string", [original, alias, original])
        nested = {"out": make_list_value("list<core.string>", [repeated, repeated])}
        assert entry_cost(nested, "ram") == 40
        await cache.put("first", nested)
        await cache.put("second", {"out": alias})
        assert cache.footprint("ram") == 40

        replacement = entry(40)
        assert replacement["out"].fingerprint == original.fingerprint
        assert replacement["out"].payload is not original.payload
        await cache.put("first", replacement)
        assert cache.footprint("ram") == 80
        await cache.put("second", replacement)
        assert cache.footprint("ram") == 40
        assert cache.clear() == 2
        assert cache.footprint("ram") == 0

    asyncio.run(scenario())


def test_distinct_equal_allocations_are_not_deduplicated() -> None:
    async def scenario() -> None:
        cache = MemoryLRUCache(max_bytes=100)
        first = bytearray(60)
        second = bytearray(60)
        assert first == second and first is not second
        metadata = ValueMeta({COST_META_KEY: {"ram": 60}})
        await cache.put(
            "first", {"out": Value("unknown.buffer", "equal", metadata, PyObjPayload(first))}
        )
        await cache.put(
            "second", {"out": Value("unknown.buffer", "equal", metadata, PyObjPayload(second))}
        )
        assert cache.footprint("ram") == 60
        assert await cache.get("first") is None
        assert await cache.get("second") is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("as_list", [False, True])
def test_cache_rehydration_charges_receiver_ram_per_leaf(tmp_path: Path, as_list: bool) -> None:
    async def scenario() -> None:
        data = b"encoded media bytes"
        metadata = ValueMeta(
            {"storage_dtype": "fp16", "shape": (2, 2), COST_META_KEY: {"vram:cuda:0": 4096}}
        )
        leaf = Value(
            "unknown.media",
            "producer-fingerprint",
            metadata,
            EncodedPayload("unknown.media", data, None),
        )
        output = make_list_value("unknown.media", [leaf, leaf]) if as_list else leaf
        disk = disk_store(tmp_path)
        await disk.put("key", {"out": output})
        replay = await disk_store(tmp_path).get("key")
        assert replay is not None
        assert replay["out"].fingerprint == output.fingerprint
        leaves = [
            value for value in iter_value_tree(replay["out"]) if value.type_id == leaf.type_id
        ]
        assert len(leaves) == (2 if as_list else 1)
        for value in leaves:
            assert value.fingerprint == leaf.fingerprint
            assert value.meta.get("storage_dtype") == "fp16"
            assert value.meta.get("shape") == [2, 2]
            assert value.meta.get(COST_META_KEY) == {"ram": len(data)}
        assert leaf.meta.get(COST_META_KEY) == {"vram:cuda:0": 4096}
        cache = MemoryLRUCache(max_bytes=100)
        await cache.put("key", replay)
        # CAS dedup does not alias the receiver's separately allocated payloads.
        assert cache.footprint("ram") == len(data) * len(leaves)
        assert cache.footprint("vram:cuda:0") == 0

    asyncio.run(scenario())
