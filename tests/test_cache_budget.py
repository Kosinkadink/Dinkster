"""Byte-bounded RAM admission and lazy content-addressed disk replay."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import PressureSignal
from dinkster_protocol import CachedResult
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
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    default_encode,
    register_core_types,
)
from dinkster_values.image_codec import image_array_fingerprint, image_array_meta
from dinkster_values.storage import image_input
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario


def entry(nbytes: int, *, device: str = "ram", text: str = "payload") -> dict[str, Value]:
    return {
        "out": Value(
            "STRING",
            text,
            ValueMeta({COST_META_KEY: {device: nbytes}}),
            EncodedPayload("STRING", default_encode(text), None),
        )
    }


def disk_store(root: Path) -> DiskCacheStore:
    registry = TypeRegistry()
    register_core_types(registry)
    return DiskCacheStore(root, registry)


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

    run_scenario(scenario())


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
                    InputSpec("mask", TypeExpr.concrete("INT")),
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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_clear_discards_dirty_values_without_persisting(tmp_path: Path) -> None:
    async def scenario() -> None:
        memory = MemoryLRUCache(max_bytes=100)
        disk = disk_store(tmp_path)
        cache = LayeredCache(memory, disk, disk_spill_min_bytes=1)
        await cache.put("dirty", entry(50))
        assert cache.clear() == 1
        assert await memory.shed(PressureSignal("ram", 100)) == 0
        assert await disk_store(tmp_path).get("dirty") is None

    run_scenario(scenario())


def test_clear_during_slow_hit_does_not_promote_stale_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        resume = asyncio.Event()

        class SlowStore:
            async def get(self, key: str) -> CachedResult | None:
                started.set()
                await resume.wait()
                return CachedResult(entry(50))

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

    run_scenario(scenario())
