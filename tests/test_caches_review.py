from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

from dinkster_caches import DiskCacheStore, DiskCAS, MemoryLRUCache
from dinkster_caches.wire import entry_from_wire
from dinkster_values import RESOURCES_META_KEY, ResourceHandle, TypeRegistry, ValueMeta


def test_cas_put_restores_corrupt_existing_blob(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path)
    data = b"correct bytes"
    digest = cas.put(data)
    path = tmp_path / digest.split(":", 1)[1][:2] / digest.split(":", 1)[1]
    path.write_bytes(b"corrupt")

    assert cas.put(data) == digest
    assert cas.get(digest) == data


def test_malformed_pickled_metadata_is_a_cache_miss() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        for malformed in (b"pickle:not a pickle", b"pickle:"):
            wire = {
                "version": 2,
                "key": "key",
                "outputs": {
                    "out": {
                        "typeId": "test.type",
                        "fingerprint": "fp",
                        "metaB64": base64.b64encode(malformed).decode("ascii"),
                        "digest": "unused",
                        "size": 0,
                    }
                },
            }

            async def fetch(_digest: str, _size: int) -> bytes | None:
                return b""

            assert await entry_from_wire(wire, fetch, registry) is None

    asyncio.run(scenario())


def test_disk_trim_remeasures_shared_blobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        registry.register("test.bulk")
        shared = registry.wrap("test.bulk", os.urandom(24 * 1024))
        store = DiskCacheStore(tmp_path, registry)
        await store.put("old-a", {"out": shared})
        await store.put("old-b", {"out": shared})
        entries = list((tmp_path / "entries").glob("*.json"))
        for index, path in enumerate(entries):
            os.utime(path, (index + 1, index + 1))
        usage = DiskCAS(tmp_path / "cas").total_bytes() + sum(
            path.stat().st_size for path in entries
        )
        budget = usage - entries[0].stat().st_size - 1
        tight = DiskCacheStore(tmp_path, registry, max_bytes=budget)
        await tight.put("new", {"out": registry.wrap("test.bulk", b"new")})

        remaining = list((tmp_path / "entries").glob("*.json"))
        actual = DiskCAS(tmp_path / "cas").total_bytes() + sum(
            path.stat().st_size for path in remaining
        )
        assert actual <= budget
        assert await tight.get("old-a") is None
        assert await tight.get("old-b") is None

    asyncio.run(scenario())


def test_value_meta_copies_source_mappings_and_residency_sequences() -> None:
    devices = ["cuda:0", "cuda:1"]
    resources: dict[str, object] = {"gpu": devices}
    source: dict[str, object] = {RESOURCES_META_KEY: resources, "shape": "old"}
    meta = ValueMeta(source)
    source["shape"] = "new"
    resources["gpu"] = "cuda:9"
    devices.append("cuda:2")

    assert meta.get("shape") == "old"
    assert meta.get(RESOURCES_META_KEY) == {"gpu": ("cuda:0", "cuda:1")}


def test_resource_handle_copies_source_mappings() -> None:
    residency: dict[str, str | tuple[str, ...]] = {"gpu": "cuda:0"}
    cost = {"vram:cuda:0": 100}
    handle = ResourceHandle("resource", "model", residency, cost)
    residency["gpu"] = "cuda:1"
    cost["vram:cuda:0"] = 200

    assert handle.residency == {"gpu": "cuda:0"}
    assert handle.cost == {"vram:cuda:0": 100}


def test_memory_cache_hit_cannot_mutate_stored_outputs() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        registry.register("test.value")
        cache = MemoryLRUCache()
        await cache.put("key", {"out": registry.wrap("test.value", 1)})
        first = await cache.get("key")
        assert first is not None
        assert isinstance(first, dict)
        first.clear()
        second = await cache.get("key")
        assert second is not None
        assert second["out"].resolve() == 1

    asyncio.run(scenario())
