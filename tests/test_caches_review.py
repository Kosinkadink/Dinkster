from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import cast

import pytest
from dinkster_assets import AssetError, AssetRef, digest_bytes
from dinkster_caches import DiskCacheStore, DiskCAS, MemoryLRUCache
from dinkster_caches.wire import entry_from_wire
from dinkster_protocol import CachedResult, SavedArtifact
from dinkster_values import (
    RESOURCES_META_KEY,
    EncodedPayload,
    ResourceHandle,
    TypeRegistry,
    Value,
    ValueMeta,
    default_encode,
    make_list_value,
    make_tuple_value,
)


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
        with pytest.raises(TypeError):
            cast("MutableMapping[str, Value]", first)["out"] = registry.wrap("test.value", 2)
        second = await cache.get("key")
        assert second is not None
        assert second["out"].resolve() == 1

    asyncio.run(scenario())


def test_cache_external_asset_roots_survive_restart_and_follow_invalidation(tmp_path: Path) -> None:
    async def scenario() -> None:
        media = AssetRef(digest_bytes(b"video source"), "clip", 12)
        asset = AssetRef(digest_bytes(b"asset source"), "data", 12)
        decoy = digest_bytes(b"not a reference")
        leaf = Value(
            "VIDEO",
            decoy,
            ValueMeta({"asset_refs": [media.to_wire()]}),
            EncodedPayload("VIDEO", b"opaque payload", None),
        )
        typed = Value(
            "asset<file>",
            asset.digest,
            ValueMeta(asset.to_wire()),
            EncodedPayload("asset<file>", b"opaque identity", None),
        )
        nested = make_tuple_value([make_list_value("VIDEO", [leaf, leaf]), typed])
        preview_digest = digest_bytes(b"saved file preview")
        saved = SavedArtifact(
            "save",
            digest_bytes(b"saved output"),
            "saved.txt",
            12,
            "text/plain",
            "out/saved.txt",
            preview=SavedArtifact(
                "save",
                preview_digest,
                "preview.mp4",
                18,
                "video/mp4",
                f"previews/{preview_digest[7:]}/preview.mp4",
            ),
        )
        outputs = CachedResult({"out": nested}, artifacts=(saved,))
        memory = MemoryLRUCache()
        await memory.put("one", outputs)
        started = asyncio.Event()

        async def replace_memory() -> None:
            started.set()
            await memory.put("one", {"out": leaf})

        async with memory.hold_asset_references() as roots:
            assert roots == {media.digest, asset.digest, saved.digest, preview_digest}
            assert decoy not in roots
            replacement = asyncio.create_task(replace_memory())
            await started.wait()
            assert not replacement.done()
        await replacement
        async with memory.hold_asset_references() as roots:
            assert roots == {media.digest}
        disk = DiskCacheStore(tmp_path, TypeRegistry())
        await disk.put("one", outputs)
        await disk.put("two", {"out": leaf})
        restarted = DiskCacheStore(tmp_path, TypeRegistry())
        with restarted.hold_asset_references() as roots:
            assert roots == {media.digest, asset.digest, saved.digest, preview_digest}
        restarted.discard("one")
        with restarted.hold_asset_references() as roots:
            assert roots == {media.digest}
        restarted.clear()
        with restarted.hold_asset_references() as roots:
            assert roots == set()
        memory.clear()
        async with memory.hold_asset_references() as roots:
            assert roots == set()

    asyncio.run(scenario())


@pytest.mark.parametrize("link", ["directory", "manifest"])
def test_cache_asset_roots_refuse_linked_reference_state(tmp_path: Path, link: str) -> None:
    async def scenario() -> None:
        cache_root = tmp_path / "cache"
        registry = TypeRegistry()
        registry.register("test.value")
        disk = DiskCacheStore(cache_root, registry)
        await disk.put("one", {"out": registry.wrap("test.value", 1)})
        original = cache_root / "entries"
        if link == "manifest":
            original = next(original.glob("*.json"))
        target = tmp_path / "outside"
        original.rename(target)
        try:
            original.symlink_to(target, target_is_directory=target.is_dir())
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        with pytest.raises(AssetError):
            with disk.hold_asset_references():
                pytest.fail("linked reference state was followed")
        assert target.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "corrupt",
    [
        "json",
        "version",
        "metadata",
        "references",
        "elements",
        "missing",
        "artifacts",
        "preview",
        "missing_artifacts",
    ],
)
def test_cache_asset_roots_refuse_unreadable_state(tmp_path: Path, corrupt: str) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        registry.register("test.value")
        disk = DiskCacheStore(tmp_path, registry)
        await disk.put("one", {"out": registry.wrap("test.value", 1)})
        path = next((tmp_path / "entries").glob("*.json"))
        wire = json.loads(path.read_text())
        if corrupt == "missing":
            path.unlink()
            path.parent.rmdir()
        elif corrupt == "json":
            path.write_text("{")
        else:
            if corrupt == "version":
                wire["version"] = 100
            elif corrupt == "metadata":
                wire["outputs"]["out"]["metaB64"] = "not base64"
            elif corrupt == "references":
                wire["outputs"]["out"]["metaB64"] = base64.b64encode(
                    default_encode({"asset_refs": [{}]})
                ).decode("ascii")
            elif corrupt in {"artifacts", "preview", "missing_artifacts"}:
                wire["version"] = 4
                if corrupt == "artifacts":
                    wire["artifacts"] = [{}]
                elif corrupt == "preview":
                    wire["artifacts"] = [
                        {
                            "node_id": "save",
                            "digest": digest_bytes(b"saved file"),
                            "name": "saved.txt",
                            "size": 10,
                            "media_type": "text/plain",
                            "virtual_path": "out/saved.txt",
                            "preview": {},
                        }
                    ]
            else:
                wire["outputs"]["out"]["elements"] = None
            path.write_text(json.dumps(wire))
        with pytest.raises(AssetError):
            with disk.hold_asset_references():
                pytest.fail("unreadable references were treated as empty")

    asyncio.run(scenario())
