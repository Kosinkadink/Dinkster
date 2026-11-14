from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from dinkster_assets import AssetError, AssetRef, digest_bytes, register_asset_type
from dinkster_caches import DiskCacheStore, DiskCAS, MemoryLRUCache
from dinkster_protocol import CachedResult, SavedArtifact
from dinkster_values import (
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    default_encode,
    make_list_value,
    make_tuple_value,
)

from tests.async_scenarios import run_scenario


def test_cas_put_restores_corrupt_existing_blob(tmp_path: Path) -> None:
    cas = DiskCAS(tmp_path)
    data = b"correct bytes"
    digest = cas.put(data)
    path = tmp_path / digest.split(":", 1)[1][:2] / digest.split(":", 1)[1]
    path.write_bytes(b"corrupt")

    assert cas.put(data) == digest
    assert cas.get(digest) == data


@pytest.mark.parametrize("declared", [None, False, True])
def test_cache_reference_inspection_requires_complete_atoms(
    tmp_path: Path, declared: bool | None
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        if declared is not None:
            registry.register("test.opaque", asset_references_complete=declared)
        value = Value(
            "test.opaque",
            "unchanged",
            ValueMeta(),
            EncodedPayload("test.opaque", b"original codec bytes", None),
        )
        outputs = {"out": make_list_value("test.opaque", [value])}
        memory = MemoryLRUCache()
        await memory.put("one", outputs)
        if declared:
            async with memory.hold_asset_references(registry) as roots:
                assert roots == set()
        else:
            with pytest.raises(AssetError, match="incomplete"):
                async with memory.hold_asset_references(registry):
                    pytest.fail("opaque RAM atoms were certified empty")
        disk = DiskCacheStore(tmp_path, registry)
        await disk.put("one", outputs)
        restarted = DiskCacheStore(tmp_path, registry)
        if declared:
            with restarted.hold_asset_references() as roots:
                assert roots == set()
        else:
            with pytest.raises(AssetError, match="incomplete"):
                with restarted.hold_asset_references():
                    pytest.fail("opaque cached atoms were certified empty")

    asyncio.run(scenario())


def test_cache_external_asset_roots_survive_restart_and_follow_invalidation(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_asset_type(registry)
        for type_id in ("VIDEO", "file"):
            registry.register(type_id, asset_references_complete=True)
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

        async with memory.hold_asset_references(registry) as roots:
            assert roots == {media.digest, asset.digest, saved.digest, preview_digest}
            assert decoy not in roots
            replacement = asyncio.create_task(replace_memory())
            await started.wait()
            assert not replacement.done()
        await replacement
        async with memory.hold_asset_references(registry) as roots:
            assert roots == {media.digest}
        disk = DiskCacheStore(tmp_path, registry)
        await disk.put("one", outputs)
        await disk.put("two", {"out": leaf})
        restarted = DiskCacheStore(tmp_path, registry)
        with restarted.hold_asset_references() as roots:
            assert roots == {media.digest, asset.digest, saved.digest, preview_digest}
        restarted.discard("one")
        with restarted.hold_asset_references() as roots:
            assert roots == {media.digest}
        restarted.clear()
        with restarted.hold_asset_references() as roots:
            assert roots == set()
        memory.clear()
        async with memory.hold_asset_references(registry) as roots:
            assert roots == set()

    run_scenario(scenario())


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

    run_scenario(scenario())
