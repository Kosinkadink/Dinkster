"""Filesystem mounts: N explicit directory grants, runtime-mutable.

The mount table replaces ComfyUI's single input/output directory with an
explicit, operator-granted set: real paths never enter workflows (files
catalog under ``mounts/<id>/...`` and travel as digests), the table is
LIVE (grant/revoke while running, mounts.toml is the durable record
underneath), and workers see changes through the published snapshot file
without a restart or a control-channel message.
"""

from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import mount_roots_for_kind
from dinkster_assets import (
    AssetError,
    AssetScanProgress,
    AssetVault,
    LibraryStore,
    LocalAssetLibrary,
    MountDef,
    MountsError,
    MountSnapshotResolver,
    MountTable,
    digest_bytes,
    dump_mounts,
    load_mounts,
    load_output_mount,
    parse_mounts,
)
from dinkster_assets.integrity import digest_file_with_record as real_digest_file_with_record
from dinkster_assets.model import AssetRef
from dinkster_assets.resolution import MountMaterialization, ResolutionStore
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_protocol import AssetFolderContext
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import ServerLibrary, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker, load_manifest
from dinkster_workers.host import load_pack

from dinkster.mounts_api import MountService, add_mount_routes

STRING = TypeExpr.concrete("STRING")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


NODES = (Echo,)
SCHEMAS = build_schemas(NODES)


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


def seed(root: Path, files: Mapping[str, bytes]) -> None:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


# --- config: parse / dump ----------------------------------------------------


def test_mount_priority_validation_and_config_round_trip() -> None:
    mounts = (
        MountDef(
            id="renders",
            path=Path("/data/renders"),
            mode="readwrite",
            priority=-4,
        ),
        MountDef(id="models-main", path=Path("C:\\Users\\me\\models"), kind="model/vae"),
    )
    text = dump_mounts(mounts)
    assert parse_mounts(text) == tuple(sorted(mounts, key=lambda m: m.id))
    # Defaults are omitted: a read mount writes no mode line.
    assert 'mode = "readwrite"' in text
    assert text.count("mode =") == 1
    assert "priority = -4" in text
    assert text.count("priority =") == 1
    assert 'kind = "model/vae"' in text
    assert parse_mounts('[mounts.default]\npath = "/data"') == (
        MountDef(id="default", path=Path("/data"), priority=0),
    )

    for invalid in (True, False, "1", 1.0, None):
        with pytest.raises(MountsError, match="priority must be an integer"):
            MountDef(id="invalid", path=Path("/data"), priority=invalid)  # type: ignore[arg-type]
    for invalid_toml in ("true", '"1"'):
        with pytest.raises(MountsError, match="priority must be an integer"):
            parse_mounts(f'[mounts.invalid]\npath = "/data"\npriority = {invalid_toml}')


def test_parse_is_strict() -> None:
    with pytest.raises(MountsError, match="unknown top-level"):
        parse_mounts('[mount.x]\npath = "/a"')
    with pytest.raises(MountsError, match="unknown keys"):
        parse_mounts('[mounts.x]\npath = "/a"\nwritable = true')
    with pytest.raises(MountsError, match="lowercase"):
        parse_mounts('[mounts.BadId]\npath = "/a"')
    with pytest.raises(MountsError, match="mode"):
        parse_mounts('[mounts.x]\npath = "/a"\nmode = "rw"')
    with pytest.raises(MountsError, match="non-empty path"):
        parse_mounts('[mounts.x]\npath = ""')
    with pytest.raises(MountsError, match="already mounted"):
        parse_mounts('[mounts.a]\npath = "/same"\n[mounts.b]\npath = "/same"')
    with pytest.raises(MountsError, match="invalid TOML"):
        parse_mounts("[mounts.x\n")
    with pytest.raises(MountsError, match="output mount 'missing' is not configured"):
        parse_mounts('[settings]\noutput-mount = "missing"')
    with pytest.raises(MountsError, match="must be readwrite"):
        parse_mounts('[settings]\noutput-mount = "readonly"\n[mounts.readonly]\npath = "/data"')
    for kind in ("true", "42", '"../models"', '"model/*"'):
        with pytest.raises(MountsError, match="kind"):
            parse_mounts(f'[mounts.invalid]\npath = "/data"\nkind = {kind}')


def test_load_missing_file_is_empty_table(tmp_path: Path) -> None:
    assert load_mounts(tmp_path / "mounts.toml") == ()


def test_output_mount_config_round_trip(tmp_path: Path) -> None:
    output = MountDef(id="output", path=tmp_path / "output", mode="readwrite")
    path = tmp_path / "mounts.toml"
    path.write_text(dump_mounts((output,), output_mount="output"), "utf-8")

    assert load_mounts(path) == (output,)
    assert load_output_mount(path) == "output"


# --- the live table ----------------------------------------------------------


def test_scan_catalogs_under_mount_namespace(tmp_path: Path) -> None:
    root = tmp_path / "granted"
    seed(root, {"a.png": b"aaa", "sub/b.png": b"bbb"})
    table = MountTable()
    table.add(MountDef(id="pics", path=root))
    assert table.pending() == ("pics",)

    assert table.scan("pics") == 2
    entries = table.entries("pics")
    assert [e.virtual_path for e in entries] == [
        "mounts/pics/a.png",
        "mounts/pics/sub/b.png",
    ]
    # Identity is content; the ref materializes to the real file.
    ref = table.ref("mounts/pics/a.png")
    assert ref.digest == digest_bytes(b"aaa")
    assert ref.local_path() == root / "a.png"
    assert table.resolve(ref.digest) == root / "a.png"
    # Folder navigation, one level at a time.
    listing = table.list_folder("pics")
    assert listing.folders == ("sub",)
    assert [e.name for e in listing.entries] == ["a.png"]

    (row,) = table.descriptors()
    assert row["id"] == "pics"
    assert row["state"] == "ready"
    assert row["entryCount"] == 2
    assert row["source"] == "config"
    assert row["path"] == str(root)
    assert row["priority"] == 0


def test_mount_retains_missing_identity_and_recovers_it_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"weights.bin": b"original weights"})
    index_root = tmp_path / "indexes"
    table = MountTable(index_root=index_root)
    table.add(MountDef(id="models", path=root))
    assert table.scan("models") == 1
    old = table.ref("mounts/models/weights.bin")
    (root / "weights.bin").unlink()
    assert table.scan("models") == 0
    assert table.entries("models") == ()
    (missing,) = table.missing_entries("models")
    assert missing.digest == old.digest
    assert missing.metadata["missing"] is True
    assert table.resolve(old.digest) is None
    assert table.descriptors()[0]["missingEntryCount"] == 1

    restored = MountTable(index_root=index_root)
    restored.add(MountDef(id="models", path=root))
    assert restored.scan("models") == 0
    assert restored.missing_entries("models") == (missing,)
    (root / "weights.bin").write_bytes(b"original weights")
    assert restored.scan("models") == 1
    assert restored.missing_entries("models") == ()
    assert restored.ref("mounts/models/weights.bin").digest == old.digest
    assert restored.descriptors()[0]["missingEntryCount"] == 0


def test_mount_splits_changed_content_without_reusing_old_identity(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"weights.bin": b"first"})
    table = MountTable(index_root=tmp_path / "indexes")
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    old = table.ref("mounts/models/weights.bin")
    (root / "weights.bin").write_bytes(b"second weights")
    table.scan("models")
    replacement = table.ref("mounts/models/weights.bin")
    assert replacement.digest == digest_bytes(b"second weights")
    assert replacement.digest != old.digest
    assert table.resolve(old.digest) is None
    assert [entry.digest for entry in table.missing_entries("models")] == [old.digest]
    table.scan("models")
    assert [entry.digest for entry in table.missing_entries("models")] == [old.digest]
    (root / "weights.bin").write_bytes(b"first")
    table.scan("models")
    assert table.ref("mounts/models/weights.bin").digest == old.digest
    assert [entry.digest for entry in table.missing_entries("models")] == [replacement.digest]


def test_scan_publishes_cached_assets_and_progress_before_changed_file_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    seed(root, {"cached.bin": b"cached"})
    snapshot = tmp_path / "library" / "worker-state" / "mounts.json"
    index_root = tmp_path / "library" / "asset-indexes"
    table = MountTable(snapshot, index_root=index_root)
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    cached_digest = digest_bytes(b"cached")

    slow = root / "slow.bin"
    slow.write_bytes(b"slow payload")
    import dinkster_assets.library as library_module

    hashing = Event()
    release = Event()

    def held_digest(path: Path):
        if path == slow:
            hashing.set()
            assert release.wait(5)
        return real_digest_file_with_record(path)

    monkeypatch.setattr(library_module, "digest_file_with_record", held_digest)
    persist_calls = 0
    real_persist_index = library_module.LocalAssetLibrary.persist_index

    def counted_persist_index(library: LocalAssetLibrary) -> None:
        nonlocal persist_calls
        persist_calls += 1
        real_persist_index(library)

    monkeypatch.setattr(library_module.LocalAssetLibrary, "persist_index", counted_persist_index)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(table.scan, "models", progress_interval=0)
        assert hashing.wait(5)

        assert table.ref("mounts/models/cached.bin").digest == cached_digest
        persists_before_refs = persist_calls
        assert table.ref("mounts/models/cached.bin").digest == cached_digest
        assert persist_calls == persists_before_refs
        assert MountSnapshotResolver(snapshot).resolve(cached_digest) == root / "cached.bin"
        with pytest.raises(AssetError, match="mounts/models/slow.bin.*not indexed"):
            table.ref("mounts/models/slow.bin")
        (descriptor,) = table.descriptors()
        progress = cast("dict[str, object]", descriptor["scanProgress"])
        assert descriptor["state"] == "scanning"
        assert descriptor["entryCount"] == 1
        assert progress["filesDone"] == 1
        assert progress["filesTotal"] == 2
        assert progress["bytesDone"] == len(b"cached")
        assert progress["bytesTotal"] == len(b"cached") + len(b"slow payload")
        assert isinstance(progress["elapsedSeconds"], float)

        release.set()
        assert future.result(timeout=5) == 2


def test_rescan_keeps_verified_assets_available_before_cache_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"cached.bin": b"cached"})
    snapshot = tmp_path / "library" / "worker-state" / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "library" / "asset-indexes")
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    digest = digest_bytes(b"cached")
    initial_progress = Event()
    release = Event()

    def hold_initial_progress(progress: AssetScanProgress) -> None:
        if progress.files_done == 0:
            initial_progress.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(table.scan, "models", on_progress=hold_initial_progress)
        assert initial_progress.wait(5)
        assert table.ref("mounts/models/cached.bin").digest == digest
        assert MountSnapshotResolver(snapshot).resolve(digest) == root / "cached.bin"
        release.set()
        assert future.result(timeout=5) == 1


def test_mount_indexes_live_under_library_root_and_migrate_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    seed(root, {"model.bin": b"weights"})
    legacy = root / ".dinkster-asset-index.json"
    legacy_table = MountTable()
    legacy_table.add(MountDef(id="models", path=root))
    legacy_table.scan("models")
    legacy_bytes = legacy.read_bytes()

    import dinkster_assets.library as library_module

    def reject_hash(path: Path) -> tuple[str, None]:
        raise AssertionError(f"migration rehashed unchanged file: {path}")

    monkeypatch.setattr(library_module, "digest_file_with_record", reject_hash)
    library_root = tmp_path / "library"
    index_root = library_root / "asset-indexes"
    table = MountTable(library_root / "mounts.json", index_root=index_root)
    table.add(MountDef(id="models", path=root))
    assert table.scan("models") == 1
    central = index_root / "models.json"
    assert central.is_file()
    assert legacy.read_bytes() == legacy_bytes

    legacy.write_text("{}", "utf-8")
    restarted = MountTable(library_root / "restart.json", index_root=index_root)
    restarted.add(MountDef(id="models", path=root))
    assert restarted.scan("models") == 1
    assert restarted.ref("mounts/models/model.bin").digest == digest_bytes(b"weights")


def test_mount_kind_is_validated_and_published_to_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    seed(root, {"model.safetensors": b"weights"})
    snapshot = tmp_path / "mounts-snapshot.json"
    table = MountTable(snapshot)
    table.add(
        MountDef(id="comfy-model-checkpoints-1", path=root),
        source="derived",
        kind="model/checkpoint",
    )
    assert table.kind("comfy-model-checkpoints-1") == "model/checkpoint"
    assert table.mounts_for_kind("model/checkpoint") == ("comfy-model-checkpoints-1",)
    table.scan("comfy-model-checkpoints-1")
    (descriptor,) = table.descriptors()
    assert descriptor["kind"] == "model/checkpoint"
    assert "path" not in descriptor
    (snapshot_row,) = json.loads(snapshot.read_text("utf-8"))["mounts"]
    assert snapshot_row["kind"] == "model/checkpoint"
    assert MountSnapshotResolver(snapshot).resolve(digest_bytes(b"weights")) == (
        root / "model.safetensors"
    )

    with pytest.raises(MountsError, match="kind"):
        table.add(
            MountDef(id="bad-kind", path=root),
            kind="checkpoint",
        )

    missing = tmp_path / "private-missing-model-root"
    hidden = MountTable()
    hidden.add(
        MountDef(id="missing-model", path=missing),
        source="derived",
        kind="model/checkpoint",
    )
    with pytest.raises(AssetError):
        hidden.scan("missing-model")
    (failed,) = hidden.descriptors()
    assert failed["error"] == "model root unavailable"
    assert str(missing) not in json.dumps(failed)


def test_snapshot_resolver_filters_kind_and_ranks_duplicate_digest_roots(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "mounts-snapshot.json"
    checkpoint_one = tmp_path / "checkpoints-one"
    checkpoint_two = tmp_path / "checkpoints-two"
    loras = tmp_path / "loras"
    payload = b"same model bytes"
    seed(checkpoint_one, {"first.safetensors": payload})
    seed(checkpoint_two, {"second.safetensors": payload})
    seed(loras, {"copy.safetensors": payload})
    digest = digest_bytes(payload)

    table = MountTable(snapshot)
    table.add(
        MountDef(id="checkpoint-one", path=checkpoint_one, priority=2),
        kind="model/checkpoint",
    )
    table.add(MountDef(id="loras", path=loras), kind="model/lora")
    table.scan("checkpoint-one")
    table.scan("loras")
    resolver = MountSnapshotResolver(snapshot)
    assert resolver.resolve_for_kind(digest, "model/checkpoint") == (
        checkpoint_one / "first.safetensors"
    )
    assert resolver.resolve_for_kind(digest, "model/lora") == (loras / "copy.safetensors")

    table.add(
        MountDef(id="checkpoint-two", path=checkpoint_two, priority=-1),
        kind="model/checkpoint",
    )
    table.scan("checkpoint-two")
    assert resolver.resolve_for_kind(digest, "model/checkpoint") == (
        checkpoint_two / "second.safetensors"
    )
    assert resolver.resolve_for_kind(digest, "model/lora") == (loras / "copy.safetensors")


def test_add_remove_refusals(tmp_path: Path) -> None:
    table = MountTable()
    table.add(MountDef(id="a", path=tmp_path))
    with pytest.raises(MountsError, match="already exists"):
        table.add(MountDef(id="a", path=tmp_path / "elsewhere"))
    with pytest.raises(MountsError, match="unknown mount"):
        table.remove("nope")
    with pytest.raises(MountsError, match="unknown mount"):
        table.scan("nope")
    with pytest.raises(AssetError, match="mounts/a/x.png.*not ready"):
        table.ref("mounts/a/x.png")  # registered but never scanned
    with pytest.raises(AssetError, match="not a mount virtual path"):
        table.ref("models/x.png")


def test_missing_directory_is_a_failed_row_not_a_crash(tmp_path: Path) -> None:
    table = MountTable()
    table.add(MountDef(id="gone", path=tmp_path / "unplugged"))
    with pytest.raises(AssetError):
        table.scan("gone")
    (row,) = table.descriptors()
    assert row["state"] == "failed"
    assert "unplugged" in str(row["error"])
    # A failed mount serves nothing but stays visible and re-scannable.
    assert table.entries("gone") == ()
    seed(tmp_path / "unplugged", {"late.txt": b"now it exists"})
    assert table.scan("gone") == 1
    (row,) = table.descriptors()
    assert row["state"] == "ready"


def test_ready_snapshot_is_atomic_immutable_and_excludes_nonready_rows(tmp_path: Path) -> None:
    ready_root = tmp_path / "ready"
    seed(ready_root, {"model.bin": b"ready"})
    table = MountTable()
    table.add(MountDef(id="ready", path=ready_root, priority=-1), kind="model/checkpoint")
    table.add(MountDef(id="pending", path=tmp_path / "pending"))
    table.add(MountDef(id="failed", path=tmp_path / "failed"))
    table.scan("ready")
    with pytest.raises(AssetError):
        table.scan("failed")

    (row,) = table.ready_snapshot()
    assert (row.mount_id, row.priority, row.asset_kind) == (
        "ready",
        -1,
        "model/checkpoint",
    )
    assert [ref.virtual_path for ref in row.refs] == ["mounts/ready/model.bin"]
    with pytest.raises(FrozenInstanceError):
        row.mount_id = "changed"  # type: ignore[misc]


def test_config_defs_exclude_derived_mounts(tmp_path: Path) -> None:
    table = MountTable()
    configured = MountDef(id="mine", path=tmp_path)
    table.add(configured, source="config")
    table.add(MountDef(id="comfy-input", path=tmp_path / "x"), source="derived")
    assert table.config_defs() == (configured,)


# --- the worker snapshot: runtime changes without a restart -------------------


def test_snapshot_resolver_follows_runtime_mount_changes(tmp_path: Path) -> None:
    snapshot = tmp_path / "mounts-snapshot.json"
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    seed(root_a, {"one.bin": b"content-one"})
    seed(root_b, {"two.bin": b"content-two"})
    digest_a = digest_bytes(b"content-one")
    digest_b = digest_bytes(b"content-two")

    table = MountTable(snapshot)
    resolver = MountSnapshotResolver(snapshot)
    # Before any scan there is no snapshot: resolves nothing, no error.
    assert resolver.resolve(digest_a) is None

    table.add(MountDef(id="a", path=root_a))
    table.scan("a")
    assert resolver.resolve(digest_a) == root_a / "one.bin"
    assert resolver.resolve(digest_b) is None

    # A mount granted AFTER the resolver was constructed becomes visible
    # to the same resolver instance: this is the live-worker story.
    table.add(MountDef(id="b", path=root_b))
    table.scan("b")
    assert resolver.resolve(digest_b) == root_b / "two.bin"

    # Revocation is equally live.
    table.remove("a")
    assert resolver.resolve(digest_a) is None
    assert resolver.resolve(digest_b) == root_b / "two.bin"


def test_snapshot_resolver_refresh_is_atomic_across_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "mounts-snapshot.json"
    root = tmp_path / "models"
    seed(root, {"model.bin": b"weights"})
    digest = digest_bytes(b"weights")
    table = MountTable(snapshot)
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    resolver = MountSnapshotResolver(snapshot)
    entered = Event()
    release = Event()
    original_read_text = Path.read_text

    def gated_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == snapshot and not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return original_read_text(path, encoding, errors)

    monkeypatch.setattr(Path, "read_text", gated_read_text)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(resolver.resolve, digest)
        assert entered.wait(timeout=5)
        second = executor.submit(resolver.resolve, digest)
        with pytest.raises(FutureTimeoutError):
            second.result(timeout=0.1)
        release.set()
        assert first.result(timeout=5) == root / "model.bin"
        assert second.result(timeout=5) == root / "model.bin"


def test_mount_priority_orders_engine_and_worker_resolution_deterministically(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "mounts-snapshot.json"
    payload = b"identical content"
    roots = {
        "a-tie": tmp_path / "a-tie",
        "b-tie": tmp_path / "b-tie",
        "z-lower": tmp_path / "z-lower",
    }
    for mount_id, root in roots.items():
        seed(root, {f"{mount_id}.bin": payload})
    digest = digest_bytes(payload)
    mounts = (
        MountDef(id="b-tie", path=roots["b-tie"], priority=2),
        MountDef(id="z-lower", path=roots["z-lower"], priority=-1),
        MountDef(id="a-tie", path=roots["a-tie"], priority=2),
    )

    table = MountTable(snapshot)
    for mount in mounts:  # deliberately not priority or id order
        table.add(mount, kind="model/checkpoint")
    for mount_id in ("a-tie", "z-lower", "b-tie"):  # reverse scan order
        table.scan(mount_id)
    resolver = MountSnapshotResolver(snapshot)

    assert table.mounts_for_kind("model/checkpoint") == (
        "z-lower",
        "a-tie",
        "b-tie",
    )
    assert table.resolve(digest) == roots["z-lower"] / "z-lower.bin"
    assert resolver.resolve(digest) == roots["z-lower"] / "z-lower.bin"
    assert resolver.resolve_for_kind(digest, "model/checkpoint") == (
        roots["z-lower"] / "z-lower.bin"
    )
    assert [row["id"] for row in json.loads(snapshot.read_text())["mounts"]] == [
        "z-lower",
        "a-tie",
        "b-tie",
    ]
    assert [row["priority"] for row in json.loads(snapshot.read_text())["mounts"]] == [-1, 2, 2]

    table.remove("z-lower")
    assert table.resolve(digest) == roots["a-tie"] / "a-tie.bin"
    assert resolver.resolve(digest) == roots["a-tie"] / "a-tie.bin"

    # A restart from reversed config order and another scan order has the
    # same winner; neither TOML position nor insertion/scan order is authority.
    reversed_config = "\n".join(dump_mounts((mount,)).strip() for mount in reversed(mounts))
    restarted = MountTable(tmp_path / "restart-snapshot.json")
    for mount in parse_mounts(reversed_config):
        restarted.add(mount, kind="model/checkpoint")
    for mount in mounts:
        restarted.scan(mount.id)
    assert restarted.resolve(digest) == roots["z-lower"] / "z-lower.bin"
    assert (
        MountSnapshotResolver(tmp_path / "restart-snapshot.json").resolve(digest)
        == roots["z-lower"] / "z-lower.bin"
    )


def test_snapshot_priority_backward_compatibility_and_malformed_row_tolerance(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "mounts-snapshot.json"
    payload = b"same bytes"
    good_root = tmp_path / "z-good"
    malformed_root = tmp_path / "a-malformed"
    seed(good_root, {"good.bin": payload})
    seed(malformed_root, {"malformed.bin": payload})
    table = MountTable(snapshot)
    table.add(MountDef(id="z-good", path=good_root))
    table.add(MountDef(id="a-malformed", path=malformed_root, priority=-5))
    table.scan("z-good")
    table.scan("a-malformed")

    body = json.loads(snapshot.read_text("utf-8"))
    rows = {row["id"]: row for row in body["mounts"]}
    rows["z-good"].pop("priority")  # old snapshots mean default priority zero
    rows["a-malformed"]["priority"] = True
    snapshot.write_text(
        json.dumps({"mounts": [rows["a-malformed"], rows["z-good"]]}),
        "utf-8",
    )

    assert MountSnapshotResolver(snapshot).resolve(digest_bytes(payload)) == (
        good_root / "good.bin"
    )


def test_snapshot_resolver_tolerates_malformed_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    snapshot.write_text("{not json", "utf-8")
    resolver = MountSnapshotResolver(snapshot)
    assert resolver.resolve(digest_bytes(b"whatever")) is None
    snapshot.write_text(json.dumps({"mounts": "nope"}), "utf-8")
    assert resolver.resolve(digest_bytes(b"whatever")) is None


def test_mount_roots_are_kind_scoped_ranked_and_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot.json"
    env = {"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    resolver = MountSnapshotResolver(snapshot)
    assert resolver.roots_for_kind("model/embedding") == ()
    assert mount_roots_for_kind("model/embedding", env=env) == ()
    table = MountTable(snapshot)
    roots = {name: tmp_path / name for name in ("b-tie", "a-tie", "z-first", "other")}
    for name, priority, kind in (
        ("b-tie", 3, "model/embedding"),
        ("other", -20, "model/checkpoint"),
        ("a-tie", 3, "model/embedding"),
        ("z-first", -2, "model/embedding"),
    ):
        roots[name].mkdir()
        table.add(MountDef(id=name, path=roots[name], priority=priority), kind=kind)
        table.scan(name)
    # Snapshot row order is not the authority: priority and id are.
    body = json.loads(snapshot.read_text())
    body["mounts"].reverse()
    snapshot.write_text(json.dumps(body))
    expected = (roots["z-first"], roots["a-tie"], roots["b-tie"])
    captured = resolver.roots_for_kind("model/embedding")
    assert captured == expected
    assert mount_roots_for_kind("model/embedding") == expected
    assert mount_roots_for_kind("model/checkpoint", env=env) == (roots["other"],)
    assert mount_roots_for_kind("model/embedding/child", env=env) == ()
    # Directory authority survives an unavailable file index.
    for row in body["mounts"]:
        Path(row["index"]).unlink()
    assert MountSnapshotResolver(snapshot).roots_for_kind("model/embedding") == expected
    table.remove("z-first")
    assert captured == expected
    assert resolver.roots_for_kind("model/embedding") == expected[1:]
    assert mount_roots_for_kind("model/embedding", env=env) == expected[1:]
    snapshot.unlink()
    assert resolver.roots_for_kind("model/embedding") == ()
    assert mount_roots_for_kind("model/embedding", env=env) == ()


def test_mount_roots_do_not_infer_unconfigured_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    monkeypatch.setenv("DINKSTER_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    assert mount_roots_for_kind("model/embedding") == ()
    assert mount_roots_for_kind("model/embedding", env={}) == ()
    snapshot = tmp_path / "malformed.json"
    resolver = MountSnapshotResolver(snapshot)
    env = {"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
    for text in ("{not json", "[]", '{"mounts": "nope"}', '{"mounts": [{}]}'):
        snapshot.write_text(text)
        assert resolver.roots_for_kind("model/embedding") == ()
        assert mount_roots_for_kind("model/embedding", env=env) == ()
    with pytest.raises(AssetError, match="not a valid asset kind"):
        mount_roots_for_kind("embedding", env={})
    with pytest.raises(AssetError, match="not a valid asset kind"):
        resolver.roots_for_kind("model/*")


@pytest.mark.parametrize("isolated", [False, True])
def test_public_mount_roots_in_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated: bool
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest_path = pack / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "root-probe"\npublisher = "dinkster"\n'
        'namespaces = ["rootprobe"]\n[pack.entry]\nnodes = "root_probe:NODES"\n'
    )
    (pack / "root_probe.py").write_text(
        "import json\n"
        "from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr, mount_roots_for_kind\n"
        "class Probe(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='rootprobe.read', inputs=(), "
        "outputs=(OutputSpec('roots', TypeExpr.concrete('STRING')),))\n"
        "    @classmethod\n"
        "    def execute(cls):\n"
        "        roots = mount_roots_for_kind('model/embedding')\n"
        "        return cls.outputs(roots=json.dumps([(str(root), "
        "(root / 'sentinel.txt').read_text()) for root in roots]))\n"
        "NODES = (Probe,)\n"
    )
    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker: InProcessWorker | IsolatedWorker
        if isolated:
            worker = IsolatedWorker(
                manifest_path, registry, extra_env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
            )
            await worker.start()
        else:
            worker, registry, _, _ = load_pack(
                load_manifest(manifest_path), import_from_pack_root=True, registry=registry
            )
        try:
            expected: list[list[str]] = []
            for name, priority, kind, text in (
                ("late", 4, "model/embedding", "late embedding"),
                ("unrelated", -9, "model/checkpoint", "not an embedding"),
                ("first", -1, "model/embedding", "preferred embedding"),
            ):
                seed(tmp_path / name, {"sentinel.txt": text.encode()})
                table.add(MountDef(id=name, path=tmp_path / name, priority=priority), kind=kind)
                table.scan(name)
                if kind == "model/embedding":
                    expected.insert(0, [str(tmp_path / name), text])
                engine = Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    cache=MemoryLRUCache(),
                )
                result = await engine.run(
                    Graph(nodes={"read": GraphNode("rootprobe.read", {})}), ["read"]
                )
                actual = result.outputs["read"]["roots"].resolve()
                assert isinstance(actual, str)
                assert json.loads(actual) == expected
            graph = Graph(nodes={"read": GraphNode("rootprobe.read", {})})
            base = engine.pin_execution()
            contexts = (
                AssetFolderContext(("late", "first")),
                AssetFolderContext(("first", "late")),
                AssetFolderContext(()),
            )
            results = await asyncio.gather(
                *(
                    engine.run(graph, ["read"], execution=replace(base, asset_folders=context))
                    for context in contexts
                )
            )
            assert [
                json.loads(cast(str, result.outputs["read"]["roots"].resolve()))
                for result in results
            ] == [
                [
                    [str(tmp_path / "late"), "late embedding"],
                    [str(tmp_path / "first"), "preferred embedding"],
                ],
                [
                    [str(tmp_path / "first"), "preferred embedding"],
                    [str(tmp_path / "late"), "late embedding"],
                ],
                [],
            ]
            service = MountService(table, tmp_path / "mounts.toml", allow_changes=True)
            service.projects.update(
                "project-a", ["late", "first"], available=service.mount_ids(), output=None
            )
            service.projects.update(
                "project-b", ["first", "late"], available=service.mount_ids(), output=None
            )
            store = LibraryStore(tmp_path / "library.sqlite")
            library = ServerLibrary(
                AssetVault(tmp_path / "vault"),
                store,
                resolver=table,
                project_folders_for=service.project_context,
            )

            def serve_engine(on_event: EventListener | None = None) -> Engine:
                return Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    cache=MemoryLRUCache(),
                    on_event=on_event,
                )

            app = create_app(
                serve_engine, dict(worker.schemas), library=library, max_running_jobs=2
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                for project in ("project-a", "project-b"):
                    response = await client.post(
                        "/api/jobs",
                        json={
                            "clientId": "projects",
                            "jobId": project,
                            "project": project,
                            "graph": graph_to_wire(graph),
                            "targets": ["read"],
                        },
                    )
                    assert response.status == 202, await response.text()
                async with asyncio.timeout(10):
                    for project, first_name in (("project-a", "late"), ("project-b", "first")):
                        while True:
                            status = await (
                                await client.get(f"/api/jobs/projects/{project}")
                            ).json()
                            assert status["state"] != "failed", status
                            if status["state"] == "completed":
                                actual = json.loads(status["outputs"]["read"]["roots"]["value"])
                                assert actual[0][0] == str(tmp_path / first_name)
                                break
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "projects",
                        "jobId": "bad",
                        "project": "Bad Project",
                        "graph": graph_to_wire(graph),
                        "targets": ["read"],
                    },
                )
                assert response.status == 400
            finally:
                await client.close()
                store.close()
            # A project preference never grants a revoked root, even in a pinned run context.
            table.remove("late")
            result = await engine.run(
                graph,
                ["read"],
                execution=replace(base, asset_folders=contexts[0]),
                cache_enabled=False,
            )
            assert json.loads(cast(str, result.outputs["read"]["roots"].resolve())) == [
                [str(tmp_path / "first"), "preferred embedding"]
            ]
        finally:
            await worker.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("operation", ["add", "add-project", "remove", "output", "project"])
def test_failed_folder_persistence_preserves_live_and_saved_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart: bool, operation: str
) -> None:
    import os

    config = tmp_path / "mounts.toml"
    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot, output_mount="first")
    for name in ("first", "second"):
        seed(tmp_path / name, {"file.bin": name.encode()})
        table.add(MountDef(name, tmp_path / name, "readwrite"))
        table.scan(name)
    service = MountService(table, config, allow_changes=True, restart_for_changes=restart)
    service.persist()
    before = config.read_bytes()
    snapshot_before = snapshot.read_bytes()
    replace = os.replace

    def fail_settings_replace(source: object, destination: object) -> None:
        target = tmp_path / (
            "project-folders.json" if operation in ("project", "add-project") else "mounts.toml"
        )
        if destination == target:
            raise OSError("settings volume is unavailable")
        replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", fail_settings_replace)

    async def scenario() -> None:
        client = await make_client(service)
        try:
            if operation in ("add", "add-project"):
                seed(tmp_path / "third", {"file.bin": b"third"})
                response = await client.post(
                    "/api/mounts",
                    json={"id": "third", "path": str(tmp_path / "third"), "mode": "readwrite"},
                )
            elif operation == "remove":
                response = await client.delete("/api/mounts/second")
            elif operation == "output":
                response = await client.put("/api/mounts/output", json={"id": "second"})
            else:
                response = await client.put(
                    "/api/mounts/projects/default",
                    json={"mounts": ["second"], "outputMount": "second"},
                )
            assert response.status == 500
            current = await (await client.get("/api/mounts")).json()
            assert [row["id"] for row in current["mounts"]] == ["first", "second"]
            assert current["outputMount"] == "first"
            assert current["restartRequired"] is False
            assert snapshot.read_bytes() == snapshot_before
            assert config.read_bytes() == before
            assert (await client.get("/api/mounts/second/entries")).status == 200
        finally:
            await service.close()
            await client.close()

    asyncio.run(scenario())


# --- the API surface ----------------------------------------------------------


async def make_client(service: MountService) -> TestClient:
    app = create_app(make_engine, SCHEMAS)
    add_mount_routes(app, service)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_selection_options_api_uses_live_grants_without_exposing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot.json"
    root = tmp_path / "private-model-directory"
    seed(root, {"vendor/model/model_index.json": b"{}"})
    table = MountTable(snapshot)
    (mount,) = parse_mounts(
        dump_mounts((MountDef(id="diffusers", path=root, kind="model/diffusers"),))
    )
    table.add(mount)
    table.scan("diffusers")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        client = await make_client(MountService(table))
        try:
            response = await client.get("/api/assets/selections?kind=model/diffusers")
            assert response.status == 200
            assert await response.json() == {
                "kind": "model/diffusers",
                "options": [{"name": "vendor/model", "shape": "directory"}],
            }
            table.remove("diffusers")
            response = await client.get("/api/assets/selections?kind=model/diffusers")
            assert await response.json() == {"kind": "model/diffusers", "options": []}
            for kind in ("", "../private-model-directory", "model/checkpoint"):
                response = await client.get("/api/assets/selections", params={"kind": kind})
                assert response.status == 400
                assert await response.json() == {"error": "unsupported-selection-kind"}
        finally:
            await client.close()

    asyncio.run(scenario())


async def wait_for_state(
    client: TestClient,
    mount_id: str,
    state: str,
    timeout: float = 5.0,
    *,
    project: str = "default",
) -> None:
    async with asyncio.timeout(timeout):
        while True:
            resp = await client.get("/api/mounts", params={"project": project})
            rows = (await resp.json())["mounts"]
            row = next((r for r in rows if r["id"] == mount_id), None)
            if row is not None and row["state"] == state:
                return
            await asyncio.sleep(0.02)


def test_mounts_read_surface(tmp_path: Path) -> None:
    root = tmp_path / "granted"
    seed(
        root,
        {
            "cat.png": b"cat",
            "dog.png": b"dog",
            "notes/readme.txt": b"hello",
        },
    )
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="pics", path=root))
    table.scan("pics")

    async def scenario() -> None:
        client = await make_client(MountService(table))
        resp = await client.get("/api/mounts")
        assert resp.status == 200
        listing = await resp.json()
        assert listing["mountChangesAllowed"] is False
        (row,) = listing["mounts"]
        assert (row["id"], row["state"], row["entryCount"]) == ("pics", "ready", 3)

        # Folder navigation.
        resp = await client.get("/api/mounts/pics/list")
        body = await resp.json()
        assert body["folders"] == ["notes"]
        assert [e["name"] for e in body["entries"]] == ["cat.png", "dog.png"]
        assert [e["kind"] for e in body["entries"]] == ["media/image", "media/image"]
        resp = await client.get("/api/mounts/pics/list", params={"path": "notes"})
        body = await resp.json()
        assert [e["virtualPath"] for e in body["entries"]] == ["mounts/pics/notes/readme.txt"]

        # Query-first paged browse: the cursor binds the query.
        resp = await client.get("/api/mounts/pics/entries", params={"limit": "2"})
        body = await resp.json()
        assert [e["name"] for e in body["entries"]] == ["cat.png", "dog.png"]
        cursor = body["cursor"]
        resp = await client.get("/api/mounts/pics/entries", params={"limit": "2", "cursor": cursor})
        body = await resp.json()
        assert [e["name"] for e in body["entries"]] == ["readme.txt"]
        assert "cursor" not in body

        resp = await client.get("/api/mounts/pics/entries", params={"q": "dog"})
        body = await resp.json()
        assert [e["name"] for e in body["entries"]] == ["dog.png"]
        # A cursor minted for one query refuses to page another.
        resp = await client.get("/api/mounts/pics/entries", params={"q": "dog", "cursor": cursor})
        assert resp.status == 400

        resp = await client.get("/api/mounts/nope/entries")
        assert resp.status == 404
        # Mutation without the opt-in refuses with a machine-readable key
        # a client can distinguish from "no such endpoint".
        resp = await client.post("/api/mounts", json={"id": "x", "path": str(tmp_path)})
        assert resp.status == 403
        assert (await resp.json())["error"] == "mount-changes-disabled"
        resp = await client.delete("/api/mounts/pics")
        assert resp.status == 403
        assert (await resp.json())["error"] == "mount-changes-disabled"
        await client.close()

    asyncio.run(scenario())


def test_mounts_api_exposes_scan_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "models"
    seed(root, {"model.bin": b"model bytes"})
    table = MountTable(tmp_path / "library" / "mounts.json")
    table.add(MountDef(id="models", path=root))
    import dinkster_assets.library as library_module

    hashing = Event()
    release = Event()

    def held_digest(path: Path):
        hashing.set()
        assert release.wait(5)
        return real_digest_file_with_record(path)

    monkeypatch.setattr(library_module, "digest_file_with_record", held_digest)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        try:
            scan = asyncio.create_task(asyncio.to_thread(table.scan, "models"))
            assert await asyncio.to_thread(hashing.wait, 5)
            response = await client.get("/api/mounts")
            assert response.status == 200
            (row,) = (await response.json())["mounts"]
            assert row["state"] == "scanning"
            assert row["scanProgress"]["filesDone"] == 0
            assert row["scanProgress"]["filesTotal"] == 1
            assert row["scanProgress"]["bytesDone"] == 0
            assert row["scanProgress"]["bytesTotal"] == len(b"model bytes")
            assert row["scanProgress"]["elapsedSeconds"] >= 0
            release.set()
            assert await scan == 1
        finally:
            release.set()
            await client.close()

    asyncio.run(scenario())


def test_mount_service_does_not_republish_unchanged_elapsed_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    seed(root, {"model.bin": b"model bytes"})
    table = MountTable(tmp_path / "library" / "mounts.json")
    table.add(MountDef(id="models", path=root))
    import dinkster_assets.library as library_module

    import dinkster.mounts_api as mounts_api_module

    hashing = Event()
    release = Event()

    def held_digest(path: Path):
        hashing.set()
        assert release.wait(5)
        return real_digest_file_with_record(path)

    monkeypatch.setattr(library_module, "digest_file_with_record", held_digest)
    monkeypatch.setattr(mounts_api_module, "_SCAN_PROGRESS_INTERVAL", 0.01)
    service = MountService(table)
    refreshes: list[None] = []
    publishes: list[None] = []
    monkeypatch.setattr(service, "refresh_resolution_store", lambda: refreshes.append(None))
    monkeypatch.setattr(
        service,
        "publish",
        lambda _app: publishes.append(None),
    )

    async def scenario() -> None:
        polled = asyncio.Event()
        polls = 0
        original_progress = table.scan_progress

        def scan_progress(mount_id: str):
            nonlocal polls
            polls += 1
            if polls == 2:
                polled.set()
            return original_progress(mount_id)

        monkeypatch.setattr(table, "scan_progress", scan_progress)
        scan = asyncio.create_task(service.scan(cast("web.Application", {}), "models"))
        assert await asyncio.to_thread(hashing.wait, 5)
        await asyncio.wait_for(polled.wait(), 5)
        assert refreshes == []
        assert publishes == []
        release.set()
        await scan
        assert len(refreshes) == 2
        assert len(publishes) == 2

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_mount_service_cancelled_scan_consumes_background_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    seed(root, {"model.bin": b"model bytes"})
    table = MountTable(tmp_path / "library" / "mounts.json")
    table.add(MountDef(id="models", path=root))
    import dinkster_assets.library as library_module

    import dinkster.mounts_api as mounts_api_module

    hashing = Event()
    release = Event()
    completed = Event()

    def held_digest(_path: Path):
        hashing.set()
        assert release.wait(5)
        completed.set()
        raise AssetError("late scan failure")

    monkeypatch.setattr(library_module, "digest_file_with_record", held_digest)
    monkeypatch.setattr(mounts_api_module, "_SCAN_PROGRESS_INTERVAL", 0.01)
    service = MountService(table)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop_errors: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            service.scan_soon(cast("web.Application", {}), "models")
            (scan,) = service._tasks
            assert await asyncio.to_thread(hashing.wait, 5)
            scan.cancel()
            with pytest.raises(asyncio.CancelledError):
                await scan
            await asyncio.sleep(0)
            assert service._tasks == set()

            release.set()
            assert await asyncio.to_thread(completed.wait, 5)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gc.collect()
            await asyncio.sleep(0)
            assert loop_errors == []
        finally:
            release.set()
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_mount_entries_emit_and_filter_semantic_model_kind(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    seed(
        root,
        {
            "a.safetensors": b"first",
            "b.safetensors": b"second",
        },
    )
    table = MountTable(tmp_path / "snap.json")
    mount_id = "comfy-model-checkpoints-1"
    table.add(
        MountDef(id=mount_id, path=root),
        source="derived",
        kind="model/checkpoint",
    )
    table.scan(mount_id)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        response = await client.get(
            f"/api/mounts/{mount_id}/entries",
            params={"kind": "model/checkpoint", "limit": "1"},
        )
        assert response.status == 200
        body = await response.json()
        assert body["entries"][0]["kind"] == "model/checkpoint"
        assert "cursor" in body

        mismatch = await client.get(
            f"/api/mounts/{mount_id}/entries",
            params={"kind": "model/lora", "cursor": body["cursor"]},
        )
        assert mismatch.status == 400
        assert await mismatch.json() == {"error": "cursor does not match this query"}

        empty = await client.get(
            f"/api/mounts/{mount_id}/entries",
            params={"kind": "model/lora"},
        )
        assert empty.status == 200
        assert (await empty.json())["entries"] == []

        invalid = await client.get(
            f"/api/mounts/{mount_id}/entries",
            params={"kind": "checkpoint"},
        )
        assert invalid.status == 400
        await client.close()

    asyncio.run(scenario())


def _folder_paging_table(tmp_path: Path) -> MountTable:
    root = tmp_path / "granted"
    seed(
        root,
        {
            "alpha.txt": b"alpha",
            "root-dog.txt": b"root dog",
            "models/a-cat.safetensors": b"cat",
            "models/b-dog.safetensors": b"dog",
            "models/deep/c-dog.safetensors": b"deep dog",
            "models/deep/d-cat.safetensors": b"deep cat",
            "models/other/e-dog.safetensors": b"other dog",
            "notes/readme.txt": b"notes",
        },
    )
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="mixed", path=root))
    table.scan("mixed")
    return table


def test_mount_entries_omitted_params_match_explicit_defaults(tmp_path: Path) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        for query in (None, "dog"):
            omitted = {"limit": "2"}
            explicit = {"limit": "2", "path": "", "recursive": "true"}
            if query is not None:
                omitted["q"] = query
                explicit["q"] = query
            while True:
                omitted_resp = await client.get("/api/mounts/mixed/entries", params=omitted)
                explicit_resp = await client.get("/api/mounts/mixed/entries", params=explicit)
                omitted_payload = await omitted_resp.read()
                explicit_payload = await explicit_resp.read()
                assert omitted_resp.status == explicit_resp.status == 200
                assert omitted_payload == explicit_payload
                body = json.loads(omitted_payload)
                if "cursor" not in body:
                    break
                omitted["cursor"] = body["cursor"]
                explicit["cursor"] = json.loads(explicit_payload)["cursor"]
        await client.close()

    asyncio.run(scenario())


def test_mount_entries_recursive_subtree_pages_stay_scoped(tmp_path: Path) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        params = {"path": "models", "recursive": "true", "limit": "2"}
        resp = await client.get("/api/mounts/mixed/entries", params=params)
        first = await resp.json()
        assert [row["virtualPath"] for row in first["entries"]] == [
            "mounts/mixed/models/a-cat.safetensors",
            "mounts/mixed/models/b-dog.safetensors",
        ]
        resp = await client.get(
            "/api/mounts/mixed/entries",
            params={**params, "cursor": first["cursor"]},
        )
        second = await resp.json()
        assert [row["virtualPath"] for row in second["entries"]] == [
            "mounts/mixed/models/deep/c-dog.safetensors",
            "mounts/mixed/models/deep/d-cat.safetensors",
        ]
        resp = await client.get(
            "/api/mounts/mixed/entries",
            params={**params, "cursor": second["cursor"]},
        )
        third = await resp.json()
        assert [row["virtualPath"] for row in third["entries"]] == [
            "mounts/mixed/models/other/e-dog.safetensors"
        ]
        assert "cursor" not in third
        await client.close()

    asyncio.run(scenario())


def test_mount_entries_immediate_children_include_first_page_folders(
    tmp_path: Path,
) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        list_resp = await client.get("/api/mounts/mixed/list", params={"path": "models"})
        listed = await list_resp.json()
        params = {"path": "models", "recursive": "false", "limit": "1"}
        resp = await client.get("/api/mounts/mixed/entries", params=params)
        first = await resp.json()
        assert first["folders"] == listed["folders"] == ["deep", "other"]
        assert [row["virtualPath"] for row in first["entries"]] == [
            "mounts/mixed/models/a-cat.safetensors"
        ]
        resp = await client.get(
            "/api/mounts/mixed/entries",
            params={**params, "cursor": first["cursor"]},
        )
        second = await resp.json()
        assert "folders" not in second
        assert [row["virtualPath"] for row in second["entries"]] == [
            "mounts/mixed/models/b-dog.safetensors"
        ]

        resp = await client.get("/api/mounts/mixed/entries", params={"recursive": "false"})
        root = await resp.json()
        assert root["folders"] == ["models", "notes"]
        assert [row["name"] for row in root["entries"]] == [
            "alpha.txt",
            "root-dog.txt",
        ]
        await client.close()

    asyncio.run(scenario())


def test_mount_entries_query_composes_with_folder_scope(tmp_path: Path) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        resp = await client.get(
            "/api/mounts/mixed/entries",
            params={"path": "models/deep", "q": "dog"},
        )
        body = await resp.json()
        assert [row["virtualPath"] for row in body["entries"]] == [
            "mounts/mixed/models/deep/c-dog.safetensors"
        ]
        await client.close()

    asyncio.run(scenario())


def test_mount_entries_cursor_binds_every_filter(tmp_path: Path) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        base = {
            "path": "models",
            "recursive": "true",
            "q": "safe",
            "limit": "1",
        }
        resp = await client.get("/api/mounts/mixed/entries", params=base)
        cursor = (await resp.json())["cursor"]
        mismatches = (
            {**base, "path": "models/deep", "cursor": cursor},
            {**base, "recursive": "false", "cursor": cursor},
            {**base, "q": "dog", "cursor": cursor},
        )
        for params in mismatches:
            resp = await client.get("/api/mounts/mixed/entries", params=params)
            assert resp.status == 400
            assert await resp.json() == {"error": "cursor does not match this query"}
        resp = await client.get(
            "/api/mounts/mixed/entries", params={**base, "cursor": "not-a-cursor"}
        )
        assert resp.status == 400
        assert await resp.json() == {"error": "malformed cursor"}
        await client.close()

    asyncio.run(scenario())


def test_mount_entries_path_errors_match_list_and_unknown_mount_is_404(
    tmp_path: Path,
) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        for path in ("../models", "/models"):
            list_resp = await client.get("/api/mounts/mixed/list", params={"path": path})
            entries_resp = await client.get("/api/mounts/mixed/entries", params={"path": path})
            assert list_resp.status == entries_resp.status == 400
            assert await list_resp.json() == await entries_resp.json()
        resp = await client.get("/api/mounts/mixed/entries", params={"recursive": "1"})
        assert resp.status == 400
        assert await resp.json() == {"error": "recursive must be true or false"}
        resp = await client.get(
            "/api/mounts/nope/entries",
            params={"path": "../models", "recursive": "1"},
        )
        assert resp.status == 404
        await client.close()

    asyncio.run(scenario())


def test_mount_entries_scoped_limit_bounds_are_unchanged(tmp_path: Path) -> None:
    table = _folder_paging_table(tmp_path)

    async def scenario() -> None:
        client = await make_client(MountService(table))
        for limit, error in (
            ("0", "limit must be in 1..500"),
            ("501", "limit must be in 1..500"),
            ("many", "limit must be an integer"),
        ):
            resp = await client.get(
                "/api/mounts/mixed/entries",
                params={"path": "models", "limit": limit},
            )
            assert resp.status == 400
            assert await resp.json() == {"error": error}
        await client.close()

    asyncio.run(scenario())


def test_mount_grant_and_revoke_at_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "mounts.toml"
    snapshot = tmp_path / "snap.json"
    granted = tmp_path / "granted"
    seed(granted, {"nested/render.png": b"pixels"})
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    table = MountTable(snapshot)
    table.add(MountDef(id="comfy-input", path=tmp_path), source="derived")
    service = MountService(table, config, allow_changes=True)

    async def scenario() -> None:
        client = await make_client(service)
        ws = await client.ws_connect("/api/events?clientId=c1")

        listing = await (await client.get("/api/mounts")).json()
        assert listing["mountChangesAllowed"] is True

        resp = await client.post(
            "/api/mounts",
            json={
                "id": "renders",
                "path": str(granted),
                "mode": "readwrite",
                "priority": -4,
                "kind": "directory/input",
            },
        )
        assert resp.status == 201
        created = await resp.json()
        assert (created["id"], created["source"]) == ("renders", "config")
        assert created["priority"] == -4
        assert created["kind"] == "directory/input"

        # The grant is durable immediately (mounts.toml), scanned behind
        # the response, and announced on the event stream.
        assert load_mounts(config) == (
            MountDef(
                id="renders", path=granted, mode="readwrite", priority=-4, kind="directory/input"
            ),
        )
        await wait_for_state(client, "renders", "ready")
        async with asyncio.timeout(5):
            while True:
                event = cast("dict[str, object]", await ws.receive_json())
                if event.get("type") == "mounts_changed":
                    break
        # The worker-facing snapshot now names the granted mount.
        snap = json.loads(snapshot.read_text("utf-8"))
        assert [m["id"] for m in snap["mounts"]] == ["renders"]
        assert [m["priority"] for m in snap["mounts"]] == [-4]
        options = await (await client.get("/api/assets/selections?kind=directory/input")).json()
        assert options["options"] == [{"name": "nested", "shape": "directory"}]

        # Refusals: duplicate id, malformed id/mode, missing directory.
        resp = await client.post("/api/mounts", json={"id": "renders", "path": str(granted)})
        assert resp.status == 409
        resp = await client.post("/api/mounts", json={"id": "Bad Id", "path": str(granted)})
        assert resp.status == 400
        resp = await client.post(
            "/api/mounts", json={"id": "x", "path": str(granted), "mode": "rw"}
        )
        assert resp.status == 400
        resp = await client.post("/api/mounts", json={"id": "x", "path": str(tmp_path / "absent")})
        assert resp.status == 400
        for mount_id, priority in (("bool-priority", True), ("str-priority", "1")):
            resp = await client.post(
                "/api/mounts",
                json={"id": mount_id, "path": str(granted), "priority": priority},
            )
            assert resp.status == 400
            assert "priority must be an integer" in (await resp.json())["error"]
        for kind in (True, 42, "model/*", "../input"):
            resp = await client.post(
                "/api/mounts", json={"id": "invalid-kind", "path": str(granted), "kind": kind}
            )
            assert resp.status == 400

        # Derived mounts are not deletable here; config mounts are.
        resp = await client.delete("/api/mounts/comfy-input")
        assert resp.status == 409
        resp = await client.delete("/api/mounts/renders")
        assert resp.status == 200
        assert load_mounts(config) == ()
        options = await (await client.get("/api/assets/selections?kind=directory/input")).json()
        assert options["options"] == []
        resp = await client.delete("/api/mounts/renders")
        assert resp.status == 404
        await ws.close()
        await client.close()

    asyncio.run(scenario())


def test_output_mount_selection_is_persisted_and_published(tmp_path: Path) -> None:
    config = tmp_path / "mounts.toml"
    snapshot = tmp_path / "snapshot.json"
    output = tmp_path / "output"
    renders = tmp_path / "renders"
    output.mkdir()
    renders.mkdir()
    table = MountTable(snapshot, output_mount="output")
    table.add(MountDef("output", output, "readwrite"), source="config")
    table.add(MountDef("renders", renders, "readwrite"), source="config")
    table.scan("output")
    table.scan("renders")
    service = MountService(table, config, allow_changes=True)
    service.persist()

    async def scenario() -> None:
        client = await make_client(service)
        response = await client.put("/api/mounts/output", json={"id": "renders"})
        assert response.status == 200
        assert await response.json() == {
            "outputMount": "renders",
            "restartRequired": False,
            "pendingRemovals": [],
        }
        listing = await (await client.get("/api/mounts")).json()
        assert listing["outputMount"] == "renders"
        await client.close()

    asyncio.run(scenario())
    assert load_output_mount(config) == "renders"
    assert json.loads(snapshot.read_text("utf-8"))["outputMount"] == "renders"


def test_output_mount_selection_refuses_derived_mounts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    table = MountTable(output_mount="output")
    table.add(MountDef("output", output, "readwrite"), source="derived")

    with pytest.raises(MountsError, match="not persisted in mounts.toml"):
        table.select_output_mount("output")


def test_mount_service_keeps_configured_resolution_scopes_in_sync(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"old.bin": b"old"})
    table = MountTable()
    table.add(MountDef(id="models", path=root, priority=3), source="config")
    table.scan("models")
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    stale = MountMaterialization(
        "scope-a",
        "stale",
        0,
        "",
        AssetRef(
            digest_bytes(b"stale"),
            "stale.bin",
            5,
            virtual_path="mounts/stale/stale.bin",
        ),
    )
    store.replace_mount_snapshot(("scope-a",), (stale,))
    service = MountService(table, allow_changes=True)
    service.attach_resolution_store(store, ("scope-a", "scope-b"))
    assert [row.mount_id for row in store.snapshot("scope-a").mounts] == ["models"]
    assert store.snapshot("scope-b").mounts[0].ref.digest == digest_bytes(b"old")

    async def scenario() -> None:
        client = await make_client(service)
        try:
            (root / "old.bin").unlink()
            seed(root, {"new.bin": b"new"})
            await service.scan(cast(TestServer, client.server).app, "models")
            assert [row.ref.name for row in store.snapshot("scope-a").mounts] == ["new.bin"]

            for path in root.iterdir():
                path.unlink()
            root.rmdir()
            await service.scan(cast(TestServer, client.server).app, "models")
            assert store.snapshot("scope-a").mounts == ()
            assert store.snapshot("scope-b").mounts == ()

            response = await client.delete("/api/mounts/models")
            assert response.status == 200
            assert store.snapshot("scope-a").mounts == ()
        finally:
            await client.close()

    asyncio.run(scenario())
    store.close()


def test_mount_service_attachment_clears_stale_rows_before_pending_scan(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    digest = digest_bytes(b"stale")
    stale = MountMaterialization(
        "scope-a",
        "stale",
        0,
        "",
        AssetRef(digest, "stale", 5, virtual_path="mounts/stale/x"),
    )
    store.replace_mount_snapshot(("scope-a",), (stale,))
    table = MountTable()
    table.add(MountDef(id="pending", path=tmp_path / "pending"))
    MountService(table).attach_resolution_store(store, ("scope-a",))
    assert store.snapshot("scope-a").mounts == ()
    store.close()


def test_mount_revoke_persists_even_when_catalog_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "mounts.toml"
    table = MountTable()
    table.add(MountDef(id="models", path=tmp_path / "models"), source="config")
    service = MountService(table, config, allow_changes=True)
    service.persist()
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    service.attach_resolution_store(store, ("scope-a",))

    def fail_refresh(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(store, "replace_mount_snapshot", fail_refresh)

    async def scenario() -> None:
        client = await make_client(service)
        try:
            response = await client.delete("/api/mounts/models")
            assert response.status == 500
            assert load_mounts(config) == ()
            assert table.get("models") is None
        finally:
            await client.close()

    asyncio.run(scenario())
    store.close()


def test_project_folders_persist_order_membership_and_derived_exclusions(tmp_path: Path) -> None:
    config = tmp_path / "mounts.toml"
    table = MountTable()
    table.add(MountDef("derived", tmp_path / "derived"), source="derived")
    service = MountService(table, config, allow_changes=True)
    for name in ("first", "second", "common"):
        seed(tmp_path / name, {"same.bin": name.encode()})

    async def scenario() -> None:
        client = await make_client(service)
        try:
            for name in ("first", "second", "common"):
                response = await client.post(
                    "/api/mounts",
                    json={
                        "id": name,
                        "path": str(tmp_path / name),
                        "project": "project-a",
                        "shared": name == "common",
                        "mode": "readwrite",
                        "kind": "model/vae",
                    },
                )
                assert response.status == 201
                await wait_for_state(client, name, "ready", project="project-a")
            response = await client.get("/api/mounts?project=project-b")
            assert [row["id"] for row in (await response.json())["mounts"]] == ["derived", "common"]
            for project, order in (
                ("project-a", ["second", "first", "common"]),
                ("project-b", ["common", "first"]),
            ):
                response = await client.put(
                    f"/api/mounts/projects/{project}",
                    json={
                        "mounts": order,
                        "outputMount": "common",
                    },
                )
                assert response.status == 200
                assert [row["id"] for row in (await response.json())["mounts"]] == order
            assert (await client.delete("/api/mounts/common")).status == 409
            assert table.get("derived") is not None
            assert all(mount.id != "derived" for mount in load_mounts(config))
            saved = config.with_name("project-folders.json").read_bytes()
            for mounts in ([], ["first"], ["common", "first"]):
                response = await client.put(
                    "/api/mounts/projects/project-b",
                    json={"mounts": mounts, "outputMount": None},
                )
                assert response.status == 409
                assert config.with_name("project-folders.json").read_bytes() == saved
                current = await (await client.get("/api/mounts?project=project-b")).json()
                assert current["outputMount"] == "common"
                assert current["activeOutputMount"] == "common"
                assert [row["id"] for row in current["mounts"]] == ["common", "first"]
            response = await client.put(
                "/api/mounts/projects/project-b",
                json={"mounts": ["first"], "outputMount": "first"},
            )
            assert response.status == 200
            assert (await response.json())["outputMount"] == "first"
            response = await client.put(
                "/api/mounts/projects/without-output",
                json={"mounts": [], "outputMount": None},
            )
            assert response.status == 200
            assert (await response.json())["outputMount"] is None
        finally:
            await service.close()
            await client.close()

    asyncio.run(scenario())
    restarted_table = MountTable()
    for mount in load_mounts(config):
        restarted_table.add(mount)
    restarted_table.add(MountDef("derived", tmp_path / "derived"), source="derived")
    restarted = MountService(restarted_table, config, allow_changes=True)
    project_a = cast(list[dict[str, object]], restarted.project_response("project-a")["mounts"])
    assert [row["id"] for row in project_a] == [
        "second",
        "first",
        "common",
    ]
    project_b = restarted.project_response("project-b")
    assert [row["id"] for row in cast(list[dict[str, object]], project_b["mounts"])] == ["first"]
    assert project_b["outputMount"] == "first"
    assert restarted.project_response("project-a")["outputMount"] == "common"
    assert restarted.project_response("without-output")["mounts"] == []
    assert restarted.project_response("without-output")["outputMount"] is None
    new_project = cast(list[dict[str, object]], restarted.project_response("new-project")["mounts"])
    assert [row["id"] for row in new_project] == [
        "derived",
        "common",
    ]
    assert [mount.priority for mount in load_mounts(config)] == [0, 0, 0]
    for name in ("first", "second", "common"):
        assert (tmp_path / name / "same.bin").read_bytes() == name.encode()


def test_project_folder_validation_keeps_previous_settings(tmp_path: Path) -> None:
    table = MountTable()
    table.add(MountDef("existing", tmp_path, mode="readwrite"))
    service = MountService(table, tmp_path / "mounts.toml", allow_changes=True)

    async def scenario() -> None:
        client = await make_client(service)
        try:
            for mounts, output in (
                (["unknown"], None),
                (["existing", "existing"], None),
                (["existing"], "unknown"),
                ([], "existing"),
                (False, None),
            ):
                response = await client.put(
                    "/api/mounts/projects/default",
                    json={
                        "mounts": mounts,
                        "outputMount": output,
                    },
                )
                assert response.status == 400
            response = await client.get("/api/mounts?project=Bad%20Project")
            assert response.status == 400
            response = await client.get("/api/mounts?project=default")
            assert [row["id"] for row in (await response.json())["mounts"]] == ["existing"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_project_name_order_never_substitutes_for_digest_identity(tmp_path: Path) -> None:
    from dinkster_assets import resolve_asset_selection
    from dinkster_values.asset_folders import use_asset_folders

    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    for name, data in (("first", b"first model"), ("second", b"different model")):
        seed(tmp_path / name, {"same.safetensors": data})
        table.add(
            MountDef(
                name, tmp_path / name, kind="model/vae", priority=-8 if name == "second" else 2
            )
        )
        table.scan(name)
    service = MountService(table, tmp_path / "mounts.toml", allow_changes=True)
    assert service.project_context("new-project").mounts == ("second", "first")
    first = table.ref("mounts/first/same.safetensors")
    resolver = MountSnapshotResolver(snapshot)
    for order, expected in (
        (("second", "first"), b"different model"),
        (("first", "second"), b"first model"),
    ):
        with use_asset_folders(AssetFolderContext(order)):
            ref = table.bind_name("same.safetensors", "model/vae")
            assert isinstance(ref, AssetRef)
            assert ref.digest == digest_bytes(expected)
            assert table.resolve(first.digest) == tmp_path / "first" / "same.safetensors"
            assert resolver.resolve(first.digest) == tmp_path / "first" / "same.safetensors"
    with use_asset_folders(AssetFolderContext(())):
        assert table.bind_name("same.safetensors", "model/vae") is None
        assert resolver.resolve(first.digest) == tmp_path / "first" / "same.safetensors"
        assert (
            resolve_asset_selection(
                "model/vae", first, env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
            )
            is first
        )
    assert not (tmp_path / "vault").exists()
    assert not (tmp_path / "produced-assets").exists()


def test_project_folder_paths_and_mutations_require_operator(tmp_path: Path) -> None:
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    table = MountTable()
    table.add(MountDef("private-location", tmp_path / "private-location"))
    service = MountService(table, allow_changes=True)
    authenticator = StaticBearerAuthenticator(
        {
            "asset-editor-token": Principal(
                "editor", {"local": frozenset({"assets:read", "assets:write"})}
            ),
            "agent-token": Principal(
                "agent",
                {
                    "local": frozenset(
                        {
                            "assets:read",
                            "assets:write",
                            "settings:write",
                        }
                    )
                },
                kind="agent",
            ),
        }
    )

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        add_mount_routes(app, service)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for token in ("asset-editor-token", "agent-token"):
                headers = {"Authorization": f"Bearer {token}"}
                for path in ("/api/mounts", "/api/mounts?project=default"):
                    response = await client.get(path, headers=headers)
                    assert response.status == 200
                    assert str(tmp_path) not in await response.text()
                for method, path, body in (
                    ("GET", "/api/mounts/locations", None),
                    ("POST", "/api/mounts", {"id": "new", "path": str(tmp_path)}),
                    ("PUT", "/api/mounts/projects/default", {"mounts": [], "outputMount": None}),
                    ("DELETE", "/api/mounts/private-location", None),
                    ("POST", "/api/mounts/private-location/retry", None),
                ):
                    response = await client.request(method, path, json=body, headers=headers)
                    assert response.status == 403
                    assert str(tmp_path) not in await response.text()
            assert table.get("private-location") is not None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_sandbox_folder_edits_preserve_live_grants_until_restart(tmp_path: Path) -> None:
    config = tmp_path / "mounts.toml"
    snapshot = tmp_path / "snapshot.json"
    for name in ("old", "next"):
        seed(tmp_path / name, {"file.bin": name.encode()})
    table = MountTable(snapshot, output_mount="old")
    table.add(MountDef("old", tmp_path / "old", "readwrite"))
    table.scan("old")
    old_digest = table.ref("mounts/old/file.bin").digest
    original_snapshot = snapshot.read_bytes()
    service = MountService(table, config, allow_changes=True, restart_for_changes=True)
    service.persist()

    async def scenario() -> None:
        client = await make_client(service)
        try:
            response = await client.post(
                "/api/mounts",
                json={"id": "next", "path": str(tmp_path / "next"), "mode": "readwrite"},
            )
            assert response.status == 201
            added = await response.json()
            assert added["state"] == "restart-required"
            assert added["restartRequired"] is True
            assert (await client.get("/api/mounts/next/list")).status == 404
            response = await client.put(
                "/api/mounts/projects/default",
                json={"mounts": ["next"], "outputMount": None},
            )
            assert response.status == 409
            assert snapshot.read_bytes() == original_snapshot
            assert (await client.delete("/api/mounts/old")).status == 409
            assert (await client.put("/api/mounts/output", json={"id": "next"})).status == 200
            response = await client.delete("/api/mounts/old")
            assert response.status == 200
            removed = await response.json()
            assert removed["restartRequired"] is True
            assert removed["pendingRemovals"] == ["old"]
            listing = await (await client.get("/api/mounts")).json()
            assert listing["outputMount"] == "next"
            assert listing["activeOutputMount"] == "old"
            locations = await (await client.get("/api/mounts/locations")).json()
            assert [row["id"] for row in locations["mounts"]] == ["next"]
            assert [(row["id"], row["path"]) for row in locations["activeMounts"]] == [
                ("old", str(tmp_path / "old"))
            ]
            response = await client.post(
                "/api/mounts", json={"id": "old", "path": str(tmp_path / "next")}
            )
            assert response.status == 409
            assert table.output_mount == "old"
            assert (
                MountSnapshotResolver(snapshot).resolve(old_digest) == tmp_path / "old" / "file.bin"
            )
            assert snapshot.read_bytes() == original_snapshot
        finally:
            await service.close()
            await client.close()

    asyncio.run(scenario())
    assert [mount.id for mount in load_mounts(config)] == ["next"]
    restarted = MountTable(output_mount=load_output_mount(config))
    for mount in load_mounts(config):
        restarted.add(mount)
        restarted.scan(mount.id)
    assert restarted.get("old") is None
    assert restarted.output_mount == "next"
    assert restarted.ref("mounts/next/file.bin").digest == digest_bytes(b"next")
    assert (tmp_path / "old" / "file.bin").read_bytes() == b"old"


def test_offline_folder_retry_retains_location_and_digest(tmp_path: Path) -> None:
    root = tmp_path / "mounted-network-share" / "Project A"
    detached = tmp_path / "detached-share"
    seed(root, {"source.bin": b"network source"})
    table = MountTable(tmp_path / "snapshot.json", index_root=tmp_path / "indexes")
    table.add(MountDef("network", root, kind="model/vae"))
    table.scan("network")
    digest = table.ref("mounts/network/source.bin").digest
    root.rename(detached)
    service = MountService(table, tmp_path / "mounts.toml", allow_changes=True)
    service.persist()

    async def scenario() -> None:
        client = await make_client(service)
        try:
            response = await client.post("/api/mounts/network/retry")
            assert response.status == 202
            await wait_for_state(client, "network", "failed")
            response = await client.get("/api/mounts/locations")
            (row,) = (await response.json())["mounts"]
            assert row["path"] == str(root)
            assert row["state"] == "failed"
            assert not root.exists()
            assert table.resolve(digest) is None
            detached.rename(root)
            assert (await client.post("/api/mounts/network/retry")).status == 202
            await wait_for_state(client, "network", "ready")
            response = await client.get("/api/mounts/network/entries")
            (entry,) = (await response.json())["entries"]
            assert entry["digest"] == digest
            assert entry["virtualPath"] == "mounts/network/source.bin"
            assert (root / "source.bin").read_bytes() == b"network source"
        finally:
            await service.close()
            await client.close()

    asyncio.run(scenario())
