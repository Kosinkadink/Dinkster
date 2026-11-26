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
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import mount_roots_for_kind
from dinkster_assets import (
    AssetError,
    AssetVault,
    AssetWriter,
    LibraryStore,
    LocalAssetLibrary,
    MountDef,
    MountSnapshotResolver,
    MountTable,
    digest_bytes,
    dump_mounts,
    load_mounts,
    load_output_mount,
    parse_mounts,
)
from dinkster_assets.integrity import digest_file_with_record as real_digest_file_with_record
from dinkster_assets.model import AssetRef, LocalAssetRef
from dinkster_assets.resolution import MountMaterialization, ResolutionStore
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_protocol import AssetFolderContext
from dinkster_schema import (
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import ServerLibrary, create_app
from dinkster_values import INLINE_STRING_CAP, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker, load_manifest
from dinkster_workers.host import load_pack

from dinkster.mounts_api import MountService, add_mount_routes
from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip

STRING = TypeExpr.concrete("STRING")


def test_engine_relative_selection_keeps_exact_root_and_project_authority(tmp_path: Path) -> None:
    from dinkster_values.asset_folders import use_asset_folders

    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "indexes")
    for name, data in (("first", b"first bytes"), ("second", b"different second bytes")):
        root = tmp_path / name
        root.mkdir()
        (root / "shared.bin").write_bytes(data)
        table.add(MountDef(name, root, kind="media/model3d"))
    root = tmp_path / "second"
    selected = table.bind_relative_file("media/model3d", root, "shared.bin")
    assert selected.read_bytes() == b"different second bytes"
    named = table.bind_name("shared.bin", "media/model3d")
    assert named is not None and named.read_bytes() == b"first bytes"
    for relative in ("../first/shared.bin", "/shared.bin", "./shared.bin", "C:/shared.bin"):
        with pytest.raises(AssetError, match="safe relative"):
            table.bind_relative_file("media/model3d", root, relative)
    with pytest.raises(AssetError, match="not granted"):
        table.bind_relative_file("media/image", root, "shared.bin")
    with use_asset_folders(AssetFolderContext(("first",))):
        with pytest.raises(AssetError, match="not granted"):
            table.bind_relative_file("media/model3d", root, "shared.bin")
    table.remove("second")
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        selected.read_bytes()


def test_relative_selection_reuses_engine_grant_without_indexing(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    table.add(MountDef("models", root, kind="media/model3d"))
    table.scan("models")
    (root / "sidecar.bin").write_bytes(b"first unindexed resource")
    granted = table.bind_name("sidecar.bin", "media/model3d")
    assert isinstance(granted, LocalAssetRef)
    worker = MountSnapshotResolver(snapshot)
    selected = worker.bind_relative_file("media/model3d", root, "sidecar.bin")
    assert isinstance(selected, LocalAssetRef)
    received = LocalAssetRef.from_wire(selected.to_wire(), table)
    assert received.read_bytes() == b"first unindexed resource"
    assert selected.binding == granted.binding
    (root / "sidecar.bin").write_bytes(b"a changed resource revision")
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        received.read_bytes()
    replacement = table.bind_name("sidecar.bin", "media/model3d")
    assert isinstance(replacement, LocalAssetRef)
    selected = worker.bind_relative_file("media/model3d", root, "sidecar.bin")
    assert isinstance(selected, LocalAssetRef)
    received = LocalAssetRef.from_wire(selected.to_wire(), table)
    assert received.read_bytes() == b"a changed resource revision"
    assert selected.binding != granted.binding
    table.remove("models")
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        received.read_bytes()
    with pytest.raises(AssetError, match="not granted"):
        worker.bind_relative_file("media/model3d", root, "sidecar.bin")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            body=(Input("text", STRING), Output("out", STRING)),
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


def test_engine_mount_scans_share_helper_hashes_across_aliases_and_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server.projects import add_project_routes

    models = tmp_path / "models"
    seed(
        models,
        {
            "model.bin": b"shared model payload",
            "earlier-copy.txt": b"selected without a folder scan",
        },
    )
    payload = models / "model.bin"
    opened = tmp_path / "payload-reads.txt"
    code = f"""
import io, os, runpy

class CountedReader:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.recorded = False

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.source.__exit__(*args)

    def read(self, *args):
        data = self.source.read(*args)
        if data and not self.recorded:
            with open({str(opened)!r}, "a") as output:
                output.write(os.path.basename(self.name) + "\\n")
            self.recorded = True
        return data

original_open = io.open

def counted_open(file, *args, **kwargs):
    source = original_open(file, *args, **kwargs)
    if isinstance(file, (str, os.PathLike)) and os.path.normcase(file) in (
        os.path.normcase({str(payload)!r}),
        os.path.normcase({str(models / "selected.txt")!r}),
    ) and "r" in source.mode:
        return CountedReader(source, file)
    return source

io.open = counted_open
runpy.run_module("dinkster_library", run_name="__main__")
"""
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "account"))
    monkeypatch.setenv("DINKSTER_LIBRARY_COMMAND", json.dumps([sys.executable, "-c", code]))

    async def scenario() -> None:
        for name in ("first-library", "second-library"):
            library = tmp_path / name
            snapshot = library / "worker-state" / "mounts.json"
            table = MountTable(snapshot, index_root=snapshot.parent / "asset-indexes")
            for identifier in ("models-upscale", "models-upscaler"):
                table.add(MountDef(identifier, models))
            service = MountService(table, library / "mounts.toml")
            app = create_app(make_engine, SCHEMAS)
            add_project_routes(app, library, engine_version="test")
            add_mount_routes(app, service)
            async with TestClient(TestServer(app)) as client:
                for identifier in ("models-upscale", "models-upscaler"):
                    await service.scan(app, identifier)
                    assert table.descriptors()[0]["state"] == "ready"
                    assert table.ref(f"mounts/{identifier}/model.bin").digest == digest_bytes(
                        b"shared model payload"
                    )
                if name == "first-library":
                    (models / "selected.txt").write_bytes(b"selected without a folder scan")
                for _ in range(2):
                    response = await client.post(
                        "/api/mounts/resolve-path",
                        json={
                            "path": str(models / "selected.txt"),
                            "project": "default",
                            "scope": "local",
                        },
                    )
                    assert response.status == 200
                    body = await response.json()
                    assert body["status"] == "resolved"
                    assert body["asset"]["digest"] == digest_bytes(
                        b"selected without a folder scan"
                    )
                    assert table.ref(body["asset"]["virtualPath"]).local_path() == (
                        models / "selected.txt"
                    )
                assert opened.read_text("utf-8").splitlines() == ["model.bin", "selected.txt"]
                await service.scan(app, "models-upscale")
                assert (
                    MountSnapshotResolver(snapshot).resolve(digest_bytes(b"shared model payload"))
                    == payload
                )
                rows = json.loads(snapshot.read_text("utf-8"))["mounts"]
                assert all(Path(row["index"]).is_relative_to(snapshot.parent) for row in rows)
            await service.close()
        assert opened.read_text("utf-8").splitlines() == ["model.bin", "selected.txt"]

    run_scenario(scenario())


def test_publication_before_scan_preserves_worker_index(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"weights.bin": b"original weights"})
    snapshot = tmp_path / "worker-state" / "mounts.json"
    index_root = tmp_path / "indexes"
    table = MountTable(snapshot, index_root=index_root)
    table.add(MountDef(id="models", path=root))
    assert table.scan("models") == 1
    digest = digest_bytes(b"original weights")
    index = index_root / "models.json"
    warm = json.loads(index.read_text("utf-8"))

    restarted = MountTable(snapshot, index_root=index_root)
    restarted.add(MountDef(id="models", path=root))
    restarted.refresh_publication("models")

    assert restarted.entries("models") == ()
    assert json.loads(index.read_text("utf-8")) == warm
    worker = MountSnapshotResolver(snapshot)
    assert worker.resolve(digest) == root / "weights.bin"

    # Another scanner can publish while this engine retains its earlier catalog.
    (root / "new.bin").write_bytes(b"new weights")
    assert table.scan("models") == 2
    updated = index.read_bytes()
    restarted.refresh_publication("models")
    assert index.read_bytes() == updated
    assert MountSnapshotResolver(snapshot).resolve(digest_bytes(b"new weights")) == root / "new.bin"
    assert restarted.scan("models") == 2
    assert restarted.ref("mounts/models/weights.bin").read_bytes() == b"original weights"


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


@pytest.mark.parametrize("isolated", [False, True])
def test_public_mount_roots_in_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated: bool
) -> None:
    late_text = "late embedding " * 24
    first_text = "preferred embedding " * 20
    mounts_root = tmp_path
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest_path = pack / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "root-probe"\npublisher = "dinkster"\n'
        'namespaces = ["rootprobe"]\n[pack.entry]\nnodes = "root_probe:NODES"\n'
    )
    (pack / "root_probe.py").write_text(
        "import json\n"
        "from dinkster_api.v1 import Node, NodeSchema, Output, TypeExpr, mount_roots_for_kind\n"
        "class Probe(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='rootprobe.read',  "
        "body=(Output('roots', TypeExpr.concrete('STRING')),))\n"
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
                ("late", 4, "model/embedding", late_text),
                ("unrelated", -9, "model/checkpoint", "not an embedding"),
                ("first", -1, "model/embedding", first_text),
            ):
                seed(mounts_root / name, {"sentinel.txt": text.encode()})
                table.add(MountDef(id=name, path=mounts_root / name, priority=priority), kind=kind)
                table.scan(name)
                if kind == "model/embedding":
                    expected.insert(0, [str(mounts_root / name), text])
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
                    [str(mounts_root / "late"), late_text],
                    [str(mounts_root / "first"), first_text],
                ],
                [
                    [str(mounts_root / "first"), first_text],
                    [str(mounts_root / "late"), late_text],
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
                    for project, roots in (("project-a", expected[::-1]), ("project-b", expected)):
                        assert len(json.dumps(roots)) > INLINE_STRING_CAP
                        while True:
                            status = await (
                                await client.get(f"/api/jobs/projects/{project}")
                            ).json()
                            assert status["state"] != "failed", status
                            if status["state"] == "completed":
                                descriptor = status["outputs"]["read"]["roots"]
                                assert descriptor["typeId"] == "STRING"
                                assert "value" not in descriptor
                                response = await client.get(
                                    "/api/values",
                                    params={
                                        "clientId": "projects",
                                        "jobId": project,
                                        "nodeId": "read",
                                        "outputId": "roots",
                                        "rendition": "text",
                                    },
                                )
                                assert response.status == 200, await response.text()
                                actual = json.loads(await response.text())
                                assert actual == roots
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
                [str(mounts_root / "first"), first_text]
            ]
        finally:
            await worker.close()

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


def test_automatic_mount_scans_preserve_ready_catalog_until_explicit_retry(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"first.bin": b"first"})
    table = MountTable(tmp_path / "mounts.json")
    table.add(MountDef(id="models", path=root), source="shared-models")
    service = MountService(table, allow_changes=True)

    async def scenario() -> None:
        client = await make_client(service)
        try:
            async with service._scan_lock:
                startup = asyncio.create_task(service.scan_all(client.app))
                await asyncio.sleep(0)
                assert (await client.get("/api/mounts")).status == 200
                table.scan("models")
                seed(root, {"later.bin": b"later"})
            await startup
            await asyncio.gather(*tuple(service._tasks))
            response = await (await client.get("/api/mounts")).json()
            (row,) = response["mounts"]
            assert row["state"] == "ready"
            assert row["entryCount"] == 1

            assert (await client.post("/api/mounts/models/retry")).status == 202
            await asyncio.gather(*tuple(service._tasks))
            response = await (await client.get("/api/mounts")).json()
            (row,) = response["mounts"]
            assert row["state"] == "ready"
            assert row["entryCount"] == 2
        finally:
            await service.close()
            await client.close()

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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


@pytest.mark.parametrize(
    ("sort", "expected"),
    [
        ("newest", ["second/Zebra.png", "first/Zebra.png", "first/deep/apple.png"]),
        ("name", ["first/deep/apple.png", "first/Zebra.png", "second/Zebra.png"]),
        ("path", ["first/Zebra.png", "first/deep/apple.png", "second/Zebra.png"]),
    ],
)
def test_mount_entries_order_across_enabled_mounts(
    tmp_path: Path, sort: str, expected: list[str]
) -> None:
    import os

    table = MountTable(tmp_path / "snap.json", index_root=tmp_path / "indexes")
    files = {
        "first/Zebra.png": 1_700_000_100.25,
        "first/deep/apple.png": 1_700_000_100.25,
        "second/Zebra.png": 1_700_000_200.5,
        "second/notes.txt": 1_700_000_300.0,
        "disabled/hidden.png": 1_700_000_400.0,
    }
    for relative, modified in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        os.utime(path, (modified, modified))
    for mount_id in ("first", "second", "disabled"):
        table.add(MountDef(mount_id, tmp_path / mount_id))
        table.scan(mount_id)
    service = MountService(table)
    service.projects.update(
        "default", ["first", "second"], available=service.mount_ids(), output=None
    )

    async def scenario() -> None:
        async with await make_client(service) as client:
            params = {"sort": sort, "kind": "media/image", "limit": "1"}
            rows = []
            cursor = None
            for _ in expected:
                response = await client.get(
                    "/api/mounts/entries",
                    params={**params, **({"cursor": cursor} if cursor else {})},
                )
                assert response.status == 200
                body = await response.json()
                assert body["total"] == 3
                rows.extend(body["entries"])
                cursor = body.get("cursor")
            assert cursor is None
            assert [row["virtualPath"] for row in rows] == [f"mounts/{p}" for p in expected]
            assert [row["modifiedAt"] for row in rows] == [files[p] for p in expected]

            response = await client.get("/api/mounts/entries", params={**params, "mount": "first"})
            body = await response.json()
            assert body["total"] == 2
            selected_rows = body["entries"]
            selected_cursor = body["cursor"]
            response = await client.get(
                "/api/mounts/entries",
                params={**params, "mount": "first", "cursor": selected_cursor},
            )
            body = await response.json()
            assert body["total"] == 2
            assert "cursor" not in body
            selected_rows.extend(body["entries"])
            assert [row["virtualPath"] for row in selected_rows] == [
                f"mounts/{p}" for p in expected if p.startswith("first/")
            ]
            for scope in ({}, {"mount": "second"}):
                response = await client.get(
                    "/api/mounts/entries",
                    params={**params, **scope, "cursor": selected_cursor},
                )
                assert response.status == 400
                assert await response.json() == {"error": "cursor does not match this query"}
            response = await client.get(
                "/api/mounts/entries",
                params={**params, "mount": ["second", "first", "first"], "limit": "10"},
            )
            body = await response.json()
            assert body["total"] == 3
            assert [row["virtualPath"] for row in body["entries"]] == [
                f"mounts/{p}" for p in expected
            ]
            for mount, status in (("disabled", 403), ("unknown", 404), ("", 400)):
                response = await client.get(
                    "/api/mounts/entries", params={"mount": ["first", mount]}
                )
                assert response.status == status
            response = await client.get("/api/mounts/first/entries", params={"mount": "second"})
            assert response.status == 400

            response = await client.get(
                "/api/mounts/first/entries",
                params={"sort": sort, "path": "deep", "q": "APPLE", "recursive": "false"},
            )
            body = await response.json()
            assert body["total"] == 1
            assert [row["name"] for row in body["entries"]] == ["apple.png"]
            response = await client.get(
                "/api/mounts/entries", params={**params, "recursive": "false"}
            )
            assert (await response.json())["total"] == 2
            response = await client.get("/api/mounts/disabled/entries", params=params)
            assert response.status == 403
            response = await client.get("/api/mounts/entries", params={"path": "deep"})
            assert response.status == 400
            response = await client.get("/api/mounts/entries", params={"sort": "invalid"})
            assert response.status == 400
            response = await client.get("/api/mounts/entries", params=params)
            cursor = (await response.json())["cursor"]
            service.projects.update(
                "default", ["first"], available=service.mount_ids(), output=None
            )
            response = await client.get("/api/mounts/entries", params={**params, "cursor": cursor})
            assert response.status == 400
            assert await response.json() == {"error": "cursor does not match this query"}
            response = await client.get(
                "/api/mounts/entries",
                params={**params, "mount": "first", "cursor": selected_cursor},
            )
            assert response.status == 400
            assert await response.json() == {"error": "cursor does not match this query"}

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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
            {**base, "sort": "newest", "cursor": cursor},
            {**base, "sort": "name", "cursor": cursor},
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

    run_scenario(scenario())


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

    run_scenario(scenario())


@pytest.mark.parametrize("restart", [False, True])
def test_helper_mount_routes_publish_grants_and_membership_atomically(
    tmp_path: Path, restart: bool
) -> None:
    from dinkster_library import LibraryClient

    class RacingClient(LibraryClient):
        race = False

        def call(self, operation: str, **arguments: object) -> dict[str, object]:
            if operation == "folders.patch" and self.race:
                self.race = False
                state = cast("dict[str, object]", peer.call("folders.projects.get")["folders"])
                peer.call(
                    "folders.projects.patch", revision=state["revision"], patch={"future": "peer"}
                )
            return super().call(operation, **arguments)

    root = tmp_path / "library"
    granted = tmp_path / "renders"
    seed(granted, {"image.png": b"pixels"})
    snapshot = tmp_path / "worker.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    table.add(MountDef("input", tmp_path / "input"), source="derived")
    with RacingClient(root) as helper, LibraryClient(root) as peer:
        service = MountService(
            table, library=helper, allow_changes=True, restart_for_changes=restart
        )
        baseline = snapshot.read_bytes()

        async def scenario() -> None:
            client = await make_client(service)
            try:
                response = await client.post(
                    "/api/mounts", json={"id": "renders", "path": str(granted), "mode": "readwrite"}
                )
                assert response.status == 201
                state = peer.call("folders.get")
                grants = cast("dict[str, object]", state["grants"])
                projects = cast("dict[str, object]", state["projects"])
                values = cast("dict[str, object]", grants["values"])
                assert cast("dict[str, object]", values["mounts"])["renders"] == {
                    "path": str(granted),
                    "mode": "readwrite",
                    "priority": 0,
                    "kind": "",
                }
                assert cast("dict[str, object]", projects["values"])["projects"] == {
                    "default": ["renders"]
                }
                assert service.projects.ordered("other", service.mount_ids(settings=True)) == (
                    "input",
                )
                if restart:
                    assert table.get("renders") is None
                    assert snapshot.read_bytes() == baseline
                else:
                    await wait_for_state(client, "renders", "ready")
                    assert table.get("renders") is not None
                response = await client.put("/api/mounts/output", json={"id": "renders"})
                assert response.status == 200
                assert service.settings_table.output_mount == "renders"
                current = await (await client.get("/api/mounts")).json()
                assert current["outputMount"] == "renders"
                assert current["restartRequired"] is restart
                response = await client.put(
                    "/api/mounts/projects/default",
                    json={"mounts": ["input", "renders"], "outputMount": "renders"},
                )
                assert response.status == 200
                protected = peer.call("folders.get")
                response = await client.delete("/api/mounts/renders")
                assert response.status == 409
                assert peer.call("folders.get") == protected
                snapshot_before = snapshot.read_bytes()
                grants_before = peer.call("folders.grants.get")
                helper.race = True
                response = await client.post(
                    "/api/mounts", json={"id": "stale", "path": str(granted), "shared": True}
                )
                # Use a distinct path so the refusal exercises the late project CAS.
                assert response.status == 400
                other = tmp_path / "other"
                other.mkdir()
                helper.race = True
                response = await client.post(
                    "/api/mounts", json={"id": "stale", "path": str(other), "shared": True}
                )
                assert response.status == 409
                assert peer.call("folders.grants.get") == grants_before
                assert table.get("stale") is None
                assert service.settings_table.get("stale") is None
                assert snapshot.read_bytes() == snapshot_before
                current = await (await client.get("/api/mounts")).json()
                assert [row["id"] for row in current["mounts"]] == ["input", "renders"]
                projects = cast("dict[str, object]", peer.call("folders.projects.get")["folders"])
                assert cast("dict[str, object]", projects["values"])["future"] == "peer"
                response = await client.post(
                    "/api/mounts", json={"id": "stale", "path": str(other), "shared": True}
                )
                assert response.status == 201
                state = peer.call("folders.get")
                peer.call(
                    "folders.patch",
                    grants={
                        "revision": cast("dict[str, object]", state["grants"])["revision"],
                        "remove": [["mounts", "stale"]],
                    },
                    projects={"revision": cast("dict[str, object]", state["projects"])["revision"]},
                )
                current = await (await client.get("/api/mounts")).json()
                assert "stale" not in [row["id"] for row in current["mounts"]]
                assert service.settings_table.get("stale") is None
                if not restart:
                    assert table.get("stale") is None
                helper.close()
                response = await client.get("/api/mounts")
                assert response.status == 503
                assert "error" in await response.json()
            finally:
                await service.close()
                await client.close()

        run_scenario(scenario())
    assert not (root / "mounts.toml").exists()
    assert not (root / "project-folders.json").exists()


def test_helper_project_edit_after_shared_model_location_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_library import LibraryClient

    from dinkster.model_settings import ModelLocation, ModelSettings

    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    output = tmp_path / "output"
    output.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    table = MountTable()
    table.add(MountDef("output", output, "readwrite"), source="derived")
    with LibraryClient(tmp_path / "library") as helper:
        helper.call(
            "folders.models.patch",
            revision=0,
            patch={"locations": {}, "order": [], "downloadLocation": None},
        )
        service = MountService(table, library=helper, shared_models=True, allow_changes=True)
        service.refresh_models()

        async def scenario() -> None:
            client = await make_client(service)
            try:
                ModelSettings(
                    (ModelLocation("shared", models, "read", kind="model/lora"),),
                    None,
                    library=helper,
                ).save(ModelSettings.load(helper).revision)
                response = await client.put(
                    "/api/mounts/projects/default",
                    json={"mounts": ["shared-lora", "output"], "outputMount": "output"},
                )
                assert response.status == 200, await response.text()
                body = await response.json()
                assert [row["id"] for row in body["mounts"]] == ["shared-lora", "output"]
                assert body["outputMount"] == "output"
                record = cast("dict[str, object]", helper.call("folders.projects.get")["folders"])
                values = cast("dict[str, object]", record["values"])
                assert cast("dict[str, object]", values["projects"])["default"] == [
                    "shared-lora",
                    "output",
                ]
                assert ModelSettings.load(helper).locations[0].path == models
            finally:
                await service.close()
                await client.close()

        run_scenario(scenario())


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

    run_scenario(scenario())
    assert load_output_mount(config) == "renders"
    assert json.loads(snapshot.read_text("utf-8"))["outputMount"] == "renders"


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

    run_scenario(scenario())
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

    run_scenario(scenario())
    store.close()


@pytest.mark.network("pypi.org", "files.pythonhosted.org")
def test_project_folder_preferences_remain_readable_by_0_0_14(tmp_path: Path) -> None:
    import importlib.util
    import subprocess

    from dinkster_assets import MountsError

    from dinkster.project_folders import HiddenModelPreference, ProjectFolders

    legacy = tmp_path / "legacy"
    subprocess.run(
        ["uv", "pip", "install", "--target", str(legacy), "--no-deps", "dinkster==0.0.14"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    spec = importlib.util.spec_from_file_location(
        "legacy_project_folders", legacy / "dinkster/project_folders.py"
    )
    assert spec is not None and spec.loader is not None
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    path = tmp_path / "project-folders.json"
    folders = ProjectFolders(path)
    folders.initialize(("shared", "other"))
    folders.update("first", ["other", "shared"], available=("shared", "other"), output="other")
    folders.update("second", ["shared"], available=("shared", "other"), output="shared")

    def assert_preferences() -> None:
        restored = reader.ProjectFolders(path)
        assert restored.ordered("first", ("shared", "other")) == ("other", "shared")
        assert restored.output("first", None) == "other"
        assert restored.ordered("second", ("shared", "other")) == ("shared",)
        assert restored.output("second", None) == "shared"

    assert_preferences()
    for hidden in ({}, {"first": [], "second": []}):
        saved = json.loads(path.read_text())
        saved["hiddenModels"] = hidden
        path.write_text(json.dumps(saved))
        folders = ProjectFolders(path)
        assert_preferences()
    preference = HiddenModelPreference(
        "selected.safetensors", "model/checkpoint", ("blake3:" + "a" * 64,)
    )
    folders.set_model_hidden("first", preference, True)
    folders = ProjectFolders(path)
    assert folders.hidden_models("first") == preference.identities
    assert folders.hidden_models("second") == ()
    with pytest.raises(MountsError, match="invalid project folders document"):
        reader.ProjectFolders(path)
    folders.set_model_hidden("first", preference, False)
    assert_preferences()


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

    run_scenario(scenario())
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


def test_project_asset_routes_require_live_library_projects(
    tmp_path: Path, shared_model_folders: dict[str, Path]
) -> None:
    from dinkster_server.projects import add_project_routes

    from dinkster.guess_api import add_guess_routes
    from dinkster.workflow_requirements_api import add_workflow_requirement_routes

    root = tmp_path / "input"
    seed(root, {"source.txt": b"source"})
    table = MountTable(index_root=tmp_path / "indexes")
    table.add(MountDef("input", root))
    table.scan("input")
    service = MountService(table, tmp_path / "mounts.toml", allow_changes=True, shared_models=True)
    document = {
        "format": "dinkster-workflow",
        "formatVersion": 1,
        "graphs": {"root": {"nodes": {}}},
    }

    async def scenario() -> None:
        library = ServerLibrary(
            AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library.sqlite")
        )
        app = create_app(make_engine, SCHEMAS, library=library)
        add_project_routes(app, tmp_path, engine_version="test")
        add_mount_routes(app, service)
        add_guess_routes(app)
        add_workflow_requirement_routes(app, manifests=lambda: {}, engine_variant="linux-cpu")
        async with TestClient(TestServer(app)) as client:
            registered = await client.post("/api/projects", json={"id": "film"})
            assert registered.status == 200
            settings = await (await client.get("/api/models/settings")).json()
            for project, available in (("film", True), ("missing", False), ("film", False)):
                requests = (
                    ("GET", f"/api/mounts?project={project}", None),
                    ("GET", f"/api/mounts/locations?project={project}", None),
                    ("GET", f"/api/mounts/input/list?project={project}", None),
                    ("GET", f"/api/mounts/input/entries?project={project}", None),
                    ("GET", f"/api/mounts/entries?project={project}", None),
                    (
                        "POST",
                        "/api/mounts/resolve-path",
                        {"path": str(root / "source.txt"), "scope": "local", "project": project},
                    ),
                    (
                        "PUT",
                        f"/api/mounts/projects/{project}",
                        {"mounts": ["input"], "outputMount": None},
                    ),
                    ("GET", f"/api/models/settings?project={project}", None),
                    (
                        "PUT",
                        f"/api/models/settings?project={project}",
                        {"settings": settings["settings"], "revision": settings["revision"]},
                    ),
                    (
                        "PUT",
                        f"/api/models/locations/shared/projects/{project}",
                        {"enabled": True},
                    ),
                    ("POST", "/api/assets/guess", {"names": ["source.txt"], "project": project}),
                    ("POST", "/api/workflows/plan", {"workflows": [document], "project": project}),
                )
                for method, path, body in requests:
                    response = await client.request(method, path, json=body)
                    assert response.status == (200 if available else 400), (
                        method,
                        path,
                        await response.text(),
                    )
                    if not available:
                        assert "project is unavailable" in await response.text()
                if available:
                    deleted = await client.delete("/api/projects/film", json={"revision": 1})
                    assert deleted.status == 200
                else:
                    for method, path, body in (
                        (
                            "POST",
                            "/api/mounts",
                            {"id": "extra", "path": str(root), "project": project},
                        ),
                        ("PUT", f"/api/models/hidden/{project}", {}),
                        ("DELETE", f"/api/models/hidden/{project}", {"identity": "unknown"}),
                    ):
                        response = await client.request(method, path, json=body)
                        assert response.status == 400, await response.text()
                        assert "project is unavailable" in await response.text()
            assert table.get("extra") is None
            assert (root / "source.txt").read_bytes() == b"source"

    run_scenario(scenario())


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
                    ("GET", "/api/models/settings", None),
                    ("PUT", "/api/models/settings", {}),
                    ("PUT", "/api/models/locations/models/projects/default", {"enabled": True}),
                    (
                        "POST",
                        "/api/mounts/resolve-path",
                        {"path": str(tmp_path), "project": "default", "scope": "local"},
                    ),
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

    run_scenario(scenario())


def test_operator_default_import_destination_matches_upload_and_survives_revocation(
    tmp_path: Path,
) -> None:
    table = MountTable(output_mount="renders")
    (tmp_path / "actual-imports").mkdir()
    (tmp_path / "output").mkdir()
    table.add(MountDef("incoming", tmp_path / "actual-imports", "readwrite"))
    table.add(MountDef("renders", tmp_path / "output", "readwrite"))
    table.scan("incoming")
    table.scan("renders")
    service = MountService(table, allow_changes=True)

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "catalog.sqlite"),
            local_resolver=table,
            project_folders_for=service.project_context,
            upload_writer=AssetWriter(table),
            import_mount="incoming",
        )
        app = create_app(make_engine, SCHEMAS, library=library)
        add_mount_routes(app, service)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/mounts/locations")
            assert response.status == 200
            body = await response.json()
            assert body["importMount"] == "incoming"
            row = next(row for row in body["activeMounts"] if row["id"] == body["importMount"])
            assert row["path"] == str(tmp_path / "actual-imports")
            assert table.output_mount == "renders"
            response = await client.post(
                "/api/assets",
                params={"scope": "local", "name": "original.txt"},
                data=b"named default import",
            )
            assert response.status == 201
            assert (await response.json())["asset"]["virtualPath"] == "mounts/incoming/original.txt"
            assert (
                tmp_path / "actual-imports" / "original.txt"
            ).read_bytes() == b"named default import"
            assert not (tmp_path / "output" / "original.txt").exists()
            assert (await client.delete("/api/mounts/incoming")).status == 200
            body = await (await client.get("/api/mounts/locations")).json()
            assert body["importMount"] == "incoming"
            assert all(row["id"] != "incoming" for row in body["activeMounts"])
            response = await client.post(
                "/api/assets",
                params={"scope": "local", "name": "must-not-fallback.txt"},
                data=b"refused",
            )
            assert response.status == 403
            assert not (tmp_path / "output" / "must-not-fallback.txt").exists()
        finally:
            await client.close()

    run_scenario(scenario())


def test_project_input_choices_route_uploads_and_survive_restart(tmp_path: Path) -> None:
    from dinkster_assets import MountConfiguration
    from dinkster_library import LibraryClient
    from dinkster_server.projects import add_project_routes

    root = tmp_path / "library"
    mounts = tuple(
        MountDef(name, tmp_path / name, "read" if name == "reference" else "readwrite")
        for name in ("imports", "renders", "reference")
    )
    for mount in mounts:
        mount.path.mkdir()

    async def scenario() -> None:
        for restarted in (False, True):
            with LibraryClient(root) as helper:
                if not restarted:
                    MountConfiguration.load(helper).save(mounts, output_mount="renders")
                table = MountTable()
                service = MountService(table, library=helper, allow_changes=True)
                for mount in mounts:
                    table.scan(mount.id)
                library = ServerLibrary(
                    vault=AssetVault(root / "vault"),
                    store=LibraryStore(helper),
                    local_resolver=table,
                    project_folders_for=service.project_context,
                    upload_writer=AssetWriter(table),
                    import_mount=lambda project, folders=service.projects: (
                        folders.input(project, "imports") or "imports"
                    ),
                )
                app = create_app(make_engine, SCHEMAS, library=library)
                add_project_routes(app, helper, engine_version="0.0.18")
                add_mount_routes(app, service)
                async with TestClient(TestServer(app)) as client:
                    if not restarted:
                        for project in ("film", "portrait"):
                            response = await client.post(
                                "/api/projects",
                                json={"id": project, "filesFolder": str(tmp_path / project)},
                            )
                            assert response.status == 200, await response.text()
                            await wait_for_state(
                                client, f"{project}-input", "ready", project=project
                            )
                            response = await client.post(
                                "/api/assets",
                                params={"scope": "local", "project": project, "name": "first.txt"},
                                data=project.encode(),
                            )
                            assert response.status == 201, await response.text()
                            assert (
                                tmp_path / project / "input" / "first.txt"
                            ).read_bytes() == project.encode()
                            response = await client.put(
                                f"/api/mounts/projects/{project}",
                                json={
                                    "mounts": [f"{project}-input", "renders"],
                                    "outputMount": "renders",
                                    "inputMount": f"{project}-input",
                                },
                            )
                            assert response.status == 200, await response.text()
                            response = await client.put(
                                f"/api/mounts/projects/{project}",
                                json={
                                    "mounts": ["renders", f"{project}-input"],
                                    "outputMount": "renders",
                                },
                            )
                            assert response.status == 200, await response.text()
                            response = await client.put(
                                f"/api/mounts/projects/{project}",
                                json={"mounts": ["renders"], "outputMount": "renders"},
                            )
                            assert response.status == 400, await response.text()
                            response = await client.post(
                                "/api/assets",
                                params={"scope": "local", "project": project, "name": "source.txt"},
                                data=project.encode(),
                            )
                            assert response.status == 201, await response.text()
                            assert (
                                tmp_path / project / "input" / "source.txt"
                            ).read_bytes() == project.encode()
                        for destination in ("reference", "unknown", "portrait-input"):
                            response = await client.put(
                                "/api/mounts/projects/film",
                                json={
                                    "mounts": ["film-input", "renders", "reference"],
                                    "outputMount": "renders",
                                    "inputMount": destination,
                                },
                            )
                            assert response.status == 400, await response.text()
                    else:
                        for project in ("film", "portrait"):
                            await wait_for_state(
                                client, f"{project}-input", "ready", project=project
                            )
                            locations = await (
                                await client.get(
                                    "/api/mounts/locations", params={"project": project}
                                )
                            ).json()
                            assert locations["importMount"] == f"{project}-input"
                        response = await client.put(
                            "/api/mounts/projects/portrait",
                            json={
                                "mounts": ["film-input", "renders"],
                                "outputMount": "renders",
                                "inputMount": "film-input",
                            },
                        )
                        assert response.status == 200, await response.text()
                        response = await client.post(
                            "/api/assets",
                            params={"scope": "local", "project": "portrait", "name": "source.txt"},
                            data=b"portrait",
                        )
                        assert response.status == 201, await response.text()
                        assert (
                            tmp_path / "film" / "input" / "source_00001.txt"
                        ).read_bytes() == b"portrait"
                        assert (tmp_path / "film" / "input" / "source.txt").read_bytes() == b"film"
                        assert (await client.delete("/api/mounts/film-input")).status == 200
                        response = await client.post(
                            "/api/assets",
                            params={"scope": "local", "project": "portrait", "name": "refused.txt"},
                            data=b"refused",
                        )
                        assert response.status == 403, await response.text()
                        assert not (tmp_path / "imports" / "refused.txt").exists()
                        assert not (tmp_path / "renders" / "refused.txt").exists()
                await service.close()

    run_scenario(scenario())


def test_operator_asset_locations_verify_scoped_project_content(tmp_path: Path) -> None:
    table = MountTable()
    seed(tmp_path / "imports", {"first.bin": b"first", "other.bin": b"other"})
    table.add(MountDef("imports", tmp_path / "imports", "readwrite"))
    table.scan("imports")
    service = MountService(table, allow_changes=True)
    store = LibraryStore(tmp_path / "catalog.sqlite")
    vault = AssetVault(tmp_path / "vault")
    first = table.ref("mounts/imports/first.bin")
    other = table.ref("mounts/imports/other.bin")
    for ref in (first, other):
        store.record_asset(
            "local", ref, original_name=ref.name, kind="asset/file", origin={"kind": "upload"}
        )
    with vault.writer(first.digest) as writer:
        writer.write(b"first")
        writer.commit()

    async def scenario() -> None:
        library = ServerLibrary(
            vault=vault,
            store=store,
            local_resolver=table,
            project_folders_for=service.project_context,
            upload_writer=AssetWriter(table),
        )
        app = create_app(make_engine, SCHEMAS, library=library)
        add_mount_routes(app, service)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:

            async def locations(**params: str) -> dict[str, object]:
                response = await client.get("/api/mounts/locations", params=params)
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                return await response.json()

            response = await client.post(
                "/api/assets",
                params={"scope": "local", "name": "just-uploaded.bin"},
                data=b"newly published bytes",
            )
            assert response.status == 201
            uploaded = await response.json()
            body = await locations(scope="local", digest=uploaded["digest"])
            assert body["locations"] == [
                {
                    "storage": "mount",
                    "mount": "imports",
                    "virtualPath": "mounts/imports/just-uploaded.bin",
                    "path": str(tmp_path / "imports" / "just-uploaded.bin"),
                }
            ]

            body = await locations(scope="local", digest=first.digest)
            assert body["vault"] == str(vault.root)
            assert body["locations"] == [
                {"storage": "vault", "path": str(vault.resolve(first.digest))},
                {
                    "storage": "mount",
                    "mount": "imports",
                    "virtualPath": first.virtual_path,
                    "path": str(tmp_path / "imports" / "first.bin"),
                },
            ]
            seed(tmp_path / "imports", {"alias.bin": b"first"})
            table.scan("imports")
            alias = table.ref("mounts/imports/alias.bin")
            store.record_asset(
                "local",
                alias,
                original_name=alias.name,
                kind="asset/file",
                origin={"kind": "upload"},
            )
            alias_location = {
                "storage": "mount",
                "mount": "imports",
                "virtualPath": alias.virtual_path,
                "path": str(tmp_path / "imports" / "alias.bin"),
            }
            body = await locations(digest=first.digest)
            assert body["locations"] == [
                {"storage": "vault", "path": str(vault.resolve(first.digest))},
                {
                    "storage": "mount",
                    "mount": "imports",
                    "virtualPath": first.virtual_path,
                    "path": str(tmp_path / "imports" / "first.bin"),
                },
                alias_location,
            ]
            service.projects.update("default", [], available=("imports",), output=None)
            assert (await locations(digest=first.digest))["locations"] == [
                {"storage": "vault", "path": str(vault.resolve(first.digest))}
            ]
            service.projects.update("default", ["imports"], available=("imports",), output=None)
            (tmp_path / "imports" / "alias.bin").unlink()
            table.scan("imports")
            assert (await locations(scope="elsewhere", digest=first.digest))["locations"] == []
            assert (await locations(project="other", digest=first.digest))["locations"] == []
            ordinary = await client.get("/api/assets/catalog", params={"scope": "local"})
            assert str(tmp_path) not in await ordinary.text()
            (tmp_path / "imports" / "first.bin").write_bytes(b"wrong")
            table.scan("imports")
            body = await locations(digest=first.digest)
            assert body["locations"] == [
                {"storage": "vault", "path": str(vault.resolve(first.digest))}
            ]
            path = vault.resolve(first.digest)
            assert path is not None
            path.write_bytes(b"wrong")
            assert (await locations(digest=first.digest))["locations"] == []
            table.remove("imports")
            assert (await locations(digest=other.digest))["locations"] == []
            for params in ({"digest": "bad"}, {"unknown": "field"}):
                response = await client.get("/api/mounts/locations", params=params)
                assert response.status == 400
        finally:
            await client.close()

    try:
        run_scenario(scenario())
    finally:
        store.close()


@pytest.mark.parametrize("boundary", ("snapshot", "verification"))
@pytest.mark.parametrize("change", ("revoke", "replace", "exclude", "content", "escape"))
def test_asset_locations_refuse_changed_files_and_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, change: str
) -> None:
    import dinkster_assets.library as library_module

    import dinkster.mounts_api as mounts_api_module

    root = tmp_path / "input"
    seed(root, {"source.bin": b"source"})
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"source")
    table = MountTable()
    table.add(MountDef("input", root))
    table.scan("input")
    service = MountService(table)
    store = LibraryStore(tmp_path / "catalog.sqlite")
    ref = table.ref("mounts/input/source.bin")
    store.record_asset(
        "local", ref, original_name=ref.name, kind="asset/file", origin={"kind": "upload"}
    )
    entered, release = Event(), Event()
    module = library_module if boundary == "snapshot" else mounts_api_module
    name = "load_write_records" if boundary == "snapshot" else "verified_local_path"
    original = getattr(module, name)

    def held(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(module, name, held)

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"), store=store, local_resolver=table
        )
        app = create_app(make_engine, SCHEMAS, library=library)
        add_mount_routes(app, service)
        client = TestClient(TestServer(app))
        await client.start_server()
        pending = asyncio.ensure_future(
            client.get("/api/mounts/locations", params={"digest": ref.digest})
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            if change in ("revoke", "replace"):
                table.remove("input")
                if change == "replace":
                    table.add(MountDef("input", root))
            elif change == "exclude":
                service.projects.update("default", [], available=("input",), output=None)
            elif change == "content":
                (root / "source.bin").write_bytes(b"changed")
            else:
                (root / "source.bin").unlink()
                symlink_or_skip(root / "source.bin", outside)
            release.set()
            response = await pending
            assert response.status == 200
            assert (await response.json())["locations"] == []
        finally:
            release.set()
            await pending
            await client.close()

    try:
        run_scenario(scenario())
    finally:
        store.close()


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

    run_scenario(scenario())
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

    run_scenario(scenario())


@pytest.mark.parametrize("helper", (False, True))
def test_resolve_path_catalogs_exact_current_bytes_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper: bool
) -> None:
    import os

    from dinkster_server.projects import add_project_routes

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "account"))

    root = tmp_path / "read-only-input"
    seed(root, {"indexed.txt": b"old bytes", "sibling.bin": b"keep sibling"})
    table = MountTable(tmp_path / "snapshot.json", index_root=tmp_path / "indexes")
    table.add(MountDef("input", root))
    table.scan("input")
    service = MountService(table, tmp_path / "mounts.toml")
    original_stat = (root / "indexed.txt").stat()
    (root / "indexed.txt").write_bytes(b"new bytes")
    os.utime(root / "indexed.txt", ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    seed(root, {"new.txt": b"uncataloged", "empty.txt": b""})
    original_names = {path.name for path in root.iterdir()}

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        if helper:
            add_project_routes(app, tmp_path / "library", engine_version="test")
        add_mount_routes(app, service)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/mounts/locations")
            assert (await response.json())["resolvePath"] is True
            for name, data in (
                ("indexed.txt", b"new bytes"),
                ("new.txt", b"uncataloged"),
                ("empty.txt", b""),
            ):
                before = (root / name).stat()
                response = await client.post(
                    "/api/mounts/resolve-path",
                    json={"path": str(root / name), "project": "default", "scope": "local"},
                )
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert await response.json() == {
                    "status": "resolved",
                    "asset": {
                        "digest": digest_bytes(data),
                        "name": name,
                        "size": len(data),
                        "mediaType": "text/plain",
                        "virtualPath": f"mounts/input/{name}",
                    },
                }
                assert str(root) not in await response.text()
                after = (root / name).stat()
                assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                assert (root / name).read_bytes() == data
            assert {path.name for path in root.iterdir()} == original_names
            assert not (tmp_path / "vault").exists()
            assert not (tmp_path / "imports").exists()
            restored = MountSnapshotResolver(tmp_path / "snapshot.json")
            assert restored.resolve(digest_bytes(b"uncataloged")) == root / "new.txt"
            assert table.ref("mounts/input/sibling.bin").digest == digest_bytes(b"keep sibling")
            response = await client.get("/api/mounts/input/entries")
            assert {row["digest"] for row in (await response.json())["entries"]} == {
                digest_bytes(data) for data in (b"new bytes", b"keep sibling", b"uncataloged", b"")
            }
            if helper:
                index = tmp_path / "indexes" / "input.json"
                index.unlink()
                index.mkdir()
                response = await client.post(
                    "/api/mounts/resolve-path",
                    json={"path": str(root / "new.txt"), "project": "default", "scope": "local"},
                )
                assert response.status == 500
                assert await response.json() == {"error": "mount-resolution-failed"}
        finally:
            await client.close()

    run_scenario(scenario())


def test_resolve_path_refusals_validation_and_scope(tmp_path: Path) -> None:
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    root = tmp_path / "input"
    seed(root, {"source.txt": b"source"})
    seed(tmp_path / "outside", {"secret.txt": b"secret"})
    seed(root, {"partial.tmp": b"not complete", ".private.txt": b"hidden"})
    symlink_or_skip(root / "escape.txt", tmp_path / "outside" / "secret.txt")
    table = MountTable(index_root=tmp_path / "indexes")
    table.add(MountDef("input", root))
    table.scan("input")
    service = MountService(table)
    service.projects.update("excluded", [], available=("input",), output=None)

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=StaticBearerAuthenticator(
                {
                    "operator": Principal(
                        "operator", {"local": frozenset({"assets:read", "settings:write"})}
                    )
                }
            ),
        )
        add_mount_routes(app, service)
        client = TestClient(TestServer(app), headers={"Authorization": "Bearer operator"})
        await client.start_server()
        try:
            for path, project in (
                (root, "default"),
                (root / "missing.txt", "default"),
                (tmp_path / "outside" / "secret.txt", "default"),
                (root / "escape.txt", "default"),
                (root / "source.txt", "excluded"),
                (root / "partial.tmp", "default"),
                (root / ".private.txt", "default"),
            ):
                response = await client.post(
                    "/api/mounts/resolve-path",
                    json={"path": str(path), "project": project, "scope": "local"},
                )
                assert response.status == 200
                assert await response.json() == {"status": "refused", "reason": "not-resolvable"}
            valid = {"path": str(root / "source.txt"), "project": "default", "scope": "local"}
            for malformed in (
                {},
                [],
                {**valid, "extra": True},
                {**valid, "path": "relative.txt"},
                {**valid, "path": ""},
                {**valid, "path": "\0"},
                {**valid, "path": 42},
                {**valid, "project": "Bad Project"},
                {**valid, "scope": " "},
            ):
                response = await client.post("/api/mounts/resolve-path", json=malformed)
                assert response.status == 400
                assert await response.json() == {"error": "invalid-path-request"}
                assert str(tmp_path) not in await response.text()
            foreign_path = (
                "C:\\outside\\file.txt" if Path("/").is_absolute() else "/outside/file.txt"
            )
            response = await client.post(
                "/api/mounts/resolve-path", json={**valid, "path": foreign_path}
            )
            assert response.status == 200
            assert await response.json() == {"status": "refused", "reason": "not-resolvable"}
            response = await client.post("/api/mounts/resolve-path?extra=yes", json=valid)
            assert response.status == 400
            response = await client.post("/api/mounts/resolve-path", data="not json")
            assert response.status == 400
            response = await client.post(
                "/api/mounts/resolve-path", json={**valid, "scope": "ungranted"}
            )
            assert response.status == 403
            table.remove("input")
            response = await client.post("/api/mounts/resolve-path", json=valid)
            assert await response.json() == {"status": "refused", "reason": "not-resolvable"}
        finally:
            await client.close()

    run_scenario(scenario())


@pytest.mark.parametrize("change", ("revoke", "exclude", "replace", "persistence"))
def test_resolve_path_rechecks_live_authority_and_reports_storage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    import dinkster_assets.library as library_module

    root = tmp_path / "input"
    seed(root, {"source.txt": b"source"})
    table = MountTable(index_root=tmp_path / "indexes")
    table.add(MountDef("input", root))
    table.scan("input")
    service = MountService(table)
    entered, release = Event(), Event()

    def digest(path: Path):
        entered.set()
        assert release.wait(10)
        return real_digest_file_with_record(path)

    monkeypatch.setattr(library_module, "digest_file_with_record", digest)
    if change == "persistence":
        monkeypatch.setattr(LocalAssetLibrary, "_save_index", lambda *_: False)

    async def scenario() -> None:
        client = await make_client(service)
        pending = asyncio.ensure_future(
            client.post(
                "/api/mounts/resolve-path",
                json={"path": str(root / "source.txt"), "project": "default", "scope": "local"},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            if change in ("revoke", "replace"):
                table.remove("input")
                if change == "replace":
                    table.add(MountDef("input", root))
            elif change == "exclude":
                service.projects.update("default", [], available=("input",), output=None)
            release.set()
            response = await pending
            assert response.status == (500 if change == "persistence" else 200)
            assert await response.json() == (
                {"error": "mount-resolution-failed"}
                if change == "persistence"
                else {"status": "refused", "reason": "not-resolvable"}
            )
            assert str(tmp_path) not in await response.text()
        finally:
            release.set()
            await pending
            await client.close()

    run_scenario(scenario())
