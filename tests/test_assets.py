"""M3 asset core (DESIGN 3.12): identity is a content digest, references
are virtual paths with file-like query semantics, nodes receive AssetRefs
and never filesystem paths, and an asset's cache fingerprint IS its digest
(hazard H4: location-independent cache keys)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_assets import (
    ASSET_TYPE,
    AssetCatalog,
    AssetEntry,
    AssetError,
    AssetRef,
    AssetResolver,
    AssetVault,
    IndexedAssetResolver,
    LibraryStore,
    LocalAssetLibrary,
    LocalAssetRef,
    MountDef,
    MountSnapshotResolver,
    MountTable,
    digest_bytes,
    is_digest,
    register_asset_type,
)
from dinkster_assets.value import value_asset_references
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_schema import (
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    AUDIO_RECORD_TYPE,
    CORE_INT,
    CORE_STRING,
    TypeRegistry,
    make_list_value,
    make_tuple_value,
    register_core_types,
)
from dinkster_workers import InProcessWorker, ValueCodec

from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip

# --- identity ---------------------------------------------------------------


def test_bad_digests_rejected() -> None:
    for bad in ("", "blake3:short", "sha256:" + "0" * 64, "blake3:" + "G" * 64):
        assert not is_digest(bad)
        with pytest.raises(AssetError):
            AssetRef(digest=bad, name="x", size=1)


# --- catalog: file-like semantics without file access ------------------------


def entry(path: str, data: bytes) -> AssetEntry:
    return AssetEntry(virtual_path=path, digest=digest_bytes(data), size=len(data))


def test_catalog_folder_listing_and_glob() -> None:
    catalog = AssetCatalog(
        [
            entry("models/checkpoints/sd15.safetensors", b"sd15"),
            entry("models/checkpoints/flux/dev.safetensors", b"flux"),
            entry("models/loras/detail.safetensors", b"lora"),
            entry("inputs/pose.png", b"png"),
        ]
    )
    top = catalog.list_folder()
    assert top.folders == ("inputs", "models")
    assert top.entries == ()

    checkpoints = catalog.list_folder("models/checkpoints")
    assert checkpoints.folders == ("flux",)
    assert [e.name for e in checkpoints.entries] == ["sd15.safetensors"]

    assert [e.virtual_path for e in catalog.glob("models/*/*.safetensors")] == [
        "models/checkpoints/sd15.safetensors",
        "models/loras/detail.safetensors",
    ]


def test_catalog_duplicate_content_one_identity() -> None:
    data = b"the same weights twice"
    catalog = AssetCatalog(
        [entry("models/a.safetensors", data), entry("models/copies/b.safetensors", data)]
    )
    hits = catalog.by_digest(digest_bytes(data))
    assert sorted(e.virtual_path for e in hits) == [
        "models/a.safetensors",
        "models/copies/b.safetensors",
    ]


def test_virtual_paths_reject_traversal() -> None:
    for bad in ("../x", "a/../b", "a//b", "a\\b", "./a"):
        with pytest.raises(AssetError):
            AssetEntry(virtual_path=bad, digest=digest_bytes(b"x"), size=1)


# --- library: real directories, scanned once, digests cached -----------------


def make_models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    (root / "checkpoints").mkdir(parents=True)
    (root / "loras").mkdir()
    (root / "checkpoints" / "tiny.safetensors").write_bytes(b"tiny checkpoint bytes")
    (root / "loras" / "style.safetensors").write_bytes(b"lora bytes")
    (root / ".hidden").write_bytes(b"skip me")
    return root


def test_library_scan_catalog_and_resolve(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    assert library.scan() == 2

    listing = library.catalog.list_folder("models")
    assert listing.folders == ("checkpoints", "loras")

    ref = library.ref("models/checkpoints/tiny.safetensors")
    assert ref.digest == digest_bytes(b"tiny checkpoint bytes")
    assert ref.read_bytes() == b"tiny checkpoint bytes"
    # resolve() is by identity, not by name
    assert library.resolve(ref.digest) == root / "checkpoints" / "tiny.safetensors"
    assert library.resolve(digest_bytes(b"absent")) is None


def test_indexed_resolver_serves_digests_without_hashing(tmp_path: Path) -> None:
    """The worker-side resolver: reads the scan's index, never hashes.
    A file changed since the scan is absent, not served with a stale
    identity."""
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()
    ref = library.ref("models/checkpoints/tiny.safetensors")

    resolver = library.resolver()
    assert resolver.resolve(ref.digest) == root / "checkpoints" / "tiny.safetensors"
    assert resolver.resolve(digest_bytes(b"absent")) is None

    # Modify the file after the scan: the index row no longer matches.
    (root / "checkpoints" / "tiny.safetensors").write_bytes(b"different bytes now")
    fresh = IndexedAssetResolver(root)
    assert fresh.resolve(ref.digest) is None


def test_indexed_resolver_reuses_ingest_verification_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()
    ref = library.ref("models/checkpoints/tiny.safetensors")
    worker_ref = AssetRef.from_wire(ref.to_wire(), IndexedAssetResolver(root))

    import dinkster_assets.integrity as integrity_module

    def explode(_handle: object) -> str:
        raise AssertionError("recorded asset was rehashed")

    monkeypatch.setattr(integrity_module, "_hash_handle", explode)
    assert worker_ref.read_bytes() == b"tiny checkpoint bytes"


def test_library_default_ignore_skips_placeholders_junk_and_partials(
    tmp_path: Path,
) -> None:
    root = make_models_dir(tmp_path)
    (root / "checkpoints" / "put_checkpoints_here").write_bytes(b"")
    (root / "checkpoints" / "Thumbs.db").write_bytes(b"windows junk")
    (root / "loras" / "big-model.safetensors.part").write_bytes(b"half a download")
    (root / "loras" / "queued.crdownload").write_bytes(b"chrome partial")
    library = LocalAssetLibrary(root)
    assert library.scan() == 2  # only the two real model files
    assert library.catalog.get("models/checkpoints/put_checkpoints_here") is None


def test_library_custom_ignore_replaces_defaults(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    (root / "checkpoints" / "put_checkpoints_here").write_bytes(b"")
    library = LocalAssetLibrary(root, ignore=("loras/*",))
    count = library.scan()
    # loras/ pattern anchors to the relative path; the placeholder is now
    # cataloged because custom patterns replace DEFAULT_IGNORE.
    assert count == 2
    assert library.catalog.get("models/checkpoints/put_checkpoints_here") is not None
    assert library.catalog.get("models/loras/style.safetensors") is None


@pytest.mark.parametrize(
    "extension",
    [".part", ".partial", ".CRDOWNLOAD", ".download", ".tmp", ".aria2", ".!qb", ".opdownload"],
)
def test_library_rejects_partial_downloads_even_with_custom_ignore(
    tmp_path: Path, extension: str
) -> None:
    (tmp_path / f"model.safetensors{extension}").write_bytes(b"partial model")
    (tmp_path / "finished.safetensors").write_bytes(b"finished model")
    library = LocalAssetLibrary(tmp_path, ignore=())
    assert library.scan() == 1
    assert [entry.name for entry in library.entries()] == ["finished.safetensors"]


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_library_defers_unstable_file_until_a_later_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = tmp_path / "downloading.safetensors"
    path.write_bytes(b"initial")
    stable = tmp_path / "stable.safetensors"
    stable.write_bytes(b"stable")
    first = path.stat()

    def change_between_stats(_delay: float) -> None:
        if change == "size":
            path.write_bytes(b"longer download")
            os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
        else:
            os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns + 1_000_000))

    library = LocalAssetLibrary(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr("dinkster_assets.library.time.sleep", change_between_stats)
        assert library.scan() == 1
        assert [entry.name for entry in library.entries()] == ["stable.safetensors"]
    assert library.scan() == 2
    assert library.ref("models/downloading.safetensors").read_bytes() == path.read_bytes()


def test_library_never_catalogs_its_own_index(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    # A non-dot index name inside the root would otherwise look like a file.
    library = LocalAssetLibrary(root, index_path=root / "index.json")
    library.scan()
    library2 = LocalAssetLibrary(root, index_path=root / "index.json")
    assert library2.scan() == 2
    assert library2.catalog.get("models/index.json") is None


def test_library_index_skips_rehash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()

    import dinkster_assets.library as library_module

    def explode(path: Path) -> tuple[str, None]:
        raise AssertionError(f"rehash of unchanged file: {path}")

    monkeypatch.setattr(library_module, "digest_file_with_record", explode)
    fresh = LocalAssetLibrary(root)
    assert fresh.scan() == 2  # digests come from the index; no hashing

    # Touching content invalidates that entry only.
    monkeypatch.undo()
    target = root / "loras" / "style.safetensors"
    target.write_bytes(b"different lora bytes")
    rescanned = LocalAssetLibrary(root)
    rescanned.scan()
    assert rescanned.ref("models/loras/style.safetensors").digest == digest_bytes(
        b"different lora bytes"
    )


# --- value type: fingerprint is the digest; boundary carries no bytes --------


def asset_registry(resolver: AssetResolver | None = None) -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry, resolver)
    register_asset_type(registry, resolver, type_id=AUDIO_RECORD_TYPE)
    return registry


def test_complete_asset_references_traverse_beyond_display_limits() -> None:
    registry = asset_registry()
    refs = [AssetRef(digest_bytes(str(i).encode()), f"asset-{i}", i) for i in range(70)]
    value = make_tuple_value(
        [
            registry.wrap(f"asset<{CORE_STRING}>", refs[0]),
            registry.wrap(f"list<{ASSET_TYPE}>", refs[1:]),
        ]
    )
    for _ in range(36):
        value = make_list_value("*", [value])
    assert value_asset_references({"output": value}, complete_registry=registry) == {
        ref.digest: ref for ref in refs
    }
    with pytest.raises(AssetError, match="cannot exclude"):
        value_asset_references(
            {"output": value}, complete_registry=registry, include_asset_values=False
        )


def test_complete_asset_references_require_explicit_atom_declarations() -> None:
    registry = asset_registry()
    registry.register("example.opaque", encode=lambda _: b"opaque", decode=lambda _: None)
    opaque = registry.wrap("example.opaque", None)
    assert registry.spec("example.opaque").declared_codec
    assert value_asset_references({"output": opaque}) == {}
    with pytest.raises(AssetError, match="incomplete"):
        value_asset_references(
            {"output": make_list_value("*", [opaque])}, complete_registry=registry
        )
    ref = AssetRef(digest_bytes(b"alias"), "alias", 5)
    with pytest.raises(AssetError, match="incomplete"):
        value_asset_references(
            {"output": registry.wrap(AUDIO_RECORD_TYPE, ref)}, complete_registry=registry
        )
    registry.register("example.empty", asset_references_complete=True)
    assert (
        value_asset_references(
            {"output": registry.wrap("example.empty", None)}, complete_registry=registry
        )
        == {}
    )


def test_wrap_rejects_paths() -> None:
    registry = asset_registry()
    for type_id in (ASSET_TYPE, AUDIO_RECORD_TYPE):
        with pytest.raises(AssetError, match="never.*filesystem paths"):
            registry.wrap(type_id, "/etc/passwd")


def test_codec_crossing_carries_identity_not_bytes(tmp_path: Path) -> None:
    """An AssetRef over the boundary is a tiny JSON envelope; the receiving
    side rebinds its own resolver. Fingerprints are unchanged by the hop."""
    root = make_models_dir(tmp_path)
    sender_library = LocalAssetLibrary(root)
    sender_library.scan()
    sender = ValueCodec(asset_registry(sender_library))
    receiver_library = LocalAssetLibrary(root, index_path=tmp_path / "idx2.json")
    receiver_library.scan()
    receiver = ValueCodec(asset_registry(receiver_library))

    value = asset_registry(sender_library).wrap(
        ASSET_TYPE, sender_library.ref("models/loras/style.safetensors").to_wire()
    )
    blobs: list[bytes] = []
    wire, stat = sender.encode(value, blobs, [])
    assert stat.size_bytes < 512  # identity + metadata, never content
    crossed, _ = receiver.decode(wire, blobs, [])
    assert crossed.fingerprint == value.fingerprint
    ref = crossed.resolve()
    assert isinstance(ref, AssetRef)
    assert ref.read_bytes() == b"lora bytes"


def test_unverified_local_binding_crosses_worker_codec_and_reconciles_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    data = b"not yet hashed model bytes"
    (root / "weights.safetensors").write_bytes(data)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    table.add(MountDef(id="checkpoints", path=root), kind="model/checkpoint")

    def forbid_hash(*args: object, **kwargs: object) -> None:
        raise AssertionError("unverified name binding must not hash file content")

    with monkeypatch.context() as patch:
        patch.setattr("dinkster_assets.library.digest_file_with_record", forbid_hash)
        patch.setattr("dinkster_assets.identity.digest_file", forbid_hash)
        ref = table.bind_name("weights.safetensors", "model/checkpoint")
        assert isinstance(ref, LocalAssetRef)
        assert ref.read_bytes() == data
        registry = asset_registry(table)
        value = registry.wrap(ASSET_TYPE, ref)
        assert value.meta.get("verification") == "unverified"
        assert "digest" not in value.meta.entries
        assert not is_digest(value.fingerprint)
        assert "digest" not in ref.to_wire()
        assert str(root) not in json.dumps(ref.to_wire())
        worker_registry = asset_registry(MountSnapshotResolver(snapshot))
        blobs: list[bytes] = []
        wire, _ = ValueCodec(registry).encode(value, blobs, [])
        crossed, _ = ValueCodec(worker_registry).decode(wire, blobs, [])
        assert crossed.fingerprint == value.fingerprint
        worker_ref = crossed.resolve()
        assert isinstance(worker_ref, LocalAssetRef)
        assert worker_ref.read_bytes() == data
        foreign = LocalAssetRef.from_wire(ref.to_wire(), MountTable())
        with pytest.raises(AssetError, match="unavailable"):
            foreign.read_bytes()

    assert table.scan("checkpoints") == 1
    reconciled = registry.wrap(ASSET_TYPE, ref)
    verified = reconciled.resolve()
    assert isinstance(verified, AssetRef)
    assert verified.digest == digest_bytes(data)
    assert reconciled.fingerprint == verified.digest
    assert reconciled.meta.get("digest") == verified.digest
    assert isinstance(worker_registry.wrap(ASSET_TYPE, worker_ref).resolve(), AssetRef)


@pytest.mark.parametrize("verified", (False, True))
def test_text_encoder_relative_preflight_preserves_ranked_file_authority(
    tmp_path: Path, verified: bool
) -> None:
    from dinkster_server.library import ServerLibrary
    from dinkster_server.preflight import bind_model_names

    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    for mount_id, kind, priority in (
        ("wrong", "model/checkpoint", -10),
        ("preferred", "model/text-encoder", 0),
        ("fallback", "model/text-encoder", 10),
    ):
        root = tmp_path / mount_id
        for subdir in ("vendor", "other"):
            path = root / subdir / "encoder.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"{mount_id}/{subdir}".encode())
        table.add(MountDef(mount_id, root, priority=priority, kind=kind))
        if verified:
            table.scan(mount_id)
    assert table.bind_name("encoder.bin", "model/text-encoder") is None
    schema = NodeSchema(
        "encoder.load",
        inputs=(
            InputSpec(
                "file", TypeExpr.concrete(ASSET_TYPE), widget=AssetWidget(kind="model/text-encoder")
            ),
        ),
    )
    graph = Graph({"load": GraphNode("encoder.load", {"file": "vendor/encoder.bin"})})
    library = ServerLibrary(
        AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library.sqlite"), resolver=table
    )
    bound, missing = bind_model_names(graph, {schema.node_type: schema}, library)
    assert not missing
    wire = bound.nodes["load"].inputs["file"]
    ref = asset_registry(MountSnapshotResolver(snapshot)).wrap(ASSET_TYPE, wire).resolve()
    assert isinstance(ref, AssetRef if verified else LocalAssetRef)
    assert ref.read_bytes() == b"preferred/vendor"
    assert str(tmp_path) not in json.dumps(wire)
    table.remove("preferred")
    next_ref = table.bind_name("vendor/encoder.bin", "model/text-encoder")
    assert next_ref is not None and next_ref.read_bytes() == b"fallback/vendor"
    assert table.bind_name("vendor/encoder.bin", "model/checkpoint") is None


@pytest.mark.parametrize("change", ("replace", "revoke", "escape"))
def test_text_encoder_relative_binding_rechecks_host_and_worker_authority(
    tmp_path: Path, change: str
) -> None:
    root = tmp_path / "models"
    path = root / "vendor" / "encoder.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot)
    table.add(MountDef("encoders", root, kind="model/text-encoder"))
    ref = table.bind_name("vendor/encoder.bin", "model/text-encoder")
    assert isinstance(ref, LocalAssetRef)
    worker_ref = LocalAssetRef.from_wire(ref.to_wire(), MountSnapshotResolver(snapshot))
    assert worker_ref.read_bytes() == b"original"
    if change == "revoke":
        table.remove("encoders")
    else:
        path.unlink()
        if change == "replace":
            path.write_bytes(b"changed")
        else:
            outside = tmp_path / "outside.bin"
            outside.write_bytes(b"secret")
            symlink_or_skip(path, outside)
            assert table.bind_name("vendor/encoder.bin", "model/text-encoder") is None
    for current in (ref, worker_ref):
        with pytest.raises(AssetError, match="unavailable"):
            current.read_bytes()


@pytest.mark.parametrize(
    "name",
    (
        "../outside.bin",
        "/outside.bin",
        "C:/outside.bin",
        "C:outside.bin",
        "vendor/../outside.bin",
        "vendor/./encoder.bin",
        "vendor//encoder.bin",
        "vendor\\encoder.bin",
        "vendor/encoder.bin/",
        "vendor/\x00encoder.bin",
        "vendor/\nencoder.bin",
    ),
)
def test_text_encoder_relative_binding_rejects_unsafe_source_names(tmp_path: Path, name: str):
    table = MountTable()
    table.add(MountDef("encoders", tmp_path, kind="model/text-encoder"))
    assert table.bind_name(name, "model/text-encoder") is None


@pytest.mark.parametrize("change", ["replace", "revoke", "regrant"])
def test_unverified_local_binding_cannot_reuse_cache_after_change_or_revocation(
    tmp_path: Path, change: str
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    path = root / "weights.safetensors"
    path.write_bytes(b"original model")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    mount = MountDef(id="models", path=root)
    table.add(mount, kind="model/checkpoint")
    ref = table.bind_name(path.name, "model/checkpoint")
    assert isinstance(ref, LocalAssetRef)
    worker_ref = LocalAssetRef.from_wire(ref.to_wire(), MountSnapshotResolver(snapshot))
    if change == "replace":
        path.unlink()
        path.write_bytes(b"different data")
    else:
        table.remove("models")
        if change == "regrant":
            table.add(mount, kind="model/checkpoint")
    with pytest.raises(AssetError, match="unavailable"):
        asset_registry(table).wrap(ASSET_TYPE, ref)
    with pytest.raises(AssetError, match="unavailable"):
        worker_ref.read_bytes()


def test_unresolvable_ref_still_flows_but_fails_to_materialize() -> None:
    registry = asset_registry()  # no resolver anywhere
    ref_wire = {"digest": digest_bytes(b"elsewhere"), "name": "far.safetensors", "size": 9}
    value = registry.wrap(ASSET_TYPE, ref_wire)
    assert value.fingerprint == ref_wire["digest"]  # caching/interrogation fine
    ref = value.resolve()
    assert isinstance(ref, AssetRef)
    with pytest.raises(AssetError, match="no resolver bound"):
        ref.local_path()


# --- end to end: a node consumes an asset input via the engine ---------------

ASSET = TypeExpr.concrete(ASSET_TYPE)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)


class AssetStat(Node):
    """Reads an asset's bytes: proof node code sees an AssetRef and needs
    neither paths nor any idea of where content lives."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.asset.stat",
            display_name="Asset Stat",
            category="test",
            inputs=(InputSpec("asset", ASSET),),
            outputs=(OutputSpec("bytes", INT), OutputSpec("name", STRING)),
        )

    @classmethod
    def execute(cls, *, asset: AssetRef) -> Mapping[str, object]:
        return cls.outputs(bytes=len(asset.read_bytes()), name=asset.name)


def test_engine_run_with_asset_literal(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = make_models_dir(tmp_path)
        library = LocalAssetLibrary(root)
        library.scan()
        registry = asset_registry(library)

        nodes: list[type[Node]] = [AssetStat]
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        # The graph literal is wire-shaped identity - what a frontend submits
        # after picking from the catalog. Never a path.
        literal = library.ref("models/checkpoints/tiny.safetensors").to_wire()
        graph = Graph(nodes={"s": GraphNode("test.asset.stat", {"asset": literal})})
        result = await engine.run(graph, ["s"])
        assert result.outputs["s"]["bytes"].resolve() == len(b"tiny checkpoint bytes")
        assert result.outputs["s"]["name"].resolve() == "tiny.safetensors"

        # Same digest under a different name: same cache key, no re-execution.
        (root / "checkpoints" / "alias.safetensors").write_bytes(b"tiny checkpoint bytes")
        library.scan()
        alias = library.ref("models/checkpoints/alias.safetensors").to_wire()
        again = await engine.run(
            Graph(nodes={"s": GraphNode("test.asset.stat", {"asset": alias})}), ["s"]
        )
        assert again.executed == ()

    run_scenario(scenario())


def test_engine_cache_separates_asset_and_audio_record_type(tmp_path: Path) -> None:
    class Identity(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            generic = TypeExpr.variable("T")
            return NodeSchema(
                node_type="test.asset.identity",
                inputs=(InputSpec("asset", generic),),
                outputs=(OutputSpec("asset", generic),),
            )

        @classmethod
        def execute(cls, *, asset: AssetRef) -> Mapping[str, object]:
            return cls.outputs(asset=asset)

    async def scenario() -> None:
        root = make_models_dir(tmp_path)
        library = LocalAssetLibrary(root)
        library.scan()
        registry = asset_registry(library)
        nodes: list[type[Node]] = [Identity]
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        literal = library.ref("models/checkpoints/tiny.safetensors").to_wire()

        def graph(type_id: str) -> Graph:
            return Graph(
                nodes={
                    "s": GraphNode("test.asset.identity", {"asset": TypedLiteral(type_id, literal)})
                }
            )

        asset_result = await engine.run(graph(ASSET_TYPE), ["s"])
        record_result = await engine.run(graph(AUDIO_RECORD_TYPE), ["s"])
        assert asset_result.executed == ("s",)
        assert record_result.executed == ("s",)
        assert asset_result.outputs["s"]["asset"].type_id == ASSET_TYPE
        assert record_result.outputs["s"]["asset"].type_id == AUDIO_RECORD_TYPE
        assert (
            asset_result.outputs["s"]["asset"].fingerprint
            == record_result.outputs["s"]["asset"].fingerprint
        )
        record = record_result.outputs["s"]["asset"].resolve()
        assert isinstance(record, AssetRef)
        assert record.read_bytes() == b"tiny checkpoint bytes"
        again = await engine.run(graph(ASSET_TYPE), ["s"])
        assert again.cached == ("s",)
        assert again.outputs["s"]["asset"].type_id == ASSET_TYPE

    run_scenario(scenario())


@pytest.mark.parametrize("format_id", ["gltf", "obj"])
@pytest.mark.parametrize(
    "case", ["valid", "empty-format", "missing", "extra", "malformed", "wrong-format"]
)
def test_file3d_canonical_submission_checks_declared_resources(
    tmp_path: Path, format_id: str, case: str
) -> None:
    from dinkster_assets import load_file3d, register_file3d_types
    from dinkster_server.library import ServerLibrary
    from dinkster_server.preflight import bind_model_names
    from dinkster_values import FILE_3D_TYPE

    root = tmp_path / "models"
    root.mkdir()
    if format_id == "gltf":
        primary = b'{"buffers":[{"uri":"mesh.bin"}]}'
        resources = {"mesh.bin": b"mesh bytes"}
    else:
        primary = b"mtllib surface.mtl\nv 1 3 7\n"
        resources = {"surface.mtl": b"newmtl surface\nKd 0.1 0.3 0.7\n"}
    (root / f"scene.{format_id}").write_bytes(primary)
    for name, data in resources.items():
        (root / name).write_bytes(data)
    (root / "extra.bin").write_bytes(b"granted but undeclared")
    table = MountTable(tmp_path / "mounts.json")
    table.add(MountDef("models", root, kind="media/model3d"))
    source = table.bind_name(f"scene.{format_id}", "media/model3d")
    assert isinstance(source, LocalAssetRef)
    packet = load_file3d(source)
    assert packet is not None
    registry = TypeRegistry()
    register_file3d_types(registry, table)
    wire = json.loads(registry.spec(FILE_3D_TYPE).encode(packet))
    if case == "missing":
        wire["resources"] = {}
    elif case == "extra":
        extra = table.bind_name("extra.bin", "media/model3d")
        assert isinstance(extra, LocalAssetRef)
        wire["resources"]["extra.bin"] = extra.to_wire()
    elif case == "malformed":
        wire["source"] = "scene.gltf"
    elif case == "wrong-format":
        wire["format"] = "ply"
    elif case == "empty-format":
        wire["format"] = ""
    literal = TypedLiteral(FILE_3D_TYPE, wire)
    graph = Graph({"preview": GraphNode("preview-file", {"model": literal})})
    schema = NodeSchema(
        "preview-file", inputs=(InputSpec("model", TypeExpr.union(CORE_STRING, FILE_3D_TYPE)),)
    )
    library = ServerLibrary(
        AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library.sqlite"), resolver=table
    )
    if case in {"valid", "empty-format"}:
        prepared, missing = bind_model_names(graph, {schema.node_type: schema}, library)
        assert not missing
        assert prepared.nodes["preview"].inputs["model"] is literal
        assert source.read_bytes() == primary
    else:
        error = {
            "missing": "requires an explicit asset",
            "extra": "undeclared",
            "malformed": "asset identities",
            "wrong-format": "format does not match",
        }
        with pytest.raises(AssetError, match=error[case]):
            bind_model_names(graph, {schema.node_type: schema}, library)
