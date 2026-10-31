"""M3 asset core (DESIGN 3.12): identity is a content digest, references
are virtual paths with file-like query semantics, nodes receive AssetRefs
and never filesystem paths, and an asset's cache fingerprint IS its digest
(hazard H4: location-independent cache keys)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest
from dinkster_assets import (
    ASSET_TYPE,
    AssetCatalog,
    AssetEntry,
    AssetError,
    AssetRef,
    AssetResolver,
    AssetVault,
    ChainResolver,
    IndexedAssetResolver,
    LibraryStore,
    LocalAssetLibrary,
    LocalAssetRef,
    MountDef,
    MountSnapshotResolver,
    MountTable,
    digest_bytes,
    digest_file,
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

# --- identity ---------------------------------------------------------------


def test_digest_is_canonical_and_content_based(tmp_path: Path) -> None:
    payload = b"model weights, allegedly"
    file_a = tmp_path / "a.safetensors"
    file_a.write_bytes(payload)
    digest = digest_file(file_a)
    assert is_digest(digest)
    assert digest == digest_bytes(payload)

    # Renames, copies, different folders: same bytes, same identity.
    file_b = tmp_path / "renamed" / "b.ckpt"
    file_b.parent.mkdir()
    file_b.write_bytes(payload)
    assert digest_file(file_b) == digest

    assert digest_bytes(b"other bytes") != digest


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


def test_legacy_index_row_hashes_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()
    ref = library.ref("models/checkpoints/tiny.safetensors")
    index = json.loads(library.index_path.read_text("utf-8"))
    for row in index.values():
        row.pop("verification", None)
    library.index_path.write_text(json.dumps(index), "utf-8")

    import dinkster_assets.integrity as integrity_module

    real = integrity_module._hash_handle
    calls = [0]

    def counted(handle: BinaryIO) -> str:
        calls[0] += 1
        return real(handle)

    monkeypatch.setattr(integrity_module, "_hash_handle", counted)
    worker_ref = AssetRef.from_wire(ref.to_wire(), IndexedAssetResolver(root))
    assert worker_ref.read_bytes() == b"tiny checkpoint bytes"
    assert calls == [1]


def test_recorded_resolver_miss_does_not_repeat_legacy_resolution(tmp_path: Path) -> None:
    class RecordedMiss:
        def __init__(self) -> None:
            self.recorded_calls = 0
            self.legacy_calls = 0

        def resolve_asset(self, digest: str) -> None:
            del digest
            self.recorded_calls += 1
            return None

        def resolve(self, digest: str) -> Path | None:
            del digest
            self.legacy_calls += 1
            return tmp_path / "unexpected"

    digest = digest_bytes(b"missing")
    resolver = RecordedMiss()
    ref = AssetRef(digest, "missing", 7, resolver=resolver)
    with pytest.raises(AssetError, match="not materializable"):
        ref.open()
    assert (resolver.recorded_calls, resolver.legacy_calls) == (1, 0)

    chained = RecordedMiss()
    chain_ref = AssetRef(digest, "missing", 7, resolver=ChainResolver(chained))
    with pytest.raises(AssetError, match="not materializable"):
        chain_ref.open()
    assert (chained.recorded_calls, chained.legacy_calls) == (1, 0)


def test_indexed_resolver_requires_an_existing_index(tmp_path: Path) -> None:
    with pytest.raises(AssetError, match="no readable asset index"):
        IndexedAssetResolver(tmp_path)


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


def test_scan_keeps_existing_index_until_assets_are_published(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()
    existing_index = json.loads(library.index_path.read_text("utf-8"))
    indexes_seen: list[dict[str, object]] = []

    LocalAssetLibrary(root).scan(
        on_progress=lambda _progress: indexes_seen.append(
            json.loads(library.index_path.read_text("utf-8"))
        )
    )

    assert indexes_seen[0] == existing_index
    assert indexes_seen[-1] == existing_index


def test_library_index_writes_are_atomic_and_quiescent(tmp_path: Path) -> None:
    """The index is shared state between Dinkster instances pointing at one
    model folder: writes go through temp + os.replace (a reader never sees
    a torn file that would cost it a full rehash), and a scan that learned
    nothing new does not rewrite the file at all."""
    root = make_models_dir(tmp_path)
    LocalAssetLibrary(root).scan()
    index_path = root / ".dinkster-asset-index.json"
    before = index_path.stat().st_mtime_ns

    LocalAssetLibrary(root).scan()  # nothing changed: no rewrite
    assert index_path.stat().st_mtime_ns == before
    assert not list(root.glob(".dinkster-asset-index.json.tmp-*"))  # no temp litter

    # And the temp file is never cataloged even mid-write (dot-prefixed).
    library = LocalAssetLibrary(root)
    (root / ".dinkster-asset-index.json.tmp-99999").write_text("{", "utf-8")
    library.scan()
    assert all(".tmp-" not in entry.virtual_path for entry in library.catalog.entries())


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


def test_complete_asset_references_validate_declared_metadata() -> None:
    registry = asset_registry()
    registry.register(
        "example.invalid", meta=lambda _: {"asset_refs": None}, asset_references_complete=True
    )
    with pytest.raises(AssetError, match="asset_refs must be a list"):
        value_asset_references(
            {"output": registry.wrap("example.invalid", None)}, complete_registry=registry
        )
    ref = AssetRef(digest_bytes(b"external"), "external", 8)
    registry.register(
        "example.external",
        meta=lambda _: {"asset_refs": [ref.to_wire()]},
        asset_references_complete=True,
    )
    assert value_asset_references(
        {"output": registry.wrap("example.external", None)}, complete_registry=registry
    ) == {ref.digest: ref}


@pytest.mark.parametrize("verified", (False, True))
@pytest.mark.parametrize("preexisting", (False, True))
def test_comfy_manifest_registers_bound_assets_without_replacing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified: bool, preexisting: bool
) -> None:
    from dinkster_workers import load_manifest
    from dinkster_workers.manifest import resolve_entry

    root = tmp_path / "models"
    root.mkdir()
    payload = b"comfy model asset"
    (root / "weights.bin").write_bytes(payload)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    table.add(MountDef(id="models", path=root), kind="model/checkpoint")
    if verified:
        table.scan("models")
    ref = table.bind_name("weights.bin", "model/checkpoint")
    assert isinstance(ref, AssetRef if verified else LocalAssetRef)
    registry = TypeRegistry()
    register_core_types(registry)
    if preexisting:
        register_asset_type(registry, table)
        original = registry.spec(ASSET_TYPE)
        monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "absent.json"))
    else:
        original = None
        monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    manifest = load_manifest(
        Path(__file__).parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert manifest.types_entry is not None
    register = resolve_entry(manifest.types_entry)
    assert callable(register)
    register(registry)
    first = registry.spec(ASSET_TYPE)
    first_recording = registry.spec(AUDIO_RECORD_TYPE)
    register(registry)
    assert registry.spec(ASSET_TYPE) is first
    assert registry.spec(AUDIO_RECORD_TYPE) is first_recording
    if original is not None:
        assert first is original
    rebound = registry.wrap(ASSET_TYPE, ref.to_wire()).resolve()
    assert isinstance(rebound, AssetRef if verified else LocalAssetRef)
    assert rebound.local_path() == root / "weights.bin"
    assert rebound.read_bytes() == payload


def test_wrap_literal_mapping_becomes_resolver_bound_ref(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()
    registry = asset_registry(library)

    wire = library.ref("models/checkpoints/tiny.safetensors").to_wire()
    value = registry.wrap(ASSET_TYPE, json.loads(json.dumps(wire)))
    assert value.fingerprint == wire["digest"]  # cache identity IS content identity
    assert value.meta.get("virtualPath") == "models/checkpoints/tiny.safetensors"
    ref = value.resolve()
    assert isinstance(ref, AssetRef)
    assert ref.read_bytes() == b"tiny checkpoint bytes"


def test_wrap_rejects_paths() -> None:
    registry = asset_registry()
    for type_id in (ASSET_TYPE, AUDIO_RECORD_TYPE):
        with pytest.raises(AssetError, match="never.*filesystem paths"):
            registry.wrap(type_id, "/etc/passwd")


def test_audio_record_uses_asset_identity_with_its_own_type_stamp(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    library = LocalAssetLibrary(root)
    library.scan()
    registry = asset_registry(library)
    first = library.ref("models/loras/style.safetensors")
    second = AssetRef(first.digest, "renamed.wav", first.size, resolver=library)

    asset = registry.wrap(ASSET_TYPE, first.to_wire())
    recorded = registry.wrap(AUDIO_RECORD_TYPE, second.to_wire())

    assert asset.fingerprint == recorded.fingerprint == first.digest
    assert asset.type_id == ASSET_TYPE
    assert recorded.type_id == AUDIO_RECORD_TYPE
    record_ref = recorded.resolve()
    assert isinstance(record_ref, AssetRef)
    assert record_ref.read_bytes() == b"lora bytes"
    with pytest.raises(AssetError, match="digest"):
        registry.wrap(AUDIO_RECORD_TYPE, {"name": "recording.wav", "size": 1})


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


def test_audio_record_crossing_keeps_asset_cache_identity_separate_by_type(tmp_path: Path) -> None:
    root = make_models_dir(tmp_path)
    sender_library = LocalAssetLibrary(root)
    sender_library.scan()
    receiver_library = LocalAssetLibrary(root, index_path=tmp_path / "idx2.json")
    receiver_library.scan()
    sender_registry = asset_registry(sender_library)
    receiver_registry = asset_registry(receiver_library)
    value = sender_registry.wrap(
        AUDIO_RECORD_TYPE, sender_library.ref("models/loras/style.safetensors").to_wire()
    )

    blobs: list[bytes] = []
    wire, stat = ValueCodec(sender_registry).encode(value, blobs, [])
    crossed, _ = ValueCodec(receiver_registry).decode(wire, blobs, [])

    assert stat.size_bytes < 512
    assert crossed.type_id == AUDIO_RECORD_TYPE
    assert crossed.fingerprint == value.fingerprint
    record_ref = crossed.resolve()
    assert isinstance(record_ref, AssetRef)
    assert record_ref.read_bytes() == b"lora bytes"


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
            path.symlink_to(outside)
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


@pytest.mark.parametrize(
    ("platform", "changed_field", "allowed"),
    [
        ("nt", "st_ctime_ns", True),
        ("posix", "st_ctime_ns", False),
        ("nt", "st_size", False),
        ("nt", "st_mtime_ns", False),
        ("nt", "st_dev", False),
        ("nt", "st_ino", False),
    ],
)
def test_local_binding_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    changed_field: str,
    allowed: bool,
) -> None:
    import dinkster_assets.model as model

    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"unhashed model")
    table = MountTable()
    table.add(MountDef(id="models", path=tmp_path), kind="model/checkpoint")
    ref = table.bind_name(path.name, "model/checkpoint")
    assert isinstance(ref, LocalAssetRef)

    def changed_stat(fd: int) -> SimpleNamespace:
        actual = os.fstat(fd)
        values = {
            field: getattr(actual, field)
            for field in (
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_dev",
                "st_ino",
            )
        }
        values[changed_field] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(model, "os", SimpleNamespace(name=platform, fstat=changed_stat))
    if allowed:
        assert ref.read_bytes() == b"unhashed model"
    else:
        with pytest.raises(AssetError, match="changed while opening"):
            ref.read_bytes()


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())
