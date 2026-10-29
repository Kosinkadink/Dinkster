"""M3 asset core (DESIGN 3.12): identity is a content digest, references
are virtual paths with file-like query semantics, nodes receive AssetRefs
and never filesystem paths, and an asset's cache fingerprint IS its digest
(hazard H4: location-independent cache keys)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import pytest
from dinkster_assets import (
    ASSET_TYPE,
    AssetCatalog,
    AssetEntry,
    AssetError,
    AssetRef,
    ChainResolver,
    IndexedAssetResolver,
    LocalAssetLibrary,
    digest_bytes,
    digest_file,
    is_digest,
    register_asset_type,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import CORE_INT, CORE_STRING, TypeRegistry, register_core_types
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


def asset_registry(resolver: LocalAssetLibrary | None = None) -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry, resolver)
    return registry


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
    with pytest.raises(AssetError, match="never.*filesystem paths"):
        registry.wrap(ASSET_TYPE, "/etc/passwd")


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
