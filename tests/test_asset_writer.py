"""Structured save targets + the mount-scoped AssetWriter.

The write half of filesystem mounts: a save destination is a
``dinkster.save_target`` ({mount, prefix}) - never a raw host path - and the
one gate to disk is AssetWriter, which checks the mount is currently
granted readwrite, lands bytes atomically under collision-free counter
names, and records every write in the sidecar so the digest resolves
immediately, before any rescan.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dinkster_assets import (
    WRITES_SIDECAR_NAME,
    AssetError,
    AssetWriter,
    IndexedAssetResolver,
    LocalAssetLibrary,
    MountDef,
    MountSnapshotWriter,
    MountTable,
    SaveTarget,
    digest_bytes,
    load_write_records,
    register_save_target_type,
)
from dinkster_values import TypeRegistry

from tests.platform_support import symlink_or_skip

# ---------------------------------------------------------------------------
# SaveTarget: the value type


def test_save_target_valid_shapes() -> None:
    flat = SaveTarget(mount="out", prefix="ComfyUI")
    assert flat.subfolder == ""
    assert flat.stem == "ComfyUI"
    nested = SaveTarget(mount="comfy-output", prefix="renders/scene/final")
    assert nested.subfolder == "renders/scene"
    assert nested.stem == "final"
    assert nested.to_wire() == {"mount": "comfy-output", "prefix": "renders/scene/final"}
    assert SaveTarget.from_wire(nested.to_wire()) == nested


@pytest.mark.parametrize(
    ("mount", "prefix"),
    [
        ("out", "/etc/passwd"),  # absolute
        ("out", "a/../b"),  # traversal
        ("out", "./a"),  # dot segment
        ("out", "a//b"),  # empty segment
        ("out", "a/"),  # empty stem
        ("out", ""),  # nothing
        ("out", "a\\b"),  # backslash separator
        ("OUT", "a"),  # mount grammar: no uppercase
        ("-out", "a"),  # mount grammar: leading hyphen
        ("", "a"),  # mount grammar: empty
    ],
)
def test_save_target_rejects_unsafe(mount: str, prefix: str) -> None:
    with pytest.raises(AssetError):
        SaveTarget(mount=mount, prefix=prefix)


def test_save_target_wire_is_strict() -> None:
    with pytest.raises(AssetError):
        SaveTarget.from_wire({"mount": "out"})  # missing prefix
    with pytest.raises(AssetError):
        SaveTarget.from_wire({"mount": "out", "prefix": "a", "path": "/x"})  # unknown key
    with pytest.raises(AssetError):
        SaveTarget.from_wire({"mount": "out", "prefix": 3})  # wrong type


def test_save_target_type_registration_round_trip() -> None:
    registry = TypeRegistry()
    register_save_target_type(registry)
    spec = registry.spec("dinkster.save_target")
    target = SaveTarget(mount="out", prefix="renders/scene")
    encoded = spec.encode(target)
    assert spec.decode(encoded) == target
    # Graph literals arrive as plain mappings; coercion applies one rule,
    # so the mapping and the dataclass are the same value (same
    # fingerprint, same bytes).
    as_wire = registry.wrap("dinkster.save_target", {"mount": "out", "prefix": "renders/scene"})
    as_object = registry.wrap("dinkster.save_target", target)
    assert as_wire.fingerprint == as_object.fingerprint
    assert spec.encode({"mount": "out", "prefix": "renders/scene"}) == encoded
    with pytest.raises(AssetError):
        spec.encode("out/renders/scene")  # raw strings never name destinations


# ---------------------------------------------------------------------------
# Write authority: who may write where, live


def one_mount_snapshot(tmp_path: Path, root: Path, mode: str = "readwrite") -> Path:
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "out", "root": str(root), "mode": mode}]}),
        "utf-8",
    )
    return snapshot


def test_snapshot_writer_refusals(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    # No snapshot file at all: no write grants exist.
    with pytest.raises(AssetError, match="no readable mounts snapshot"):
        MountSnapshotWriter(tmp_path / "missing.json").writable_root("out")
    # Malformed snapshot.
    bad = tmp_path / "bad.json"
    bad.write_text('{"mounts": "nope"}', "utf-8")
    with pytest.raises(AssetError, match="malformed"):
        MountSnapshotWriter(bad).writable_root("out")
    # Unknown mount id.
    snapshot = one_mount_snapshot(tmp_path, root)
    with pytest.raises(AssetError, match="no ready mount"):
        MountSnapshotWriter(snapshot).writable_root("other")
    # Read-only grant refuses writes by name.
    readonly = one_mount_snapshot(tmp_path, root, mode="read")
    with pytest.raises(AssetError, match="read-only"):
        MountSnapshotWriter(readonly).writable_root("out")
    # The happy path names the root.
    snapshot = one_mount_snapshot(tmp_path, root)
    assert MountSnapshotWriter(snapshot).writable_root("out") == root


def test_snapshot_writer_sees_live_grant_and_revoke(tmp_path: Path) -> None:
    """The snapshot is re-read per call: a revoke lands on the NEXT save
    with no restart - the runtime-mounts contract, held for writes."""
    root = tmp_path / "root"
    root.mkdir()
    snapshot = one_mount_snapshot(tmp_path, root)
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    ref = writer.save_bytes({"mount": "out", "prefix": "a"}, b"first", suffix=".bin")
    assert (root / "a_00001.bin").read_bytes() == b"first"
    assert ref.digest == digest_bytes(b"first")
    # Revoke: rewrite the snapshot without the mount.
    snapshot.write_text(json.dumps({"mounts": []}), "utf-8")
    with pytest.raises(AssetError, match="no ready mount"):
        writer.save_bytes({"mount": "out", "prefix": "a"}, b"second", suffix=".bin")
    # Re-grant: the next save works again, counter continues.
    one_mount_snapshot(tmp_path, root)
    again = writer.save_bytes({"mount": "out", "prefix": "a"}, b"third", suffix=".bin")
    assert again.name == "a_00002.bin"


def test_mount_table_writable_root(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    ready.mkdir()
    table = MountTable()
    table.add(MountDef(id="rw", path=ready, mode="readwrite"))
    table.add(MountDef(id="ro", path=ready, mode="read"))
    table.scan("rw")
    table.scan("ro")
    assert table.writable_root("rw") == ready
    with pytest.raises(AssetError, match="read-only"):
        table.writable_root("ro")
    with pytest.raises(AssetError, match="never granted or was revoked"):
        table.writable_root("nope")
    # A mount whose directory is gone is failed, not writable.
    gone = tmp_path / "gone"
    gone.mkdir()
    table.add(MountDef(id="lost", path=gone, mode="readwrite"))
    gone.rmdir()
    with pytest.raises(AssetError):
        table.scan("lost")  # scan records the failure and re-raises
    with pytest.raises(AssetError, match="not ready"):
        table.writable_root("lost")


# ---------------------------------------------------------------------------
# AssetWriter: landing bytes


def make_writer(tmp_path: Path) -> tuple[AssetWriter, Path]:
    root = tmp_path / "root"
    root.mkdir()
    snapshot = one_mount_snapshot(tmp_path, root)
    return AssetWriter(MountSnapshotWriter(snapshot)), root


def test_save_bytes_lands_counter_named_files(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    first = writer.save_bytes(
        {"mount": "out", "prefix": "renders/scene"},
        b"one",
        suffix=".png",
        media_type="image/png",
    )
    second = writer.save_bytes(
        {"mount": "out", "prefix": "renders/scene"},
        b"two",
        suffix=".png",
        media_type="image/png",
    )
    assert (root / "renders" / "scene_00001.png").read_bytes() == b"one"
    assert (root / "renders" / "scene_00002.png").read_bytes() == b"two"
    assert first.virtual_path == "mounts/out/renders/scene_00001.png"
    assert second.virtual_path == "mounts/out/renders/scene_00002.png"
    assert first.digest == digest_bytes(b"one")
    assert first.size == 3
    assert first.media_type == "image/png"
    assert first.name == "scene_00001.png"


def test_save_bytes_never_overwrites(tmp_path: Path) -> None:
    """A pre-existing counter-named file (from a previous session, another
    process, or the user) is skipped, never clobbered."""
    writer, root = make_writer(tmp_path)
    (root / "a_00001.bin").write_bytes(b"precious")
    ref = writer.save_bytes({"mount": "out", "prefix": "a"}, b"new", suffix=".bin")
    assert (root / "a_00001.bin").read_bytes() == b"precious"
    assert ref.name == "a_00002.bin"


def test_save_bytes_rejects_bad_suffix(tmp_path: Path) -> None:
    writer, _ = make_writer(tmp_path)
    for suffix in ("png", "", ".p/ng", "..", ".png/../x"):
        with pytest.raises(AssetError, match="suffix"):
            writer.save_bytes({"mount": "out", "prefix": "a"}, b"x", suffix=suffix)


def test_save_bytes_rejects_symlink_escape(tmp_path: Path) -> None:
    """The prefix grammar cannot express traversal; the remaining escape -
    a symlinked folder inside the mount pointing outside - is caught by
    resolved containment."""
    writer, root = make_writer(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(root / "leak", outside, target_is_directory=True)
    with pytest.raises(AssetError, match="escapes its mount"):
        writer.save_bytes({"mount": "out", "prefix": "leak/file"}, b"x", suffix=".bin")
    assert list(outside.iterdir()) == []


def test_save_bytes_no_temp_litter(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    writer.save_bytes({"mount": "out", "prefix": "a"}, b"x", suffix=".bin")
    leftovers = [p.name for p in root.rglob(".dinkster-save-*")]
    assert leftovers == []


def test_save_stream_rewinds_and_publishes_exact_bounded_bytes(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    source = io.BytesIO(b"streamed")
    source.seek(4)
    ref = writer.save_stream(
        {"mount": "out", "prefix": "latent"},
        source,
        suffix=".latent",
        media_type="application/x-comfy-latent",
        limit=8,
    )
    assert (root / ref.name).read_bytes() == b"streamed"
    assert ref.digest == digest_bytes(b"streamed")
    assert ref.size == 8


def test_save_stream_bound_failure_leaves_nothing(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    with pytest.raises(AssetError, match="exceeds limit"):
        writer.save_stream(
            {"mount": "out", "prefix": "latent"},
            io.BytesIO(b"too large"),
            suffix=".latent",
            limit=3,
        )
    assert list(root.iterdir()) == []


def test_concurrent_saves_get_distinct_names(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(
            pool.map(
                lambda i: writer.save_bytes(
                    {"mount": "out", "prefix": "burst"},
                    f"payload-{i}".encode(),
                    suffix=".bin",
                ),
                range(16),
            )
        )
    names = {ref.name for ref in refs}
    assert len(names) == 16  # no collisions, no overwrites
    for ref in refs:
        assert (root / ref.name).exists()


# ---------------------------------------------------------------------------
# Immediate resolvability: the writes sidecar


def test_written_digest_resolves_before_any_rescan(tmp_path: Path) -> None:
    """The whole point of the sidecar: a save's digest resolves through a
    snapshot-frozen index resolver immediately, no rescan required."""
    writer, root = make_writer(tmp_path)
    library = LocalAssetLibrary(root, namespace="mounts/out")
    library.scan()  # index exists, but predates the write
    resolver = IndexedAssetResolver(root)
    ref = writer.save_bytes({"mount": "out", "prefix": "fresh"}, b"hot", suffix=".bin")
    assert resolver.resolve(ref.digest) == root / "fresh_00001.bin"
    # The scanning library resolves it too, without rescanning.
    assert library.resolve(ref.digest) == root / "fresh_00001.bin"


def test_written_asset_is_listed_before_any_rescan(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    library = LocalAssetLibrary(root, namespace="mounts/out")
    library.scan()

    writer.save_bytes({"mount": "out", "prefix": "fresh"}, b"hot", suffix=".bin")

    assert [entry.virtual_path for entry in library.entries()] == ["mounts/out/fresh_00001.bin"]
    assert [entry.virtual_path for entry in library.list_folder("mounts/out").entries] == [
        "mounts/out/fresh_00001.bin"
    ]


def test_scan_absorbs_and_prunes_sidecar(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    library = LocalAssetLibrary(root, namespace="mounts/out")
    library.scan()
    ref = writer.save_bytes({"mount": "out", "prefix": "fresh"}, b"hot", suffix=".bin")
    assert (root / WRITES_SIDECAR_NAME).exists()
    library.scan()
    # Absorbed: resolvable from the index proper; sidecar row pruned.
    assert library.resolve(ref.digest) == root / "fresh_00001.bin"
    assert load_write_records(root) == {}
    index = json.loads((root / ".dinkster-asset-index.json").read_text("utf-8"))
    assert index["fresh_00001.bin"]["digest"] == ref.digest
    # And the sidecar/index themselves never enter the catalog.
    listed = [entry.virtual_path for entry in library.catalog.entries()]
    assert listed == ["mounts/out/fresh_00001.bin"]


def test_sidecar_tolerates_torn_lines(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    ref = writer.save_bytes({"mount": "out", "prefix": "a"}, b"good", suffix=".bin")
    sidecar = root / WRITES_SIDECAR_NAME
    with sidecar.open("a", encoding="utf-8") as handle:
        handle.write('{"path": "torn')  # crashed writer mid-line
    records = load_write_records(root)
    assert list(records) == ["a_00001.bin"]
    assert records["a_00001.bin"]["digest"] == ref.digest


def test_sidecar_hit_requires_unchanged_file(tmp_path: Path) -> None:
    """A file modified after its write record is treated as absent - the
    same size+mtime discipline index hits get."""
    writer, root = make_writer(tmp_path)
    ref = writer.save_bytes({"mount": "out", "prefix": "a"}, b"original", suffix=".bin")
    (root / "a_00001.bin").write_bytes(b"tampered!!")
    resolver_hit = _sidecar_resolve(root, ref.digest)
    assert resolver_hit is None


def _sidecar_resolve(root: Path, digest: str) -> Path | None:
    library = LocalAssetLibrary(root, namespace="mounts/out")
    return library.resolve(digest)
