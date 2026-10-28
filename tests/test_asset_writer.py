"""Structured save targets + the mount-scoped AssetWriter.

The write half of filesystem mounts: a save destination is a
``SAVE_TARGET`` ({mount, prefix}) - never a raw host path - and the
one gate to disk is AssetWriter, which checks the mount is currently
granted readwrite, lands bytes atomically under collision-free counter
names, and records every write in the sidecar so the digest resolves
immediately, before any rescan.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import BinaryIO, Protocol

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
from dinkster_protocol import AssetFolderContext
from dinkster_values import TypeRegistry
from dinkster_values.asset_folders import use_asset_folders

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
    spec = registry.spec("SAVE_TARGET")
    target = SaveTarget(mount="out", prefix="renders/scene")
    encoded = spec.encode(target)
    assert spec.decode(encoded) == target
    # Graph literals arrive as plain mappings; coercion applies one rule,
    # so the mapping and the dataclass are the same value (same
    # fingerprint, same bytes).
    as_wire = registry.wrap("SAVE_TARGET", {"mount": "out", "prefix": "renders/scene"})
    as_object = registry.wrap("SAVE_TARGET", target)
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


@pytest.mark.parametrize(
    ("name", "collision_name"),
    [
        ("hero image.png", "hero image_00001.png"),
        ("weights.safetensors", "weights_00001.safetensors"),
        ("README", "README_00001"),
        (".env", ".env_00001"),
    ],
)
def test_named_import_preserves_name_and_never_overwrites(
    tmp_path: Path, name: str, collision_name: str
) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    LocalAssetLibrary(root, namespace="mounts/out").scan()
    snapshot = one_mount_snapshot(tmp_path, root)
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    target = {"mount": "out", "prefix": f"project/{name}"}
    first = writer.save_named_stream(target, io.BytesIO(b"first bytes"), limit=11)
    assert first.name == name
    assert first.virtual_path == f"mounts/out/project/{name}"
    assert (root / "project" / name).read_bytes() == b"first bytes"
    second = writer.save_named_stream(target, io.BytesIO(b"other"), limit=5)
    assert second.name == collision_name
    assert (root / "project" / second.name).read_bytes() == b"other"
    assert (root / "project" / first.name).read_bytes() == b"first bytes"
    resolver = IndexedAssetResolver(root)
    assert resolver.resolve(first.digest) == root / "project" / name
    assert resolver.resolve(second.digest) == root / "project" / second.name
    assert (first.digest, second.digest) == (digest_bytes(b"first bytes"), digest_bytes(b"other"))
    assert not list(root.rglob(".dinkster-save-*"))


def test_named_import_concurrent_collisions_keep_every_payload(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    snapshot = one_mount_snapshot(tmp_path, root)
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    payloads = [f"input {index}".encode() for index in range(7)]

    def upload(payload: bytes):
        return writer.save_named_stream(
            {"mount": "out", "prefix": "original.png"},
            io.BytesIO(payload),
            limit=len(payload),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        refs = list(executor.map(upload, payloads))
    assert {ref.name for ref in refs} == {"original.png"} | {
        f"original_{index:05d}.png" for index in range(1, 7)
    }
    for ref, payload in zip(refs, payloads, strict=True):
        assert (root / ref.name).read_bytes() == payload
        assert ref.digest == digest_bytes(payload)


def test_named_import_refuses_failed_or_unsafe_destinations(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    snapshot = one_mount_snapshot(tmp_path, root)
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(root / "escape", outside, target_is_directory=True)
    target = {"mount": "out", "prefix": "escape/hero.png"}
    with pytest.raises(AssetError, match="escapes"):
        writer.save_named_stream(target, io.BytesIO(b"unchanged"), limit=9)
    assert list(outside.iterdir()) == []
    target["prefix"] = "hero.png"
    with pytest.raises(AssetError, match="exceeds limit"):
        writer.save_named_stream(target, io.BytesIO(b"too big"), limit=6)
    assert not (root / "hero.png").exists()
    assert not list(root.rglob(".dinkster-save-*"))
    one_mount_snapshot(tmp_path, root, mode="read")
    with pytest.raises(AssetError, match="read-only"):
        writer.save_named_stream(target, io.BytesIO(b"read only"), limit=9)
    one_mount_snapshot(tmp_path, root)
    detached = tmp_path / "detached"
    root.rename(detached)
    with pytest.raises(AssetError, match="unavailable"):
        writer.save_named_stream(target, io.BytesIO(b"offline"), limit=7)
    assert not root.exists()
    detached.rename(root)
    snapshot.write_text('{"mounts": []}', "utf-8")
    with pytest.raises(AssetError, match="no ready mount"):
        writer.save_named_stream(target, io.BytesIO(b"revoked"), limit=7)
    assert not (root / "hero.png").exists()


def test_named_import_rechecks_grant_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    snapshot = one_mount_snapshot(tmp_path, root)
    writer = AssetWriter(MountSnapshotWriter(snapshot))

    class RevokingStream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            data = super().read(size)
            if not data:
                snapshot.write_text('{"mounts": []}', "utf-8")
            return data

    with pytest.raises(AssetError, match="no ready mount"):
        writer.save_named_stream(
            {"mount": "out", "prefix": "hero.png"}, RevokingStream(b"received"), limit=8
        )
    assert not (root / "hero.png").exists()
    assert not list(root.glob(".dinkster-save-*"))


@pytest.mark.parametrize(
    "name",
    [
        "../hero.png",
        "/hero.png",
        "x\\hero.png",
        "CON.png",
        "a/NUL",
        "a/hero.",
        "a ",
        "a:b.png",
        ".dinkster-asset-index.json",
        "sub/.dinkster-asset-writer.lock",
    ],
)
def test_named_import_rejects_nonportable_paths(tmp_path: Path, name: str) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    writer = AssetWriter(MountSnapshotWriter(one_mount_snapshot(tmp_path, root)))
    with pytest.raises(AssetError):
        writer.save_named_stream({"mount": "out", "prefix": name}, io.BytesIO(b"x"), limit=1)
    assert list(root.iterdir()) == []


def test_project_output_overrides_default_without_granting_writes(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    table = MountTable(snapshot)
    for name in ("global", "project"):
        root = tmp_path / name
        root.mkdir()
        table.add(MountDef(name, root, mode="readwrite"))
        table.scan(name)
    table.select_output_mount("global")
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    context = AssetFolderContext(("project",), "project")
    with use_asset_folders(context):
        ref = writer.save_bytes(writer.output_target("render"), b"project result", suffix=".bin")
        assert ref.virtual_path == "mounts/project/render_00001.bin"
        assert (tmp_path / "project" / ref.name).read_bytes() == b"project result"
        assert writer.output_target("render", "global")["mount"] == "global"
        detached = tmp_path / "detached-project"
        (tmp_path / "project").rename(detached)
        with pytest.raises(AssetError, match="unavailable"):
            writer.save_bytes(
                writer.output_target("nested/frames/render"), b"offline", suffix=".bin"
            )
        assert not (tmp_path / "project").exists()
        assert not (detached / "nested").exists()
        detached.rename(tmp_path / "project")
        restored = writer.save_bytes(
            writer.output_target("nested/frames/render"), b"restored", suffix=".bin"
        )
        assert (
            tmp_path / "project" / "nested" / "frames" / restored.name
        ).read_bytes() == b"restored"
        table.remove("project")
        with pytest.raises(AssetError, match="no ready mount"):
            writer.save_bytes(writer.output_target("render"), b"revoked", suffix=".bin")
    with use_asset_folders(AssetFolderContext(("global",))):
        with pytest.raises(AssetError, match="for this project"):
            writer.output_target("render")
    assert writer.output_target("render")["mount"] == "global"
    assert list((tmp_path / "global").glob("render*")) == []


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


class StringQueue(Protocol):
    def put(self, item: str) -> None: ...


def _save_batch_process(snapshot: str, queue: StringQueue) -> None:
    ref = AssetWriter(MountSnapshotWriter(snapshot)).save_stream_batch(
        {"mount": "out", "prefix": "burst-%batch_num%"},
        [io.BytesIO(b"process")],
        suffix="_.bin",
        limit=7,
    )
    queue.put(ref[0].name)


def _save_encoded_process(snapshot: str, queue: StringQueue) -> None:
    def encoder(stem: str, counter: int) -> BinaryIO:
        return io.BytesIO(f"{stem} {counter}".encode())

    ref = AssetWriter(MountSnapshotWriter(snapshot)).save_encoded(
        {"mount": "out", "prefix": "checkpoint"},
        encoder,
        suffix=".safetensors",
        limit=32,
    )
    queue.put(ref.name)


def test_separate_prefix_and_mount_write_to_selected_grant(tmp_path: Path) -> None:
    roots = {name: tmp_path / name for name in ("default", "selected", "readonly")}
    for root in roots.values():
        root.mkdir()
    snapshot = tmp_path / "mounts.json"
    mounts = [
        {"id": name, "root": str(root), "mode": "read" if name == "readonly" else "readwrite"}
        for name, root in roots.items()
    ]
    snapshot.write_text(json.dumps({"mounts": mounts, "outputMount": "default"}), "utf-8")
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    default = writer.save_bytes(writer.output_target("renders/default"), b"default", suffix=".png")
    selected_target = writer.output_target("other/selected", mount="selected")
    selected = writer.save_bytes(selected_target, b"selected", suffix=".png")
    assert default.virtual_path == "mounts/default/renders/default_00001.png"
    assert selected.virtual_path == "mounts/selected/other/selected_00001.png"
    assert (roots["default"] / "renders/default_00001.png").read_bytes() == b"default"
    assert (roots["selected"] / "other/selected_00001.png").read_bytes() == b"selected"
    with pytest.raises(AssetError, match="read-only"):
        writer.save_bytes(writer.output_target("blocked", "readonly"), b"no", suffix=".png")
    snapshot.write_text(json.dumps({"mounts": mounts[:1], "outputMount": "default"}), "utf-8")
    with pytest.raises(AssetError, match="no ready mount"):
        writer.save_bytes(selected_target, b"revoked", suffix=".png")
    assert len(list(roots["selected"].rglob("*.png"))) == 1


@pytest.mark.parametrize(
    ("dimensions", "expected"),
    [(None, "0x0"), ((1, 1), "1x1"), ((640, 360), "640x360")],
)
def test_output_prefix_variables_are_expanded_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions: tuple[int, int] | None,
    expected: str,
) -> None:
    monkeypatch.setattr(
        time, "localtime", lambda: time.struct_time((2026, 2, 3, 4, 5, 6, 1, 34, -1))
    )
    writer, root = make_writer(tmp_path)
    prefix = "%year%/%month%%day%/%hour%%minute%%second%-%width%x%height%-%custom%"
    target = (
        writer.output_target(prefix, "out")
        if dimensions is None
        else writer.output_target(
            prefix, "out", image_width=dimensions[0], image_height=dimensions[1]
        )
    )
    ref = writer.save_bytes(target, b"dimensions\n", suffix=".txt")
    relative = f"2026/0203/040506-{expected}-%custom%_00001.txt"
    assert ref.virtual_path == f"mounts/out/{relative}"
    assert (root / relative).read_bytes() == b"dimensions\n"
    with pytest.raises(AssetError):
        writer.save_bytes(writer.output_target("../%width%/escape", "out"), b"no", suffix=".txt")


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


def test_save_bytes_rejects_symlinked_writer_lock(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"")
    lock_path = root / ".dinkster-asset-writer.lock"
    symlink_or_skip(lock_path, outside)
    with pytest.raises(AssetError, match="unsafe writer lock path"):
        writer.save_bytes({"mount": "out", "prefix": "probe"}, b"payload", suffix=".bin")
    assert outside.read_bytes() == b""
    assert lock_path.is_symlink()
    assert not (root / "probe_00001.bin").exists()


def test_save_bytes_rejects_hardlinked_writer_lock(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"")
    lock_path = root / ".dinkster-asset-writer.lock"
    os.link(outside, lock_path)
    with pytest.raises(AssetError, match="unsafe writer lock path"):
        writer.save_bytes({"mount": "out", "prefix": "probe"}, b"payload", suffix=".bin")
    assert outside.read_bytes() == b""
    assert not (root / "probe_00001.bin").exists()


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
    assert {path.name for path in root.iterdir()} == {".dinkster-asset-writer.lock"}


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


def test_save_stream_batch_uses_one_counter_for_expanded_stems(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    (root / "render-%batch_num%_00004_.webp").write_bytes(b"previous")
    (root / "render-1_00099_.png").write_bytes(b"other stem")
    LocalAssetLibrary(root, namespace="mounts/out").scan()

    refs = writer.save_stream_batch(
        {"mount": "out", "prefix": "render-%batch_num%"},
        (io.BytesIO(data) for data in (b"zero", b"one", b"two")),
        suffix="_.png",
        media_type="image/png",
        limit=4,
    )

    assert [ref.name for ref in refs] == [
        "render-0_00005_.png",
        "render-1_00006_.png",
        "render-2_00007_.png",
    ]
    assert [ref.virtual_path for ref in refs] == [f"mounts/out/{ref.name}" for ref in refs]
    resolver = IndexedAssetResolver(root)
    assert [resolver.resolve(ref.digest) for ref in refs] == [root / ref.name for ref in refs]
    assert (root / "render-1_00099_.png").read_bytes() == b"other stem"


def test_save_stream_batch_repeats_with_next_source_counter(tmp_path: Path) -> None:
    writer, _ = make_writer(tmp_path)
    first = writer.save_stream_batch(
        {"mount": "out", "prefix": "render-%batch_num%"},
        [io.BytesIO(b"first"), io.BytesIO(b"second")],
        suffix="_.png",
        limit=6,
    )
    second = writer.save_stream_batch(
        {"mount": "out", "prefix": "render-%batch_num%"},
        [io.BytesIO(b"third")],
        suffix="_.png",
        limit=5,
    )
    assert [ref.name for ref in first + second] == [
        "render-0_00001_.png",
        "render-1_00002_.png",
        "render-0_00002_.png",
    ]


def test_save_stream_batch_rejects_later_oversize_source_without_temp_files(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    with pytest.raises(AssetError, match="exceeds limit"):
        writer.save_stream_batch(
            {"mount": "out", "prefix": "render-%batch_num%"},
            [io.BytesIO(b"ok"), io.BytesIO(b"too large")],
            suffix="_.png",
            limit=2,
        )
    assert (root / "render-0_00001_.png").read_bytes() == b"ok"
    assert list(root.glob(".dinkster-save-*")) == []


def test_save_stream_batch_rechecks_revocation_and_containment(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(root / "leak", outside, target_is_directory=True)
    with pytest.raises(AssetError, match="escapes its mount"):
        writer.save_stream_batch(
            {"mount": "out", "prefix": "leak/render-%batch_num%"},
            [io.BytesIO(b"x")],
            suffix="_.png",
            limit=1,
        )
    assert list(outside.iterdir()) == []

    snapshot = tmp_path / "mounts.json"

    def revoked_sources() -> Iterator[BinaryIO]:
        yield io.BytesIO(b"first")
        snapshot.write_text(json.dumps({"mounts": []}), "utf-8")
        yield io.BytesIO(b"second")

    with pytest.raises(AssetError, match="no ready mount"):
        writer.save_stream_batch(
            {"mount": "out", "prefix": "render-%batch_num%"},
            revoked_sources(),
            suffix="_.png",
            limit=6,
        )
    assert (root / "render-0_00001_.png").read_bytes() == b"first"


def test_concurrent_process_batches_never_collide(tmp_path: Path) -> None:
    _, root = make_writer(tmp_path)
    snapshot = tmp_path / "mounts.json"
    context = get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_save_batch_process, args=(str(snapshot), queue)) for _ in range(4)
    ]
    for process in processes:
        process.start()
    names = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert len(set(names)) == 4
    assert all((root / name).read_bytes() == b"process" for name in names)


def test_save_encoded_reserves_source_title_before_encoding(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    (root / "checkpoint_00004.safetensors").write_bytes(b"existing")
    seen: list[tuple[str, int]] = []

    def encoder(stem: str, counter: int) -> BinaryIO:
        seen.append((stem, counter))
        return io.BytesIO(f"modelspec.title={stem} {counter}".encode())

    ref = writer.save_encoded(
        {"mount": "out", "prefix": "checkpoint"},
        encoder,
        suffix=".safetensors",
        limit=64,
    )
    assert seen == [("checkpoint", 5)]
    assert ref.name == "checkpoint_00005.safetensors"
    assert (root / ref.name).read_bytes() == b"modelspec.title=checkpoint 5"


def test_save_encoded_refuses_late_collision_without_renaming_metadata(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)

    def encoder(stem: str, counter: int) -> BinaryIO:
        (root / f"{stem}_{counter:05d}.safetensors").write_bytes(b"external")
        return io.BytesIO(b"encoded")

    with pytest.raises(AssetError, match="selected save name"):
        writer.save_encoded(
            {"mount": "out", "prefix": "checkpoint"},
            encoder,
            suffix=".safetensors",
            limit=16,
        )
    assert (root / "checkpoint_00001.safetensors").read_bytes() == b"external"
    assert list(root.glob(".dinkster-save-*")) == []


def test_save_encoded_encoder_failure_releases_its_claim(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)

    def failing_encoder(stem: str, counter: int) -> BinaryIO:
        del stem, counter
        raise RuntimeError("encoding failed")

    with pytest.raises(RuntimeError, match="encoding failed"):
        writer.save_encoded(
            {"mount": "out", "prefix": "checkpoint"},
            failing_encoder,
            suffix=".safetensors",
            limit=16,
        )
    ref = writer.save_bytes({"mount": "out", "prefix": "after-failure"}, b"ok", suffix=".bin")
    assert (root / ref.name).read_bytes() == b"ok"


def test_save_encoded_rechecks_revocation_after_encoder_returns(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    snapshot = tmp_path / "mounts.json"

    def encoder(stem: str, counter: int) -> BinaryIO:
        del stem, counter
        snapshot.write_text(json.dumps({"mounts": []}), "utf-8")
        return io.BytesIO(b"encoded")

    with pytest.raises(AssetError, match="no ready mount"):
        writer.save_encoded(
            {"mount": "out", "prefix": "checkpoint"},
            encoder,
            suffix=".safetensors",
            limit=16,
        )
    assert {path.name for path in root.iterdir()} == {".dinkster-asset-writer.lock"}


def test_save_encoded_rejects_mount_remap_after_encoder_returns(tmp_path: Path) -> None:
    writer, root = make_writer(tmp_path)
    remapped = tmp_path / "remapped"
    remapped.mkdir()
    snapshot = tmp_path / "mounts.json"

    def encoder(stem: str, counter: int) -> BinaryIO:
        del stem, counter
        snapshot.write_text(
            json.dumps({"mounts": [{"id": "out", "root": str(remapped), "mode": "readwrite"}]}),
            "utf-8",
        )
        return io.BytesIO(b"encoded")

    with pytest.raises(AssetError, match="changed its root"):
        writer.save_encoded(
            {"mount": "out", "prefix": "checkpoint"},
            encoder,
            suffix=".safetensors",
            limit=16,
        )
    assert {path.name for path in root.iterdir()} == {".dinkster-asset-writer.lock"}
    assert list(remapped.iterdir()) == []


def test_concurrent_process_encoders_receive_distinct_selected_counters(tmp_path: Path) -> None:
    _, root = make_writer(tmp_path)
    snapshot = tmp_path / "mounts.json"
    context = get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_save_encoded_process, args=(str(snapshot), queue)) for _ in range(4)
    ]
    for process in processes:
        process.start()
    names = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted(names) == [f"checkpoint_{counter:05d}.safetensors" for counter in range(1, 5)]
    assert {path.read_bytes() for path in root.iterdir() if path.suffix == ".safetensors"} == {
        f"checkpoint {counter}".encode() for counter in range(1, 5)
    }


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
