"""Host-authoritative capture of untrusted output reports and native save claims."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
import zlib
from pathlib import Path

import pytest
from dinkster_api.v1 import attach_saved_artifact
from dinkster_assets import (
    AssetError,
    AssetWriter,
    LocalAssetLibrary,
    MountSnapshotWriter,
    SaveTarget,
    digest_bytes,
    load_write_records,
)
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import SavedArtifactCandidate
from dinkster_schema.reporting import capture_saved_artifacts
from dinkster_workers.saved_artifacts import (
    SavedArtifactAuthority,
)

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"")
        + _png_chunk(b"IEND", b"")
    )


def _authority(tmp_path: Path) -> tuple[SavedArtifactAuthority, Path]:
    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "comfy-output", "root": str(root), "mode": "readwrite"}]}),
        "utf-8",
    )
    return SavedArtifactAuthority(snapshot), root


def _candidate(
    filename: str = "result.png", subfolder: str = "video", folder_type: str = "output"
) -> SavedArtifactCandidate:
    return SavedArtifactCandidate("save", filename, subfolder, folder_type)


def test_host_classifies_hashes_and_indexes_valid_candidate(tmp_path: Path) -> None:
    authority, root = _authority(tmp_path)
    folder = root / "video"
    folder.mkdir()
    payload = _png()
    path = folder / "result.png"
    path.write_bytes(payload)
    artifact = authority.capture((_candidate(),), "save", 0)[0]
    assert artifact.digest == digest_bytes(payload)
    assert artifact.size == len(payload)
    assert artifact.media_type == "image/png"
    assert artifact.virtual_path == "mounts/comfy-output/video/result.png"
    assert load_write_records(root)["video/result.png"]["digest"] == artifact.digest


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        (_candidate("../result.png"), "safe basename"),
        (_candidate(subfolder="../outside"), "unsafe path"),
        (_candidate(folder_type="temp"), "type must be 'output'"),
        (SavedArtifactCandidate("other", "result.png", "video", "output"), "nodeId"),
    ],
)
def test_host_rejects_untrusted_candidate_fields(
    tmp_path: Path, candidate: SavedArtifactCandidate, match: str
) -> None:
    authority, _root = _authority(tmp_path)
    with pytest.raises(AssetError, match=match):
        authority.capture((candidate,), "save", 0)


def test_host_rejects_symlink_candidate(tmp_path: Path) -> None:
    authority, root = _authority(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png())
    symlink_or_skip(root / "result.png", outside)
    with pytest.raises(AssetError, match="opened safely|symlink|junction"):
        authority.capture((_candidate(subfolder=""),), "save", 0)


def test_host_rejects_outside_file_hard_link(tmp_path: Path) -> None:
    authority, root = _authority(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png())
    try:
        os.link(outside, root / "result.png")
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    with pytest.raises(AssetError, match="hard link"):
        authority.capture((_candidate(subfolder=""),), "save", 0)


def test_host_rejects_stale_candidate(tmp_path: Path) -> None:
    authority, root = _authority(tmp_path)
    path = root / "result.png"
    path.write_bytes(_png())
    old = time.time_ns() - 10_000_000_000
    os.utime(path, ns=(old, old))
    with pytest.raises(AssetError, match="stale"):
        authority.capture((_candidate(subfolder=""),), "save", time.time_ns())


def test_host_rejects_file_changed_during_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, root = _authority(tmp_path)
    path = root / "result.png"
    path.write_bytes(_png())
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(fd: int):  # noqa: ANN202
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(_png() + b"changed")
        return real_fstat(fd)

    monkeypatch.setattr(os, "fstat", changing_fstat)
    with pytest.raises(AssetError, match="changed while"):
        authority.capture((_candidate(subfolder=""),), "save", 0)


def test_native_save_attachment_requires_open_context_and_snapshots_claim() -> None:
    claim = {
        "digest": digest_bytes(b"x"),
        "name": "x.txt",
        "size": 1,
        "mediaType": "text/plain",
        "virtualPath": "mounts/custom/x.txt",
    }
    with pytest.raises(RuntimeError, match="active invocation"):
        attach_saved_artifact(claim)
    with capture_saved_artifacts() as claims:
        attach_saved_artifact(claim)
        claim["size"] = 999
    assert claims[0]["size"] == 1

    async def scenario() -> None:
        release = asyncio.Event()

        async def late() -> None:
            await release.wait()
            attach_saved_artifact({**claim, "size": 1})

        with capture_saved_artifacts():
            task = asyncio.create_task(late())
        release.set()
        with pytest.raises(RuntimeError, match="closed"):
            await task

    run_scenario(scenario())


@pytest.mark.parametrize(
    "fault",
    (
        "revoked",
        "readonly",
        "digest",
        "size",
        "large-size",
        "unpublished",
        "stale",
        "hardlink",
        "symlink",
    ),
)
def test_native_save_authority_rejects_invalid_or_revoked_publication(
    tmp_path: Path, fault: str
) -> None:
    root = tmp_path / "custom"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    row = {"id": "custom", "root": str(root), "mode": "readwrite"}
    snapshot.write_text(json.dumps({"mounts": [row]}), "ascii")
    ref = AssetWriter(MountSnapshotWriter(snapshot)).save_bytes(
        SaveTarget("custom", "nested/report"), b"valid text", suffix=".txt", media_type="text/plain"
    )
    claim = ref.to_wire()
    started = 0
    if fault == "revoked":
        snapshot.write_text('{"mounts": []}', "ascii")
    elif fault == "readonly":
        snapshot.write_text(json.dumps({"mounts": [{**row, "mode": "read"}]}), "ascii")
    elif fault == "digest":
        (root / "nested" / ref.name).write_bytes(b"evil! text")
    elif fault == "size":
        claim["size"] = ref.size + 1
    elif fault == "large-size":
        claim["size"] = 1073741825
    elif fault == "unpublished":
        (root / "unpublished.txt").write_bytes(b"valid text")
        claim.update(name="unpublished.txt", virtualPath="mounts/custom/unpublished.txt")
    elif fault == "stale":
        published = load_write_records(root)["nested/" + ref.name]["publishedAtNs"]
        assert type(published) is int
        started = published + 1
    elif fault == "hardlink":
        try:
            os.link(root / "nested" / ref.name, tmp_path / "second-link.txt")
        except OSError as exc:
            pytest.skip(f"hard links are unavailable: {exc}")
    elif fault == "symlink":
        original = root / "nested" / ref.name
        outside = tmp_path / "outside.txt"
        original.rename(outside)
        symlink_or_skip(original, outside)
    with pytest.raises(AssetError):
        SavedArtifactAuthority(snapshot).capture_mounted((claim,), "save", started)


@pytest.mark.parametrize("persist_index", (True, False))
def test_native_save_attachment_survives_scan_without_refreshing_publication(
    tmp_path: Path,
    persist_index: bool,
) -> None:
    root = tmp_path / "custom"
    root.mkdir()
    index = tmp_path / "host-index.json"
    if not persist_index:
        index.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "custom", "root": str(root), "mode": "readwrite", "index": str(index)}
                ]
            }
        ),
        "ascii",
    )
    ref = AssetWriter(MountSnapshotWriter(snapshot)).save_bytes(
        SaveTarget("custom", "report"), b"scan-safe text", suffix=".txt", media_type="text/plain"
    )
    publication = load_write_records(root)[ref.name]["publishedAtNs"]
    assert type(publication) is int
    library = LocalAssetLibrary(root, index_path=index)
    authority = SavedArtifactAuthority(snapshot)
    for _ in range(2):
        library.scan()
        if persist_index:
            assert load_write_records(root) == {}
            assert json.loads(index.read_text("ascii"))[ref.name]["publishedAtNs"] == publication
        else:
            assert load_write_records(root)[ref.name]["publishedAtNs"] == publication
        (artifact,) = authority.capture_mounted((ref.to_wire(),), "save", publication)
        assert artifact.digest == digest_bytes(b"scan-safe text")
        assert artifact.virtual_path == f"mounts/custom/{ref.name}"
        with pytest.raises(AssetError, match="fresh writer publication"):
            authority.capture_mounted((ref.to_wire(),), "save", publication + 1)
    (root / ref.name).write_bytes(b"new scan bytes")
    library.scan()
    with pytest.raises(AssetError):
        authority.capture_mounted((ref.to_wire(),), "save", publication)


def test_native_save_published_after_scan_snapshot_keeps_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import library as library_module

    root = tmp_path / "custom"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "custom", "root": str(root), "mode": "readwrite"}]}), "ascii"
    )
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    real_load = library_module.load_write_records
    published: list[dict[str, object]] = []
    publication: list[int] = []

    def load_then_publish(path: Path) -> dict[str, dict[str, object]]:
        records = real_load(path)
        if not published:
            ref = writer.save_bytes(
                SaveTarget("custom", "during-scan"),
                b"racing save",
                suffix=".txt",
                media_type="text/plain",
            )
            published.append(ref.to_wire())
            timestamp = real_load(root)[ref.name]["publishedAtNs"]
            assert type(timestamp) is int
            publication.append(timestamp)
        return records

    monkeypatch.setattr(library_module, "load_write_records", load_then_publish)
    library = LocalAssetLibrary(root)
    library.scan()
    authority = SavedArtifactAuthority(snapshot)
    (artifact,) = authority.capture_mounted(published, "save", publication[0])
    assert artifact.digest == digest_bytes(b"racing save")
    assert artifact.virtual_path == "mounts/custom/during-scan_00001.txt"
    library.scan()
    assert authority.capture_mounted(published, "save", publication[0]) == (artifact,)
    with pytest.raises(AssetError, match="fresh writer publication"):
        authority.capture_mounted(published, "save", publication[0] + 1)


@pytest.mark.parametrize("in_process", (True, False))
def test_native_mounted_save_results_use_host_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    root = tmp_path / "custom"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "custom", "root": str(root), "mode": "readwrite"}]}), "ascii"
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    (tmp_path / "native_save_fixture.py").write_text(
        r"""
import os
from dinkster_api.v1 import (CORE_STRING, InputSpec, Node, NodeSchema, OutputSpec,
    TypeExpr, AssetWriter, MountSnapshotWriter, SaveTarget, attach_saved_artifact,
    ASSET_TYPE, register_asset_type)
class Save(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(node_type='native-save-fixture.save',
            inputs=(InputSpec('mode', TypeExpr.concrete(CORE_STRING)),),
            outputs=(OutputSpec('out', TypeExpr.concrete(CORE_STRING)),
                     OutputSpec('asset', TypeExpr.concrete(ASSET_TYPE))),
            idempotent=False)
    @classmethod
    def execute(cls, *, mode):
        authority = MountSnapshotWriter(os.environ['DINKSTER_MOUNTS_SNAPSHOT'])
        data = b'' if mode == 'empty' else b'asymmetric text 1,2,7\n'
        if mode == 'batch':
            for index in range(65):
                ref = AssetWriter(authority).save_bytes(
                    SaveTarget('custom', f'batch/frame-{index}'), str(index).encode('ascii'),
                    suffix='.txt', media_type='text/plain')
                attach_saved_artifact(ref.to_wire())
            return cls.outputs(out=ref.name, asset=ref)
        ref = AssetWriter(authority).save_bytes(SaveTarget('custom', 'nested/report'),
            data, suffix='.txt', media_type='text/plain')
        if mode != 'unattached':
            attach_saved_artifact(ref.to_wire())
        if mode == 'failed':
            raise RuntimeError('save fixture failed after attachment')
        if mode == 'tampered':
            path = authority.writable_root('custom') / ref.virtual_path.split('/', 2)[2]
            path.write_bytes(b'changed bytes')
        if mode == 'bad-output':
            return cls.outputs(unknown='bad')
        return cls.outputs(out=ref.name, asset=ref)
NODES = (Save,)
def register_types(registry):
    if ASSET_TYPE not in registry:
        register_asset_type(registry)
""",
        "ascii",
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        """[pack]
name = "native-save-fixture"
publisher = "dinkster"
namespaces = ["native-save-fixture"]
[pack.entry]
nodes = "native_save_fixture:NODES"
types = "native_save_fixture:register_types"
""",
        "ascii",
    )
    env = {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path.cwd()))),
        "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
    }

    async def scenario() -> None:
        composer = ServingComposer(worker_env=env)
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, env=env))
            engine = composer.composition.make_engine(lambda _event: None)
            for mode in ("text", "empty", "unattached", "batch"):
                result = await engine.run(
                    Graph(nodes={"save": GraphNode("native-save-fixture.save", {"mode": mode})}),
                    ["save"],
                )
                if mode == "unattached":
                    assert result.artifacts == ()
                    continue
                if mode == "batch":
                    assert len(result.artifacts) == 65
                    assert len({artifact.virtual_path for artifact in result.artifacts}) == 65
                    for index, artifact in enumerate(result.artifacts):
                        data = str(index).encode("ascii")
                        assert artifact.node_id == "save"
                        assert artifact.digest == digest_bytes(data)
                        assert artifact.size == len(data)
                        assert artifact.media_type == "text/plain"
                        assert artifact.name.startswith(f"frame-{index}_")
                        assert artifact.virtual_path == f"mounts/custom/batch/{artifact.name}"
                        assert (root / "batch" / artifact.name).read_bytes() == data
                    continue
                (artifact,) = result.artifacts
                data = b"" if mode == "empty" else b"asymmetric text 1,2,7\n"
                assert artifact.node_id == "save"
                assert artifact.digest == digest_bytes(data)
                assert artifact.size == len(data)
                assert artifact.media_type == "text/plain"
                assert artifact.virtual_path == f"mounts/custom/nested/{artifact.name}"
                assert (root / "nested" / artifact.name).read_bytes() == data
            for mode in ("failed", "tampered", "bad-output"):
                with pytest.raises(ExecutionError):
                    await engine.run(
                        Graph(
                            nodes={"save": GraphNode("native-save-fixture.save", {"mode": mode})}
                        ),
                        ["save"],
                    )
        finally:
            await composer.close()

    run_scenario(scenario())
