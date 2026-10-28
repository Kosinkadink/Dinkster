"""Host-authoritative capture of untrusted compatibility output reports."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
import zlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_assets import (
    ASSET_TYPE,
    AssetError,
    AssetRef,
    AssetWriter,
    MountSnapshotWriter,
    digest_bytes,
    load_write_records,
    register_asset_type,
)
from dinkster_compat_comfy.saved_results import capture_saved_results
from dinkster_protocol import Invocation, SavedArtifactCandidate
from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import ExecutionContext, InProcessWorker
from dinkster_workers.boundary import BoundaryError, decode_result_artifact_candidates
from dinkster_workers.execution import use_execution_context
from dinkster_workers.saved_artifacts import (
    MAX_ARTIFACT_FILENAME,
    MAX_SAVED_ARTIFACT_BYTES,
    MAX_SAVED_ARTIFACTS,
    SavedArtifactAuthority,
)

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


def test_worker_reports_only_raw_saved_result_candidates() -> None:
    candidates: list[SavedArtifactCandidate] = []
    context = ExecutionContext(arm=None, expected_execution_identity=None, node_id="save")
    context = replace(
        context,
        artifact_sink=lambda *fields: candidates.append(SavedArtifactCandidate(*fields)),
    )
    ui = {
        "videos": [
            {"filename": "result.webm", "subfolder": "video", "type": "output"},
            {"filename": "result.webm", "subfolder": "video", "type": "output"},
        ],
        "presentation": [object()],
    }
    with use_execution_context(context):
        capture_saved_results(ui)
    assert candidates == [SavedArtifactCandidate("save", "result.webm", "video", "output")]


def test_forged_authoritative_descriptor_cannot_cross_boundary() -> None:
    with pytest.raises(BoundaryError, match="forged authoritative"):
        decode_result_artifact_candidates(
            {
                "artifacts": [
                    {
                        "nodeId": "save",
                        "digest": "blake3:" + "0" * 64,
                        "name": "forged.png",
                        "size": 1,
                        "mediaType": "image/png",
                        "virtualPath": "mounts/comfy-output/forged.png",
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "header",
    [
        {"artifactCandidates": "not-a-list"},
        {"artifactCandidates": [{}]},
        {
            "artifactCandidates": [
                {
                    "nodeId": "save",
                    "filename": "x" * (MAX_ARTIFACT_FILENAME + 1),
                    "subfolder": "",
                    "type": "output",
                }
            ]
        },
        {
            "artifactCandidates": [
                {"nodeId": "save", "filename": "x.png", "subfolder": "", "type": "output"}
            ]
            * (MAX_SAVED_ARTIFACTS + 1)
        },
    ],
)
def test_candidate_wire_is_bounded(header: object) -> None:
    with pytest.raises(BoundaryError):
        decode_result_artifact_candidates(header)  # type: ignore[arg-type]


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


@pytest.mark.parametrize("change", ("bytes", "size", "path", "readonly", "revoked", "symlink"))
def test_native_saved_ref_requires_exact_bytes_and_current_mount(
    tmp_path: Path, change: str
) -> None:
    authority, root = _authority(tmp_path)
    snapshot = tmp_path / "mounts.json"
    ref = AssetWriter(MountSnapshotWriter(snapshot)).save_bytes(
        {"mount": "comfy-output", "prefix": "notes/result"},
        b"saved text",
        suffix=".txt",
        media_type="text/plain",
    )
    wire = ref.to_wire()
    path = root / "notes" / ref.name
    if change == "bytes":
        path.write_bytes(b"other text")
    elif change == "size":
        wire["size"] = ref.size + 1
    elif change == "path":
        wire["virtualPath"] = "mounts/comfy-output/elsewhere/" + ref.name
    elif change == "readonly":
        data = json.loads(snapshot.read_text("utf-8"))
        data["mounts"][0]["mode"] = "read"
        snapshot.write_text(json.dumps(data), "utf-8")
    elif change == "revoked":
        snapshot.write_text('{"mounts": []}', "utf-8")
    else:
        outside = tmp_path / "outside.txt"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        symlink_or_skip(path, outside)
    candidate = SavedArtifactCandidate("save", ref.name, "notes", "output", wire)
    with pytest.raises(AssetError):
        authority.capture((candidate,), "save", 0)


@pytest.mark.parametrize("behavior", ("forward", "fail", "revoke"))
def test_only_successful_authorized_writer_calls_report_saved_files(
    tmp_path: Path, behavior: str
) -> None:
    authority, _ = _authority(tmp_path)
    snapshot = tmp_path / "mounts.json"
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    existing = writer.save_bytes(
        {"mount": "comfy-output", "prefix": "existing"}, b"input", suffix=".txt"
    )

    class OutputNode(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.save",
                outputs=(OutputSpec("asset", TypeExpr.concrete(ASSET_TYPE)),),
                output_node=True,
            )

        @classmethod
        def execute(cls) -> dict[str, AssetRef]:
            if behavior == "forward":
                return {"asset": existing}
            ref = writer.save_bytes(
                {"mount": "comfy-output", "prefix": "written"}, b"new", suffix=".txt"
            )
            if behavior == "fail":
                raise ValueError("after save")
            snapshot.write_text('{"mounts": []}', "utf-8")
            return {"asset": ref}

    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry)
    worker = InProcessWorker(
        {"test.save": OutputNode}, registry, artifact_authority=authority.capture
    )
    result = asyncio.run(
        worker.invoke(Invocation("invocation", "save", "test.save", {}, OutputNode.schema()))
    )
    assert not result.artifacts
    assert not result.artifact_candidates
    if behavior == "forward":
        assert result.error is None
        assert result.outputs is not None
        assert result.outputs["asset"].resolve() == existing
    else:
        assert result.outputs is None
        assert result.error is not None
        assert ("after save" if behavior == "fail" else "no ready mount") in result.error.message


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


def test_host_rejects_oversize_before_classifying(tmp_path: Path) -> None:
    authority, root = _authority(tmp_path)
    path = root / "result.png"
    path.write_bytes(_png())
    with path.open("r+b") as handle:
        handle.truncate(MAX_SAVED_ARTIFACT_BYTES + 1)
    with pytest.raises(AssetError, match="1 GiB"):
        authority.capture((_candidate(subfolder=""),), "save", 0)


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


def test_host_rejects_candidate_path_replaced_before_indexing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import saved_artifacts

    authority, root = _authority(tmp_path)
    target = root / "video" / "result.png"
    target.parent.mkdir()
    target.write_bytes(_png())
    replacement = root / "replacement.png"
    replacement.write_bytes(_png() + b"replacement")
    real_open = saved_artifacts._open_candidate

    @contextmanager
    def replacing_open(root_path: Path, parts: tuple[str, ...], filename: str):  # noqa: ANN202
        with real_open(root_path, parts, filename) as (handle, _restat_path):
            yield handle, replacement.stat

    monkeypatch.setattr(saved_artifacts, "_open_candidate", replacing_open)
    with pytest.raises(AssetError) as raised:
        authority.capture((_candidate(),), "save", 0)
    assert str(raised.value) == "saved artifact path changed while it was being indexed"
    assert load_write_records(root) == {}


def _stat_override(original: os.stat_result, **overrides: object) -> SimpleNamespace:
    """Return a stat-like object with specified fields overridden, preserving all others."""
    names = (
        "st_mode",
        "st_ino",
        "st_dev",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_atime",
        "st_mtime",
        "st_ctime",
        "st_atime_ns",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return SimpleNamespace(**{name: overrides.get(name, getattr(original, name)) for name in names})


def test_path_stable_fields_contract() -> None:
    """Verify the platform-specific field comparison contract.

    The before/after handle comparison always checks all five identity
    fields.  The after-handle vs path-stat comparison checks all five on
    POSIX but excludes st_ctime_ns on Windows, where CPython maps
    st_ctime to CreationTime for stat(path) but ChangeTime for fstat(fd).
    """
    from dinkster_workers.saved_artifacts import _HANDLE_STABLE_FIELDS, _PATH_STABLE_FIELDS

    all_fields = {"st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"}
    assert set(_HANDLE_STABLE_FIELDS) == all_fields
    if os.name == "nt":
        assert "st_ctime_ns" not in _PATH_STABLE_FIELDS
        assert set(_PATH_STABLE_FIELDS) == all_fields - {"st_ctime_ns"}
    else:
        assert set(_PATH_STABLE_FIELDS) == all_fields


def test_capture_is_stable_across_repeated_calls(tmp_path: Path) -> None:
    """Repeatedly capturing a freshly-written file must not flake on Windows.

    Without the st_ctime_ns exclusion, os.fstat and os.stat(path) return
    different ctime semantics on Windows (ChangeTime vs CreationTime),
    causing intermittent failures for freshly written files.
    """
    authority, root = _authority(tmp_path)
    folder = root / "video"
    folder.mkdir()
    payload = _png()
    for _ in range(50):
        path = folder / "result.png"
        path.write_bytes(payload)
        artifact = authority.capture((_candidate(),), "save", 0)[0]
        assert artifact.digest == digest_bytes(payload)
        path.unlink()
    assert load_write_records(root)["video/result.png"]["digest"] == digest_bytes(payload)


def test_handle_comparison_detects_ctime_only_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The before/after handle comparison must still check st_ctime_ns on every platform."""
    authority, root = _authority(tmp_path)
    path = root / "result.png"
    path.write_bytes(_png())
    real_fstat = os.fstat
    calls = 0

    def ctime_changing_fstat(fd: int) -> object:
        nonlocal calls
        result = real_fstat(fd)
        calls += 1
        if calls == 2:
            return _stat_override(result, st_ctime_ns=result.st_ctime_ns + 1_000_000)
        return result

    monkeypatch.setattr(os, "fstat", ctime_changing_fstat)
    with pytest.raises(AssetError, match="changed while"):
        authority.capture((_candidate(subfolder=""),), "save", 0)


def test_path_comparison_detects_inode_only_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handle-vs-path comparison must still catch non-ctime field divergence on Windows."""
    from dinkster_workers import saved_artifacts

    authority, root = _authority(tmp_path)
    target = root / "video" / "result.png"
    target.parent.mkdir()
    target.write_bytes(_png())
    real_open = saved_artifacts._open_candidate

    @contextmanager
    def inode_diverging_open(root_path: Path, parts: tuple[str, ...], filename: str):
        with real_open(root_path, parts, filename) as (handle, restat_path):

            def diverging_restat() -> object:
                original = restat_path()
                return _stat_override(original, st_ino=original.st_ino + 1)

            yield handle, diverging_restat

    monkeypatch.setattr(saved_artifacts, "_open_candidate", inode_diverging_open)
    with pytest.raises(AssetError, match="path changed while"):
        authority.capture((_candidate(),), "save", 0)
