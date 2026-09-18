"""Pack artifacts: deterministic bytes, verified unpacking (DESIGN M8).

What this proves: the same pack tree produces the same artifact digest
regardless of mtimes, permissions, or build order (the identity the whole
pipeline shares); junk and symlinks never enter an artifact; and unpacking
verifies the digest before extraction and refuses archives that would
write outside their store directory - a hostile archive fails loudly with
nothing extracted.
"""

from __future__ import annotations

import os
import sys
import warnings
import zipfile
from pathlib import Path

import pytest
from dinkster_registry import (
    ArtifactError,
    artifact_digest,
    build_artifact,
    unpack_artifact,
    verify_artifact,
)

from tests.platform_support import symlink_or_skip

MANIFEST = '[pack]\nname = "demo"\n\n[pack.entry]\nnodes = "demo_nodes:NODES"\n'


def write_pack(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dinkster-pack.toml").write_text(MANIFEST)
    (directory / "demo_nodes.py").write_text("NODES = []\n")
    (directory / "data").mkdir()
    (directory / "data" / "table.json").write_text("{}")
    return directory


def test_build_is_deterministic_across_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Platform, mtimes, permissions, and rebuilds do not change the digest."""
    first = write_pack(tmp_path / "a")
    second = write_pack(tmp_path / "b")
    os.utime(second / "demo_nodes.py", (0, 0))
    (second / "demo_nodes.py").chmod(0o755)
    digest_a = build_artifact(first, tmp_path / "a.zip")
    monkeypatch.setattr(
        sys,
        "platform",
        "linux" if sys.platform == "win32" else "win32",
    )
    digest_b = build_artifact(second, tmp_path / "b.zip")
    assert digest_a == digest_b
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    assert digest_a == artifact_digest((tmp_path / "a.zip").read_bytes())


def test_build_tracks_content(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    before = build_artifact(pack, tmp_path / "before.zip")
    (pack / "demo_nodes.py").write_text("NODES = [1]\n")
    after = build_artifact(pack, tmp_path / "after.zip")
    assert before != after


def test_build_excludes_junk(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    baseline = build_artifact(pack, tmp_path / "baseline.zip")
    (pack / "__pycache__").mkdir()
    (pack / "__pycache__" / "demo.pyc").write_bytes(b"\x00")
    (pack / ".git").mkdir()
    (pack / ".git" / "HEAD").write_text("ref")
    assert build_artifact(pack, tmp_path / "junky.zip") == baseline


def test_build_refuses_symlinks_and_non_packs(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    symlink_or_skip(pack / "link.py", pack / "demo_nodes.py")
    with pytest.raises(ArtifactError, match="symlink"):
        build_artifact(pack, tmp_path / "out.zip")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactError, match="dinkster-pack.toml"):
        build_artifact(empty, tmp_path / "out.zip")
    with pytest.raises(ArtifactError, match="not a pack directory"):
        build_artifact(tmp_path / "missing", tmp_path / "out.zip")


def test_verify_and_unpack_round_trip(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    digest = build_artifact(pack, tmp_path / "pack.zip")
    verify_artifact(tmp_path / "pack.zip", digest)
    dest = unpack_artifact(tmp_path / "pack.zip", tmp_path / "store", digest)
    assert (dest / "dinkster-pack.toml").read_text() == MANIFEST
    assert (dest / "data" / "table.json").read_text() == "{}"
    # the unpacked tree rebuilds to the identical digest - store is faithful
    assert build_artifact(dest, tmp_path / "again.zip") == digest


def test_unpack_refuses_digest_mismatch_without_extracting(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    build_artifact(pack, tmp_path / "pack.zip")
    wrong = artifact_digest(b"other bytes")
    with pytest.raises(ArtifactError, match="digest mismatch"):
        unpack_artifact(tmp_path / "pack.zip", tmp_path / "store", wrong)
    assert not (tmp_path / "store").exists()
    with pytest.raises(ArtifactError, match="expected digest"):
        unpack_artifact(tmp_path / "pack.zip", tmp_path / "store", "md5:nope")


def _hostile_archive(path: Path, name: str) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("dinkster-pack.toml", MANIFEST)
        archive.writestr(name, "owned")
    return artifact_digest(path.read_bytes())


def test_unpack_refuses_traversal_and_absolute_entries(tmp_path: Path) -> None:
    """Even a correctly-digested archive cannot write outside dest."""
    for hostile in ("../escape.txt", "/tmp/escape.txt", "nested/../../escape.txt"):
        archive = tmp_path / "hostile.zip"
        digest = _hostile_archive(archive, hostile)
        dest = tmp_path / "store"
        with pytest.raises(ArtifactError, match="escapes the destination"):
            unpack_artifact(archive, dest, digest)
        assert not dest.exists()  # refused before a single byte was written
        assert not (tmp_path / "escape.txt").exists()


def test_unpack_refuses_duplicate_entries(tmp_path: Path) -> None:
    archive = tmp_path / "dupes.zip"
    with (
        warnings.catch_warnings(),
        zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as out,
    ):
        warnings.simplefilter("ignore")  # zipfile itself warns about the dupe
        out.writestr("dinkster-pack.toml", MANIFEST)
        out.writestr("a.py", "first")
        out.writestr("a.py", "second")
    digest = artifact_digest(archive.read_bytes())
    with pytest.raises(ArtifactError, match="duplicate entry"):
        unpack_artifact(archive, tmp_path / "store", digest)
