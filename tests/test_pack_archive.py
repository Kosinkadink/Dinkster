from __future__ import annotations

import os
import sys
import unicodedata
import zipfile
from pathlib import Path

import pytest
from blake3 import blake3
from dinkster_workers import ManifestError

from dinkster import manager
from dinkster.pack_archive import (
    MAX_ARCHIVE_BYTES,
    MAX_ENTRIES,
    MAX_EXPANDED_BYTES,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PATH_BYTES,
    PackArchiveError,
    _enforce_limit,
    build_pack_archive,
)
from tests.platform_support import symlink_or_skip

MANIFEST = '[pack]\nname = "archive-test"\n\n[pack.entry]\nnodes = "nodes:NODES"\n'


def write_pack(root: Path) -> Path:
    root.mkdir()
    (root / "dinkster-pack.toml").write_bytes(MANIFEST.encode())
    (root / "nodes.py").write_bytes(b"NODES = []\n")
    return root


def test_archive_is_deterministic_and_has_registry_metadata(tmp_path: Path) -> None:
    first = write_pack(tmp_path / "first")
    second = write_pack(tmp_path / "second")
    os.utime(second / "nodes.py", (0, 0))
    (second / "nodes.py").chmod(0o755)

    first_output = tmp_path / "first.zip"
    second_output = tmp_path / "second.zip"
    first_digest = build_pack_archive(first, first_output)
    second_digest = build_pack_archive(second, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert (
        first_digest == second_digest == f"blake3:{blake3(first_output.read_bytes()).hexdigest()}"
    )
    assert first_digest == "blake3:ccbd5bd1ece43a67a6256b92b58c4787036e85129b24bf08f59070c285ed3fe6"
    with zipfile.ZipFile(first_output) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.external_attr == 0o644 << 16
            assert info.extra == b""
            assert info.comment == b""


def test_archive_normalizes_paths_and_omits_excluded_directories(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    decomposed = unicodedata.normalize("NFD", f"caf{chr(0xE9)}.py")
    (pack / decomposed).write_text("content")
    for excluded in (
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    ):
        directory = pack / "nested" / excluded
        directory.mkdir(parents=True)
        (directory / "ignored").write_text("ignored")

    output = tmp_path / "pack.zip"
    build_pack_archive(pack, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert unicodedata.normalize("NFC", decomposed) in names
    assert all(not name.startswith("nested/") for name in names)


def test_archive_rejects_symlinks_and_normalized_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    symlink_or_skip(pack / "link.py", pack / "nodes.py")
    with pytest.raises(PackArchiveError, match="symlink"):
        build_pack_archive(pack, tmp_path / "symlink.zip")

    (pack / "link.py").unlink()
    entries = list(pack.rglob("*"))
    composed = f"{chr(0xE9)}.py"
    decomposed = unicodedata.normalize("NFD", composed)
    (pack / composed).write_text("one")
    (pack / decomposed).write_text("two")
    # Normalization-insensitive filesystems cannot store both directory entries.
    monkeypatch.setattr(
        Path, "rglob", lambda self, pattern: iter([*entries, pack / composed, pack / decomposed])
    )
    with pytest.raises(PackArchiveError, match="collides"):
        build_pack_archive(pack, tmp_path / "collision.zip")


def test_archive_validates_manifest_before_writing(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    (pack / "dinkster-pack.toml").write_text("not toml = [")
    output = tmp_path / "pack.zip"
    with pytest.raises(ManifestError):
        build_pack_archive(pack, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("limit", "subject"),
    [
        (MAX_ARCHIVE_BYTES, "archive"),
        (MAX_FILE_BYTES, "file"),
        (MAX_EXPANDED_BYTES, "expanded archive"),
        (MAX_PATH_BYTES, "archive path"),
        (MAX_MANIFEST_BYTES, "pack manifest"),
    ],
)
def test_byte_limits_accept_exact_and_reject_plus_one(limit: int, subject: str) -> None:
    _enforce_limit(limit, limit, subject)
    with pytest.raises(PackArchiveError, match="limit"):
        _enforce_limit(limit + 1, limit, subject)


def test_entry_limit_accepts_exact_and_rejects_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    monkeypatch.setattr("dinkster.pack_archive.MAX_ENTRIES", 2)
    build_pack_archive(pack, tmp_path / "exact.zip")
    (pack / "third.py").write_text("")
    with pytest.raises(PackArchiveError, match="3 entries; limit is 2"):
        build_pack_archive(pack, tmp_path / "too-many.zip")
    assert MAX_ENTRIES == 10_000


def test_file_manifest_expanded_and_archive_limits_are_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    largest = pack / "largest.bin"
    largest.write_bytes(b"x" * 100)
    monkeypatch.setattr("dinkster.pack_archive.MAX_FILE_BYTES", 100)
    build_pack_archive(pack, tmp_path / "file-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_FILE_BYTES", 99)
    with pytest.raises(PackArchiveError, match="file 'largest.bin'"):
        build_pack_archive(pack, tmp_path / "file.zip")

    monkeypatch.setattr("dinkster.pack_archive.MAX_FILE_BYTES", MAX_FILE_BYTES)
    monkeypatch.setattr("dinkster.pack_archive.MAX_MANIFEST_BYTES", len(MANIFEST))
    build_pack_archive(pack, tmp_path / "manifest-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_MANIFEST_BYTES", len(MANIFEST) - 1)
    with pytest.raises(PackArchiveError, match="pack manifest"):
        build_pack_archive(pack, tmp_path / "manifest.zip")

    monkeypatch.setattr("dinkster.pack_archive.MAX_MANIFEST_BYTES", MAX_MANIFEST_BYTES)
    expanded = len(MANIFEST.encode()) + len(b"NODES = []\n") + 100
    monkeypatch.setattr("dinkster.pack_archive.MAX_EXPANDED_BYTES", expanded)
    build_pack_archive(pack, tmp_path / "expanded-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_EXPANDED_BYTES", expanded - 1)
    with pytest.raises(PackArchiveError, match="expanded archive"):
        build_pack_archive(pack, tmp_path / "expanded.zip")

    monkeypatch.setattr("dinkster.pack_archive.MAX_EXPANDED_BYTES", MAX_EXPANDED_BYTES)
    baseline = tmp_path / "archive-baseline.zip"
    build_pack_archive(pack, baseline)
    archive_size = baseline.stat().st_size
    monkeypatch.setattr("dinkster.pack_archive.MAX_ARCHIVE_BYTES", archive_size)
    build_pack_archive(pack, tmp_path / "archive-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_ARCHIVE_BYTES", archive_size - 1)
    with pytest.raises(PackArchiveError, match="archive"):
        build_pack_archive(pack, tmp_path / "archive.zip")


def test_path_limit_counts_normalized_utf8_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    unicode_file = pack / (chr(0xE9) * 8 + ".py")
    unicode_file.write_text("")
    normalized_bytes = len(unicode_file.name.encode("utf-8"))
    monkeypatch.setattr("dinkster.pack_archive.MAX_PATH_BYTES", normalized_bytes)
    build_pack_archive(pack, tmp_path / "exact.zip")
    longer = pack / (unicode_file.name + "x")
    unicode_file.rename(longer)
    with pytest.raises(PackArchiveError, match="archive path"):
        build_pack_archive(pack, tmp_path / "too-long.zip")


def test_cli_writes_archive_and_prints_written_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = write_pack(tmp_path / "pack")
    output = tmp_path / "output" / "pack.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster pack", "archive", str(pack), "--output", str(output)],
    )
    manager.main()
    printed = capsys.readouterr().out.strip()
    assert printed == f"blake3:{blake3(output.read_bytes()).hexdigest()}"
