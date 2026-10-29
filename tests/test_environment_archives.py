from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import py7zr
import pytest
from dinkster_supervisor.environment_archives import ArchiveError, extract_base


def make_parts(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, Path]:
    source = tmp_path / "source"
    for name, data in members.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        for name in members:
            writer.write(source / name, arcname=name)
    data = archive.read_bytes()
    split_at = max(1, len(data) // 2)
    first = tmp_path / "base.001"
    second = tmp_path / "base.002"
    first.write_bytes(data[:split_at])
    second.write_bytes(data[split_at:])
    return first, second


def test_extracts_split_archive_to_promoted_base(tmp_path: Path) -> None:
    parts = make_parts(
        tmp_path,
        {"python/bin/python": b"interpreter", "wheels/example-1.0.whl": b"wheel"},
    )
    destination = tmp_path / "bases" / "base-id"

    result = extract_base(parts, destination, platform="linux", executable="python/bin/python")

    assert result == destination / "python"
    assert result.joinpath("bin", "python").read_bytes() == b"interpreter"
    assert destination.joinpath("wheels", "example-1.0.whl").read_bytes() == b"wheel"
    assert not list(destination.parent.glob(".base-id.*"))


@pytest.mark.skipif(os.name == "nt", reason="Linux archive requires a case-sensitive filesystem")
def test_linux_terminfo_preserves_distinct_case_paths(tmp_path: Path) -> None:
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        writer.writestr(b"upper", "python/share/terminfo/A/entry")
        writer.writestr(b"lower", "python/share/terminfo/a/entry")
        writer.writestr(b"interpreter", "python/bin/python3")
        writer.writestr(b"wheel", "wheels/example.whl")
    destination = tmp_path / "bases/base-id"
    extract_base((archive,), destination, platform="linux", executable="python/bin/python3")
    assert (destination / "python/share/terminfo/A/entry").read_bytes() == b"upper"
    assert (destination / "python/share/terminfo/a/entry").read_bytes() == b"lower"


@pytest.mark.parametrize("platform", ["windows", "macos", "linux"])
def test_case_ambiguous_terminfo_refused_for_insensitive_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        writer.writestr(b"upper", "python/share/terminfo/A/entry")
        writer.writestr(b"lower", "python/share/terminfo/a/entry")
        writer.writestr(b"wheel", "wheels/example.whl")
    if platform == "linux":
        monkeypatch.setattr(
            "dinkster_supervisor.environment_archives._case_insensitive", lambda _: True
        )
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError, match="ambiguous"):
        extract_base((archive,), destination, platform=platform, executable="python/bin/python3")
    assert not list(destination.parent.iterdir())


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_exact_duplicate_members_always_refused(tmp_path: Path, platform: str) -> None:
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        writer.writestr(b"first", "python/module.py")
        writer.writestr(b"second", "python/module.py")
        writer.writestr(b"wheel", "wheels/example.whl")
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError, match="duplicate path"):
        extract_base((archive,), destination, platform=platform, executable="python/bin/python3")
    assert not list(destination.parent.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="Unix executable permissions")
def test_manifest_interpreter_gets_execute_mode_without_changing_other_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "python3"
    source.write_bytes(Path(sys.executable).resolve().read_bytes())
    source.chmod(0o644)
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        writer.write(source, "python/bin/python3")
        writer.write(source, "python/bin/other")
        writer.writestr(b"wheel", "wheels/example.whl")
    destination = tmp_path / "bases/base-id"
    extract_base((archive,), destination, platform="linux", executable="python/bin/python3")
    python = destination / "python/bin/python3"
    assert stat.S_IMODE(python.stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "python/bin/other").stat().st_mode) == 0o644
    assert not (destination / "wheels/example.whl").stat().st_mode & 0o111
    result = subprocess.run(
        [str(python), "-c", "import sys"],
        env={**os.environ, "PYTHONHOME": sys.base_prefix},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_attribute_free_archive_preserves_directories_and_empty_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        writer.write(source, "python")
        writer.write(source, "python/empty-directory")
        writer.writestr(b"interpreter", "python/python.exe")
        writer.writestr(b"", "python/empty-file")
        writer.writestr(b"wheel", "wheels/example.whl")
    original = py7zr.SevenZipFile._real_get_contents

    def without_attributes(reader: py7zr.SevenZipFile, password: str | None) -> None:
        original(reader, password)
        info = reader.header.files_info
        assert info is not None
        info.emptyfiles = [not member.is_directory for member in reader.files if member.emptystream]
        for member in info.files:
            member.pop("attributes", None)

    monkeypatch.setattr(py7zr.SevenZipFile, "_real_get_contents", without_attributes)
    destination = tmp_path / "bases/base-id"
    extract_base((archive,), destination, platform="linux", executable="python/python.exe")
    assert (destination / "python/empty-directory").is_dir()
    assert (destination / "python/empty-file").is_file()
    assert (destination / "python/empty-file").read_bytes() == b""


def test_rejects_malformed_archive_and_cleans_temporary_files(tmp_path: Path) -> None:
    part = tmp_path / "base.001"
    part.write_bytes(b"not a 7z archive")
    destination = tmp_path / "bases" / "base-id"

    with pytest.raises(ArchiveError, match="extraction failed"):
        extract_base((part,), destination, platform="linux", executable="python/bin/python")

    assert not destination.exists()
    assert not destination.parent.exists() or not list(destination.parent.iterdir())


def test_missing_later_part_cleans_reconstructed_archive(tmp_path: Path) -> None:
    first = tmp_path / "base.001"
    first.write_bytes(b"first part")
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError, match="part is not a regular file"):
        extract_base(
            (first, tmp_path / "missing.002"),
            destination,
            platform="linux",
            executable="python/bin/python",
        )
    assert not list(destination.parent.iterdir())


@pytest.mark.parametrize(
    "members",
    [
        ("python/module.py", "python/MODULE.py"),
        ("python/Lib/a.py", "python/lib/b.py"),
        ("python/module.py", "python/module.py"),
        ("other/file",),
        ("python/C:/escape",),
        ("python/NUL.txt",),
        ("python/module.py.",),
    ],
)
def test_rejects_unsafe_or_ambiguous_members_before_promotion(
    tmp_path: Path, members: tuple[str, ...]
) -> None:
    archive = tmp_path / "unsafe.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        for name in members:
            writer.writestr(b"payload", name)
        writer.writestr(b"wheel", "wheels/example.whl")
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base((archive,), destination, platform="windows", executable="python/python.exe")
    assert not list(destination.parent.iterdir())


def test_rejects_symbolic_link_member_without_touching_its_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    source = tmp_path / "source"
    source.mkdir()
    link = source / "link"
    link.symlink_to(outside)
    archive = tmp_path / "link.7z"
    with py7zr.SevenZipFile(archive, "w", dereference=False) as writer:
        writer.write(link, "python/link")
        writer.writestr(b"wheel", "wheels/example.whl")
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError, match="links are not supported"):
        extract_base((archive,), destination, platform="linux", executable="python/bin/python")
    assert outside.read_bytes() == b"preserve"
    assert not list(destination.parent.iterdir())


def test_rejects_parent_traversal_member(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "payload"
    payload.write_bytes(b"escape")
    archive = tmp_path / "base.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        writer.write(payload, arcname="../python/escape")
        writer.writestr(b"wheel", "wheels/example.whl")
    data = archive.read_bytes()
    parts = (tmp_path / "base.001", tmp_path / "base.002")
    parts[0].write_bytes(data[:3])
    parts[1].write_bytes(data[3:])

    with pytest.raises(ArchiveError, match="unsafe archive member path"):
        extract_base(
            parts, tmp_path / "bases" / "base-id", platform="linux", executable="python/bin/python"
        )


def test_rejects_archive_without_python_tree(tmp_path: Path) -> None:
    parts = make_parts(tmp_path, {"wheels/example.whl": b"wheel"})

    with pytest.raises(ArchiveError, match="both python/ and wheels/ trees"):
        extract_base(
            parts, tmp_path / "bases" / "base-id", platform="linux", executable="python/bin/python"
        )


def test_preserves_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "bases" / "base-id"
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("existing base")

    with pytest.raises(ArchiveError, match="already exists"):
        extract_base(
            (tmp_path / "missing.001",),
            destination,
            platform="linux",
            executable="python/bin/python",
        )

    assert marker.read_text() == "existing base"
