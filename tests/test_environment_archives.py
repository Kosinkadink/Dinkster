from __future__ import annotations

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

    result = extract_base(parts, destination)

    assert result == destination / "python"
    assert result.joinpath("bin", "python").read_bytes() == b"interpreter"
    assert destination.joinpath("wheels", "example-1.0.whl").read_bytes() == b"wheel"
    assert not list(destination.parent.glob(".base-id.*"))


def test_rejects_malformed_archive_and_cleans_temporary_files(tmp_path: Path) -> None:
    part = tmp_path / "base.001"
    part.write_bytes(b"not a 7z archive")
    destination = tmp_path / "bases" / "base-id"

    with pytest.raises(ArchiveError, match="extraction failed"):
        extract_base((part,), destination)

    assert not destination.exists()
    assert not destination.parent.exists() or not list(destination.parent.iterdir())


def test_missing_later_part_cleans_reconstructed_archive(tmp_path: Path) -> None:
    first = tmp_path / "base.001"
    first.write_bytes(b"first part")
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError, match="part is not a regular file"):
        extract_base((first, tmp_path / "missing.002"), destination)
    assert not list(destination.parent.iterdir())


@pytest.mark.parametrize(
    "members",
    [
        ("python/module.py", "python/MODULE.py"),
        ("python/Lib/a.py", "python/lib/b.py"),
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
        extract_base((archive,), destination)
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
        extract_base((archive,), destination)
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
        extract_base(parts, tmp_path / "bases" / "base-id")


def test_rejects_archive_without_python_tree(tmp_path: Path) -> None:
    parts = make_parts(tmp_path, {"wheels/example.whl": b"wheel"})

    with pytest.raises(ArchiveError, match="both python/ and wheels/ trees"):
        extract_base(parts, tmp_path / "bases" / "base-id")


def test_preserves_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "bases" / "base-id"
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("existing base")

    with pytest.raises(ArchiveError, match="already exists"):
        extract_base((tmp_path / "missing.001",), destination)

    assert marker.read_text() == "existing base"
