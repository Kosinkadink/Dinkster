from __future__ import annotations

import hashlib
import io
import os
import stat
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

import pytest
import zstandard
from dinkster_supervisor.environment_archives import ArchiveError, extract_base


def member(
    name: str,
    data: bytes = b"payload",
    *,
    mode: int = 0o644,
    kind: bytes = tarfile.REGTYPE,
    target: str = "",
) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode
    info.linkname = target
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    return info, data


def base_members() -> list[tuple[tarfile.TarInfo, bytes]]:
    return [
        member("python/bin/python3", b"interpreter", mode=0o755),
        member("wheels/example.whl", b"wheel"),
    ]


def split_bytes(tmp_path: Path, data: bytes) -> tuple[Path, ...]:
    # Split inside the zstd frame header as well as the compressed tar payload.
    offsets = (0, 3, max(4, len(data) // 2), len(data))
    parts = tuple(tmp_path / f"base.tar.zst.{index:03}" for index in range(1, 4))
    for part, start, end in zip(parts, offsets[:-1], offsets[1:], strict=True):
        part.write_bytes(data[start:end])
    return parts


def make_parts(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]] | None = None,
) -> tuple[Path, ...]:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for info, data in base_members() if members is None else members:
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return split_bytes(tmp_path, zstandard.ZstdCompressor().compress(stream.getvalue()))


def expected_parts(parts: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "filename": part.name,
            "size": part.stat().st_size,
            "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
        }
        for part in parts
    )


def assert_clean(destination: Path) -> None:
    assert not os.path.lexists(destination)
    assert not destination.parent.exists() or not list(destination.parent.iterdir())


@pytest.fixture
def runnable_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr("dinkster_supervisor.environment_archives.subprocess.run", run)


@pytest.mark.usefixtures("runnable_dummy")
def test_extracts_split_archive_to_promoted_base(tmp_path: Path) -> None:
    parts = make_parts(tmp_path)
    destination = tmp_path / "bases/base-id"
    result = extract_base(
        parts,
        destination,
        platform="linux",
        executable="python/bin/python3",
        expected_parts=expected_parts(parts),
    )
    assert result == destination / "python"
    assert (result / "bin/python3").read_bytes() == b"interpreter"
    assert (destination / "wheels/example.whl").read_bytes() == b"wheel"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.usefixtures("runnable_dummy")
def test_preserves_directories_empty_files_and_modes_without_external_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_path = "python/" + "/".join(["nested-directory" * 2] * 10) + "/module.py"
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/empty-directory", kind=tarfile.DIRTYPE, mode=0o750),
            member("python/empty-file", b""),
            member("python/bin/helper", b"helper", mode=0o751),
            member(long_path, b"long path payload"),
        ],
    )
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    destination = tmp_path / "bases/base-id"
    extract_base(
        parts,
        destination,
        platform="linux",
        executable="python/bin/python3",
        expected_parts=expected_parts(parts),
    )
    assert (destination / "python/empty-directory").is_dir()
    assert (destination / "python/empty-file").read_bytes() == b""
    assert (destination / long_path).read_bytes() == b"long path payload"
    if os.name != "nt":
        assert stat.S_IMODE((destination / "python/empty-directory").stat().st_mode) == 0o750
        assert stat.S_IMODE((destination / "python/bin/helper").stat().st_mode) == 0o751
        assert stat.S_IMODE((destination / "wheels/example.whl").stat().st_mode) == 0o644


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize(
    "problem",
    ["tampered", "size", "missing", "reordered", "filename", "missing-row", "extra-row", "empty"],
)
def test_rejects_unverified_parts_and_cleans_staging(tmp_path: Path, problem: str) -> None:
    parts = make_parts(tmp_path)
    rows = expected_parts(parts)
    if problem == "tampered":
        data = parts[-1].read_bytes()
        parts[-1].write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif problem == "size":
        parts[-1].write_bytes(parts[-1].read_bytes() + b"extra")
    elif problem == "missing":
        parts[-1].unlink()
    elif problem == "reordered":
        parts = tuple(reversed(parts))
    elif problem == "filename":
        renamed = parts[-1].with_name("base.tar.zst.004")
        parts[-1].rename(renamed)
        parts = (*parts[:-1], renamed)
        rows = expected_parts(parts)
    elif problem == "missing-row":
        rows = rows[:-1]
    elif problem == "extra-row":
        rows = (*rows, dict(rows[-1]))
    else:
        parts, rows = (), ()
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=rows,
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
def test_checks_integrity_before_decoding(tmp_path: Path) -> None:
    parts = split_bytes(tmp_path, b"this is not a zstd frame")
    rows = expected_parts(parts)
    rows[-1]["sha256"] = "0" * 64
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError, match="(?i)sha256|digest|hash|integrity"):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=rows,
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize("payload", [b"not zstd", zstandard.ZstdCompressor().compress(b"not tar")])
def test_rejects_malformed_archive_and_cleans_temporary_files(
    tmp_path: Path, payload: bytes
) -> None:
    parts = split_bytes(tmp_path, payload)
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize(
    "damage", ["missing-tar-end", "partial-tar-header", "truncated-checksum", "truncated-footer"]
)
def test_rejects_hash_verified_truncated_archive(tmp_path: Path, damage: str) -> None:
    parts = make_parts(tmp_path)
    compressed = b"".join(part.read_bytes() for part in parts)
    tar = zstandard.ZstdDecompressor().decompress(compressed)
    if damage in {"missing-tar-end", "partial-tar-header"}:
        # Both complete regular files occupy one header and one payload block each.
        tar = tar[: 4 * tarfile.BLOCKSIZE]
        if damage == "partial-tar-header":
            tar += tarfile.TarInfo("python/unfinished").tobuf()[:128]
        compressed = zstandard.ZstdCompressor().compress(tar)
    else:
        compressed = zstandard.ZstdCompressor(write_checksum=True).compress(tar)
        compressed = compressed[:-1] if damage == "truncated-checksum" else compressed[:-4]
    parts = split_bytes(tmp_path, compressed)
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.skipif(os.name == "nt", reason="Requires a case-sensitive filesystem")
def test_linux_terminfo_preserves_distinct_case_paths(tmp_path: Path) -> None:
    probe = tmp_path / "case-probe-A"
    probe.touch()
    if probe.with_name("case-probe-a").exists():
        pytest.skip("Filesystem is case-insensitive")
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/share/terminfo/A/entry", b"upper"),
            member("python/share/terminfo/a/entry", b"lower"),
        ],
    )
    destination = tmp_path / "bases/base-id"
    extract_base(
        parts,
        destination,
        platform="linux",
        executable="python/bin/python3",
        expected_parts=expected_parts(parts),
    )
    assert (destination / "python/share/terminfo/A/entry").read_bytes() == b"upper"
    assert (destination / "python/share/terminfo/a/entry").read_bytes() == b"lower"


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize("platform", ["windows", "macos", "linux"])
@pytest.mark.parametrize(
    "names",
    [
        ("python/module.py", "python/MODULE.py"),
        ("python/share/terminfo/A/entry", "python/share/terminfo/a/other"),
    ],
)
def test_case_collisions_refused_for_insensitive_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    names: tuple[str, str],
) -> None:
    if platform == "linux":
        monkeypatch.setattr(
            "dinkster_supervisor.environment_archives._case_insensitive", lambda _: True
        )
    parts = make_parts(tmp_path, base_members() + [member(name) for name in names])
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform=platform,
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize("platform", ["linux", "windows", "macos"])
def test_exact_duplicate_members_always_refused(tmp_path: Path, platform: str) -> None:
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/module.py", b"first"),
            member("python/module.py", b"second"),
        ],
    )
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform=platform,
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "python/../../escape",
        "/absolute/escape",
        "other/file",
        "python/C:/escape",
        "python/NUL.txt",
        "python/module.py.",
        "python/module?.py",
        "python/module*.py",
        "python/module<.py",
        "python/module>.py",
        "python/module|.py",
        'python/module".py',
        "python/..\\escape",
        "C:/escape",
    ],
)
def test_rejects_unsafe_member_paths_before_promotion(tmp_path: Path, name: str) -> None:
    outside = tmp_path / "escape"
    outside.write_bytes(b"preserve")
    parts = make_parts(tmp_path, base_members() + [member(name, b"overwrite")])
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="windows",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert outside.read_bytes() == b"preserve"
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.skipif(os.name == "nt", reason="Requires unprivileged symlink creation")
def test_internal_relative_symlinks_are_preserved(tmp_path: Path) -> None:
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/lib/value", b"value"),
            member("python/bin/value", kind=tarfile.SYMTYPE, target="../lib/value"),
            member("python/bin/python", kind=tarfile.SYMTYPE, target="python3"),
        ],
    )
    destination = tmp_path / "bases/base-id"
    extract_base(
        parts,
        destination,
        platform="linux",
        executable="python/bin/python",
        expected_parts=expected_parts(parts),
    )
    assert (destination / "python/bin/value").is_symlink()
    assert (destination / "python/bin/value").read_bytes() == b"value"
    assert (destination / "python/bin/python").is_symlink()
    assert os.readlink(destination / "python/bin/value") == "../lib/value"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.usefixtures("runnable_dummy")
def test_symlink_parent_traversal_preserves_original_target_and_payload(tmp_path: Path) -> None:
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/other/sub", kind=tarfile.DIRTYPE, mode=0o755),
            member("python/other/value", b"intended payload"),
            member("python/lib/value", b"different payload"),
            member("python/lib/hop", kind=tarfile.SYMTYPE, target="../other/sub"),
            member("python/lib/alias", kind=tarfile.SYMTYPE, target="hop/../value"),
        ],
    )
    destination = tmp_path / "bases/base-id"
    if os.name == "nt":
        # Windows cannot preserve POSIX '..' traversal through a directory symlink.
        with pytest.raises(ArchiveError):
            extract_base(
                parts,
                destination,
                platform="windows",
                executable="python/bin/python3",
                expected_parts=expected_parts(parts),
            )
        assert_clean(destination)
        return
    extract_base(
        parts,
        destination,
        platform="linux",
        executable="python/bin/python3",
        expected_parts=expected_parts(parts),
    )
    alias = destination / "python/lib/alias"
    assert os.readlink(destination / "python/lib/hop") == "../other/sub"
    assert os.readlink(alias) == "hop/../value"
    assert alias.read_bytes() == b"intended payload"
    assert (destination / "python/lib/value").read_bytes() == b"different payload"


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize(
    "kind,target",
    [
        (tarfile.SYMTYPE, "../../../outside"),
        (tarfile.SYMTYPE, "/outside"),
        (tarfile.SYMTYPE, "C:/outside"),
        (tarfile.LNKTYPE, "python/bin/python3"),
        (tarfile.LNKTYPE, "../../outside"),
        (tarfile.CHRTYPE, ""),
        (tarfile.BLKTYPE, ""),
        (tarfile.FIFOTYPE, ""),
    ],
)
def test_rejects_escaping_links_hardlinks_and_special_files(
    tmp_path: Path,
    kind: bytes,
    target: str,
) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/link", kind=kind, target=target),
        ],
    )
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert outside.read_bytes() == b"preserve"
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
def test_rejects_absolute_symlink_to_existing_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/link", kind=tarfile.SYMTYPE, target=str(outside)),
        ],
    )
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert outside.read_bytes() == b"preserve"
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize(
    "members,executable",
    [
        ([member("wheels/example.whl")], "python/bin/python3"),
        ([member("python/bin/python3", mode=0o755)], "python/bin/python3"),
        (base_members(), "python/bin/missing"),
        (base_members(), "wheels/example.whl"),
    ],
)
def test_requires_both_trees_and_manifest_interpreter(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    executable: str,
) -> None:
    parts = make_parts(tmp_path, members)
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable=executable,
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.skipif(os.name == "nt", reason="Unix executable permissions")
def test_missing_execute_mode_warns_and_repairs_only_manifest_interpreter(tmp_path: Path) -> None:
    parts = make_parts(
        tmp_path,
        [
            member("python/bin/python3", b"interpreter", mode=0o644),
            member("python/bin/other", mode=0o640),
            member("wheels/example.whl"),
        ],
    )
    destination = tmp_path / "bases/base-id"
    with pytest.warns(RuntimeWarning):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert stat.S_IMODE((destination / "python/bin/python3").stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "python/bin/other").stat().st_mode) == 0o640
    assert stat.S_IMODE((destination / "wheels/example.whl").stat().st_mode) == 0o644


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.skipif(os.name == "nt", reason="Unix executable permissions")
def test_executable_mode_is_preserved_without_repair_warning(tmp_path: Path) -> None:
    parts = make_parts(
        tmp_path,
        [
            member("python/bin/python3", b"interpreter", mode=0o751),
            member("wheels/example.whl"),
        ],
    )
    destination = tmp_path / "bases/base-id"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert stat.S_IMODE((destination / "python/bin/python3").stat().st_mode) == 0o751


def test_invalid_interpreter_is_not_promoted(tmp_path: Path) -> None:
    parts = make_parts(tmp_path)
    destination = tmp_path / "bases/base-id"
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


@pytest.mark.parametrize("failure", ["exit", "timeout", "not-executable"])
def test_interpreter_must_run_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    parts = make_parts(tmp_path)
    destination = tmp_path / "bases/base-id"

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert not destination.exists()
        if failure == "timeout":
            raise subprocess.TimeoutExpired("python", 30)
        if failure == "not-executable":
            raise OSError("invalid executable")
        return subprocess.CompletedProcess([], 1, stdout="", stderr="broken stdlib")

    monkeypatch.setattr("dinkster_supervisor.environment_archives.subprocess.run", run)
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert_clean(destination)


def test_probe_failure_cleans_populated_inaccessible_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = make_parts(
        tmp_path,
        base_members()
        + [
            member("python/locked", kind=tarfile.DIRTYPE, mode=0o000),
            member("python/locked/value", b"preserve", mode=0o444),
        ],
    )
    destination = tmp_path / "bases/base-id"
    probed = False

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal probed
        probed = True
        return subprocess.CompletedProcess([], 1, stdout="", stderr="broken stdlib")

    monkeypatch.setattr("dinkster_supervisor.environment_archives.subprocess.run", run)
    try:
        with pytest.raises(ArchiveError, match="probe failed"):
            extract_base(
                parts,
                destination,
                platform="linux",
                executable="python/bin/python3",
                expected_parts=expected_parts(parts),
            )
        assert probed
        assert_clean(destination)
    finally:
        for remaining in destination.parent.iterdir():
            locked = remaining / "python/locked"
            if locked.exists():
                locked.chmod(0o700)


@pytest.mark.usefixtures("runnable_dummy")
@pytest.mark.parametrize("parent_first", [True, False])
def test_directory_modes_are_restored_deepest_first_regardless_of_member_order(
    tmp_path: Path, parent_first: bool
) -> None:
    directories = [
        member("python/locked", kind=tarfile.DIRTYPE, mode=0o000),
        member("python/locked/child", kind=tarfile.DIRTYPE, mode=0o500),
    ]
    if not parent_first:
        directories.reverse()
    parts = make_parts(
        tmp_path,
        base_members() + directories + [member("python/locked/child/value")],
    )
    destination = tmp_path / "bases/base-id"
    try:
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
        locked = destination / "python/locked"
        if os.name != "nt":
            assert stat.S_IMODE(locked.stat().st_mode) == 0o000
        locked.chmod(0o700)
        if os.name != "nt":
            assert stat.S_IMODE((locked / "child").stat().st_mode) == 0o500
        value = locked / "child/value"
        assert value.read_bytes() == b"payload"
        assert list(destination.parent.iterdir()) == [destination]
    finally:
        for remaining in destination.parent.iterdir():
            locked = remaining / "python/locked"
            if locked.exists():
                locked.chmod(0o700)
                child = locked / "child"
                if child.exists():
                    child.chmod(0o700)


@pytest.mark.usefixtures("runnable_dummy")
def test_read_only_regular_file_mode_is_preserved(tmp_path: Path) -> None:
    parts = make_parts(
        tmp_path, base_members() + [member("python/value", b"read only", mode=0o444)]
    )
    destination = tmp_path / "bases/base-id"
    extract_base(
        parts,
        destination,
        platform="linux",
        executable="python/bin/python3",
        expected_parts=expected_parts(parts),
    )
    value = destination / "python/value"
    assert stat.S_IMODE(value.stat().st_mode) == 0o444
    assert value.read_bytes() == b"read only"


def test_extracts_and_runs_complete_standalone_interpreter(tmp_path: Path) -> None:
    from test_environment_generations import _standalone_tree

    source = tmp_path / "source"
    _standalone_tree(source / "python")
    (source / "wheels").mkdir()
    (source / "wheels/example.whl").write_bytes(b"wheel")
    archive_path = tmp_path / "standalone.tar.zst"
    with archive_path.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", dereference=True) as archive:
                archive.add(source / "python", arcname="python")
                archive.add(source / "wheels", arcname="wheels")
    parts = split_bytes(tmp_path, archive_path.read_bytes())
    executable = "python/python.exe" if os.name == "nt" else "python/bin/python3"
    platform = "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux"
    destination = tmp_path / "bases/base-id"
    extract_base(
        parts,
        destination,
        platform=platform,
        executable=executable,
        expected_parts=expected_parts(parts),
    )
    result = subprocess.run(
        [str(destination / executable), "-I", "-c", "import sys, sysconfig"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.usefixtures("runnable_dummy")
def test_preserves_existing_destination(tmp_path: Path) -> None:
    parts = make_parts(tmp_path)
    destination = tmp_path / "bases/base-id"
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("existing base")
    with pytest.raises(ArchiveError):
        extract_base(
            parts,
            destination,
            platform="linux",
            executable="python/bin/python3",
            expected_parts=expected_parts(parts),
        )
    assert marker.read_text() == "existing base"
    assert list(destination.iterdir()) == [marker]
    assert list(destination.parent.iterdir()) == [destination]
