"""Safe streaming extraction of environment base archives."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import warnings
from collections.abc import Buffer
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

import zstandard

from .limits import ENVIRONMENT_STREAM_CHUNK_BYTES

__all__ = ["ArchiveError", "extract_base"]

_ALLOWED_TREES = frozenset({"python", "wheels"})
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


class ArchiveError(ValueError):
    """A base archive is malformed or cannot be safely staged."""


class _PartsReader(io.RawIOBase):
    def __init__(self, files: list[io.BufferedReader]) -> None:
        self.files = iter(files)
        self.current = next(self.files, None)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        while self.current is not None:
            size = self.current.readinto(buffer)
            if size:
                return size
            self.current = next(self.files, None)
        return 0


class _TarReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.size = 0
        self.zero_tail = 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        self.size += len(block)
        trailing = len(block) - len(block.rstrip(b"\0"))
        self.zero_tail = self.zero_tail + trailing if trailing == len(block) else trailing
        return block


def _verify_frame(source: BinaryIO) -> None:
    # stream_reader permits incomplete EOF; require a complete single frame before decoding.
    def read(size: int) -> bytes:
        data = source.read(size)
        if len(data) != size:
            raise ArchiveError("incomplete zstd frame")
        return data

    header = read(6)
    header += read(zstandard.frame_header_size(header) - len(header))
    parameters = zstandard.get_frame_parameters(header)
    while True:
        block = int.from_bytes(read(3), "little")
        kind, size = (block >> 1) & 3, block >> 3
        if kind == 3 or size > 128 * 1024:
            raise ArchiveError("invalid zstd block")
        read(1 if kind == 1 else size)
        if block & 1:
            break
    if parameters.has_checksum:
        read(4)
    if source.read(1):
        raise ArchiveError("base archive must contain one zstd frame")


def _data_filter(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
    filtered = tarfile.data_filter(member, destination)
    if member.issym():
        # Normalizing '..' through a directory symlink changes its filesystem meaning.
        target = Path(destination) / member.name
        if (
            not (target.parent / member.linkname)
            .resolve()
            .is_relative_to(Path(destination).resolve())
        ):
            raise ArchiveError("archive link escapes staging")
        filtered = filtered.replace(linkname=member.linkname)
    return filtered


def extract_base(
    parts: tuple[Path, ...],
    destination: Path,
    *,
    platform: str,
    executable: str,
    expected_parts: tuple[dict[str, object], ...],
) -> Path:
    """Verify ordered parts, then stage one tar+zstd stream before atomic promotion."""
    destination = Path(destination)
    if os.path.lexists(destination):
        raise ArchiveError(f"base destination already exists: {destination}")
    if not parts or len(parts) != len(expected_parts):
        raise ArchiveError("base archive parts differ from the manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        with ExitStack() as stack:
            files: list[io.BufferedReader] = []
            for index, (part, row) in enumerate(zip(parts, expected_parts, strict=True), 1):
                name = f"base.tar.zst.{index:03d}"
                if part.name != name or row.get("filename") != name or not part.is_file():
                    raise ArchiveError("base archive parts are missing or out of order")
                source = stack.enter_context(part.open("rb"))
                sha = hashlib.sha256()
                size = 0
                while block := source.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
                    sha.update(block)
                    size += len(block)
                if size != row.get("size") or sha.hexdigest() != row.get("sha256"):
                    raise ArchiveError(f"base archive part digest or size mismatch: {name}")
                source.seek(0)
                files.append(source)
            with io.BufferedReader(_PartsReader(files)) as compressed:
                _verify_frame(compressed)
            for source in files:
                source.seek(0)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
            )
            insensitive = platform in {"windows", "macos"} or _case_insensitive(staging)
            names: set[str] = set()
            spellings: dict[str, str] = {}
            directories: list[tuple[Path, int]] = []
            modes: list[tuple[Path, int]] = []
            with (
                io.BufferedReader(_PartsReader(files)) as source,
                zstandard.ZstdDecompressor().stream_reader(source) as decoded,
                tarfile.open(
                    fileobj=cast("BinaryIO", tracked := _TarReader(decoded)), mode="r|"
                ) as archive,
            ):
                end = 0
                for member in archive:
                    path = _member_path(member.name)
                    if path.parts[0] not in _ALLOWED_TREES:
                        raise ArchiveError("archive contains an unknown top-level tree")
                    name = path.as_posix()
                    key = name.casefold() if insensitive else name
                    if key in names:
                        raise ArchiveError(f"archive contains ambiguous duplicate path: {name!r}")
                    names.add(key)
                    if insensitive:
                        for end in range(1, len(path.parts) + 1):
                            prefix = "/".join(path.parts[:end])
                            if spellings.setdefault(prefix.casefold(), prefix) != prefix:
                                raise ArchiveError(
                                    f"archive contains case-ambiguous path: {name!r}"
                                )
                    if not (member.isfile() or member.isdir() or member.issym()):
                        raise ArchiveError(f"archive contains unsupported member: {name!r}")
                    if member.issym() and (
                        PurePosixPath(member.linkname).is_absolute()
                        or "\\" in member.linkname
                        or ":" in member.linkname
                    ):
                        raise ArchiveError(f"unsafe archive link: {name!r}")
                    if any((staging / parent).is_symlink() for parent in path.parents):
                        raise ArchiveError(f"archive member has a symbolic-link parent: {name!r}")
                    archive.extract(member, staging, filter=_data_filter)
                    if member.isdir():
                        directories.append((staging / path, member.mode & 0o777))
                    elif member.isfile():
                        modes.append((staging / path, member.mode & 0o755))
                    end = member.offset_data + ((member.size + 511) // 512) * 512
                # Consume the zstd footer as well as the tar end markers.
                while tracked.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
                    pass
                padding = tracked.size - end
                if padding < 1024 or tracked.size % 512 or tracked.zero_tail < padding:
                    raise ArchiveError("incomplete tar end markers or trailing archive data")
        _validate_staging(staging)
        for path, mode in modes:
            path.chmod(mode)
        interpreter = staging / _member_path(executable)
        if not executable.startswith("python/") or not interpreter.is_file():
            raise ArchiveError("manifest interpreter is missing from the python tree")
        if os.name != "nt" and not os.access(interpreter, os.X_OK):
            warnings.warn(
                "base interpreter lacked execute permission; restoring 0755",
                RuntimeWarning,
                stacklevel=2,
            )
            interpreter.chmod(0o755)
        result = subprocess.run(
            [str(interpreter.resolve()), "-I", "-B", "-c", "import sys, sysconfig"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise ArchiveError(f"base interpreter probe failed with exit code {result.returncode}")
        for directory, mode in sorted(directories, key=lambda row: len(row[0].parts), reverse=True):
            directory.chmod(mode)
        if os.path.lexists(destination):
            raise ArchiveError(f"base destination was created while staging: {destination}")
        staging.rename(destination)
        staging = None
        return destination / "python"
    except Exception as exc:
        raise ArchiveError(f"base archive extraction failed: {exc}") from exc
    finally:
        if staging is not None:
            # Restore traversal before cleanup, including modes already applied before a failure.
            staging.chmod(0o700)
            for directory, children, files_in_directory in os.walk(staging):
                Path(directory).chmod(0o700)
                for name in children:
                    path = Path(directory) / name
                    if not path.is_symlink():
                        path.chmod(0o700)
                for name in files_in_directory:
                    path = Path(directory) / name
                    if not path.is_symlink():
                        path.chmod(0o600)
            shutil.rmtree(staging)


def _case_insensitive(directory: Path) -> bool:
    with tempfile.NamedTemporaryFile(prefix=".case-A-", dir=directory) as probe:
        path = Path(probe.name)
        return path.with_name(path.name.swapcase()).exists()


def _member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or any(character in name for character in '*?<>|"')
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in name.split("/"))
        or any(ord(character) < 32 for character in name)
        or any(
            part.endswith((".", " ")) or part.split(".", 1)[0].casefold() in _WINDOWS_DEVICES
            for part in path.parts
        )
    ):
        raise ArchiveError(f"unsafe archive member path: {name!r}")
    return path


def _validate_staging(staging: Path) -> None:
    for path in staging.rglob("*"):
        try:
            path.resolve(strict=True).relative_to(staging.resolve())
        except (ValueError, OSError) as exc:
            raise ArchiveError(f"archive member escaped staging or is dangling: {path}") from exc
        if not path.is_file() and not path.is_dir():
            raise ArchiveError(f"archive created an unsupported member: {path}")
    if not (staging / "python").is_dir() or not (staging / "wheels").is_dir():
        raise ArchiveError("archive must extract both python/ and wheels/ directories")
