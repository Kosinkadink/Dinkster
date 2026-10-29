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

import zstandard

__all__ = ["ArchiveError", "extract_base"]

_CHUNK_SIZE = 1024 * 1024
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
                while block := source.read(_CHUNK_SIZE):
                    sha.update(block)
                    size += len(block)
                if size != row.get("size") or sha.hexdigest() != row.get("sha256"):
                    raise ArchiveError(f"base archive part digest or size mismatch: {name}")
                source.seek(0)
                files.append(source)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
            )
            insensitive = platform in {"windows", "macos"} or _case_insensitive(staging)
            names: set[str] = set()
            spellings: dict[str, str] = {}
            directories: list[tuple[Path, int]] = []
            with (
                io.BufferedReader(_PartsReader(files)) as source,
                zstandard.ZstdDecompressor().stream_reader(source) as decoded,
                tarfile.open(fileobj=decoded, mode="r|") as archive,
            ):
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
                    archive.extract(member, staging, filter="data")
                    if member.isdir():
                        directories.append((staging / path, member.mode & 0o777))
                # Consume the zstd footer as well as the tar end markers.
                while decoded.read(_CHUNK_SIZE):
                    pass
        _validate_staging(staging)
        # The data filter intentionally discards directory modes; restore safe permission bits.
        for directory, mode in reversed(directories):
            directory.chmod(mode)
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
            [str(interpreter.resolve()), "-I", "-c", "import sys, sysconfig"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise ArchiveError(f"base interpreter probe failed with exit code {result.returncode}")
        if os.path.lexists(destination):
            raise ArchiveError(f"base destination was created while staging: {destination}")
        staging.rename(destination)
        staging = None
        return destination / "python"
    except Exception as exc:
        raise ArchiveError(f"base archive extraction failed: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


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
