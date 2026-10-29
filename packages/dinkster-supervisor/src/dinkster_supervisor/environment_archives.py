"""Safe reconstruction and extraction of environment base archives."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath

import py7zr

__all__ = ["ArchiveError", "extract_base"]

_CHUNK_SIZE = 1024 * 1024
_ALLOWED_TREES = frozenset({"python", "wheels"})
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


class ArchiveError(ValueError):
    """A base archive is malformed or cannot be safely staged."""


def extract_base(
    parts: tuple[Path, ...], destination: Path, *, platform: str, executable: str
) -> Path:
    """Preflight and extract verified archive parts using the real 7-Zip binary."""
    destination = Path(destination)
    if os.path.lexists(destination):
        raise ArchiveError(f"base destination already exists: {destination}")
    if not parts:
        raise ArchiveError("base archive has no parts")

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    staging: Path | None = None
    try:
        archive_path = _reconstruct_archive(parts, destination)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
        )
        binary = os.environ.get("DINKSTER_7ZIP_BINARY")
        if binary:
            if not Path(binary).is_absolute() or not Path(binary).is_file():
                raise ArchiveError("DINKSTER_7ZIP_BINARY must name an existing absolute executable")
        else:
            binary = str(
                distribution("py7zz").locate_file(
                    f"py7zz/bin/{'7zz.exe' if os.name == 'nt' else '7zz'}"
                )
            )
            if not Path(binary).is_file():
                raise ArchiveError("installed py7zz wheel has no bundled decoder")
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            _validate_members(
                archive,
                case_insensitive=platform in {"windows", "macos"} or _case_insensitive(staging),
            )
        result = subprocess.run(
            [binary, "x", "-y", "-bd", f"-o{staging.resolve()}", str(archive_path.resolve())],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise ArchiveError(f"7-Zip extraction failed with exit code {result.returncode}")
        _validate_staging(staging)
        interpreter = staging / _member_path(executable)
        if not executable.startswith("python/") or not interpreter.is_file():
            raise ArchiveError("manifest interpreter is missing from the python tree")
        interpreter.chmod(0o755)
        if os.path.lexists(destination):
            raise ArchiveError(f"base destination was created while staging: {destination}")
        staging.rename(destination)
        staging = None
        return destination / "python"
    except Exception as exc:
        raise ArchiveError(f"base archive extraction failed: {exc}") from exc
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _reconstruct_archive(parts: tuple[Path, ...], destination: Path) -> Path:
    archive_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".7z", dir=destination.parent, delete=False
        ) as archive:
            archive_path = Path(archive.name)
            for part in parts:
                part = Path(part)
                if not part.is_file():
                    raise ArchiveError(f"base archive part is not a regular file: {part}")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, archive, length=_CHUNK_SIZE)
    except BaseException:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def _case_insensitive(directory: Path) -> bool:
    with tempfile.NamedTemporaryFile(prefix=".case-A-", dir=directory) as probe:
        path = Path(probe.name)
        return path.with_name(path.name.swapcase()).exists()


def _validate_members(archive: py7zr.SevenZipFile, *, case_insensitive: bool) -> None:
    names: set[str] = set()
    spellings: dict[str, str] = {}
    top_levels: set[str] = set()
    for member in archive.files:
        name = member.filename
        path = _member_path(name)
        attributes = member.file_properties().get("attributes") or 0
        # Native 7-Zip honors link bits even without py7zr's Unix-extension marker.
        if (
            member.is_symlink
            or member.is_junction
            or stat.S_ISLNK(attributes >> 16)
            or attributes & 0x400
        ):
            raise ArchiveError(f"archive links are not supported: {name!r}")
        normalized = path.as_posix()
        if case_insensitive:
            normalized = normalized.casefold()
        if normalized in names:
            raise ArchiveError(f"archive contains ambiguous duplicate path: {name!r}")
        names.add(normalized)
        if case_insensitive:
            for end in range(1, len(path.parts) + 1):
                prefix = "/".join(path.parts[:end])
                if spellings.setdefault(prefix.casefold(), prefix) != prefix:
                    raise ArchiveError(f"archive contains case-ambiguous path: {name!r}")
        top_levels.add(path.parts[0])
    if top_levels - _ALLOWED_TREES:
        unknown = sorted(top_levels - _ALLOWED_TREES)
        raise ArchiveError(f"archive contains unknown top-level tree: {unknown}")
    if top_levels != set(_ALLOWED_TREES):
        raise ArchiveError("archive must contain both python/ and wheels/ trees")


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
        if path.is_symlink():
            raise ArchiveError(f"archive created a symbolic link: {path.relative_to(staging)}")
        try:
            path.resolve(strict=True).relative_to(staging.resolve())
        except ValueError as exc:
            relative = path.relative_to(staging)
            raise ArchiveError(f"archive member escaped staging: {relative}") from exc
        relative = path.relative_to(staging)
        if relative.parts[0] not in _ALLOWED_TREES:
            raise ArchiveError(f"archive created an unknown top-level tree: {relative.parts[0]}")
        if not path.is_file() and not path.is_dir():
            raise ArchiveError(f"archive created an unsupported member: {relative}")
    if not (staging / "python").is_dir() or not (staging / "wheels").is_dir():
        raise ArchiveError("archive must extract both python/ and wheels/ directories")
