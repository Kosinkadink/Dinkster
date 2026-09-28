"""Build registry-compatible deterministic pack archives."""

from __future__ import annotations

import io
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

from blake3 import blake3
from dinkster_values import MEBIBYTE
from dinkster_workers import load_manifest

MANIFEST_FILENAME = "dinkster-pack.toml"

MAX_ARCHIVE_BYTES = 64 * MEBIBYTE
MAX_FILE_BYTES = 64 * MEBIBYTE
MAX_EXPANDED_BYTES = 256 * MEBIBYTE
MAX_ENTRIES = 10_000
MAX_PATH_BYTES = 512
MAX_MANIFEST_BYTES = MEBIBYTE

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_SYSTEM_UNIX = 3
_FILE_MODE = 0o644 << 16


class PackArchiveError(Exception):
    """A pack tree cannot be represented as a registry archive."""


def _enforce_limit(actual: int, maximum: int, subject: str) -> None:
    if actual > maximum:
        raise PackArchiveError(f"{subject} is {actual} bytes; limit is {maximum} bytes")


def _collect_entries(pack_root: Path) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    folded_paths: set[str] = set()
    for path in pack_root.rglob("*"):
        raw_parts = path.relative_to(pack_root).parts
        if any(part in _SKIP_DIRS for part in raw_parts):
            continue
        if path.is_symlink():
            raise PackArchiveError(f"{path}: symlinks are not allowed in pack archives")
        if not path.is_file():
            continue

        relative = unicodedata.normalize("NFC", path.relative_to(pack_root).as_posix())
        if relative.startswith("/") or "\0" in relative or ".." in PurePosixPath(relative).parts:
            raise PackArchiveError(f"{path}: unsafe archive path {relative!r}")
        _enforce_limit(len(relative.encode("utf-8")), MAX_PATH_BYTES, f"archive path {relative!r}")

        folded = relative.casefold()
        if folded in folded_paths:
            raise PackArchiveError(
                f"{path}: archive path {relative!r} collides after Unicode normalization"
            )
        folded_paths.add(folded)

        payload = path.read_bytes()
        _enforce_limit(len(payload), MAX_FILE_BYTES, f"file {relative!r}")
        if relative == MANIFEST_FILENAME:
            _enforce_limit(len(payload), MAX_MANIFEST_BYTES, "pack manifest")
        entries.append((relative, payload))

    entries.sort(key=lambda item: item[0])
    if len(entries) > MAX_ENTRIES:
        raise PackArchiveError(f"archive has {len(entries)} entries; limit is {MAX_ENTRIES}")
    expanded = sum(len(payload) for _, payload in entries)
    _enforce_limit(expanded, MAX_EXPANDED_BYTES, "expanded archive")
    return entries


def _archive_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for relative, payload in entries:
            info = zipfile.ZipInfo(relative, date_time=_ZIP_EPOCH)
            info.create_system = _ZIP_SYSTEM_UNIX
            info.external_attr = _FILE_MODE
            info.compress_type = zipfile.ZIP_STORED
            info.extra = b""
            info.comment = b""
            archive.writestr(info, payload)
    result = output.getvalue()
    _enforce_limit(len(result), MAX_ARCHIVE_BYTES, "archive")
    return result


def build_pack_archive(pack_root: Path | str, output: Path | str) -> str:
    """Validate and archive a pack, returning its ``blake3:<hex>`` digest."""
    root = Path(pack_root)
    destination = Path(output)
    if not root.is_dir():
        raise PackArchiveError(f"not a pack directory: {root}")
    manifest = root / MANIFEST_FILENAME
    if manifest.is_symlink():
        raise PackArchiveError(f"{manifest}: symlinks are not allowed in pack archives")
    if not manifest.is_file():
        raise PackArchiveError(f"{root}: no {MANIFEST_FILENAME}; not a pack")
    _enforce_limit(manifest.stat().st_size, MAX_MANIFEST_BYTES, "pack manifest")
    load_manifest(manifest)

    archive = _archive_bytes(_collect_entries(root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(archive)
    return f"blake3:{blake3(archive).hexdigest()}"


__all__ = [
    "MANIFEST_FILENAME",
    "MAX_ARCHIVE_BYTES",
    "MAX_ENTRIES",
    "MAX_EXPANDED_BYTES",
    "MAX_FILE_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_PATH_BYTES",
    "PackArchiveError",
    "build_pack_archive",
]
