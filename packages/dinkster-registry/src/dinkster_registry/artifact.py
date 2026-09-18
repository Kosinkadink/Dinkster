"""Pack artifacts: deterministic archives with content identity.

The artifact is the unit the whole distribution pipeline shares: the
publisher builds it, the registry digests and probes it (the admission
verdict binds to this digest), mirrors verify it, and the manager unpacks
it into the content-addressed store. One byte format, one digest,
computable identically everywhere.

Determinism rules (so "same tree -> same digest" holds across machines,
filesystems, and rebuild times):

- entries are the pack directory's files, sorted by POSIX relative path;
- timestamps are fixed to the zip epoch and permissions normalized -
  mtimes and umasks are machine noise, not pack content;
- no compression (``ZIP_STORED``): compressor output varies across zlib
  builds, and pack sources are small;
- junk is excluded (``__pycache__``, VCS dirs, venvs) and symlinks are
  refused - an artifact contains regular files only.

Unpacking verifies the digest BEFORE extracting and refuses entries that
would escape the destination (absolute paths, ``..`` traversal) - a
malicious archive fails loudly, never writes outside its store directory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from .model import artifact_digest, validate_artifact_digest

MANIFEST_FILENAME = "dinkster-pack.toml"

_SKIP_DIRS = frozenset({"__pycache__", ".git", ".hg", ".venv", "node_modules", ".tox"})
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_SYSTEM_UNIX = 3
_FILE_MODE = 0o644 << 16


class ArtifactError(Exception):
    """A pack artifact could not be built, verified, or unpacked."""


def _artifact_entries(pack_dir: Path) -> list[tuple[str, Path]]:
    """(posix relative path, file) pairs, sorted - the canonical order."""
    entries: list[tuple[str, Path]] = []
    stack = [pack_dir]
    while stack:
        directory = stack.pop()
        for child in directory.iterdir():
            if child.is_symlink():
                raise ArtifactError(
                    f"{child}: symlinks cannot enter an artifact; ship regular files"
                )
            if child.is_dir():
                if child.name not in _SKIP_DIRS:
                    stack.append(child)
            elif child.is_file():
                entries.append((child.relative_to(pack_dir).as_posix(), child))
    entries.sort(key=lambda entry: entry[0])
    return entries


def build_artifact(pack_dir: Path, out_path: Path) -> str:
    """Archive ``pack_dir`` deterministically; returns the artifact digest.

    The directory must contain a pack manifest - an artifact without one
    could never pass admission or install, so it fails at build instead.
    """
    if not pack_dir.is_dir():
        raise ArtifactError(f"not a pack directory: {pack_dir}")
    if not (pack_dir / MANIFEST_FILENAME).is_file():
        raise ArtifactError(f"{pack_dir}: no {MANIFEST_FILENAME}; not a pack")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative, file in _artifact_entries(pack_dir):
            info = zipfile.ZipInfo(filename=relative, date_time=_ZIP_EPOCH)
            info.create_system = _ZIP_SYSTEM_UNIX
            info.external_attr = _FILE_MODE
            archive.writestr(info, file.read_bytes())
    return artifact_digest(out_path.read_bytes())


def verify_artifact(archive_path: Path, expected_digest: str) -> None:
    """Check the archive bytes against the expected digest."""
    problem = validate_artifact_digest(expected_digest)
    if problem is not None:
        raise ArtifactError(f"expected digest {expected_digest!r} {problem}")
    actual = artifact_digest(archive_path.read_bytes())
    if actual != expected_digest:
        raise ArtifactError(
            f"{archive_path}: digest mismatch - expected {expected_digest}, "
            f"got {actual}; the bytes are not the published artifact"
        )


def unpack_artifact(archive_path: Path, dest: Path, expected_digest: str) -> Path:
    """Verify, then extract into ``dest``. Returns ``dest``.

    Every entry must resolve inside ``dest``; absolute paths and ``..``
    traversal are refused before a single byte is written.
    """
    verify_artifact(archive_path, expected_digest)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ArtifactError(f"{archive_path}: duplicate entry names; not a canonical artifact")
        for name in names:
            target = dest_resolved / name
            if not target.resolve().is_relative_to(dest_resolved):
                raise ArtifactError(
                    f"{archive_path}: entry {name!r} escapes the destination; refusing to unpack"
                )
        for name in sorted(names):
            target = dest_resolved / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return dest


__all__ = [
    "MANIFEST_FILENAME",
    "ArtifactError",
    "build_artifact",
    "unpack_artifact",
    "verify_artifact",
]
