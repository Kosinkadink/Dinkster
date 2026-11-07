"""Resolve content identities independently of readable local folder names."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import cast

_IDENTITY = re.compile(r"(?:sha256|blake3):[0-9a-f]{64}")
_COMPONENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._+-]{0,95}")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?", re.IGNORECASE)
_INDEX = ".objects.json"
_JOURNAL = ".objects.pending.json"


class StorePathError(ValueError):
    """A store allocation cannot be read or published safely."""


def readable_label(*components: str) -> str:
    """Normalize display metadata into bounded, portable path components."""
    labels: list[str] = []
    for component in components:
        label = re.sub(r"[^a-zA-Z0-9._+-]+", "-", component).strip(".-+")[:80].rstrip(".")
        if not label:
            label = "unknown"
        if _RESERVED.fullmatch(label):
            label = "object-" + label
        labels.append(label)
    return relative_path("/".join(labels))


def relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise StorePathError("store paths must be bounded portable relative paths")
    parts = value.split("/")
    if any(
        _COMPONENT.fullmatch(part) is None or part.endswith(".") or _RESERVED.fullmatch(part)
        for part in parts
    ):
        raise StorePathError("store paths must be bounded portable relative paths")
    return value


def _identity(value: str) -> str:
    if _IDENTITY.fullmatch(value) is None:
        raise StorePathError("store objects require a full sha256 or blake3 identity")
    return value


def _unlinked(root: Path, relative: str = "") -> Path:
    path = root
    if any(parent.is_symlink() or parent.is_junction() for parent in (root, *root.parents)):
        raise StorePathError("store paths cannot follow symbolic links or junctions")
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise StorePathError("store paths cannot follow symbolic links or junctions")
    return path


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".objects-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _lock(root: Path) -> Generator[None]:
    path = _unlinked(root, ".objects.lock")
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(root: Path) -> dict[str, str]:
    # The durable journal is the committed allocation if index publication was interrupted.
    journal = _unlinked(root, _JOURNAL)
    index = _unlinked(root, _INDEX)

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, value in pairs:
            if key in values:
                raise StorePathError("duplicate store identity metadata")
            values[key] = value
        return values

    try:
        try:
            data = journal.read_bytes()
        except FileNotFoundError:
            try:
                data = index.read_bytes()
            except FileNotFoundError:
                return {}
        value: object = json.loads(data, object_pairs_hook=unique)
    except (ValueError, OSError) as exc:
        raise StorePathError("cannot read store object identities") from exc
    if not isinstance(value, dict):
        raise StorePathError("invalid store object index")
    document = cast("dict[str, object]", value)
    objects = document.get("objects")
    if document.get("format") != "dinkster.store-paths/1" or not isinstance(objects, dict):
        raise StorePathError("invalid store object index")
    result: dict[str, str] = {}
    paths: list[str] = []
    for identity, relative in cast("dict[str, object]", objects).items():
        result[_identity(identity)] = relative_path(relative)
        paths.append(cast("str", relative).casefold())
    path_set = set(paths)
    if len(path_set) != len(paths):
        raise StorePathError("store object paths must be distinct and non-overlapping")
    for relative in paths:
        if any(str(parent) in path_set for parent in PurePosixPath(relative).parents):
            raise StorePathError("store object paths must be distinct and non-overlapping")
    return result


def _publish(root: Path, objects: Mapping[str, str]) -> None:
    data = (
        json.dumps({"format": "dinkster.store-paths/1", "objects": objects}, sort_keys=True) + "\n"
    ).encode("ascii")
    _write(_unlinked(root, _JOURNAL), data)
    _write(_unlinked(root, _INDEX), data)
    _unlinked(root, _JOURNAL).unlink()
    _sync_directory(root)


def object_path(
    root: Path,
    identity: str,
    *,
    label: str | None = None,
    legacy: str | None = None,
) -> Path:
    """Resolve without writes, or reserve a label before downloading under the caller's lock.

    A reservation identifies a location, not verified or complete content. Acquisition
    must still verify its bytes. Missing content never releases its allocated name.
    """
    _identity(identity)
    old = relative_path(legacy if legacy is not None else identity.split(":", 1)[1])

    def existing(objects: Mapping[str, str]) -> Path | None:
        if identity in objects:
            return _unlinked(root, objects[identity])
        path = _unlinked(root, old)
        return path if path.exists() else None

    _unlinked(root)
    if label is None:
        return existing(_read(root)) or _unlinked(root, old)
    relative_path(label)
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        objects = _read(root)
        path = existing(objects)
        if path is not None:
            if _unlinked(root, _JOURNAL).exists():
                _publish(root, objects)
            return path
        used = {value.casefold() for value in objects.values()}
        if any(label.casefold().startswith(value + "/") for value in used):
            raise StorePathError("store label parent is already a content object")
        for parent in PurePosixPath(label).parents:
            candidate_parent = _unlinked(root, str(parent))
            if candidate_parent.exists() and not candidate_parent.is_dir():
                raise StorePathError("store label parent is not a directory")
        candidate = label
        number = 1
        while True:
            path = _unlinked(root, candidate)
            overlaps = any(
                candidate.casefold() == value
                or candidate.casefold().startswith(value + "/")
                or value.startswith(candidate.casefold() + "/")
                for value in used
            )
            occupied = path.exists() or (
                path.parent.exists()
                and any(
                    child.name.casefold() == path.name.casefold() for child in path.parent.iterdir()
                )
            )
            if not overlaps and not occupied:
                break
            number += 1
            parent, _, name = label.rpartition("/")
            suffix = f"--{number}"
            name = name[: 96 - len(suffix)] + suffix
            candidate = f"{parent}/{name}" if parent else name
        objects[identity] = candidate
        _publish(root, objects)
        return path


def object_entries(
    root: Path, *, algorithm: str = "sha256", legacy_suffix: str = ""
) -> tuple[tuple[str, Path], ...]:
    """List existing indexed objects and legacy digest entries, not namespace parents."""
    _unlinked(root)
    objects = _read(root)
    result = {
        identity: _unlinked(root, relative)
        for identity, relative in objects.items()
        if _unlinked(root, relative).exists()
    }
    if root.exists():
        for path in root.iterdir():
            name = path.name.removesuffix(legacy_suffix) if legacy_suffix else path.name
            if re.fullmatch(r"[0-9a-f]{64}", name):
                identity = _identity(f"{algorithm}:{name}")
                if identity in objects:
                    if path != root / objects[identity]:
                        raise StorePathError("store identity has both indexed and legacy content")
                else:
                    result[identity] = _unlinked(root, path.name)
    return tuple(sorted(result.items()))
