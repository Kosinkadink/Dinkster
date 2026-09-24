"""Host authority for untrusted ComfyUI saved-output reports."""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import contextlib
import os
import stat
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from dinkster_assets import (
    KIND_MEDIA_AUDIO,
    KIND_MEDIA_IMAGE,
    KIND_MEDIA_VIDEO,
    MOUNT_NAMESPACE,
    AssetError,
    MountSnapshotWriter,
    classify_media_handle,
)
from dinkster_assets.identity import new_hasher
from dinkster_assets.integrity import verification_record
from dinkster_assets.library import append_write_record
from dinkster_protocol import SavedArtifact, SavedArtifactCandidate
from dinkster_values import SAVED_ARTIFACT_LIMIT_BYTES

MAX_SAVED_ARTIFACTS = 64
MAX_SAVED_ARTIFACT_BYTES = SAVED_ARTIFACT_LIMIT_BYTES
MAX_ARTIFACT_NODE_ID = 512
MAX_ARTIFACT_FILENAME = 255
MAX_ARTIFACT_SUBFOLDER = 1024
MAX_ARTIFACT_FOLDER_TYPE = 16
_SUPPORTED_KINDS = frozenset((KIND_MEDIA_IMAGE, KIND_MEDIA_AUDIO, KIND_MEDIA_VIDEO))

# Fields compared between two fstat calls on the same open handle.  Both
# calls read through the same fd, so every identity field is compared
# exactly on every platform.
_HANDLE_STABLE_FIELDS: tuple[str, ...] = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

# Fields compared between the after-fstat (handle fd) and the path-based
# stat.  On Windows, CPython maps st_ctime to different semantic fields
# depending on the call: os.stat(path) exposes CreationTime (birth time)
# for legacy compatibility, while os.fstat(fd) exposes ChangeTime
# (metadata-change time).  For a freshly written file these differ by
# ~1ms (the time between creation and write completion), so st_ctime_ns
# is excluded from the handle-vs-path comparison on Windows.  POSIX
# reads the same inode through both calls, so all fields are compared.
_PATH_STABLE_FIELDS: tuple[str, ...] = (
    _HANDLE_STABLE_FIELDS if os.name != "nt" else ("st_dev", "st_ino", "st_size", "st_mtime_ns")
)


def _bounded_text(value: object, field: str, limit: int, *, empty: bool = False) -> str:
    if type(value) is not str or (not value and not empty) or len(value) > limit:
        qualifier = "possibly empty" if empty else "non-empty"
        raise AssetError(f"saved artifact {field} must be a {qualifier} string of at most {limit}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AssetError(f"saved artifact {field} contains a control character")
    return value


def validate_candidate(candidate: SavedArtifactCandidate) -> tuple[str, tuple[str, ...]]:
    """Validate one raw report without touching the filesystem."""
    _bounded_text(candidate.node_id, "nodeId", MAX_ARTIFACT_NODE_ID)
    filename = _bounded_text(candidate.filename, "filename", MAX_ARTIFACT_FILENAME)
    subfolder = _bounded_text(candidate.subfolder, "subfolder", MAX_ARTIFACT_SUBFOLDER, empty=True)
    folder_type = _bounded_text(candidate.folder_type, "type", MAX_ARTIFACT_FOLDER_TYPE)
    if folder_type != "output":
        raise AssetError("saved artifact type must be 'output'")
    if filename in (".", "..") or "/" in filename or "\\" in filename:
        raise AssetError("saved artifact filename must be a safe basename")
    if "\\" in subfolder:
        raise AssetError("saved artifact subfolder must use '/' separators")
    parts = tuple(subfolder.split("/")) if subfolder else ()
    if any(part in ("", ".", "..") for part in parts):
        raise AssetError("saved artifact subfolder contains an unsafe path segment")
    return filename, parts


def _is_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction())


@contextmanager
def _open_candidate(
    root: Path, parts: tuple[str, ...], filename: str
) -> Generator[tuple[BinaryIO, Callable[[], os.stat_result]]]:
    """Open without following any candidate-controlled link component."""
    if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptors: list[int] = [os.open(root, directory_flags)]
        try:
            for part in parts:
                descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
            descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptors[-1])
            with os.fdopen(descriptor, "rb") as handle:
                yield (
                    handle,
                    lambda: os.stat(filename, dir_fd=descriptors[-1], follow_symlinks=False),
                )
        finally:
            for descriptor in reversed(descriptors):
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        return

    candidate = root.joinpath(*parts, filename)
    current = root
    for part in (*parts, filename):
        current /= part
        if _is_link(current):
            raise AssetError("saved artifact path contains a symlink or junction")
    resolved = candidate.resolve(strict=True)
    if resolved != root and not resolved.is_relative_to(root):
        raise AssetError("saved artifact path escapes the comfy-output mount")
    with candidate.open("rb") as handle:
        yield handle, lambda: os.stat(candidate, follow_symlinks=False)


def _digest_handle(handle: BinaryIO) -> str:
    handle.seek(0)
    hasher = new_hasher()
    while chunk := handle.read(MEBIBYTE):
        hasher.update(chunk)
    return "blake3:" + hasher.hexdigest()


class SavedArtifactAuthority:
    """Validate worker reports against one host-configured mount snapshot."""

    def __init__(self, mounts_snapshot: Path | str) -> None:
        self._snapshot = Path(mounts_snapshot)

    def capture(
        self,
        candidates: Sequence[SavedArtifactCandidate],
        node_id: str,
        started_at_ns: int,
    ) -> tuple[SavedArtifact, ...]:
        if len(candidates) > MAX_SAVED_ARTIFACTS:
            raise AssetError(f"saved artifact candidate count exceeds {MAX_SAVED_ARTIFACTS}")
        root = (
            MountSnapshotWriter(self._snapshot).writable_root("comfy-output").resolve(strict=True)
        )
        artifacts: list[SavedArtifact] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            if candidate.node_id != node_id:
                raise AssetError("saved artifact nodeId does not match the invocation")
            filename, parts = validate_candidate(candidate)
            relative = Path(*parts, filename)
            relative_text = relative.as_posix()
            identity = (candidate.node_id, relative_text)
            if identity in seen:
                continue
            try:
                with _open_candidate(root, parts, filename) as (handle, restat_path):
                    before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise AssetError("saved artifact is not a regular file")
                    if before.st_nlink != 1:
                        raise AssetError("saved artifact must not be a hard link")
                    if before.st_size <= 0:
                        raise AssetError("saved artifact is empty")
                    if before.st_size > MAX_SAVED_ARTIFACT_BYTES:
                        raise AssetError("saved artifact exceeds the 1 GiB limit")
                    if before.st_mtime_ns < started_at_ns:
                        raise AssetError("saved artifact is stale")
                    classification = classify_media_handle(handle)
                    if classification.kind not in _SUPPORTED_KINDS:
                        raise AssetError("saved artifact is not supported media")
                    digest = _digest_handle(handle)
                    after = os.fstat(handle.fileno())
                    path_stat = restat_path()
            except AssetError:
                raise
            except OSError as exc:
                raise AssetError(f"saved artifact could not be opened safely: {exc}") from exc
            if any(
                getattr(before, field) != getattr(after, field) for field in _HANDLE_STABLE_FIELDS
            ):
                raise AssetError("saved artifact changed while it was being indexed")
            if any(
                getattr(after, field) != getattr(path_stat, field) for field in _PATH_STABLE_FIELDS
            ):
                raise AssetError("saved artifact path changed while it was being indexed")
            append_write_record(
                root,
                relative_text,
                digest,
                before.st_size,
                before.st_mtime_ns,
                verification_record(digest, before),
            )
            artifacts.append(
                SavedArtifact(
                    node_id=node_id,
                    digest=digest,
                    name=filename,
                    size=before.st_size,
                    media_type=classification.media_type,
                    virtual_path=f"{MOUNT_NAMESPACE}/comfy-output/{relative_text}",
                )
            )
            seen.add(identity)
        return tuple(artifacts)


__all__ = [
    "MAX_ARTIFACT_FILENAME",
    "MAX_ARTIFACT_FOLDER_TYPE",
    "MAX_ARTIFACT_NODE_ID",
    "MAX_ARTIFACT_SUBFOLDER",
    "MAX_SAVED_ARTIFACTS",
    "SavedArtifactAuthority",
    "validate_candidate",
]
