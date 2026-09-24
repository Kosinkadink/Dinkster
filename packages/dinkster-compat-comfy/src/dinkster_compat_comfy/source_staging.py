"""Invocation-owned staging of verified media for Comfy source filenames."""

from __future__ import annotations
from dinkster_values import GIBIBYTE

import contextlib
import os
import secrets
import stat
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

from dinkster_assets import AssetError, AssetRef, classify_media, digest_bytes

from .bootstrap import initialize_comfy_paths

_CATEGORIES = frozenset(("input", "output", "temp"))
_MAX_SOURCE_BYTES = GIBIBYTE
_HEX = frozenset("0123456789abcdef")
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


class SourceStagingError(AssetError):
    pass


def _supports_confined_staging() -> bool:
    return os.name == "posix"


class _MediaAuthority(Protocol):
    digest: str
    kind: str
    media_type: str
    extension: str
    byte_size: int


def _lock(handle: BinaryIO, *, blocking: bool) -> bool:
    if os.name != "posix":
        raise SourceStagingError("source staging owner locks require POSIX flock")
    import fcntl

    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(handle.fileno(), flags)
    except BlockingIOError:
        return False
    return True


def _lexical_roots(folder_paths: Any) -> dict[str, Path]:
    return {
        "input": Path(os.path.abspath(folder_paths.get_input_directory())),
        "output": Path(os.path.abspath(folder_paths.get_output_directory())),
        "temp": Path(os.path.abspath(folder_paths.get_temp_directory())),
    }


def _roots(folder_paths: Any) -> dict[str, Path]:
    roots = _lexical_roots(folder_paths)
    identities: set[tuple[int, int]] = set()
    for root in roots.values():
        try:
            descriptor = _open_root(root)
        except SourceStagingError as exc:
            raise SourceStagingError(
                "Comfy source category roots must be real non-symlink directories"
            ) from exc
        try:
            details = os.fstat(descriptor)
            identity = (details.st_dev, details.st_ino)
            if identity in identities:
                raise SourceStagingError("Comfy source category roots must be distinct")
            identities.add(identity)
        finally:
            os.close(descriptor)
    return roots


def _read_bounded(handle: BinaryIO, limit: int) -> bytes:
    data = handle.read(limit + 1)
    if len(data) > limit:
        raise SourceStagingError("media source exceeds the bounded staging limit")
    return data


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_root(root: Path) -> int:
    try:
        before = os.stat(root, follow_symlinks=False)
        descriptor = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise SourceStagingError(f"could not securely open source category root: {exc}") from exc
    if not stat.S_ISDIR(before.st_mode) or not _same_inode(before, os.fstat(descriptor)):
        os.close(descriptor)
        raise SourceStagingError("source category root changed while it was opened")
    return descriptor


def _open_child_directory(parent: int, name: str) -> int:
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    if not stat.S_ISDIR(before.st_mode) or not _same_inode(before, os.fstat(descriptor)):
        os.close(descriptor)
        raise SourceStagingError("source staging directory changed while it was opened")
    return descriptor


def _ensure_staging_root(root: int) -> int:
    try:
        os.mkdir(".dinkster-source", mode=0o700, dir_fd=root)
    except FileExistsError:
        pass
    try:
        return _open_child_directory(root, ".dinkster-source")
    except OSError as exc:
        raise SourceStagingError(
            "source staging root must not be a symlink and must be a real directory"
        ) from exc


def _bind_removal(parent: int, name: str, descriptor: int) -> str:
    """Move the currently named directory aside only if it is the pinned directory."""
    removal = f".remove-{os.getpid()}-{secrets.token_hex(16)}"
    os.rename(name, removal, src_dir_fd=parent, dst_dir_fd=parent)
    moved = os.stat(removal, dir_fd=parent, follow_symlinks=False)
    if not _same_inode(moved, os.fstat(descriptor)):
        raise SourceStagingError("source staging directory was rebound during cleanup")
    return removal


def _finish_removal(parent: int, removal: str, descriptor: int) -> None:
    moved = os.stat(removal, dir_fd=parent, follow_symlinks=False)
    if not _same_inode(moved, os.fstat(descriptor)):
        raise SourceStagingError("source staging tombstone was rebound during cleanup")
    os.rmdir(removal, dir_fd=parent)


def _is_staged_name(name: str) -> bool:
    stem, separator, extension = name.partition(".")
    return (
        bool(separator)
        and len(stem) == 64
        and all(character in _HEX for character in stem)
        and extension in {"png", "jpg", "webp", "wav", "flac", "mp3", "ogg", "mp4", "webm"}
    )


def _is_session_name(name: str) -> bool:
    return len(name) == 32 and all(character in _HEX for character in name)


def _owned_transition_pid(name: str, prefix: str, identity: bool) -> int | None:
    parts = name.split("-")
    if (
        len(parts) != (4 if identity else 3)
        or parts[0] != prefix
        or not parts[1].isascii()
        or not parts[1].isdigit()
        or len(parts[1]) > 20
    ):
        return None
    nonce = parts[3] if identity else parts[2]
    nonce_length = 16 if identity else 32
    if (
        (identity and not _is_session_name(parts[2]))
        or len(nonce) != nonce_length
        or any(character not in _HEX for character in nonce)
    ):
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    return pid if pid > 0 and str(pid) == parts[1] else None


def _pid_is_definitely_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError, OverflowError):
        return False
    return False


def _is_known_entry(name: str) -> bool:
    return name == ".owner" or name.startswith(".tmp-") or _is_staged_name(name)


class ComfySourceStagingProvider:
    def __init__(self, folder_paths: Any | None = None) -> None:
        self._folder_paths = folder_paths
        if self._folder_paths is None and os.environ.get("DINKSTER_COMFY_NATIVE_ONLY") != "1":
            self._folder_paths = initialize_comfy_paths()
        self._supported = _supports_confined_staging()
        self._roots: dict[str, Path] = {}
        if self._folder_paths is not None:
            self._roots = (
                _roots(self._folder_paths)
                if self._supported
                else _lexical_roots(self._folder_paths)
            )

    def sweep(self) -> None:
        if not self._supported:
            return
        for root_path in self._roots.values():
            root = _open_root(root_path)
            staging: int | None = None
            try:
                try:
                    staging = _open_child_directory(root, ".dinkster-source")
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise SourceStagingError(
                        "source staging root must not be a symlink and must be a real directory"
                    ) from exc
                for name in os.listdir(staging):
                    claim_pid = _owned_transition_pid(name, ".claim", True)
                    removal_pid = _owned_transition_pid(name, ".remove", False)
                    is_claim = claim_pid is not None
                    is_removal = removal_pid is not None
                    if not (_is_session_name(name) or is_claim or is_removal):
                        continue
                    if claim_pid is not None and not _pid_is_definitely_dead(claim_pid):
                        continue
                    if removal_pid is not None and not _pid_is_definitely_dead(removal_pid):
                        continue
                    try:
                        session = _open_child_directory(staging, name)
                    except (OSError, SourceStagingError):
                        continue
                    try:
                        try:
                            owner_fd = os.open(".owner", _FILE_FLAGS, dir_fd=session)
                        except FileNotFoundError:
                            if (is_claim or is_removal) and not os.listdir(session):
                                removal = _bind_removal(staging, name, session)
                                _finish_removal(staging, removal, session)
                            continue
                        except OSError:
                            continue
                        owner = os.fdopen(owner_fd, "rb", closefd=True)
                        with owner:
                            if not stat.S_ISREG(os.fstat(owner.fileno()).st_mode):
                                continue
                            if not _lock(owner, blocking=False):
                                continue
                            entries = tuple(os.listdir(session))
                            if any(not _is_known_entry(entry) for entry in entries):
                                continue
                            safe = True
                            for entry in entries:
                                try:
                                    details = os.stat(entry, dir_fd=session, follow_symlinks=False)
                                except OSError:
                                    safe = False
                                    break
                                if not stat.S_ISREG(details.st_mode):
                                    safe = False
                                    break
                            if not safe:
                                continue
                            removal = _bind_removal(staging, name, session)
                            for entry in entries:
                                if entry != ".owner":
                                    os.unlink(entry, dir_fd=session)
                            os.unlink(".owner", dir_fd=session)
                        _finish_removal(staging, removal, session)
                    except (OSError, SourceStagingError):
                        # A concurrently changing or draining session is not stale.
                        continue
                    finally:
                        os.close(session)
            finally:
                if staging is not None:
                    os.close(staging)
                os.close(root)

    def open(self, invocation_id: str, authorities: Sequence[object]) -> ComfySourceStagingSession:
        if not invocation_id:
            raise SourceStagingError("source staging requires an opaque invocation id")
        if self._folder_paths is None:
            raise SourceStagingError(
                "ComfyUI source filenames require --comfy-root; native nodes consume assets"
            )
        return ComfySourceStagingSession(
            self._folder_paths,
            self._roots,
            cast("tuple[_MediaAuthority, ...]", tuple(authorities)),
            secrets.token_hex(16),
            supported=self._supported,
        )


class _CategorySession:
    def __init__(
        self,
        root: int,
        staging: int,
        session: int,
        owner: BinaryIO,
    ) -> None:
        self.root = root
        self.staging = staging
        self.session = session
        self.owner = owner


class ComfySourceStagingSession:
    def __init__(
        self,
        folder_paths: Any,
        roots: Mapping[str, Path],
        authorities: tuple[_MediaAuthority, ...],
        session_id: str,
        *,
        supported: bool = True,
    ) -> None:
        self._folder_paths = folder_paths
        self._roots = dict(roots)
        self._authorities = {authority.digest: authority for authority in authorities}
        if len(self._authorities) != len(authorities):
            raise SourceStagingError("source authorities must have unique digests")
        self._session_id = session_id
        self._supported = supported
        self._sessions: dict[str, _CategorySession] = {}
        self._closed = False
        self._mutex = threading.RLock()

    def _category_session(self, category: str) -> _CategorySession:
        existing = self._sessions.get(category)
        if existing is not None:
            return existing
        root = _open_root(self._roots[category])
        staging: int | None = None
        claim: str | None = None
        owner: BinaryIO | None = None
        session: int | None = None
        try:
            staging = _ensure_staging_root(root)
            claim = f".claim-{os.getpid()}-{self._session_id}-{secrets.token_hex(8)}"
            os.mkdir(claim, mode=0o700, dir_fd=staging)
            claim_fd = _open_child_directory(staging, claim)
            try:
                owner_fd = os.open(
                    ".owner",
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=claim_fd,
                )
                owner = os.fdopen(owner_fd, "r+b", closefd=True)
                _lock(owner, blocking=True)
                os.rename(claim, self._session_id, src_dir_fd=staging, dst_dir_fd=staging)
                claim = None
                session = claim_fd
            except BaseException:
                os.close(claim_fd)
                raise
            opened = _CategorySession(root, staging, session, owner)
            self._sessions[category] = opened
            return opened
        except BaseException:
            if owner is not None:
                owner.close()
            if claim is not None and staging is not None:
                with contextlib.suppress(OSError):
                    claim_fd = _open_child_directory(staging, claim)
                    try:
                        for entry in os.listdir(claim_fd):
                            os.unlink(entry, dir_fd=claim_fd)
                    finally:
                        os.close(claim_fd)
                    os.rmdir(claim, dir_fd=staging)
            if session is not None:
                os.close(session)
            if staging is not None:
                os.close(staging)
            os.close(root)
            raise

    def _mounted_bytes(self, asset: AssetRef) -> bytes:
        path = asset.local_path()
        candidate = path if path.is_absolute() else Path.cwd() / path
        match = next(
            (root for root in self._roots.values() if candidate.is_relative_to(root)),
            None,
        )
        if match is None:
            raise SourceStagingError(
                "media source has no carried authority and is not operator-mounted"
            )
        parts = candidate.relative_to(match).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise SourceStagingError("operator mount source must be a file")
        descriptor = _open_root(match)
        directories: list[int] = [descriptor]
        try:
            for component in parts[:-1]:
                descriptor = _open_child_directory(descriptor, component)
                directories.append(descriptor)
            file_descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=descriptor)
            with os.fdopen(file_descriptor, "rb", closefd=True) as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise SourceStagingError("operator mount source must be a regular file")
                return _read_bounded(handle, _MAX_SOURCE_BYTES)
        except OSError as exc:
            raise SourceStagingError("operator mount source changed during staging") from exc
        finally:
            for directory in reversed(directories):
                os.close(directory)

    def _source_bytes(self, asset: AssetRef) -> bytes:
        authority = self._authorities.get(asset.digest)
        if authority is not None:
            if authority.byte_size > _MAX_SOURCE_BYTES:
                raise SourceStagingError("carried authority exceeds the bounded staging limit")
            with asset.open() as handle:
                data = _read_bounded(handle, authority.byte_size)
            if len(data) != authority.byte_size:
                raise SourceStagingError("media source byte size does not match carried authority")
            return data
        return self._mounted_bytes(asset)

    def materialize(self, asset: object, kind: str, category: str) -> str:
        with self._mutex:
            if not self._supported:
                raise SourceStagingError("Comfy source staging is unavailable on this platform")
            if self._closed:
                raise SourceStagingError("source staging session is closed")
            if not isinstance(asset, AssetRef):
                raise SourceStagingError("source staging accepts only AssetRef values")
            if kind not in {"media/image", "media/audio", "media/video"}:
                raise SourceStagingError("requested media kind is invalid")
            if category not in _CATEGORIES:
                raise SourceStagingError("source category must be input, output, or temp")
            data = self._source_bytes(asset)
            classification = classify_media(data)
            if digest_bytes(data) != asset.digest:
                raise SourceStagingError("media source digest changed during staging")
            if classification.kind != kind:
                raise SourceStagingError("media source kind does not match the requested kind")
            authority = self._authorities.get(asset.digest)
            if authority is not None and (
                classification.kind,
                classification.media_type,
                classification.extension,
                len(data),
            ) != (
                authority.kind,
                authority.media_type,
                authority.extension,
                authority.byte_size,
            ):
                raise SourceStagingError("media source bytes do not match carried authority")

            opened = self._category_session(category)
            filename = f"{asset.digest[7:]}.{classification.extension}"
            try:
                target_fd = os.open(filename, _FILE_FLAGS, dir_fd=opened.session)
            except FileNotFoundError:
                temp = f".tmp-{secrets.token_hex(16)}"
                temp_fd = os.open(
                    temp,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=opened.session,
                )
                try:
                    with os.fdopen(temp_fd, "wb", closefd=True) as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        os.link(
                            temp,
                            filename,
                            src_dir_fd=opened.session,
                            dst_dir_fd=opened.session,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        pass
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temp, dir_fd=opened.session)
                target_fd = os.open(filename, _FILE_FLAGS, dir_fd=opened.session)
            except OSError as exc:
                raise SourceStagingError("source staging target is unsafe") from exc
            with os.fdopen(target_fd, "rb", closefd=True) as target:
                target_details = os.fstat(target.fileno())
                if not stat.S_ISREG(target_details.st_mode):
                    raise SourceStagingError("source staging target is not a regular file")
                if _read_bounded(target, len(data)) != data:
                    raise SourceStagingError("source staging target collision changed bytes")
            if not _same_inode(
                os.stat(filename, dir_fd=opened.session, follow_symlinks=False),
                target_details,
            ):
                raise SourceStagingError("source staging target was rebound")

            relative = f".dinkster-source/{self._session_id}/{filename}"
            annotated = relative if category == "input" else f"{relative} [{category}]"
            round_trip = Path(self._folder_paths.get_annotated_filepath(annotated))
            expected = self._roots[category] / relative
            if round_trip != expected:
                raise SourceStagingError("Comfy source filename failed exact round-trip validation")
            if not _same_inode(
                os.stat(self._roots[category], follow_symlinks=False),
                os.fstat(opened.root),
            ):
                raise SourceStagingError("source category root was rebound during staging")
            if not _same_inode(
                os.stat(".dinkster-source", dir_fd=opened.root, follow_symlinks=False),
                os.fstat(opened.staging),
            ) or not _same_inode(
                os.stat(self._session_id, dir_fd=opened.staging, follow_symlinks=False),
                os.fstat(opened.session),
            ):
                raise SourceStagingError("source staging containment changed")
            return annotated

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            failures: list[Exception] = []
            for opened in tuple(self._sessions.values()):
                try:
                    entries = tuple(os.listdir(opened.session))
                    for entry in entries:
                        details = os.stat(entry, dir_fd=opened.session, follow_symlinks=False)
                        if not _is_known_entry(entry) or not stat.S_ISREG(details.st_mode):
                            raise SourceStagingError("source staging cleanup found an unsafe entry")
                    removal = _bind_removal(opened.staging, self._session_id, opened.session)
                    for entry in entries:
                        if entry != ".owner":
                            os.unlink(entry, dir_fd=opened.session)
                    os.unlink(".owner", dir_fd=opened.session)
                    _finish_removal(opened.staging, removal, opened.session)
                except Exception as exc:
                    failures.append(exc)
                finally:
                    opened.owner.close()
                    os.close(opened.session)
                    os.close(opened.staging)
                    os.close(opened.root)
            self._sessions.clear()
            if failures:
                raise SourceStagingError(f"could not clean source staging session: {failures[0]}")


def source_staging_provider() -> ComfySourceStagingProvider:
    return ComfySourceStagingProvider()
