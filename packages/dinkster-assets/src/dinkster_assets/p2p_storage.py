"""Confined resumable storage and publication for P2P asset transfers."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import json
import os
import stat
import struct
import sys
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from .identity import CHUNK_SIZE, AssetError, new_hasher, require_digest
from .integrity import AssetVerificationRecord, verification_record
from .p2p_descriptor import (
    P2PDescriptorResult,
    P2PDescriptorV1,
    derive_p2p_descriptor,
    validate_p2p_descriptor,
)

if os.name == "nt":  # pragma: no cover - exercised by Windows CI
    from . import p2p_windows as _p2p_windows
else:
    _p2p_windows = None

P2P_FORMAT_POLICY_VERSION = 1
P2P_PARTIAL_RETENTION_SECONDS = 7 * 24 * 60 * 60

_HEX = frozenset("0123456789abcdef")
_RESUME_VERSION = 1
_MAX_RESUME_BYTES = 16 * 1024 * 1024
_MAX_HEADER_BYTES = 100_000_000
_MAX_GGUF_ITEMS = 1_000_000
_MAX_GGUF_RANK = 4
_GGUF_ALIGNMENT = 32
_UINT64_MAX = (1 << 64) - 1
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_GGUF_VALUE_BYTES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_TYPES = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    8: (32, 34),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    30: (1, 2),
}


class P2PStorageError(AssetError):
    """P2P bytes or paths could not preserve the vault invariant."""


@dataclass(frozen=True)
class P2PStagingUsage:
    actual_bytes: int
    logical_bytes: int
    partials: int


@dataclass(frozen=True)
class P2PStagingPurge:
    removed_partials: int
    reclaimed_actual_bytes: int
    reclaimed_logical_bytes: int


@dataclass(frozen=True)
class P2PLocalFileMapping:
    """A no-copy local path whose verified identity can be revoked."""

    digest: str
    size: int
    path: Path
    format_policy_version: int
    verification: AssetVerificationRecord | None
    _fingerprint: tuple[int, ...]

    @property
    def fingerprint(self) -> tuple[int, ...]:
        return self._fingerprint

    def is_current(self) -> bool:
        if os.name == "nt":
            try:
                with _open_regular(self.path, writable=False) as handle:
                    current = os.fstat(handle.fileno())
                    _require_path_binding(self.path, current)
                    return _local_file_fingerprint(handle, current) == self._fingerprint
            except (OSError, P2PStorageError):
                return False
        try:
            current = os.lstat(self.path)
        except OSError:
            return False
        return (
            stat.S_ISREG(current.st_mode)
            and not _is_link_or_junction(self.path, current)
            and _stable_fingerprint(current) == self._fingerprint
        )

    def require_current(self) -> Path:
        if not self.is_current():
            raise P2PStorageError(f"P2P local file mapping is stale for {self.digest}")
        return self.path


@dataclass(frozen=True)
class _ResumeState:
    info_hash: str
    digest: str
    size: int
    completed: tuple[tuple[int, int], ...]
    updated_at_ns: int

    def to_json(self) -> dict[str, object]:
        return {
            "version": _RESUME_VERSION,
            "infoHash": self.info_hash,
            "digest": self.digest,
            "sizeBytes": self.size,
            "completedRanges": [list(row) for row in self.completed],
            "updatedAtNs": self.updated_at_ns,
        }


@dataclass(frozen=True)
class _ConfinedDirectory:
    path: Path
    descriptor: int | None
    device: int
    parent: _ConfinedDirectory | None = None
    name: str | None = None

    def require_bound(self) -> None:
        if self.parent is None or self.name is None:
            if self.descriptor is not None:
                if _p2p_windows is not None:
                    current_handle = _p2p_windows.open_root(self.path)
                    try:
                        current_identity = _p2p_windows.identity(current_handle)
                        opened_identity = _p2p_windows.identity(self.descriptor)
                    finally:
                        _p2p_windows.close(current_handle)
                    if current_identity != opened_identity:
                        raise P2PStorageError("P2P vault root was rebound")
                else:
                    current = os.lstat(self.path)
                    opened = os.fstat(self.descriptor)
                    if (current.st_dev, current.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise P2PStorageError("P2P vault root was rebound")
            return
        self.parent.require_bound()
        current = self.parent.stat_entry(self.name)
        opened_ino = (
            _p2p_windows.identity(self.descriptor)[1]
            if _p2p_windows is not None and self.descriptor is not None
            else (
                os.fstat(self.descriptor).st_ino
                if self.descriptor is not None
                else os.lstat(self.path).st_ino
            )
        )
        if not stat.S_ISDIR(current.st_mode) or current.st_ino != opened_ino:
            raise P2PStorageError("P2P storage directory was rebound")

    @contextmanager
    def child(self, name: str, *, create: bool) -> Generator[_ConfinedDirectory]:
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise P2PStorageError("P2P storage directory name is not canonical")
        child_path = self.path / name
        self.require_bound()
        if _p2p_windows is not None and self.descriptor is not None:
            try:
                descriptor = _p2p_windows.open_directory(self.descriptor, name, create=create)
            except OSError as error:
                raise P2PStorageError(f"P2P storage directory is not confined: {error}") from error
            try:
                yield _ConfinedDirectory(
                    child_path,
                    descriptor,
                    self.device,
                    self,
                    name,
                )
            finally:
                _p2p_windows.close(descriptor)
            return
        if self.descriptor is not None:
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=self.descriptor)
                except FileExistsError:
                    pass
            flags = os.O_RDONLY | _directory_flag() | _nofollow_flag() | _cloexec_flag()
            try:
                descriptor = os.open(name, flags, dir_fd=self.descriptor)
            except OSError as error:
                raise P2PStorageError(f"P2P storage directory is not confined: {error}") from error
            try:
                item = os.fstat(descriptor)
                if not stat.S_ISDIR(item.st_mode) or item.st_dev != self.device:
                    raise P2PStorageError("P2P staging and canonical vault must use one filesystem")
                yield _ConfinedDirectory(
                    child_path,
                    descriptor,
                    self.device,
                    self,
                    name,
                )
            finally:
                os.close(descriptor)
            return

        if create:
            child_path.mkdir(mode=0o700, exist_ok=True)
        item = os.lstat(child_path)
        if (
            not stat.S_ISDIR(item.st_mode)
            or _is_link_or_junction(child_path, item)
            or item.st_dev != self.device
        ):
            raise P2PStorageError("P2P storage directory is not confined")
        yield _ConfinedDirectory(child_path, None, self.device, self, name)

    def open_file(
        self,
        name: str,
        *,
        writable: bool,
        create_exclusive: bool = False,
        deny_other_writers: bool = False,
        delete_access: bool = False,
    ) -> BinaryIO:
        self.require_bound()
        if _p2p_windows is not None and self.descriptor is not None:
            raw_handle: int | None = None
            try:
                raw_handle = _p2p_windows.open_file(
                    self.descriptor,
                    name,
                    writable=writable,
                    exclusive=create_exclusive,
                    share_write=not deny_other_writers,
                    delete=delete_access,
                )
                flags = (os.O_RDWR if writable else os.O_RDONLY) | _binary_flag()
                descriptor = _p2p_windows.take_file_descriptor(raw_handle, flags)
                raw_handle = None
            except FileExistsError:
                raise
            except OSError as error:
                if error.errno in (errno.EISDIR, errno.ELOOP, errno.ENOTDIR):
                    raise P2PStorageError("P2P file must be regular non-symlink") from error
                with contextlib.suppress(OSError):
                    item = self.stat_entry(name)
                    if not stat.S_ISREG(item.st_mode):
                        raise P2PStorageError("P2P file must be regular non-symlink") from error
                    if item.st_dev != self.device:
                        raise P2PStorageError(
                            "P2P file must be regular and on the vault filesystem"
                        ) from error
                raise P2PStorageError(f"could not open confined P2P file: {error}") from error
            finally:
                if raw_handle is not None:
                    _p2p_windows.close(raw_handle)
            return os.fdopen(descriptor, "r+b" if writable else "rb")
        flags = (os.O_RDWR if writable else os.O_RDONLY) | _binary_flag() | _nofollow_flag()
        if create_exclusive:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            if self.descriptor is None:
                descriptor = os.open(self.path / name, flags, 0o600)
            else:
                descriptor = os.open(name, flags | _cloexec_flag(), 0o600, dir_fd=self.descriptor)
        except FileExistsError:
            raise
        except OSError as error:
            if error.errno in (errno.EISDIR, errno.ELOOP, errno.ENOTDIR):
                raise P2PStorageError("P2P file must be regular non-symlink") from error
            raise P2PStorageError(f"could not open confined P2P file: {error}") from error
        handle = os.fdopen(descriptor, "r+b" if writable else "rb")
        item = os.fstat(descriptor)
        if not stat.S_ISREG(item.st_mode) or item.st_dev != self.device:
            handle.close()
            raise P2PStorageError("P2P file must be regular and on the vault filesystem")
        if self.descriptor is None:
            _require_path_binding(self.path / name, item)
        return handle

    def stat_entry(self, name: str) -> os.stat_result:
        if _p2p_windows is not None and self.descriptor is not None:
            return _p2p_windows.stat_entry(self.descriptor, name)
        if self.descriptor is None:
            return os.lstat(self.path / name)
        return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)

    def allocated_bytes(self, name: str, item: os.stat_result) -> int:
        if _p2p_windows is not None and self.descriptor is not None:
            return _p2p_windows.allocated_size(self.descriptor, name)
        return _allocated_bytes(item)

    def unlink_entry(self, name: str, expected: os.stat_result | None = None) -> None:
        """Remove one confined name, refusing a stale expected identity.

        POSIX has no inode-conditional unlink; a concurrent substitution remains
        bounded to this already-writable directory.
        """
        self.require_bound()
        try:
            current = self.stat_entry(name)
        except FileNotFoundError:
            return
        if expected is not None and (current.st_dev, current.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise P2PStorageError("refusing to unlink a rebound P2P entry")
        if _p2p_windows is not None and self.descriptor is not None:
            handle = _p2p_windows.open_entry(
                self.descriptor,
                name,
                delete=True,
            )
            try:
                if expected is not None and _p2p_windows.identity(handle)[1] != expected.st_ino:
                    raise P2PStorageError("refusing to unlink a rebound P2P entry")
                _p2p_windows.delete(handle)
            finally:
                _p2p_windows.close(handle)
        elif self.descriptor is None:
            (self.path / name).unlink()
        else:
            os.unlink(name, dir_fd=self.descriptor)

    def replace(self, source: str, target: str) -> None:
        self.require_bound()
        if _p2p_windows is not None and self.descriptor is not None:
            handle = _p2p_windows.open_file(
                self.descriptor,
                source,
                writable=True,
                exclusive=False,
                delete=True,
            )
            try:
                _p2p_windows.replace(handle, self.descriptor, target)
            finally:
                _p2p_windows.close(handle)
        elif self.descriptor is None:
            os.replace(self.path / source, self.path / target)
        else:
            os.replace(
                source,
                target,
                src_dir_fd=self.descriptor,
                dst_dir_fd=self.descriptor,
            )

    def rename_to(
        self,
        source: str,
        target_dir: _ConfinedDirectory,
        target: str,
        opened_handle: BinaryIO,
    ) -> None:
        self.require_bound()
        target_dir.require_bound()
        if (
            _p2p_windows is not None
            and self.descriptor is not None
            and target_dir.descriptor is not None
        ):
            _p2p_windows.rename(
                _p2p_windows.raw_file_handle(opened_handle.fileno()),
                target_dir.descriptor,
                target,
                replace=False,
            )
        elif self.descriptor is None or target_dir.descriptor is None:
            raise P2PStorageError(
                "atomic no-replace P2P publication is unavailable on this platform"
            )
        else:
            _rename_noreplace(
                self.descriptor,
                source,
                target_dir.descriptor,
                target,
            )

    def fsync(self) -> None:
        if self.descriptor is None:
            return
        if _p2p_windows is not None:
            _p2p_windows.flush(self.descriptor)
        else:
            os.fsync(self.descriptor)

    def entry_names(self) -> list[str]:
        if _p2p_windows is not None and self.descriptor is not None:
            return _p2p_windows.entry_names(self.descriptor)
        return os.listdir(self.descriptor if self.descriptor is not None else self.path)

    def rmdir_child(self, name: str) -> None:
        self.require_bound()
        if _p2p_windows is not None and self.descriptor is not None:
            handle = _p2p_windows.open_directory(
                self.descriptor,
                name,
                create=False,
                delete=True,
            )
            try:
                _p2p_windows.delete(handle)
            finally:
                _p2p_windows.close(handle)
        elif self.descriptor is None:
            (self.path / name).rmdir()
        else:
            os.rmdir(name, dir_fd=self.descriptor)


@contextmanager
def _vault_directories(
    vault_root: Path,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> Generator[tuple[_ConfinedDirectory, ...]]:
    root_path = Path(os.path.abspath(vault_root))
    if os.name == "posix":
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(root_path, flags)
        except OSError as error:
            raise P2PStorageError(
                f"P2P vault root must be a regular non-symlink directory: {error}"
            ) from error
        root_item = os.fstat(descriptor)
        path_item = os.lstat(root_path)
        if not stat.S_ISDIR(root_item.st_mode) or (root_item.st_dev, root_item.st_ino) != (
            path_item.st_dev,
            path_item.st_ino,
        ):
            os.close(descriptor)
            raise P2PStorageError("P2P vault root must be a regular non-symlink directory")
        root = _ConfinedDirectory(root_path, descriptor, root_item.st_dev)
    elif _p2p_windows is not None:
        try:
            descriptor = _p2p_windows.open_root(root_path)
        except OSError as error:
            raise P2PStorageError(
                f"P2P vault root must be a regular non-symlink directory: {error}"
            ) from error
        root_item = os.lstat(root_path)
        root = _ConfinedDirectory(root_path, descriptor, root_item.st_dev)
    else:
        root_item = os.lstat(root_path)
        if not stat.S_ISDIR(root_item.st_mode) or _is_link_or_junction(root_path, root_item):
            raise P2PStorageError("P2P vault root must be a regular directory")
        descriptor = None
        root = _ConfinedDirectory(root_path, None, root_item.st_dev)
    try:
        with ExitStack() as stack:
            directories = [root]
            current = root
            for part in parts:
                current = stack.enter_context(current.child(part, create=create))
                directories.append(current)
            yield tuple(directories)
    finally:
        _close_optional_descriptor(descriptor)


def _close_optional_descriptor(descriptor: int | None) -> None:
    if descriptor is not None:
        if _p2p_windows is not None:
            _p2p_windows.close(descriptor)
        else:
            os.close(descriptor)


def _rename_noreplace(
    source_directory: int,
    source: str,
    target_directory: int,
    target: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = libc.renameat2
        flag = 1  # RENAME_NOREPLACE
    elif sys.platform == "darwin":
        rename = libc.renameatx_np
        flag = 4  # RENAME_EXCL
    else:
        raise P2PStorageError("atomic no-replace P2P publication is unavailable on this platform")
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    if (
        rename(
            source_directory,
            os.fsencode(source),
            target_directory,
            os.fsencode(target),
            flag,
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), source, target)


@contextmanager
def _lock_file(handle: BinaryIO, *, blocking: bool = True) -> Generator[bool]:
    descriptor = handle.fileno()
    if os.name == "posix":
        import fcntl

        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        return

    if _p2p_windows is None:
        raise P2PStorageError("P2P file locking is unavailable on this platform")
    lock = _p2p_windows.lock_file(descriptor, blocking=blocking)
    if lock is None:
        yield False
        return
    try:
        yield True
    finally:
        _p2p_windows.unlock_file(descriptor, lock)


@contextmanager
def _open_stable_file(
    directory: _ConfinedDirectory,
    name: str,
    *,
    publication_source: bool,
) -> Generator[BinaryIO]:
    deadline = time.monotonic() + 5
    while True:
        try:
            handle = directory.open_file(
                name,
                writable=publication_source,
                deny_other_writers=True,
                delete_access=publication_source,
            )
        except P2PStorageError as error:
            cause = error.__cause__
            if (
                _p2p_windows is None
                or not isinstance(cause, OSError)
                or getattr(cause, "winerror", None) != 32
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.01)
        else:
            with handle:
                yield handle
            return


class P2PPartial:
    """A sparse staged file with durable random-write resume ranges."""

    def __init__(
        self,
        vault_root: Path,
        path: Path,
        state_path: Path,
        info_hash: str,
        digest: str,
        size: int,
    ) -> None:
        self._vault_root = vault_root
        self.path = path
        self.resume_path = state_path
        self.info_hash = info_hash
        self.digest = digest
        self.size = size
        self._lock = threading.Lock()

    @property
    def completed_ranges(self) -> tuple[tuple[int, int], ...]:
        with _vault_directories(
            self._vault_root,
            (".p2p", "staging", self.info_hash),
            create=False,
        ) as directories:
            parent = directories[-1]
            with parent.open_file(self.digest[7:], writable=True) as handle:
                with _lock_file(handle) as locked:
                    if not locked:
                        raise P2PStorageError("could not lock the P2P partial")
                    _require_staged_stat(parent.device, os.fstat(handle.fileno()), self.size)
                    return _load_resume_state(
                        parent, self.info_hash, self.digest, self.size
                    ).completed

    @property
    def complete(self) -> bool:
        return self.completed_ranges == ((0, self.size),)

    def write_piece(self, offset: int, data: bytes) -> None:
        if type(offset) is not int or offset < 0:
            raise P2PStorageError("P2P piece offset must be a non-negative integer")
        if not data:
            raise P2PStorageError("P2P piece bytes must be non-empty")
        end = offset + len(data)
        if end > self.size:
            raise P2PStorageError("P2P piece extends beyond the declared asset size")

        with self._lock:
            with _vault_directories(
                self._vault_root,
                (".p2p", "staging", self.info_hash),
                create=False,
            ) as directories:
                parent = directories[-1]
                with parent.open_file(self.digest[7:], writable=True) as handle:
                    with _lock_file(handle) as locked:
                        if not locked:
                            raise P2PStorageError("could not lock the P2P partial")
                        state = _load_resume_state(parent, self.info_hash, self.digest, self.size)
                        before = os.fstat(handle.fileno())
                        _require_staged_stat(parent.device, before, self.size)
                        pwrite = cast(
                            "Callable[[int, bytes, int], int] | None",
                            getattr(os, "pwrite", None),
                        )
                        if pwrite is not None:
                            written = pwrite(handle.fileno(), data, offset)
                        else:  # pragma: no cover - Windows
                            handle.seek(offset)
                            written = handle.write(data)
                        if written != len(data):
                            raise P2PStorageError("P2P piece write was incomplete")
                        handle.flush()
                        os.fsync(handle.fileno())
                        after = os.fstat(handle.fileno())
                        if (after.st_dev, after.st_ino, after.st_size) != (
                            before.st_dev,
                            before.st_ino,
                            before.st_size,
                        ):
                            raise P2PStorageError(
                                "P2P staged file changed identity during a piece write"
                            )
                        _require_staged_stat(parent.device, after, self.size)
                        completed = _merge_ranges((*state.completed, (offset, end)))
                        _write_resume_state(
                            parent,
                            self.digest[7:],
                            _ResumeState(
                                self.info_hash,
                                self.digest,
                                self.size,
                                completed,
                                time.time_ns(),
                            ),
                        )


def p2p_staging_root(vault_root: Path) -> Path:
    with _vault_directories(vault_root, (".p2p", "staging"), create=True) as directories:
        return directories[-1].path


def p2p_staging_path(vault_root: Path, info_hash: str, digest: str) -> Path:
    info_hash = _require_hex(info_hash, "P2P info hash")
    digest_hex = require_digest(digest).split(":", 1)[1]
    with _vault_directories(vault_root, (".p2p", "staging", info_hash), create=True) as directories:
        return directories[-1].path / digest_hex


def open_p2p_partial(
    vault_root: Path,
    info_hash: str,
    digest: str,
    expected_size: int,
) -> P2PPartial:
    size = _require_size(expected_size)
    digest = require_digest(digest)
    info_hash = _require_hex(info_hash, "P2P info hash")
    digest_hex = digest[7:]
    with _vault_directories(vault_root, (".p2p", "staging", info_hash), create=True) as directories:
        parent = directories[-1]
        path = parent.path / digest_hex
        state_path = path.with_name(path.name + ".resume.json")
        created = False
        try:
            handle = parent.open_file(digest_hex, writable=True, create_exclusive=True)
        except FileExistsError:
            handle = parent.open_file(digest_hex, writable=True)
        else:
            created = True
        with handle:
            with _lock_file(handle) as locked:
                if not locked:
                    raise P2PStorageError("could not lock the P2P partial")
                item = os.fstat(handle.fileno())
                if created:
                    _initialize_partial(parent, handle, info_hash, digest, size)
                else:
                    _require_staged_identity(parent.device, item)
                    try:
                        _load_resume_state(parent, info_hash, digest, size)
                    except P2PStorageError as error:
                        if not isinstance(error.__cause__, FileNotFoundError):
                            raise
                        _initialize_partial(parent, handle, info_hash, digest, size)
                    else:
                        _require_staged_stat(parent.device, item, size)
        parent.fsync()
    return P2PPartial(vault_root, path, state_path, info_hash, digest, size)


def _initialize_partial(
    directory: _ConfinedDirectory,
    handle: BinaryIO,
    info_hash: str,
    digest: str,
    size: int,
) -> None:
    try:
        if _p2p_windows is not None:
            _p2p_windows.make_sparse(handle.fileno(), size)
        else:
            os.ftruncate(handle.fileno(), size)
        handle.flush()
        os.fsync(handle.fileno())
    except OSError as error:
        raise P2PStorageError(f"could not initialize sparse P2P partial: {error}") from error
    _write_resume_state(
        directory,
        digest[7:],
        _ResumeState(info_hash, digest, size, (), time.time_ns()),
    )


def adopt_staged_asset(
    vault_root: Path,
    target: Path,
    info_hash: str,
    digest: str,
    expected_size: int,
    staged_path: Path | str,
    format_policy_version: int,
) -> tuple[Path, AssetVerificationRecord | None]:
    vault_root = Path(os.path.abspath(vault_root))
    digest = require_digest(digest)
    digest_hex = digest[7:]
    target = Path(os.path.abspath(target))
    expected_target = vault_root / digest_hex[:2] / digest_hex
    if target != expected_target:
        raise P2PStorageError("P2P publication target is not the canonical vault path")
    size = _require_size(expected_size)
    _require_format_policy(format_policy_version)
    candidate = Path(staged_path)
    expected_info_hash = _require_hex(info_hash, "P2P info hash")
    if _staged_info_hash(vault_root, candidate, digest) != expected_info_hash:
        raise P2PStorageError("P2P staged path does not match the validated descriptor")

    with ExitStack() as stack:
        source_directories = stack.enter_context(
            _vault_directories(
                vault_root,
                (".p2p", "staging", expected_info_hash),
                create=False,
            )
        )
        target_directories = stack.enter_context(
            _vault_directories(vault_root, (digest_hex[:2],), create=True)
        )
        staging = source_directories[-1]
        canonical = target_directories[-1]
        with _open_stable_file(
            staging,
            digest_hex,
            publication_source=True,
        ) as handle:
            with _lock_file(handle) as locked:
                if not locked:
                    raise P2PStorageError("could not lock the P2P staged asset")
                before = os.fstat(handle.fileno())
                before_fingerprint = _local_file_fingerprint(handle, before)
                _require_staged_stat(staging.device, before, size)
                _validate_safe_format(handle, format_policy_version)
                actual = _hash_handle(handle)
                if actual != digest:
                    raise P2PStorageError(
                        f"staged asset did not verify: expected {digest}, bytes hash to {actual}"
                    )
                handle.flush()
                os.fsync(handle.fileno())
                after = os.fstat(handle.fileno())
                changed = _local_file_fingerprint(handle, after) != before_fingerprint
                if _p2p_windows is not None and not changed:
                    changed = _hash_handle(handle) != actual
                if changed:
                    raise P2PStorageError("P2P staged file changed during verification")

                try:
                    staging.rename_to(digest_hex, canonical, digest_hex, handle)
                except FileExistsError:
                    record = _verify_existing_target(
                        canonical,
                        digest_hex,
                        digest,
                        size,
                        format_policy_version,
                    )
                    staging.unlink_entry(digest_hex, after)
                except OSError as error:
                    raise P2PStorageError(
                        f"could not publish verified staged asset {digest}: {error}"
                    ) from error
                else:
                    published = canonical.stat_entry(digest_hex)
                    try:
                        current = os.fstat(handle.fileno())
                        if (
                            (published.st_dev, published.st_ino) != (after.st_dev, after.st_ino)
                            or current.st_nlink != 1
                            or (current.st_size, current.st_mtime_ns)
                            != (after.st_size, after.st_mtime_ns)
                        ):
                            raise P2PStorageError(
                                "canonical vault path was rebound during publication"
                            )
                        final = canonical.stat_entry(digest_hex)
                        current = os.fstat(handle.fileno())
                        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != (
                            after.st_dev,
                            after.st_ino,
                            after.st_size,
                            after.st_mtime_ns,
                        ) or current.st_nlink != 1:
                            raise P2PStorageError(
                                "published vault asset changed during atomic rename"
                            )
                    except BaseException:
                        if (published.st_dev, published.st_ino) == (after.st_dev, after.st_ino):
                            with contextlib.suppress(OSError, P2PStorageError):
                                canonical.unlink_entry(digest_hex, after)
                        raise
                    record = verification_record(digest, final)
        canonical.fsync()
        _remove_resume_state(staging, digest_hex)
        staging.fsync()
        return canonical.path / digest_hex, record


def verify_p2p_local_file(
    vault_root: Path,
    digest: str,
    expected_size: int,
    local_path: Path | str,
    format_policy_version: int,
) -> P2PLocalFileMapping:
    digest = require_digest(digest)
    size = _require_size(expected_size)
    _require_format_policy(format_policy_version)
    path = Path(local_path)
    if not path.is_absolute():
        raise P2PStorageError("P2P local file mapping requires an absolute path")
    cached = cached_p2p_local_file(vault_root, path)
    if cached is not None and (
        cached.digest == digest
        and cached.size == size
        and cached.format_policy_version == format_policy_version
    ):
        return cached
    verified_p2p_seed_descriptor(vault_root, digest, size, path)
    mapping = cached_p2p_local_file(vault_root, path)
    if mapping is None or mapping.digest != digest or mapping.size != size:
        raise P2PStorageError("P2P local file changed after verification")
    return mapping


def _verification_name(path: Path) -> str:
    hasher = new_hasher()
    hasher.update(os.fsencode(path))
    return hasher.hexdigest() + ".json"


def _read_local_verification(vault_root: Path, path: Path) -> dict[str, object]:
    try:
        with _vault_directories(vault_root, (".p2p", "verified-local"), create=False) as dirs:
            with dirs[-1].open_file(_verification_name(path), writable=False) as handle:
                data = handle.read(_MAX_RESUME_BYTES + 1)
                if len(data) > _MAX_RESUME_BYTES:
                    return {}
                value: object = json.loads(data)
        return cast(dict[str, object], value) if isinstance(value, dict) else {}
    except (OSError, P2PStorageError, ValueError):
        return {}


def cached_p2p_local_file(vault_root: Path, path: Path) -> P2PLocalFileMapping | None:
    """Reuse private local verification state, never provider-supplied metadata."""
    row = _read_local_verification(vault_root, path)
    if set(row) != {"version", "path", "digest", "size", "policy", "fingerprint", "descriptor"}:
        return None
    fingerprint = row["fingerprint"]
    if (
        type(row["version"]) is not int
        or row["version"] != 1
        or row["path"] != str(path)
        or not path.is_absolute()
        or not isinstance(row["digest"], str)
        or type(row["size"]) is not int
        or row["size"] <= 0
        or type(row["policy"]) is not int
        or row["policy"] != P2P_FORMAT_POLICY_VERSION
        or not isinstance(fingerprint, list)
        or len(cast(list[object], fingerprint)) not in (6, 7)
        or any(type(item) is not int for item in cast(list[object], fingerprint))
    ):
        return None
    try:
        digest = require_digest(row["digest"])
        with _open_regular(path, writable=False) as handle:
            current = os.fstat(handle.fileno())
            _require_path_binding(path, current)
            identity = _local_file_fingerprint(handle, current)
            if list(identity) != fingerprint or current.st_size != row["size"]:
                return None
        return P2PLocalFileMapping(
            digest,
            row["size"],
            path,
            P2P_FORMAT_POLICY_VERSION,
            verification_record(digest, current),
            identity,
        )
    except (AssetError, OSError):
        return None


def _save_local_verification(
    vault_root: Path,
    mapping: P2PLocalFileMapping,
    descriptor: P2PDescriptorResult | None = None,
) -> None:
    mapping.require_current()
    row = {
        "version": 1,
        "path": str(mapping.path),
        "digest": mapping.digest,
        "size": mapping.size,
        "policy": mapping.format_policy_version,
        "fingerprint": list(mapping.fingerprint),
        "descriptor": None
        if descriptor is None
        else {
            "descriptor": descriptor.descriptor.to_wire(),
            "info": descriptor.info.hex(),
            "pieceLayer": descriptor.piece_layer.hex(),
        },
    }
    with _vault_directories(vault_root, (".p2p", "verified-local"), create=True) as dirs:
        directory = dirs[-1]
        name = _verification_name(mapping.path)
        temporary = name + ".tmp-" + uuid.uuid4().hex
        try:
            with directory.open_file(temporary, writable=True, create_exclusive=True) as handle:
                handle.write(json.dumps(row, separators=(",", ":")).encode())
                handle.flush()
                os.fsync(handle.fileno())
            directory.replace(temporary, name)
            directory.fsync()
        finally:
            directory.unlink_entry(temporary)


def verified_p2p_seed_descriptor(
    vault_root: Path,
    digest: str | None,
    size: int,
    path: Path,
) -> P2PDescriptorResult:
    """Verify safe bytes and derive their identity in one scan, or reuse current state."""
    if digest is not None:
        digest = require_digest(digest)
    size = _require_size(size)
    if not path.is_absolute():
        raise P2PStorageError("P2P local file mapping requires an absolute path")
    mapping = cached_p2p_local_file(vault_root, path)
    row = _read_local_verification(vault_root, path)
    value = row.get("descriptor")
    if (
        mapping is not None
        and mapping.size == size
        and (digest is None or mapping.digest == digest)
        and isinstance(value, dict)
        and row.get("fingerprint") == list(mapping.fingerprint)
    ):
        material = cast(dict[str, object], value)
        try:
            if set(material) == {"descriptor", "info", "pieceLayer"}:
                wire = material["descriptor"]
                if (
                    isinstance(wire, dict)
                    and isinstance(material["info"], str)
                    and isinstance(material["pieceLayer"], str)
                ):
                    result = P2PDescriptorResult(
                        mapping.digest,
                        size,
                        P2PDescriptorV1.from_wire(cast(dict[str, object], wire)),
                        bytes.fromhex(material["info"]),
                        bytes.fromhex(material["pieceLayer"]),
                    )
                    validate_p2p_descriptor(
                        result.descriptor,
                        asset_digest=mapping.digest,
                        size=size,
                        info=result.info,
                        piece_layer=result.piece_layer,
                    )
                    mapping.require_current()
                    return result
        except (ValueError, AssetError):
            pass
    with _open_regular(path, writable=False) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise P2PStorageError("P2P local file must be a regular file of the declared size")
        _require_path_binding(path, before)
        before_fingerprint = _local_file_fingerprint(handle, before)
        _validate_safe_format(handle, P2P_FORMAT_POLICY_VERSION)
        result = derive_p2p_descriptor(path, handle=handle)
        if digest is not None and result.asset_digest != digest:
            raise P2PStorageError(
                f"P2P local file did not verify: expected {digest}, "
                f"bytes hash to {result.asset_digest}"
            )
        after = os.fstat(handle.fileno())
        fingerprint = _local_file_fingerprint(handle, after)
        if fingerprint != before_fingerprint or result.size != size:
            raise P2PStorageError("P2P local file changed during verification")
        _require_path_binding(path, after)
    mapping = P2PLocalFileMapping(
        result.asset_digest,
        size,
        path,
        P2P_FORMAT_POLICY_VERSION,
        verification_record(result.asset_digest, after),
        fingerprint,
    )
    _save_local_verification(vault_root, mapping, result)
    return result


def p2p_partial_growth(vault_root: Path, info_hash: str, digest: str, expected_size: int) -> int:
    """Unallocated artifact bytes, including sparse holes, without creating staging."""
    size = _require_size(expected_size)
    info_hash = _require_hex(info_hash, "P2P info hash")
    name = require_digest(digest)[7:]
    try:
        with _vault_directories(
            vault_root, (".p2p", "staging", info_hash), create=False
        ) as directories:
            parent = directories[-1]
            item = parent.stat_entry(name)
            _require_staged_identity(parent.device, item)
            if item.st_size > size:
                raise P2PStorageError("P2P partial exceeds the declared size")
            return max(0, size - parent.allocated_bytes(name, item))
    except FileNotFoundError:
        return size
    except P2PStorageError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return size
        raise


def p2p_staging_usage(vault_root: Path) -> P2PStagingUsage:
    actual = 0
    logical = 0
    partials = 0
    with _vault_directories(vault_root, (".p2p", "staging"), create=True) as directories:
        staging = directories[-1]
        for info_hash in staging.entry_names():
            try:
                item = staging.stat_entry(info_hash)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(item.st_mode) or not _is_digest_hex(info_hash):
                logical += item.st_size
                actual += staging.allocated_bytes(info_hash, item)
                continue
            try:
                context = staging.child(info_hash, create=False)
                with context as parent:
                    for name in parent.entry_names():
                        try:
                            child = parent.stat_entry(name)
                        except FileNotFoundError:
                            continue
                        logical += child.st_size
                        actual += parent.allocated_bytes(name, child)
                        if stat.S_ISREG(child.st_mode) and _is_digest_hex(name):
                            partials += 1
            except P2PStorageError:
                logical += item.st_size
                actual += staging.allocated_bytes(info_hash, item)
                if stat.S_ISREG(item.st_mode) and _is_digest_hex(info_hash):
                    partials += 1
    return P2PStagingUsage(actual, logical, partials)


def _allocated_bytes(item: os.stat_result) -> int:
    blocks = getattr(item, "st_blocks", None)
    return blocks * 512 if blocks is not None else item.st_size


def purge_inactive_p2p_partials(
    vault_root: Path,
    *,
    now: float | None = None,
    retention_seconds: int = P2P_PARTIAL_RETENTION_SECONDS,
) -> P2PStagingPurge:
    if type(retention_seconds) is not int or retention_seconds < 0:
        raise P2PStorageError("P2P retention must be a non-negative integer")
    before = p2p_staging_usage(vault_root)
    cutoff_ns = int((time.time() if now is None else now) * 1_000_000_000) - (
        retention_seconds * 1_000_000_000
    )
    removed = 0
    with _vault_directories(vault_root, (".p2p", "staging"), create=True) as directories:
        staging = directories[-1]
        for info_hash in staging.entry_names():
            if not _is_digest_hex(info_hash):
                continue
            try:
                with staging.child(info_hash, create=False) as parent:
                    for name in parent.entry_names():
                        if not _is_digest_hex(name):
                            continue
                        try:
                            with parent.open_file(name, writable=True) as handle:
                                with _lock_file(handle, blocking=False) as locked:
                                    if not locked:
                                        continue
                                    item = os.fstat(handle.fileno())
                                    if (
                                        not stat.S_ISREG(item.st_mode)
                                        or item.st_nlink != 1
                                        or item.st_mtime_ns > cutoff_ns
                                    ):
                                        continue
                                    related = {name: item}
                                    resume_name = _resume_name(name)
                                    try:
                                        resume = parent.stat_entry(resume_name)
                                    except FileNotFoundError:
                                        pass
                                    else:
                                        if (
                                            not stat.S_ISREG(resume.st_mode)
                                            or resume.st_mtime_ns > cutoff_ns
                                        ):
                                            continue
                                        related[resume_name] = resume
                                    for related_name, related_stat in related.items():
                                        parent.unlink_entry(related_name, related_stat)
                                    parent.fsync()
                                    removed += 1
                        except (FileNotFoundError, P2PStorageError):
                            continue
            except P2PStorageError:
                continue
            with contextlib.suppress(OSError):
                staging.rmdir_child(info_hash)
        staging.fsync()
    after = p2p_staging_usage(vault_root)
    return P2PStagingPurge(
        removed,
        max(0, before.actual_bytes - after.actual_bytes),
        max(0, before.logical_bytes - after.logical_bytes),
    )


def _staged_info_hash(vault_root: Path, path: Path, digest: str) -> str:
    if ".." in path.parts:
        raise P2PStorageError("P2P staged path contains traversal components")
    candidate = Path(os.path.abspath(path))
    staging_root = Path(os.path.abspath(vault_root)) / ".p2p" / "staging"
    try:
        relative = candidate.relative_to(staging_root)
    except ValueError as error:
        raise P2PStorageError("P2P staged path is outside the staging root") from error
    if len(relative.parts) != 2 or relative.name != digest[7:]:
        raise P2PStorageError("P2P staged path is not a canonical infohash/digest path")
    return _require_hex(relative.parts[0], "P2P info hash")


def _require_staged_identity(vault_device: int | Path, item: os.stat_result) -> None:
    if not stat.S_ISREG(item.st_mode):
        raise P2PStorageError("P2P staged asset must be a regular file")
    device = vault_device if isinstance(vault_device, int) else os.stat(vault_device).st_dev
    if item.st_dev != device:
        raise P2PStorageError("P2P staged asset is on a different filesystem")
    if item.st_nlink != 1:
        raise P2PStorageError("P2P staged asset must not have external hard links")


def _require_staged_stat(vault_device: int | Path, item: os.stat_result, size: int) -> None:
    _require_staged_identity(vault_device, item)
    if item.st_size != size:
        raise P2PStorageError(
            f"P2P staged asset size mismatch: expected {size}, got {item.st_size}"
        )


def _require_path_binding(path: Path, item: os.stat_result) -> None:
    current = os.lstat(path)
    if (
        _is_link_or_junction(path, current)
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (item.st_dev, item.st_ino)
    ):
        raise P2PStorageError("P2P file path was rebound during verification")


def _open_regular(path: Path, *, writable: bool) -> BinaryIO:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | _binary_flag() | _nofollow_flag()
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise P2PStorageError(f"could not open regular P2P file: {error}") from error
    handle = os.fdopen(descriptor, "r+b" if writable else "rb")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        handle.close()
        raise P2PStorageError("P2P file must be regular")
    return handle


def _verify_existing_target(
    directory: _ConfinedDirectory,
    name: str,
    digest: str,
    size: int,
    format_policy_version: int,
) -> AssetVerificationRecord | None:
    with _open_stable_file(
        directory,
        name,
        publication_source=False,
    ) as handle:
        with _lock_file(handle) as locked:
            if not locked:
                raise P2PStorageError("could not lock the existing vault asset")
            before = os.fstat(handle.fileno())
            if before.st_nlink != 1 or before.st_size != size:
                raise P2PStorageError(f"existing canonical asset does not match {digest}")
            _validate_safe_format(handle, format_policy_version)
            if _hash_handle(handle) != digest:
                raise P2PStorageError(f"existing canonical asset does not match {digest}")
            after = os.fstat(handle.fileno())
            if _stable_fingerprint(after) != _stable_fingerprint(before):
                raise P2PStorageError("existing canonical asset changed during verification")
            bound = directory.stat_entry(name)
            if (bound.st_dev, bound.st_ino) != (after.st_dev, after.st_ino):
                raise P2PStorageError("existing canonical asset path was rebound")
    return verification_record(digest, after)


def _remove_resume_state(directory: _ConfinedDirectory, asset_name: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        directory.unlink_entry(_resume_name(asset_name))


def _load_resume_state(
    directory: _ConfinedDirectory,
    info_hash: str,
    digest: str,
    size: int,
) -> _ResumeState:
    try:
        with directory.open_file(_resume_name(digest[7:]), writable=False) as handle:
            encoded = handle.read(_MAX_RESUME_BYTES + 1)
            if len(encoded) > _MAX_RESUME_BYTES:
                raise P2PStorageError("P2P resume state exceeds the size limit")
            raw: object = json.loads(encoded.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise P2PStorageError(f"P2P resume state is missing or malformed: {error}") from error
    if not isinstance(raw, Mapping):
        raise P2PStorageError("P2P resume state must be an object")
    row = cast("Mapping[str, object]", raw)
    if set(row) != {
        "version",
        "infoHash",
        "digest",
        "sizeBytes",
        "completedRanges",
        "updatedAtNs",
    }:
        raise P2PStorageError("P2P resume state fields are not canonical")
    ranges = row["completedRanges"]
    if (
        row["version"] != _RESUME_VERSION
        or row["infoHash"] != info_hash
        or row["digest"] != digest
        or row["sizeBytes"] != size
        or type(row["updatedAtNs"]) is not int
        or not isinstance(ranges, list)
    ):
        raise P2PStorageError("P2P resume state identity does not match the staged asset")
    completed: list[tuple[int, int]] = []
    for value in cast("list[object]", ranges):
        bounds = cast("list[object]", value) if isinstance(value, list) else None
        if bounds is None or len(bounds) != 2 or any(type(bound) is not int for bound in bounds):
            raise P2PStorageError("P2P resume ranges are malformed")
        start, end = cast("list[int]", bounds)
        if start < 0 or end <= start or end > size:
            raise P2PStorageError("P2P resume range is outside the staged asset")
        completed.append((start, end))
    canonical = _merge_ranges(completed)
    if tuple(completed) != canonical:
        raise P2PStorageError("P2P resume ranges are not canonical")
    return _ResumeState(info_hash, digest, size, canonical, row["updatedAtNs"])


def _write_resume_state(
    directory: _ConfinedDirectory,
    asset_name: str,
    state: _ResumeState,
) -> None:
    state_name = _resume_name(asset_name)
    tmp_name = state_name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with directory.open_file(tmp_name, writable=True, create_exclusive=True) as handle:
            encoded = json.dumps(state.to_json(), sort_keys=True, separators=(",", ":")).encode()
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory.replace(tmp_name, state_name)
        directory.fsync()
    finally:
        with contextlib.suppress(FileNotFoundError):
            directory.unlink_entry(tmp_name)


def _resume_name(asset_name: str) -> str:
    return asset_name + ".resume.json"


def _merge_ranges(
    ranges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _hash_handle(handle: BinaryIO) -> str:
    handle.seek(0)
    hasher = new_hasher()
    while chunk := handle.read(CHUNK_SIZE):
        hasher.update(chunk)
    return "blake3:" + hasher.hexdigest()


def _validate_safe_format(handle: BinaryIO, version: int) -> str:
    _require_format_policy(version)
    handle.seek(0)
    prefix = handle.read(4)
    if prefix == b"GGUF":
        _validate_gguf(handle)
        return "gguf-v3"
    _validate_safetensors(handle)
    return "safetensors"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _validate_safetensors(handle: BinaryIO) -> None:
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(0)
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise P2PStorageError("P2P asset is not a supported safe model format")
    (header_size,) = struct.unpack("<Q", prefix)
    if header_size == 0 or header_size > min(_MAX_HEADER_BYTES, size - 8):
        raise P2PStorageError("P2P safetensors header size is invalid")
    try:
        raw: object = json.loads(
            handle.read(header_size),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise P2PStorageError(f"P2P safetensors header is malformed: {error}") from error
    if not isinstance(raw, dict):
        raise P2PStorageError("P2P safetensors header must be an object")
    header = cast("dict[str, object]", raw)
    metadata = header.pop("__metadata__", None)
    metadata_items = (
        cast("dict[object, object]", metadata).items() if isinstance(metadata, dict) else ()
    )
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in metadata_items
        )
    ):
        raise P2PStorageError("P2P safetensors metadata must contain only strings")
    if not header:
        raise P2PStorageError("P2P safetensors must contain at least one tensor")
    spans: list[tuple[int, int]] = []
    for name, raw_tensor in header.items():
        if not name or not isinstance(raw_tensor, Mapping):
            raise P2PStorageError("P2P safetensors tensor entry is malformed")
        tensor = cast("Mapping[str, object]", raw_tensor)
        if set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise P2PStorageError("P2P safetensors tensor fields are not canonical")
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        offsets = tensor["data_offsets"]
        dimensions = cast("list[object]", shape) if isinstance(shape, list) else None
        data_offsets = cast("list[object]", offsets) if isinstance(offsets, list) else None
        if (
            not isinstance(dtype, str)
            or dtype not in _SAFETENSORS_DTYPE_BYTES
            or dimensions is None
            or not all(type(dim) is int and dim >= 0 for dim in dimensions)
            or data_offsets is None
            or len(data_offsets) != 2
            or not all(type(offset) is int for offset in data_offsets)
        ):
            raise P2PStorageError("P2P safetensors tensor geometry is invalid")
        start, end = cast("list[int]", data_offsets)
        elements = 1
        for dimension in cast("list[int]", dimensions):
            elements *= dimension
            if elements > _UINT64_MAX:
                raise P2PStorageError("P2P safetensors tensor size overflows uint64")
        if start < 0 or end < start or end - start != elements * _SAFETENSORS_DTYPE_BYTES[dtype]:
            raise P2PStorageError("P2P safetensors data offsets do not match tensor geometry")
        spans.append((start, end))
    expected = 0
    for start, end in sorted(spans):
        if start != expected:
            raise P2PStorageError("P2P safetensors data offsets overlap or leave gaps")
        expected = end
    if expected != size - 8 - header_size:
        raise P2PStorageError("P2P safetensors data length does not match its header")


class _GGUFReader:
    def __init__(self, handle: BinaryIO, size: int) -> None:
        self.handle = handle
        self.size = size
        self.position = 0

    def read(self, length: int, field: str) -> bytes:
        end = self.position + length
        if length < 0 or end > self.size or end > _MAX_HEADER_BYTES:
            raise P2PStorageError(f"P2P GGUF {field} exceeds the bounded header")
        value = self.handle.read(length)
        if len(value) != length:
            raise P2PStorageError(f"P2P GGUF {field} is truncated")
        self.position += length
        return value

    def unpack(self, code: str, field: str) -> int:
        return cast("int", struct.unpack("<" + code, self.read(struct.calcsize(code), field))[0])

    def string(self, field: str) -> str:
        length = self.unpack("Q", field + " length")
        try:
            return self.read(length, field).decode("utf-8")
        except UnicodeDecodeError as error:
            raise P2PStorageError(f"P2P GGUF {field} is not UTF-8") from error

    def value(self, value_type: int, field: str) -> object:
        if value_type in _GGUF_VALUE_BYTES:
            raw = self.read(_GGUF_VALUE_BYTES[value_type], field)
            if value_type == 7 and raw not in (b"\0", b"\1"):
                raise P2PStorageError(f"P2P GGUF {field} boolean is invalid")
            if value_type == 4:
                return struct.unpack("<I", raw)[0]
            return None
        if value_type == _GGUF_STRING:
            return self.string(field)
        if value_type == _GGUF_ARRAY:
            element_type = self.unpack("I", field + " element type")
            if element_type == _GGUF_ARRAY or (
                element_type not in _GGUF_VALUE_BYTES and element_type != _GGUF_STRING
            ):
                raise P2PStorageError(f"P2P GGUF {field} array type is invalid")
            count = self.unpack("Q", field + " count")
            if count > _MAX_GGUF_ITEMS:
                raise P2PStorageError(f"P2P GGUF {field} array is too large")
            for index in range(count):
                self.value(element_type, f"{field}[{index}]")
            return None
        raise P2PStorageError(f"P2P GGUF {field} type is invalid")


def _validate_gguf(handle: BinaryIO) -> None:
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(0)
    reader = _GGUFReader(handle, size)
    if reader.read(4, "magic") != b"GGUF":
        raise P2PStorageError("P2P GGUF magic is invalid")
    version_bytes = reader.read(4, "version")
    if version_bytes == struct.pack(">I", 3) or struct.unpack("<I", version_bytes)[0] != 3:
        raise P2PStorageError("P2P GGUF must be little-endian version 3")
    tensor_count = reader.unpack("Q", "tensor count")
    metadata_count = reader.unpack("Q", "metadata count")
    if tensor_count == 0 or tensor_count > _MAX_GGUF_ITEMS or metadata_count > _MAX_GGUF_ITEMS:
        raise P2PStorageError("P2P GGUF item counts are invalid")
    metadata: dict[str, tuple[int, object]] = {}
    for index in range(metadata_count):
        key = reader.string(f"metadata[{index}] key")
        if not key or key in metadata:
            raise P2PStorageError("P2P GGUF metadata keys must be unique and non-empty")
        value_type = reader.unpack("I", f"metadata {key!r} type")
        metadata[key] = (value_type, reader.value(value_type, f"metadata {key!r}"))
    alignment = _GGUF_ALIGNMENT
    if "general.alignment" in metadata:
        value_type, value = metadata["general.alignment"]
        if value_type != 4 or type(value) is not int:
            raise P2PStorageError("P2P GGUF general.alignment must be UINT32")
        alignment = value
    if alignment == 0 or alignment & (alignment - 1):
        raise P2PStorageError("P2P GGUF alignment must be a nonzero power of two")

    names: set[str] = set()
    expected_offset = 0
    quantized = False
    for index in range(tensor_count):
        name = reader.string(f"tensor[{index}] name")
        if not name or len(name.encode()) > 127 or name in names:
            raise P2PStorageError("P2P GGUF tensor names must be unique, non-empty, and bounded")
        names.add(name)
        rank = reader.unpack("I", f"tensor {name!r} rank")
        if rank > _MAX_GGUF_RANK:
            raise P2PStorageError("P2P GGUF tensor rank exceeds the safe format policy")
        dimensions = [reader.unpack("Q", f"tensor {name!r} dimension") for _ in range(rank)]
        elements = 1
        for dimension in dimensions:
            if dimension and elements > _UINT64_MAX // dimension:
                raise P2PStorageError("P2P GGUF tensor element count overflows uint64")
            elements *= dimension
        type_code = reader.unpack("I", f"tensor {name!r} type")
        geometry = _GGUF_TYPES.get(type_code)
        if geometry is None:
            raise P2PStorageError("P2P GGUF tensor type is not admitted by format policy 1")
        block_elements, block_bytes = geometry
        row_elements = dimensions[0] if dimensions else 1
        if row_elements % block_elements or elements % block_elements:
            raise P2PStorageError("P2P GGUF tensor dimensions do not fit the encoded block type")
        blocks = elements // block_elements
        if blocks > _UINT64_MAX // block_bytes:
            raise P2PStorageError("P2P GGUF tensor size overflows uint64")
        relative_offset = reader.unpack("Q", f"tensor {name!r} offset")
        if relative_offset != expected_offset:
            raise P2PStorageError("P2P GGUF tensor offsets overlap or leave a gap")
        expected_offset = _align(relative_offset + blocks * block_bytes, alignment)
        quantized |= block_elements != 1
    if quantized and (
        "general.quantization_version" not in metadata
        or metadata["general.quantization_version"][0] != 4
        or type(metadata["general.quantization_version"][1]) is not int
        or metadata["general.quantization_version"][1] <= 0
    ):
        raise P2PStorageError("P2P quantized GGUF requires a positive quantization version")
    data_offset = _align(reader.position, alignment)
    if any(reader.read(data_offset - reader.position, "tensor data padding")):
        raise P2PStorageError("P2P GGUF tensor data padding must be zero")
    if data_offset + expected_offset != size:
        raise P2PStorageError("P2P GGUF tensor data size does not match its header")


def _align(value: int, alignment: int) -> int:
    if value > _UINT64_MAX - (alignment - 1):
        raise P2PStorageError("P2P GGUF aligned size overflows uint64")
    return (value + alignment - 1) & -alignment


def _require_format_policy(version: int) -> None:
    if type(version) is not int or version != P2P_FORMAT_POLICY_VERSION:
        raise P2PStorageError(
            f"unsupported P2P format policy {version!r}; expected {P2P_FORMAT_POLICY_VERSION}"
        )


def _require_size(size: int) -> int:
    if type(size) is not int or size <= 0:
        raise P2PStorageError("P2P asset size must be a positive integer")
    return size


def _require_hex(value: str, field: str) -> str:
    if not _is_digest_hex(value):
        raise P2PStorageError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _is_digest_hex(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)


def _stable_fingerprint(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _local_file_fingerprint(
    handle: BinaryIO,
    item: os.stat_result,
) -> tuple[int, ...]:
    fingerprint = _stable_fingerprint(item)
    if _p2p_windows is None:
        return fingerprint
    return (*fingerprint, _p2p_windows.change_token(handle.fileno()))


def _is_link_or_junction(path: Path, item: os.stat_result) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return stat.S_ISLNK(item.st_mode) or bool(is_junction is not None and is_junction())


def _binary_flag() -> int:
    return getattr(os, "O_BINARY", 0)


def _nofollow_flag() -> int:
    return getattr(os, "O_NOFOLLOW", 0)


def _directory_flag() -> int:
    return getattr(os, "O_DIRECTORY", 0)


def _cloexec_flag() -> int:
    return getattr(os, "O_CLOEXEC", 0)
