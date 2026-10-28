"""AssetWriter: the one gate through which node output bytes reach disk.

Writes are scoped by MOUNTS, never by paths (DESIGN 3.12 extension): a
node hands the writer a :class:`SaveTarget` (mount id + relative prefix)
and bytes; the writer checks the mount is currently granted readwrite,
derives a safe location under its root, lands the file atomically with a
collision-free counter name, and returns an :class:`AssetRef` - digest
identity, virtual path, no host path. Everything ComfyUI's save path got
wrong is structural here: no traversal (the prefix grammar cannot express
it), no symlink escape (resolved containment is checked), no torn files
(same-directory temp + atomic link), no counter races (names are published
with no-replace link semantics, which are atomic across processes).

Authority comes from a :class:`MountWriteAuthority`. Workers use
:class:`MountSnapshotWriter` over the engine-published snapshot file
(``DINKSTER_MOUNTS_SNAPSHOT``), re-read per save: a mount granted mid-session
becomes writable in a live worker with no restart, and a revoked mount
refuses the very next save. The engine side can pass a
:class:`MountTable` method directly.

Every landed file is recorded in the writes sidecar (library.py), so its
digest resolves immediately - the engine can stream it to a client before
any rescan of the mount.
"""

from __future__ import annotations

import errno
import importlib
import json
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from dinkster_values import MEBIBYTE
from dinkster_values.asset_folders import current_asset_folders

from .identity import DIGEST_PREFIX, AssetError, new_hasher
from .integrity import verification_record_for_publication
from .library import append_write_record
from .model import AssetRef
from .mounts import MOUNT_NAMESPACE
from .save_target import SaveTarget, coerce_save_target

__all__ = ["AssetWriter", "MountSnapshotWriter", "MountWriteAuthority"]

_SUFFIX_RE = re.compile(r"^_?\.[A-Za-z0-9][A-Za-z0-9.]*$")
_WRITER_PATH_LOCKS_GUARD = threading.Lock()
_WRITER_PATH_LOCKS: dict[str, threading.RLock] = {}


class _Fcntl(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, fd: int, operation: int, /) -> None: ...


class _Msvcrt(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, fd: int, mode: int, nbytes: int, /) -> None: ...

    def open_osfhandle(self, handle: int, flags: int, /) -> int: ...


class MountWriteAuthority(Protocol):
    """Where mount ids become writable roots. Raises AssetError for an
    unknown, revoked, not-ready, or read-only mount - the refusal carries
    the reason, never a silent fallback directory."""

    def writable_root(self, mount_id: str) -> Path: ...


class MountSnapshotWriter:
    """Write authority over the engine-published mount snapshot.

    Stdlib-only and stateless: the snapshot is re-read on every call, so
    grants and revokes made while this process runs apply to the next
    save with no restart and no control-channel message (the same
    contract MountSnapshotResolver gives reads). Only READY mounts appear
    in the snapshot, so writing into a mount that never scanned - or
    whose directory is gone - refuses with a reason."""

    def __init__(self, snapshot_path: Path | str) -> None:
        self._path = Path(snapshot_path)

    def writable_root(self, mount_id: str) -> Path:
        try:
            loaded: object = json.loads(self._path.read_text("utf-8"))
        except (OSError, ValueError):
            raise AssetError(
                f"no readable mounts snapshot at {self._path}; this process "
                "has no filesystem write grants"
            ) from None
        rows: object = (
            cast("dict[str, object]", loaded).get("mounts") if isinstance(loaded, dict) else None
        )
        if not isinstance(rows, list):
            raise AssetError(f"malformed mounts snapshot at {self._path}")
        for row in cast("list[object]", rows):
            if not isinstance(row, dict):
                continue
            entry = cast("dict[str, object]", row)
            if entry.get("id") != mount_id:
                continue
            root = entry.get("root")
            if not isinstance(root, str):
                continue
            if entry.get("mode") != "readwrite":
                raise AssetError(
                    f"mount {mount_id!r} is granted read-only; saving needs a readwrite mount"
                )
            return Path(root)
        raise AssetError(
            f"no ready mount {mount_id!r}: it was never granted, was "
            "revoked, or has not finished scanning"
        )

    def output_mount(self) -> str:
        context = current_asset_folders()
        if context is not None:
            if context.output_mount is None:
                raise AssetError("no default output mount is configured for this project")
            return context.output_mount
        try:
            loaded: object = json.loads(self._path.read_text("utf-8"))
        except (OSError, ValueError):
            raise AssetError(f"no readable mounts snapshot at {self._path}") from None
        selected = (
            cast("dict[str, object]", loaded).get("outputMount")
            if isinstance(loaded, dict)
            else None
        )
        if not isinstance(selected, str) or not selected:
            raise AssetError("no default output mount is configured")
        return selected


class AssetWriter:
    """Save bytes into a readwrite mount and get back an AssetRef."""

    def __init__(self, authority: MountWriteAuthority) -> None:
        self._authority = authority

    def output_target(
        self,
        prefix: str,
        mount: str | None = None,
        *,
        image_width: int = 0,
        image_height: int = 0,
    ) -> dict[str, str]:
        """Combine separate node inputs without resolving a host path or granting writes."""
        if mount is None:
            if not isinstance(self._authority, MountSnapshotWriter):
                raise AssetError("default output selection requires a mounts snapshot")
            mount = self._authority.output_mount()
        if "%" in prefix:
            now = time.localtime()
            variables = {
                "width": str(image_width),
                "height": str(image_height),
                "year": str(now.tm_year),
                "month": f"{now.tm_mon:02}",
                "day": f"{now.tm_mday:02}",
                "hour": f"{now.tm_hour:02}",
                "minute": f"{now.tm_min:02}",
                "second": f"{now.tm_sec:02}",
            }
            for name, value in variables.items():
                prefix = prefix.replace(f"%{name}%", value)
        return {"mount": mount, "prefix": prefix}

    def save_bytes(
        self,
        target: object,
        data: bytes,
        *,
        suffix: str,
        media_type: str = "application/octet-stream",
    ) -> AssetRef:
        """Land ``data`` under ``target`` as ``<stem>_<NNNNN><suffix>``.

        ``target`` is a SaveTarget or its wire mapping. ``suffix`` is the
        node-owned extension (".png"); the caller decided the encoding, so
        the caller names it. Returns the ref whose digest resolves
        immediately (writes sidecar) wherever this mount is readable."""
        import io

        return self.save_stream(
            target, io.BytesIO(data), suffix=suffix, media_type=media_type, limit=len(data)
        )

    def save_stream(
        self,
        target: object,
        source: BinaryIO,
        *,
        suffix: str,
        media_type: str = "application/octet-stream",
        limit: int,
    ) -> AssetRef:
        """Boundedly copy a held seekable source and atomically publish it."""
        parsed = self._validated_target(target, suffix, limit)
        _, folder = self._writable_folder(parsed)
        with self._directory_claim(folder):
            ref, _ = self._write_stream(
                parsed,
                source,
                suffix=suffix,
                media_type=media_type,
                limit=limit,
            )
        return ref

    def save_named_stream(
        self,
        target: object,
        source: BinaryIO,
        *,
        media_type: str = "application/octet-stream",
        limit: int,
        before_publish: Callable[[AssetRef], None] | None = None,
    ) -> AssetRef:
        """Import bytes under the target's full filename, numbering collisions only."""
        if type(limit) is not int or limit < 0:
            raise AssetError("limit must be a non-negative integer")
        parsed = coerce_save_target(target)
        reserved = {"con", "prn", "aux", "nul"} | {
            f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
        }
        for segment in parsed.prefix.split("/"):
            if (
                len(segment) > 255
                or segment.endswith((".", " "))
                or segment.casefold().startswith(".dinkster-")
                or segment.split(".", 1)[0].casefold() in reserved
                or any(ord(character) < 32 or character in '<>:"|?*\x7f' for character in segment)
            ):
                raise AssetError("import target must use portable folder and file names")
        suffix = Path(parsed.stem).suffix
        named = SaveTarget(
            parsed.mount, parsed.prefix.removesuffix(suffix) if suffix else parsed.prefix
        )
        root, folder = self._writable_folder(named)
        with self._directory_claim(folder):
            ref, _ = self._write_stream(
                named,
                source,
                suffix=suffix,
                media_type=media_type,
                limit=limit,
                expected_root=root,
                prefer_exact_name=True,
                before_publish=before_publish,
            )
        return ref

    def recover_named_asset(self, ref: AssetRef) -> Path | None:
        """Verify a journaled publication under current authority without creating bytes."""
        namespace, mount, relative = ref.virtual_path.split("/", 2)
        if namespace != MOUNT_NAMESPACE:
            raise AssetError("recovery requires a mounted asset")
        target = SaveTarget(mount, relative)
        root = self._authority.writable_root(mount)
        path = root / relative
        self._require_contained(root, path, target)
        if not path.is_file():
            return None
        from .integrity import open_verified

        with open_verified(path, ref.digest) as handle:
            observed = os.fstat(handle.fileno())
            if observed.st_size != ref.size:
                raise AssetError("journaled asset size changed")
        if self._authority.writable_root(mount).resolve() != root.resolve():
            raise AssetError("journaled mount changed its root")
        self._require_contained(root, path, target)
        append_write_record(root, relative, ref.digest, ref.size, observed.st_mtime_ns)
        return path

    def save_encoded(
        self,
        target: object,
        encoder: Callable[[str, int], BinaryIO],
        *,
        suffix: str,
        media_type: str = "application/octet-stream",
        limit: int,
    ) -> AssetRef:
        """Reserve a source-compatible name while an encoder records it.

        ``encoder`` receives the filename stem and integer counter, never a
        host path or an output placeholder. It owns its returned stream; the
        writer reads it without closing it. A pre-existing final name after
        encoding is an error rather than a renamed publication, because the
        encoded metadata may carry that selected name.
        """
        parsed = self._validated_target(target, suffix, limit)
        root, folder = self._writable_folder(parsed)
        with self._directory_claim(folder):
            counter = self._next_counter(folder, parsed.stem)
            source = encoder(parsed.stem, counter)
            ref, _ = self._write_stream(
                parsed,
                source,
                suffix=suffix,
                media_type=media_type,
                limit=limit,
                counter=counter,
                expected_root=root,
                strict_counter=True,
            )
        return ref

    def save_stream_batch(
        self,
        target: object,
        sources: Iterable[BinaryIO],
        *,
        suffix: str,
        media_type: str = "application/octet-stream",
        limit: int,
    ) -> list[AssetRef]:
        """Publish a source-ordered batch with one source-compatible counter.

        ``%batch_num%`` expands in the filename stem after the initial
        counter is selected from the unexpanded target. ``suffix`` may start
        with ``_`` (``_.png``) when the caller needs the source filename tail.
        Files already published before a later source fails remain complete.
        """
        parsed = self._validated_target(target, suffix, limit)
        root, folder = self._writable_folder(parsed)
        with self._directory_claim(folder):
            counter = self._next_counter(folder, parsed.stem)
            refs: list[AssetRef] = []
            for batch_num, source in enumerate(sources):
                stem = parsed.stem.replace("%batch_num%", str(batch_num))
                ref, counter = self._write_stream(
                    parsed,
                    source,
                    suffix=suffix,
                    media_type=media_type,
                    limit=limit,
                    counter=counter,
                    stem=stem,
                    expected_root=root,
                )
                refs.append(ref)
        return refs

    @staticmethod
    def _validated_target(target: object, suffix: str, limit: int) -> SaveTarget:
        if type(limit) is not int or limit < 0:
            raise AssetError("limit must be a non-negative integer")
        parsed = coerce_save_target(target)
        if not _SUFFIX_RE.match(suffix):
            raise AssetError(
                f"save suffix must be a bare extension like '.png' or '_.png': {suffix!r}"
            )
        return parsed

    def _writable_folder(self, parsed: SaveTarget) -> tuple[Path, Path]:
        root = self._authority.writable_root(parsed.mount)
        if not root.is_dir():
            raise AssetError(f"output mount {parsed.mount!r} is unavailable at {root}")
        folder = root
        for part in Path(parsed.subfolder).parts:
            folder = folder / part
            self._require_contained(root, folder, parsed)
            folder.mkdir(exist_ok=True)
        return root, folder

    def _write_stream(
        self,
        parsed: SaveTarget,
        source: BinaryIO,
        *,
        suffix: str,
        media_type: str,
        limit: int,
        counter: int | None = None,
        stem: str | None = None,
        expected_root: Path | None = None,
        strict_counter: bool = False,
        prefer_exact_name: bool = False,
        before_publish: Callable[[AssetRef], None] | None = None,
    ) -> tuple[AssetRef, int]:
        root, folder = self._writable_folder(parsed)
        if expected_root is not None and root.resolve() != expected_root.resolve():
            raise AssetError(f"mount {parsed.mount!r} changed its root during batch save")
        stem = parsed.stem if stem is None else stem
        tmp = folder / f".dinkster-save-{uuid.uuid4().hex}"
        final: Path | None = None
        used_counter: int | None = None
        try:
            try:
                source.seek(0)
                before = os.fstat(source.fileno())
            except (AttributeError, OSError):
                before = None
                try:
                    source.seek(0)
                except (AttributeError, OSError) as exc:
                    raise AssetError("save source must be seekable binary input") from exc
            hasher = new_hasher()
            size = 0
            with tmp.open("xb") as output:
                while chunk := source.read(8 * MEBIBYTE):
                    size += len(chunk)
                    if size > limit:
                        raise AssetError("save source exceeds limit")
                    hasher.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
                verified_stat = os.fstat(output.fileno())
            if before is not None:
                after = os.fstat(source.fileno())
                fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                if any(getattr(before, field) != getattr(after, field) for field in fields):
                    raise AssetError("save source changed while it was copied")
            digest = DIGEST_PREFIX + hasher.hexdigest()
            if prefer_exact_name:
                current_root, _ = self._writable_folder(parsed)
                if current_root.resolve() != root.resolve():
                    raise AssetError(f"mount {parsed.mount!r} changed its root during import")

            def prepare(final: Path) -> None:
                if before_publish is not None:
                    before_publish(
                        AssetRef(
                            digest,
                            final.name,
                            size,
                            media_type,
                            f"{MOUNT_NAMESPACE}/{parsed.mount}/{final.relative_to(root).as_posix()}",
                        )
                    )
                    if self._authority.writable_root(parsed.mount).resolve() != root.resolve():
                        raise AssetError("import mount changed its root before publication")
                    self._require_contained(root, final, parsed)

            final, used_counter = self._claim_and_land(
                folder,
                stem,
                suffix,
                tmp,
                counter,
                strict_counter=strict_counter,
                prefer_exact_name=prefer_exact_name,
                before_publish=prepare if before_publish is not None else None,
            )
        finally:
            tmp.unlink(missing_ok=True)
        assert final is not None and used_counter is not None
        stat = final.stat()
        record = verification_record_for_publication(digest, verified_stat, stat)
        relative = final.relative_to(root).as_posix()
        try:
            append_write_record(
                root,
                relative,
                digest,
                size,
                stat.st_mtime_ns,
                record,
            )
        except OSError:
            pass  # resolvability degrades to next-scan; the save itself stood
        return (
            AssetRef(
                digest=digest,
                name=final.name,
                size=size,
                media_type=media_type,
                virtual_path=f"{MOUNT_NAMESPACE}/{parsed.mount}/{relative}",
            ),
            used_counter + 1,
        )

    @staticmethod
    def _require_contained(root: Path, folder: Path, target: SaveTarget) -> None:
        """The prefix grammar already forbids traversal; this catches the
        remaining escape - a symlinked subfolder inside the mount pointing
        outside it."""
        resolved_root = root.resolve()
        resolved_folder = folder.resolve()
        if resolved_folder != resolved_root and not resolved_folder.is_relative_to(resolved_root):
            raise AssetError(
                f"save target {target.mount}/{target.prefix} escapes its "
                f"mount (symlinked folder leaves {resolved_root})"
            )

    @contextmanager
    def _directory_claim(self, folder: Path) -> Generator[None]:
        lock_path = folder / ".dinkster-asset-writer.lock"
        with self._path_lock(lock_path), self._open_lock_file(lock_path) as handle:
            handle.seek(0)
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                msvcrt = cast("_Msvcrt", importlib.import_module("msvcrt"))
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError as exc:
                        if (
                            exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
                            and getattr(exc, "winerror", None) != 33
                        ):
                            raise
                        time.sleep(0.05)
                    else:
                        break
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = cast("_Fcntl", importlib.import_module("fcntl"))
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _open_lock_file(lock_path: Path) -> BinaryIO:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            return AssetWriter._open_windows_lock_file(lock_path)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise AssetError("safe writer lock opening is unavailable on this platform")
        try:
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | nofollow, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise AssetError(f"unsafe writer lock path: {lock_path}") from None
            raise
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            os.close(descriptor)
            raise AssetError(f"unsafe writer lock path: {lock_path}")
        return os.fdopen(descriptor, "r+b")

    @staticmethod
    def _open_windows_lock_file(lock_path: Path) -> BinaryIO:
        ctypes = importlib.import_module("ctypes")
        wintypes = importlib.import_module("ctypes.wintypes")
        msvcrt = cast("_Msvcrt", importlib.import_module("msvcrt"))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        get_file_information = kernel32.GetFileInformationByHandle
        get_file_information.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
        get_file_information.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        class FileInformation(ctypes.Structure):
            _fields_ = [
                ("attributes", wintypes.DWORD),
                ("creation_time", wintypes.FILETIME),
                ("last_access_time", wintypes.FILETIME),
                ("last_write_time", wintypes.FILETIME),
                ("volume_serial_number", wintypes.DWORD),
                ("file_size_high", wintypes.DWORD),
                ("file_size_low", wintypes.DWORD),
                ("number_of_links", wintypes.DWORD),
                ("file_index_high", wintypes.DWORD),
                ("file_index_low", wintypes.DWORD),
            ]

        generic_read_write = 0xC0000000
        share_all = 0x00000007
        create_new = 1
        open_always = 4
        file_attribute_normal = 0x00000080
        file_flag_open_reparse_point = 0x00200000
        file_attribute_reparse_point = 0x00000400
        invalid_handle_value = wintypes.HANDLE(-1).value
        while True:
            handle = create_file(
                str(lock_path),
                generic_read_write,
                0,
                None,
                create_new,
                file_attribute_normal | file_flag_open_reparse_point,
                None,
            )
            if handle != invalid_handle_value:
                with AssetWriter._windows_file_from_handle(handle, msvcrt, close_handle) as created:
                    created.write(b"\0")
                    created.flush()
                continue
            error = ctypes.get_last_error()
            if error != 80:
                raise OSError(error, f"cannot create writer lock {lock_path}")
            handle = create_file(
                str(lock_path),
                generic_read_write,
                share_all,
                None,
                open_always,
                file_attribute_normal | file_flag_open_reparse_point,
                None,
            )
            if handle != invalid_handle_value:
                break
            if ctypes.get_last_error() != 32:
                raise OSError(ctypes.get_last_error(), f"cannot open writer lock {lock_path}")
            time.sleep(0.05)
        information = FileInformation()
        try:
            if not get_file_information(handle, ctypes.byref(information)):
                raise OSError(ctypes.get_last_error(), f"cannot inspect writer lock {lock_path}")
            if (
                information.attributes & file_attribute_reparse_point
                or information.number_of_links != 1
            ):
                raise AssetError(f"unsafe writer lock path: {lock_path}")
        except BaseException:
            close_handle(handle)
            raise
        return AssetWriter._windows_file_from_handle(handle, msvcrt, close_handle)

    @staticmethod
    def _windows_file_from_handle(
        handle: int,
        msvcrt: _Msvcrt,
        close_handle: Callable[[int], object],
    ) -> BinaryIO:
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR)
        except BaseException:
            close_handle(handle)
            raise
        try:
            return os.fdopen(descriptor, "r+b")
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _path_lock(path: Path) -> threading.RLock:
        key = str(path.absolute())
        with _WRITER_PATH_LOCKS_GUARD:
            return _WRITER_PATH_LOCKS.setdefault(key, threading.RLock())

    @staticmethod
    def _next_counter(folder: Path, stem: str) -> int:
        """Match ComfyUI's ``get_save_image_path`` filename scan exactly."""
        counters: list[int] = []
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    prefix = f"{stem}_"
                    if not entry.name.startswith(prefix):
                        continue
                    try:
                        counter = int(entry.name[len(prefix) :].split(".", 1)[0].split("_", 1)[0])
                    except ValueError:
                        counter = 0
                    counters.append(counter)
        except OSError:
            pass
        return max(counters, default=0) + 1

    @staticmethod
    def _next_counter_for_suffix(folder: Path, stem: str, suffix: str) -> int:
        pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
        counter = 1
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    match = pattern.match(entry.name)
                    if match:
                        counter = max(counter, int(match.group(1)) + 1)
        except OSError:
            pass
        return counter

    @staticmethod
    def _claim_and_land(
        folder: Path,
        stem: str,
        suffix: str,
        tmp: Path,
        counter: int | None = None,
        *,
        strict_counter: bool = False,
        prefer_exact_name: bool = False,
        before_publish: Callable[[Path], None] | None = None,
    ) -> tuple[Path, int]:
        """Publish the temp file under the next free counter name.

        A hard link publishes the already-complete same-directory temp with
        atomic no-replace semantics. Concurrent savers therefore get distinct
        names, without exposing an empty placeholder to readers.
        """
        if prefer_exact_name:
            final = folder / f"{stem}{suffix}"
            if before_publish is not None:
                before_publish(final)
            try:
                os.link(tmp, final)
            except FileExistsError:
                pass
            else:
                return final, 0
        if counter is None:
            counter = AssetWriter._next_counter_for_suffix(folder, stem, suffix)
        while True:
            final = folder / f"{stem}_{counter:05d}{suffix}"
            if before_publish is not None:
                before_publish(final)
            try:
                os.link(tmp, final)
            except FileExistsError:
                if strict_counter:
                    raise AssetError("selected save name was claimed before publication") from None
                counter += 1
                continue
            return final, counter
