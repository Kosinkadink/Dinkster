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

import json
import os
import re
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from .identity import DIGEST_PREFIX, AssetError, new_hasher
from .integrity import verification_record_for_publication
from .library import append_write_record
from .model import AssetRef
from .mounts import MOUNT_NAMESPACE
from .save_target import SaveTarget, coerce_save_target

__all__ = ["AssetWriter", "MountSnapshotWriter", "MountWriteAuthority"]

_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9.]*$")


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


class AssetWriter:
    """Save bytes into a readwrite mount and get back an AssetRef."""

    def __init__(self, authority: MountWriteAuthority) -> None:
        self._authority = authority

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
        if type(limit) is not int or limit < 0:
            raise AssetError("limit must be a non-negative integer")
        parsed = coerce_save_target(target)
        if not _SUFFIX_RE.match(suffix):
            raise AssetError(f"save suffix must be a bare extension like '.png': {suffix!r}")
        root = self._authority.writable_root(parsed.mount)
        folder = root / parsed.subfolder if parsed.subfolder else root
        folder.mkdir(parents=True, exist_ok=True)
        self._require_contained(root, folder, parsed)
        tmp = folder / f".dinkster-save-{uuid.uuid4().hex}"
        final: Path | None = None
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
                while chunk := source.read(8 * 1024 * 1024):
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
            final = self._claim_and_land(folder, parsed.stem, suffix, tmp)
        finally:
            tmp.unlink(missing_ok=True)
        assert final is not None
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
        return AssetRef(
            digest=digest,
            name=final.name,
            size=size,
            media_type=media_type,
            virtual_path=f"{MOUNT_NAMESPACE}/{parsed.mount}/{relative}",
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

    @staticmethod
    def _claim_and_land(folder: Path, stem: str, suffix: str, tmp: Path) -> Path:
        """Publish the temp file under the next free counter name.

        A hard link publishes the already-complete same-directory temp with
        atomic no-replace semantics. Concurrent savers therefore get distinct
        names, without exposing an empty placeholder to readers.
        """
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
        while True:
            final = folder / f"{stem}_{counter:05d}{suffix}"
            try:
                os.link(tmp, final)
            except FileExistsError:
                counter += 1
                continue
            return final
