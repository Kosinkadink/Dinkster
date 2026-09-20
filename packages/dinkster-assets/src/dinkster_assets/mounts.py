"""Filesystem mounts: N explicit directory grants, not one input dir.

A mount is the unit of filesystem authority (DESIGN 3.12 extension): a
real directory the operator granted, with an id, a mode, and nothing
implicit. ComfyUI's single input-directory / single output-directory
model is the failure this replaces - users work with files all over a
machine, and "the one blessed folder" forces copies. Real paths never
enter workflows: a mount's files are cataloged under the virtual
namespace ``mounts/<id>/<relative-path>`` and referenced by content
digest, so granting, revoking, or moving a mount never invalidates a
document.

Three pieces, one file:

- ``MountDef`` + ``parse_mounts``/``dump_mounts``: the ``mounts.toml``
  config contract, strict like the installs registry (typos fail loudly,
  never default).
- ``MountTable``: the LIVE, engine-owned table. Mounts are runtime state
  with config persistence underneath - the desktop "grant a folder while
  running" flow mutates the table first and the TOML is the durable
  record, so add/remove never needs a restart. Each mount wraps a
  scanning :class:`LocalAssetLibrary`; scans are blocking by design
  (callers thread them off) and publish a worker-facing SNAPSHOT file
  after every change.
- ``MountSnapshotResolver``: the worker side. Stdlib-only, env-pointed
  at the snapshot; re-checks the file per resolve and rebuilds its
  per-mount resolvers when it changed. This is what makes runtime mount
  changes visible to long-lived workers without a control-channel
  message or a worker restart (an idle torch process holds real VRAM;
  restarting workers to grant a folder would be absurd).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .catalog import AssetEntry, FolderListing
from .identity import AssetError, require_digest
from .integrity import AssetVerificationRecord
from .kind import KIND_MODEL_EMBEDDING, require_asset_kind
from .library import INDEX_NAME, AssetScanProgress, IndexedAssetResolver, LocalAssetLibrary
from .model import AssetRef, AssetResolution

__all__ = [
    "MOUNT_MODES",
    "MOUNT_NAMESPACE",
    "EmbeddingNameIndex",
    "MountDef",
    "MountSnapshotResolver",
    "MountTable",
    "MountsError",
    "dump_mounts",
    "load_mounts",
    "parse_mounts",
]

MOUNT_NAMESPACE = "mounts"
MOUNT_MODES = ("read", "readwrite")

MOUNT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
"""The mount id grammar - shared with save targets, which name mounts."""
_ENTRY_KEYS = frozenset({"path", "mode", "priority"})


class MountsError(Exception):
    """A mount is misconfigured or unknown - named loudly, never defaulted."""


@dataclass(frozen=True)
class _ExactIndexedPathResolver:
    root: Path
    relative: str
    digest: str
    size: int
    mtime_ns: int
    verification: AssetVerificationRecord | None

    def resolve(self, digest: str) -> Path | None:
        if digest != self.digest:
            return None
        path = self.root / self.relative
        try:
            stat = path.stat()
        except OSError:
            return None
        actual = (stat.st_size, stat.st_mtime_ns)
        return path if actual == (self.size, self.mtime_ns) else None

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        if digest != self.digest:
            return None
        path = self.root / self.relative
        return AssetResolution(path, self.verification) if path.is_file() else None


class EmbeddingNameIndex:
    """Immutable textual-inversion namespace captured from one snapshot."""

    _UNSUPPORTED = frozenset({".pt", ".bin", ".ckpt", ".pth", ".zip", ".pickle", ".pkl"})

    def __init__(self, snapshot_path: Path | str) -> None:
        path = Path(snapshot_path)
        try:
            loaded: object = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise MountsError(f"malformed configured mounts snapshot {path}: {exc}") from exc
        snapshot = cast("dict[str, object]", loaded) if isinstance(loaded, dict) else {}
        if (
            not isinstance(loaded, dict)
            or set(snapshot) != {"mounts"}
            or not isinstance(snapshot["mounts"], list)
        ):
            raise MountsError(f"malformed configured mounts snapshot {path}")
        aliases: dict[str, list[AssetRef]] = {}
        unsupported: set[str] = set()
        state: list[tuple[str, str, str, str]] = []
        rows = cast("list[object]", snapshot["mounts"])
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                raise MountsError(f"malformed mount row in {path}")
            raw = cast("dict[str, object]", raw_row)
            if raw.get("kind", "") != KIND_MODEL_EMBEDDING:
                continue
            mount_id = raw.get("id")
            root = raw.get("root")
            index = raw.get("index")
            if not all(isinstance(value, str) and value for value in (mount_id, root, index)):
                raise MountsError(f"malformed embedding mount row in {path}")
            root_path, index_path = Path(cast("str", root)), Path(cast("str", index))
            try:
                document: object = json.loads(index_path.read_text("utf-8"))
            except (OSError, ValueError) as exc:
                raise MountsError(f"malformed embedding asset index {index_path}: {exc}") from exc
            if not isinstance(document, dict):
                raise MountsError(f"malformed embedding asset index {index_path}")
            entries = cast("dict[object, object]", document)
            for relative, raw_entry in entries.items():
                entry_keys: set[object] = (
                    set(cast("dict[object, object]", raw_entry))
                    if isinstance(raw_entry, dict)
                    else set()
                )
                if (
                    not isinstance(relative, str)
                    or not isinstance(raw_entry, dict)
                    or entry_keys != {"digest", "size", "mtimeNs"}
                ):
                    raise MountsError(f"malformed embedding asset index {index_path}")
                entry = cast("dict[str, object]", raw_entry)
                if (
                    not relative
                    or Path(relative).is_absolute()
                    or "\\" in relative
                    or any(ord(c) < 32 or ord(c) == 127 for c in relative)
                    or re.match(r"^[A-Za-z]:", relative)
                    or any(part in ("", ".", "..") for part in relative.split("/"))
                ):
                    raise MountsError(f"unsafe path in embedding asset index {index_path}")
                digest = entry["digest"]
                size = entry["size"]
                mtime_ns = entry["mtimeNs"]
                try:
                    if not isinstance(digest, str):
                        raise AssetError("digest is not a string")
                    require_digest(digest)
                except AssetError as exc:
                    raise MountsError(
                        f"malformed embedding asset index {index_path}: {exc}"
                    ) from exc
                if type(size) is not int or size < 0 or type(mtime_ns) is not int or mtime_ns < 0:
                    raise MountsError(f"malformed embedding asset index {index_path}")
                suffix = Path(relative).suffix.lower()
                if suffix == ".safetensors":
                    names = (relative, relative[: -len(suffix)])
                    row_state = "safe"
                elif suffix in self._UNSUPPORTED:
                    names = (
                        (relative, relative[: -len(suffix)])
                        if suffix in {".pt", ".bin"}
                        else (relative,)
                    )
                    row_state = "unsupported"
                else:
                    names = (relative,)
                    row_state = "unsupported"
                for name in names:
                    state.append((name, row_state, cast("str", mount_id), digest))
                if row_state == "unsupported":
                    unsupported.update(names)
                    continue
                if row_state != "safe":
                    continue
                ref = AssetRef(
                    digest,
                    Path(relative).name,
                    size,
                    virtual_path=f"mounts/{mount_id}/{relative}",
                    resolver=_ExactIndexedPathResolver(
                        root_path,
                        relative,
                        digest,
                        size,
                        mtime_ns,
                        AssetVerificationRecord.from_json(digest, entry.get("verification")),
                    ),
                )
                for name in names:
                    aliases.setdefault(name, []).append(ref)
        self._aliases = {name: tuple(refs) for name, refs in aliases.items()}
        self._unsupported = frozenset(unsupported)
        canonical = json.dumps(sorted(state), ensure_ascii=True, separators=(",", ":"))
        self.binding_digest = hashlib.sha256(canonical.encode()).hexdigest() if state else None

    def resolve(self, name: str) -> AssetRef | None:
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or re.match(r"^[A-Za-z]:", name)
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise AssetError(f"unsafe embedding name: {name!r}")
        refs = self._aliases.get(name, ())
        if refs and name in self._unsupported:
            locations = sorted(ref.virtual_path for ref in refs)
            raise AssetError(
                f"ambiguous embedding name {name!r}: unsupported exact name "
                f"collides with {locations}"
            )
        if len(refs) > 1:
            locations = sorted(ref.virtual_path for ref in refs)
            raise AssetError(f"ambiguous embedding name {name!r}: {locations}")
        if refs:
            return refs[0]
        if name in self._unsupported:
            raise AssetError(
                f"unsupported textual-inversion format: {name!r}; only safetensors is accepted"
            )
        return None


@dataclass(frozen=True)
class MountDef:
    """One granted directory, as configured."""

    id: str
    path: Path
    mode: str = "read"
    priority: int = 0

    def __post_init__(self) -> None:
        if not MOUNT_ID_PATTERN.match(self.id):
            raise MountsError(
                f"mount ids are lowercase alphanumerics and hyphens, "
                f"starting alphanumeric: {self.id!r}"
            )
        if self.mode not in MOUNT_MODES:
            raise MountsError(
                f"mount {self.id!r}: mode must be one of {MOUNT_MODES}, got {self.mode!r}"
            )
        priority = cast("object", self.priority)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise MountsError(f"mount {self.id!r}: priority must be an integer, got {priority!r}")


@dataclass(frozen=True)
class ReadyMountSnapshot:
    mount_id: str
    priority: int
    asset_kind: str
    refs: tuple[AssetRef, ...]


def parse_mounts(text: str, source: str = "mounts config") -> tuple[MountDef, ...]:
    """Decode and validate mounts config text. Order follows the file."""
    try:
        data: dict[str, object] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise MountsError(f"{source}: invalid TOML: {exc}") from exc
    unknown_top = set(data) - {"mounts"}
    if unknown_top:
        raise MountsError(
            f"{source}: unknown top-level keys {sorted(unknown_top)} "
            f"(everything lives under [mounts.<id>])"
        )
    tables = data.get("mounts", {})
    if not isinstance(tables, dict):
        raise MountsError(f"{source}: 'mounts' must be a table of mounts")
    mounts: list[MountDef] = []
    paths: dict[str, str] = {}
    for id_raw, entry_raw in cast("dict[object, object]", tables).items():
        mount_id = str(id_raw)
        where = f"{source}: [mounts.{mount_id}]"
        if not isinstance(entry_raw, dict):
            raise MountsError(f"{where}: must be a table")
        entry = cast("dict[object, object]", entry_raw)
        unknown = {str(key) for key in entry} - _ENTRY_KEYS
        if unknown:
            raise MountsError(f"{where}: unknown keys {sorted(unknown)}")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise MountsError(f"{where}: 'path' must be a non-empty path string")
        holder = paths.get(path)
        if holder is not None:
            raise MountsError(
                f"{where}: path {path!r} is already mounted as "
                f"[mounts.{holder}] - one directory, one mount"
            )
        paths[path] = mount_id
        mode = entry.get("mode", "read")
        if not isinstance(mode, str):
            raise MountsError(f"{where}: 'mode' must be a string")
        priority = entry.get("priority", 0)
        try:
            mounts.append(
                MountDef(
                    id=mount_id,
                    path=Path(path),
                    mode=mode,
                    priority=cast("int", priority),
                )
            )
        except MountsError as exc:
            raise MountsError(f"{where}: {exc}") from None
    return tuple(mounts)


def load_mounts(path: Path) -> tuple[MountDef, ...]:
    """Parse the mounts config file; a missing file is an empty table
    (an install with no grants is valid, not an error)."""
    if not path.is_file():
        return ()
    return parse_mounts(path.read_text("utf-8"), str(path))


def _toml_string(value: str) -> str:
    """A TOML basic string. JSON string escaping is a subset of TOML's
    basic-string escapes, so this is exact, including Windows paths."""
    return json.dumps(value)


def dump_mounts(mounts: Sequence[MountDef]) -> str:
    """Deterministic config text (sorted by id); ``parse_mounts`` of the
    output round-trips exactly. Default mode and priority are omitted."""
    chunks: list[str] = []
    for mount in sorted(mounts, key=lambda entry: entry.id):
        lines = [
            f"[mounts.{mount.id}]",
            f"path = {_toml_string(str(mount.path))}",
        ]
        if mount.mode != "read":
            lines.append(f"mode = {_toml_string(mount.mode)}")
        if mount.priority != 0:
            lines.append(f"priority = {mount.priority}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + ("\n" if chunks else "")


class _MountRow:
    """Live state for one mount. States: pending (registered, not yet
    scanned), scanning (publishing entries as they are indexed), ready, failed
    (directory missing/unreadable - recorded, never silently dropped)."""

    def __init__(self, mount: MountDef, source: str, kind: str) -> None:
        self.mount = mount
        self.source = source
        self.kind = kind
        self.state = "pending"
        self.error: str | None = None
        self.library: LocalAssetLibrary | None = None
        self.entry_count: int | None = None
        self.progress: AssetScanProgress | None = None
        self.scan_started: float | None = None


class MountTable:
    """The engine-owned live mount table.

    Mutations (``add``/``remove``) are event-loop-cheap; ``scan`` is the
    blocking piece (hashing a model library is real work) and is meant to
    run on a thread - a lock guards row state so a scan finishing off-loop
    never races an add/remove on it. Every state change that affects what
    workers can resolve rewrites the snapshot file atomically.
    """

    def __init__(
        self,
        snapshot_path: Path | str | None = None,
        *,
        index_root: Path | str | None = None,
    ) -> None:
        self._rows: dict[str, _MountRow] = {}
        self._lock = threading.RLock()
        self._snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self._index_root = Path(index_root) if index_root is not None else None

    def add(self, mount: MountDef, *, source: str = "config", kind: str = "") -> None:
        """Register a mount (state "pending"; call ``scan`` to catalog it).
        A missing directory is not an add-time error - a config mount whose
        drive is unplugged should surface as a failed row, not take the
        whole server down - but a duplicate id is always a refusal."""
        if kind:
            try:
                require_asset_kind(kind)
            except AssetError as exc:
                raise MountsError(f"mount {mount.id!r}: {exc}") from exc
        with self._lock:
            if mount.id in self._rows:
                raise MountsError(f"mount id {mount.id!r} already exists")
            self._rows[mount.id] = _MountRow(mount, source, kind)

    def remove(self, mount_id: str) -> MountDef:
        """Drop a mount and republish the snapshot without it. Documents
        referencing its assets stay valid (identity is the digest); they
        just stop being materializable here until re-granted."""
        with self._lock:
            row = self._rows.pop(mount_id, None)
        if row is None:
            raise MountsError(f"unknown mount: {mount_id!r}")
        self.write_snapshot()
        return row.mount

    def get(self, mount_id: str) -> MountDef | None:
        with self._lock:
            row = self._rows.get(mount_id)
        return row.mount if row is not None else None

    def source(self, mount_id: str) -> str | None:
        with self._lock:
            row = self._rows.get(mount_id)
        return row.source if row is not None else None

    def kind(self, mount_id: str) -> str:
        """The mount-wide semantic asset kind, or empty when unclassified."""
        with self._lock:
            row = self._rows.get(mount_id)
        return row.kind if row is not None else ""

    def mounts_for_kind(self, kind: str) -> tuple[str, ...]:
        """Mount ids for a semantic kind in authoritative priority order."""
        with self._lock:
            return tuple(
                row.mount.id
                for row in sorted(
                    (row for row in self._rows.values() if row.kind == kind),
                    key=lambda row: (row.mount.priority, row.mount.id),
                )
            )

    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(mount_id for mount_id, row in self._rows.items() if row.state == "pending")

    def config_defs(self) -> tuple[MountDef, ...]:
        """The mounts that belong in ``mounts.toml``: config-sourced only.
        Derived mounts (e.g. a ComfyUI install's directories) are re-derived
        every boot and never persisted."""
        with self._lock:
            return tuple(row.mount for row in self._rows.values() if row.source == "config")

    def descriptors(self) -> list[dict[str, object]]:
        """Wire rows for GET /api/mounts, sorted by id.

        Operator-granted and conventional derived mounts expose their real
        paths on this settings surface. Semantic model mounts are picker
        infrastructure derived from ComfyUI's live folder table, not grants;
        they expose only stable ids and kinds so an arbitrary number of roots
        remains addressable without publishing host paths or path-bearing
        failure text.
        """
        with self._lock:
            rows = sorted(self._rows.values(), key=lambda row: row.mount.id)
            out: list[dict[str, object]] = []
            for row in rows:
                hidden_semantic_root = row.source == "derived" and bool(row.kind)
                wire: dict[str, object] = {
                    "id": row.mount.id,
                    "mode": row.mount.mode,
                    "priority": row.mount.priority,
                    "source": row.source,
                    "state": row.state,
                }
                if row.kind:
                    wire["kind"] = row.kind
                if not hidden_semantic_root:
                    wire["path"] = str(row.mount.path)
                if row.entry_count is not None:
                    wire["entryCount"] = row.entry_count
                progress = self._current_progress(row)
                if progress is not None:
                    wire["scanProgress"] = {
                        "filesDone": progress.files_done,
                        "filesTotal": progress.files_total,
                        "bytesDone": progress.bytes_done,
                        "bytesTotal": progress.bytes_total,
                        "elapsedSeconds": progress.elapsed_seconds,
                    }
                if row.error is not None:
                    wire["error"] = "model root unavailable" if hidden_semantic_root else row.error
                out.append(wire)
            return out

    @staticmethod
    def _current_progress(row: _MountRow) -> AssetScanProgress | None:
        progress = row.progress
        if progress is None or row.scan_started is None:
            return progress
        return AssetScanProgress(
            progress.files_done,
            progress.files_total,
            progress.bytes_done,
            progress.bytes_total,
            max(progress.elapsed_seconds, time.monotonic() - row.scan_started),
        )

    def scan_progress(self, mount_id: str) -> AssetScanProgress | None:
        with self._lock:
            row = self._rows.get(mount_id)
            return self._current_progress(row) if row is not None else None

    def scan(
        self,
        mount_id: str,
        *,
        on_progress: Callable[[AssetScanProgress], None] | None = None,
        progress_interval: float = 1.0,
    ) -> int:
        """(Re)catalog one mount. BLOCKING - hashes new/changed files - so
        run it on a thread. On success the mount is "ready" and enters the
        worker snapshot; on failure it is "failed" with the error recorded
        (never silently absent). Returns the entry count."""
        with self._lock:
            row = self._rows.get(mount_id)
            if row is None:
                raise MountsError(f"unknown mount: {mount_id!r}")
            row.state = "scanning"
            row.error = None
            row.progress = None
            row.scan_started = time.monotonic()
        try:
            library = row.library
            if library is None:
                index_path = (
                    self._index_root / f"{row.mount.id}.json"
                    if self._index_root is not None
                    else None
                )
                library = LocalAssetLibrary(
                    row.mount.path,
                    namespace=f"{MOUNT_NAMESPACE}/{row.mount.id}",
                    index_path=index_path,
                    legacy_index_path=(
                        row.mount.path / INDEX_NAME if index_path is not None else None
                    ),
                )

            with self._lock:
                if self._rows.get(mount_id) is row:
                    row.library = library

            def progress(update: AssetScanProgress) -> None:
                with self._lock:
                    if self._rows.get(mount_id) is not row:
                        return
                    row.progress = update
                    row.entry_count = update.files_done
                self.write_snapshot()
                if on_progress is not None:
                    on_progress(update)

            count = library.scan(
                on_progress=progress,
                progress_interval=progress_interval,
            )
        except (AssetError, OSError) as exc:
            with self._lock:
                if self._rows.get(mount_id) is row:
                    row.state = "failed"
                    row.error = str(exc)
                    row.library = None
                    row.entry_count = None
                    row.progress = None
                    row.scan_started = None
            raise
        with self._lock:
            if self._rows.get(mount_id) is not row:
                # Removed while scanning: the removal already republished
                # the snapshot; this scan's result is nobody's state.
                return count
            row.library = library
            row.entry_count = count
            self.write_snapshot()
            row.state = "ready"
            row.progress = None
            row.scan_started = None
        return count

    def entries(self, mount_id: str) -> tuple[AssetEntry, ...]:
        """Entries indexed so far for one mount (empty until scanning starts)."""
        with self._lock:
            row = self._rows.get(mount_id)
            if row is None:
                raise MountsError(f"unknown mount: {mount_id!r}")
            library = row.library
        return library.entries() if library is not None else ()

    def ready_snapshot(self) -> tuple[ReadyMountSnapshot, ...]:
        """Capture indexed mount rows and their exact catalog generation."""
        with self._lock:
            return tuple(
                ReadyMountSnapshot(
                    row.mount.id,
                    row.mount.priority,
                    row.kind,
                    tuple(entry.ref() for entry in row.library.entries()),
                )
                for row in sorted(
                    self._rows.values(),
                    key=lambda row: (row.mount.priority, row.mount.id),
                )
                if row.library is not None
            )

    def list_folder(self, mount_id: str, path: str = "") -> FolderListing:
        """One level of a mount's virtual tree (file-picker navigation).
        ``path`` is relative to the mount ("" = its root)."""
        with self._lock:
            row = self._rows.get(mount_id)
            if row is None:
                raise MountsError(f"unknown mount: {mount_id!r}")
            library = row.library
        if library is None:
            return FolderListing(folders=(), entries=())
        prefix = f"{MOUNT_NAMESPACE}/{mount_id}"
        if path:
            prefix = f"{prefix}/{path}"
        return library.list_folder(prefix)

    def writable_root(self, mount_id: str) -> Path:
        """MountWriteAuthority (engine-side): the root of a READY readwrite
        mount, or a refusal that names why - the same contract workers get
        from MountSnapshotWriter."""
        with self._lock:
            row = self._rows.get(mount_id)
            if row is None:
                raise AssetError(f"no mount {mount_id!r}: it was never granted or was revoked")
            if row.mount.mode != "readwrite":
                raise AssetError(
                    f"mount {mount_id!r} is granted read-only; saving needs a readwrite mount"
                )
            if row.state != "ready":
                raise AssetError(
                    f"mount {mount_id!r} is not ready (state {row.state!r}"
                    + (f": {row.error}" if row.error else "")
                    + ")"
                )
            return row.mount.path

    def resolve(self, digest: str) -> Path | None:
        """AssetResolver across mounts with indexed entries (engine-side)."""
        with self._lock:
            libraries = [
                cast("LocalAssetLibrary", row.library)
                for row in sorted(
                    (row for row in self._rows.values() if row.library is not None),
                    key=lambda row: (row.mount.priority, row.mount.id),
                )
            ]
        for library in libraries:
            path = library.resolve(digest)
            if path is not None:
                return path
        return None

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        with self._lock:
            libraries = [
                cast("LocalAssetLibrary", row.library)
                for row in sorted(
                    (row for row in self._rows.values() if row.library is not None),
                    key=lambda row: (row.mount.priority, row.mount.id),
                )
            ]
        for library in libraries:
            resolution = library.resolve_asset(digest)
            if resolution is not None:
                return resolution
        return None

    def ref(self, virtual_path: str) -> AssetRef:
        """Mint a resolver-bound AssetRef for ``mounts/<id>/...``."""
        parts = virtual_path.split("/", 2)
        if len(parts) < 3 or parts[0] != MOUNT_NAMESPACE:
            raise AssetError(
                f"not a mount virtual path (expected 'mounts/<id>/...'): {virtual_path!r}"
            )
        with self._lock:
            row = self._rows.get(parts[1])
            library = row.library if row is not None else None
        if library is None:
            raise AssetError(
                f"asset {virtual_path!r} is not indexed because its mount is not ready"
            )
        try:
            ref = library.ref(virtual_path)
        except AssetError as exc:
            raise AssetError(f"asset {virtual_path!r} is not indexed yet") from exc
        if row is not None and row.state == "scanning":
            library.persist_index()
            self.write_snapshot()
        return ref

    def write_snapshot(self) -> None:
        """Publish each indexed mount's root and current index path.

        Scanning mounts participate once their first entries are available.
        Atomic replacement keeps workers from observing a torn snapshot.
        """
        if self._snapshot_path is None:
            return
        with self._lock:
            rows = [
                {
                    "id": row.mount.id,
                    "root": str(row.mount.path),
                    "index": str(row.library.index_path),
                    "mode": row.mount.mode,
                    "kind": row.kind,
                    "priority": row.mount.priority,
                }
                for row in sorted(
                    self._rows.values(),
                    key=lambda row: (row.mount.priority, row.mount.id),
                )
                if row.library is not None
            ]
            serialized = json.dumps({"mounts": rows}, indent=1)
            tmp = self._snapshot_path.with_name(
                self._snapshot_path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
            )
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(serialized, "utf-8")
            os.replace(tmp, self._snapshot_path)


class MountSnapshotResolver:
    """AssetResolver over the engine-published mount snapshot, stdlib-only.

    The worker half of runtime mounts: bound once at type registration,
    it re-checks the snapshot file (mtime+size) on every resolve and
    rebuilds its per-mount :class:`IndexedAssetResolver` chain when the
    engine rewrote it - so a folder granted mid-session materializes in a
    live worker with no restart and no control-channel message. A missing
    or empty snapshot resolves nothing (the engine simply has not scanned
    yet); a mount whose index vanished is skipped, not fatal.
    """

    def __init__(self, snapshot_path: Path | str) -> None:
        self._path = Path(snapshot_path)
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._resolvers: list[tuple[str, IndexedAssetResolver]] = []

    def resolve(self, digest: str) -> Path | None:
        for _kind, resolver in self._refresh():
            path = resolver.resolve(digest)
            if path is not None:
                return path
        return None

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        for _kind, resolver in self._refresh():
            resolution = resolver.resolve_asset(digest)
            if resolution is not None:
                return resolution
        return None

    def resolve_for_kind(self, digest: str, kind: str) -> Path | None:
        """Resolve in the first ranked mount carrying ``kind``.

        Model widgets use this instead of the general vault/mount chain so
        an equal digest in a vault or another model category cannot redirect
        execution. Equal content in multiple roots is resolved by the
        snapshot's authoritative priority and mount-id order.
        """
        require_asset_kind(kind)
        for row_kind, resolver in self._refresh():
            if row_kind != kind:
                continue
            path = resolver.resolve(digest)
            if path is not None:
                return path
        return None

    def resolve_asset_for_kind(self, digest: str, kind: str) -> AssetResolution | None:
        require_asset_kind(kind)
        for row_kind, resolver in self._refresh():
            if row_kind != kind:
                continue
            resolution = resolver.resolve_asset(digest)
            if resolution is not None:
                return resolution
        return None

    def _refresh(self) -> tuple[tuple[str, IndexedAssetResolver], ...]:
        with self._lock:
            self._refresh_locked()
            return tuple(self._resolvers)

    def _refresh_locked(self) -> None:
        try:
            stat = self._path.stat()
            stamp: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        if stamp == self._stamp:
            return
        self._stamp = stamp
        self._resolvers = []
        if stamp is None:
            return
        try:
            loaded: object = json.loads(self._path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(loaded, dict):
            return
        rows = cast("dict[str, object]", loaded).get("mounts")
        if not isinstance(rows, list):
            return
        prepared: list[tuple[int, str, str, str, str]] = []
        for row in cast("list[object]", rows):
            if not isinstance(row, dict):
                continue
            entry = cast("dict[str, object]", row)
            mount_id, root, index, kind, priority = (
                entry.get("id"),
                entry.get("root"),
                entry.get("index"),
                entry.get("kind", ""),
                entry.get("priority", 0),
            )
            if (
                not isinstance(mount_id, str)
                or not isinstance(root, str)
                or not isinstance(index, str)
                or isinstance(priority, bool)
                or not isinstance(priority, int)
            ):
                continue
            if not isinstance(kind, str):
                kind = ""
            prepared.append((priority, mount_id, kind, root, index))
        for _priority, _mount_id, kind, root, index in sorted(prepared):
            try:
                self._resolvers.append((kind, IndexedAssetResolver(root, index_path=index)))
            except AssetError:
                continue  # index gone since the snapshot: that mount is dark
