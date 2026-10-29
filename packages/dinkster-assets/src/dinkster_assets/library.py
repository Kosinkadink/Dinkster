"""LocalAssetLibrary: a real directory scanned into the virtual namespace.

The first AssetResolver implementation (DESIGN 3.12): existing model
directories become catalogs without moving a byte. Scanning hashes every
file once and remembers digests plus descriptor metadata in an index file.
Rescans and loads compare that metadata instead of rereading multi-GB payloads.
A mismatch fails closed and requires an explicit rescan to ingest the changed
file. Legacy index rows without a verification record retain load-time hashing.

The library is both sides of the asset story for local files: a catalog
populator (virtual paths under a namespace prefix mirror the directory
layout) and a resolver (digest -> real path) for materialization.
"""

from __future__ import annotations

import importlib
import json
import mimetypes
import os
import re
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .catalog import AssetCatalog, AssetEntry, FolderListing, glob_regex
from .identity import AssetError, is_digest
from .integrity import AssetVerificationRecord, digest_file_with_record
from .model import AssetRef, AssetResolution

INDEX_NAME = ".dinkster-asset-index.json"


@dataclass(frozen=True)
class AssetScanProgress:
    files_done: int
    files_total: int
    bytes_done: int
    bytes_total: int
    elapsed_seconds: float


WRITES_SIDECAR_NAME = ".dinkster-asset-writes.jsonl"
"""Append-only log of files an AssetWriter just landed in this root.

The full index is rewritten only by a scan, and a scan of a model library
is real work - but a freshly saved image must resolve by digest NOW (the
frontend previews it seconds after the job finishes). So writers append
one self-contained JSON line per file here, resolvers consult it after the
index misses, and the next scan absorbs its rows into the index (their
digests are trusted via the same size+mtime key, so just-written files are
never rehashed) and prunes them from the sidecar."""


class _Fcntl(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, fd: int, operation: int, /) -> None: ...


@contextmanager
def _sidecar_lock(root: Path) -> Generator[None]:
    """Serialize sidecar append/prune across processes when flock exists."""
    lock_path = root / f"{WRITES_SIDECAR_NAME}.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl = cast("_Fcntl", importlib.import_module("fcntl"))
        except ImportError:  # pragma: no cover - non-POSIX fallback
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_write_record(
    root: Path,
    relative: str,
    digest: str,
    size: int,
    mtime_ns: int,
    verification: AssetVerificationRecord | None = None,
) -> None:
    """Record one just-written file. Appends are one small line each, so
    concurrent writers (multiple worker processes saving into one mount)
    interleave whole records, never bytes of records."""
    line = json.dumps(
        {
            "path": relative,
            "digest": digest,
            "size": size,
            "mtimeNs": mtime_ns,
            **({"verification": verification.to_json()} if verification is not None else {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with _sidecar_lock(root):
        with (root / WRITES_SIDECAR_NAME).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_write_records(root: Path) -> dict[str, dict[str, object]]:
    """The sidecar's current rows, path-keyed, later lines winning.
    Malformed lines (a torn write from a crashed process) are skipped,
    never fatal - the file is a cache of scan-pending truth, and the
    files themselves are still on disk for the next scan."""
    rows: dict[str, dict[str, object]] = {}
    try:
        text = (root / WRITES_SIDECAR_NAME).read_text("utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            loaded: object = json.loads(line)
        except ValueError:
            continue
        if not isinstance(loaded, dict):
            continue
        row = cast("dict[str, object]", loaded)
        relative = row.get("path")
        digest = row.get("digest")
        size = row.get("size")
        mtime = row.get("mtimeNs")
        if (
            isinstance(relative, str)
            and isinstance(digest, str)
            and is_digest(digest)
            and isinstance(size, int)
            and isinstance(mtime, int)
        ):
            verification = AssetVerificationRecord.from_json(digest, row.get("verification"))
            rows[relative] = {
                "size": size,
                "mtimeNs": mtime,
                "digest": digest,
                **({"verification": verification.to_json()} if verification is not None else {}),
            }
    return rows


def _resolve_from_sidecar(root: Path, digest: str) -> Path | None:
    """Digest -> path via the writes sidecar, using size+mtime as the same
    cache-invalidation heuristic as index hits."""
    for relative, row in load_write_records(root).items():
        if row.get("digest") != digest:
            continue
        path = root / relative
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size == row.get("size") and stat.st_mtime_ns == row.get("mtimeNs"):
            return path
    return None


def _resolve_asset_from_sidecar(root: Path, digest: str) -> AssetResolution | None:
    for relative, row in load_write_records(root).items():
        if row.get("digest") != digest:
            continue
        path = root / relative
        if not path.is_file():
            continue
        record = AssetVerificationRecord.from_json(digest, row.get("verification"))
        return AssetResolution(path, record)
    return None


DEFAULT_IGNORE: tuple[str, ...] = (
    "put_*_here",  # ComfyUI folder placeholders
    "Thumbs.db",  # Windows Explorer thumbnails
    "desktop.ini",  # Windows folder config
    "*.tmp",  # partial downloads: generic
    "*.part",  # partial downloads: firefox/wget
    "*.crdownload",  # partial downloads: chrome
    "*.aria2",  # partial downloads: aria2 control files
)
"""Files that are never assets: placeholders, OS junk, partial downloads.

Passing ``ignore=`` to LocalAssetLibrary replaces this list; include it
explicitly to extend rather than replace.
"""


class _IgnoreRules:
    """Gitignore-flavored matching: a pattern with ``/`` anchors to the
    relative path (catalog glob semantics: ``*`` stays within a folder,
    ``**`` crosses); a pattern without ``/`` matches the file name at any
    depth."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self._name_rules: list[re.Pattern[str]] = []
        self._path_rules: list[re.Pattern[str]] = []
        for pattern in patterns:
            if "/" in pattern:
                self._path_rules.append(glob_regex(pattern))
            else:
                self._name_rules.append(glob_regex(pattern))

    def matches(self, relative_path: str, name: str) -> bool:
        return any(rule.match(name) for rule in self._name_rules) or any(
            rule.match(relative_path) for rule in self._path_rules
        )


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


class LocalAssetLibrary:
    """Scan ``root`` into virtual paths ``<namespace>/<relative-path>``.

    ``index_path`` defaults to ``<root>/.dinkster-asset-index.json``; pass an
    explicit path to keep the root read-only. Hidden files (dot-prefixed),
    the index file itself, and anything matching ``ignore`` patterns
    (DEFAULT_IGNORE unless given: placeholders, OS junk, partial downloads)
    are skipped.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        namespace: str = "models",
        index_path: Path | str | None = None,
        legacy_index_path: Path | str | None = None,
        ignore: Iterable[str] = DEFAULT_IGNORE,
    ) -> None:
        self._root = Path(root)
        if not self._root.is_dir():
            raise AssetError(f"asset library root is not a directory: {self._root}")
        self._namespace = namespace
        self._index_path = Path(index_path) if index_path is not None else self._root / INDEX_NAME
        self._legacy_index_path = Path(legacy_index_path) if legacy_index_path is not None else None
        self._ignore = _IgnoreRules(ignore)
        self._lock = threading.RLock()
        self._index_write_lock = threading.RLock()
        self._catalog = AssetCatalog()
        self._paths_by_digest: dict[str, Path] = {}
        self._resolutions_by_digest: dict[str, AssetResolution] = {}
        self._current_index: dict[str, dict[str, object]] = {}

    @property
    def catalog(self) -> AssetCatalog:
        snapshot = AssetCatalog()
        with self._lock:
            for entry in self._catalog.entries():
                snapshot.add(entry)
        return snapshot

    @property
    def index_path(self) -> Path:
        """Where this library persists digests - what a worker-facing
        snapshot must point its IndexedAssetResolver at."""
        return self._index_path

    def scan(
        self,
        *,
        on_progress: Callable[[AssetScanProgress], None] | None = None,
        progress_interval: float = 1.0,
    ) -> int:
        """(Re)build the catalog from the directory. Returns the number of
        cataloged files. Digests come from the index when the size+mtime
        heuristic matches, from hashing otherwise; the index is rewritten
        after every scan."""
        if progress_interval < 0:
            raise ValueError("progress_interval must be nonnegative")
        started = time.monotonic()
        index = self._load_index()
        # Just-written files carry known digests in the writes sidecar;
        # absorbing them here means a scan right after a save costs a
        # stat, not a rehash.
        index.update(load_write_records(self._root))
        candidates: list[tuple[Path, str, os.stat_result]] = []
        for path in sorted(self._root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if any(part.startswith(".") for part in path.relative_to(self._root).parts):
                continue
            if path == self._index_path:
                continue
            relative = path.relative_to(self._root).as_posix()
            if self._ignore.matches(relative, path.name):
                continue
            stat = path.stat()
            candidates.append((path, relative, stat))

        fresh_index: dict[str, dict[str, object]] = {}
        fresh_catalog = AssetCatalog()
        fresh_paths_by_digest: dict[str, Path] = {}
        fresh_resolutions_by_digest: dict[str, AssetResolution] = {}
        files_total = len(candidates)
        bytes_total = sum(stat.st_size for _, _, stat in candidates)
        files_done = 0
        bytes_done = 0
        last_report = started

        def report(*, force: bool = False) -> None:
            nonlocal last_report
            if on_progress is None:
                return
            now = time.monotonic()
            if not force and now - last_report < progress_interval:
                return
            if files_done:
                self.persist_index()
            last_report = now
            on_progress(
                AssetScanProgress(
                    files_done,
                    files_total,
                    bytes_done,
                    bytes_total,
                    now - started,
                )
            )

        report(force=True)
        pending: list[tuple[Path, str, os.stat_result]] = []
        for path, relative, stat in candidates:
            cached = index.get(relative)
            digest: str | None = None
            record: AssetVerificationRecord | None = None
            if isinstance(cached, Mapping):
                row = cast("Mapping[str, object]", cached)
                if row.get("size") == stat.st_size and row.get("mtimeNs") == stat.st_mtime_ns:
                    candidate = row.get("digest")
                    if isinstance(candidate, str) and is_digest(candidate):
                        candidate_record = AssetVerificationRecord.from_json(
                            candidate, row.get("verification")
                        )
                        if candidate_record is not None and candidate_record.matches(stat):
                            digest = candidate
                            record = candidate_record
            if digest is None:
                pending.append((path, relative, stat))
                continue
            with self._lock:
                fresh_index[relative] = {
                    "size": stat.st_size,
                    "mtimeNs": stat.st_mtime_ns,
                    "digest": digest,
                    **({"verification": record.to_json()} if record is not None else {}),
                }
            fresh_catalog.add(
                AssetEntry(
                    virtual_path=f"{self._namespace}/{relative}",
                    digest=digest,
                    size=stat.st_size,
                    media_type=_guess_media_type(path),
                )
            )
            fresh_paths_by_digest.setdefault(digest, path)
            fresh_resolutions_by_digest.setdefault(digest, AssetResolution(path, record))
            files_done += 1
            bytes_done += stat.st_size

        with self._lock:
            self._catalog = fresh_catalog
            self._paths_by_digest = fresh_paths_by_digest
            self._resolutions_by_digest = fresh_resolutions_by_digest
            self._current_index = fresh_index
        report(force=True)
        for path, relative, stat in pending:
            digest, record = digest_file_with_record(path)
            stat = path.stat()
            if record is not None and not record.matches(stat):
                raise AssetError(f"asset path changed while being ingested: {path}")
            with self._lock:
                fresh_index[relative] = {
                    "size": stat.st_size,
                    "mtimeNs": stat.st_mtime_ns,
                    "digest": digest,
                    **({"verification": record.to_json()} if record is not None else {}),
                }
            self._publish_entry(path, relative, stat.st_size, digest, record)
            files_done += 1
            bytes_done += stat.st_size
            report()
        self._save_index(fresh_index)
        self._prune_sidecar(fresh_index)
        report(force=True)
        return files_done

    def _publish_entry(
        self,
        path: Path,
        relative: str,
        size: int,
        digest: str,
        record: AssetVerificationRecord | None,
    ) -> None:
        with self._lock:
            self._catalog.add(
                AssetEntry(
                    virtual_path=f"{self._namespace}/{relative}",
                    digest=digest,
                    size=size,
                    media_type=_guess_media_type(path),
                )
            )
            self._paths_by_digest.setdefault(digest, path)
            self._resolutions_by_digest.setdefault(digest, AssetResolution(path, record))

    def persist_index(self) -> None:
        with self._index_write_lock:
            with self._lock:
                current = {relative: dict(row) for relative, row in self._current_index.items()}
            self._save_index(current)

    def _prune_sidecar(self, fresh_index: dict[str, dict[str, object]]) -> None:
        """Drop sidecar rows the index now covers. Rows for files written
        DURING the scan (not yet in fresh_index) survive, so a save racing
        a scan never loses its immediate resolvability."""
        sidecar = self._root / WRITES_SIDECAR_NAME
        try:
            with _sidecar_lock(self._root):
                remaining = {
                    relative: row
                    for relative, row in load_write_records(self._root).items()
                    if relative not in fresh_index
                }
                if not remaining:
                    sidecar.unlink(missing_ok=True)
                    return
                tmp = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
                tmp.write_text(
                    "".join(
                        json.dumps({"path": relative, **row}, sort_keys=True) + "\n"
                        for relative, row in remaining.items()
                    ),
                    "utf-8",
                )
                os.replace(tmp, sidecar)
        except OSError:
            pass  # a read-only root keeps its sidecar; resolution still works

    def resolve(self, digest: str) -> Path | None:
        """AssetResolver: digest -> real path, if this library holds it
        (cataloged at the last scan, or written since via the sidecar)."""
        with self._lock:
            path = self._paths_by_digest.get(digest)
        if path is not None:
            return path
        return _resolve_from_sidecar(self._root, digest)

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        with self._lock:
            resolution = self._resolutions_by_digest.get(digest)
        if resolution is not None:
            return resolution
        return _resolve_asset_from_sidecar(self._root, digest)

    def digests(self) -> list[str]:
        """Every distinct identity this library holds (announcements)."""
        with self._lock:
            return list(self._paths_by_digest)

    def resolver(self) -> IndexedAssetResolver:
        """A read-only resolver over this library's persisted index - what a
        worker process gets instead of the scanning library itself."""
        return IndexedAssetResolver(self._root, index_path=self._index_path)

    def ref(self, virtual_path: str) -> AssetRef:
        """Mint a resolver-bound AssetRef for a cataloged virtual path -
        the host-side entry point for feeding assets into graphs."""
        with self._lock:
            entry = self._catalog.get(virtual_path)
        if entry is None:
            raise AssetError(f"no asset cataloged at virtual path: {virtual_path!r}")
        return entry.ref(resolver=self)

    def refresh_writes(self) -> None:
        """Publish valid writer-sidecar rows into the live catalog."""
        for relative, row in load_write_records(self._root).items():
            path = self._root / relative
            try:
                stat = path.stat()
            except OSError:
                continue
            digest, size, mtime_ns = row.get("digest"), row.get("size"), row.get("mtimeNs")
            if (
                not isinstance(digest, str)
                or not isinstance(size, int)
                or not isinstance(mtime_ns, int)
                or (stat.st_size, stat.st_mtime_ns) != (size, mtime_ns)
            ):
                continue
            record = AssetVerificationRecord.from_json(digest, row.get("verification"))
            self._publish_entry(path, relative, size, digest, record)

    def entries(self) -> tuple[AssetEntry, ...]:
        self.refresh_writes()
        with self._lock:
            return self._catalog.entries()

    def list_folder(self, prefix: str = "") -> FolderListing:
        self.refresh_writes()
        with self._lock:
            return self._catalog.list_folder(prefix)

    def _load_index(self) -> dict[str, object]:
        source = self._index_path
        if (
            not source.exists()
            and self._legacy_index_path is not None
            and self._legacy_index_path != source
        ):
            source = self._legacy_index_path
        try:
            loaded: object = json.loads(source.read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {str(key): val for key, val in cast("dict[object, object]", loaded).items()}

    def _save_index(self, index: dict[str, dict[str, object]]) -> None:
        """Atomic (temp + os.replace) because the index is *shared* state:
        multiple Dinkster instances pointing at one model folder share hash
        work through this file, and a torn write another instance reads as
        malformed JSON would cost that instance a full rehash. Unchanged
        indexes are not rewritten - concurrent scanners of a stable library
        do not race each other over identical bytes."""
        with self._index_write_lock:
            serialized = json.dumps(index, indent=1)
            try:
                if self._index_path.read_text("utf-8") == serialized:
                    return
            except (OSError, ValueError):
                pass
            tmp = self._index_path.with_name(self._index_path.name + f".tmp-{os.getpid()}")
            try:
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(serialized, "utf-8")
                os.replace(tmp, self._index_path)
            except OSError:
                # A read-only root degrades to rehash-on-scan, not failure.
                try:
                    tmp.unlink()
                except OSError:
                    pass


class IndexedAssetResolver:
    """AssetResolver over a persisted index file, stdlib-only by design.

    A worker process resolves digests the graph references; it never
    catalogs and never hashes (it may not even have blake3 - compat
    children ride a foreign interpreter). This reads the index a scanning
    LocalAssetLibrary wrote and answers digest -> path, re-checking
    size+mtime per hit as a cache-invalidation heuristic. A same-size,
    same-mtime replacement can pass this check despite different content.
    Recorded descriptor metadata closes that gap without rereading the payload;
    legacy rows retain descriptor-bound hashing at consumption.
    """

    def __init__(self, root: Path | str, *, index_path: Path | str | None = None) -> None:
        self._root = Path(root)
        self._index_path = Path(index_path) if index_path is not None else self._root / INDEX_NAME
        self._rows_by_digest: dict[
            str, list[tuple[str, int, int, AssetVerificationRecord | None]]
        ] = {}
        try:
            loaded: object = json.loads(self._index_path.read_text("utf-8"))
        except (OSError, ValueError) as error:
            raise AssetError(
                f"no readable asset index at {self._index_path}: {error} "
                "(scan the library where Dinkster is installed first)"
            ) from None
        if not isinstance(loaded, Mapping):
            raise AssetError(f"malformed asset index at {self._index_path}")
        for relative, row in cast("Mapping[object, object]", loaded).items():
            if not isinstance(row, Mapping):
                continue
            entry = cast("Mapping[str, object]", row)
            digest, size, mtime = entry.get("digest"), entry.get("size"), entry.get("mtimeNs")
            if isinstance(digest, str) and isinstance(size, int) and isinstance(mtime, int):
                record = AssetVerificationRecord.from_json(digest, entry.get("verification"))
                self._rows_by_digest.setdefault(digest, []).append(
                    (str(relative), size, mtime, record)
                )

    def resolve(self, digest: str) -> Path | None:
        """AssetResolver: digest -> a path whose size+mtime still match the
        cached row - or a file written since the last scan,
        via the writes sidecar (re-read per miss: writes are rare and the
        sidecar is small, while a snapshot-frozen index would otherwise
        make a just-saved output unresolvable until the next scan)."""
        for relative, size, mtime_ns, _record in self._rows_by_digest.get(digest, ()):
            path = self._root / relative
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == size and stat.st_mtime_ns == mtime_ns:
                return path
        return _resolve_from_sidecar(self._root, digest)

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        stale: AssetResolution | None = None
        for relative, _size, _mtime_ns, record in self._rows_by_digest.get(digest, ()):
            path = self._root / relative
            try:
                stat = path.stat()
            except OSError:
                continue
            resolution = AssetResolution(path, record)
            if record is not None and record.matches(stat):
                return resolution
            if stale is None:
                stale = resolution
        return stale or _resolve_asset_from_sidecar(self._root, digest)
