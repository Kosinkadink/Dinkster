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
import stat
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from dinkster_library.index_format import DEFAULT_IGNORE as DEFAULT_IGNORE
from dinkster_library.index_format import INDEX_NAME as INDEX_NAME
from dinkster_library.index_format import PARTIAL_DOWNLOAD_EXTENSIONS as PARTIAL_DOWNLOAD_EXTENSIONS
from dinkster_library.index_format import WRITES_SIDECAR_NAME as WRITES_SIDECAR_NAME
from dinkster_library.index_format import IgnoreRules, retain_missing_versions

from .catalog import AssetCatalog, AssetEntry, FolderListing, validate_virtual_path
from .identity import AssetError, is_digest
from .integrity import AssetVerificationRecord, digest_file_with_record
from .model import AssetRef, AssetResolution


@dataclass(frozen=True)
class AssetScanProgress:
    files_done: int
    files_total: int
    bytes_done: int
    bytes_total: int
    elapsed_seconds: float


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
    *,
    published_at_ns: int | None = None,
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
            **({"publishedAtNs": published_at_ns} if published_at_ns is not None else {}),
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
                **(
                    {"publishedAtNs": row["publishedAtNs"]}
                    if type(row.get("publishedAtNs")) is int
                    else {}
                ),
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
        self._ignore = IgnoreRules(ignore)
        self._lock = threading.RLock()
        self._index_write_lock = threading.RLock()
        self._catalog = AssetCatalog()
        self._paths_by_digest: dict[str, Path] = {}
        self._resolutions_by_digest: dict[str, AssetResolution] = {}
        # Publication can precede the first scan; it must retain the worker's warm index.
        self._current_index: dict[str, dict[str, object]] = {
            relative: dict(cast("Mapping[str, object]", row))
            for relative, row in self._load_index().items()
            if isinstance(row, Mapping)
        }

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

    def folder_size(self, *, deadline: float) -> int | None:
        """Observe logical file bytes without reading contents. Run off-loop.

        Hidden/ignored files, partial downloads, the index and child symlinks
        are excluded. Each regular pathname counts once, including hardlinks.
        A failed or budget-limited traversal never reports a partial total.
        """
        try:
            root = self._root.resolve(strict=True)
            original = root.stat()
            visited = 0
            descriptor_reads = os.scandir in os.supports_fd and os.open in os.supports_dir_fd
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

            def read(directory: Path, descriptor: int | None) -> int | None:
                nonlocal visited
                if (
                    time.monotonic() >= deadline
                    or len(directory.relative_to(root).parts) >= 64
                    or directory.resolve(strict=True) != directory
                ):
                    return None
                before = directory.stat(follow_symlinks=False)
                if not stat.S_ISDIR(before.st_mode) or (
                    descriptor is not None and not os.path.samestat(before, os.fstat(descriptor))
                ):
                    return None
                total = 0
                with os.scandir(descriptor if descriptor is not None else directory) as entries:
                    for entry in entries:
                        visited += 1
                        if visited > 100_000 or time.monotonic() >= deadline:
                            return None
                        member = directory / entry.name
                        relative = member.relative_to(root).as_posix()
                        if (
                            entry.name.startswith(".")
                            or self._ignore.matches(relative, entry.name)
                            or member.suffix.lower() in PARTIAL_DOWNLOAD_EXTENSIONS
                            or member == self._index_path
                        ):
                            continue
                        metadata = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(metadata.st_mode) or getattr(
                            metadata, "st_file_attributes", 0
                        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                            continue
                        if stat.S_ISDIR(metadata.st_mode):
                            child = (
                                os.open(entry.name, flags, dir_fd=descriptor)
                                if descriptor is not None
                                else None
                            )
                            try:
                                size = read(member, child)
                            finally:
                                if child is not None:
                                    os.close(child)
                            if size is None:
                                return None
                            total += size
                        elif stat.S_ISREG(metadata.st_mode):
                            total += metadata.st_size
                if directory.resolve(strict=True) != directory or not os.path.samestat(
                    before, directory.stat(follow_symlinks=False)
                ):
                    return None
                return total

            descriptor = os.open(root, flags) if descriptor_reads else None
            try:
                total = read(root, descriptor)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            current = self._root.stat()
            if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                return None
            return total
        except (OSError, RuntimeError):
            return None

    def refresh_index(self) -> int:
        """Load a helper-published catalog without scanning or hashing files."""
        index = {
            relative: dict(cast("Mapping[str, object]", row))
            for relative, row in self._load_index().items()
            if isinstance(row, Mapping)
        }
        catalog = AssetCatalog()
        paths: dict[str, Path] = {}
        resolutions: dict[str, AssetResolution] = {}
        root = self._root.resolve()
        for relative, row in index.items():
            digest = row.get("digest")
            if row.get("missing") or not isinstance(digest, str) or not is_digest(digest):
                continue
            try:
                validate_virtual_path(relative)
                path = self._root / relative
                if not path.resolve(strict=True).is_relative_to(root):
                    continue
                metadata = path.stat()
                if (metadata.st_size, metadata.st_mtime_ns) != (
                    row.get("size"),
                    row.get("mtimeNs"),
                ):
                    continue
                record = AssetVerificationRecord.from_json(digest, row.get("verification"))
                if record is not None and not record.matches(path):
                    continue
                catalog.add(
                    AssetEntry(
                        virtual_path=f"{self._namespace}/{relative}",
                        digest=digest,
                        size=metadata.st_size,
                        media_type=_guess_media_type(path),
                        modified_at=metadata.st_mtime,
                    )
                )
                paths.setdefault(digest, path)
                resolutions.setdefault(digest, AssetResolution(path, record))
            except (AssetError, OSError, RuntimeError):
                continue
        with self._lock:
            self._current_index = index
            self._catalog = catalog
            self._paths_by_digest = paths
            self._resolutions_by_digest = resolutions
        return len(catalog.entries())

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
        index.update(
            (relative, row)
            for relative, row in load_write_records(self._root).items()
            if not self._ignore.matches(relative, Path(relative).name)
        )
        candidates: list[tuple[Path, str, os.stat_result]] = []
        for path in sorted(self._root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if any(part.startswith(".") for part in path.relative_to(self._root).parts):
                continue
            if path == self._index_path:
                continue
            if path.suffix.lower() in PARTIAL_DOWNLOAD_EXTENSIONS:
                continue
            relative = path.relative_to(self._root).as_posix()
            if self._ignore.matches(relative, path.name):
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            candidates.append((path, relative, stat))

        # A previous identity remains durable when its location disappears or changes.
        fresh_index: dict[str, dict[str, object]] = {
            relative: {**cast("Mapping[str, object]", row), "missing": True}
            for relative, row in index.items()
            if isinstance(row, Mapping)
        }
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
        if candidates:
            time.sleep(0.1)
        pending: list[tuple[Path, str, os.stat_result]] = []
        for path, relative, stat in candidates:
            try:
                settled = path.stat()
            except FileNotFoundError:
                continue
            if (stat.st_size, stat.st_mtime_ns) != (settled.st_size, settled.st_mtime_ns):
                continue
            stat = settled
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
                        if candidate_record is not None and candidate_record.matches(path):
                            digest = candidate
                            record = candidate_record
            if digest is None:
                pending.append((path, relative, stat))
                continue
            with self._lock:
                fresh_index[relative] = retain_missing_versions(
                    {
                        "size": stat.st_size,
                        "mtimeNs": stat.st_mtime_ns,
                        "digest": digest,
                        **({"verification": record.to_json()} if record is not None else {}),
                    },
                    index.get(relative),
                )
            fresh_catalog.add(
                AssetEntry(
                    virtual_path=f"{self._namespace}/{relative}",
                    digest=digest,
                    size=stat.st_size,
                    media_type=_guess_media_type(path),
                    modified_at=stat.st_mtime,
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
            try:
                digest, record = digest_file_with_record(path)
                stat = path.stat()
            except FileNotFoundError:
                continue
            if record is not None and not record.matches(path):
                raise AssetError(f"asset path changed while being ingested: {path}")
            with self._lock:
                fresh_index[relative] = retain_missing_versions(
                    {
                        "size": stat.st_size,
                        "mtimeNs": stat.st_mtime_ns,
                        "digest": digest,
                        **({"verification": record.to_json()} if record is not None else {}),
                    },
                    index.get(relative),
                )
            self._publish_entry(path, relative, stat.st_size, digest, record, stat.st_mtime)
            files_done += 1
            bytes_done += stat.st_size
            report()
        if self._save_index(fresh_index):
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
        modified_at: float,
    ) -> None:
        with self._lock:
            self._catalog.add(
                AssetEntry(
                    virtual_path=f"{self._namespace}/{relative}",
                    digest=digest,
                    size=size,
                    media_type=_guess_media_type(path),
                    modified_at=modified_at,
                )
            )
            self._paths_by_digest.setdefault(digest, path)
            self._resolutions_by_digest.setdefault(digest, AssetResolution(path, record))

    def persist_index(self, *, only_if_missing: bool = False) -> None:
        with self._index_write_lock:
            with self._lock:
                current = {relative: dict(row) for relative, row in self._current_index.items()}
            self._save_index(current, only_if_missing=only_if_missing)

    def index_file(
        self,
        relative: str,
        *,
        publish_index: Callable[[Path, Path, str], None] | None = None,
    ) -> AssetRef | None:
        """Verify and catalog one selected file without rescanning its directory."""
        try:
            validate_virtual_path(relative)
            if any(part.startswith(".") for part in Path(relative).parts) or self._ignore.matches(
                relative, Path(relative).name
            ):
                return None
            root = self._root.resolve(strict=True)
            path = root / relative
            if (
                not path.is_file()
                or path.suffix.lower() in PARTIAL_DOWNLOAD_EXTENSIONS
                or not path.resolve(strict=True).is_relative_to(root)
                or path == self._index_path.resolve()
            ):
                return None
        except (AssetError, OSError, RuntimeError):
            return None
        if publish_index is not None:
            publish_index(root, self._index_path, relative)
            self.refresh_index()
        try:
            if publish_index is None:
                digest, record = digest_file_with_record(path)
            else:
                with self._lock:
                    row = self._current_index.get(relative, {})
                digest = row.get("digest")
                if row.get("missing") or not isinstance(digest, str) or not is_digest(digest):
                    return None
                record = AssetVerificationRecord.from_json(digest, row.get("verification"))
            stat = path.stat()
            if (
                record is None
                or not record.matches(path)
                or not path.resolve(strict=True).is_relative_to(root)
            ):
                return None
        except (AssetError, OSError, RuntimeError):
            return None
        with self._index_write_lock:
            if publish_index is None:
                with self._lock:
                    index = dict(self._current_index)
                    index[relative] = retain_missing_versions(
                        {
                            "size": stat.st_size,
                            "mtimeNs": stat.st_mtime_ns,
                            "digest": digest,
                            "verification": record.to_json(),
                        },
                        index.get(relative),
                    )
                if not self._save_index(index):
                    raise OSError("asset index publication failed")
                with self._lock:
                    self._current_index = index
            with self._lock:
                self._publish_entry(path, relative, stat.st_size, digest, record, stat.st_mtime)
                self._paths_by_digest[digest] = path
                self._resolutions_by_digest[digest] = AssetResolution(path, record)
                entry = self._catalog.get(f"{self._namespace}/{relative}")
                assert entry is not None
                return entry.ref(resolver=self)

    def _prune_sidecar(self, fresh_index: dict[str, dict[str, object]]) -> None:
        """Drop only publications fully covered by the successfully persisted index."""
        sidecar = self._root / WRITES_SIDECAR_NAME
        try:
            with _sidecar_lock(self._root):
                remaining = {
                    relative: row
                    for relative, row in load_write_records(self._root).items()
                    if any(
                        fresh_index.get(relative, {}).get(key) != value
                        for key, value in row.items()
                    )
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
        self.refresh_writes()
        with self._lock:
            entry = self._catalog.get(virtual_path)
        if entry is None:
            raise AssetError(f"no asset cataloged at virtual path: {virtual_path!r}")
        return entry.ref(resolver=self)

    def refresh_writes(self) -> None:
        """Publish valid writer-sidecar rows into the live catalog."""
        for relative, row in load_write_records(self._root).items():
            path = self._root / relative
            if self._ignore.matches(relative, path.name):
                continue
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
            if record is not None and not record.matches(path):
                continue
            self._publish_entry(path, relative, size, digest, record, stat.st_mtime)

    def entries(self) -> tuple[AssetEntry, ...]:
        self.refresh_writes()
        with self._lock:
            return self._catalog.entries()

    def missing_entries(self) -> tuple[AssetEntry, ...]:
        """Retained identities with no live location, excluded from resolution."""
        entries: list[AssetEntry] = []
        with self._lock:
            for relative, row in self._current_index.items():
                previous = row.get("missingVersions", [])
                versions = (
                    list(cast("list[object]", previous)) if isinstance(previous, list) else []
                )
                if row.get("missing") is True:
                    versions.append(row)
                for version in versions:
                    if not isinstance(version, Mapping):
                        continue
                    old = cast("Mapping[str, object]", version)
                    digest, size = old.get("digest"), old.get("size")
                    if (
                        not isinstance(digest, str)
                        or not is_digest(digest)
                        or type(size) is not int
                    ):
                        continue
                    entries.append(
                        AssetEntry(
                            virtual_path=f"{self._namespace}/{relative}",
                            digest=digest,
                            size=size,
                            metadata={"missing": True},
                        )
                    )
        return tuple(entries)

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
        return {
            str(key): val
            for key, val in cast("dict[object, object]", loaded).items()
            if not self._ignore.matches(str(key), Path(str(key)).name)
        }

    def _save_index(
        self, index: dict[str, dict[str, object]], *, only_if_missing: bool = False
    ) -> bool:
        """Publish complete JSON atomically, without rewriting unchanged indexes.

        Initial writer publication uses a hard link so it cannot replace an
        index that the helper published after this library loaded its rows.
        """
        with self._index_write_lock:
            serialized = json.dumps(index, indent=1)
            try:
                if self._index_path.read_text("utf-8") == serialized:
                    return True
            except (OSError, ValueError):
                pass
            tmp = self._index_path.with_name(
                self._index_path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
            )
            try:
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(serialized, "utf-8")
                if only_if_missing:
                    try:
                        os.link(tmp, self._index_path)
                    except FileExistsError:
                        pass
                    tmp.unlink()
                else:
                    os.replace(tmp, self._index_path)
                return True
            except OSError:
                # A read-only root degrades to rehash-on-scan, not failure.
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return False


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
        self._entries_by_relative: dict[str, AssetEntry] = {}
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
            if entry.get("missing") is True:
                continue
            digest, size, mtime = entry.get("digest"), entry.get("size"), entry.get("mtimeNs")
            if isinstance(digest, str) and isinstance(size, int) and isinstance(mtime, int):
                record = AssetVerificationRecord.from_json(digest, entry.get("verification"))
                self._rows_by_digest.setdefault(digest, []).append(
                    (str(relative), size, mtime, record)
                )
                if record is not None:
                    self._entries_by_relative[str(relative)] = AssetEntry(
                        virtual_path=str(relative),
                        digest=digest,
                        size=size,
                        media_type=_guess_media_type(Path(str(relative))),
                        modified_at=mtime / 1_000_000_000,
                    )

    def ref(self, relative: str) -> AssetRef:
        """Look up a selected file in the engine-written index without hashing it."""
        validate_virtual_path(relative)
        entry = self._entries_by_relative.get(relative)
        if entry is None:
            raise AssetError(f"no verified asset indexed at relative path: {relative!r}")
        path = self._root / relative
        try:
            if not path.resolve(strict=True).is_relative_to(self._root.resolve()):
                raise AssetError("indexed file is outside its root")
            records = self._rows_by_digest[entry.digest]
            if not any(
                name == relative and record is not None and record.matches(path)
                for name, _size, _mtime, record in records
            ):
                raise AssetError("indexed file changed; scan the mount before selection")
        except OSError as error:
            raise AssetError("indexed file is unavailable") from error
        return entry.ref(resolver=self)

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
                path.stat()
            except OSError:
                continue
            resolution = AssetResolution(path, record)
            if record is not None and record.matches(path):
                return resolution
            if stale is None:
                stale = resolution
        return stale or _resolve_asset_from_sidecar(self._root, digest)
