"""DiskCacheStore: cache entries that survive the process (DESIGN 3.4).

Layout under one root:

- ``cas/`` - payload bytes, content-addressed (DiskCAS). Blobs are shared:
  two entries whose outputs carry identical bytes store them once.
- ``entries/<hash-of-key>.json`` - one manifest per cache key (wire.py's
  format, verbatim - the same shape the peer endpoint serves).

Because fingerprints travel in the envelope and cache keys derive from
them, a manifest written by one process is a hit in any other - restart
persistence and cross-machine sharing are the same mechanism. A lock file
serializes manifest publication, trimming, and garbage collection across
store instances and processes sharing the root. Queued-write cancellation
is store-local; overlapping writes from another store serialize in lock
order and can publish after an invalidation.

What never lands here: entries carrying resource stubs (references to live
process state - dangling after restart) and values with no bytes to store
(unregistered type, payload never encoded). Refusal is silent and total
per entry; the memory layer above still serves those within the run.

Budgeting is self-contained: after each put, evict least-recently-used
manifests (mtime is the LRU clock, refreshed on hit) until under
``max_bytes``, then drop CAS blobs no remaining manifest references.
Entries and blobs are separate concepts: eviction of one key never deletes
bytes another key still needs. Governor-driven shedding of *disk* is
deliberately not wired yet - disk is not a governed device today.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

from dinkster_values import GIBIBYTE

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from dinkster_protocol import CacheKey
from dinkster_values import TypeRegistry, Value, stable_hash

from .cas import DiskCAS
from .wire import (
    EncodedValue,
    encode_entry,
    entry_from_wire,
    entry_to_wire,
    iter_manifest_payloads,
)

DEFAULT_DISK_CACHE_BYTES = 10 * GIBIBYTE
DEFAULT_MAX_BYTES = DEFAULT_DISK_CACHE_BYTES

_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_T = TypeVar("_T")


def _root_thread_lock(root: Path) -> threading.RLock:
    key = str(root.resolve())
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_file_lock(path: Path) -> Generator[None, None, None]:
    """Cross-platform advisory lock for one cache root."""
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if (
                        exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)
                        and getattr(exc, "winerror", None) != 33
                    ):
                        raise
                    time.sleep(0.05)
                else:
                    break
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DiskCacheStore:
    cache_layer = "disk"

    def __init__(
        self,
        root: Path,
        registry: TypeRegistry,
        *,
        max_bytes: int = DEFAULT_DISK_CACHE_BYTES,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self._registry = registry
        self._max_bytes = max_bytes
        self._cas = DiskCAS(root / "cas")
        self._entries_dir = root / "entries"
        self._entries_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = _root_thread_lock(root)
        self._lock_path = root / ".lock"
        self._pending_writes: dict[CacheKey, object] = {}
        self.hits = 0
        self.misses = 0
        self._with_lock(self._recover)

    def _entry_path(self, key: CacheKey) -> Path:
        return self._entries_dir / (stable_hash([key.encode("utf-8")]) + ".json")

    # -- CacheStore -------------------------------------------------------

    async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
        entry = await asyncio.to_thread(self._with_lock, self._get, key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        encoded = encode_entry(outputs, self._registry)
        if encoded is None:
            return  # structurally unpersistable (stubs / no bytes); not an error
        token = object()
        self._pending_writes[key] = token
        try:
            await asyncio.to_thread(self._with_lock, self._put_pending, key, encoded, token)
        finally:
            if self._pending_writes.get(key) is token:
                del self._pending_writes[key]

    def _put_pending(self, key: CacheKey, encoded: dict[str, EncodedValue], token: object) -> None:
        # This store's invalidation can precede a queued writer taking the lock.
        if self._pending_writes.get(key) is token:
            self._put(key, encoded)

    # -- invalidation (ReleaseGuard contract) -----------------------------

    def drop_referencing(self, resource_id: str) -> int:
        """Nothing referencing a live resource is ever persisted (put
        refuses stub entries), so there is never anything to drop."""
        return 0

    def clear(self) -> int:
        """Remove every manifest and its now-orphaned payload blobs."""
        self._pending_writes.clear()
        return self._with_lock(self._clear)

    def discard(self, key: CacheKey) -> None:
        """Invalidate a replaced key, including this store's queued publication."""
        self._pending_writes.pop(key, None)
        self._with_lock(self._discard, key)

    def _discard(self, key: CacheKey) -> None:
        path = self._entry_path(key)
        if path.exists():
            self._remove_entry(path)
            self._gc_blobs()

    # -- peer export (the server's cache-sharing endpoints call these) ----

    async def entry_wire(self, key: str) -> Mapping[str, Any] | None:
        """The raw manifest for a key - what GET /cache/entry/{key} serves."""
        return await asyncio.to_thread(self._with_lock, self._entry_wire, key)

    async def blob(self, digest: str) -> bytes | None:
        """Verified blob bytes - what GET /cache/cas/{digest} serves."""
        return await asyncio.to_thread(self._cas.get, digest)

    # -- blocking internals (called via to_thread) -------------------------

    def _with_lock(self, operation: Callable[..., _T], *args: Any) -> _T:
        with self._thread_lock, _exclusive_file_lock(self._lock_path):
            return operation(*args)

    def _get(self, key: CacheKey) -> Mapping[str, Value] | None:
        path = self._entry_path(key)
        wire = self._read_manifest(path)
        if wire is None or wire.get("key") != key:
            return None

        async def fetch_blob(digest: str, size: int) -> bytes | None:
            del size
            try:
                return self._cas.get(digest)
            except ValueError:
                return None

        entry = asyncio.run(entry_from_wire(wire, fetch_blob, self._registry))
        if entry is None:
            # The lock prevents a same-key replacement from landing between
            # this read and removal of its unservable manifest.
            self._remove_entry(path)
            self._gc_blobs()
            return None
        with contextlib.suppress(OSError):
            os.utime(path)  # LRU clock
        return entry

    def _put(self, key: CacheKey, encoded: dict[str, EncodedValue]) -> None:
        try:
            self._write_entry(key, encoded)
        except BaseException:
            self._gc_blobs()
            raise
        self._trim()

    def _entry_wire(self, key: str) -> Mapping[str, Any] | None:
        path = self._entry_path(key)
        wire = self._read_manifest(path)
        if wire is None or wire.get("key") != key:
            return None
        with contextlib.suppress(OSError):
            os.utime(path)
        return wire

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any] | None:
        try:
            raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        return cast("dict[str, Any]", raw)

    @staticmethod
    def _remove_entry(path: Path) -> None:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

    def _write_entry(self, key: CacheKey, encoded: dict[str, EncodedValue]) -> None:
        manifest = entry_to_wire(key, encoded, self._cas.put)
        path = self._entry_path(key)
        tmp = self._entries_dir / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            tmp.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()

    def _manifests(self) -> list[Path]:
        return [p for p in self._entries_dir.iterdir() if p.suffix == ".json"]

    def _trim(self) -> None:
        """Evict LRU manifests until under budget, then GC orphaned blobs."""
        manifests = self._manifests()
        total = self._disk_usage(manifests)
        if total <= self._max_bytes:
            return
        # Replacing an existing key can leave old payloads unreferenced. Reclaim
        # those before evicting a live entry to satisfy the budget.
        self._gc_blobs()
        manifests = self._manifests()
        total = self._disk_usage(manifests)
        for path in sorted(manifests, key=lambda p: p.stat().st_mtime):
            if total <= self._max_bytes:
                break
            self._remove_entry(path)
            self._gc_blobs()
            # A manifest's declared payload sizes are not eviction credit:
            # another manifest may still reference those blobs. GC first,
            # then measure what is actually left on disk.
            total = self._disk_usage(self._manifests())

    def _recover(self) -> None:
        """Remove interrupted publications and enforce the current budget."""
        for path in self._entries_dir.iterdir():
            if path.suffix == ".tmp":
                self._remove_entry(path)
            elif path.suffix == ".json" and self._read_manifest(path) is None:
                self._remove_entry(path)
        self._gc_blobs()
        self._trim()

    def _clear(self) -> int:
        manifests = self._manifests()
        for path in manifests:
            self._remove_entry(path)
        self._gc_blobs()
        return len(manifests)

    def _disk_usage(self, manifests: list[Path]) -> int:
        return self._cas.total_bytes() + sum(
            path.stat().st_size for path in manifests if path.is_file()
        )

    def _gc_blobs(self) -> None:
        referenced: set[str] = set()
        for path in self._manifests():
            manifest = self._read_manifest(path)
            if manifest is None:
                continue
            for payload in iter_manifest_payloads(manifest):
                digest = payload.get("digest")
                if isinstance(digest, str):
                    referenced.add(digest)
        for digest in self._cas.digests():
            if digest not in referenced:
                self._cas.delete(digest)
