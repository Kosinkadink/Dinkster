"""DiskCAS: content-addressed payload bytes on disk (DESIGN 3.4).

Blobs are keyed by the canonical ``blake3:<64 hex>`` digest - deliberately
the same namespace dinkster-assets uses for asset identity, so payload blobs
and asset blobs can eventually share one store. The store is dumb on
purpose: bytes in, bytes out, digest-verified both ways. What a blob
*means* (which cache entry, which asset) lives in whatever references it;
the CAS never knows and never needs to.

Concurrency and corruption posture:

- Writes are atomic: bytes land in a temp file first, then ``os.replace``
  onto the digest path (atomic on POSIX and Windows). Striped locks prevent
  threads from repeatedly replacing the same immutable blob; process races
  verify the winning bytes before returning.
- Reads verify: bytes that no longer hash to their name (bit rot, torn
  write from a crashed process) are deleted and reported as absent - a
  conservative miss, never a malformed payload.
- Paths derive only from validated digests, so a hostile "digest" cannot
  traverse outside the root.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from collections.abc import Collection
from pathlib import Path

from dinkster_assets import digest_bytes, digest_file, require_digest
from dinkster_values import GIBIBYTE

DEFAULT_VALUE_STORE_BYTES = 10 * GIBIBYTE


class CASError(Exception):
    """A blob could not be stored or addressed."""


class DiskCAS:
    """Synchronous by design: callers that live on an event loop wrap
    calls in ``asyncio.to_thread`` (DiskCacheStore does)."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._publication_locks = tuple(threading.Lock() for _ in range(64))

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, digest: str) -> Path:
        hexpart = require_digest(digest).split(":", 1)[1]
        return self._root / hexpart[:2] / hexpart

    def _publication_lock(self, digest: str) -> threading.Lock:
        return self._publication_locks[int(digest[-2:], 16) % len(self._publication_locks)]

    def has(self, digest: str) -> bool:
        return self._path(digest).is_file()

    def resolve(self, digest: str) -> Path | None:
        """Locate a blob for bounded reads; callers pin it and verify streamed bytes."""
        path = self._path(digest)
        return path if path.is_file() else None

    def put(self, data: bytes, *, protect: Collection[str] = ()) -> str:
        """Store bytes; returns their digest. Idempotent.

        ``protect`` names digests a budgeted store must not evict while
        storing these bytes; a plain DiskCAS never evicts, so it ignores it."""
        del protect
        digest = digest_bytes(data)
        path = self._path(digest)
        with self._publication_lock(digest):
            return self._put_locked(data, digest, path)

    def _put_locked(self, data: bytes, digest: str, path: Path) -> str:
        if path.is_file():
            try:
                if digest_file(path) == digest:
                    return digest
            except OSError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._root / f"tmp-{uuid.uuid4().hex}"
        try:
            tmp.write_bytes(data)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise CASError(f"could not store blob {digest}: {exc}") from exc
        self._replace_candidate(tmp, path, digest)
        return digest

    @staticmethod
    def _replace_candidate(candidate: Path, target: Path, digest: str) -> None:
        try:
            os.replace(candidate, target)
        except OSError as exc:
            with contextlib.suppress(OSError):
                candidate.unlink()
            # Windows can deny both the losing replace and immediate reads
            # while the winning writer settles the identical target.
            for delay in (0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064):
                time.sleep(delay)
                try:
                    if target.is_file():
                        if digest_file(target) == digest:
                            return
                        break
                except OSError:
                    continue
            raise CASError(f"could not store blob {digest}: {exc}") from exc

    def get(self, digest: str) -> bytes | None:
        """Bytes for a digest, verified; None if absent or corrupt (corrupt
        blobs are deleted - the next put restores them from good bytes)."""
        path = self._path(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        if digest_bytes(data) != digest:
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        return data

    def adopt_file(self, path: Path, digest: str, *, protect: Collection[str] = ()) -> None:
        """Adopt a file whose bytes must hash to ``digest``. The file is
        consumed: moved onto the digest path on success (atomic, same
        filesystem expected), deleted on any failure. This is the landing
        step for bytes that streamed to disk instead of memory - the hash
        check happens exactly once, here, before the blob becomes visible.

        ``protect`` names digests a budgeted store must not evict while
        landing this one; a plain DiskCAS never evicts, so it ignores it."""
        del protect
        canonical = require_digest(digest)
        try:
            actual = digest_file(path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                path.unlink()
            raise CASError(f"cannot read candidate blob for {canonical}: {exc}") from exc
        if actual != canonical:
            with contextlib.suppress(OSError):
                path.unlink()
            raise CASError(f"candidate blob does not hash to {canonical} (got {actual})")
        target = self._path(canonical)
        if path == target:
            return
        with self._publication_lock(canonical):
            target_matches = False
            if target.is_file():
                try:
                    target_matches = digest_file(target) == canonical
                except OSError:
                    pass
            if target_matches:
                try:
                    path.unlink()
                except OSError as exc:
                    raise CASError(f"could not consume candidate blob {canonical}: {exc}") from exc
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            self._replace_candidate(path, target, canonical)

    def pin(self, owner: object, digests: Collection[str]) -> None:
        """Keep ``digests`` safe from eviction until ``owner`` re-pins or
        unpins; a plain DiskCAS never evicts, so this is a no-op."""
        del owner, digests

    def unpin(self, owner: object) -> None:
        del owner

    def delete(self, digest: str) -> bool:
        try:
            self._path(digest).unlink()
            return True
        except FileNotFoundError:
            return False

    def digests(self) -> list[str]:
        """Every digest currently stored (for GC scans)."""
        found: list[str] = []
        for shard in self._root.iterdir():
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for blob in shard.iterdir():
                if blob.is_file():
                    found.append(f"blake3:{blob.name}")
        return found

    def total_bytes(self) -> int:
        total = 0
        for shard in self._root.iterdir():
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for blob in shard.iterdir():
                with contextlib.suppress(OSError):
                    total += blob.stat().st_size
        return total


class BudgetedDiskCAS(DiskCAS):
    """DiskCAS with a byte budget: once stored blobs exceed it, the least
    recently used are evicted. Recency is the blob file's mtime, refreshed
    on every hit, so eviction follows use rather than creation order.
    Eviction is a conservative miss for whoever asks next - the peer that
    still holds the bytes re-sends them - so an undersized budget degrades
    to re-transfers, never to an error."""

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_VALUE_STORE_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        super().__init__(root)
        self._max_bytes = max_bytes
        # Owner -> pinned blob paths. Written from event loops (pin/unpin),
        # read by trims on executor threads. The lock is held across each
        # victim's check-and-unlink, so once pin() returns, no trim - even
        # one that started earlier - can evict the pinned blobs.
        self._pins_lock = threading.Lock()
        self._pins: dict[object, frozenset[str]] = {}

    def _touch(self, digest: str) -> None:
        with contextlib.suppress(OSError):
            os.utime(self._path(digest))

    def has(self, digest: str) -> bool:
        hit = super().has(digest)
        if hit:
            self._touch(digest)
        return hit

    def get(self, digest: str) -> bytes | None:
        data = super().get(digest)
        if data is not None:
            self._touch(digest)
        return data

    def put(self, data: bytes, *, protect: Collection[str] = ()) -> str:
        digest = super().put(data)
        self._touch(digest)
        self._trim(protect={digest, *protect})
        return digest

    def adopt_file(self, path: Path, digest: str, *, protect: Collection[str] = ()) -> None:
        super().adopt_file(path, digest)
        self._trim(protect={require_digest(digest), *protect})

    def pin(self, owner: object, digests: Collection[str]) -> None:
        """Protect ``digests`` from eviction until ``owner`` re-pins or
        unpins. One pin set per owner: a new pin replaces the previous one.
        Trims consult the registry as they pick victims, so once pin()
        returns, the pinned blobs are safe from every trim, including ones
        already running."""
        pinned = frozenset(str(self._path(digest)) for digest in digests)
        with self._pins_lock:
            if pinned:
                self._pins[owner] = pinned
            else:
                self._pins.pop(owner, None)

    def unpin(self, owner: object) -> None:
        with self._pins_lock:
            self._pins.pop(owner, None)

    def _trim(self, protect: Collection[str]) -> None:
        # A protected or pinned blob is never an eviction victim. The
        # just-stored blob must land even when larger than the whole budget
        # (or the transfer that produced it could never complete), and a
        # pinned set is one frame's working set: evicting one blob to fit
        # another would make that frame permanently undecodable. Protected
        # and pinned sets larger than the budget temporarily exceed it.
        protected = {str(self._path(digest)) for digest in protect}
        entries: list[tuple[float, int, str]] = []
        total = 0
        for shard in self._root.iterdir():
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for blob in shard.iterdir():
                try:
                    stat = blob.stat()
                except OSError:
                    continue
                total += stat.st_size
                entries.append((stat.st_mtime, stat.st_size, str(blob)))
        if total <= self._max_bytes:
            return
        for _, size, blob_path in sorted(entries):
            if blob_path in protected:
                continue
            with self._pins_lock:
                if any(blob_path in pinned for pinned in self._pins.values()):
                    continue
                try:
                    os.unlink(blob_path)
                except OSError:
                    continue
            total -= size
            if total <= self._max_bytes:
                return
