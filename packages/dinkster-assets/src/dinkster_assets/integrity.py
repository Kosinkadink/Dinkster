"""Descriptor-bound verification for content-addressed asset reads.

Ingest records bind a digest to stable descriptor metadata. Loaders can trust
that record without rereading the payload when the opened descriptor still
matches it. Missing records retain full descriptor-bound hashing, and stale
records fail closed.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

from .identity import CHUNK_SIZE, DIGEST_PREFIX, AssetError, new_hasher

IntegrityFailure = Literal[
    "digest_mismatch",
    "changed_during_verification",
    "path_rebound",
    "stale_ingest_record",
]


class AssetIntegrityError(AssetError):
    """The bytes at a resolved path do not (or can no longer be proven to)
    match the digest that named them. Carries the expected digest, the
    actual digest when a hash completed (None when the file changed under
    verification or the path stopped naming the verified inode), and the
    offending path - for logs; public surfaces should not echo the path."""

    def __init__(
        self,
        expected_digest: str,
        actual_digest: str | None,
        path: Path,
        reason: IntegrityFailure,
    ) -> None:
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest
        self.path = path
        self.reason = reason
        detail = {
            "digest_mismatch": f"content hashes to {actual_digest}",
            "changed_during_verification": "the file changed while being hashed",
            "path_rebound": "the path stopped naming the verified file",
            "stale_ingest_record": "the file metadata differs from its ingest record",
        }[reason]
        super().__init__(
            f"asset integrity failure ({reason}): {path} does not hold the "
            f"bytes named by {expected_digest} - {detail}"
        )


_Fingerprint = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class AssetVerificationRecord:
    """Descriptor metadata captured by a content-verified ingest."""

    digest: str
    scheme: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int | None

    def matches(self, stat: os.stat_result) -> bool:
        if self.scheme != _stat_scheme():
            return False
        actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        expected = (self.device, self.inode, self.size, self.mtime_ns)
        return actual == expected and (self.ctime_ns is None or stat.st_ctime_ns == self.ctime_ns)

    def to_json(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "scheme": self.scheme,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtimeNs": self.mtime_ns,
            **({"ctimeNs": self.ctime_ns} if self.ctime_ns is not None else {}),
        }

    @classmethod
    def from_json(cls, digest: str, value: object) -> AssetVerificationRecord | None:
        if not isinstance(value, Mapping):
            return None
        row = cast("Mapping[str, object]", value)
        recorded_digest = row.get("digest")
        scheme = row.get("scheme")
        device = row.get("device")
        inode = row.get("inode")
        size = row.get("size")
        mtime_ns = row.get("mtimeNs")
        ctime_ns = row.get("ctimeNs")
        if (
            recorded_digest != digest
            or scheme not in ("posix-v1", "windows-v1")
            or not all(isinstance(field, int) for field in (device, inode, size, mtime_ns))
            or (ctime_ns is not None and not isinstance(ctime_ns, int))
        ):
            return None
        if scheme == "posix-v1" and not isinstance(ctime_ns, int):
            return None
        return cls(
            digest,
            cast("str", scheme),
            cast("int", device),
            cast("int", inode),
            cast("int", size),
            cast("int", mtime_ns),
            ctime_ns,
        )


def _stat_scheme() -> str | None:
    if os.name == "posix":
        return "posix-v1"
    if os.name == "nt":
        return "windows-v1"
    return None


def verification_record(digest: str, stat: os.stat_result) -> AssetVerificationRecord | None:
    """Create a load-time proof only where device/inode identity is usable."""
    scheme = _stat_scheme()
    if scheme is None or stat.st_dev == 0 or stat.st_ino == 0:
        return None
    return AssetVerificationRecord(
        digest,
        scheme,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns if scheme == "posix-v1" else None,
    )


def verification_record_for_publication(
    digest: str,
    verified: os.stat_result,
    published: os.stat_result,
) -> AssetVerificationRecord | None:
    """Bind verified temporary bytes to their atomically published name."""
    verified_identity = (verified.st_dev, verified.st_ino)
    published_identity = (published.st_dev, published.st_ino)
    identities_stable = all(verified_identity) and all(published_identity)
    if identities_stable and verified_identity != published_identity:
        raise AssetError("asset path was rebound while it was being published")
    if (verified.st_size, verified.st_mtime_ns) != (published.st_size, published.st_mtime_ns):
        raise AssetError("asset changed while it was being published")
    if not identities_stable:
        return None
    return verification_record(digest, published)


_CACHE_LIMIT = 4096
"""Verified fingerprints retained per process. Bounds memory and shrinks
the window in which a recycled inode number could meet a matching stale
entry; real libraries hold thousands of assets, not millions."""

_TRUST_FINGERPRINTS = os.name == "posix"
"""Whether a matching fingerprint may skip the rehash. Requires st_ctime_ns
to be a change stamp (POSIX); on Windows it is creation time, which an
in-place rewrite with a restored mtime would not move."""

_verified: OrderedDict[_Fingerprint, str] = OrderedDict()
_verified_lock = threading.Lock()


def _reset_after_fork() -> None:
    # A forked child inherits the dict but not the right to trust it being
    # in sync with whatever the parent does next - and possibly a lock a
    # now-nonexistent parent thread held. Fresh lock, empty cache.
    global _verified_lock
    _verified_lock = threading.Lock()
    _verified.clear()


register_at_fork = getattr(os, "register_at_fork", None)
if register_at_fork is not None:  # pragma: no branch - POSIX
    register_at_fork(after_in_child=_reset_after_fork)


def clear_verified_cache() -> None:
    """Forget every recorded verification (tests; long-lived hosts that
    want to force a re-hash)."""
    with _verified_lock:
        _verified.clear()


def _fingerprint(stat: os.stat_result) -> _Fingerprint:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _path_binding_fingerprint(fingerprint: _Fingerprint) -> tuple[int, ...]:
    # Windows creation time can differ between stat() and fstat() for the same file.
    return fingerprint[:4] if os.name == "nt" else fingerprint


def _hash_handle(handle: BinaryIO) -> str:
    """Hash the handle's content from position 0. A seam on purpose:
    integrity tests monkeypatch this to count hashes and to interleave
    replacements mid-verification."""
    handle.seek(0)
    hasher = new_hasher()
    while chunk := handle.read(CHUNK_SIZE):
        hasher.update(chunk)
    return DIGEST_PREFIX + hasher.hexdigest()


def digest_file_with_record(
    path: Path,
) -> tuple[str, AssetVerificationRecord | None]:
    """Hash one ingest candidate and bind its digest to the same descriptor."""
    with path.open("rb") as handle:
        before = _fingerprint(os.fstat(handle.fileno()))
        digest = _hash_handle(handle)
        after_stat = os.fstat(handle.fileno())
        if _fingerprint(after_stat) != before:
            raise AssetError(f"asset changed while being ingested: {path}")
        return digest, verification_record(digest, after_stat)


def open_verified(
    path: Path,
    expected_digest: str,
    record: AssetVerificationRecord | None = None,
) -> BinaryIO:
    """Open ``path`` and return a handle positioned at 0 whose bytes are
    proven to hash to ``expected_digest``.

    The proof and the returned handle share one file descriptor, so a
    rename/replace after (or during) verification cannot swap the bytes:
    the descriptor keeps the verified inode. Raises AssetIntegrityError
    on any mismatch; raises AssetError when hashing is unavailable in
    this interpreter (missing blake3) - never a silent downgrade."""
    handle = path.open("rb")
    try:
        opened_stat = os.fstat(handle.fileno())
        before = _fingerprint(opened_stat)
        if record is not None:
            if record.digest != expected_digest or not record.matches(opened_stat):
                raise AssetIntegrityError(expected_digest, None, path, "stale_ingest_record")
            return handle
        actual: str | None = None
        if _TRUST_FINGERPRINTS:
            with _verified_lock:
                actual = _verified.get(before)
                if actual is not None:
                    _verified.move_to_end(before)
        if actual is None:
            actual = _hash_handle(handle)
            after = _fingerprint(os.fstat(handle.fileno()))
            if after != before:
                raise AssetIntegrityError(
                    expected_digest, None, path, "changed_during_verification"
                )
            if _TRUST_FINGERPRINTS:
                # Cache what the content IS (its proven hash), not what the
                # caller hoped for: a later open with the honest digest may
                # hit; one with a different digest still compares and fails.
                with _verified_lock:
                    _verified[after] = actual
                    _verified.move_to_end(after)
                    while len(_verified) > _CACHE_LIMIT:
                        _verified.popitem(last=False)
        if actual != expected_digest:
            raise AssetIntegrityError(expected_digest, actual, path, "digest_mismatch")
        handle.seek(0)
        return handle
    except BaseException:
        handle.close()
        raise


def verified_local_path(
    path: Path,
    expected_digest: str,
    record: AssetVerificationRecord | None = None,
) -> Path:
    """Verify ``path``'s content against ``expected_digest`` and return the
    path, for consumers that need a real filesystem name (checkpoint
    loaders). Weaker than :func:`open_verified` by nature: after the final
    check the name can be rebound, so the guarantee is "the path named the
    verified bytes at return time", with the window shrunk from unbounded
    to the instant of return. Prefer open_verified where a handle works."""
    verified = (
        open_verified(path, expected_digest)
        if record is None
        else open_verified(path, expected_digest, record)
    )
    with verified as handle:
        fd_print = _fingerprint(os.fstat(handle.fileno()))
        try:
            path_print = _fingerprint(os.stat(path))
        except OSError:
            raise AssetIntegrityError(expected_digest, None, path, "path_rebound") from None
        if _path_binding_fingerprint(path_print) != _path_binding_fingerprint(fd_print):
            raise AssetIntegrityError(expected_digest, None, path, "path_rebound")
    return path
