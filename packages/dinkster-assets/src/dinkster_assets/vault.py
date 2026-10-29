"""AssetVault: where fetched and transferred bytes land (DESIGN 3.12).

A content-addressed file store, sharded like the payload CAS and sharing
its canonical ``blake3:<64 hex>`` namespace: ``<root>/<2 hex>/<64 hex>``.
The vault is the materialization target for asset *distribution* - bytes
pulled from provenance URLs or peers - while model libraries stay what
they are: user-managed folders the vault never touches.

Derived conversion sidecars live beside a source as
``<64 hex>.<sha256-16>.dinkster.safetensors``. They are not vault assets:
announcements ignore them, and deleting the source deletes every sibling
whose name begins with the source's complete 64-hex name plus a dot.

Ingest streams: multi-gigabyte models never pass through memory whole.
A writer hashes chunks as they arrive and lands the file atomically
(dot-prefixed temp + ``os.replace``) only if the bytes hash to the digest
the caller expected - a failed or tampered download leaves nothing behind
but the temp file it cleans up. Reads trust the name: content was verified
on the way in, and re-hashing many-GB files on every resolve would make
materialization unusable (the deliberate asymmetry with the payload CAS,
whose small blobs are cheap to verify per read).
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

from . import p2p_storage
from .identity import AssetError, new_hasher, require_digest
from .integrity import (
    AssetVerificationRecord,
    digest_file_with_record,
    verification_record_for_publication,
)
from .model import AssetResolution
from .p2p_descriptor import P2PDescriptorV1, validate_p2p_descriptor

_HEX = frozenset("0123456789abcdef")


class VaultError(AssetError):
    """A vault operation could not preserve content-addressed integrity."""


class AssetVault:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, digest: str) -> Path:
        hexpart = require_digest(digest).split(":", 1)[1]
        return self._root / hexpart[:2] / hexpart

    def has(self, digest: str) -> bool:
        return self._path(digest).is_file()

    def resolve(self, digest: str) -> Path | None:
        """AssetResolver: digest -> real path, if the vault holds it."""
        path = self._path(digest)
        return path if path.is_file() else None

    def resolve_asset(self, digest: str) -> AssetResolution | None:
        path = self.resolve(digest)
        if path is None:
            return None
        record: AssetVerificationRecord | None = None
        try:
            loaded: object = json.loads(_record_path(path).read_text("utf-8"))
        except (OSError, ValueError):
            pass
        else:
            record = AssetVerificationRecord.from_json(digest, loaded)
        return AssetResolution(path, record)

    def digests(self) -> list[str]:
        """Every digest currently held (announcements, GC scans)."""
        found: list[str] = []
        for shard in self._root.iterdir():
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for blob in shard.iterdir():
                if (
                    blob.is_file()
                    and len(blob.name) == 64
                    and all(character in _HEX for character in blob.name)
                ):
                    found.append(f"blake3:{blob.name}")
        return found

    def writer(self, expected_digest: str) -> VaultWriter:
        """Streaming, verifying ingest for one asset. Feed chunks with
        ``write()``; ``commit()`` lands the file only if the bytes hash to
        ``expected_digest``. Use as a context manager: an exception or a
        missing commit rolls back to nothing."""
        digest = require_digest(expected_digest)
        return VaultWriter(self._root, self._path(digest), digest)

    @property
    def p2p_staging_root(self) -> Path:
        return p2p_storage.p2p_staging_root(self._root)

    def p2p_staging_path(
        self,
        descriptor: P2PDescriptorV1 | Mapping[str, object],
        digest: str,
        expected_size: int,
    ) -> Path:
        info_hash = _validated_p2p_info_hash(descriptor, digest, expected_size)
        return p2p_storage.p2p_staging_path(self._root, info_hash, digest)

    def open_p2p_partial(
        self,
        descriptor: P2PDescriptorV1 | Mapping[str, object],
        digest: str,
        expected_size: int,
    ) -> p2p_storage.P2PPartial:
        info_hash = _validated_p2p_info_hash(descriptor, digest, expected_size)
        return p2p_storage.open_p2p_partial(
            self._root,
            info_hash,
            digest,
            expected_size,
        )

    def adopt_staged_asset(
        self,
        descriptor: P2PDescriptorV1 | Mapping[str, object],
        digest: str,
        expected_size: int,
        staged_path: Path | str,
        format_policy_version: int,
    ) -> Path:
        """Verify and atomically publish a complete confined P2P partial."""
        info_hash = _validated_p2p_info_hash(descriptor, digest, expected_size)
        target = self._path(digest)
        path, record = p2p_storage.adopt_staged_asset(
            self._root,
            target,
            info_hash,
            digest,
            expected_size,
            staged_path,
            format_policy_version,
        )
        _persist_verification_record(path, record)
        return path

    def p2p_staging_usage(self) -> p2p_storage.P2PStagingUsage:
        return p2p_storage.p2p_staging_usage(self._root)

    def p2p_partial_growth(
        self,
        descriptor: P2PDescriptorV1 | Mapping[str, object],
        digest: str,
        expected_size: int,
    ) -> int:
        info_hash = _validated_p2p_info_hash(descriptor, digest, expected_size)
        return p2p_storage.p2p_partial_growth(self._root, info_hash, digest, expected_size)

    def purge_inactive_p2p_partials(
        self,
        *,
        now: float | None = None,
        retention_seconds: int | None = None,
    ) -> p2p_storage.P2PStagingPurge:
        if retention_seconds is None:
            return p2p_storage.purge_inactive_p2p_partials(self._root, now=now)
        return p2p_storage.purge_inactive_p2p_partials(
            self._root,
            now=now,
            retention_seconds=retention_seconds,
        )

    def verify_p2p_local_file(
        self,
        digest: str,
        expected_size: int,
        local_path: Path | str,
        format_policy_version: int,
    ) -> p2p_storage.P2PLocalFileMapping:
        return p2p_storage.verify_p2p_local_file(
            self._root,
            digest,
            expected_size,
            local_path,
            format_policy_version,
        )

    def delete(self, digest: str) -> bool:
        path = self._path(digest)
        deleted = False
        try:
            path.unlink()
            deleted = True
        except FileNotFoundError:
            pass
        for sibling in path.parent.glob(f"{path.name}.*"):
            if sibling.is_file():
                sibling.unlink()
        return deleted


class VaultWriter:
    def __init__(self, root: Path, target: Path, expected_digest: str) -> None:
        self._target = target
        self._expected = expected_digest
        # Dot-prefixed so a library scan pointed at (or near) a vault root
        # never catalogs a partial download - the same rule that protects
        # the asset index's own temp files.
        self._tmp = root / f".ingest-{uuid.uuid4().hex}"
        self._handle = self._tmp.open("wb")
        self._hasher = new_hasher()
        self._committed = False

    def write(self, chunk: bytes) -> None:
        self._handle.write(chunk)
        self._hasher.update(chunk)

    def commit(self) -> Path:
        """Verify and land the file; raises AssetError (and rolls back) if
        the bytes do not hash to the expected digest."""
        self._handle.flush()
        verified_stat = os.fstat(self._handle.fileno())
        self._handle.close()
        actual = "blake3:" + self._hasher.hexdigest()
        if actual != self._expected:
            self._discard()
            raise AssetError(
                f"ingest did not verify: expected {self._expected}, bytes hash to {actual}"
            )
        self._target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(self._tmp, self._target)
        except OSError as exc:
            self._discard()
            # A racing ingest of the same content already landed identical
            # bytes: idempotent success (Windows refuses replacing a file
            # a reader holds open).
            record: AssetVerificationRecord | None = None
            try:
                target_digest, record = digest_file_with_record(self._target)
                target_matches = target_digest == self._expected
            except OSError:
                target_matches = False
            if target_matches:
                self._persist_verification(record)
                self._committed = True
                return self._target
            raise VaultError(f"could not store verified asset {self._expected}: {exc}") from exc
        published_stat = self._target.stat()
        record = verification_record_for_publication(self._expected, verified_stat, published_stat)
        self._persist_verification(record)
        self._committed = True
        return self._target

    def commit_with_result(self) -> tuple[Path, bool]:
        """Verify and atomically land bytes, reporting whether this writer won.

        Unlike a preflight ``has()`` check, the hard-link publication is one
        filesystem operation, so concurrent identical ingests have exactly one
        creator. The existing ``commit()`` contract remains unchanged for
        callers that do not need creation identity.
        """
        self._handle.flush()
        verified_stat = os.fstat(self._handle.fileno())
        self._handle.close()
        actual = "blake3:" + self._hasher.hexdigest()
        if actual != self._expected:
            self._discard()
            raise AssetError(
                f"ingest did not verify: expected {self._expected}, bytes hash to {actual}"
            )
        self._target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(self._tmp, self._target)
        except FileExistsError:
            record: AssetVerificationRecord | None = None
            try:
                try:
                    target_digest, record = digest_file_with_record(self._target)
                except AssetError:
                    # The winning writer can unlink its staging hard link during hashing,
                    # changing ctime. Require a fresh, stable verification before accepting.
                    target_digest, record = digest_file_with_record(self._target)
                target_matches = target_digest == self._expected
            except OSError:
                target_matches = False
            if not target_matches:
                self._discard()
                raise VaultError(
                    f"existing asset does not match its digest {self._expected}"
                ) from None
            self._discard()
            self._persist_verification(record)
            self._committed = True
            return self._target, False
        except OSError as exc:
            self._discard()
            raise VaultError(f"could not store verified asset {self._expected}: {exc}") from exc
        with contextlib.suppress(OSError):
            self._tmp.unlink()
        published_stat = self._target.stat()
        record = verification_record_for_publication(self._expected, verified_stat, published_stat)
        self._persist_verification(record)
        self._committed = True
        return self._target, True

    def _persist_verification(self, record: AssetVerificationRecord | None) -> None:
        _persist_verification_record(self._target, record)

    def _discard(self) -> None:
        with contextlib.suppress(OSError):
            self._handle.close()
        with contextlib.suppress(OSError):
            self._tmp.unlink()

    def __enter__(self) -> VaultWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._committed:
            self._discard()


def _validated_p2p_info_hash(
    descriptor: P2PDescriptorV1 | Mapping[str, object],
    digest: str,
    expected_size: int,
) -> str:
    return validate_p2p_descriptor(
        descriptor,
        asset_digest=digest,
        size=expected_size,
    ).info_hash


def _persist_verification_record(
    target: Path,
    record: AssetVerificationRecord | None,
) -> None:
    tmp: Path | None = None
    try:
        if record is None or not record.matches(target.stat()):
            return
        record_target = _record_path(target)
        tmp = record_target.with_name(record_target.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(record.to_json(), sort_keys=True), "utf-8")
        os.replace(tmp, record_target)
    except OSError:
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink()


def _record_path(path: Path) -> Path:
    return path.with_name(path.name + ".verified.json")
