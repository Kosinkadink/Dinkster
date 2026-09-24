"""Durable evidence for verified acquisition from public HTTPS sources."""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import contextlib
import errno
import importlib
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast
from urllib.parse import urlsplit

from .fetch import FetchResult
from .identity import AssetError, require_digest

PUBLIC_ACQUISITION_RECEIPT_VERSION = 1
PUBLIC_ACQUISITION_MAX_RECEIPTS = 100_000
PUBLIC_ACQUISITION_MAX_BYTES = 64 * MEBIBYTE
PUBLIC_ACQUISITION_MAX_URL = 4096

PublicSourceType = Literal[
    "official-provider",
    "declarative-resolver",
    "code-resolver",
    "manual",
]
_SOURCE_TYPES = frozenset({"official-provider", "declarative-resolver", "code-resolver", "manual"})
_RECEIPT_PATH_LOCKS_GUARD = threading.Lock()
_RECEIPT_PATH_LOCKS: dict[str, threading.RLock] = {}


class _Fcntl(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, fd: int, operation: int, /) -> None: ...


class _Msvcrt(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, fd: int, mode: int, nbytes: int, /) -> None: ...


def _receipt_thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _RECEIPT_PATH_LOCKS_GUARD:
        return _RECEIPT_PATH_LOCKS.setdefault(key, threading.RLock())


@contextlib.contextmanager
def _receipt_file_lock(path: Path) -> Generator[None]:
    """Serialize receipt read-modify-write across store instances and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with _receipt_thread_lock(path), lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
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


def _opaque(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetError(f"{field} must be a non-empty trimmed string")
    if len(value) > 512 or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AssetError(f"{field} must be bounded and contain no control characters")
    return value


def _timestamp(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssetError(f"{field} must be a finite non-negative timestamp")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise AssetError(f"{field} must be a finite non-negative timestamp") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise AssetError(f"{field} must be a finite non-negative timestamp")
    return parsed


def _size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssetError("sizeBytes must be a non-negative integer")
    return value


def _public_url(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetError(f"{field} must be a public HTTPS URL")
    if len(value) > PUBLIC_ACQUISITION_MAX_URL or any(
        character.isspace() or ord(character) < 32 for character in value
    ):
        raise AssetError(f"{field} must contain no whitespace or control characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise AssetError(f"{field} is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise AssetError(f"{field} must use HTTPS without userinfo, query, or fragment")
    return value


@dataclass(frozen=True)
class PublicAcquisitionSourceV1:
    """Trusted listing identity used to attempt receipt-bearing acquisition."""

    digest: str
    size_bytes: int
    source_type: PublicSourceType
    source_id: str
    source_revision: str
    listed_urls: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", require_digest(self.digest))
        object.__setattr__(self, "size_bytes", _size(self.size_bytes))
        if self.source_type not in _SOURCE_TYPES:
            raise AssetError(f"unknown public source type: {self.source_type!r}")
        object.__setattr__(self, "source_id", _opaque(self.source_id, "sourceId"))
        object.__setattr__(
            self,
            "source_revision",
            _opaque(self.source_revision, "sourceRevision"),
        )
        urls = tuple(_public_url(url, "listedUrl") for url in self.listed_urls)
        if not urls or len(urls) != len(set(urls)):
            raise AssetError("listed URLs must be a non-empty unique sequence")
        object.__setattr__(self, "listed_urls", urls)


@dataclass(frozen=True)
class PublicAcquisitionReceiptV1:
    version: int
    receipt_id: str
    digest: str
    size_bytes: int
    source_type: PublicSourceType
    source_id: str
    source_revision: str
    listed_url: str
    final_url: str
    fetched_at: float

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != PUBLIC_ACQUISITION_RECEIPT_VERSION:
            raise AssetError(f"unsupported public acquisition receipt version {self.version}")
        receipt_id = _opaque(self.receipt_id, "receiptId")
        if len(receipt_id) != 32 or any(
            character not in "0123456789abcdef" for character in receipt_id
        ):
            raise AssetError("receiptId must be 32 lowercase hexadecimal characters")
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "digest", require_digest(self.digest))
        object.__setattr__(self, "size_bytes", _size(self.size_bytes))
        if self.source_type not in _SOURCE_TYPES:
            raise AssetError(f"unknown public source type: {self.source_type!r}")
        object.__setattr__(self, "source_id", _opaque(self.source_id, "sourceId"))
        object.__setattr__(
            self,
            "source_revision",
            _opaque(self.source_revision, "sourceRevision"),
        )
        object.__setattr__(self, "listed_url", _public_url(self.listed_url, "listedUrl"))
        object.__setattr__(self, "final_url", _public_url(self.final_url, "finalUrl"))
        object.__setattr__(self, "fetched_at", _timestamp(self.fetched_at, "fetchedAt"))

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version,
            "receiptId": self.receipt_id,
            "digest": self.digest,
            "sizeBytes": self.size_bytes,
            "sourceType": self.source_type,
            "sourceId": self.source_id,
            "sourceRevision": self.source_revision,
            "listedUrl": self.listed_url,
            "finalUrl": self.final_url,
            "fetchedAt": self.fetched_at,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> PublicAcquisitionReceiptV1:
        expected = {
            "version",
            "receiptId",
            "digest",
            "sizeBytes",
            "sourceType",
            "sourceId",
            "sourceRevision",
            "listedUrl",
            "finalUrl",
            "fetchedAt",
        }
        if set(wire) != expected:
            raise AssetError("public acquisition receipt fields do not match version 1")
        return cls(
            version=cast("int", wire["version"]),
            receipt_id=cast("str", wire["receiptId"]),
            digest=cast("str", wire["digest"]),
            size_bytes=cast("int", wire["sizeBytes"]),
            source_type=cast("PublicSourceType", wire["sourceType"]),
            source_id=cast("str", wire["sourceId"]),
            source_revision=cast("str", wire["sourceRevision"]),
            listed_url=cast("str", wire["listedUrl"]),
            final_url=cast("str", wire["finalUrl"]),
            fetched_at=cast("float", wire["fetchedAt"]),
        )


class PublicAcquisitionReceiptStore:
    """Append-only receipt records persisted atomically in one JSON document."""

    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._path = Path(path)
        self._clock = clock
        self._id_factory = id_factory
        self._lock = threading.RLock()
        self._receipts = self._load()

    def _load(self) -> dict[str, PublicAcquisitionReceiptV1]:
        try:
            with self._path.open("rb") as handle:
                if os.fstat(handle.fileno()).st_size > PUBLIC_ACQUISITION_MAX_BYTES:
                    raise AssetError("public acquisition receipt store exceeds its byte limit")
                data = handle.read(PUBLIC_ACQUISITION_MAX_BYTES + 1)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise AssetError(f"cannot load public acquisition receipts: {exc}") from exc
        if len(data) > PUBLIC_ACQUISITION_MAX_BYTES:
            raise AssetError("public acquisition receipt store exceeds its byte limit")

        def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise AssetError(f"duplicate public acquisition receipt field {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise AssetError(f"invalid public acquisition receipt number {value}")

        try:
            loaded: object = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=object_pairs,
                parse_constant=reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise AssetError(f"cannot load public acquisition receipts: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise AssetError("public acquisition receipt store must be an object")
        document = cast("Mapping[str, object]", loaded)
        if set(document) != {"publicAcquisitionReceipts", "receipts"}:
            raise AssetError("public acquisition receipt store has unknown fields")
        store_version = document["publicAcquisitionReceipts"]
        if type(store_version) is not int or store_version != PUBLIC_ACQUISITION_RECEIPT_VERSION:
            raise AssetError("unsupported public acquisition receipt store version")
        rows_raw = document["receipts"]
        if not isinstance(rows_raw, Sequence) or isinstance(rows_raw, (str, bytes)):
            raise AssetError("public acquisition receipt store requires a receipt list")
        rows = cast("Sequence[object]", rows_raw)
        if len(rows) > PUBLIC_ACQUISITION_MAX_RECEIPTS:
            raise AssetError("public acquisition receipt store exceeds its receipt limit")
        receipts: dict[str, PublicAcquisitionReceiptV1] = {}
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise AssetError(f"receipts[{position}] must be an object")
            receipt = PublicAcquisitionReceiptV1.from_wire(cast("Mapping[str, object]", row))
            if receipt.receipt_id in receipts:
                raise AssetError("public acquisition receipt ids must be unique")
            receipts[receipt.receipt_id] = receipt
        return receipts

    def record(
        self,
        source: PublicAcquisitionSourceV1,
        result: FetchResult,
    ) -> PublicAcquisitionReceiptV1:
        """Persist a receipt only for a complete verified public fetch result."""
        if not result.public_https_verified:
            raise AssetError("fetch result does not prove a public HTTPS chain")
        if result.digest != source.digest:
            raise AssetError("fetch result digest does not match the trusted declaration")
        if result.listed_url not in source.listed_urls:
            raise AssetError("fetch result does not match the trusted listed URL")
        if result.size_bytes != source.size_bytes:
            raise AssetError("fetch result size does not match the trusted declaration")
        receipt = PublicAcquisitionReceiptV1(
            version=PUBLIC_ACQUISITION_RECEIPT_VERSION,
            receipt_id=self._id_factory(),
            digest=source.digest,
            size_bytes=source.size_bytes,
            source_type=source.source_type,
            source_id=source.source_id,
            source_revision=source.source_revision,
            listed_url=result.listed_url,
            final_url=result.final_url,
            fetched_at=_timestamp(self._clock(), "receipt clock"),
        )
        with self._lock, _receipt_file_lock(self._path):
            latest = self._load()
            previous = self._receipts
            self._receipts = latest
            if len(self._receipts) >= PUBLIC_ACQUISITION_MAX_RECEIPTS:
                raise AssetError("public acquisition receipt store is full")
            if receipt.receipt_id in self._receipts:
                raise AssetError("public acquisition receipt id already exists")
            self._receipts[receipt.receipt_id] = receipt
            try:
                self._save()
            except BaseException:
                self._receipts = previous
                raise
        return receipt

    def records(self, digest: str | None = None) -> tuple[PublicAcquisitionReceiptV1, ...]:
        if digest is not None:
            digest = require_digest(digest)
        with self._lock:
            return tuple(
                sorted(
                    (
                        receipt
                        for receipt in self._receipts.values()
                        if digest is None or receipt.digest == digest
                    ),
                    key=lambda receipt: (receipt.fetched_at, receipt.receipt_id),
                )
            )

    def matching(self, source: PublicAcquisitionSourceV1) -> tuple[PublicAcquisitionReceiptV1, ...]:
        return tuple(
            receipt
            for receipt in self.records(source.digest)
            if receipt.size_bytes == source.size_bytes
            and receipt.source_type == source.source_type
            and receipt.source_id == source.source_id
            and receipt.source_revision == source.source_revision
            and receipt.listed_url in source.listed_urls
        )

    def _save(self) -> None:
        serialized = json.dumps(
            {
                "publicAcquisitionReceipts": PUBLIC_ACQUISITION_RECEIPT_VERSION,
                "receipts": [
                    receipt.to_wire()
                    for receipt in sorted(
                        self._receipts.values(),
                        key=lambda receipt: (receipt.fetched_at, receipt.receipt_id),
                    )
                ],
            },
            indent=1,
            allow_nan=False,
        )
        if len(serialized.encode("utf-8")) > PUBLIC_ACQUISITION_MAX_BYTES:
            raise AssetError("public acquisition receipt store exceeds its byte limit")
        temporary = self._path.with_name(self._path.name + f".tmp-{uuid.uuid4().hex}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.write_text(serialized, "utf-8")
            os.replace(temporary, self._path)
        except OSError:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise
