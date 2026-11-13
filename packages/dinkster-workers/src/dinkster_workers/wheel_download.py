"""Bounded transport retries for wheel bytes, before lock verification."""

from __future__ import annotations

import hashlib
import http.client
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from dinkster_store import object_path, readable_label
from dinkster_values import MEBIBYTE

if TYPE_CHECKING:
    from .pack_lock import LockedWheel

from .provision import ProvisionError

_ATTEMPTS = 3
_TIMEOUT = 30


def verify_wheel(path: Path, wheel: LockedWheel) -> bool:
    if not path.is_file() or path.stat().st_size != wheel.size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == wheel.sha256


def fetch_wheel(wheel: LockedWheel, store: Path, *, offline: bool = False) -> Path:
    """Admit bytes only after size and digest verification; prefer stored bytes."""
    target = (
        object_path(
            store,
            "sha256:" + wheel.sha256,
            label=None if offline else readable_label(Path(wheel.filename).stem),
        )
        / wheel.filename
    )
    for candidate in (target, store / wheel.filename):
        if verify_wheel(candidate, wheel):
            return candidate
    if offline:
        raise ProvisionError(f"offline pack wheel missing or corrupt: {wheel.filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    staging = Path(name)
    os.close(descriptor)
    try:
        download_wheel(wheel.url, staging, max_size=wheel.size)
        if not verify_wheel(staging, wheel):
            raise ProvisionError(f"wheel hash or size mismatch: {wheel.filename}")
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def download_wheel(url: str, path: Path, *, max_size: int | None = None) -> None:
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as source, path.open("wb") as output:
                received = 0
                while block := source.read(
                    MEBIBYTE if max_size is None else min(MEBIBYTE, max_size - received + 1)
                ):
                    received += len(block)
                    if max_size is not None and received > max_size:
                        raise ProvisionError(f"wheel exceeds locked size: {url}")
                    output.write(block)
                length = source.headers.get("Content-Length")
                if length is not None and received < int(length):
                    raise http.client.IncompleteRead(b"", int(length) - received)
            return
        except (
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            ssl.SSLError,
            http.client.HTTPException,
        ) as exc:
            if (
                url.startswith("file:")
                or isinstance(exc, urllib.error.HTTPError)
                and not 500 <= exc.code < 600
                or isinstance(exc, ssl.SSLCertVerificationError)
                or isinstance(exc, urllib.error.URLError)
                and isinstance(exc.reason, ssl.SSLCertVerificationError)
            ):
                raise
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            if attempt == _ATTEMPTS:
                raise ProvisionError(
                    f"wheel download failed after {attempt} tries: {url}: {exc}"
                ) from exc
            time.sleep(attempt)
