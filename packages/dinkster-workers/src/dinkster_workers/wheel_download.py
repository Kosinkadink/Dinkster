"""Bounded transport retries for wheel bytes, before lock verification."""

from __future__ import annotations

import http.client
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from dinkster_values import MEBIBYTE

from .provision import ProvisionError

_ATTEMPTS = 3
_TIMEOUT = 30


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
