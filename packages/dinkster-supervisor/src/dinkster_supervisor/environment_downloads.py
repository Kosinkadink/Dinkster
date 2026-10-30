"""Verified artifact downloads for environment installation."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from http.client import HTTPException, HTTPMessage
from pathlib import Path, PurePosixPath
from typing import IO, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .limits import ENVIRONMENT_METADATA_LIMIT_BYTES, ENVIRONMENT_STREAM_CHUNK_BYTES

__all__ = ["DownloadError", "fetch_artifact", "read_metadata"]

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")
_USER_AGENT = "dinkster-env"


class DownloadError(ValueError):
    """An artifact could not be obtained intact from any configured mirror."""


class _SafeRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        _validate_remote_url(newurl, allow_query=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_metadata(mirror: str, relative_path: str) -> bytes:
    """Read a bounded channel or manifest without creating a local project/cache."""
    _validate_path(relative_path)
    maximum = ENVIRONMENT_METADATA_LIMIT_BYTES
    parsed = urlparse(mirror)
    if parsed.scheme and not Path(mirror).is_absolute():
        _validate_remote_url(mirror)
        url = mirror.rstrip("/") + "/" + quote(relative_path, safe="/")
        request = Request(url, headers={"User-Agent": _USER_AGENT})
        with build_opener(_SafeRedirects()).open(request, timeout=30) as response:
            if response.status != 200:
                raise DownloadError("unexpected metadata response status")
            data = response.read(maximum + 1)
    else:
        root = Path(mirror).resolve(strict=True)
        source = (root / relative_path).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise DownloadError("metadata is missing or escapes its mirror")
        with source.open("rb") as handle:
            data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise DownloadError("metadata exceeds eight MiB")
    return data


def fetch_artifact(
    mirrors: tuple[str, ...],
    relative_path: str,
    destination: Path,
    sha256: str,
    size: int,
) -> Path:
    """Fetch one verified artifact, trying mirrors in the caller's order.

    ``destination.part`` keeps incomplete bytes so a later call can resume an
    interrupted HTTP transfer. The completed destination is only published
    after its exact size and SHA-256 have been verified.
    """
    _validate_request(mirrors, relative_path, sha256, size)
    destination = Path(destination)
    partial = destination.with_name(f"{destination.name}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        destination.unlink()
    if partial.is_symlink():
        partial.unlink()
    if destination.exists():
        if _matches(destination, sha256, size):
            return destination
        destination.unlink()

    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    if partial.exists() and partial.stat().st_size == size:
        if _matches(partial, sha256, size):
            partial.replace(destination)
            return destination
        partial.unlink()

    failures: list[str] = []
    for mirror in mirrors:
        try:
            resumed = partial.exists() and partial.stat().st_size > 0
            streamed_digest = _download_from_mirror(mirror, relative_path, partial, size)
            if streamed_digest != sha256 and partial.stat().st_size == size and resumed:
                # A saved prefix may be corrupt even when the resumed suffix is valid.
                partial.unlink()
                streamed_digest = _download_from_mirror(mirror, relative_path, partial, size)
            if partial.stat().st_size == size and streamed_digest == sha256:
                partial.replace(destination)
                return destination
            if partial.stat().st_size == size:
                partial.unlink()
            failures.append(f"{_mirror_label(mirror)} returned corrupt or short data")
        except (HTTPError, HTTPException, URLError, OSError, ValueError) as exc:
            failures.append(f"{_mirror_label(mirror)}: {_failure_reason(exc)}")
            if partial.exists() and partial.stat().st_size > size:
                partial.unlink()

    detail = "; ".join(failures) if failures else "no mirrors configured"
    raise DownloadError(f"artifact {relative_path!r} could not be downloaded: {detail}")


def _validate_request(mirrors: tuple[str, ...], relative_path: str, sha256: str, size: int) -> None:
    if not mirrors:
        raise DownloadError("artifact download requires at least one mirror")
    _validate_path(relative_path)
    if not _DIGEST_PATTERN.fullmatch(sha256):
        raise DownloadError("artifact SHA-256 must be 64 lowercase hexadecimal characters")
    if type(size) is not int or size < 0:
        raise DownloadError("artifact size must be a non-negative integer")


def _validate_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or urlparse(relative_path).scheme
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in relative_path.split("/"))
        or any(ord(character) < 32 for character in relative_path)
    ):
        raise DownloadError(f"unsafe artifact path {relative_path!r}")


def _download_from_mirror(mirror: str, relative_path: str, partial: Path, size: int) -> str:
    parsed = urlparse(mirror)
    if parsed.scheme and not Path(mirror).is_absolute():
        return _download_http(mirror, relative_path, partial, size)
    return _download_local(Path(mirror), relative_path, partial, size)


def _download_local(mirror: Path, relative_path: str, partial: Path, size: int) -> str:
    root = mirror.resolve(strict=True)
    source = (root / PurePosixPath(relative_path)).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("artifact is missing")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    with source.open("rb") as stream:
        stream.seek(offset)
        return _append_stream(stream, partial, offset, size)


def _download_http(mirror: str, relative_path: str, partial: Path, size: int) -> str:
    _validate_remote_url(mirror)
    base = mirror if mirror.endswith("/") else f"{mirror}/"
    url = urljoin(base, quote(relative_path, safe="/"))
    _validate_remote_url(url)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with build_opener(_SafeRedirects()).open(request, timeout=30) as response:
        status = response.status
        if offset and status == 206:
            _validate_content_range(response.headers.get("Content-Range"), offset, size)
        elif offset and status == 200:
            partial.unlink(missing_ok=True)
            offset = 0
        elif status != 200:
            raise ValueError(f"unexpected HTTP status {status}")
        return _append_stream(response, partial, offset, size)


def _append_stream(stream: BinaryIO, partial: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as prefix:
            while chunk := prefix.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
                digest.update(chunk)
    mode = "ab" if offset else "wb"
    received = offset
    with partial.open(mode) as output:
        while chunk := stream.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
            received += len(chunk)
            if received > size:
                raise ValueError("response exceeds expected size")
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _validate_content_range(value: str | None, offset: int, size: int) -> None:
    match = _CONTENT_RANGE_PATTERN.fullmatch(value or "")
    if match is None or tuple(int(part) for part in match.groups()) != (offset, size - 1, size):
        raise ValueError("invalid Content-Range for resumed download")


def _matches(path: Path, sha256: str, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def _validate_remote_url(url: str, *, allow_query: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.username or parsed.password or parsed.fragment or (parsed.query and not allow_query):
        raise ValueError("unsafe mirror URL")
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname and _is_loopback(parsed.hostname):
        return
    raise ValueError("mirror URL must use HTTPS or loopback HTTP")


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _mirror_label(mirror: str) -> str:
    parsed = urlparse(mirror)
    if not parsed.scheme:
        return "local mirror"
    return f"{parsed.scheme} mirror {parsed.hostname or 'unknown host'}"


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        return "network error"
    return str(exc) or exc.__class__.__name__
