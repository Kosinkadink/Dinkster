"""Strict engine feed documents and resumable downloads from one mirror.

An engine feed is a set of static JSON documents plus binary artifacts
hosted under a single mirror base URL (Cloudflare R2 in production, a
loopback S3-compatible endpoint in development). Two document types exist:

- Engine manifest (``format="dinkster.engine/1"``): one document per engine
  commit and cell at ``engine/<commit>/<cell>.json``. It names the base
  archive (the relocatable interpreter plus torch), the interpreter path
  inside the extracted base, the base package set, and the code-layer
  wheels for that commit and cell.
- Engine channel (``format="dinkster.engine-channel/1"``): a stable or
  github-live pointer that maps each cell to its manifest artifact.

Parsing is strict: unknown fields, duplicate JSON keys, and any value
outside the documented shape are rejected, so a buggy or compromised
mirror cannot smuggle unexpected content past a consumer.

:class:`Mirror` downloads artifacts with byte-range resume and verifies
the exact size and SHA-256 digest before the destination is atomically
replaced. Redirects are followed only when they stay on the mirror's
origin and under its base prefix, so a redirect can neither leave the
mirror nor leak a request to another host.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "CELLS",
    "CHANNEL_FORMAT",
    "ENVIRONMENTS",
    "EngineBase",
    "EngineChannel",
    "EngineFeedError",
    "EngineManifest",
    "MANIFEST_FORMAT",
    "MAX_JSON_BYTES",
    "MINIMUM_LAUNCHER_VERSION",
    "Artifact",
    "BASE_ARCHIVE_SUFFIXES",
    "Mirror",
    "Wheel",
    "parse_channel",
    "parse_manifest",
]

MANIFEST_FORMAT = "dinkster.engine/1"
"""Format identifier of engine manifest documents."""

CHANNEL_FORMAT = "dinkster.engine-channel/1"
"""Format identifier of engine channel documents."""

MINIMUM_LAUNCHER_VERSION = "0.0.1"
"""Default minimum launcher version a channel writer publishes."""

CELLS = (
    "win-cu128",
    "linux-cu128",
    "mac-arm64",
    "linux-cpu",
    "win-cpu",
    "linux-rocm",
    "windows-rocm",
    "linux-xpu",
    "windows-xpu",
)
"""Cells an engine manifest may be published for."""

ENVIRONMENTS = ("control", "execution")
"""Venv kinds a wheel may be installed into, in canonical order."""

MAX_JSON_BYTES = 4 * 1024 * 1024
"""Largest feed document a mirror will return."""

BASE_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.zst")
"""Suffixes a base archive object key may use; gz archives bootstrap with the stdlib."""

_SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")

_MAX_DOWNLOAD_ATTEMPTS = 5
"""Total attempts (fresh or resumed) per download call."""

_REQUEST_TIMEOUT_S = 60.0
_CHUNK_BYTES = 64 * 1024
_RETRY_BACKOFF_S = 0.1

_HEX_64 = re.compile(r"[0-9a-f]{64}")
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_NORMALIZED_NAME = re.compile(r"[a-z0-9]([a-z0-9._-]*[a-z0-9])?")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class EngineFeedError(ValueError):
    """A feed document, mirror URL, or downloaded artifact is invalid."""


def _validate_digest(value: object, what: str) -> None:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise EngineFeedError(f"{what} must be 64 lowercase hex characters")


def _validate_commit(value: object, what: str) -> None:
    if not isinstance(value, str) or not _HEX_40.fullmatch(value):
        raise EngineFeedError(f"{what} must be a 40 character lowercase hex commit id")


def _validate_size(value: object, what: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EngineFeedError(f"{what} must be a nonnegative integer")


def _validate_version(value: object, what: str) -> None:
    if not isinstance(value, str) or not value:
        raise EngineFeedError(f"{what} must be a nonempty string")
    if _CONTROL_CHARS.search(value) or any(ch.isspace() for ch in value):
        raise EngineFeedError(f"{what} must not contain whitespace or control characters")


def _validate_object_key(value: object, what: str) -> None:
    """Require a strictly relative POSIX object key with nothing to exploit."""
    if not isinstance(value, str) or not value:
        raise EngineFeedError(f"{what} must be a nonempty string")
    if "\\" in value:
        raise EngineFeedError(f"{what} must not contain backslashes")
    if _CONTROL_CHARS.search(value):
        raise EngineFeedError(f"{what} must not contain control characters")
    for character in "?#@:":
        if character in value:
            raise EngineFeedError(f"{what} must not contain {character!r}")
    lowered = value.lower()
    if "%2f" in lowered or "%5c" in lowered:
        raise EngineFeedError(f"{what} must not contain encoded path separators")
    if value.startswith("/") or value.endswith("/"):
        raise EngineFeedError(f"{what} must be relative without leading or trailing slashes")
    for segment in value.split("/"):
        if not segment or segment.startswith("."):
            raise EngineFeedError(f"{what} must not contain empty or traversal path segments")


def _validate_wheel_filename(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise EngineFeedError("wheel filename must be a nonempty string")
    if "/" in value or "\\" in value:
        raise EngineFeedError("wheel filename must be a bare filename without path separators")
    if value.startswith("."):
        raise EngineFeedError("wheel filename must not start with a dot")
    if _CONTROL_CHARS.search(value) or any(ch.isspace() for ch in value):
        raise EngineFeedError("wheel filename must not contain whitespace or control characters")
    for character in "?#@:%":
        if character in value:
            raise EngineFeedError(f"wheel filename must not contain {character!r}")


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _validate_normalized_name(value: object, what: str) -> None:
    if not isinstance(value, str) or not value:
        raise EngineFeedError(f"{what} must be a nonempty string")
    normalized = _normalized_distribution_name(value) if isinstance(value, str) else None
    if not _NORMALIZED_NAME.fullmatch(value) or normalized != value:
        raise EngineFeedError(f"{what} {value!r} is not a normalized distribution name")


def _validate_cell(value: object, what: str) -> None:
    if value not in CELLS:
        raise EngineFeedError(f"{what} must be one of {', '.join(CELLS)}")


def _validate_semver(value: object, what: str) -> None:
    """Require a safe major.minor.patch version a consumer can compare."""
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        raise EngineFeedError(f"{what} must be a numeric major.minor.patch version")


def _canonical_environments(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise EngineFeedError("wheel environments must be a nonempty list")
    present: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value not in ENVIRONMENTS:
            raise EngineFeedError(f"unknown wheel environment {value!r}")
        if value in present:
            raise EngineFeedError(f"duplicate wheel environment {value!r}")
        present.add(value)
    return tuple(environment for environment in ENVIRONMENTS if environment in present)


@dataclass(frozen=True)
class Artifact:
    """A downloadable feed object.

    Attributes:
        path: strictly relative POSIX object key resolved under the mirror
            base URL.
        sha256: lowercase 64-hex digest of the object contents.
        size: exact object size in bytes.
    """

    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_object_key(self.path, "artifact path")
        _validate_digest(self.sha256, "artifact sha256")
        _validate_size(self.size, "artifact size")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON record for this artifact."""
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class Wheel(Artifact):
    """A wheel in an engine manifest's code layer.

    Attributes:
        path: strictly relative POSIX object key of the wheel file.
        sha256: lowercase 64-hex digest of the wheel contents.
        size: exact wheel size in bytes.
        filename: bare wheel file name used when staging the wheelhouse.
        name: normalized distribution name of the wheel.
        version: exact installed version of the distribution.
        environments: nonempty subset of ``control`` and ``execution``, in
            canonical order.
    """

    filename: str
    name: str
    version: str
    environments: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_wheel_filename(self.filename)
        _validate_normalized_name(self.name, "wheel name")
        _validate_version(self.version, "wheel version")
        object.__setattr__(self, "environments", _canonical_environments(self.environments))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON record for this wheel."""
        record = super().to_record()
        record.update(
            {
                "filename": self.filename,
                "name": self.name,
                "version": self.version,
                "environments": list(self.environments),
            }
        )
        return record


@dataclass(frozen=True)
class EngineBase:
    """The per-cell base archive of an engine manifest.

    Attributes:
        id: 64 lowercase hex digest identifying the pinned base set.
        archive: the base archive artifact (interpreter plus torch). The
            builder bundles uv into the base root as ``tools/uv`` (or
            ``tools/uv.exe`` on Windows); the manifest carries no separate
            field for it.
        python: relative path of the interpreter inside the extracted base.
        packages: normalized distribution name to exact version for the
            packages installed in the base: torch, torchvision, and their
            non-Dinkster transitive closure.
    """

    id: str
    archive: Artifact
    python: str
    packages: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_digest(self.id, "base id")
        _validate_object_key(self.python, "base python path")
        if not self.archive.path.lower().endswith(BASE_ARCHIVE_SUFFIXES):
            raise EngineFeedError(
                f"base archive must be one of {', '.join(BASE_ARCHIVE_SUFFIXES)}:"
                f" {self.archive.path!r}"
            )
        packages = dict(self.packages)
        for name, version in packages.items():
            _validate_normalized_name(name, "base package name")
            if name.startswith("dinkster-"):
                raise EngineFeedError(
                    f"base package {name!r}: dinkster distributions ship in the"
                    " code layer, never in the base"
                )
            _validate_version(version, f"base package {name!r} version")
        object.__setattr__(self, "packages", MappingProxyType(packages))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON record for this base."""
        return {
            "id": self.id,
            "archive": self.archive.to_record(),
            "python": self.python,
            "packages": dict(self.packages),
        }


@dataclass(frozen=True)
class EngineManifest:
    """One cell's engine feed for one commit.

    Attributes:
        format: always ``dinkster.engine/1``.
        commit: the 40 character lowercase hex engine commit.
        cell: the cell this manifest was published for.
        base: the base archive and package set.
        wheels: the code-layer wheels; filenames and distribution names are
            unique, and no base distribution may appear as a wheel.
    """

    format: str
    commit: str
    cell: str
    base: EngineBase
    wheels: tuple[Wheel, ...]

    def __post_init__(self) -> None:
        if self.format != MANIFEST_FORMAT:
            raise EngineFeedError(f"manifest format must be {MANIFEST_FORMAT!r}")
        _validate_commit(self.commit, "manifest commit")
        _validate_cell(self.cell, "manifest cell")
        names: set[str] = set()
        filenames: set[str] = set()
        for wheel in self.wheels:
            if wheel.name in names:
                raise EngineFeedError(f"duplicate wheel distribution name {wheel.name!r}")
            if wheel.filename in filenames:
                raise EngineFeedError(f"duplicate wheel filename {wheel.filename!r}")
            if wheel.name in self.base.packages:
                raise EngineFeedError(
                    f"wheel {wheel.name!r} repeats a base distribution; base"
                    " packages never appear in the code layer"
                )
            names.add(wheel.name)
            filenames.add(wheel.filename)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON record for this manifest."""
        return {
            "format": self.format,
            "commit": self.commit,
            "cell": self.cell,
            "base": self.base.to_record(),
            "wheels": [wheel.to_record() for wheel in self.wheels],
        }


@dataclass(frozen=True)
class EngineChannel:
    """A stable or github-live pointer to per-cell engine manifests.

    Attributes:
        format: always ``dinkster.engine-channel/1``.
        channel: ``stable`` or ``github-live``.
        commit: the 40 character lowercase hex engine commit the channel names.
        minimum_launcher_version: the oldest launcher that may consume this
            channel, as a numeric major.minor.patch version; a consumer that
            runs older reports upgrade-required.
        cells: cell name to the manifest artifact ``engine/<commit>/<cell>.json``.
    """

    format: str
    channel: str
    commit: str
    minimum_launcher_version: str
    cells: Mapping[str, Artifact]

    def __post_init__(self) -> None:
        if self.format != CHANNEL_FORMAT:
            raise EngineFeedError(f"channel format must be {CHANNEL_FORMAT!r}")
        if self.channel not in ("stable", "github-live"):
            raise EngineFeedError("channel must be 'stable' or 'github-live'")
        _validate_commit(self.commit, "channel commit")
        _validate_semver(self.minimum_launcher_version, "minimum launcher version")
        cells = dict(self.cells)
        for cell, artifact in cells.items():
            _validate_cell(cell, "channel cell")
            expected = f"engine/{self.commit}/{cell}.json"
            if artifact.path != expected:
                raise EngineFeedError(f"cell {cell!r} must point at {expected!r}")
        object.__setattr__(self, "cells", MappingProxyType(cells))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON record for this channel."""
        return {
            "format": self.format,
            "channel": self.channel,
            "commit": self.commit,
            "minimumLauncherVersion": self.minimum_launcher_version,
            "cells": {cell: artifact.to_record() for cell, artifact in self.cells.items()},
        }


def _reject_constant(value: str) -> Any:
    raise EngineFeedError(f"feed documents must not contain {value}")


def _load_json(data: bytes, what: str) -> Any:
    if not isinstance(data, bytes):
        raise EngineFeedError(f"{what} must be parsed from bytes")
    if b"\xef\xbb\xbf" == data[:3]:
        raise EngineFeedError(f"{what} must be UTF-8 without a byte order mark")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EngineFeedError(f"{what} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EngineFeedError(f"{what} is not valid UTF-8") from error
    try:
        return json.loads(
            text, object_pairs_hook=reject_duplicates, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        raise EngineFeedError(f"{what} is not valid JSON") from error


def _expect_keys(record: object, expected: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise EngineFeedError(f"{what} must be a JSON object")
    missing = expected - record.keys()
    unknown = record.keys() - expected
    if missing:
        raise EngineFeedError(f"{what} is missing fields {sorted(missing)}")
    if unknown:
        raise EngineFeedError(f"{what} has unknown fields {sorted(unknown)}")
    return record


def _parse_artifact(record: object, what: str) -> Artifact:
    body = _expect_keys(record, frozenset({"path", "sha256", "size"}), what)
    return Artifact(path=body["path"], sha256=body["sha256"], size=body["size"])


def _parse_wheel(record: object, what: str) -> Wheel:
    body = _expect_keys(
        record,
        frozenset({"path", "sha256", "size", "filename", "name", "version", "environments"}),
        what,
    )
    return Wheel(
        path=body["path"],
        sha256=body["sha256"],
        size=body["size"],
        filename=body["filename"],
        name=body["name"],
        version=body["version"],
        environments=body["environments"],
    )


def parse_manifest(data: bytes) -> EngineManifest:
    """Parse strict engine manifest bytes into an :class:`EngineManifest`."""
    body = _expect_keys(
        _load_json(data, "engine manifest"),
        frozenset({"format", "commit", "cell", "base", "wheels"}),
        "engine manifest",
    )
    if body["format"] != MANIFEST_FORMAT:
        raise EngineFeedError(f"manifest format must be {MANIFEST_FORMAT!r}")
    base = _expect_keys(
        body["base"],
        frozenset({"id", "archive", "python", "packages"}),
        "engine manifest base",
    )
    packages = base["packages"]
    if not isinstance(packages, dict):
        raise EngineFeedError("engine manifest base packages must be a JSON object")
    wheels = body["wheels"]
    if not isinstance(wheels, list):
        raise EngineFeedError("engine manifest wheels must be a JSON list")
    return EngineManifest(
        format=body["format"],
        commit=body["commit"],
        cell=body["cell"],
        base=EngineBase(
            id=base["id"],
            archive=_parse_artifact(base["archive"], "engine manifest base archive"),
            python=base["python"],
            packages=packages,
        ),
        wheels=tuple(
            _parse_wheel(wheel, f"engine manifest wheel {index}")
            for index, wheel in enumerate(wheels)
        ),
    )


def parse_channel(data: bytes) -> EngineChannel:
    """Parse strict engine channel bytes into an :class:`EngineChannel`."""
    body = _expect_keys(
        _load_json(data, "engine channel"),
        frozenset({"format", "channel", "commit", "minimumLauncherVersion", "cells"}),
        "engine channel",
    )
    if body["format"] != CHANNEL_FORMAT:
        raise EngineFeedError(f"channel format must be {CHANNEL_FORMAT!r}")
    cells = body["cells"]
    if not isinstance(cells, dict) or not cells:
        raise EngineFeedError("engine channel cells must be a nonempty JSON object")
    return EngineChannel(
        format=body["format"],
        channel=body["channel"],
        commit=body["commit"],
        minimum_launcher_version=body["minimumLauncherVersion"],
        cells={
            cell: _parse_artifact(record, f"engine channel cell {cell!r}")
            for cell, record in cells.items()
        },
    )


def _is_loopback_host(host: str) -> bool:
    return host == "localhost" or host == "::1" or host.startswith("127.")


def _path_under_prefix(path: str, base_path: str) -> bool:
    segments = path.split("/")
    if ".." in segments or "%2e%2e" in path.lower():
        return False
    if not base_path:
        return True
    return path == base_path or path.startswith(base_path + "/")


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    """Redirect handler that never leaves the mirror origin or base prefix."""

    def __init__(self, mirror: Mirror) -> None:
        self._mirror = mirror

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        split = urllib.parse.urlsplit(newurl)
        if split.username is not None or split.password is not None:
            raise EngineFeedError(f"mirror redirect carries credentials: {newurl}")
        if split.query or split.fragment:
            raise EngineFeedError(f"mirror redirect carries a query or fragment: {newurl}")
        try:
            port = split.port
        except ValueError as error:
            raise EngineFeedError(f"mirror redirect has an invalid port: {newurl}") from error
        if (
            split.scheme != self._mirror._scheme
            or split.hostname != self._mirror._host
            or port != self._mirror._port
        ):
            raise EngineFeedError(f"mirror redirect leaves the mirror origin: {newurl}")
        if not _path_under_prefix(split.path, self._mirror._base_path):
            raise EngineFeedError(f"mirror redirect escapes the base prefix: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _RecoverableTransfer(Exception):
    """A transfer failure a later attempt may resolve."""


class Mirror:
    """Feed mirror client with verified, resumable artifact downloads.

    Attributes:
        base_url: mirror root every artifact path resolves under. Must be
            HTTPS, or plain HTTP for a loopback host when
            ``allow_local_http`` is set (local development mirrors only).
        allow_local_http: when True, plain HTTP is accepted for loopback
            hosts only; every other host requires HTTPS. Credentials, query
            strings, and fragments are never allowed in the base URL.
    """

    def __init__(self, base_url: str, allow_local_http: bool = False) -> None:
        self._scheme, self._host, self._port, self._base_path = _parse_base_url(
            base_url, allow_local_http
        )
        self.base_url = base_url
        self.allow_local_http = allow_local_http
        self._opener = urllib.request.build_opener(_SameOriginRedirects(self))

    def get_json(self, path: str) -> bytes:
        """Fetch the feed document at ``path``; returns at most 4 MiB of bytes."""
        _validate_object_key(path, "feed path")
        try:
            response = self._open(urllib.request.Request(self._url(path)))
        except _RecoverableTransfer as error:
            raise EngineFeedError(f"fetching {path} from the mirror failed: {error}") from error
        with response:
            if response.status != 200:
                raise EngineFeedError(f"mirror returned HTTP {response.status} for {path}")
            payload = response.read(MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            raise EngineFeedError(f"feed document {path} exceeds {MAX_JSON_BYTES} bytes")
        return payload

    def download(self, artifact: Artifact, destination: Path) -> Path:
        """Download ``artifact`` to ``destination`` and return the path.

        Data streams into a sibling ``.part`` file. A partial ``.part`` is
        resumed with a byte-range request; an ignored or invalid range resets
        the part instead of concatenating bad bytes. Every interruption a
        later attempt may resolve is retried up to a bounded number of times
        and leaves the partial bytes in place. Data that completes but fails
        verification is deleted. An existing destination is reverified
        against the artifact's size and digest first; a corrupt destination
        is deleted. The destination is replaced atomically only after the
        exact size and SHA-256 digest have been verified.
        """
        destination = Path(destination)
        if destination.is_dir():
            raise EngineFeedError(f"download destination is a directory: {destination}")
        if destination.exists():
            if _file_matches(destination, artifact):
                return destination
            destination.unlink()  # complete data that no longer verifies is corrupt
        part = destination.with_name(destination.name + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
            if self._stream_once(artifact, part):
                os.replace(part, destination)
                return destination
            if attempt + 1 < _MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(min(_RETRY_BACKOFF_S * attempt, 1.0))
        raise EngineFeedError(
            f"download of {artifact.path} did not complete in {_MAX_DOWNLOAD_ATTEMPTS} attempts"
        )

    def _stream_once(self, artifact: Artifact, part: Path) -> bool:
        """Run one download attempt; True means ``part`` now verifies.

        Raises :class:`EngineFeedError` only for failures no retry can fix;
        every recoverable failure returns False and leaves the partial bytes
        in place for the next attempt.
        """
        part_size = part.stat().st_size if part.exists() else 0
        if part_size > artifact.size:
            _truncate(part)
            part_size = 0
        if part_size == artifact.size:
            if _file_matches(part, artifact):
                return True
            _truncate(part)
            part_size = 0
        resuming = part_size > 0
        headers = {"Range": f"bytes={part_size}-"} if resuming else {}
        try:
            response = self._open(urllib.request.Request(self._url(artifact.path), headers=headers))
            with response:
                status = response.status
                if status == 206:
                    if not resuming or not _content_range_continues(
                        response.headers.get("Content-Range"), part_size, artifact.size
                    ):
                        _truncate(part)
                        return False
                elif status == 200 and resuming:
                    # The server ignored the range request: retry from zero
                    # rather than appending a second copy of the object.
                    _truncate(part)
                    return False
                elif status == 416:
                    _truncate(part)
                    return False
                elif status != 200:
                    raise EngineFeedError(f"mirror returned HTTP {status} for {artifact.path}")
                digest = hashlib.sha256()
                if part_size:
                    _hash_file(part, digest)
                remaining = artifact.size - part_size
                with part.open("ab") as sink:
                    while remaining > 0:
                        chunk = response.read(min(_CHUNK_BYTES, remaining))
                        if not chunk:
                            break  # the connection ended before the object did
                        sink.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
        except _RecoverableTransfer:
            return False
        if remaining:
            return False  # truncated transfer; partial bytes stay for the next resume
        if digest.hexdigest() != artifact.sha256:
            part.unlink()  # staged bytes are corrupt; never promote them
            raise EngineFeedError(f"downloaded {artifact.path} does not match its sha256 digest")
        return True

    def _open(self, request: urllib.request.Request) -> urllib.response.addinfourl:
        try:
            return self._opener.open(request, timeout=_REQUEST_TIMEOUT_S)
        except urllib.error.HTTPError as error:
            if 500 <= error.code < 600:
                raise _RecoverableTransfer(f"HTTP {error.code}") from error
            raise EngineFeedError(
                f"mirror returned HTTP {error.code} for {request.full_url}"
            ) from error
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as error:
            raise _RecoverableTransfer(str(error)) from error

    def _url(self, path: str) -> str:
        host = f"[{self._host}]" if ":" in self._host else self._host
        netloc = host if self._port is None else f"{host}:{self._port}"
        return urllib.parse.urlunsplit((self._scheme, netloc, f"{self._base_path}/{path}", "", ""))


def _parse_base_url(base_url: object, allow_local_http: bool) -> tuple[str, str, int | None, str]:
    what = "mirror base URL"
    if not isinstance(base_url, str) or not base_url or base_url.strip() != base_url:
        raise EngineFeedError(f"{what} must be a nonempty URL without surrounding whitespace")
    if _CONTROL_CHARS.search(base_url) or any(ch.isspace() for ch in base_url):
        raise EngineFeedError(f"{what} must not contain whitespace or control characters")
    split = urllib.parse.urlsplit(base_url)
    if split.username is not None or split.password is not None:
        raise EngineFeedError(f"{what} must not carry credentials")
    if split.query or split.fragment:
        raise EngineFeedError(f"{what} must not carry a query or fragment")
    if not split.hostname:
        raise EngineFeedError(f"{what} must name a host")
    scheme = split.scheme.lower()
    host = split.hostname
    if scheme == "https":
        pass
    elif scheme == "http" and allow_local_http and _is_loopback_host(host):
        pass
    else:
        raise EngineFeedError(
            f"{what} must use HTTPS (plain HTTP is allowed for loopback hosts only"
            " when allow_local_http is set)"
        )
    try:
        port = split.port
    except ValueError as error:
        raise EngineFeedError(f"{what} has an invalid port") from error
    if ".." in split.path.split("/"):
        raise EngineFeedError(f"{what} must not contain traversal segments")
    return scheme, host, port, split.path.rstrip("/")


def _content_range_continues(header: object, offset: int, total: int) -> bool:
    if not isinstance(header, str):
        return False
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", header.strip())
    if match is None:
        return False
    return int(match.group(1)) == offset and int(match.group(3)) == total


def _hash_file(path: Path, digest: Any) -> None:
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)


def _file_matches(path: Path, artifact: Artifact) -> bool:
    try:
        if path.stat().st_size != artifact.size:
            return False
        digest = hashlib.sha256()
        _hash_file(path, digest)
    except OSError:
        return False
    return digest.hexdigest() == artifact.sha256


def _truncate(part: Path) -> None:
    with part.open("wb"):
        pass
