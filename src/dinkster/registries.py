"""Named registries: client config + pack@version resolution.

The client half of the registry index. Three rules carried over from the
design, enforced here so no caller can weaken them:

- RESOLUTION IS NEVER A SEARCH PATH. ``resolve_release`` asks exactly
  ONE registry - the one the user named (``--registry NAME_OR_URL``) or
  the one configured default - never an ordered fallback list across
  registries (pip's --extra-index-url model is exactly the
  dependency-confusion attack surface where a public name shadows a
  private one). With multiple registries configured and no default,
  resolution refuses loudly and names the choices.
- THE REGISTRY IS PROVENANCE, NEVER IDENTITY. Resolution records
  ``registry:<name>`` (configured) or ``registry:<url>`` (ad-hoc) as the
  entry's source; identity stays the artifact digest, and acquisition
  verifies the bytes against it, so even a lying index cannot
  substitute content - only refuse or confess.
- CREDENTIALS ARE PER REGISTRY ENDPOINT, NEVER IN THE CONFIG FILE.
  ``registries.toml`` names an ENVIRONMENT VARIABLE per registry
  (``token-env``); the plaintext token lives in the environment, exactly
  like $DINKSTER_REGISTRY_TOKEN does for the ad-hoc ``--registry URL``
  form. A config file that embedded tokens would end up committed.

Config shape (``registries.toml``, by default ``<root>/registries.toml``
or $DINKSTER_REGISTRIES)::

    [registries.public]
    endpoint = "https://registry.example.org"
    default = true

    [registries.corp]
    endpoint = "https://registry.corp.internal"
    token-env = "CORP_REGISTRY_TOKEN"

Registry names ride the one name grammar and are keyed by canonical
form, so ``corp.x`` and ``corp-x`` can never be two registries. Loading
is loud:
unknown keys, non-http(s) endpoints, duplicate canonical names, and
multiple defaults all refuse with the fix named - config is
configuration, never best-effort.

``routing_registry_fetcher`` is the acquisition side: one
``RegistryFetcher`` that dispatches each lockfile entry on its recorded
``source`` qualifier - bare ``"registry"`` means the selected/default
registry, ``registry:<name>`` means that configured registry (unknown
names refuse naming the configured ones), ``registry:<url>`` fetches
from that endpoint directly (with the matching configured credential
when one exists). Downloads stay content-addressed and digest-verified
in ``Installer.acquire`` - routing chooses WHERE to ask, never WHAT
counts as the right bytes.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from dinkster_registry import (
    ArtifactError,
    InstallError,
    LockedPack,
    Lockfile,
    artifact_digest,
    unpack_artifact,
    validate_artifact_digest,
    validate_version,
)
from dinkster_schema import PackCompatibility, canonical_claim, canonical_name, validate_name
from dinkster_schema.units import MEBIBYTE
from dinkster_store import pack_archive_path, pack_label
from dinkster_workers import ManifestError, load_manifest
from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
from dinkster_workers.pack_wheel import pack_manifest_path
from dinkster_workers.provision import ProvisionError
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

from dinkster.installer import RegistryFetcher, http_registry_fetcher, read_registry_response

DEFAULT_REGISTRIES_FILENAME = "registries.toml"
DEFAULT_PACKS_PATH = Path(__file__).with_name("default-packs.toml")
AD_HOC_TOKEN_ENV = "DINKSTER_REGISTRY_TOKEN"
DEFAULT_RESOLVE_TIMEOUT = 60.0


class RegistryPublishError(Exception):
    """The publish client could not complete or decode the registry exchange."""


@dataclass(frozen=True)
class PublishFinding:
    code: str
    message: str


@dataclass(frozen=True)
class PublishVerdict:
    state: str
    findings: tuple[PublishFinding, ...]
    candidate_id: str = ""


def _publish_request(
    registry: NamedRegistry,
    request: urllib.request.Request,
    *,
    action: str,
    timeout: float,
) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as exc:
        try:
            body: object = json.loads(exc.read())
            error = cast("dict[str, object]", body).get("error")
            message = (
                cast("dict[str, object]", error).get("message")
                if isinstance(error, dict)
                else error
            )
        except (ValueError, AttributeError):
            message = None
        detail = f": {message}" if isinstance(message, str) else ""
        raise RegistryPublishError(
            f"{action} against {registry.label} failed: HTTP {exc.code}{detail}"
        ) from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RegistryPublishError(f"{action} against {registry.label} failed: {exc}") from exc


def publish_release(
    registry: NamedRegistry,
    archive: Path,
    pack_name: str,
    version: str,
    *,
    locks: Path | None = None,
    engine: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> PublishVerdict:
    """Submit exact artifact bytes with optional base locks and scan engine identity."""
    if engine is not None and locks is None:
        raise RegistryPublishError("engine binding requires publication locks")
    try:
        data = archive.read_bytes()
        lock_data = locks.read_bytes() if locks is not None else None
    except OSError as exc:
        raise RegistryPublishError(f"cannot read publication input: {exc}") from exc
    if re.fullmatch(r"[A-Za-z0-9_.+\-]+\.whl", archive.name) is None:
        raise RegistryPublishError("wheel.required: publication requires a wheel")
    content_type = "application/octet-stream"
    if lock_data is not None:
        boundary = "dinkster-" + uuid.uuid4().hex
        content_type = f"multipart/form-data; boundary={boundary}"
        data = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="archive"; '
            f'filename="{archive.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(
                "ascii"
            )
            + data
            + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="locks"\r\n'
            "Content-Type: application/json\r\n\r\n".encode("ascii")
            + lock_data
        )
        if engine is not None:
            data += (
                f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="engine"\r\n'
                "Content-Type: application/json\r\n\r\n".encode("ascii")
                + json.dumps(dict(engine), sort_keys=True, separators=(",", ":")).encode("ascii")
            )
        data += f"\r\n--{boundary}--\r\n".encode("ascii")
    headers = {"User-Agent": "dinkster-pack"}
    token = registry.token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    submit = urllib.request.Request(
        f"{registry.endpoint}/v1/publish/{pack_name}/versions/{version}",
        data=data,
        headers={**headers, "Content-Type": content_type},
        method="POST",
    )
    payload = _publish_request(registry, submit, action="publish submission", timeout=timeout)
    try:
        decoded: object = json.loads(payload)
        document = cast("dict[str, object]", decoded)
        state = document["status"]
        candidate_id = document["candidateId"]
        if (
            not isinstance(state, str)
            or state not in {"queued", "scanning", "retrying", "accepted"}
            or not isinstance(candidate_id, str)
        ):
            raise TypeError
    except (ValueError, KeyError, TypeError) as exc:
        raise RegistryPublishError(
            f"registry {registry.label} answered publish submission with a malformed receipt"
        ) from exc
    return PublishVerdict(state=state, findings=(), candidate_id=candidate_id)


@dataclass(frozen=True)
class NamedRegistry:
    """One configured registry: a local label over an endpoint plus the
    NAME of the environment variable holding its credential."""

    name: str
    """Canonical local label; empty for an ad-hoc ``--registry URL``."""
    endpoint: str
    token_env: str = ""
    default: bool = False

    def token(self) -> str:
        return os.environ.get(self.token_env, "") if self.token_env else ""

    @property
    def source(self) -> str:
        """The provenance qualifier resolution records: the NAME when
        configured (portable across machines that configure the same
        label), the URL for ad-hoc endpoints."""
        return f"registry:{self.name or self.endpoint}"

    @property
    def label(self) -> str:
        return self.name or self.endpoint


@dataclass(frozen=True)
class RegistryConfig:
    """The registries table. Empty is valid (no named registries)."""

    registries: tuple[NamedRegistry, ...] = ()

    def get(self, name: str) -> NamedRegistry | None:
        wanted = canonical_name(name)
        for registry in self.registries:
            if registry.name == wanted:
                return registry
        return None

    def default_registry(self) -> NamedRegistry | None:
        """The one registry a bare ``"registry"`` source or unqualified
        ``pack@version`` means: the flagged default, or the single
        configured entry. Multiple entries without a flag mean NO
        default - the caller must name one."""
        for registry in self.registries:
            if registry.default:
                return registry
        if len(self.registries) == 1:
            return self.registries[0]
        return None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(registry.name for registry in self.registries)

    def registry_for(
        self, entry: LockedPack, selected: NamedRegistry | None = None
    ) -> NamedRegistry:
        """Route recorded provenance without searching another registry."""
        source = entry.source
        if source == "registry":
            chosen = selected if selected is not None else self.default_registry()
            if chosen is None:
                raise InstallError(
                    f"cannot acquire {entry.pack}: source 'registry' names no "
                    f"specific registry and none is the default - pass "
                    f"--registry or flag one 'default = true'"
                )
            return chosen
        qualifier = source.removeprefix("registry:")
        if qualifier.startswith(("http://", "https://")):
            endpoint = qualifier.rstrip("/")
            if selected is not None and selected.endpoint == endpoint:
                return selected
            for registry in self.registries:
                if registry.endpoint == endpoint:
                    return registry
            return NamedRegistry(name="", endpoint=endpoint)
        named = self.get(qualifier)
        if named is None:
            configured = ", ".join(self.names) if self.names else "none configured"
            raise InstallError(
                f"cannot acquire {entry.pack}: recorded source {source!r} names "
                f"registry {qualifier!r}, which is not configured "
                f"(configured: {configured}); add it to registries.toml"
            )
        return named


def _normalized_endpoint(label: str, endpoint: str) -> str:
    if not endpoint.startswith(("http://", "https://")):
        raise InstallError(f"registry {label!r} endpoint {endpoint!r} must be an http(s) URL")
    return endpoint.rstrip("/")


def load_registries(path: Path) -> RegistryConfig:
    """Parse ``registries.toml`` - loudly. A missing file is the
    caller's decision (an explicit --registries path must exist; the
    default location is only read when present)."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InstallError(f"cannot read registries config {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise InstallError(f"registries config {path} is not valid TOML: {exc}") from exc

    unknown_top = set(document) - {"registries"}
    if unknown_top:
        raise InstallError(
            f"registries config {path} has unknown top-level keys "
            f"{sorted(unknown_top)}; expected only [registries.<name>] tables"
        )
    table = document.get("registries", {})
    if not isinstance(table, dict):
        raise InstallError(f"registries config {path}: 'registries' must be a table")

    entries: list[NamedRegistry] = []
    for raw_name, raw_entry in cast("dict[str, object]", table).items():
        problem = validate_name(raw_name)
        if problem is not None:
            raise InstallError(f"registry name {raw_name!r} {problem}")
        name = canonical_name(raw_name)
        if any(entry.name == name for entry in entries):
            raise InstallError(f"registries config {path} names {name!r} twice")
        if not isinstance(raw_entry, dict):
            raise InstallError(f"registry {name!r} must be a table")
        fields = cast("dict[str, object]", raw_entry)
        unknown = set(fields) - {"endpoint", "token-env", "default"}
        if unknown:
            raise InstallError(
                f"registry {name!r} has unknown keys {sorted(unknown)}; "
                f"expected endpoint, token-env, default"
            )
        endpoint = fields.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise InstallError(f"registry {name!r} requires an endpoint string")
        token_env = fields.get("token-env", "")
        if not isinstance(token_env, str):
            raise InstallError(f"registry {name!r} token-env must be a string")
        default = fields.get("default", False)
        if not isinstance(default, bool):
            raise InstallError(f"registry {name!r} default must be a boolean")
        entries.append(
            NamedRegistry(
                name=name,
                endpoint=_normalized_endpoint(name, endpoint),
                token_env=token_env,
                default=default,
            )
        )

    defaults = [entry.name for entry in entries if entry.default]
    if len(defaults) > 1:
        raise InstallError(
            f"registries config {path} flags multiple defaults {defaults}; "
            f"exactly one registry may be the default"
        )
    return RegistryConfig(registries=tuple(entries))


def select_registry(config: RegistryConfig, selector: str) -> NamedRegistry:
    """The ONE registry a resolution or bare-source fetch means.

    ``selector`` is the --registry value: a configured name, an http(s)
    URL (ad-hoc, credential from $DINKSTER_REGISTRY_TOKEN), or empty -
    which means the configured default and refuses loudly when several
    registries compete. Never a search across registries."""
    if not selector:
        chosen = config.default_registry()
        if chosen is None:
            if config.registries:
                raise InstallError(
                    f"multiple registries are configured "
                    f"({', '.join(config.names)}) and none is the default; "
                    f"name one with --registry or flag one 'default = true'"
                )
            raise InstallError(
                "no registry is configured: pass --registry NAME_OR_URL, set "
                "$DINKSTER_REGISTRY, or add one to registries.toml"
            )
        return chosen
    if selector.startswith(("http://", "https://")):
        return NamedRegistry(
            name="",
            endpoint=_normalized_endpoint(selector, selector),
            token_env=AD_HOC_TOKEN_ENV,
        )
    named = config.get(selector)
    if named is None:
        configured = ", ".join(config.names) if config.names else "none configured"
        raise InstallError(
            f"unknown registry {selector!r} (configured: {configured}); "
            f"--registry takes a configured name or an http(s) URL"
        )
    return named


def routing_registry_fetcher(
    config: RegistryConfig, selected: NamedRegistry | None, *, index: StaticPackIndex | None = None
) -> RegistryFetcher:
    """One fetcher for every registry-sourced entry, dispatching on the
    entry's recorded provenance. ``selected`` is the --registry choice
    (already resolved through :func:`select_registry`); bare
    ``"registry"`` sources go there, falling back to the configured
    default."""
    index = index or StaticPackIndex()

    def fetch(entry: LockedPack) -> bytes:
        registry = config.registry_for(entry, selected)
        return http_registry_fetcher(registry.endpoint, token=registry.token(), index=index)(entry)

    return fetch


def _index_get(
    registry: NamedRegistry,
    path: str,
    *,
    action: str,
    subject: str,
    missing: str = "",
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
    token: str | None = None,
) -> dict[str, object]:
    """Read a registry index path as a JSON object, with bounded transport retries,
    or refused as an :class:`InstallError` that names the registry and
    what was being asked. ``missing`` is the message a 404 earns when the
    caller knows what absence means; without it a 404 is just another
    failed HTTP status."""
    url = f"{registry.endpoint}{path}"
    headers = {"User-Agent": "dinkster-pack"}
    token = registry.token() if token is None else token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - scheme checked at construction
    try:
        payload, warning = read_registry_response(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and missing:
            raise InstallError(missing) from exc
        raise InstallError(f"{action} against {registry.label} failed: HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError, http.client.IncompleteRead) as exc:
        raise InstallError(f"{action} against {registry.label} failed: {exc}") from exc

    try:
        document: object = json.loads(payload)
    except ValueError as exc:
        raise InstallError(
            f"registry {registry.label} answered {subject} with invalid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise InstallError(
            f"registry {registry.label} answered {subject} with a non-object document"
        )
    if warning:
        warnings.warn(warning, UserWarning, stacklevel=2)
    return cast("dict[str, object]", document)


class StaticPackIndex:
    """Cache one operation's documents, isolated by endpoint and credential.

    A caller-owned scratch directory shares the snapshot across preparation
    subprocesses. The caller removes it with the rest of the operation's inputs.
    Credentials are hashed into filenames, never written to the snapshot.
    """

    def __init__(
        self, directory: Path | None = None, *, offline: bool = False, read_only: bool = False
    ) -> None:
        self._documents: dict[tuple[str, str, str], dict[str, object]] = {}
        self._locks: dict[tuple[str, str, str], bytes] = {}
        self._directory = directory
        self._offline = offline
        self._read_only = read_only

    def _cache_path(self, key: tuple[str, str, str], suffix: str) -> Path | None:
        return (
            self._directory / (hashlib.sha256(json.dumps(key).encode()).hexdigest() + suffix)
            if self._directory is not None
            else None
        )

    def _read(
        self, registry: NamedRegistry, path: str, timeout: float, token: str | None
    ) -> dict[str, object]:
        credential = registry.token() if token is None else token
        key = (registry.endpoint, credential, path)
        if key not in self._documents:
            cached = self._cache_path(key, ".json")
            if cached is not None and cached.exists():
                try:
                    raw: object = json.loads(cached.read_bytes())
                except ValueError as exc:
                    raise InstallError("operation static index snapshot is invalid") from exc
                if not isinstance(raw, dict):
                    raise InstallError("operation static index snapshot is not an object")
                document = cast("dict[str, object]", raw)
            else:
                if self._offline:
                    raise InstallError(
                        f"offline registry metadata missing: {registry.label} {path}"
                    )
                document = _index_get(
                    registry,
                    path,
                    action="reading static pack index",
                    subject="static pack index",
                    timeout=timeout,
                    token=credential,
                )
            schema = document.get("schemaVersion")
            if type(schema) is not int or schema not in (1, 2):
                raise InstallError(
                    f"registry static index has unsupported schema {schema!r}; update Dinkster"
                )
            if not isinstance(document.get("items"), list):
                raise InstallError(
                    "registry static index has an unsupported schema or missing items"
                )
            if (
                cached is not None
                and not self._offline
                and not self._read_only
                and not cached.exists()
            ):
                cached.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                cached.write_text(json.dumps(document), encoding="utf-8")
            self._documents[key] = document
        return self._documents[key]

    def lock_bytes(
        self, registry: NamedRegistry, path: str, digest: str, size: int, *, timeout: float
    ) -> bytes:
        """Read bound lock metadata; offline snapshots never fetch or repair missing inputs."""
        token = registry.token()
        key = (registry.endpoint, token, path)
        cached = self._cache_path(key, ".lock")
        if key in self._locks:
            payload = self._locks[key]
        elif cached is not None and cached.is_file():
            with cached.open("rb") as handle:
                payload = handle.read(size + 1)
        else:
            if self._offline:
                raise InstallError(f"offline dependency lock missing: {registry.label} {path}")
            headers = {"User-Agent": "dinkster-pack"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(registry.endpoint + path, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = response.read(size + 1)
        if len(payload) != size or artifact_digest(payload) != digest:
            raise InstallError("published lock bytes differ from binding")
        if cached is not None and not self._offline and not self._read_only and not cached.exists():
            cached.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            cached.write_bytes(payload)
        self._locks[key] = payload
        return payload

    def catalog(
        self,
        registry: NamedRegistry,
        *,
        timeout: float = DEFAULT_RESOLVE_TIMEOUT,
        token: str | None = None,
        include_libraries: bool = False,
    ) -> tuple[dict[str, object], ...]:
        document = self._read(registry, "/static/packs/index.json", timeout, token)
        if document["schemaVersion"] == 1:
            return _static_releases(registry, document, include_libraries=include_libraries)
        entries: list[dict[str, object]] = []
        names: set[str] = set()
        for value in cast("list[object]", document["items"]):
            if not isinstance(value, dict):
                raise InstallError("registry static catalog contains a non-object pack")
            row = cast("dict[str, object]", value)
            name, version = row.get("name"), row.get("latestVersion")
            if (
                not isinstance(name, str)
                or validate_name(name) is not None
                or canonical_name(name) != name
                or not isinstance(version, str)
                or validate_version(version) is not None
                or row.get("url") != f"packs/{name}/index.json"
                or type(row.get("library")) is not bool
                or "publisher" not in row
                or row["publisher"] is not None
                and not isinstance(row["publisher"], str)
            ):
                raise InstallError("registry static catalog has invalid pack fields")
            if name in names:
                raise InstallError(f"registry static catalog repeats {name}")
            names.add(name)
            if include_libraries or row["library"] is not True:
                entries.append(row)
        return tuple(entries)

    def releases(
        self,
        registry: NamedRegistry,
        pack: str,
        *,
        timeout: float = DEFAULT_RESOLVE_TIMEOUT,
        token: str | None = None,
        include_libraries: bool = False,
    ) -> tuple[dict[str, object], ...]:
        catalog = self.catalog(
            registry, timeout=timeout, token=token, include_libraries=include_libraries
        )
        document = self._read(registry, "/static/packs/index.json", timeout, token)
        if document["schemaVersion"] == 1:
            return tuple(row for row in catalog if row["name"] == pack)
        entry = next((row for row in catalog if row["name"] == pack), None)
        if entry is None:
            return ()
        document = self._read(registry, "/static/" + str(entry["url"]), timeout, token)
        if document["schemaVersion"] != 2 or document.get("name") != pack:
            raise InstallError(f"registry static per-pack index identity differs for {pack}")
        releases = _static_releases(registry, document, include_libraries=include_libraries)
        if any(row["name"] != pack for row in releases):
            raise InstallError(
                f"registry static per-pack index substitutes another pack for {pack}"
            )
        return releases


def _static_releases(
    registry: NamedRegistry,
    document: Mapping[str, object],
    *,
    include_libraries: bool = False,
) -> tuple[dict[str, object], ...]:
    rows = document.get("items")
    releases: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    for value in cast("list[object]", rows):
        if not isinstance(value, dict):
            raise InstallError("registry static index contains a non-object release")
        row = cast("dict[str, object]", value)
        if row.get("library") is True and not include_libraries:
            continue
        name, version = row.get("name"), row.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise InstallError("registry static release needs name and version strings")
        if (
            validate_name(name) is not None
            or canonical_name(name) != name
            or validate_version(version) is not None
        ):
            raise InstallError("registry static release has invalid name or version")
        identity = (name, version)
        if identity in identities:
            raise InstallError(f"registry static index repeats {name}@{version}")
        identities.add(identity)
        static_release_wheel(registry, row)
        releases.append(row)
    return tuple(releases)


def static_release_wheel(
    registry: NamedRegistry, row: Mapping[str, object]
) -> tuple[LockedWheel, str]:
    """Select compatible indexed bytes and constrain their URL to the static prefix."""
    candidates = row.get("wheels")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise InstallError("registry release must contain one universal wheel in wheels")
    value = candidates[0]
    if not isinstance(value, dict):
        raise InstallError("registry release wheel must be an object")
    fields = cast("dict[str, object]", value)
    filename, sha256, size, relative = (
        fields.get("filename"),
        fields.get("sha256"),
        fields.get("size"),
        fields.get("url"),
    )
    digest = fields.get("digest")
    name, version = row.get("name"), row.get("version")
    if (
        not isinstance(filename, str)
        or not isinstance(sha256, str)
        or not isinstance(digest, str)
        or validate_artifact_digest(digest) is not None
        or type(size) is not int
        or not isinstance(name, str)
        or not isinstance(version, str)
        or relative != f"wheels/{sha256}/{filename}"
    ):
        raise InstallError("registry static release has invalid wheel fields")
    try:
        tags = parse_wheel_filename(filename)[3]
        if fields.get("tags") != ["py3-none-any"] or {str(tag) for tag in tags} != {"py3-none-any"}:
            raise ValueError("wheel.platform-tag: only py3-none-any packs are supported")
        if not tags.intersection(sys_tags()):
            raise ValueError("wheel.incompatible: no wheel matches this interpreter")
        wheel = LockedWheel(
            filename,
            cast("int", size),
            sha256,
            registry.endpoint + "/static/" + cast("str", relative),
        )
        LockedDistribution(name, version, (wheel,))
    except (ValueError, ProvisionError) as exc:
        raise InstallError(f"registry static release has invalid wheel identity: {exc}") from exc
    return wheel, digest


def static_pack_release(
    registry: NamedRegistry,
    pack: str,
    version: str,
    *,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
    token: str | None = None,
    index: StaticPackIndex | None = None,
) -> dict[str, object]:
    for row in (index or StaticPackIndex()).releases(
        registry, pack, timeout=timeout, token=token, include_libraries=True
    ):
        if row["name"] == pack and row["version"] == version:
            return row
    raise InstallError(f"registry {registry.label} publishes no release {pack}@{version}")


def resolve_release(
    registry: NamedRegistry,
    pack: str,
    version: str,
    *,
    engine_version: str | None = None,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
    index: StaticPackIndex | None = None,
) -> LockedPack:
    """Turn ``pack@version`` into a digest-pinned :class:`LockedPack`
    by asking ONE registry's index for the exact release - never a
    nearest version, never another registry. The answer is
    cross-checked against the request (a lying index refuses, it does
    not redirect) and only NAMES the digest; acquisition separately
    verifies the actual bytes against it."""
    wanted_pack = canonical_name(pack)
    fields = static_pack_release(registry, wanted_pack, version, timeout=timeout, index=index)
    answered_pack = fields.get("name")
    answered_version = fields.get("version")
    _, digest = static_release_wheel(registry, fields)
    publisher = fields.get("publisher")
    if (
        not isinstance(answered_pack, str)
        or not isinstance(answered_version, str)
        or not isinstance(digest, str)
        or "publisher" not in fields
        or publisher is not None
        and not isinstance(publisher, str)
    ):
        raise InstallError(
            f"registry {registry.label} answered {wanted_pack}@{version} with a "
            f"malformed release record (need name/version, artifact digest, "
            f"and a publishing principal string or null)"
        )
    if answered_pack != wanted_pack or answered_version != version:
        raise InstallError(
            f"registry {registry.label} answered the request for "
            f"{wanted_pack}@{version} with {answered_pack}@{answered_version}; "
            f"refusing the substitution"
        )
    if engine_version is not None:
        engine = fields.get("engine")
        if not isinstance(engine, str):
            raise InstallError(
                f"registry release {wanted_pack}@{version} is not active "
                "with a declared engine range"
            )
        try:
            PackCompatibility(version, engine).require_engine(wanted_pack, engine_version)
        except ValueError as exc:
            raise InstallError(f"registry release {wanted_pack}@{version}: {exc}") from exc
    digest_problem = validate_artifact_digest(digest)
    if digest_problem is not None:
        raise InstallError(
            f"registry {registry.label} answered {wanted_pack}@{version} with "
            f"artifact digest {digest!r}: {digest_problem}"
        )
    return LockedPack(
        pack=wanted_pack,
        version=version,
        artifact_digest=digest,
        publisher=canonical_name(publisher) if publisher is not None else "",
        claims=(),
        source=registry.source,
    )


def resolve_compatible_release(
    registry: NamedRegistry,
    pack: str,
    minimum: str,
    engine_version: str,
    *,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
    index: StaticPackIndex | None = None,
) -> LockedPack:
    """Select the newest active compatible version from ONE registry, then pin its bytes.

    A pack's latest release need not support the target engine.
    """
    wanted_pack = canonical_name(pack)
    minimum_version = Version(minimum)
    target_engine = Version(engine_version)
    selected: Version | None = None
    minimum_engine: str | None = None
    subject = f"versions of {wanted_pack}"
    index = index or StaticPackIndex()
    for entry in index.releases(registry, wanted_pack, timeout=timeout):
        if entry["name"] != wanted_pack:
            continue
        version, engine = entry.get("version"), entry.get("engine")
        if not isinstance(version, str) or not isinstance(engine, str):
            raise InstallError(
                f"registry {registry.label} answered {subject} without version/engine strings"
            )
        try:
            compatibility = PackCompatibility(version, engine)
        except ValueError as exc:
            raise InstallError(f"registry {registry.label} answered {subject}: {exc}") from exc
        candidate = Version(compatibility.version)
        if candidate == minimum_version:
            minimum_engine = compatibility.engine
        if (
            candidate >= minimum_version
            and (selected is None or candidate > selected)
            and SpecifierSet(compatibility.engine).contains(target_engine, prereleases=True)
        ):
            selected = candidate
    if selected is None:
        declaration = (
            f"; {wanted_pack}@{minimum} declares engine {minimum_engine}" if minimum_engine else ""
        )
        raise InstallError(
            f"registry {registry.label} has no active {wanted_pack} >= {minimum} "
            f"compatible with engine {engine_version}{declaration}"
        )
    return resolve_release(
        registry,
        wanted_pack,
        str(selected),
        engine_version=engine_version,
        timeout=timeout,
        index=index,
    )


def registry_source_lock(selected: LockedPack, archive: Path, *, engine_version: str) -> LockedPack:
    """Bind the selected release to its safely unpacked, digest-verified manifest.

    Preserve the registry's authenticated publishing principal. A manifest's
    self-declared publisher cannot grant reserved-namespace authority.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="dinkster-registry-source-") as directory:
            root = unpack_artifact(archive, Path(directory), selected.artifact_digest)
            manifest = load_manifest(pack_manifest_path(root))
            if canonical_name(manifest.name) != selected.pack:
                raise ValueError(f"archive manifest names {manifest.name!r}")
            compatibility = manifest.compatibility
            if compatibility is None:
                raise ValueError("archive manifest has no pack version and engine range")
            if compatibility.version != selected.version:
                raise ValueError(f"archive manifest version is {compatibility.version}")
            compatibility.require_engine(selected.pack, engine_version)
            if not manifest.publisher:
                raise ValueError("archive manifest has no publisher")
            return replace(
                selected,
                claims=tuple(canonical_claim(claim) for claim in manifest.namespaces),
            )
    except (ArtifactError, ManifestError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise InstallError(
            f"registry source {selected.pack}@{selected.version} refused: {exc}"
        ) from exc


def default_pack_minimums() -> dict[str, str]:
    """Read default ids and minimum versions, never installation byte identities."""
    try:
        document = tomllib.loads(DEFAULT_PACKS_PATH.read_text(encoding="utf-8"))
        packs = document.get("packs")
        if set(document) != {"packs"} or not isinstance(packs, dict) or not packs:
            raise ValueError("expected a nonempty packs table")
        minimums: dict[str, str] = {}
        for name, minimum in packs.items():
            if validate_name(name) or canonical_name(name) != name:
                raise ValueError(f"invalid default pack id {name!r}")
            if not isinstance(minimum, str) or validate_version(minimum):
                raise ValueError(f"invalid minimum version for {name}")
            minimums[name] = minimum
        return minimums
    except (OSError, ValueError) as exc:
        raise InstallError(f"default pack declaration refused: {exc}") from exc


def prepare_registry_sources(
    registry: NamedRegistry,
    minimums: Mapping[str, str],
    artifacts: Path,
    *,
    engine_version: str,
    allow_missing: bool = False,
    index: StaticPackIndex | None = None,
) -> tuple[Lockfile, dict[str, str]]:
    """Resolve target-compatible sources without mutating any generation.

    Fresh setup may retain available defaults and report unavailable packs.
    Update requires the complete requested set before returning a source lock.
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    index = index or StaticPackIndex()
    fetch = http_registry_fetcher(registry.endpoint, token=registry.token(), index=index)
    entries: list[LockedPack] = []
    unavailable: dict[str, str] = {}
    for pack, minimum in minimums.items():
        try:
            selected = resolve_compatible_release(
                registry, pack, minimum, engine_version, index=index
            )
            cached = pack_archive_path(artifacts, selected.artifact_digest)
            if cached.exists():
                entry = registry_source_lock(selected, cached, engine_version=engine_version)
            else:
                with tempfile.TemporaryDirectory(dir=artifacts, prefix="source-") as directory:
                    staged = Path(directory) / "pack.zip"
                    staged.write_bytes(fetch(selected))
                    entry = registry_source_lock(selected, staged, engine_version=engine_version)
                    cached = pack_archive_path(
                        artifacts,
                        selected.artifact_digest,
                        label=pack_label(selected.pack, selected.version),
                    )
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    staged.replace(cached)
            entries.append(entry)
        except (InstallError, OSError) as exc:
            unavailable[pack] = f"{pack}: {exc}"
    if unavailable and not allow_missing:
        raise InstallError("registry pack preparation refused: " + "; ".join(unavailable.values()))
    return Lockfile.of(entries), unavailable


def prepare_registry_update_sources(
    config: RegistryConfig,
    installed: Lockfile,
    artifacts: Path,
    *,
    engine_version: str,
    selected: NamedRegistry | None = None,
    index: StaticPackIndex | None = None,
) -> Lockfile:
    """Re-resolve installed official registry packs without adding or dropping packs.

    Other selections remain unchanged for generation admission. Official registry
    selections cannot downgrade or switch their recorded source on engine update.
    """
    index = index or StaticPackIndex()
    entries: list[LockedPack] = []
    for entry in installed.packs:
        if entry.publisher != "dinkster" or (
            entry.source != "registry" and not entry.source.startswith("registry:")
        ):
            entries.append(entry)
            continue
        try:
            registry = config.registry_for(entry, selected)
            lock, _ = prepare_registry_sources(
                registry,
                {entry.pack: entry.version},
                artifacts,
                engine_version=engine_version,
                index=index,
            )
            entries.extend(
                replace(updated, source=entry.source) if entry.source != "registry" else updated
                for updated in lock.packs
            )
        except InstallError as exc:
            raise InstallError(
                f"updating installed {entry.pack}@{entry.version} "
                f"for engine {engine_version} refused: {exc}"
            ) from exc
    return Lockfile.of(entries)


def read_registry_pack_locks(
    config: RegistryConfig,
    sources: Lockfile,
    *,
    variant: str,
    base_id: str,
    selected: NamedRegistry | None = None,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
    index: StaticPackIndex | None = None,
    errors: list[str] | None = None,
) -> tuple[PackLock, ...]:
    """Inspect every binding before staging; never resolve dependency ranges."""
    index = index or StaticPackIndex()
    if (
        re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", base_id) is None
    ):
        raise InstallError("published locks require a variant and exact sha256 base identity")
    locks: list[PackLock] = []
    failures: list[str] = []
    for entry in sources.packs:
        subject = f"{entry.pack}@{entry.version} dependency lock for {variant}"
        try:
            if entry.source != "registry" and not entry.source.startswith("registry:"):
                raise InstallError("published lock requires registry provenance")
            registry = config.registry_for(entry, selected)
            fields = static_pack_release(
                registry, entry.pack, entry.version, timeout=timeout, index=index
            )
            wheel, wheel_digest = static_release_wheel(registry, fields)
            bindings = fields.get("installLocks")
            binding = (
                cast("dict[str, object]", bindings).get(variant)
                if isinstance(bindings, dict)
                else None
            )
            if wheel_digest != entry.artifact_digest or not isinstance(binding, dict):
                raise InstallError("selected release or published binding differs")
            facts = cast("dict[str, object]", binding)
            digest, size = facts.get("digest"), facts.get("size")
            lock_path = "locks/" + str(digest).replace(":", "/") + ".json"
            if (
                facts.get("baseId") != base_id
                or facts.get("artifactDigest") != entry.artifact_digest
                or facts.get("url") != lock_path
                or not isinstance(digest, str)
                or validate_artifact_digest(digest)
                or type(size) is not int
                or not 1 <= cast("int", size) <= MEBIBYTE
            ):
                raise InstallError("invalid archive/base/lock binding")
            payload = index.lock_bytes(
                registry, "/static/" + lock_path, digest, cast("int", size), timeout=timeout
            )
            lock = PackLock.from_json(payload)
            if lock.pack != entry.pack or lock.variant != variant:
                raise InstallError("dependency lock pack or variant differs")
            if any(package.name == entry.pack for package in lock.packages):
                raise InstallError("dependency lock repeats the outer pack wheel")
            locks.append(
                replace(
                    lock,
                    indexes=(registry.endpoint + f"/static/{variant}/simple/",),
                    packages=(
                        *lock.packages,
                        LockedDistribution(entry.pack, entry.version, (wheel,)),
                    ),
                )
            )
        except (
            InstallError,
            OSError,
            ValueError,
            ProvisionError,
            http.client.HTTPException,
        ) as exc:
            failures.append(f"{subject}: {exc}")
    if errors is not None:
        errors.extend(failures)
    elif failures:
        raise InstallError("published dependency locks refused:\n" + "\n".join(failures))
    return tuple(locks)


def plan_registry_sources(
    config: RegistryConfig,
    registry: NamedRegistry,
    minimums: Mapping[str, str],
    *,
    engine_version: str,
    variant: str,
    base_id: str,
    index: StaticPackIndex,
) -> tuple[Lockfile, tuple[PackLock, ...]]:
    """Select sources and bound locks from metadata without acquiring pack archives."""
    entries: list[LockedPack] = []
    errors: list[str] = []
    for pack, minimum in minimums.items():
        try:
            entries.append(
                resolve_compatible_release(registry, pack, minimum, engine_version, index=index)
            )
        except (InstallError, OSError, ValueError, http.client.HTTPException) as exc:
            errors.append(f"{pack}: {exc}")
    sources = Lockfile.of(entries)
    locks: tuple[PackLock, ...] = ()
    try:
        locks = read_registry_pack_locks(
            config, sources, variant=variant, base_id=base_id, selected=registry, index=index
        )
    except InstallError as exc:
        errors.append(str(exc))
    if errors:
        raise InstallError("registry plan refused:\n" + "\n".join(errors))
    return sources, locks


def plan_registry_changes(
    config: RegistryConfig,
    installed: Lockfile,
    *,
    selections: tuple[str, ...] = (),
    updates: tuple[str, ...] | None = None,
    engine_version: str,
    variant: str,
    base_id: str,
    selected: NamedRegistry | None = None,
    index: StaticPackIndex,
    errors: list[str] | None = None,
) -> tuple[Lockfile, tuple[PackLock, ...]]:
    """Inspect published replacements, retaining every unselected source verbatim."""
    if selections and updates is not None:
        raise InstallError("exact selections and compatible updates are mutually exclusive")
    current = {entry.pack: entry for entry in installed.packs}
    requested: dict[str, str | None] = {}
    if updates is not None:
        for name in updates or tuple(
            entry.pack
            for entry in installed.packs
            if entry.source == "registry" or entry.source.startswith("registry:")
        ):
            if name != canonical_name(name) or name in requested:
                raise InstallError("updates require distinct canonical installed pack names")
            entry = current.get(name)
            if entry is None or not (
                entry.source == "registry" or entry.source.startswith("registry:")
            ):
                raise InstallError(f"{name}: update requires an installed registry pack")
            requested[name] = None
    else:
        for raw in selections:
            parsed = parse_registry_spec(raw)
            if parsed is None or raw != f"{parsed[0]}@{parsed[1]}":
                raise InstallError("installation requires canonical PACK@VERSION selections")
            name, version = parsed
            if name in requested:
                raise InstallError(f"{name}: selections require a distinct pack")
            requested[name] = version
        if not requested:
            raise InstallError("installation requires at least one exact selection")
    replacements: list[LockedPack] = []
    failures: list[str] = []
    for name, version in requested.items():
        try:
            previous = current.get(name)
            registry = (
                config.registry_for(previous, selected)
                if previous is not None
                and (previous.source == "registry" or previous.source.startswith("registry:"))
                else selected
                or select_registry(
                    config, "" if config.registries else "https://registry.dinkster.org"
                )
            )
            if selected is not None and registry.endpoint != selected.endpoint:
                raise InstallError("selection cannot change the installed registry")
            if version is None:
                assert previous is not None
                entry = resolve_compatible_release(
                    registry, name, previous.version, engine_version, index=index
                )
            else:
                entry = resolve_release(
                    registry, name, version, engine_version=engine_version, index=index
                )
            if previous is not None and previous.source.startswith("registry:"):
                entry = replace(entry, source=previous.source)
            replacements.append(entry)
        except (InstallError, OSError, ValueError, http.client.HTTPException) as exc:
            failures.append(f"{name}: {exc}")
    sources = Lockfile.of(replacements)
    locks = read_registry_pack_locks(
        config,
        sources,
        variant=variant,
        base_id=base_id,
        selected=selected,
        index=index,
        errors=failures,
    )
    if errors is not None:
        errors.extend(failures)
    elif failures:
        raise InstallError("registry pack changes refused:\n" + "\n".join(failures))
    return (
        Lockfile.of(
            [entry for entry in installed.packs if entry.pack not in requested] + replacements
        ),
        locks,
    )


def prepare_registry_pack_locks(
    config: RegistryConfig,
    sources: Lockfile,
    output: Path,
    *,
    variant: str,
    base_id: str,
    selected: NamedRegistry | None = None,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
    index: StaticPackIndex | None = None,
) -> tuple[Path, ...]:
    """Stage the complete verified lock set only after metadata inspection succeeds."""
    if output.exists():
        raise InstallError("published lock output directory already exists")
    locks = read_registry_pack_locks(
        config,
        sources,
        variant=variant,
        base_id=base_id,
        selected=selected,
        timeout=timeout,
        index=index,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=output.parent, prefix="published-locks-") as directory:
            staged = Path(directory)
            names: list[str] = []
            for lock in locks:
                name = lock.pack + ".pack-lock.json"
                lock.write(staged / name)
                names.append(name)
            staged.rename(output)
        return tuple(output / name for name in names)
    except (OSError, ValueError, ProvisionError) as exc:
        raise InstallError(f"published dependency lock preparation failed: {exc}") from exc


@dataclass(frozen=True)
class PackSummary:
    """One row of the pack index: enough to pick a pack to install."""

    pack: str
    publisher: str
    latest_version: str


@dataclass(frozen=True)
class PackPage:
    """One page of the pack index. ``cursor`` is empty when the listing
    is complete; otherwise it resumes the SAME query (keyset contract)."""

    packs: tuple[PackSummary, ...]
    cursor: str


@dataclass(frozen=True)
class TemplateSummary:
    """One row of the remote template catalog (latest release per pack)."""

    pack: str
    version: str
    id: str
    name: str
    description: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class TemplatePage:
    templates: tuple[TemplateSummary, ...]
    cursor: str


def _page_cursor(
    fields: dict[str, object], registry: NamedRegistry, subject: str, *, key: str = "cursor"
) -> str:
    cursor = fields.get(key) or ""
    if not isinstance(cursor, str):
        raise InstallError(f"registry {registry.label} answered {subject} with a malformed cursor")
    return cursor


def browse_packs(
    registry: NamedRegistry,
    *,
    query: str = "",
    limit: int = 50,
    cursor: str = "",
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> PackPage:
    """One page of ONE registry's pack index (``GET /index/packs``).
    Browse is discovery only: what it lists still installs through
    :func:`resolve_release` + digest-verified acquisition, so a lying
    index can advertise, never substitute."""
    params = [("q", query)] if query else []
    params.append(("limit", str(limit)))
    if cursor:
        params.append(("cursor", cursor))
    fields = _index_get(
        registry,
        "/v1/packs?" + urllib.parse.urlencode(params),
        action="browsing the pack index",
        subject="the pack index",
        timeout=timeout,
    )
    rows = fields.get("items")
    if not isinstance(rows, list):
        raise InstallError(
            f"registry {registry.label} answered the pack index without an items list"
        )
    packs: list[PackSummary] = []
    for row in cast("list[object]", rows):
        if not isinstance(row, dict):
            raise InstallError(
                f"registry {registry.label} answered the pack index with a non-object row"
            )
        entry = cast("dict[str, object]", row)
        pack = entry.get("name")
        owner = entry.get("owner")
        publisher = (
            cast("dict[str, object]", owner).get("slug") if isinstance(owner, dict) else None
        )
        latest = entry.get("latestVersion")
        if (
            not isinstance(pack, str)
            or not isinstance(publisher, str)
            or not isinstance(latest, str)
        ):
            raise InstallError(
                f"registry {registry.label} answered the pack index with a malformed "
                f"row (need name/latestVersion strings and an owner slug)"
            )
        packs.append(PackSummary(pack=pack, publisher=publisher, latest_version=latest))
    return PackPage(
        packs=tuple(packs),
        cursor=_page_cursor(fields, registry, "the pack index", key="nextCursor"),
    )


def browse_templates(
    registry: NamedRegistry,
    *,
    query: str = "",
    tag: str = "",
    pack: str = "",
    limit: int = 50,
    cursor: str = "",
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> TemplatePage:
    """One page of ONE registry's template catalog
    (``GET /index/templates``): each pack's latest release's starter
    workflows, browsable before anything is installed. Bodies never ride
    the listing - they hang off the release's digest-pinned template
    endpoint."""
    params = [("q", query)] if query else []
    if tag:
        params.append(("tag", tag))
    if pack:
        params.append(("pack", pack))
    params.append(("limit", str(limit)))
    if cursor:
        params.append(("cursor", cursor))
    fields = _index_get(
        registry,
        "/index/templates?" + urllib.parse.urlencode(params),
        action="browsing the template catalog",
        subject="the template catalog",
        timeout=timeout,
    )
    rows = fields.get("templates")
    if not isinstance(rows, list):
        raise InstallError(
            f"registry {registry.label} answered the template catalog without a templates list"
        )
    templates: list[TemplateSummary] = []
    for row in cast("list[object]", rows):
        if not isinstance(row, dict):
            raise InstallError(
                f"registry {registry.label} answered the template catalog with a non-object row"
            )
        entry = cast("dict[str, object]", row)
        owner = entry.get("pack")
        version = entry.get("version")
        template_id = entry.get("id")
        name = entry.get("name")
        description = entry.get("description", "")
        tags = entry.get("tags", [])
        if (
            not isinstance(owner, str)
            or not isinstance(version, str)
            or not isinstance(template_id, str)
            or not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(tags, list)
            or not all(isinstance(item, str) for item in cast("list[object]", tags))
        ):
            raise InstallError(
                f"registry {registry.label} answered the template catalog with a "
                f"malformed row (need pack/version/id/name strings)"
            )
        templates.append(
            TemplateSummary(
                pack=owner,
                version=version,
                id=template_id,
                name=name,
                description=description,
                tags=tuple(cast("list[str]", tags)),
            )
        )
    return TemplatePage(
        templates=tuple(templates),
        cursor=_page_cursor(fields, registry, "the template catalog"),
    )


def parse_registry_spec(raw: str) -> tuple[str, str] | None:
    """``pack@version`` when ``raw`` reads as one, else None (the caller
    falls through to path handling). Both halves must parse - a valid
    name left of the LAST ``@``, a strict major.minor.patch right of it
    - so paths and git URLs never trip this. A local directory literally
    named like a spec can be forced with a ``./`` prefix."""
    if "@" not in raw or "/" in raw or "\\" in raw:
        return None
    name, _, version = raw.rpartition("@")
    if not name or validate_version(version) is not None:
        return None
    if validate_name(name) is not None:
        return None
    return canonical_name(name), version


__all__ = [
    "AD_HOC_TOKEN_ENV",
    "DEFAULT_REGISTRIES_FILENAME",
    "DEFAULT_RESOLVE_TIMEOUT",
    "NamedRegistry",
    "PackPage",
    "PackSummary",
    "PublishFinding",
    "PublishVerdict",
    "RegistryConfig",
    "RegistryPublishError",
    "TemplatePage",
    "TemplateSummary",
    "browse_packs",
    "browse_templates",
    "default_pack_minimums",
    "load_registries",
    "parse_registry_spec",
    "prepare_registry_sources",
    "publish_release",
    "registry_source_lock",
    "resolve_compatible_release",
    "resolve_release",
    "routing_registry_fetcher",
    "select_registry",
]
