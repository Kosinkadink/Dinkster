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

import json
import os
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_registry import (
    InstallError,
    LockedPack,
    validate_artifact_digest,
    validate_version,
)
from dinkster_schema import canonical_name, validate_name

from dinkster.installer import RegistryFetcher, http_registry_fetcher

DEFAULT_REGISTRIES_FILENAME = "registries.toml"
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
            message = cast("dict[str, object]", body).get("error")
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
    digest: str,
    version: str,
    *,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> PublishVerdict:
    """Upload one canonical artifact and submit it for server-side admission."""
    headers = {"User-Agent": "dinkster-pack"}
    token = registry.token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    digest_hex = digest.partition(":")[2]
    upload = urllib.request.Request(
        f"{registry.endpoint}/artifacts/{digest_hex}.zip",
        data=archive.read_bytes(),
        headers={**headers, "Content-Type": "application/zip"},
        method="PUT",
    )
    _publish_request(registry, upload, action="artifact upload", timeout=timeout)
    body = json.dumps({"version": version, "artifactDigest": digest}).encode()
    submit = urllib.request.Request(
        f"{registry.endpoint}/publish",
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    payload = _publish_request(registry, submit, action="publish admission", timeout=timeout)
    try:
        decoded: object = json.loads(payload)
        document = cast("dict[str, object]", decoded)
        state = document["state"]
        raw_findings = document["findings"]
        if state not in ("accepted", "needs_review", "rejected") or not isinstance(
            raw_findings, list
        ):
            raise TypeError
        findings: list[PublishFinding] = []
        for raw in cast("list[object]", raw_findings):
            if not isinstance(raw, dict):
                raise TypeError
            finding = cast("dict[str, object]", raw)
            code, message = finding.get("code"), finding.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise TypeError
            findings.append(PublishFinding(code=code, message=message))
    except (ValueError, KeyError, TypeError) as exc:
        raise RegistryPublishError(
            f"registry {registry.label} answered publish admission with a malformed verdict"
        ) from exc
    return PublishVerdict(state=state, findings=tuple(findings))


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
    config: RegistryConfig, selected: NamedRegistry | None
) -> RegistryFetcher:
    """One fetcher for every registry-sourced entry, dispatching on the
    entry's recorded provenance. ``selected`` is the --registry choice
    (already resolved through :func:`select_registry`); bare
    ``"registry"`` sources go there, falling back to the configured
    default."""

    def registry_for(entry: LockedPack) -> NamedRegistry:
        source = entry.source
        if source == "registry":
            chosen = selected if selected is not None else config.default_registry()
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
            for registry in config.registries:
                if registry.endpoint == endpoint:
                    return registry  # the matching credential rides along
            return NamedRegistry(name="", endpoint=endpoint)
        named = config.get(qualifier)
        if named is None:
            configured = ", ".join(config.names) if config.names else "none configured"
            raise InstallError(
                f"cannot acquire {entry.pack}: recorded source {source!r} names "
                f"registry {qualifier!r}, which is not configured "
                f"(configured: {configured}); add it to registries.toml"
            )
        return named

    def fetch(entry: LockedPack) -> bytes:
        registry = registry_for(entry)
        return http_registry_fetcher(registry.endpoint, token=registry.token())(entry)

    return fetch


def _index_get(
    registry: NamedRegistry,
    path: str,
    *,
    action: str,
    subject: str,
    missing: str = "",
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> dict[str, object]:
    """One GET against a registry index path, answered as a JSON object
    or refused as an :class:`InstallError` that names the registry and
    what was being asked. ``missing`` is the message a 404 earns when the
    caller knows what absence means; without it a 404 is just another
    failed HTTP status."""
    url = f"{registry.endpoint}{path}"
    headers = {"User-Agent": "dinkster-pack"}
    token = registry.token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - scheme checked at construction
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = response.read()
            warning = response.headers.get("Warning", "")
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and missing:
            raise InstallError(missing) from exc
        raise InstallError(f"{action} against {registry.label} failed: HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
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


def resolve_release(
    registry: NamedRegistry,
    pack: str,
    version: str,
    *,
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> LockedPack:
    """Turn ``pack@version`` into a digest-pinned :class:`LockedPack`
    by asking ONE registry's index for the exact release - never a
    nearest version, never another registry. The answer is
    cross-checked against the request (a lying index refuses, it does
    not redirect) and only NAMES the digest; acquisition separately
    verifies the actual bytes against it."""
    wanted_pack = canonical_name(pack)
    fields = _index_get(
        registry,
        f"/index/packs/{wanted_pack}/versions/{version}",
        action=f"resolving {wanted_pack}@{version}",
        subject=f"{wanted_pack}@{version}",
        missing=f"registry {registry.label} publishes no release {wanted_pack}@{version}",
        timeout=timeout,
    )
    answered_pack = fields.get("pack")
    answered_version = fields.get("version")
    digest = fields.get("artifactDigest")
    publisher = fields.get("publisher")
    claims = fields.get("claims")
    if (
        not isinstance(answered_pack, str)
        or not isinstance(answered_version, str)
        or not isinstance(digest, str)
        or not isinstance(publisher, str)
        or not isinstance(claims, list)
        or not all(isinstance(claim, str) for claim in cast("list[object]", claims))
    ):
        raise InstallError(
            f"registry {registry.label} answered {wanted_pack}@{version} with a "
            f"malformed release record (need pack/version/artifactDigest/"
            f"publisher strings and a claims list)"
        )
    if answered_pack != wanted_pack or answered_version != version:
        raise InstallError(
            f"registry {registry.label} answered the request for "
            f"{wanted_pack}@{version} with {answered_pack}@{answered_version}; "
            f"refusing the substitution"
        )
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
        publisher=canonical_name(publisher),
        claims=tuple(canonical_name(claim) for claim in cast("list[str]", claims)),
        source=registry.source,
    )


@dataclass(frozen=True)
class PackSummary:
    """One row of the pack index: enough to pick a pack to install."""

    pack: str
    publisher: str
    latest_version: str
    versions: int


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


def _page_cursor(fields: dict[str, object], registry: NamedRegistry, subject: str) -> str:
    cursor = fields.get("cursor", "")
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
        "/index/packs?" + urllib.parse.urlencode(params),
        action="browsing the pack index",
        subject="the pack index",
        timeout=timeout,
    )
    rows = fields.get("packs")
    if not isinstance(rows, list):
        raise InstallError(
            f"registry {registry.label} answered the pack index without a packs list"
        )
    packs: list[PackSummary] = []
    for row in cast("list[object]", rows):
        if not isinstance(row, dict):
            raise InstallError(
                f"registry {registry.label} answered the pack index with a non-object row"
            )
        entry = cast("dict[str, object]", row)
        pack = entry.get("pack")
        publisher = entry.get("publisher")
        latest = entry.get("latestVersion")
        versions = entry.get("versions")
        if (
            not isinstance(pack, str)
            or not isinstance(publisher, str)
            or not isinstance(latest, str)
            or not isinstance(versions, int)
        ):
            raise InstallError(
                f"registry {registry.label} answered the pack index with a malformed "
                f"row (need pack/publisher/latestVersion strings and a versions count)"
            )
        packs.append(
            PackSummary(pack=pack, publisher=publisher, latest_version=latest, versions=versions)
        )
    return PackPage(
        packs=tuple(packs),
        cursor=_page_cursor(fields, registry, "the pack index"),
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
    "load_registries",
    "parse_registry_spec",
    "publish_release",
    "resolve_release",
    "routing_registry_fetcher",
    "select_registry",
]
