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
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
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
    unpack_artifact,
    validate_artifact_digest,
    validate_version,
)
from dinkster_schema import PackCompatibility, canonical_claim, canonical_name, validate_name
from dinkster_workers import ManifestError, load_manifest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from dinkster.installer import RegistryFetcher, http_registry_fetcher

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
    timeout: float = DEFAULT_RESOLVE_TIMEOUT,
) -> PublishVerdict:
    """Submit one canonical artifact to the service's admission queue."""
    headers = {"User-Agent": "dinkster-pack"}
    token = registry.token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    submit = urllib.request.Request(
        f"{registry.endpoint}/v1/publish/{pack_name}/versions/{version}",
        data=archive.read_bytes(),
        headers={**headers, "Content-Type": "application/zip"},
        method="POST",
    )
    payload = _publish_request(registry, submit, action="publish submission", timeout=timeout)
    try:
        decoded: object = json.loads(payload)
        document = cast("dict[str, object]", decoded)
        state = document["status"]
        candidate_id = document["candidateId"]
        if not isinstance(state, str) or state != "queued" or not isinstance(candidate_id, str):
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
    engine_version: str | None = None,
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
        f"/v1/packs/{wanted_pack}/versions/{version}",
        action=f"resolving {wanted_pack}@{version}",
        subject=f"{wanted_pack}@{version}",
        missing=f"registry {registry.label} publishes no release {wanted_pack}@{version}",
        timeout=timeout,
    )
    answered_pack = fields.get("packName")
    answered_version = fields.get("version")
    artifact = fields.get("artifact")
    published_by = fields.get("publishedBy")
    digest = (
        cast("dict[str, object]", artifact).get("digest") if isinstance(artifact, dict) else None
    )
    publisher = (
        cast("dict[str, object]", published_by).get("principalId")
        if isinstance(published_by, dict)
        else None
    )
    if (
        not isinstance(answered_pack, str)
        or not isinstance(answered_version, str)
        or not isinstance(digest, str)
        or not isinstance(publisher, str)
    ):
        raise InstallError(
            f"registry {registry.label} answered {wanted_pack}@{version} with a "
            f"malformed release record (need packName/version, artifact digest, "
            f"and publishing principal strings)"
        )
    if answered_pack != wanted_pack or answered_version != version:
        raise InstallError(
            f"registry {registry.label} answered the request for "
            f"{wanted_pack}@{version} with {answered_pack}@{answered_version}; "
            f"refusing the substitution"
        )
    if engine_version is not None:
        manifest = fields.get("manifest")
        engine = (
            cast("dict[str, object]", manifest).get("dinkster")
            if isinstance(manifest, dict)
            else None
        )
        if fields.get("state") != "active" or not isinstance(engine, str):
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
        publisher=canonical_name(publisher),
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
) -> LockedPack:
    """Select the newest active compatible version from ONE registry, then pin its bytes.

    Inspect every page: a pack's latest release need not support the target engine.
    Recheck the selected release detail before trusting its installation identity.
    """
    wanted_pack = canonical_name(pack)
    minimum_version = Version(minimum)
    target_engine = Version(engine_version)
    selected: Version | None = None
    cursor = ""
    seen: set[str] = set()
    subject = f"versions of {wanted_pack}"
    while True:
        params = [("limit", "100")]
        if cursor:
            params.append(("cursor", cursor))
        fields = _index_get(
            registry,
            f"/v1/packs/{wanted_pack}/versions?" + urllib.parse.urlencode(params),
            action=f"resolving compatible {wanted_pack}",
            subject=subject,
            timeout=timeout,
        )
        rows = fields.get("items")
        if not isinstance(rows, list):
            raise InstallError(
                f"registry {registry.label} answered {subject} without an items list"
            )
        for row in cast("list[object]", rows):
            if not isinstance(row, dict):
                raise InstallError(
                    f"registry {registry.label} answered {subject} with a malformed row"
                )
            entry = cast("dict[str, object]", row)
            version, engine, state = entry.get("version"), entry.get("dinkster"), entry.get("state")
            if (
                not isinstance(version, str)
                or not isinstance(engine, str)
                or not isinstance(state, str)
            ):
                raise InstallError(
                    f"registry {registry.label} answered {subject} "
                    "without version/engine/state strings"
                )
            try:
                compatibility = PackCompatibility(version, engine)
            except ValueError as exc:
                raise InstallError(f"registry {registry.label} answered {subject}: {exc}") from exc
            candidate = Version(compatibility.version)
            if (
                state == "active"
                and candidate >= minimum_version
                and (selected is None or candidate > selected)
                and SpecifierSet(compatibility.engine).contains(target_engine, prereleases=True)
            ):
                selected = candidate
        following = fields.get("nextCursor")
        if following is not None and not isinstance(following, str):
            raise InstallError(
                f"registry {registry.label} answered {subject} with a malformed cursor"
            )
        cursor = following or ""
        if not cursor:
            break
        if cursor in seen:
            raise InstallError(f"registry {registry.label} repeated the cursor for {subject}")
        seen.add(cursor)
    if selected is None:
        raise InstallError(
            f"registry {registry.label} has no active {wanted_pack} >= {minimum} "
            f"compatible with engine {engine_version}"
        )
    return resolve_release(
        registry, wanted_pack, str(selected), engine_version=engine_version, timeout=timeout
    )


def registry_source_lock(selected: LockedPack, archive: Path, *, engine_version: str) -> LockedPack:
    """Bind the selected release to its safely unpacked, digest-verified manifest.

    Publisher and namespace claims describe the selected content, not the
    publishing audit actor. They are not proof of authenticated authority:
    reserved-name/grant admission belongs to the trusted registry's promotion.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="dinkster-registry-source-") as directory:
            root = unpack_artifact(archive, Path(directory), selected.artifact_digest)
            manifest = load_manifest(root / "dinkster-pack.toml")
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
                publisher=canonical_name(manifest.publisher),
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
) -> tuple[Lockfile, dict[str, str]]:
    """Resolve target-compatible sources without mutating any generation.

    Fresh setup may retain available defaults and report unavailable packs.
    Update requires the complete requested set before returning a source lock.
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    fetch = http_registry_fetcher(registry.endpoint, token=registry.token())
    entries: list[LockedPack] = []
    unavailable: dict[str, str] = {}
    for pack, minimum in minimums.items():
        try:
            selected = resolve_compatible_release(registry, pack, minimum, engine_version)
            cached = artifacts / f"{selected.artifact_digest.removeprefix('blake3:')}.zip"
            if cached.exists():
                entry = registry_source_lock(selected, cached, engine_version=engine_version)
            else:
                with tempfile.TemporaryDirectory(dir=artifacts, prefix="source-") as directory:
                    staged = Path(directory) / "pack.zip"
                    staged.write_bytes(fetch(selected))
                    entry = registry_source_lock(selected, staged, engine_version=engine_version)
                    staged.replace(cached)
            entries.append(entry)
        except (InstallError, OSError) as exc:
            unavailable[pack] = f"{pack}: {exc}"
    if unavailable and not allow_missing:
        raise InstallError("registry pack preparation refused: " + "; ".join(unavailable.values()))
    return Lockfile.of(entries), unavailable


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
