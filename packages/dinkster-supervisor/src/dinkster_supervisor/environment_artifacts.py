"""Discover exact release manifests and populate the verified local wheel store."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from platform import machine
from typing import cast
from urllib.parse import quote, unquote, urljoin, urlparse

from blake3 import blake3
from dinkster_store import object_path, pack_archive_path, pack_label, readable_label
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .environment_archives import extract_base
from .environment_channel_signatures import verify_channel
from .environment_constraints import effective_constraints, validate_base_requirements
from .environment_downloads import fetch_artifact, read_metadata
from .environment_frontend import (
    frontend_filename,
    frontend_requirements,
    select_frontend,
    verify_frontend_wheel,
)
from .environment_state import (
    ArtifactLocation,
    Channel,
    EnvironmentError,
    atomic_write,
    content_digest,
    digest,
    relative_file,
)
from .limits import ENVIRONMENT_STREAM_CHUNK_BYTES

_BASE_IDENTITY = (
    "format",
    "variant",
    "platform",
    "python",
    "interpreter",
    "wheels",
    "replaces",
    "workerProtocol",
    "nvidia",
)


def parse_metadata(data: bytes) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EnvironmentError(f"duplicate manifest field: {key}")
            result[key] = value
        return result

    try:
        value: object = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise EnvironmentError("invalid artifact metadata JSON") from exc
    if not isinstance(value, dict):
        raise EnvironmentError("artifact metadata must be an object")
    return cast("dict[str, object]", value)


def artifact_rows(
    value: object, key: str, *, allow_identical: bool = False
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise EnvironmentError("manifest artifacts must be an array")
    rows: list[dict[str, object]] = []
    names: dict[str, dict[str, object]] = {}
    for item in cast("list[object]", value):
        if not isinstance(item, dict):
            raise EnvironmentError("manifest artifact must be an object")
        row = cast("dict[str, object]", item)
        name = relative_file(row.get(key))
        if "/" in name or name.endswith((".", " ")):
            raise EnvironmentError("artifact names must be distinct portable basenames")
        sha, size = row.get("sha256"), row.get("size")
        if not isinstance(sha, str) or type(size) is not int or size < 0:
            raise EnvironmentError("artifact must carry a sha256 digest and nonnegative size")
        digest("sha256:" + sha)
        previous = names.get(name.casefold())
        if previous is not None:
            if allow_identical and all(
                previous[field] == row[field] for field in (key, "sha256", "size")
            ):
                continue
            raise EnvironmentError("artifact names must be distinct portable basenames")
        names[name.casefold()] = row
        rows.append(row)
    return tuple(rows)


def supports_base_protocol(engine: dict[str, object], base: dict[str, object]) -> bool:
    """Base dependency compatibility does not change the engine's worker wire protocol."""
    protocols = engine.get("baseWorkerProtocols", [engine.get("workerProtocol")])
    if (
        not isinstance(protocols, list)
        or not protocols
        or any(type(item) is not int or item < 1 for item in cast("list[object]", protocols))
        or engine.get("workerProtocol") not in protocols
    ):
        raise EnvironmentError("engine baseWorkerProtocols must include its workerProtocol")
    return base.get("workerProtocol") in protocols


def validate_manifest(
    data: bytes, expected: str, variant: str | None, *, allow_source: bool = False
) -> dict[str, object]:
    digest(expected)
    document = parse_metadata(data)
    protocol = document.get("workerProtocol")
    if type(protocol) is not int or protocol < 1:
        raise EnvironmentError("manifest workerProtocol must be a positive integer")
    if variant is None:
        supports_base_protocol(document, document)
        if document.get("format") != "dinkster.engine/1":
            raise EnvironmentError("unsupported engine manifest format")
        capabilities = document.get("capabilities", [])
        if not isinstance(capabilities, list) or any(
            not isinstance(value, str) for value in cast("list[object]", capabilities)
        ):
            raise EnvironmentError("engine capabilities must be a list of strings")
        library_api = document.get("libraryApi")
        if library_api is not None and (type(library_api) is not int or library_api < 1):
            raise EnvironmentError("engine libraryApi must be a positive integer")
        library_capabilities = document.get("libraryCapabilities", [])
        if not isinstance(library_capabilities, list) or any(
            not isinstance(value, str) for value in cast("list[object]", library_capabilities)
        ):
            raise EnvironmentError("engine libraryCapabilities must be a list of strings")
        if library_capabilities and library_api is None:
            raise EnvironmentError("engine libraryCapabilities requires libraryApi")
        if allow_source and document.get("source") == "git":
            from .environment_development import source_identity

            if source_identity(document) != expected:
                raise EnvironmentError("source engine manifest identity mismatch")
            if (
                document.get("kind") != "source"
                or document.get("published") is not False
                or document.get("reproducible") is not False
                or not isinstance(document.get("version"), str)
                or not document["version"]
            ):
                raise EnvironmentError("source engine requires explicit unpublished identity")
            artifact_rows(document.get("wheels"), "filename")
            select_frontend(document)
            return document
        if "sha256:" + hashlib.sha256(data).hexdigest() != expected:
            raise EnvironmentError("engine manifest digest mismatch")
        if not isinstance(document.get("version"), str) or not document["version"]:
            raise EnvironmentError("engine manifest version is missing")
        commit = document.get("commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise EnvironmentError("engine manifest must name its full commit")
        select_frontend(document)
        rows = artifact_rows(document.get("artifacts"), "name")
        for row in rows:
            if "url" in row:
                url = row["url"]
                if not isinstance(url, str):
                    raise EnvironmentError("engine artifact URL must be a string")
                parsed = urlparse(url)
                if (
                    not str(row["name"]).endswith(".whl")
                    or parsed.scheme != "https"
                    or parsed.netloc != "files.pythonhosted.org"
                    or unquote(parsed.path.rsplit("/", 1)[-1]) != row["name"]
                    or parsed.query
                    or parsed.fragment
                ):
                    raise EnvironmentError("engine wheel URL must name its exact PyPI file")
        if not any(row["name"] == "constraints.txt" for row in rows):
            raise EnvironmentError("engine release has no constraints.txt artifact")
        if "defaultPacks" in document:
            defaults = document["defaultPacks"]
            if not isinstance(defaults, dict) or not defaults:
                raise EnvironmentError("engine defaultPacks must be a nonempty object")
            for pack, minimum in cast("dict[str, object]", defaults).items():
                if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", pack) is None:
                    raise EnvironmentError("engine default pack names must be canonical")
                try:
                    if not isinstance(minimum, str):
                        raise ValueError("minimum version must be a string")
                    Version(minimum)
                except ValueError as exc:
                    raise EnvironmentError(f"invalid engine default pack minimum: {pack}") from exc
        if "dependencies" in document:
            dependencies = document["dependencies"]
            if not isinstance(dependencies, dict) or not dependencies:
                raise EnvironmentError("engine dependency selections must be a nonempty object")
            for selection in cast("dict[str, object]", dependencies).values():
                dependency_wheels(selection)
    else:
        if document.get("format") != "dinkster.base/1" or document.get("variant") != variant:
            raise EnvironmentError("base manifest format or variant mismatch")
        if document.get("signature") is not None:
            raise EnvironmentError("signed base verification is not supported")
        try:
            payload = {key: document[key] for key in _BASE_IDENTITY}
        except KeyError as exc:
            raise EnvironmentError("base manifest identity fields are missing") from exc
        for key in ("amd", "intel", "displayName"):
            if key in document:
                payload[key] = document[key]
        requirements = [document.get(vendor) for vendor in ("nvidia", "amd", "intel")]
        if sum(requirement is not None for requirement in requirements) > 1:
            raise EnvironmentError("a base must require at most one graphics vendor")
        for vendor in ("amd", "intel"):
            requirement = document.get(vendor)
            if requirement is None:
                continue
            if not isinstance(requirement, dict):
                raise EnvironmentError(f"invalid {vendor} base requirement")
            requirement = cast("dict[str, object]", requirement)
            minimum = requirement.get("minimumDriver")
            if (
                set(requirement) != {"minimumDriver"}
                or not isinstance(minimum, str)
                or re.fullmatch(r"\d+(?:\.\d+)*", minimum) is None
            ):
                raise EnvironmentError(f"invalid {vendor} minimum driver")
            label = document.get("displayName")
            if not isinstance(label, str) or not label.strip():
                raise EnvironmentError("graphics base requires a displayName")
        if "displayName" in document and (
            not isinstance(document["displayName"], str) or not document["displayName"].strip()
        ):
            raise EnvironmentError("base displayName must be nonempty text")
        raw_archive = document.get("archive")
        if not isinstance(raw_archive, dict):
            raise EnvironmentError("unsupported base archive format")
        archive = cast("dict[str, object]", raw_archive)
        if archive.get("format") != "tar+zstd":
            raise EnvironmentError("unsupported base archive format")
        payload["archive"] = {"format": "tar+zstd"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        actual = "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()
        if document.get("id") != expected or actual != expected:
            raise EnvironmentError("base manifest digest mismatch")
        artifact_rows(document.get("wheels"), "filename")
        parts = artifact_rows(archive.get("parts"), "filename")
        if not parts or any(
            row["filename"] != f"base.tar.zst.{index:03d}"
            or not 0 < cast("int", row["size"]) < 2_000_000_000
            for index, row in enumerate(parts, 1)
        ):
            raise EnvironmentError("base archive parts must be ordered and below 2 GB")
        # Validate the producer-owned replacement set and wheel identities before acquisition.
        effective_constraints("", document)
        platform = document.get("platform")
        if not isinstance(platform, dict):
            raise EnvironmentError("base manifest platform is invalid")
        platform = cast("dict[str, object]", platform)
        if platform.get("os") not in ("windows", "linux", "macos"):
            raise EnvironmentError("base manifest platform is invalid")
        if platform.get("arch") != "x86_64" and (platform["os"], platform.get("arch")) != (
            "macos",
            "arm64",
        ):
            raise EnvironmentError("unsupported base architecture")
        python = document.get("interpreter")
        if not isinstance(python, dict):
            raise EnvironmentError("base interpreter metadata is missing")
        python = cast("dict[str, object]", python)
        expected_python = (
            "python/python.exe" if platform["os"] == "windows" else "python/bin/python3"
        )
        if python.get("executable") != expected_python:
            raise EnvironmentError("base interpreter executable differs from the snapshot layout")
    return document


def trusted_metadata(mirror: str) -> None:
    """Restrict unsigned publication indexes, not signature-bound artifact locations."""
    parsed = urlparse(mirror)
    if not parsed.scheme or Path(mirror).is_absolute():
        return
    if parsed.hostname == "releases.dinkster.org":
        return
    if parsed.hostname == "localhost":
        return
    try:
        if parsed.hostname and ipaddress.ip_address(parsed.hostname).is_loopback:
            return
    except ValueError:
        pass
    raise EnvironmentError("unsigned metadata requires a Dinkster-operated mirror")


def _sources(
    locations: tuple[ArtifactLocation, ...], mirrors: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    sources: list[tuple[str, str]] = []
    for location in locations:
        if location.kind == "github":
            if location.repository.lower() not in {
                "kosinkadink/dinkster",
                "kosinkadink/dinkster-environments",
            }:
                raise EnvironmentError("unsigned GitHub metadata requires a Dinkster repository")
            root = f"https://github.com/{location.repository}/releases/download/" + quote(
                location.tag, safe=""
            )
            sources.append((root, location.asset))
        else:
            prefix, filename = location.path.rsplit("/", 1)
            for mirror in mirrors:
                if urlparse(mirror).scheme and not Path(mirror).is_absolute():
                    root = mirror.rstrip("/") + "/" + prefix
                else:
                    directory = Path(mirror).resolve()
                    candidate = (directory / prefix).resolve()
                    if not candidate.is_relative_to(directory):
                        raise EnvironmentError("manifest path escapes its configured mirror")
                    root = str(candidate)
                sources.append((root, filename))
    return tuple(sources)


def discover_manifest(
    locations: tuple[ArtifactLocation, ...],
    mirrors: tuple[str, ...],
    expected: str,
    variant: str | None = None,
    *,
    store: Path | None = None,
    offline: bool = False,
) -> tuple[dict[str, object], tuple[str, ...], bytes]:
    sources = _sources(locations, mirrors)
    if offline and store is None:
        raise EnvironmentError("offline discovery requires a local store")
    candidates = (
        (
            (str(store / "metadata"), f"{expected[7:]}.json"),
            *(() if offline else sources),
        )
        if store is not None
        else sources
    )
    for root, filename in candidates:
        try:
            data = read_metadata(root, filename)
            document = validate_manifest(data, expected, variant)
            return document, tuple(source for source, _ in sources), data
        except (OSError, ValueError, HTTPException):
            continue
    raise EnvironmentError("no location supplied an intact expected artifact manifest")


@dataclass
class EnvironmentCatalog:
    channel_name: str
    document: dict[str, object]
    channel: Channel
    engine: dict[str, object]
    engine_sources: tuple[str, ...]
    bases: dict[str, dict[str, object]]
    base_sources: dict[str, tuple[str, ...]]
    engine_bytes: bytes
    mirrors: tuple[str, ...] | None = None
    cached_channel: bool = False
    channel_envelope: bytes | None = None

    def variants(self) -> list[dict[str, object]]:
        return [
            {
                "variant": variant,
                "base": manifest["id"],
                "manifest": manifest,
                "downloadBytes": sum(
                    cast("int", row["size"])
                    for row in artifact_rows(
                        cast("dict[str, object]", manifest["archive"])["parts"], "filename"
                    )
                ),
            }
            for variant, manifest in self.bases.items()
        ]


def _channel_cache(store: Path, mirrors: tuple[str, ...], channel_name: str) -> Path:
    identity = hashlib.sha256(json.dumps(mirrors, ensure_ascii=True).encode("ascii")).hexdigest()
    return store / "metadata" / "channels" / identity / f"{channel_name}.json"


def discover(
    mirrors: tuple[str, ...],
    channel_name: str,
    *,
    store: Path | None = None,
    offline: bool = False,
) -> EnvironmentCatalog:
    """Read channel and exact manifests without creating project or store files."""
    if channel_name not in {"stable", "nightly"}:
        raise EnvironmentError("channel must be stable or nightly")
    document: dict[str, object] | None = None
    channel: Channel | None = None
    cached_channel = False
    envelope: bytes | None = None
    for mirror in () if offline else mirrors:
        try:
            envelope = read_metadata(mirror, f"channels/{channel_name}.json")
            document = verify_channel(envelope, channel_name)
            channel = Channel.parse(document)
            break
        except (OSError, ValueError, HTTPException):
            continue
    if channel is None and store is not None:
        cached = _channel_cache(store, mirrors, channel_name)
        try:
            envelope = read_metadata(str(cached.parent), cached.name)
            document = verify_channel(envelope, channel_name)
            channel = Channel.parse(document)
            cached_channel = True
        except (OSError, ValueError, HTTPException):
            pass
    if channel is None or document is None:
        raise EnvironmentError(
            "offline channel metadata missing or invalid"
            if offline
            else "no mirror supplied a valid trusted channel"
        )
    errors: list[str] = []
    engine = None
    engine_sources: tuple[str, ...] = ()
    engine_bytes = b""
    try:
        engine, engine_sources, engine_bytes = discover_manifest(
            channel.engine_locations, mirrors, channel.engine_release, store=store, offline=offline
        )
    except EnvironmentError as exc:
        if not offline:
            raise
        errors.append(f"engine {channel.engine_release}: {exc}")
    if engine is not None:
        if engine["version"] != channel.engine_version:
            raise EnvironmentError("engine manifest version disagrees with channel")
        select_frontend(engine, document)
    bases: dict[str, dict[str, object]] = {}
    sources: dict[str, tuple[str, ...]] = {}
    for variant, locations in channel.base_locations:
        try:
            base, roots, _ = discover_manifest(
                locations, mirrors, channel.base_for(variant), variant, store=store, offline=offline
            )
        except EnvironmentError as exc:
            if not offline:
                raise
            errors.append(f"base {variant} {channel.base_for(variant)}: {exc}")
            continue
        if engine is not None and not supports_base_protocol(engine, base):
            raise EnvironmentError("base and engine workerProtocol mismatch")
        bases[variant], sources[variant] = base, roots
    if errors:
        raise EnvironmentError("offline metadata inputs missing or invalid:\n" + "\n".join(errors))
    assert engine is not None
    return EnvironmentCatalog(
        channel_name,
        document,
        channel,
        engine,
        engine_sources,
        bases,
        sources,
        engine_bytes,
        mirrors,
        cached_channel,
        envelope,
    )


def acquisition_date(manifest: dict[str, object]) -> str:
    """Prefer producer build time; an allocation records the fallback date once."""
    built = manifest.get("builtAt")
    if isinstance(built, str):
        try:
            timestamp = datetime.fromisoformat(built)
            if timestamp.tzinfo is not None:
                return timestamp.astimezone(UTC).date().isoformat()
        except ValueError:
            pass
    return datetime.now(UTC).date().isoformat()


def base_label(base: dict[str, object]) -> str:
    wheels = artifact_rows(base["wheels"], "filename")
    torch = next((row["version"] for row in wheels if row.get("name") == "torch"), None)
    torch_version = Version(str(torch)).public if torch is not None else "none"
    return readable_label(
        str(base["variant"]),
        f"{acquisition_date(base)}-py{base['python']}-torch{torch_version}",
    )


def engine_label(engine: dict[str, object]) -> str:
    version = Version(str(engine["version"])).public
    label = engine.get("label")
    if isinstance(label, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}-" + re.escape(version), label):
        return readable_label(label)
    return readable_label(f"{acquisition_date(engine)}-{version}")


def acquire_base(
    base: dict[str, object], sources: tuple[str, ...], store: Path, *, offline: bool = False
) -> Path:
    """Acquire a validated base under a held store writer lock, without an engine release."""
    base_id = cast("str", base["id"])
    platform = cast("dict[str, object]", base["platform"])
    host = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    if platform["os"] != host:
        raise EnvironmentError("selected variant does not run on this operating system")
    architecture = machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(architecture, architecture)
    if platform["arch"] != architecture:
        raise EnvironmentError("selected variant does not run on this architecture")
    label = base_label(base)
    base_root = object_path(store / "bases", base_id, label=label)
    created = not base_root.exists()
    if created:
        parts = artifact_rows(cast("dict[str, object]", base["archive"])["parts"], "filename")
        if not parts:
            raise EnvironmentError(
                "base archive is unavailable; manifest reconstruction is required"
            )
        downloaded = tuple(
            fetch_artifact(
                (str(object_path(store / "downloads", base_id)),) if offline else sources,
                cast("str", row["filename"]),
                object_path(store / "downloads", base_id, label=label)
                / cast("str", row["filename"]),
                cast("str", row["sha256"]),
                cast("int", row["size"]),
            )
            for row in parts
        )
        extract_base(
            downloaded,
            base_root,
            platform=cast("str", platform["os"]),
            executable=cast("str", cast("dict[str, object]", base["interpreter"])["executable"]),
            expected_parts=parts,
        )
    try:
        for row in artifact_rows(base["wheels"], "filename"):
            source = base_root / "wheels" / cast("str", row["filename"])
            sha = cast("str", row["sha256"])
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != row["size"]
                or content_digest(source) != "sha256:" + sha
            ):
                raise EnvironmentError("extracted base wheel differs from its manifest")
            destination = (
                object_path(store / "wheels", "sha256:" + sha, label=readable_label(source.stem))
                / source.name
            )
            fetch_artifact(
                (str(source.parent),), source.name, destination, sha, cast("int", row["size"])
            )
        manifest = base_root / "manifest.json"
        if not manifest.exists():
            atomic_write(manifest, json.dumps(base, sort_keys=True, ensure_ascii=True) + "\n")
    except BaseException:
        if created:
            shutil.rmtree(base_root)
        raise
    return base_root


def dependency_wheels(selection: object) -> tuple[dict[str, object], ...]:
    """Validate exact install-lock identities independently of uploaded release artifacts."""
    if not isinstance(selection, dict):
        raise EnvironmentError("engine dependency selection must be an object")
    selection = cast("dict[str, object]", selection)
    base = selection.get("base")
    if not isinstance(base, str):
        raise EnvironmentError("engine dependency selection must name its base")
    digest(base)
    rows = artifact_rows(selection.get("wheels"), "filename")
    names: set[str] = set()
    for row in rows:
        name, version = row.get("name"), row.get("version")
        if not isinstance(name, str) or name != canonicalize_name(name) or name in names:
            raise EnvironmentError("dependency distribution names must be normalized and unique")
        names.add(name)
        try:
            wheel_name, wheel_version, _, _ = parse_wheel_filename(cast("str", row["filename"]))
            if (
                not isinstance(version, str)
                or wheel_name != name
                or wheel_version != Version(version)
            ):
                raise ValueError("dependency identity differs from its filename")
        except ValueError as exc:
            raise EnvironmentError("invalid locked dependency wheel identity") from exc
        for field in ("url", "index"):
            value = row.get(field)
            if not isinstance(value, str):
                raise EnvironmentError("dependency must record its source URL and index")
            parsed = urlparse(value)
            if (
                parsed.scheme not in {"https", "http"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise EnvironmentError("dependency source must be an uncredentialed HTTP(S) URL")
        if unquote(urlparse(cast("str", row["url"])).path.rsplit("/", 1)[-1]) != row["filename"]:
            raise EnvironmentError("dependency URL differs from the locked filename")
    return rows


def validate_dependency_base(selection: object, base: dict[str, object]) -> None:
    """Bind selected dependency metadata to the exact producer manifest, not generic pins."""
    selected = dependency_wheels(selection)
    if cast("dict[str, object]", selection)["base"] != base["id"]:
        raise EnvironmentError("engine dependency lock selects a different base")
    locked = {row["name"]: row for row in selected}
    fields = ("name", "version", "filename", "size", "sha256", "index", "url")
    wheels = artifact_rows(base["wheels"], "filename")
    for row in wheels:
        if locked.get(row["name"]) != {key: row[key] for key in fields}:
            raise EnvironmentError("base wheel differs from the engine dependency lock")
    if {row["name"] for row in selected if row["name"] in cast("list[str]", base["replaces"])} != {
        row["name"] for row in wheels
    }:
        raise EnvironmentError("engine lock contains a removed base-owned distribution")


def engine_package_versions(
    engine: dict[str, object],
    variant: str,
    base: dict[str, object] | None = None,
    *,
    inference: dict[str, object] | None = None,
) -> dict[str, str]:
    """Describe the target engine's shared packages without inspecting the planning host."""
    versions: dict[str, str] = {}
    if engine.get("dependencies") is not None:
        selections = cast("dict[str, object]", engine["dependencies"])
        versions.update(
            {
                cast("str", row["name"]): cast("str", row["version"])
                for row in dependency_wheels(selections.get(variant))
            }
        )
    key = "filename" if engine.get("source") == "git" else "name"
    for row in artifact_rows(engine["wheels"] if key == "filename" else engine["artifacts"], key):
        filename = cast("str", row[key])
        if filename.endswith(".whl"):
            name, version, _, _ = parse_wheel_filename(filename)
            versions[name] = str(version)
    if base is not None:
        for name in cast("list[str]", base["replaces"]):
            versions.pop(name, None)
        versions.update(
            {
                cast("str", row["name"]): cast("str", row["version"])
                for row in cast("list[dict[str, object]]", base["wheels"])
            }
        )
    if inference is not None:
        versions["dinkster-inference"] = cast("str", inference["version"])
    return versions


def engine_constraints(
    engine: dict[str, object], variant: str, *, inference: dict[str, object] | None = None
) -> str:
    """Use the same exact variant requirements for native pack builds and user installs."""
    selections = cast("dict[str, object]", engine["dependencies"])
    additions = [
        f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}"
        for row in dependency_wheels(inference or selections.get(variant))
    ]
    for row in artifact_rows(engine["artifacts"], "name"):
        filename = cast("str", row["name"])
        if filename.endswith(".whl"):
            name, version, _, _ = parse_wheel_filename(filename)
            additions.append(f"{name}=={version} --hash=sha256:{row['sha256']}")
    return "\n".join(additions) + "\n"


class WheelLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.yanked: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            links = [value for key, value in attrs if key == "href" and value is not None]
            self.links.extend(links)
            if any(key == "data-yanked" for key, _ in attrs):
                self.yanked.update(links)


def dependency_source(row: dict[str, object], package_index: str | None) -> str:
    if package_index is None:
        return cast("str", row["url"]).rsplit("/", 1)[0]
    project = cast("str", row["name"])
    page = package_index.rstrip("/") + "/" + project + "/"
    links = WheelLinks()
    links.feed(read_metadata(package_index, project).decode("utf-8"))
    matches: set[str] = set()
    for link in links.links:
        url = urljoin(page, link)
        parsed = urlparse(url)
        if unquote(parsed.path.rsplit("/", 1)[-1]) == row["filename"]:
            if parsed.query or parsed.username or parsed.password:
                raise EnvironmentError("package index supplied an unsupported wheel URL")
            matches.add(parsed._replace(fragment="").geturl().rsplit("/", 1)[0])
    if len(matches) != 1:
        raise EnvironmentError("package index must supply one URL for the exact locked filename")
    return matches.pop()


def runtime_inputs(
    catalog: EnvironmentCatalog,
    variant: str,
    store: Path,
    *,
    inference: dict[str, object] | None = None,
) -> tuple[dict[str, object], ...]:
    """Inspect exact runtime payloads without probing remote sources or writing cache files."""
    if (
        validate_manifest(catalog.engine_bytes, catalog.channel.engine_release, None)
        != catalog.engine
    ):
        raise EnvironmentError("engine manifest differs from its verified bytes")
    base = catalog.bases[variant]
    base_id = cast("str", base["id"])
    base_root = object_path(store / "bases", base_id)
    release = object_path(store / "engines", catalog.channel.engine_release)
    owned = set(cast("list[str]", base["replaces"]))
    inputs: dict[Path, dict[str, object]] = {}

    def record(path: Path, row: dict[str, object]) -> None:
        sha, size = cast("str", row["sha256"]), cast("int", row["size"])
        inputs[path] = {
            "path": str(path),
            "sha256": sha,
            "size": size,
            "cached": (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == size
                and content_digest(path) == "sha256:" + sha
            ),
        }

    if base_root.exists():
        for row in artifact_rows(base["wheels"], "filename"):
            record(base_root / "wheels" / cast("str", row["filename"]), row)
    else:
        for row in artifact_rows(cast("dict[str, object]", base["archive"])["parts"], "filename"):
            record(object_path(store / "downloads", base_id) / cast("str", row["filename"]), row)
    for row in artifact_rows(catalog.engine["artifacts"], "name"):
        name = cast("str", row["name"])
        if name == "constraints.txt":
            record(release / name, row)
        elif name.endswith(".whl") and parse_wheel_filename(name)[0] not in owned:
            record(
                object_path(store / "wheels", "sha256:" + cast("str", row["sha256"])) / name, row
            )
    if catalog.engine.get("dependencies") is not None:
        selection = (
            inference
            if inference is not None and "wheels" in inference
            else cast("dict[str, object]", catalog.engine["dependencies"]).get(variant)
        )
        validate_dependency_base(selection, base)
        for row in dependency_wheels(selection):
            if (
                inference is not None
                and row["name"] == "dinkster-inference"
                and "wheels" not in inference
            ):
                continue
            if row["name"] not in owned:
                record(
                    object_path(store / "wheels", "sha256:" + cast("str", row["sha256"]))
                    / cast("str", row["filename"]),
                    row,
                )
    frontend = select_frontend(catalog.engine, catalog.document)
    if frontend is not None:
        name = frontend_filename(frontend)
        record(
            object_path(store / "wheels", "sha256:" + cast("str", frontend["sha256"])) / name,
            frontend,
        )
    return tuple(inputs.values())


def acquire(
    catalog: EnvironmentCatalog,
    variant: str,
    store: Path,
    *,
    package_index: str | None = None,
    base: dict[str, object] | None = None,
    base_sources: tuple[str, ...] | None = None,
    offline: bool = False,
    inference: dict[str, object] | None = None,
) -> Path:
    """Populate verified artifacts and merged requirements under a held store writer lock."""
    if (
        validate_manifest(catalog.engine_bytes, catalog.channel.engine_release, None)
        != catalog.engine
    ):
        raise EnvironmentError("engine manifest differs from its verified bytes")
    if offline:
        if base is not None or base_sources is not None:
            raise EnvironmentError("offline runtime acquisition requires a published base")
        missing = [
            str(row["path"])
            for row in runtime_inputs(catalog, variant, store, inference=inference)
            if not row["cached"]
        ]
        if missing:
            raise EnvironmentError(
                "offline runtime inputs missing or corrupt:\n" + "\n".join(missing)
            )
    published_base = catalog.bases[variant]
    custom = base is not None
    if custom != (base_sources is not None):
        raise EnvironmentError("custom acquisition requires paired base and base_sources")
    if base is None:
        base = published_base
        base_sources = catalog.base_sources[variant]
    else:
        if base.get("source") != "user-defined":
            raise EnvironmentError("explicit base acquisition requires a user-defined base")
        validate_manifest(
            json.dumps(base).encode("utf-8"), cast("str", base["id"]), cast("str", base["variant"])
        )
        if any(
            base[field] != published_base[field]
            for field in ("platform", "python", "workerProtocol")
        ):
            raise EnvironmentError("custom base differs from the engine dependency target ABI")
    selections = catalog.engine.get("dependencies")
    selected = None
    if selections is not None:
        selection = (
            inference
            if inference is not None and "wheels" in inference
            else cast("dict[str, object]", selections).get(variant)
        )
        validate_dependency_base(selection, published_base)
        selected = dependency_wheels(selection)
    assert base_sources is not None
    acquire_base(base, base_sources, store, offline=offline)
    release_root = object_path(
        store / "engines", catalog.channel.engine_release, label=engine_label(catalog.engine)
    )
    owned = set(cast("list[str]", base["replaces"]))
    if custom:
        owned.update(cast("list[str]", published_base["replaces"]))
    nonbase_wheels: list[Path] = []
    for row in artifact_rows(catalog.engine["artifacts"], "name"):
        name, sha = cast("str", row["name"]), cast("str", row["sha256"])
        if name.endswith(".whl"):
            try:
                distribution, _, _, _ = parse_wheel_filename(name)
            except InvalidWheelFilename as exc:
                raise EnvironmentError("engine artifact has an invalid wheel filename") from exc
            if distribution in owned:
                continue
            destination = (
                object_path(
                    store / "wheels", "sha256:" + sha, label=readable_label(Path(name).stem)
                )
                / name
            )
            nonbase_wheels.append(destination)
        elif name == "constraints.txt":
            destination = release_root / name
        else:
            continue
        sources = catalog.engine_sources
        if "url" in row:
            sources = (cast("str", row["url"]).rsplit("/", 1)[0], *sources)
        fetch_artifact(
            (str(destination.parent),) if offline else sources,
            name,
            destination,
            sha,
            cast("int", row["size"]),
        )
    requirements = (release_root / "constraints.txt").read_text(encoding="utf-8")
    if selected is not None:
        for row in selected:
            name, sha = cast("str", row["filename"]), cast("str", row["sha256"])
            if (
                inference is not None
                and row["name"] == "dinkster-inference"
                and row["version"] != inference["version"]
            ):
                continue
            if row["name"] not in owned:
                destination = (
                    object_path(
                        store / "wheels", "sha256:" + sha, label=readable_label(Path(name).stem)
                    )
                    / name
                )
                nonbase_wheels.append(destination)
                if (
                    not destination.is_file()
                    or destination.is_symlink()
                    or destination.stat().st_size != row["size"]
                    or content_digest(destination) != "sha256:" + sha
                ):
                    source = (
                        str(destination.parent)
                        if offline
                        else dependency_source(row, package_index)
                    )
                    fetch_artifact((source,), name, destination, sha, cast("int", row["size"]))
        requirements = engine_constraints(catalog.engine, variant)
    frontend = select_frontend(catalog.engine, catalog.document)
    if frontend is not None:
        name, sha = frontend_filename(frontend), cast("str", frontend["sha256"])
        destination = (
            object_path(store / "wheels", "sha256:" + sha, label=readable_label(Path(name).stem))
            / name
        )
        source = (
            str(destination.parent) if offline else cast("str", frontend["url"]).rsplit("/", 1)[0]
        )
        fetch_artifact((source,), name, destination, sha, cast("int", frontend["size"]))
        verify_frontend_wheel(destination, frontend)
        nonbase_wheels.append(destination)
    requirements = frontend_requirements(requirements, catalog.engine, catalog.document)
    context = published_base if custom else None
    if inference is not None:
        if selected is None:
            raise EnvironmentError("explicit inference requires a published dependency lock")
        from .environment_inference import resolve_inference_wheels
        from .environment_state import interpreter

        if "wheels" not in inference:
            if offline:
                raise EnvironmentError(
                    "offline inference selection requires retained wheel identities"
                )
            base_wheels = tuple(
                object_path(store / "wheels", "sha256:" + cast("str", row["sha256"]))
                / cast("str", row["filename"])
                for row in artifact_rows(base["wheels"], "filename")
            )
            resolve_inference_wheels(
                inference,
                requirements,
                (*nonbase_wheels, *base_wheels),
                selected,
                python=interpreter(object_path(store / "bases", cast("str", base["id"]))),
                store=store,
                package_index=package_index,
            )
            inference["base"] = published_base["id"]
        validate_dependency_base(inference, published_base)
        override_rows = dependency_wheels(inference)
        if [row["version"] for row in override_rows if row["name"] == "dinkster-inference"] != [
            inference["version"]
        ]:
            raise EnvironmentError("inference wheel selection differs from the requested version")
        nonbase_wheels.extend(
            object_path(store / "wheels", "sha256:" + cast("str", row["sha256"]))
            / cast("str", row["filename"])
            for row in override_rows
            if row["name"] not in owned
        )
        requirements = engine_constraints(catalog.engine, variant, inference=inference)
        requirements = frontend_requirements(requirements, catalog.engine, catalog.document)
    if selected is not None or custom:
        validate_base_requirements(tuple(nonbase_wheels), base, published_base=context)
    merged = effective_constraints(
        requirements,
        base,
        strict=selected is not None and not custom,
        custom=custom,
        published_base=context,
    )
    identity = hashlib.sha256(merged.encode("utf-8")).hexdigest()
    constraints = release_root / f"constraints-{identity}.txt"
    atomic_write(constraints, merged)
    atomic_write(release_root / "release-manifest.json", catalog.engine_bytes)
    atomic_write(
        store / "metadata" / f"{catalog.channel.engine_release[7:]}.json", catalog.engine_bytes
    )
    for retained_base in catalog.bases.values():
        atomic_write(
            store / "metadata" / f"{cast('str', retained_base['id'])[7:]}.json",
            json.dumps(retained_base, sort_keys=True, ensure_ascii=True) + "\n",
        )
    if catalog.mirrors is not None and catalog.channel_envelope is not None:
        verify_channel(catalog.channel_envelope, catalog.channel_name)
        atomic_write(
            _channel_cache(store, catalog.mirrors, catalog.channel_name), catalog.channel_envelope
        )
    return constraints


def acquire_pack_locks(catalog: EnvironmentCatalog, variant: str, store: Path) -> tuple[Path, ...]:
    """Transport selected dependency locks without interpreting their producer-owned schema."""
    catalog.channel.base_for(variant)
    version = Version(cast("str", catalog.engine["version"])).public
    suffix = f"-{version}-{variant}.pack-lock.json"
    release = object_path(
        store / "engines", catalog.channel.engine_release, label=engine_label(catalog.engine)
    )
    return tuple(
        fetch_artifact(
            catalog.engine_sources,
            cast("str", row["name"]),
            release / cast("str", row["name"]),
            cast("str", row["sha256"]),
            cast("int", row["size"]),
        )
        for row in artifact_rows(catalog.engine["artifacts"], "name")
        if cast("str", row["name"]).endswith(suffix)
    )


def acquire_sources(
    catalog: EnvironmentCatalog, store: Path, *, download: bool = True
) -> tuple[Path, Path] | None:
    """Verify release payload transport; target admission alone interprets the pack lock."""
    rows = artifact_rows(catalog.engine["artifacts"], "name")
    if not any(row["name"] == "official-packs.lock" for row in rows):
        return None
    release = object_path(
        store / "engines",
        catalog.channel.engine_release,
        label=engine_label(catalog.engine) if download else None,
    )
    artifacts = release / "artifacts"
    if download:
        artifacts.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name, sha, size = (
            cast("str", row["name"]),
            cast("str", row["sha256"]),
            cast("int", row["size"]),
        )
        if name != "official-packs.lock" and not name.endswith(".zip"):
            continue
        downloaded = release / name
        if download:
            fetch_artifact(catalog.engine_sources, name, downloaded, sha, size)
        elif (
            downloaded.is_symlink()
            or not downloaded.is_file()
            or downloaded.stat().st_size != size
            or content_digest(downloaded) != "sha256:" + sha
        ):
            raise EnvironmentError(f"planning requires intact cached release source: {name}")
        if name.endswith(".zip"):
            identity = blake3()
            with downloaded.open("rb") as source:
                for chunk in iter(lambda: source.read(ENVIRONMENT_STREAM_CHUNK_BYTES), b""):
                    identity.update(chunk)
            pack, separator, version = name.removesuffix(".zip").rpartition("-")
            label = pack_label(pack if separator else version, version if separator else "0.0.0")
            cached = pack_archive_path(
                artifacts, "blake3:" + identity.hexdigest(), label=label if download else None
            )
            if download:
                fetch_artifact((str(release),), name, cached, sha, size)
            elif (
                cached.is_symlink()
                or not cached.is_file()
                or cached.stat().st_size != size
                or content_digest(cached) != "sha256:" + sha
            ):
                raise EnvironmentError(f"planning requires intact cached release source: {cached}")
    return release / "official-packs.lock", artifacts
