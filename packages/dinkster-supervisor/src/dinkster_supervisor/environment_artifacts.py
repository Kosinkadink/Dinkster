"""Discover exact release manifests and populate the verified local wheel store."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import cast
from urllib.parse import quote, urlparse

from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from .environment_archives import extract_base
from .environment_constraints import effective_constraints
from .environment_downloads import fetch_artifact, read_metadata
from .environment_state import (
    ArtifactLocation,
    Channel,
    EnvironmentError,
    atomic_write,
    content_digest,
    digest,
    relative_file,
)

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


def _document(data: bytes) -> dict[str, object]:
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


def _rows(value: object, key: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise EnvironmentError("manifest artifacts must be an array")
    rows: list[dict[str, object]] = []
    names: set[str] = set()
    for item in cast("list[object]", value):
        if not isinstance(item, dict):
            raise EnvironmentError("manifest artifact must be an object")
        row = cast("dict[str, object]", item)
        name = relative_file(row.get(key))
        if "/" in name or name.casefold() in names or name.endswith((".", " ")):
            raise EnvironmentError("artifact names must be distinct portable basenames")
        names.add(name.casefold())
        sha, size = row.get("sha256"), row.get("size")
        if not isinstance(sha, str) or type(size) is not int or size < 0:
            raise EnvironmentError("artifact must carry a sha256 digest and nonnegative size")
        digest("sha256:" + sha)
        rows.append(row)
    return tuple(rows)


def _manifest(data: bytes, expected: str, variant: str | None) -> dict[str, object]:
    digest(expected)
    document = _document(data)
    protocol = document.get("workerProtocol")
    if type(protocol) is not int or protocol < 1:
        raise EnvironmentError("manifest workerProtocol must be a positive integer")
    if variant is None:
        if document.get("format") != "dinkster.engine/1":
            raise EnvironmentError("unsupported engine manifest format")
        if "sha256:" + hashlib.sha256(data).hexdigest() != expected:
            raise EnvironmentError("engine manifest digest mismatch")
        if not isinstance(document.get("version"), str) or not document["version"]:
            raise EnvironmentError("engine manifest version is missing")
        commit = document.get("commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise EnvironmentError("engine manifest must name its full commit")
        rows = _rows(document.get("artifacts"), "name")
        if not any(row["name"] == "constraints.txt" for row in rows):
            raise EnvironmentError("engine release has no constraints.txt artifact")
    else:
        if document.get("format") != "dinkster.base/1" or document.get("variant") != variant:
            raise EnvironmentError("base manifest format or variant mismatch")
        if document.get("signature") is not None:
            raise EnvironmentError("signed base verification is not supported")
        try:
            payload = {key: document[key] for key in _BASE_IDENTITY}
        except KeyError as exc:
            raise EnvironmentError("base manifest identity fields are missing") from exc
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        actual = "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()
        if document.get("id") != expected or actual != expected:
            raise EnvironmentError("base manifest digest mismatch")
        _rows(document.get("wheels"), "filename")
        _rows(document.get("parts"), "filename")
        # Validate the producer-owned replacement set and wheel identities before acquisition.
        effective_constraints("", document)
        platform = document.get("platform")
        if not isinstance(platform, dict):
            raise EnvironmentError("base manifest platform is invalid")
        platform = cast("dict[str, object]", platform)
        if platform.get("os") not in ("windows", "linux"):
            raise EnvironmentError("base manifest platform is invalid")
        if platform.get("arch") != "x86_64":
            raise EnvironmentError("unsupported base architecture")
        python = document.get("interpreter")
        if not isinstance(python, dict):
            raise EnvironmentError("base interpreter metadata is missing")
        python = cast("dict[str, object]", python)
        expected_python = (
            "python/python.exe" if platform["os"] == "windows" else "python/bin/python3"
        )
        if python.get("executable") != expected_python:
            raise EnvironmentError("base interpreter executable differs from the generation layout")
    return document


def _trusted_metadata(mirror: str) -> None:
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
                _trusted_metadata(mirror)
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


def _discover_manifest(
    locations: tuple[ArtifactLocation, ...],
    mirrors: tuple[str, ...],
    expected: str,
    variant: str | None = None,
) -> tuple[dict[str, object], tuple[str, ...]]:
    sources = _sources(locations, mirrors)
    for root, filename in sources:
        try:
            document = _manifest(read_metadata(root, filename), expected, variant)
            return document, tuple(source for source, _ in sources)
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

    def variants(self) -> list[dict[str, object]]:
        return [
            {
                "variant": variant,
                "base": manifest["id"],
                "manifest": manifest,
                "downloadBytes": sum(
                    cast("int", row["size"]) for row in _rows(manifest["parts"], "filename")
                ),
            }
            for variant, manifest in self.bases.items()
        ]


def discover(mirrors: tuple[str, ...], channel_name: str) -> EnvironmentCatalog:
    """Read channel and exact manifests without creating project or store files."""
    if channel_name not in {"stable", "main"}:
        raise EnvironmentError("channel must be stable or main")
    document: dict[str, object] | None = None
    channel: Channel | None = None
    for mirror in mirrors:
        try:
            _trusted_metadata(mirror)
            document = _document(read_metadata(mirror, f"channels/{channel_name}.json"))
            channel = Channel.parse(document)
            break
        except (OSError, ValueError, HTTPException):
            continue
    if channel is None or document is None:
        raise EnvironmentError("no mirror supplied a valid trusted channel")
    engine, engine_sources = _discover_manifest(
        channel.engine_locations, mirrors, channel.engine_release
    )
    if engine["version"] != channel.engine_version:
        raise EnvironmentError("engine manifest version disagrees with channel")
    bases: dict[str, dict[str, object]] = {}
    sources: dict[str, tuple[str, ...]] = {}
    for variant, locations in channel.base_locations:
        base, roots = _discover_manifest(locations, mirrors, channel.base_for(variant), variant)
        if base["workerProtocol"] != engine["workerProtocol"]:
            raise EnvironmentError("base and engine workerProtocol mismatch")
        bases[variant], sources[variant] = base, roots
    return EnvironmentCatalog(
        channel_name, document, channel, engine, engine_sources, bases, sources
    )


def acquire(catalog: EnvironmentCatalog, variant: str, store: Path) -> Path:
    """Populate verified artifacts and merged requirements under a held store writer lock."""
    base_id = catalog.channel.base_for(variant)
    base = catalog.bases[variant]
    platform = cast("dict[str, object]", base["platform"])
    if platform["os"] != ("windows" if os.name == "nt" else "linux"):
        raise EnvironmentError("selected variant does not run on this operating system")
    base_root = store / "bases" / base_id[7:]
    created = not base_root.exists()
    if created:
        parts = _rows(base["parts"], "filename")
        if not parts:
            raise EnvironmentError(
                "base archive is unavailable; manifest reconstruction is required"
            )
        downloaded = tuple(
            fetch_artifact(
                catalog.base_sources[variant],
                cast("str", row["filename"]),
                store / "downloads" / base_id[7:] / cast("str", row["filename"]),
                cast("str", row["sha256"]),
                cast("int", row["size"]),
            )
            for row in parts
        )
        extract_base(downloaded, base_root)
    try:
        for row in _rows(base["wheels"], "filename"):
            source = base_root / "wheels" / cast("str", row["filename"])
            sha = cast("str", row["sha256"])
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != row["size"]
                or content_digest(source) != "sha256:" + sha
            ):
                raise EnvironmentError("extracted base wheel differs from its manifest")
            destination = store / "wheels" / sha / source.name
            fetch_artifact(
                (str(source.parent),), source.name, destination, sha, cast("int", row["size"])
            )
    except BaseException:
        if created:
            shutil.rmtree(base_root)
        raise
    release_root = store / "engines" / catalog.channel.engine_release[7:]
    owned = cast("list[str]", base["replaces"])
    for row in _rows(catalog.engine["artifacts"], "name"):
        name, sha = cast("str", row["name"]), cast("str", row["sha256"])
        if name.endswith(".whl"):
            try:
                distribution, _, _, _ = parse_wheel_filename(name)
            except InvalidWheelFilename as exc:
                raise EnvironmentError("engine artifact has an invalid wheel filename") from exc
            if distribution in owned:
                continue
            destination = store / "wheels" / sha / name
        elif name == "constraints.txt":
            destination = release_root / name
        else:
            continue
        fetch_artifact(catalog.engine_sources, name, destination, sha, cast("int", row["size"]))
    merged = effective_constraints(
        (release_root / "constraints.txt").read_text(encoding="utf-8"), base
    )
    constraints = release_root / f"constraints-{variant}.txt"
    atomic_write(constraints, merged)
    return constraints
