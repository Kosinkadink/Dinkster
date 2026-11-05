"""Content-verified metadata for the engine-independent native tools publication."""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from platform import machine
from typing import cast
from urllib.parse import urlsplit

from packaging.version import InvalidVersion, Version

from .environment_artifacts import (
    artifact_rows,
    discover_manifest,
    parse_metadata,
    trusted_metadata,
)
from .environment_downloads import fetch_artifact, read_metadata
from .environment_state import Channel, EnvironmentError

TOOLS_ARCHIVES = frozenset(
    f"dinkster-tools-{platform}.zip"
    for platform in ("linux-x86_64", "windows-x86_64", "macos-arm64")
)


def validate_tools_manifest(data: bytes, sha256: str) -> dict[str, object]:
    """Admit only a complete three-platform publication by its exact byte hash."""
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None or hashlib.sha256(data).hexdigest() != sha256:
        raise EnvironmentError("tools manifest SHA256 mismatch")
    manifest = parse_metadata(data)
    if (
        manifest.get("format") != "dinkster.tools/1"
        or manifest.get("repository") != "Kosinkadink/Dinkster"
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("commit"))) is None
    ):
        raise EnvironmentError("unsupported tools manifest")
    try:
        version = Version(str(manifest["version"]))
        built_at = datetime.fromisoformat(str(manifest["builtAt"]).replace("Z", "+00:00"))
    except (KeyError, ValueError, InvalidVersion) as error:
        raise EnvironmentError("invalid tools version or build date") from error
    offset = built_at.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise EnvironmentError("tools build date must be UTC")
    if manifest.get("label") != f"{built_at.date().isoformat()}-{version.public}":
        raise EnvironmentError("tools label differs from version or build date")
    rows = artifact_rows(manifest.get("artifacts"), "name")
    if {cast("str", row["name"]) for row in rows} != set(TOOLS_ARCHIVES) or any(
        row["size"] == 0 for row in rows
    ):
        raise EnvironmentError("tools manifest must contain all three native archives")
    return manifest


def tools_reference(value: object) -> dict[str, str]:
    """Validate the channel's documented absolute manifest and payload locations."""
    if not isinstance(value, dict):
        raise EnvironmentError("channel tools must be an object")
    value = cast("dict[str, object]", value)
    result: dict[str, str] = {}
    for field in ("manifest", "sha256", "release", "label"):
        entry = value.get(field)
        if not isinstance(entry, str) or not entry:
            raise EnvironmentError(f"channel tools must name {field}")
        result[field] = entry
    if re.fullmatch(r"[0-9a-f]{64}", result["sha256"]) is None:
        raise EnvironmentError("channel tools require a manifest SHA256")
    if (
        re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9][A-Za-z0-9._-]*-" + result["sha256"][:8],
            result["label"],
        )
        is None
    ):
        raise EnvironmentError("channel tools label must name its dated version and hash8")
    for field in ("manifest", "release"):
        location = result[field]
        if Path(location).is_absolute():
            continue
        url = urlsplit(location)
        if (
            url.scheme != "https"
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise EnvironmentError("tools inputs require absolute paths or credential-free HTTPS")
    if result["manifest"] != result["release"] + "/tools-manifest.json" or not result[
        "release"
    ].endswith("/tools/" + result["label"]):
        raise EnvironmentError("channel tools locations differ from the publication label")
    return result


def acquire_tools(
    mirrors: tuple[str, ...], channel_name: str, destination: Path, *, download: bool
) -> dict[str, object]:
    """Resolve native tools independently of engine/base payloads, then verify transport."""
    if channel_name not in {"main", "stable"}:
        raise EnvironmentError("channel must be stable or main")
    channel = None
    document: dict[str, object] = {}
    for mirror in mirrors:
        try:
            trusted_metadata(mirror)
            document = parse_metadata(read_metadata(mirror, f"channels/{channel_name}.json"))
            channel = Channel.parse(document)
            break
        except (OSError, ValueError, HTTPException):
            continue
    if channel is None:
        raise EnvironmentError("no mirror supplied a valid trusted channel")
    if "tools" in document:
        reference = tools_reference(document["tools"])
        data = read_metadata(reference["release"], "tools-manifest.json")
        manifest = validate_tools_manifest(data, reference["sha256"])
        label = f"{manifest['label']}-{reference['sha256'][:8]}"
        if label != reference["label"]:
            raise EnvironmentError("tools manifest identity differs from channel")
        sources = (reference["release"],)
    else:
        # Old channels retain their engine source until both consumers switch.
        manifest, sources, _ = discover_manifest(
            channel.engine_locations, mirrors, channel.engine_release
        )
        label = channel.engine_release[7:]
    architecture = machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(architecture, architecture)
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(sys.platform)
    name = f"dinkster-tools-{system}-{architecture}.zip"
    row = next(
        (row for row in artifact_rows(manifest.get("artifacts"), "name") if row["name"] == name),
        None,
    )
    if row is None:
        raise EnvironmentError("publication has no tools for this operating system/architecture")
    archive = destination / label / name
    if download:
        fetch_artifact(sources, name, archive, cast("str", row["sha256"]), cast("int", row["size"]))
    return {"label": label, "release": sources[0], "archive": str(archive), **row}
