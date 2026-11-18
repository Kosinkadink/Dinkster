"""Content-verified metadata for the engine-independent native tools publication."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from platform import machine
from typing import cast
from urllib.parse import urlsplit
from zipfile import BadZipFile

from dinkster_library.client import API_VERSION, CAPABILITIES
from dinkster_store import object_path, readable_label, relative_path
from filelock import FileLock
from packaging.version import InvalidVersion, Version

from .environment_archives import extract_tools
from .environment_artifacts import artifact_rows, parse_metadata
from .environment_channel_signatures import verify_channel
from .environment_downloads import fetch_artifact, read_metadata
from .environment_state import (
    Channel,
    EnvironmentError,
    interpreter,
    read_document,
    write_document,
)

TOOLS_ARCHIVES = frozenset(
    f"dinkster-tools-{platform}.zip"
    for platform in ("linux-x86_64", "windows-x86_64", "macos-arm64")
)


def _helper_description(command: list[str]) -> tuple[int, int, frozenset[str]]:
    try:
        result = subprocess.run(
            [*command, "--describe"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=True,
        )
        description = parse_metadata(result.stdout.encode("utf-8"))
        low, high = description.get("apiMin"), description.get("apiMax")
        capabilities = description.get("capabilities", [])
        if not isinstance(capabilities, list) or any(
            not isinstance(value, str) for value in cast("list[object]", capabilities)
        ):
            raise EnvironmentError("library helper capabilities must be a list of strings")
        if type(low) is int and type(high) is int and 0 < low <= high:
            return low, high, frozenset(cast("list[str]", capabilities))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise EnvironmentError(f"library helper compatibility check failed: {exc}") from exc
    raise EnvironmentError("library helper returned an invalid API range")


def _installed_helper(root: Path) -> tuple[list[str], int, int, frozenset[str]]:
    command = [sys.executable, "-I", "-m", "dinkster_library"]
    selection = root / "library-tools.json"
    if selection.is_file():
        directory = relative_path(read_document(selection).get("directory"))
        if "/" in directory:
            raise EnvironmentError("invalid selected library tools directory")
        selected = [
            str(interpreter(root / "tools" / directory / "runtime")),
            "-I",
            "-m",
            "dinkster_library",
        ]
        low, high, capabilities = _helper_description(selected)
        if low <= API_VERSION <= high:
            return selected, low, high, capabilities
    return command, API_VERSION, API_VERSION, frozenset(CAPABILITIES)


def library_command(
    root: Path, manifest: dict[str, object], *, sources: dict[str, object] | None = None
) -> list[str]:
    """Admit tools before engine installation or shutdown, without opening the library."""
    required = manifest.get("libraryApi")
    if required is not None and (type(required) is not int or required < 1):
        raise EnvironmentError("engine libraryApi must be a positive integer")
    requested = manifest.get("libraryCapabilities", [])
    if not isinstance(requested, list) or any(
        not isinstance(value, str) for value in cast("list[object]", requested)
    ):
        raise EnvironmentError("engine libraryCapabilities must be a list of strings")
    needed = frozenset(cast("list[str]", requested))
    if needed and required is None:
        raise EnvironmentError("engine libraryCapabilities requires libraryApi")
    command, low, high, capabilities = _installed_helper(root)
    if required is None or (low <= required <= high and needed <= capabilities):
        return command
    if required < low:
        raise EnvironmentError(
            f"Engine requests library API {required}; helper offers {low} to {high}."
        )
    settings = sources or (
        read_document(root / "environment.json") if (root / "environment.json").is_file() else {}
    )
    mirrors = settings.get("mirrors", [])
    channel = settings.get("channel", "stable")
    if isinstance(channel, dict):
        channel = cast("dict[str, object]", channel).get("channel", "stable")
    if (
        not isinstance(mirrors, list)
        or not all(isinstance(mirror, str) and mirror for mirror in cast("list[object]", mirrors))
        or channel not in {"stable", "nightly"}
    ):
        raise EnvironmentError("invalid tools update sources")
    tools = root / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    with FileLock(tools / ".update.lock"):
        command, low, high, capabilities = _installed_helper(root)
        if low <= required <= high and needed <= capabilities:
            return command
        row = acquire_tools(
            tuple(cast("list[str]", mirrors)), cast("str", channel), tools, download=True
        )
        directory = Path(cast("str", row["archive"])).parent
        runtime = directory / "runtime"
        if not runtime.exists():
            with tempfile.TemporaryDirectory(prefix=".tools-stage-", dir=directory) as temporary:
                staged = Path(temporary) / "runtime"
                try:
                    extract_tools(Path(cast("str", row["archive"])), staged)
                except (OSError, BadZipFile) as exc:
                    raise EnvironmentError(f"tools archive extraction failed: {exc}") from exc
                candidate = [str(interpreter(staged)), "-I", "-m", "dinkster_library"]
                candidate_low, candidate_high, candidate_capabilities = _helper_description(
                    candidate
                )
                missing = (capabilities | needed) - candidate_capabilities
                if missing:
                    raise EnvironmentError(
                        "New library helper lacks capabilities: " + ", ".join(sorted(missing))
                    )
                if not candidate_low <= min(low, required) <= max(high, required) <= candidate_high:
                    raise EnvironmentError(
                        f"Engine requests library API {required}; new helper offers "
                        f"{candidate_low} to {candidate_high}."
                    )
                staged.rename(runtime)
        command = [str(interpreter(runtime)), "-I", "-m", "dinkster_library"]
        candidate_low, candidate_high, candidate_capabilities = _helper_description(command)
        missing = (capabilities | needed) - candidate_capabilities
        if missing:
            raise EnvironmentError(
                "New library helper lacks capabilities: " + ", ".join(sorted(missing))
            )
        if not candidate_low <= min(low, required) <= max(high, required) <= candidate_high:
            raise EnvironmentError(
                f"Engine requests library API {required}; new helper offers "
                f"{candidate_low} to {candidate_high}."
            )
        write_document(root / "library-tools.json", {"directory": directory.name})
        return command


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
    release = Path(result["release"])
    if release.is_absolute():
        matches = (
            Path(result["manifest"]) == release / "tools-manifest.json"
            and release.name == result["label"]
            and release.parent.name == "tools"
        )
    else:
        matches = result["manifest"] == result["release"] + "/tools-manifest.json" and result[
            "release"
        ].endswith("/tools/" + result["label"])
    if not matches:
        raise EnvironmentError("channel tools locations differ from the publication label")
    return result


def acquire_tools(
    mirrors: tuple[str, ...], channel_name: str, destination: Path, *, download: bool
) -> dict[str, object]:
    """Resolve native tools independently of engine/base payloads, then verify transport."""
    if channel_name not in {"nightly", "stable"}:
        raise EnvironmentError("channel must be stable or nightly")
    architecture = machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(architecture, architecture)
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(sys.platform)
    name = f"dinkster-tools-{system}-{architecture}.zip"
    if name not in TOOLS_ARCHIVES:
        raise EnvironmentError(f"unsupported native tools host: {sys.platform}/{architecture}")
    channel = None
    document: dict[str, object] = {}
    for mirror in mirrors:
        try:
            document = verify_channel(
                read_metadata(mirror, f"channels/{channel_name}.json"), channel_name
            )
            channel = Channel.parse(document)
            break
        except (OSError, ValueError, HTTPException):
            continue
    if channel is None:
        raise EnvironmentError("no mirror supplied a valid trusted channel")
    reference = tools_reference(document.get("tools"))
    data = read_metadata(reference["release"], "tools-manifest.json")
    manifest = validate_tools_manifest(data, reference["sha256"])
    label = f"{manifest['label']}-{reference['sha256'][:8]}"
    if label != reference["label"]:
        raise EnvironmentError("tools manifest identity differs from channel")
    sources = (reference["release"],)
    row = next(
        (row for row in artifact_rows(manifest.get("artifacts"), "name") if row["name"] == name),
        None,
    )
    if row is None:
        raise EnvironmentError("publication has no tools for this operating system/architecture")
    directory = object_path(
        destination,
        "sha256:" + reference["sha256"],
        label=readable_label(cast("str", manifest["label"])),
        legacy=label,
        reserve=download,
    )
    archive = directory / name
    if download:
        fetch_artifact(sources, name, archive, cast("str", row["sha256"]), cast("int", row["size"]))
    return {"label": directory.name, "release": sources[0], "archive": str(archive), **row}
