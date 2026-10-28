"""Assemble a variant's hashed requirements without resolving dependencies."""

from __future__ import annotations

import re
from typing import cast

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .environment_state import EnvironmentError, digest, relative_file

_HASH = re.compile(r"(?<!\S)--hash=sha256:([0-9a-f]{64})(?=\s|$)")


def effective_constraints(engine: str, base: dict[str, object]) -> str:
    """Replace precisely the base producer's owned closure, including remove-only rows.

    The caller verifies manifest identity first. Local torch version suffixes
    select a backend wheel but must match the engine's locked public version.
    All unrelated requirement markers and hashes are retained verbatim.
    """
    replaces = base.get("replaces")
    if not isinstance(replaces, list) or not replaces:
        raise EnvironmentError("base manifest must provide replacement membership")
    names = cast("list[object]", replaces)
    if any(
        not isinstance(name, str) or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
        for name in names
    ):
        raise EnvironmentError("base replacement names must be normalized distribution names")
    replacement_names = cast("list[str]", names)
    if replacement_names != sorted(set(replacement_names)):
        raise EnvironmentError("base replacement names must be sorted and unique")
    owned = set(replacement_names)
    retained: list[str] = []
    locked: dict[str, set[Version]] = {}
    for line in engine.replace("\\\r\n", "").replace("\\\n", "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            retained.append(line)
            continue
        if not _HASH.search(line):
            raise EnvironmentError("engine requirements must carry exact sha256 hashes")
        try:
            requirement = Requirement(_HASH.sub("", line).strip())
        except InvalidRequirement as exc:
            raise EnvironmentError("engine constraints contain an invalid requirement") from exc
        pins = list(requirement.specifier)
        if (
            requirement.url is not None
            or len(pins) != 1
            or pins[0].operator != "=="
            or "*" in pins[0].version
        ):
            raise EnvironmentError("engine requirements must use exact version pins")
        name = canonicalize_name(requirement.name)
        try:
            locked.setdefault(name, set()).add(Version(pins[0].version))
        except InvalidVersion as exc:
            raise EnvironmentError(f"invalid engine version for {name}") from exc
        if name not in owned:
            retained.append(line)

    wheels = base.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise EnvironmentError("base manifest must provide selected wheels")
    selected: set[str] = set()
    additions: list[str] = []
    for value in cast("list[object]", wheels):
        if not isinstance(value, dict):
            raise EnvironmentError("base wheel must be an object")
        wheel = cast("dict[str, object]", value)
        name, version = wheel.get("name"), wheel.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise EnvironmentError("base wheel must name its distribution and version")
        if name != canonicalize_name(name) or name not in owned or name in selected:
            raise EnvironmentError(
                f"base wheel {name} has invalid or duplicate replacement ownership"
            )
        selected.add(name)
        filename = relative_file(wheel.get("filename"))
        try:
            wheel_name, wheel_version, _, _ = parse_wheel_filename(filename)
            parsed_version = Version(version)
            if wheel_name != name or wheel_version != parsed_version:
                raise EnvironmentError(f"base wheel identity disagrees with filename: {name}")
            versions = locked.get(name, set())
            declared = wheel.get("lockedVersion")
            if declared is not None:
                if not isinstance(declared, str):
                    raise EnvironmentError(f"invalid lockedVersion for {name}")
                versions = versions | {Version(declared)}
            if any(
                candidate != parsed_version
                and not (
                    name in {"torch", "torchvision"}
                    and candidate.public == parsed_version.public
                    and candidate.local is None
                )
                for candidate in versions
            ):
                raise EnvironmentError(f"base wheel {name} {version} disagrees with engine lock")
        except (InvalidVersion, InvalidWheelFilename) as exc:
            raise EnvironmentError(f"invalid base wheel identity: {name}") from exc
        sha = wheel.get("sha256")
        if not isinstance(sha, str):
            raise EnvironmentError(f"missing wheel digest: {name}")
        digest("sha256:" + sha)
        additions.append(f"{name}=={version} --hash=sha256:{sha}")
    return "\n".join([*retained, *sorted(additions)]).strip() + "\n"
