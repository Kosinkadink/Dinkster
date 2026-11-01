"""Assemble a variant's hashed requirements without resolving dependencies."""

from __future__ import annotations

import re
from email.parser import BytesParser
from pathlib import Path
from typing import cast
from zipfile import BadZipFile, ZipFile

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .environment_state import EnvironmentError, digest, relative_file

_HASH = re.compile(r"(?<!\S)--hash=sha256:([0-9a-f]{64})(?=\s|$)")


def requirement_hashes(requirements: str) -> frozenset[str]:
    """Keep every hashed alternative, including other-platform marker rows."""
    return frozenset(_HASH.findall(requirements.replace("\\\n", "").replace("\\\r\n", "")))


def validate_base_requirements(
    wheels: tuple[Path, ...],
    base: dict[str, object],
    *,
    published_base: dict[str, object] | None = None,
) -> None:
    """Check installed non-base wheel requirements against a native base, without resolving."""
    versions = {
        cast("str", row["name"]): Version(cast("str", row["version"]))
        for row in cast("list[dict[str, object]]", base["wheels"])
    }
    owned = set(cast("list[str]", base["replaces"]))
    if published_base is not None:
        owned.update(cast("list[str]", published_base["replaces"]))
    requirements: dict[str, list[Requirement]] = {}
    for path in wheels:
        try:
            name, version, _, _ = parse_wheel_filename(path.name)
            with ZipFile(path) as archive:
                entries = [
                    entry
                    for entry in archive.namelist()
                    if entry.endswith(".dist-info/METADATA") and len(entry.split("/")) == 2
                ]
                if len(entries) != 1:
                    raise EnvironmentError(
                        f"wheel must contain one distribution metadata: {path.name}"
                    )
                metadata = BytesParser().parsebytes(archive.read(entries[0]))
            if (
                canonicalize_name(str(metadata["Name"])) != name
                or Version(str(metadata["Version"])) != version
            ):
                raise EnvironmentError(f"wheel metadata disagrees with filename: {path.name}")
            requirements[name] = [Requirement(row) for row in metadata.get_all("Requires-Dist", [])]
        except (BadZipFile, ValueError) as exc:
            raise EnvironmentError(f"invalid wheel dependency metadata: {path.name}") from exc

    environment = cast("dict[str, str]", default_environment())
    python = Version(cast("str", base["python"]))
    environment.update(
        python_full_version=str(python),
        python_version=f"{python.major}.{python.minor}",
        implementation_version=str(python),
    )
    extras = {name: {""} for name in requirements}
    changed = True
    while changed:
        changed = False
        for name, dependencies in requirements.items():
            for requirement in dependencies:
                if requirement.marker is not None and not any(
                    requirement.marker.evaluate({**environment, "extra": extra})
                    for extra in extras[name]
                ):
                    continue
                dependency = canonicalize_name(requirement.name)
                if dependency in extras and not requirement.extras <= extras[dependency]:
                    extras[dependency].update(requirement.extras)
                    changed = True
                if dependency not in owned:
                    continue
                actual = versions.get(dependency)
                if (
                    actual is None
                    or requirement.url is not None
                    or not requirement.specifier.contains(actual, prereleases=True)
                ):
                    raise EnvironmentError(
                        f"{name} requires {requirement}; base provides {dependency} "
                        f"{actual if actual is not None else 'absent'}"
                    )


def effective_constraints(
    engine: str,
    base: dict[str, object],
    *,
    strict: bool = False,
    custom: bool = False,
    published_base: dict[str, object] | None = None,
    interfaces: set[str] | None = None,
) -> str:
    """Replace precisely the base producer's owned closure, including remove-only rows.

    The caller verifies manifest identity and non-base wheel requirements first.
    Published locks retain exact alignment. Explicit custom bases replace both
    closures and require at least the engine's applicable public torch version.
    Generic contributor exports constrain only computed dependency interfaces.
    All unrelated requirement markers and hashes are retained verbatim.
    """
    if custom and (strict or base.get("source") != "user-defined"):
        raise EnvironmentError("custom constraints require a user-defined base and strict=False")
    if published_base is not None and not custom:
        raise EnvironmentError("published_base replacement context requires custom=True")
    if interfaces is not None and (strict or custom):
        raise EnvironmentError(
            "generic interface constraints require strict=False and custom=False"
        )
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
    if published_base is not None:
        published_names = published_base.get("replaces")
        if (
            not isinstance(published_names, list)
            or not published_names
            or any(
                not isinstance(name, str) or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
                for name in cast("list[object]", published_names)
            )
        ):
            raise EnvironmentError("published base must provide normalized replacement membership")
        original_names = cast("list[str]", published_names)
        if original_names != sorted(set(original_names)):
            raise EnvironmentError("published base replacement names must be sorted and unique")
        owned.update(original_names)
    marker_environment = cast("dict[str, str]", default_environment())
    if custom or interfaces is not None:
        python = Version(cast("str", base["python"]))
        marker_environment.update(
            python_full_version=str(python),
            python_version=f"{python.major}.{python.minor}",
            implementation_version=str(python),
        )
    retained: list[str] = []
    locked: dict[str, set[Version]] = {}
    hashes: dict[str, set[str]] = {}
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
        hashes.setdefault(name, set()).update(_HASH.findall(line))
        try:
            version = Version(pins[0].version)
            if (
                requirement.marker is None
                or (interfaces is None and (not custom or name != "torch"))
                or requirement.marker.evaluate(marker_environment)
            ):
                locked.setdefault(name, set()).add(version)
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
        if name != canonicalize_name(name) or name not in replacement_names or name in selected:
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
            if custom and name == "torch":
                required = max(versions, default=None)
                if required is not None and Version(required.public) > Version(
                    parsed_version.public
                ):
                    raise EnvironmentError(
                        f"engine requires torch {required.public}; custom base provides torch "
                        f"{version}. Set torch in variants.toml to at least {required.public}, "
                        "use a new variant name, and rebuild with dinkster-env update "
                        "--variant NAME --variants PATH --yes"
                    )
            if (
                not custom
                and (interfaces is None or name in interfaces)
                and any(
                    candidate != parsed_version
                    and not (
                        not strict
                        and candidate.public == parsed_version.public
                        and (
                            interfaces is not None
                            or (name in {"torch", "torchvision"} and candidate.local is None)
                        )
                    )
                    for candidate in versions
                )
            ):
                raise EnvironmentError(f"base wheel {name} {version} disagrees with engine lock")
        except (InvalidVersion, InvalidWheelFilename) as exc:
            raise EnvironmentError(f"invalid base wheel identity: {name}") from exc
        sha = wheel.get("sha256")
        if not isinstance(sha, str):
            raise EnvironmentError(f"missing wheel digest: {name}")
        digest("sha256:" + sha)
        if strict and (not versions or hashes.get(name) != {sha}):
            raise EnvironmentError(f"base wheel {name} differs from the exact engine lock")
        additions.append(f"{name}=={version} --hash=sha256:{sha}")
    if custom and "torch" not in selected:
        raise EnvironmentError("custom base must provide a torch wheel")
    return "\n".join([*retained, *sorted(additions)]).strip() + "\n"
