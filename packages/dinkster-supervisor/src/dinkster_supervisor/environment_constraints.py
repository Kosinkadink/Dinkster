"""Assemble a variant's hashed requirements without resolving dependencies."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from email.parser import BytesParser
from pathlib import Path
from typing import cast
from zipfile import BadZipFile, ZipFile

import packaging
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .environment_state import EnvironmentError, digest, relative_file

_HASH = re.compile(r"(?<!\S)--hash=sha256:([0-9a-f]{64})(?=\s|$)")


def requirement_hashes(requirements: str) -> frozenset[str]:
    """Keep every hashed alternative, including other-platform marker rows."""
    return frozenset(_HASH.findall(requirements.replace("\\\n", "").replace("\\\r\n", "")))


def wheel_compatibility(python: Path) -> dict[str, object]:
    """Inspect target compatibility without loading its installed distributions."""
    script = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "from packaging.tags import sys_tags; "
        "from packaging.markers import default_environment; "
        "print(json.dumps({'tags': [str(tag) for tag in sys_tags()], "
        "'markers': default_environment()}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-B", "-S", "-c", script, str(Path(packaging.__file__).parent.parent)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise EnvironmentError(f"cannot inspect wheel compatibility: {result.stderr.strip()}")
    return cast("dict[str, object]", json.loads(result.stdout))


def selected_wheel_rows(
    requirements: str, wheels: list[dict[str, object]], compatibility: dict[str, object]
) -> list[dict[str, object]]:
    """Select the same immutable wheel identities for preview and construction."""
    tags = {tag: rank for rank, tag in enumerate(cast("list[str]", compatibility["tags"]))}
    candidates: dict[str, list[tuple[int, str, str, Version, dict[str, object]]]] = {}
    for row in wheels:
        filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
        try:
            name, version, _, wheel_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        ranks = [tags[str(tag)] for tag in wheel_tags if str(tag) in tags]
        if ranks:
            candidates.setdefault(name, []).append((min(ranks), filename, sha, version, row))
    selected: list[dict[str, object]] = []
    for line in requirements.replace("\\\r\n", "").replace("\\\n", "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            requirement = Requirement(_HASH.sub("", line).strip())
        except InvalidRequirement as exc:
            raise EnvironmentError("engine constraints contain an invalid requirement") from exc
        if requirement.marker is not None and not requirement.marker.evaluate(
            cast("dict[str, str]", compatibility["markers"])
        ):
            continue
        versions = list(requirement.specifier)
        hashes = set(_HASH.findall(line))
        if (
            requirement.url is not None
            or len(versions) != 1
            or versions[0].operator != "=="
            or "*" in versions[0].version
            or not hashes
        ):
            raise EnvironmentError("engine requirements must use exact version pins and hashes")
        name = canonicalize_name(requirement.name)
        for _, filename, sha, version, row in sorted(
            candidates.get(name, ()), key=lambda candidate: candidate[:3]
        ):
            if version not in requirement.specifier or sha not in hashes:
                continue
            selected.append({"filename": filename, "size": row["size"], "sha256": sha})
            break
        else:
            raise EnvironmentError(f"offline generation wheel missing or corrupt: {requirement}")
    return sorted(selected, key=lambda row: cast("str", row["filename"]))


def selected_wheel_requirements(
    python: Path, requirements: str, directories: list[Path]
) -> tuple[str, list[dict[str, object]]]:
    """Choose verified compatible bytes, then pin sync to those exact local URLs."""
    paths: dict[tuple[str, str], Path] = {}
    rows: list[dict[str, object]] = []
    for path in sorted({path for directory in directories for path in directory.glob("*.whl")}):
        sha = path.parent.name
        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
        identity = (path.name, sha)
        if identity not in paths:
            paths[identity] = path
            rows.append({"filename": path.name, "size": path.stat().st_size, "sha256": sha})
    selected = selected_wheel_rows(requirements, rows, wheel_compatibility(python))
    pins: list[str] = []
    for row in selected:
        filename, sha = cast("str", row["filename"]), cast("str", row["sha256"])
        path = paths[filename, sha]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                raise EnvironmentError(f"offline generation wheel missing or corrupt: {filename}")
        name = parse_wheel_filename(filename)[0]
        pins.append(f"{name} @ {path.resolve().as_uri()} --hash=sha256:{sha}")
    return "\n".join(pins) + "\n", selected


def installed_wheel_origins(
    wheels: list[dict[str, object]],
    base: dict[str, object],
    engine: dict[str, object],
    variant: str,
) -> list[dict[str, object]]:
    """Bind selected bytes to manifest origins, never to their acquisition cache paths."""
    origins: dict[tuple[object, object, object], object] = {}
    for row in cast("list[dict[str, object]]", base.get("wheels", [])):
        origins[row["filename"], row["size"], row["sha256"]] = row.get("index")
    selections = cast("dict[str, dict[str, object]]", engine.get("dependencies", {}))
    for row in cast("list[dict[str, object]]", selections.get(variant, {}).get("wheels", [])):
        origins[row["filename"], row["size"], row["sha256"]] = row.get("index")
    repository = engine.get("repository")
    issuer = f"https://github.com/{repository}" if repository else None
    for row in cast("list[dict[str, object]]", engine.get("artifacts", [])):
        if str(row["name"]).endswith(".whl"):
            origins[row["name"], row["size"], row["sha256"]] = issuer
    return [
        {**row, "registry": origins.get((row["filename"], row["size"], row["sha256"]))}
        for row in wheels
    ]


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
