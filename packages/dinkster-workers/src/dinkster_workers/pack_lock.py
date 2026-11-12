"""Hashed per-variant dependency locks and generation-layering plans."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .manifest import load_manifest
from .provision import ProvisionError
from .wheel_download import download_wheel

OPENCV_ALIASES = {
    "opencv-python": "opencv-contrib-python-headless",
    "opencv-python-headless": "opencv-contrib-python-headless",
    "opencv-contrib-python": "opencv-contrib-python-headless",
    "opencv-contrib-python-headless": "opencv-contrib-python-headless",
}


def normalize_requirement(text: str) -> str:
    requirement = Requirement(text)
    name = canonicalize_name(requirement.name)
    if name in OPENCV_ALIASES:
        requirement.name = OPENCV_ALIASES[name]
        requirement.extras.discard("ffmpeg")
    return str(requirement)


@dataclass(frozen=True)
class LockedWheel:
    filename: str
    size: int
    sha256: str
    url: str
    registry: str | None = None

    def __post_init__(self) -> None:
        if Path(self.filename).name != self.filename or "\\" in self.filename:
            raise ProvisionError("wheel filename must be a basename")
        parse_wheel_filename(self.filename)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or self.size < 0:
            raise ProvisionError(f"invalid wheel hash or size: {self.filename}")
        if urlparse(self.url).scheme not in {"https", "http", "file"}:
            raise ProvisionError(f"unsupported wheel URL: {self.filename}")
        if self.registry is not None:
            issuer = urlparse(self.registry)
            if (
                issuer.scheme not in {"https", "http"}
                or not issuer.hostname
                or issuer.username
                or issuer.password
                or issuer.query
                or issuer.fragment
            ):
                raise ProvisionError(f"invalid wheel issuing registry: {self.filename}")


@dataclass(frozen=True)
class LockedDistribution:
    name: str
    version: str
    wheels: tuple[LockedWheel, ...]

    def __post_init__(self) -> None:
        if canonicalize_name(self.name) != self.name or self.name in OPENCV_ALIASES.keys() - {
            "opencv-contrib-python-headless"
        }:
            raise ProvisionError(f"noncanonical lock package: {self.name}")
        version = Version(self.version)
        for wheel in self.wheels:
            wheel_name, wheel_version, _, _ = parse_wheel_filename(wheel.filename)
            if wheel_name != self.name or wheel_version != version:
                raise ProvisionError(f"wheel identity disagrees with {self.name}=={self.version}")


@lru_cache(maxsize=512)
def _wheel_requirements(wheel: LockedWheel) -> tuple[str, ...]:
    """Read dependency metadata from verified bytes, once per publication artifact."""
    with tempfile.TemporaryDirectory(prefix="dinkster-wheel-metadata-") as directory:
        path = Path(directory) / wheel.filename
        download_wheel(wheel.url, path, max_size=wheel.size)
        with path.open("rb") as source:
            if (
                path.stat().st_size != wheel.size
                or hashlib.file_digest(source, "sha256").hexdigest() != wheel.sha256
            ):
                raise ProvisionError(f"resolved wheel hash mismatch: {wheel.url}")
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                names = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA") and len(name.split("/")) == 2
                ]
                if len(names) != 1:
                    raise ProvisionError(
                        f"wheel requires one distribution metadata: {wheel.filename}"
                    )
                metadata = email.message_from_bytes(archive.read(names[0]))
        return tuple(metadata.get_all("Requires-Dist", []))


@dataclass(frozen=True)
class PackLock:
    pack: str
    variant: str
    requires_python: str
    requirements: tuple[str, ...]
    indexes: tuple[str, ...]
    packages: tuple[LockedDistribution, ...]
    engine: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.pack):
            raise ProvisionError("invalid pack identity")
        SpecifierSet(self.requires_python)
        if self.engine is not None:
            if not self.engine:
                raise ProvisionError("engine API requires a nonempty compatibility range")
            SpecifierSet(self.engine)
            if any(package.name == "dinkster-api" for package in self.packages):
                raise ProvisionError("engine API must not also be an exact lock package")
        names = [package.name for package in self.packages]
        if len(names) != len(set(names)):
            raise ProvisionError("duplicate package in dependency lock")
        for requirement in self.requirements:
            if normalize_requirement(requirement) != requirement:
                raise ProvisionError(f"unnormalized lock requirement: {requirement}")
            parsed = Requirement(requirement)
            if self.engine is not None and canonicalize_name(parsed.name) == "dinkster-api":
                if parsed.url or parsed.extras or parsed.marker:
                    raise ProvisionError(
                        "engine API requirements must be unconditional version ranges"
                    )

    def write(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="ascii")

    def to_json(self) -> str:
        data = asdict(self)
        data["format"] = (
            "dinkster.pack-lock/2" if self.engine is not None else "dinkster.pack-lock/1"
        )
        if self.engine is None:
            data.pop("engine")
        data["requiresPython"] = data.pop("requires_python")
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    @classmethod
    def read(cls, path: Path) -> PackLock:
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except (ProvisionError, UnicodeError) as exc:
            raise ProvisionError(f"{path}: {exc}") from exc

    @classmethod
    def from_json(cls, payload: str | bytes) -> PackLock:
        try:
            raw = json.loads(payload)
            if not isinstance(raw, dict):
                raise ValueError("dependency lock must be an object")
            data = cast("dict[str, Any]", raw)
            format = data.pop("format")
            if format not in {"dinkster.pack-lock/1", "dinkster.pack-lock/2"}:
                raise ValueError("unsupported dependency lock format")
            engine = data["engine"] if format == "dinkster.pack-lock/2" else None
            if format == "dinkster.pack-lock/2" and not isinstance(engine, str):
                raise ValueError("engine compatibility range must be a string")
            return cls(
                pack=data["pack"],
                variant=data["variant"],
                requires_python=data["requiresPython"],
                requirements=tuple(data["requirements"]),
                indexes=tuple(data["indexes"]),
                packages=tuple(
                    LockedDistribution(
                        item["name"],
                        item["version"],
                        tuple(LockedWheel(**wheel) for wheel in item["wheels"]),
                    )
                    for item in data["packages"]
                ),
                engine=engine,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProvisionError(f"invalid dependency lock: {exc}") from exc


@dataclass(frozen=True)
class GenerationPackages:
    """Facts read from the interpreter that will parent a pack venv."""

    versions: Mapping[str, str]
    python_version: str
    tags: frozenset[Tag]
    markers: Mapping[str, str]

    @classmethod
    def inspect(cls, python: Path) -> GenerationPackages:
        script = """
import importlib.metadata as m, json, platform
from packaging.tags import sys_tags
from packaging.markers import default_environment
print(json.dumps({"versions": {d.metadata["Name"]: d.version for d in m.distributions()
                              if d.metadata["Name"]},
                  "python": platform.python_version(),
                  "tags": [str(t) for t in sys_tags()], "markers": default_environment()}))
"""
        try:
            result = subprocess.run(
                [str(python), "-I", "-B", "-c", script], capture_output=True, text=True
            )
        except OSError as exc:
            raise ProvisionError(f"cannot inspect generation interpreter {python}: {exc}") from exc
        if result.returncode:
            raise ProvisionError(f"cannot inspect generation interpreter: {result.stderr.strip()}")
        data = json.loads(result.stdout)
        return cls(
            {canonicalize_name(k): v for k, v in data["versions"].items()},
            data["python"],
            frozenset(Tag(*tag.split("-")) for tag in data["tags"]),
            data["markers"],
        )


@dataclass(frozen=True)
class PackPlan:
    lock: PackLock
    wheels: tuple[LockedWheel, ...]
    conflicts: tuple[str, ...]
    own_environment_size: int

    @property
    def download_size(self) -> int:
        return sum(wheel.size for wheel in self.wheels)

    def require_layering(self) -> None:
        if self.conflicts:
            raise ProvisionError(
                f"pack {self.lock.pack}: {'; '.join(self.conflicts)}. "
                f"Use an own environment ({self.own_environment_size} bytes)."
            )


def plan_pack(
    lock: PackLock,
    generation: GenerationPackages,
    *,
    variant: str,
    own_environment_size: int,
) -> PackPlan:
    """Check engine compatibility and exact agreement for shared dependencies."""
    conflicts: list[str] = []
    wheels: list[LockedWheel] = []
    if lock.variant != variant:
        conflicts.append(f"variant {lock.variant} does not match {variant}")
    if not SpecifierSet(lock.requires_python).contains(generation.python_version):
        conflicts.append(
            f"Python {generation.python_version} does not satisfy {lock.requires_python}"
        )
    versions = {package.name: package.version for package in lock.packages}
    if lock.engine is not None:
        api_version = generation.versions.get("dinkster-api")
        if api_version is None or not SpecifierSet(lock.engine).contains(
            api_version, prereleases=True
        ):
            conflicts.append(
                f"dinkster-api: generation has {api_version or 'absent'}, "
                f"engine needs {lock.engine}"
            )
        if api_version is not None:
            versions["dinkster-api"] = api_version
    for text in lock.requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(dict(generation.markers)):
            continue
        name = canonicalize_name(requirement.name)
        version = versions.get(name)
        if version is None or not requirement.specifier.contains(
            version,
            prereleases=True if lock.engine is not None and name == "dinkster-api" else None,
        ):
            conflicts.append(f"{text} is not satisfied by the pack lock")
    for package in lock.packages:
        base_version = generation.versions.get(package.name)
        if base_version is not None:
            if Version(base_version) != Version(package.version):
                conflicts.append(
                    f"{package.name}: generation has {base_version}, lock needs {package.version}"
                )
            continue
        compatible = [
            wheel
            for wheel in package.wheels
            if parse_wheel_filename(wheel.filename)[3] & generation.tags
        ]
        if not compatible:
            conflicts.append(
                f"{package.name}=={package.version}: no compatible wheel for {variant}"
            )
        else:
            wheels.append(compatible[0])
    return PackPlan(lock, tuple(wheels), tuple(conflicts), own_environment_size)


def resolve_pack_lock(
    *,
    pack: str,
    variant: str,
    requirements: Sequence[str],
    python: Path,
    constraints: Path,
    indexes: Sequence[str] = (),
    find_links: Sequence[Path] = (),
    requires_python: str = ">=3.12",
    engine: str | None = None,
    uv: str = "uv",
) -> PackLock:
    """Publisher/local-only resolution; published installation never calls this."""
    normalized = tuple(normalize_requirement(text) for text in requirements)
    with tempfile.TemporaryDirectory(prefix="dinkster-pack-lock-") as directory:
        root = Path(directory)
        source = root / "requirements.in"
        output = root / "pylock.toml"
        sources = "".join(f"--find-links {path.resolve().as_uri()}\n" for path in find_links)
        source.write_text(sources + "\n".join(normalized) + "\n", encoding="ascii")
        command = [
            uv,
            "pip",
            "compile",
            str(source),
            "--python",
            str(python),
            "--constraints",
            str(constraints),
            "--only-binary",
            ":all:",
            "--format",
            "pylock.toml",
            "--output-file",
            str(output),
            "--no-header",
        ]
        if indexes:
            command.extend(["--default-index", indexes[0]])
            for index in indexes[1:]:
                command.extend(["--index", index])
        elif find_links:
            command.append("--no-index")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise ProvisionError(f"pack {pack} dependency conflict: {result.stderr.strip()}")
        data = tomllib.loads(output.read_text(encoding="utf-8"))
        packages: list[LockedDistribution] = []
        for item in cast("list[dict[str, Any]]", data.get("packages", [])):
            candidates: list[LockedWheel] = []
            archives: list[dict[str, Any]] = []
            if "archive" in item:
                archive = cast("dict[str, Any]", item["archive"])
                location = cast("str", archive.get("url", archive.get("path", "")))
                if ("path" in archive and Path(archive["path"]).suffix == ".whl") or (
                    urlparse(location).scheme == "file" and urlparse(location).path.endswith(".whl")
                ):
                    archives.append(archive)
            for wheel in [*item.get("wheels", []), *archives]:
                url = wheel.get("url")
                if url is None:
                    path = Path(wheel["path"])
                    if not path.is_absolute():
                        path = output.parent / path
                    url = path.resolve().as_uri()
                size = wheel.get("size")
                digest = wheel.get("hashes", {}).get("sha256")
                if size is None or digest is None or wheel in archives:
                    # uv omits integrity metadata for local find-links artifacts.
                    downloaded = root / "resolved-wheel"
                    try:
                        download_wheel(url, downloaded, max_size=size)
                    except ProvisionError as exc:
                        raise ProvisionError(f"pack {pack}: {exc}") from exc
                    content = downloaded.read_bytes()
                    measured = hashlib.sha256(content).hexdigest()
                    if digest is not None and digest != measured:
                        raise ProvisionError(f"resolved wheel hash mismatch: {url}")
                    if size is not None and size != len(content):
                        raise ProvisionError(f"resolved wheel size mismatch: {url}")
                    size, digest = len(content), measured
                candidates.append(
                    LockedWheel(
                        unquote(urlparse(url).path.rsplit("/", 1)[-1]),
                        size,
                        digest,
                        url,
                        item.get("index"),
                    )
                )
            if not candidates:
                raise ProvisionError(
                    f"pack {pack}: {item['name']}=={item['version']}: "
                    "source-only closure has no wheel candidates"
                )
            packages.append(
                LockedDistribution(
                    canonicalize_name(item["name"]),
                    item["version"],
                    tuple(candidates),
                )
            )
    if engine is not None and any(package.name == "dinkster-api" for package in packages):
        # Preserve transitive API constraints when replacing the resolved build
        # with the generation's API. Other dependencies stay exactly locked.
        api = next(package for package in packages if package.name == "dinkster-api")
        if not SpecifierSet(engine).contains(api.version, prereleases=True):
            raise ProvisionError(
                f"resolved dinkster-api {api.version} does not satisfy engine {engine}"
            )
        api_requirements: set[str] = set()
        for package in packages:
            if package.name == "dinkster-api":
                continue
            for wheel in package.wheels:
                try:
                    wheel_requirements = _wheel_requirements(wheel)
                except ProvisionError as exc:
                    raise ProvisionError(f"pack {pack}: {exc}") from exc
                for text in wheel_requirements:
                    if canonicalize_name(Requirement(text).name) == "dinkster-api":
                        api_requirements.add(normalize_requirement(text))
        normalized = tuple(dict.fromkeys((*normalized, *sorted(api_requirements))))
        packages = [package for package in packages if package.name != "dinkster-api"]
    else:
        engine = None
    return PackLock(
        pack, variant, requires_python, normalized, tuple(indexes), tuple(packages), engine
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a publisher or local checkout once, not during published installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_root", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--generation-python", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--find-links", type=Path, action="append", default=[])
    parser.add_argument("--requires-python", default=">=3.12")
    parser.add_argument("--accelerator")
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.pack_root / "dinkster-pack.toml")
    lock = resolve_pack_lock(
        pack=manifest.name,
        variant=args.variant,
        requirements=(
            manifest.requires_for(args.accelerator) if args.accelerator else manifest.requires
        ),
        python=args.generation_python,
        constraints=args.constraints,
        indexes=args.index,
        find_links=args.find_links,
        requires_python=args.requires_python,
        engine=manifest.compatibility.engine if manifest.compatibility else None,
        uv=args.uv,
    )
    lock.write(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
