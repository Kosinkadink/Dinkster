"""Hashed per-variant dependency locks and generation-layering plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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

    def __post_init__(self) -> None:
        if Path(self.filename).name != self.filename or "\\" in self.filename:
            raise ProvisionError("wheel filename must be a basename")
        parse_wheel_filename(self.filename)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or self.size < 0:
            raise ProvisionError(f"invalid wheel hash or size: {self.filename}")
        if urlparse(self.url).scheme not in {"https", "http", "file"}:
            raise ProvisionError(f"unsupported wheel URL: {self.filename}")


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


@dataclass(frozen=True)
class PackLock:
    pack: str
    variant: str
    requires_python: str
    requirements: tuple[str, ...]
    indexes: tuple[str, ...]
    packages: tuple[LockedDistribution, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.pack):
            raise ProvisionError("invalid pack identity")
        SpecifierSet(self.requires_python)
        names = [package.name for package in self.packages]
        if len(names) != len(set(names)):
            raise ProvisionError("duplicate package in dependency lock")
        for requirement in self.requirements:
            if normalize_requirement(requirement) != requirement:
                raise ProvisionError(f"unnormalized lock requirement: {requirement}")

    def write(self, path: Path) -> None:
        data = asdict(self)
        data["format"] = "dinkster.pack-lock/1"
        data["requiresPython"] = data.pop("requires_python")
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="ascii")

    @classmethod
    def read(cls, path: Path) -> PackLock:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("dependency lock must be an object")
            data = cast("dict[str, Any]", raw)
            if data.pop("format") != "dinkster.pack-lock/1":
                raise ValueError("unsupported dependency lock format")
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProvisionError(f"invalid dependency lock {path}: {exc}") from exc


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
        result = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
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
    """Subtract shared distributions only after checking exact version agreement."""
    conflicts: list[str] = []
    wheels: list[LockedWheel] = []
    if lock.variant != variant:
        conflicts.append(f"variant {lock.variant} does not match {variant}")
    if not SpecifierSet(lock.requires_python).contains(generation.python_version):
        conflicts.append(
            f"Python {generation.python_version} does not satisfy {lock.requires_python}"
        )
    versions = {package.name: package.version for package in lock.packages}
    for text in lock.requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(dict(generation.markers)):
            continue
        name = canonicalize_name(requirement.name)
        version = versions.get(name)
        if version is None or not requirement.specifier.contains(version):
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
    uv: str = "uv",
) -> PackLock:
    """Publisher/local-only resolution; published installation never calls this."""
    normalized = tuple(normalize_requirement(text) for text in requirements)
    with tempfile.TemporaryDirectory(prefix="dinkster-pack-lock-") as directory:
        root = Path(directory)
        source = root / "requirements.in"
        output = root / "pylock.toml"
        source.write_text("\n".join(normalized) + "\n", encoding="ascii")
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
        for path in find_links:
            command.extend(["--find-links", str(path)])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise ProvisionError(f"pack {pack} dependency conflict: {result.stderr.strip()}")
        data = tomllib.loads(output.read_text(encoding="utf-8"))
        packages: list[LockedDistribution] = []
        for item in cast("list[dict[str, Any]]", data.get("packages", [])):
            candidates: list[LockedWheel] = []
            for wheel in item.get("wheels", []):
                url = wheel.get("url")
                if url is None:
                    path = Path(wheel["path"])
                    if not path.is_absolute():
                        path = output.parent / path
                    url = path.resolve().as_uri()
                size = wheel.get("size")
                digest = wheel.get("hashes", {}).get("sha256")
                if size is None or digest is None:
                    # uv omits integrity metadata for local find-links artifacts.
                    with urllib.request.urlopen(url) as stream:
                        content = stream.read()
                    measured = hashlib.sha256(content).hexdigest()
                    if digest is not None and digest != measured:
                        raise ProvisionError(f"resolved wheel hash mismatch: {url}")
                    size, digest = len(content), measured
                candidates.append(
                    LockedWheel(
                        unquote(urlparse(url).path.rsplit("/", 1)[-1]),
                        size,
                        digest,
                        url,
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
    return PackLock(pack, variant, requires_python, normalized, tuple(indexes), tuple(packages))


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
        uv=args.uv,
    )
    lock.write(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
