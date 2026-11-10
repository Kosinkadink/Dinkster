"""Build official registry source artifacts and their existing artifact lock."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from dinkster_registry import LockedPack, Lockfile, artifact_digest
from dinkster_registry.archive_admission import extracted_archive
from dinkster_registry.model import validate_version
from dinkster_schema import canonical_claim
from dinkster_values.names import validate_name
from dinkster_workers import load_manifest
from dinkster_workers.doctor import DoctorReport, diagnose_static
from dinkster_workers.pack_lock import resolve_pack_lock
from dinkster_workers.pack_wheel import build_pack_archive, wheel_filename
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_release import is_official_library


def official_library(project: dict) -> tuple[str, str] | None:
    """Only explicitly opted-in distributions enter first-party publication."""
    if not is_official_library(project):
        return None
    metadata = project["project"]
    name, version = metadata["name"], metadata["version"]
    if validate_name(name) or canonicalize_name(name) != name or validate_version(version):
        raise ValueError("official library requires a canonical name and public version")
    return name, version


def library_metadata(archive: Path) -> tuple[str, str, tuple[str, ...]]:
    """Read library identity without importing code or treating it as a node pack."""
    with zipfile.ZipFile(archive) as wheel:
        names = wheel.namelist()
        if any(PurePosixPath(name).name == "dinkster-pack.toml" for name in names):
            raise ValueError("official library wheel must not contain a node manifest")
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("official library wheel requires one distribution")
        fields = email.parser.BytesParser().parsebytes(wheel.read(metadata[0]))
        return fields["Name"], fields["Version"], tuple(fields.get_all("Requires-Dist", []))


def is_official_pack_manifest(path: PurePosixPath) -> bool:
    """Recognize package manifests, excluding scaffolding and engine-carried schemas."""
    return path.name == "dinkster-pack.toml" and (
        len(path.parts) == 3
        and path.parts[0] == "packages"
        and path.parts[1].startswith("dinkster-nodes-")
        and path.parts[1] not in {"dinkster-nodes-dev", "dinkster-nodes-generation"}
        or len(path.parts) == 4
        and path.parent.parent == PurePosixPath("packages/dinkster-nodes-vision")
        and path.parent.name.endswith("_pack")
    )


def build(
    root: Path,
    output: Path,
    *,
    version: str | None = None,
    packs: Sequence[str] = (),
    variant: str | None = None,
    python: Path | None = None,
    constraints: Path | None = None,
    indexes: Sequence[str] = (),
    find_links: Sequence[Path] = (),
    accelerator: str | None = None,
    uv: str = "uv",
) -> tuple[Path, ...]:
    """Build only committed pack sources, without importing node implementations."""
    if version is not None:
        problem = validate_version(version)
        if problem is not None:
            raise ValueError(f"pack version {version!r} {problem}")
    if variant is not None:
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None:
            raise ValueError("variant must be a portable variant name")
        if python is None or constraints is None:
            raise ValueError("dependency locks require variant, python and constraints together")
    elif python is not None or constraints is not None or indexes or find_links or accelerator:
        raise ValueError("dependency locks require variant, python and constraints together")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    # git archive applies checkout conversion even though its input is a commit.
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "archive",
            "--format=tar",
            commit,
        ]
    )
    payloads: list[Path] = []
    locked: list[LockedPack] = []
    lock_inputs: list[tuple[str, str, tuple[str, ...], str, str | None]] = []
    with tempfile.TemporaryDirectory(prefix="dinkster-official-packs-") as directory:
        staged = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(tracked)) as archive:
            archive.extractall(staged, filter="data")
        packages = staged / "packages"
        roots = [
            manifest.parent
            for manifest in sorted(packages.glob("*/dinkster-pack.toml"))
            if is_official_pack_manifest(PurePosixPath(manifest.relative_to(staged).as_posix()))
        ]
        vision = packages / "dinkster-nodes-vision"
        for manifest in sorted(vision.glob("*_pack/dinkster-pack.toml")):
            pack_root = manifest.parent
            entry = load_manifest(manifest).nodes_entry.partition(":")[0]
            namespace, _, module = entry.partition(".")
            destination = pack_root / namespace / module.replace(".", "/")
            shutil.copytree(vision / "src" / namespace / module.replace(".", "/"), destination)
            roots.append(pack_root)
        manifests = [(path, load_manifest(path / "dinkster-pack.toml")) for path in roots]
        libraries = []
        requirements_by_name = {
            manifest.name: manifest.requires_for(accelerator) if accelerator else manifest.requires
            for _, manifest in manifests
        }
        for path in sorted(packages.glob("*/pyproject.toml")):
            project = tomllib.loads(path.read_text(encoding="utf-8"))
            if identity := official_library(project):
                if (path.parent / "dinkster-pack.toml").exists():
                    raise ValueError("official library must not declare a node manifest")
                libraries.append((path.parent, identity))
                requirements_by_name[identity[0]] = tuple(
                    project["project"].get("dependencies", [])
                )
        selected = set(packs)
        pack_names = {manifest.name for _, manifest in manifests}
        library_names = {identity[0] for _, identity in libraries}
        if pack_names & library_names or len(library_names) != len(libraries):
            raise ValueError("duplicate official library identity")
        unknown = selected - pack_names - library_names
        if unknown:
            raise ValueError(f"unknown official packs: {', '.join(sorted(unknown))}")
        pending = list(selected)
        while pending:
            for value in requirements_by_name[pending.pop()]:
                dependency = canonicalize_name(Requirement(value).name)
                if dependency in library_names and dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        output.mkdir(parents=True, exist_ok=True)
        for library_root, (name, library_version) in libraries:
            if selected and name not in selected:
                continue
            if version is not None and version != library_version:
                raise ValueError("official library version cannot be overridden")
            payload = output / f"{name.replace('-', '_')}-{library_version}-py3-none-any.whl"
            library_output = staged / "library-wheels" / name
            subprocess.run(
                [
                    uv,
                    "build",
                    "--wheel",
                    "--no-sources",
                    "--out-dir",
                    str(library_output),
                    str(library_root),
                ],
                check=True,
                env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"},
            )
            # Match pack wheel containers across hosts and compression libraries.
            # Member bytes, including RECORD hashes, remain the backend's output.
            with (
                zipfile.ZipFile(library_output / payload.name) as source,
                zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as target,
            ):
                for member in sorted(source.namelist()):
                    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o644 << 16
                    target.writestr(info, source.read(member))
            built_name, built_version, requirements = library_metadata(payload)
            if (built_name, built_version) != (name, library_version):
                raise ValueError("official library wheel identity mismatch")
            for requirement in requirements:
                if canonicalize_name(Requirement(requirement).name) in pack_names:
                    raise ValueError("wheel.requires-pack: library requires a node pack")
            locked.append(
                LockedPack(
                    name,
                    library_version,
                    artifact_digest(payload.read_bytes()),
                    "dinkster",
                    (),
                    source=f"git:Kosinkadink/Dinkster@{commit}",
                )
            )
            payloads.append(payload)
            declaration = tomllib.loads((library_root / "pyproject.toml").read_text())
            lock_inputs.append(
                (
                    name,
                    library_version,
                    requirements,
                    declaration["project"]["requires-python"],
                    declaration["tool"]["dinkster"].get("engine"),
                )
            )
        for pack_root, manifest in manifests:
            if selected and manifest.name not in selected:
                continue
            if version is None and manifest.compatibility is None:
                raise ValueError(
                    f"official pack {manifest.name!r} must declare its version and engine range"
                )
            pack_version = version
            if pack_version is None:
                assert manifest.compatibility is not None
                pack_version = manifest.compatibility.version
            manifest_path = pack_root / "dinkster-pack.toml"
            project_path = (
                vision / "pyproject.toml"
                if pack_root.parent == vision
                else pack_root / "pyproject.toml"
            )
            project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
            manifest = load_manifest(manifest_path)
            if manifest.compatibility is None or pack_version != manifest.compatibility.version:
                raise ValueError("wheel version must equal the pack manifest version")
            if manifest.publisher != "dinkster":
                raise ValueError(f"official pack {manifest.name!r} must declare publisher dinkster")
            tests = pack_root / "tests"
            if tests.is_dir():
                shutil.rmtree(tests)
            payload = output / wheel_filename(manifest)
            build_pack_archive(pack_root, payload)
            digest = artifact_digest(payload.read_bytes())
            with extracted_archive(payload.read_bytes()) as artifact_root:
                artifact_manifest = load_manifest(artifact_root / "dinkster-pack.toml")
                static = diagnose_static(artifact_root, artifact_manifest)
                report = DoctorReport(
                    pack_name=artifact_manifest.name,
                    manifest_path=str(artifact_manifest.path),
                    findings=static.findings,
                )
                print(report.to_json())
                if not report.ok:
                    raise ValueError(f"official pack {manifest.name!r} failed static doctor")
            locked.append(
                LockedPack(
                    pack=manifest.name,
                    version=pack_version,
                    artifact_digest=digest,
                    publisher=manifest.publisher,
                    claims=tuple(canonical_claim(claim) for claim in manifest.namespaces),
                    source=f"git:Kosinkadink/Dinkster@{commit}",
                )
            )
            payloads.append(payload)
            requirements = manifest.requires_for(accelerator) if accelerator else manifest.requires
            for requirement in requirements:
                if canonicalize_name(Requirement(requirement).name) in pack_names:
                    raise ValueError("wheel.requires-pack: Requires-Dist names a node pack")
            lock_inputs.append(
                (
                    manifest.name,
                    pack_version,
                    requirements,
                    project.get("requires-python", ">=3.12"),
                    manifest.compatibility.engine,
                )
            )
        if variant is not None:
            assert python is not None and constraints is not None
            library_wheels = {
                path.name: path
                for path in payloads
                if path.name.split("-")[0].replace("_", "-") in library_names
            }
            if library_wheels:
                pinned = staged / "library-constraints.txt"
                pinned.write_text(
                    constraints.read_text()
                    + "\n"
                    + "\n".join(
                        f"{pack.pack}=={pack.version}"
                        for pack in locked
                        if pack.pack in library_names
                    ),
                    encoding="utf-8",
                )
                constraints = pinned
            for name, pack_version, requirements, requires_python, engine in lock_inputs:
                dependency_lock = resolve_pack_lock(
                    pack=name,
                    variant=variant,
                    requirements=requirements,
                    python=python,
                    constraints=constraints,
                    indexes=indexes,
                    find_links=(*find_links, output) if library_wheels else find_links,
                    requires_python=requires_python,
                    engine=engine,
                    uv=uv,
                )
                distributions = []
                for distribution in dependency_lock.packages:
                    wheels = []
                    for wheel in distribution.wheels:
                        if wheel.filename in library_wheels:
                            payload = library_wheels[wheel.filename].read_bytes()
                            if wheel.sha256 != hashlib.sha256(
                                payload
                            ).hexdigest() or wheel.size != len(payload):
                                raise ValueError("resolved library differs from publication bytes")
                            wheel = replace(
                                wheel,
                                url=(
                                    f"https://registry.dinkster.org/static/wheels/{wheel.sha256}/"
                                    f"{wheel.filename}"
                                ),
                            )
                        wheels.append(wheel)
                    distributions.append(replace(distribution, wheels=tuple(wheels)))
                dependency_lock = replace(dependency_lock, packages=tuple(distributions))
                dependency_payload = output / f"{name}-{pack_version}-{variant}.pack-lock.json"
                dependency_lock.write(dependency_payload)
                payloads.append(dependency_payload)
    lock = output / "official-packs.lock"
    lock.write_text(Lockfile.of(locked).record_json() + "\n", encoding="utf-8", newline="\n")
    return (*payloads, lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Override manifest versions for legacy release builds")
    parser.add_argument("--pack", action="append", default=[], help="Exact official pack id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--generation-python", type=Path)
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--find-links", type=Path, action="append", default=[])
    parser.add_argument(
        "--find-links-stdin", action="store_true", help="Read JSON wheel directories from stdin"
    )
    parser.add_argument("--accelerator")
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    if args.find_links_stdin:
        links = json.load(sys.stdin)
        if not isinstance(links, list) or any(not isinstance(path, str) for path in links):
            parser.error("stdin must contain a JSON array of wheel directories")
        args.find_links.extend(Path(path) for path in links)
    for payload in build(
        Path(__file__).resolve().parents[1],
        args.output.resolve(),
        version=args.version,
        packs=args.pack,
        variant=args.variant,
        python=args.generation_python,
        constraints=args.constraints,
        indexes=args.index,
        find_links=args.find_links,
        accelerator=args.accelerator,
        uv=args.uv,
    ):
        print(payload)
