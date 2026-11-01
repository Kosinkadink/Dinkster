"""Build official registry source artifacts and their existing artifact lock."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path

from dinkster_registry import LockedPack, Lockfile, build_artifact, unpack_artifact
from dinkster_registry.model import validate_version
from dinkster_schema import canonical_claim
from dinkster_workers import load_manifest
from dinkster_workers.doctor import DoctorReport, diagnose_static
from dinkster_workers.pack_lock import resolve_pack_lock


def build(
    root: Path,
    output: Path,
    *,
    version: str,
    variant: str | None = None,
    python: Path | None = None,
    constraints: Path | None = None,
    indexes: Sequence[str] = (),
    find_links: Sequence[Path] = (),
    accelerator: str | None = None,
    uv: str = "uv",
) -> tuple[Path, ...]:
    """Build only committed pack sources, without importing node implementations."""
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
    output.mkdir(parents=True, exist_ok=True)
    payloads: list[Path] = []
    locked: list[LockedPack] = []
    with tempfile.TemporaryDirectory(prefix="dinkster-official-packs-") as directory:
        staged = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(tracked)) as archive:
            archive.extractall(staged, filter="data")
        packages = staged / "packages"
        roots = [
            packages / f"dinkster-nodes-{family}"
            for family in ("comfy", "foundation", "image", "media-io", "remote")
        ]
        vision = packages / "dinkster-nodes-vision"
        for manifest in sorted(vision.glob("*_pack/dinkster-pack.toml")):
            pack_root = manifest.parent
            entry = load_manifest(manifest).nodes_entry.partition(":")[0]
            namespace, _, module = entry.partition(".")
            destination = pack_root / namespace / module.replace(".", "/")
            shutil.copytree(vision / "src" / namespace / module.replace(".", "/"), destination)
            roots.append(pack_root)
        for pack_root in roots:
            manifest_path = pack_root / "dinkster-pack.toml"
            manifest = load_manifest(manifest_path)
            project_path = (
                vision / "pyproject.toml"
                if pack_root.parent == vision
                else pack_root / "pyproject.toml"
            )
            project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
            source = pack_root / "src"
            if source.is_dir():
                # Pack workers import from the artifact root, not workspace src/.
                for module in source.iterdir():
                    shutil.move(str(module), pack_root / module.name)
                source.rmdir()
                manifest_path.write_text(
                    re.sub(
                        r'(?m)^(file\s*=\s*")src/',
                        r"\1",
                        manifest_path.read_text(encoding="utf-8"),
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
            manifest = load_manifest(manifest_path)
            if manifest.publisher != "dinkster":
                raise ValueError(f"official pack {manifest.name!r} must declare publisher dinkster")
            tests = pack_root / "tests"
            if tests.is_dir():
                shutil.rmtree(tests)
            payload = output / f"{manifest.name}-{version}.zip"
            digest = build_artifact(pack_root, payload)
            with tempfile.TemporaryDirectory(prefix="dinkster-official-doctor-") as extracted:
                artifact_root = unpack_artifact(payload, Path(extracted), digest)
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
                    version=version,
                    artifact_digest=digest,
                    publisher=manifest.publisher,
                    claims=tuple(canonical_claim(claim) for claim in manifest.namespaces),
                    source=f"git:Kosinkadink/Dinkster@{commit}",
                )
            )
            payloads.append(payload)
            if variant is not None:
                assert python is not None and constraints is not None
                dependency_lock = resolve_pack_lock(
                    pack=manifest.name,
                    variant=variant,
                    requirements=(
                        manifest.requires_for(accelerator) if accelerator else manifest.requires
                    ),
                    python=python,
                    constraints=constraints,
                    indexes=indexes,
                    find_links=find_links,
                    requires_python=project.get("requires-python", ">=3.12"),
                    uv=uv,
                )
                dependency_payload = output / f"{manifest.name}-{version}-{variant}.pack-lock.json"
                dependency_lock.write(dependency_payload)
                payloads.append(dependency_payload)
    lock = output / "official-packs.lock"
    lock.write_text(Lockfile.of(locked).record_json() + "\n", encoding="utf-8", newline="\n")
    return (*payloads, lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Normalized public PEP 440 pack version")
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
        variant=args.variant,
        python=args.generation_python,
        constraints=args.constraints,
        indexes=args.index,
        find_links=args.find_links,
        accelerator=args.accelerator,
        uv=args.uv,
    ):
        print(payload)
