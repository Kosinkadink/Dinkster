"""Build local pack inputs using the selected generation's API and resolver."""

from __future__ import annotations

import email.parser
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dinkster_registry import LockedPack, Lockfile, artifact_digest
from dinkster_schema import canonical_claim
from dinkster_store import object_path, pack_archive_path, pack_label
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from .manifest import load_manifest
from .pack_lock import resolve_pack_lock
from .pack_wheel import build_pack_archive, wheel_filename
from .provision import ProvisionError


def build_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    generation, store, artifacts = (
        Path(inputs[name]) for name in ("generation", "store", "artifacts")
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    library_wheels: dict[str, Path] = {}
    library_versions: dict[str, str] = {}
    built_wheels: list[dict[str, Any]] = []
    for number, value in enumerate(inputs["libraries"]):
        source = Path(value)
        if source.is_file():
            wheel = artifacts / source.name
            shutil.copyfile(source, wheel)
        else:
            output = artifacts / f"library-{number}"
            result = subprocess.run(
                ["uv", "build", "--wheel", "--no-sources", "--out-dir", str(output), str(source)],
                cwd=source,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise ProvisionError(f"development library build refused: {result.stderr.strip()}")
            built = list(output.glob("*.whl"))
            if len(built) != 1:
                raise ProvisionError(f"expected one library wheel: {source}")
            wheel = built[0]
        with zipfile.ZipFile(wheel) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1 or any(
                name.endswith("/dinkster-pack.toml") for name in archive.namelist()
            ):
                raise ProvisionError(f"expected a library wheel, not a node pack: {wheel}")
            fields = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
            name = canonicalize_name(fields["Name"])
            library_versions[name] = fields["Version"]
        if name in library_wheels:
            raise ProvisionError(f"duplicate development library: {name}")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        retained = object_path(store / "wheels", "sha256:" + digest) / wheel.name
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheel, retained)
        library_wheels[name] = retained
        built_wheels.append(
            {"filename": wheel.name, "sha256": digest, "size": wheel.stat().st_size}
        )

    constraints = artifacts / "development-constraints.txt"
    constraints.write_text(
        (generation / "constraints.txt").read_text()
        + "\n"
        + "\n".join(f"{name}=={value}" for name, value in library_versions.items())
        + "\n",
        encoding="utf-8",
    )
    api_version = version("dinkster-api")
    links = [path for path in (store / "wheels").iterdir() if path.is_dir()]
    locks: list[str] = []
    entries: list[LockedPack] = []
    for number, value in enumerate(inputs["packs"]):
        source = Path(value)
        manifest = load_manifest(source / "dinkster-pack.toml")
        compatibility = manifest.compatibility
        if compatibility is None or not SpecifierSet(compatibility.engine).contains(
            api_version, prereleases=True
        ):
            raise ProvisionError(f"{manifest.name}: declared engine requirement is incompatible")
        if any(entry.pack == manifest.name for entry in entries):
            raise ProvisionError(f"duplicate development pack: {manifest.name}")
        wheel = artifacts / wheel_filename(manifest)
        build_pack_archive(source, wheel)
        payload = wheel.read_bytes()
        identity = artifact_digest(payload)
        retained = pack_archive_path(
            artifacts, identity, label=pack_label(manifest.name, compatibility.version)
        )
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheel, retained)
        built_wheels.append(
            {
                "filename": wheel.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
        lock = resolve_pack_lock(
            pack=manifest.name,
            variant=inputs["variant"],
            requirements=(
                *manifest.requires_for(inputs["accelerator"]),
                *(f"{name}=={value}" for name, value in library_versions.items()),
            ),
            python=Path(sys.executable),
            constraints=constraints,
            indexes=(inputs["index"],),
            find_links=links,
            engine=compatibility.engine,
        )
        for distribution in lock.packages:
            if distribution.name in library_wheels:
                selected = library_wheels[distribution.name]
                expected = hashlib.sha256(selected.read_bytes()).hexdigest()
                if not any(wheel.sha256 == expected for wheel in distribution.wheels):
                    raise ProvisionError(
                        f"resolved {distribution.name} differs from the selected local library"
                    )
        path = artifacts / f"{manifest.name}.pack-lock.json"
        lock.write(path)
        locks.append(str(path))
        entries.append(
            LockedPack(
                manifest.name,
                compatibility.version,
                identity,
                manifest.publisher,
                tuple(canonical_claim(claim) for claim in manifest.namespaces),
                source="local-build:" + inputs["packPaths"][number],
            )
        )
    (artifacts / "development-packs.lock").write_text(
        Lockfile.of(entries).record_json(), encoding="utf-8"
    )
    return {"locks": locks, "builtWheels": built_wheels}


if __name__ == "__main__":
    print(json.dumps(build_inputs(json.load(sys.stdin))))
