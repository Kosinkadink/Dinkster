"""Run engine-owned pack tests with only the selected registry implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
from dinkster_registry.archive_admission import (
    ARCHIVE_LIMIT,
    ENTRY_LIMIT,
    EXPANDED_LIMIT,
    FILE_LIMIT,
    admission_timing,
)
from dinkster_store import pack_archive_path
from dinkster_workers import load_manifest
from dinkster_workers.pack_environment import fetch_pack_wheels, provision_pack
from dinkster_workers.pack_lock import (
    GenerationPackages,
    LockedDistribution,
    LockedWheel,
    PackLock,
    plan_pack,
)
from dinkster_workers.pack_wheel import pack_manifest_path
from dinkster_workers.provision import ProvisionError
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

from dinkster.compose import _FIRST_PARTY_PACK_DISTRIBUTIONS, _installed_pack_digest
from scripts.build_official_packs import library_metadata
from scripts.build_release import is_official_library

_PACK_DIRECTORIES = {
    pack: distribution
    if distribution.startswith("dinkster-nodes-")
    else distribution.replace("dinkster-", "dinkster-nodes-", 1)
    for pack, distribution in _FIRST_PARTY_PACK_DISTRIBUTIONS.items()
}
_SUITES = {
    **{
        pack: (f"packages/{directory}/tests/test_*.py",)
        for pack, directory in _PACK_DIRECTORIES.items()
    },
    "dinkster-nodes-dev": ("tests/test_gallery.py",),
    "dinkster-utilities": (
        "packages/dinkster-nodes-utilities/tests/test_*.py",
        "tests/test_comfy_loops.py",
    ),
    "dinkster-image": (
        "packages/dinkster-nodes-image/tests/test_*.py",
        "tests/test_image_*.py",
        "tests/test_mask_operations.py",
        "tests/test_detection_nodes.py",
        "tests/test_detection_value.py",
        "tests/test_pose_nodes.py",
        "tests/test_glsl_shader.py",
        "tests/test_alpha_mask_operations.py",
        "tests/test_box_codec.py",
        "tests/test_tracked_crop.py",
        "tests/test_storage_inputs.py",
        "tests/test_vision_line_edge_schema.py",
    ),
    "dinkster-audio": (
        "packages/dinkster-nodes-audio/tests/test_*.py",
        "tests/test_audio_*.py",
        "tests/test_media_*.py",
    ),
    "dinkster-video": (
        "packages/dinkster-nodes-video/tests/test_*.py",
        "tests/test_video_*.py",
        "tests/test_media_*.py",
        "tests/test_capture_io.py",
        "tests/test_workflow_library.py",
    ),
    "dinkster-3d": (
        "packages/dinkster-nodes-3d/tests/test_*.py",
        "tests/test_model3d_*.py",
        "tests/test_splat_*.py",
        "tests/test_layer_documents.py",
    ),
    "dinkster-latent": (
        "packages/dinkster-nodes-latent/tests/test_*.py",
        "tests/test_latent_codec.py",
    ),
    "dinkster-text": (
        "tests/test_comfy_text_nodes.py",
        "tests/test_save_text.py",
        "tests/test_text_comfy_goldens.py",
    ),
}


def selected_library_wheels(catalog: Lockfile, artifacts: Path) -> dict[str, Path]:
    """Identify manifestless entries and verify the publisher's locked library identity."""
    libraries = {}
    for pack in catalog.packs:
        archive = pack_archive_path(artifacts, pack.artifact_digest)
        with zipfile.ZipFile(archive) as wheel:
            if any(Path(name).name == "dinkster-pack.toml" for name in wheel.namelist()):
                continue
        data = archive.read_bytes()
        if not 1 <= len(data) <= ARCHIVE_LIMIT or artifact_digest(data) != pack.artifact_digest:
            raise ValueError(f"selected library {pack.pack} size or digest mismatch")
        name, version, _ = library_metadata(archive)
        if (name, version) != (pack.pack, pack.version) or pack.claims:
            raise ValueError(f"selected library {pack.pack} identity mismatch")
        libraries[pack.pack] = archive
    return libraries


def selected_test_tools(root: Path, generation: GenerationPackages, *, variant: str) -> PackLock:
    """Select test-only wheels from the committed lock, never from the host installation."""
    tools = ("pytest", "pytest-cov", "pytest-timeout", "pytest-xdist", "hypothesis", "pyyaml")
    exported = subprocess.check_output(
        [
            "uv",
            "export",
            "--frozen",
            "--only-group",
            "dev",
            "--no-emit-workspace",
            "--no-header",
            "--no-annotate",
        ],
        cwd=root,
        text=True,
    )
    pins = {}
    for line in exported.replace("\\\n", "").splitlines():
        if not line.strip():
            continue
        requirement = Requirement(line.split(" --hash=", 1)[0].strip())
        if requirement.marker is None or requirement.marker.evaluate(dict(generation.markers)):
            pins[canonicalize_name(requirement.name)] = (
                requirement,
                set(re.findall(r"--hash=sha256:([0-9a-f]{64})", line)),
            )
    lock = tomllib.loads((root / "uv.lock").read_text())
    pending = list(tools)
    selected: dict[str, LockedDistribution] = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in selected:
            continue
        if name not in pins:
            raise ValueError(f"selected test tool has no applicable locked pin: {name}")
        requirement, hashes = pins[name]
        candidates = [
            package
            for package in lock["package"]
            if canonicalize_name(package["name"]) == name
            and requirement.specifier.contains(package["version"], prereleases=True)
            and (
                not package.get("resolution-markers")
                or any(
                    Marker(marker).evaluate(dict(generation.markers))
                    for marker in package["resolution-markers"]
                )
            )
        ]
        if len(candidates) != 1:
            raise ValueError(f"selected test tool requires one locked distribution: {name}")
        [package] = candidates
        wheels = []
        for wheel in package.get("wheels", []):
            filename = unquote(urlsplit(wheel["url"]).path).rsplit("/", 1)[-1]
            if (
                parse_wheel_filename(filename)[3] & generation.tags
                and wheel["hash"].removeprefix("sha256:") in hashes
            ):
                wheels.append(
                    LockedWheel(
                        filename, wheel["size"], wheel["hash"].removeprefix("sha256:"), wheel["url"]
                    )
                )
        if not wheels:
            raise ValueError(f"selected test tool has no compatible locked wheel: {name}")
        selected[name] = LockedDistribution(name, package["version"], tuple(wheels))
        pending.extend(
            dependency["name"]
            for dependency in package.get("dependencies", [])
            if "marker" not in dependency
            or Marker(dependency["marker"]).evaluate(dict(generation.markers))
        )
    return PackLock(
        "selected-test-tools",
        variant,
        ">=3.12",
        tools,
        (),
        tuple(selected[name] for name in sorted(selected)),
    )


def run_selected_tests(
    root: Path,
    catalog: Lockfile,
    artifacts: Path,
    *,
    generation_python: Path,
    locks: Mapping[str, PackLock],
    variant: str,
    test_packs: tuple[str, ...] = (),
) -> None:
    """Test only the scanner's verified selection through a disposable checkout layout."""
    if sys.platform != "linux":
        raise RuntimeError("selected pack admission tests require the Linux CI environment")
    libraries = selected_library_wheels(catalog, artifacts)
    unknown = {pack.pack for pack in catalog.packs} - libraries.keys() - _SUITES.keys()
    if unknown:
        raise ValueError(f"selected packs have no engine test suite: {sorted(unknown)}")
    if not catalog.packs:
        raise ValueError("selected pack tests require a nonempty catalog")
    if missing := set(test_packs) - {pack.pack for pack in catalog.packs}:
        raise ValueError(f"requested test packs are not in the selected catalog: {sorted(missing)}")
    if missing := {pack.pack for pack in catalog.packs} - locks.keys():
        raise ValueError(f"selected tests require dependency locks: {sorted(missing)}")
    generation = GenerationPackages.inspect(generation_python)
    for pack in catalog.packs:
        lock = locks[pack.pack]
        if lock.pack != pack.pack:
            raise ValueError(f"selected test dependency lock names {lock.pack}, not {pack.pack}")
        plan_pack(lock, generation, variant=variant, own_environment_size=0).require_layering()
        archive = pack_archive_path(artifacts, pack.artifact_digest)
        outer = next((package for package in lock.packages if package.name == pack.pack), None)
        filename = f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
        identity = (
            filename,
            archive.stat().st_size,
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        if (
            outer is None
            or outer.version != pack.version
            or not any(
                (wheel.filename, wheel.size, wheel.sha256) == identity for wheel in outer.wheels
            )
        ):
            raise ValueError(f"selected {pack.pack} dependency lock differs from its archive")
    tools = selected_test_tools(root, generation, variant=variant)
    directories = set(_PACK_DIRECTORIES.values())
    directories.update(
        path.parent.name
        for path in (root / "packages").glob("*/pyproject.toml")
        if is_official_library(tomllib.loads(path.read_text()))
    )
    with tempfile.TemporaryDirectory(prefix="selected-pack-tests-") as directory:
        layout_started = time.perf_counter()
        print(
            json.dumps({"admissionTiming": "start", "pack": "defaults", "stage": "test-layout"}),
            flush=True,
        )
        work = Path(directory)
        wheel_store = work / "wheels"
        wheel_store.mkdir()
        # Candidate archives need not be public yet; each consumer lock still
        # verifies the staged wheel's filename, size and digest before use.
        for pack in catalog.packs:
            filename = f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
            shutil.copyfile(
                pack_archive_path(artifacts, pack.artifact_digest), wheel_store / filename
            )
        checkout = work / "engine"
        tracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"], text=True
        ).split("\0")
        # Tests and build fixtures come from the engine; runtime bytes come from wheels.
        for name in filter(None, tracked):
            relative = Path(name)
            official = len(relative.parts) > 1 and relative.parts[1] in directories
            test_metadata = relative.name == "pyproject.toml" or (
                relative.name.startswith("draft-") and relative.suffix == ".json"
            )
            if official and "tests" not in relative.parts and not test_metadata:
                continue
            source = root / relative
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        module_roots: list[str] = []
        modules: dict[str, str] = {}
        if libraries:
            for name, archive in libraries.items():
                entry = catalog.get(name)
                assert entry is not None
                library_root = work / "libraries" / name
                with zipfile.ZipFile(archive) as wheel:
                    infos = wheel.infolist()
                    if (
                        len(infos) > ENTRY_LIMIT
                        or len({item.filename for item in infos}) != len(infos)
                        or sum(item.file_size for item in infos) > EXPANDED_LIMIT
                        or any(
                            item.file_size > FILE_LIMIT
                            or not (library_root / item.filename)
                            .resolve()
                            .is_relative_to(library_root.resolve())
                            or "\\" in item.filename
                            for item in infos
                        )
                    ):
                        raise ValueError(f"selected library {name} has unsafe archive entries")
                    wheel.extractall(library_root)
                    for member in wheel.namelist():
                        if member.endswith(".py") and ".dist-info/" not in member:
                            module = member.split("/")[0].removesuffix(".py")
                            modules[module] = str(library_root)
                module_roots.append(str(library_root))
        checkout_packs = []
        for pack in catalog.packs:
            if pack.pack in libraries:
                continue
            package = checkout / "packages" / _PACK_DIRECTORIES[pack.pack]
            destination = package / "src"
            archive = pack_archive_path(artifacts, pack.artifact_digest)
            unpack_artifact(archive, destination, pack.artifact_digest)
            manifest = load_manifest(pack_manifest_path(destination))
            if manifest.name != pack.pack:
                raise ValueError(f"selected test archive names {manifest.name}, not {pack.pack}")
            # Build configuration is engine test data, never installed. Runtime
            # identity and implementation come from the wheel's RECORD and manifest.
            shutil.copy2(manifest.path, package / "dinkster-pack.toml")
            # Source-layout contract tests read resources beside the manifest.
            # Keep Python modules inside their import package so cwd cannot shadow
            # standard-library modules such as types in spawned workers.
            for resource in manifest.root.iterdir():
                target = package / resource.name
                if target.exists() or resource.suffix in {".py", ".pyc"}:
                    continue
                if resource.is_dir():
                    shutil.copytree(
                        resource,
                        target,
                        ignore=shutil.ignore_patterns("*.py", "*.pyc", "__pycache__"),
                    )
                else:
                    shutil.copy2(resource, target)
            module_roots.append(str(destination))
            entry = manifest.nodes_entry.partition(":")[0]
            modules[entry] = str(destination / entry.replace(".", "/"))
            if not Path(modules[entry]).is_dir():
                raise ValueError(f"selected {pack.pack} archive is missing declared module {entry}")
            checkout_packs.append(
                replace(
                    pack,
                    artifact_digest=_installed_pack_digest(manifest.path, Path(modules[entry])),
                    source=f"local:{package}",
                )
            )
        (checkout / "src/dinkster/checkout-packs.lock").write_text(
            Lockfile.of(checkout_packs).record_json(), encoding="utf-8"
        )
        # Dev nodes are engine-owned test scaffolding, not a published pack.
        paths = [
            str(checkout),
            str(checkout / "src"),
            str(checkout / "packages/dinkster-nodes-dev/src"),
            *module_roots,
        ]
        runtimes: dict[str, str] = {}
        for pack in catalog.packs:
            if pack.pack in libraries:
                continue
            lock = locks[pack.pack]
            packages = {package.name: package for package in lock.packages}
            for tool in tools.packages:
                if tool.name in packages:
                    existing = packages[tool.name]
                    if existing.version != tool.version or not (
                        {wheel.sha256 for wheel in existing.wheels}
                        & {wheel.sha256 for wheel in tool.wheels}
                    ):
                        raise ProvisionError(
                            f"test tooling conflicts with {lock.pack}'s locked {tool.name}"
                        )
                else:
                    packages[tool.name] = tool
            test_lock = replace(
                lock,
                requirements=(*lock.requirements, *tools.requirements),
                packages=tuple(packages.values()),
            )
            with admission_timing(f"{pack.pack}@{pack.version}", "test-dependency-install"):
                plan = plan_pack(test_lock, generation, variant=variant, own_environment_size=0)
                fetch_pack_wheels(plan, work / "wheels")
                python = provision_pack(
                    plan,
                    generation_python=generation_python,
                    packs=work / "packs",
                    wheel_store=work / "wheels",
                    uv_cache=work / "uv-cache",
                    variant=variant,
                    offline=True,
                )
            site = Path(
                subprocess.check_output(
                    [
                        str(python),
                        "-I",
                        "-c",
                        "import sysconfig;print(sysconfig.get_path('purelib'))",
                    ],
                    text=True,
                ).strip()
            )
            (site / "selected.pth").write_text(
                f"import sys; sys.path[:0] = {paths!r}\n", encoding="utf-8"
            )
            runtimes[pack.pack] = str(python)
        child_env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTEST_ADDOPTS"}
        }
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["DINKSTER_ACCELERATOR"] = "cpu"
        # Historical evidence is read from Git, never added to the import search path.
        child_env["DINKSTER_TEST_HISTORY_ROOT"] = str(root)
        child = work / "test-child.py"
        shutil.copy2(Path(__file__).with_name("_selected_pack_test_child.py"), child)
        shutil.copy2(child, checkout / "_selected_pack_test_child.py")
        print(
            json.dumps(
                {
                    "admissionTiming": "end",
                    "pack": "defaults",
                    "stage": "test-layout",
                    "status": "completed",
                    "elapsedSeconds": time.perf_counter() - layout_started,
                }
            ),
            flush=True,
        )
        for pack in catalog.packs:
            if pack.pack in libraries:
                continue
            if test_packs and pack.pack not in test_packs:
                continue
            tests = sorted(
                {
                    str(path.relative_to(checkout))
                    for pattern in _SUITES[pack.pack]
                    for path in checkout.glob(pattern)
                }
            )
            if not tests:
                raise ValueError(f"selected {pack.pack} has no engine behavioral tests")
            print(f"Testing {pack.pack}@{pack.version} {pack.artifact_digest}", flush=True)
            config = work / "test-selection.json"
            config.write_text(
                json.dumps(
                    {
                        "modules": modules,
                        "tests": tests,
                        "packs": [p.pack for p in catalog.packs if p.pack not in libraries],
                        "runtimes": runtimes,
                        "testPack": f"{pack.pack}@{pack.version}",
                    }
                ),
                encoding="utf-8",
            )
            with admission_timing(f"{pack.pack}@{pack.version}", "test-child"):
                subprocess.run(
                    [
                        runtimes[pack.pack],
                        "-I",
                        "-B",
                        str(child),
                        str(config),
                    ],
                    cwd=checkout,
                    env=child_env,
                    check=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--generation-python", type=Path, required=True)
    parser.add_argument("--locks", type=Path, required=True)
    parser.add_argument("--variant", default="linux-cpu")
    parser.add_argument(
        "--pack", action="append", default=[], help="Diagnose only this pack's suite"
    )
    args = parser.parse_args()
    run_selected_tests(
        args.root.resolve(),
        Lockfile.from_record_json(args.catalog.read_text(encoding="utf-8")),
        args.artifacts.resolve(),
        generation_python=args.generation_python.absolute(),
        locks={path.stem: PackLock.read(path) for path in args.locks.glob("*.json")},
        variant=args.variant,
        test_packs=tuple(args.pack),
    )
