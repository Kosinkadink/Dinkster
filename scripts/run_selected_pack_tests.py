"""Run engine-owned pack tests with only the selected registry implementations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import venv
import zipfile
from dataclasses import replace
from pathlib import Path

from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
from dinkster_registry.archive_admission import ARCHIVE_LIMIT
from dinkster_store import pack_archive_path
from dinkster_workers import load_manifest
from dinkster_workers.pack_wheel import pack_import_name, pack_manifest_path

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
    "dinkster-nodes-dev": ("tests/test_gallery.py",),
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


def run_selected_tests(
    root: Path, catalog: Lockfile, artifacts: Path, *, test_packs: tuple[str, ...] = ()
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
    directories = set(_PACK_DIRECTORIES.values())
    directories.update(
        path.parent.name
        for path in (root / "packages").glob("*/pyproject.toml")
        if is_official_library(tomllib.loads(path.read_text()))
    )
    import_packages = {pack_import_name(pack) for pack in _PACK_DIRECTORIES}
    with tempfile.TemporaryDirectory(prefix="selected-pack-tests-") as directory:
        work = Path(directory)
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
            wheels = work / "wheels"
            wheels.mkdir()
            paths = []
            for archive in libraries.values():
                identity, version, _ = library_metadata(archive)
                wheel_path = wheels / f"{identity.replace('-', '_')}-{version}-py3-none-any.whl"
                shutil.copyfile(archive, wheel_path)
                paths.append(str(wheel_path))
                with zipfile.ZipFile(archive) as wheel:
                    for member in wheel.namelist():
                        if member.endswith(".py") and ".dist-info/" not in member:
                            module = member.split("/")[0].removesuffix(".py")
                            import_packages.add(module)
                            modules[module] = str(work / "libraries")
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--no-deps",
                    "--target",
                    str(work / "libraries"),
                    *paths,
                ],
                check=True,
            )
            module_roots.append(str(work / "libraries"))
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
        environment = work / "venv"
        venv.EnvBuilder(with_pip=False).create(environment)
        python = environment / "bin/python"
        site = Path(
            subprocess.check_output(
                [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        inherited = Path(sysconfig.get_path("purelib"))
        paths = [str(checkout), str(checkout / "src"), *module_roots]
        for source in inherited.iterdir():
            if source.suffix == ".pth":
                for line in source.read_text(encoding="utf-8").splitlines():
                    candidate = Path(line)
                    if not candidate.is_absolute() or not candidate.exists():
                        continue
                    if candidate.is_relative_to(root):
                        candidate = checkout / candidate.relative_to(root)
                    elif any(part in directories for part in candidate.parts):
                        continue
                    if candidate.exists():
                        paths.append(str(candidate))
                continue
            if source.name.startswith("_editable_") or any(
                source.name == name
                or source.name.startswith(name + "-")
                and source.name.endswith(".dist-info")
                for name in import_packages
            ):
                continue
            (site / source.name).symlink_to(source, target_is_directory=source.is_dir())
        (site / "selected.pth").write_text("\n".join(paths) + "\n", encoding="utf-8")
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
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(python),
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
    parser.add_argument(
        "--pack", action="append", default=[], help="Diagnose only this pack's suite"
    )
    args = parser.parse_args()
    run_selected_tests(
        args.root.resolve(),
        Lockfile.from_record_json(args.catalog.read_text(encoding="utf-8")),
        args.artifacts.resolve(),
        test_packs=tuple(args.pack),
    )
