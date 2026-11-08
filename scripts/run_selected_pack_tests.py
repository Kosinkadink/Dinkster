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
import venv
from dataclasses import replace
from pathlib import Path

from dinkster_registry import Lockfile, unpack_artifact
from dinkster_store import pack_archive_path
from dinkster_workers import load_manifest

from dinkster.compose import _FIRST_PARTY_PACK_DISTRIBUTIONS, _installed_pack_digest

_SUITES = {
    **{
        pack: (f"packages/{distribution}/tests/test_*.py",)
        for pack, distribution in _FIRST_PARTY_PACK_DISTRIBUTIONS.items()
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
        "tests/test_text_io.py",
        "tests/test_text_comfy_goldens.py",
    ),
}


def run_selected_tests(
    root: Path, catalog: Lockfile, artifacts: Path, *, test_packs: tuple[str, ...] = ()
) -> None:
    """Test only the scanner's verified selection through a disposable checkout layout."""
    if sys.platform != "linux":
        raise RuntimeError("selected pack admission tests require the Linux CI environment")
    unknown = {pack.pack for pack in catalog.packs} - _SUITES.keys()
    if unknown:
        raise ValueError(f"selected packs have no engine test suite: {sorted(unknown)}")
    if not catalog.packs:
        raise ValueError("selected pack tests require a nonempty catalog")
    if missing := set(test_packs) - {pack.pack for pack in catalog.packs}:
        raise ValueError(f"requested test packs are not in the selected catalog: {sorted(missing)}")
    distributions = {_FIRST_PARTY_PACK_DISTRIBUTIONS.get(pack, pack) for pack in _SUITES}
    with tempfile.TemporaryDirectory(prefix="selected-pack-tests-") as directory:
        work = Path(directory)
        checkout = work / "engine"
        tracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"], text=True
        ).split("\0")
        # Retain tests and distribution metadata for multi-pack projects, but
        # never copy official implementations or manifests from the engine tree.
        for name in filter(None, tracked):
            relative = Path(name)
            official = len(relative.parts) > 1 and relative.parts[1] in distributions
            if (
                official
                and "tests" not in relative.parts
                and not (
                    len(relative.parts) == 3
                    and relative.name == "pyproject.toml"
                    and not (root / relative.parent / "dinkster-pack.toml").is_file()
                )
            ):
                continue
            source = root / relative
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        module_roots: list[str] = []
        modules: dict[str, str] = {}
        checkout_packs = []
        for pack in catalog.packs:
            distribution = _FIRST_PARTY_PACK_DISTRIBUTIONS[pack.pack]
            package = checkout / "packages" / distribution
            destination = work / "packs" / pack.pack
            archive = pack_archive_path(artifacts, pack.artifact_digest)
            unpack_artifact(archive, destination, pack.artifact_digest)
            manifest = load_manifest(destination / "dinkster-pack.toml")
            if manifest.name != pack.pack:
                raise ValueError(f"selected test archive names {manifest.name}, not {pack.pack}")
            package.mkdir(parents=True, exist_ok=True)
            for source in destination.iterdir():
                if source.name == "tests":
                    raise ValueError(f"selected {pack.pack} cannot replace engine tests")
                target = package / source.name
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            destination = package
            # Keep manifest-relative assets and src imports on the same selected bytes.
            src = package / "src"
            src.mkdir(exist_ok=True)
            for module in destination.glob("dinkster_*"):
                if module.is_dir() and not (src / module.name).exists():
                    target = src / module.name
                    shutil.move(module, target)
                    module.symlink_to(target, target_is_directory=True)
            module_roots.append(str(src))
            entry = manifest.nodes_entry.partition(":")[0]
            modules[entry] = str(src / entry.replace(".", "/"))
            if not Path(modules[entry]).is_dir():
                raise ValueError(f"selected {pack.pack} archive is missing declared module {entry}")
            # A checkout excludes publication metadata and relocates modules.
            # Its local identity is not the selected registry archive identity.
            checkout_packs.append(
                replace(
                    pack,
                    artifact_digest=_installed_pack_digest(
                        destination / "dinkster-pack.toml", Path(modules[entry])
                    ),
                    source=f"local:{destination}",
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
                    elif any(part in distributions for part in candidate.parts):
                        continue
                    if candidate.exists():
                        paths.append(str(candidate))
                continue
            if source.name.startswith(("dinkster_nodes_", "dinkster_vision_", "_editable_")):
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
        child = work / "test-child.py"
        shutil.copy2(Path(__file__).with_name("_selected_pack_test_child.py"), child)
        shutil.copy2(child, checkout / "_selected_pack_test_child.py")
        for pack in catalog.packs:
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
                    {"modules": modules, "tests": tests, "packs": [p.pack for p in catalog.packs]}
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
