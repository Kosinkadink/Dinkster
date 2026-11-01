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
from pathlib import Path

from dinkster_registry import Lockfile, unpack_artifact
from dinkster_workers import load_manifest

_SUITES = {
    "dinkster-nodes-foundation": (
        "tests/test_primitives.py",
        "tests/test_numeric_stdlib.py",
        "tests/test_string_stdlib.py",
        "tests/test_routing_stdlib.py",
        "tests/test_conversion_curve_stdlib.py",
        "tests/test_lists.py",
        "tests/test_regions.py",
        "tests/test_foundation_guide.py",
        "tests/test_combo_nodes.py",
        "tests/test_string_comfy_aliases.py",
        "tests/test_comfy_loops.py",
        "tests/test_output_descriptors.py",
        "tests/test_composed_schema.py",
        "tests/test_editor_roles.py",
    ),
    "dinkster-nodes-image": (
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
    "dinkster-nodes-media-io": (
        "tests/test_video_*.py",
        "tests/test_audio_*.py",
        "tests/test_media_*.py",
        "tests/test_text_io.py",
        "tests/test_latent_codec.py",
        "tests/test_model3d_*.py",
        "tests/test_splat_*.py",
        "tests/test_layer_documents.py",
        "tests/test_capture_io.py",
        "tests/test_workflow_library.py",
        "tests/test_text_comfy_goldens.py",
    ),
    "dinkster-nodes-comfy": ("packages/dinkster-nodes-comfy/tests/test_*.py",),
    "dinkster-nodes-remote": ("packages/dinkster-nodes-remote/tests/test_*.py",),
    **{
        f"dinkster-vision-{name.replace('_', '-')}": (
            f"packages/dinkster-nodes-vision/tests/test_{name}.py",
        )
        for name in (
            "birefnet",
            "depth_anything_v2",
            "depth_anything_v3",
            "detr",
            "efficient_sam",
            "hed",
            "rtdetr",
            "sam31",
            "upscale",
        )
    },
}


def run_selected_tests(
    root: Path, catalog: Lockfile, artifacts: Path, *, test_packs: tuple[str, ...] = ()
) -> None:
    """Use the scanner's exact selection, never resolve or build a second archive."""
    if sys.platform != "linux":
        raise RuntimeError("selected pack admission tests require the Linux CI environment")
    unknown = {pack.pack for pack in catalog.packs} - _SUITES.keys()
    if unknown:
        raise ValueError(f"selected packs have no engine test suite: {sorted(unknown)}")
    if not catalog.packs:
        raise ValueError("selected pack tests require a nonempty catalog")
    if missing := set(test_packs) - {pack.pack for pack in catalog.packs}:
        raise ValueError(f"requested test packs are not in the selected catalog: {sorted(missing)}")
    with tempfile.TemporaryDirectory(prefix="selected-pack-tests-") as directory:
        work = Path(directory)
        checkout = work / "engine"
        tracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"], text=True
        ).split("\0")
        # Tests retain their checkout-relative manifests and helpers, but no
        # official implementation or manifest is copied from the engine tree.
        for name in filter(None, tracked):
            relative = Path(name)
            official = len(relative.parts) > 1 and relative.parts[1] in {
                "dinkster-nodes-comfy",
                "dinkster-nodes-foundation",
                "dinkster-nodes-image",
                "dinkster-nodes-media-io",
                "dinkster-nodes-remote",
                "dinkster-nodes-vision",
            }
            if official and "tests" not in relative.parts:
                continue
            source = root / relative
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        module_roots: list[str] = []
        modules: dict[str, str] = {}
        for pack in catalog.packs:
            if pack.pack.startswith("dinkster-vision-"):
                package = checkout / "packages/dinkster-nodes-vision"
                destination = package / (pack.pack.replace("-", "_") + "_pack")
            else:
                package = checkout / "packages" / pack.pack
                destination = work / "packs" / pack.pack
            archive = artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip"
            unpack_artifact(archive, destination, pack.artifact_digest)
            manifest = load_manifest(destination / "dinkster-pack.toml")
            if manifest.name != pack.pack:
                raise ValueError(f"selected test archive names {manifest.name}, not {pack.pack}")
            if not pack.pack.startswith("dinkster-vision-"):
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
            # Existing member tests explicitly add checkout-relative src paths.
            src = package / "src"
            src.mkdir(exist_ok=True)
            for module in destination.glob("dinkster_*"):
                if module.name == "dinkster_nodes_vision":
                    namespace = src / module.name
                    namespace.mkdir(exist_ok=True)
                    for provider in module.iterdir():
                        if provider.is_dir():
                            shutil.copytree(provider, namespace / provider.name)
                elif module.is_dir() and not (src / module.name).exists():
                    shutil.copytree(module, src / module.name)
            module_roots.append(str(src))
            entry = manifest.nodes_entry.partition(":")[0]
            modules[entry] = str(src / entry.replace(".", "/"))
        bundled = checkout / "src/dinkster/official_packs"
        bundled.mkdir(parents=True)
        (bundled / "official-packs.lock").write_text(catalog.record_json(), encoding="utf-8")
        for pack in catalog.packs:
            archive = artifacts / f"{pack.artifact_digest.removeprefix('blake3:')}.zip"
            shutil.copy2(archive, bundled / f"{pack.pack}-{pack.version}.zip")
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
                    elif any(
                        part in {*_SUITES, "dinkster-nodes-vision"} for part in candidate.parts
                    ):
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
        child = work / "test-child.py"
        shutil.copy2(Path(__file__).with_name("_selected_pack_test_child.py"), child)
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
