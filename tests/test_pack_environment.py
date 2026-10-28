"""Real wheel and interpreter coverage for generation-layered pack dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_workers.pack_environment import pack_drift, provision_pack, venv_python
from dinkster_workers.pack_lock import (
    OPENCV_ALIASES,
    GenerationPackages,
    LockedDistribution,
    LockedWheel,
    PackLock,
    normalize_requirement,
    plan_pack,
    resolve_pack_lock,
)
from dinkster_workers.provision import ProvisionError
from packaging.tags import Tag


def _wheel(root: Path, name: str, version: str, module: str, body: str) -> LockedWheel:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    path = root / filename
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{module}.py", body)
        archive.writestr(
            f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(f"{info}/RECORD", "")
    return LockedWheel(
        filename, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), path.as_uri()
    )


def _generation(root: Path, wheels: list[LockedWheel]) -> Path:
    """A separate interpreter prefix, like a linked python-build-standalone tree."""
    base = Path(sys.base_prefix)
    if os.name == "nt":
        root.mkdir(parents=True)
        for source in base.iterdir():
            if source.suffix in {".exe", ".dll"}:
                shutil.copy2(source, root / source.name)
        shutil.copytree(base / "DLLs", root / "DLLs")
        shutil.copytree(base / "Lib", root / "Lib", ignore=shutil.ignore_patterns("site-packages"))
        site = root / "Lib/site-packages"
        python = root / "python.exe"
    else:
        (root / "bin").mkdir(parents=True)
        shutil.copy2(Path(sys.executable).resolve(), root / "bin/python")
        lib = root / "lib"
        lib.mkdir()
        for source in (base / "lib").iterdir():
            if not source.is_dir():
                (lib / source.name).symlink_to(source)
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        stdlib = lib / version
        stdlib.mkdir()
        for source in (base / "lib" / version).iterdir():
            if source.name != "site-packages":
                (stdlib / source.name).symlink_to(source)
        site = stdlib / "site-packages"
        python = root / "bin/python"
    site.mkdir()
    import packaging

    shutil.copytree(Path(packaging.__file__).parent, site / "packaging")
    for wheel in wheels:
        with zipfile.ZipFile(
            Path(urllib.request.url2pathname(wheel.url.removeprefix("file://")))
        ) as archive:
            archive.extractall(site)
    return python


def test_layered_pack_downloads_only_extra_and_rebuilds_on_new_generation(tmp_path: Path) -> None:
    torch = _wheel(tmp_path, "torch", "2.14.1", "torch", "value = 41\n")
    cv = _wheel(tmp_path, "opencv-contrib-python-headless", "5.0", "cv2", "value = 17\n")
    extra = _wheel(
        tmp_path,
        "fixture-extra",
        "1.2",
        "fixture_extra",
        "import torch, cv2\ndef node(): return torch.value + cv2.value\n",
    )
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.12",
        (
            "torch>=2",
            normalize_requirement("opencv-python[ffmpeg]>=4"),
            "fixture-extra==1.2",
        ),
        (),
        tuple(
            LockedDistribution(name, version, (wheel,))
            for name, version, wheel in (
                ("torch", "2.14.1", torch),
                ("opencv-contrib-python-headless", "5.0", cv),
                ("fixture-extra", "1.2", extra),
            )
        ),
    )
    store = tmp_path / "wheels"
    for number in (1, 2):
        generation = tmp_path / "generations" / str(number)
        parent = _generation(generation, [torch, cv])
        facts = GenerationPackages.inspect(parent)
        plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=987654)
        assert not plan.conflicts
        assert plan.wheels == (extra,)
        python = provision_pack(
            plan,
            generation_python=parent,
            packs=generation / "packs",
            wheel_store=store,
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
        )
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import fixture_extra, torch, cv2, json; "
                "print(json.dumps([fixture_extra.node(), torch.__file__, cv2.__file__]))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value, torch_file, cv_file = json.loads(result.stdout)
        assert value == 58
        assert Path(torch_file).is_relative_to(generation)
        assert Path(cv_file).is_relative_to(generation)
        assert not Path(torch_file).is_relative_to(generation / "packs")
        assert not Path(cv_file).is_relative_to(generation / "packs")
        assert pack_drift(python, lock, facts) == ()
        assert PackLock.read(generation / "packs/example/pack-lock.json") == lock
        assert [path.name for path in store.rglob("*.whl")] == [extra.filename]
        assert not list((generation / "packs").rglob("torch*"))


def test_plan_names_conflicts_and_own_environment_size(tmp_path: Path) -> None:
    torch = _wheel(tmp_path, "torch", "2.15", "torch", "")
    compiled = LockedWheel(
        "kernel-1-cp312-cp312-win_amd64.whl", 100, "a" * 64, "https://example.invalid/kernel.whl"
    )
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.15",
        ("torch>=2.15",),
        (),
        (
            LockedDistribution("torch", "2.15", (torch,)),
            LockedDistribution("kernel", "1", (compiled,)),
        ),
    )
    facts = GenerationPackages(
        {"torch": "2.14.1"}, "3.14.6", frozenset({Tag("py3", "none", "any")}), {}
    )
    plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=7654321)
    assert any("torch" in conflict and "2.14.1" in conflict for conflict in plan.conflicts)
    assert any("Python" in conflict for conflict in plan.conflicts)
    assert any(
        "kernel" in conflict and "no compatible wheel" in conflict for conflict in plan.conflicts
    )
    with pytest.raises(ProvisionError, match=r"own environment \(7654321 bytes\)"):
        plan.require_layering()


@pytest.mark.parametrize("name", OPENCV_ALIASES)
@pytest.mark.parametrize("extra", ("", "[ffmpeg]"))
def test_opencv_alias_preserves_constraint_and_marker(name: str, extra: str) -> None:
    assert normalize_requirement(f'{name}{extra}>=4,<6; python_version >= "3.12"') == (
        'opencv-contrib-python-headless<6,>=4; python_version >= "3.12"'
    )


def test_local_resolution_uses_constraints_and_provision_rejects_bad_hash(tmp_path: Path) -> None:
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 13\n")
    _wheel(tmp_path, "fixture-extra", "2.0", "fixture_extra", "value = 99\n")
    parent = _generation(tmp_path / "generation", [])
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("fixture-extra==1.2\n")
    lock = resolve_pack_lock(
        pack="example",
        variant="linux-cpu",
        requirements=["fixture-extra>=1"],
        python=parent,
        constraints=constraints,
        find_links=[tmp_path],
    )
    assert [(package.name, package.version) for package in lock.packages] == [
        ("fixture-extra", "1.2")
    ]
    assert lock.packages[0].wheels[0].sha256 == extra.sha256
    bad = replace(
        lock, packages=(replace(lock.packages[0], wheels=(replace(extra, sha256="b" * 64),)),)
    )
    plan = plan_pack(
        bad, GenerationPackages.inspect(parent), variant="linux-cpu", own_environment_size=9
    )
    with pytest.raises(ProvisionError, match="hash or size mismatch"):
        provision_pack(
            plan,
            generation_python=parent,
            packs=tmp_path / "packs",
            wheel_store=tmp_path / "store",
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
        )
    assert not venv_python(tmp_path / "packs/example").exists()
