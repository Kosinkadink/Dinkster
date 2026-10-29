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
from dinkster_workers.boundary import PROTOCOL_VERSION
from dinkster_workers.pack_environment import (
    GenerationContext,
    doctor_generation,
    main,
    pack_drift,
    preflight_generation,
    provision_pack,
    rebuild_generation,
    venv_python,
)
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
        python = root / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"], text=True
        ).strip()
    )
    site.mkdir(parents=True, exist_ok=True)
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


def test_official_payloads_lock_committed_dependencies_with_variant_constraints(
    tmp_path: Path,
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_workers import load_manifest

    from scripts.build_official_packs import build

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    pinned = _wheel(wheels, "fixture-extra", "1.2", "fixture_extra", "value = 13\n")
    _wheel(wheels, "fixture-extra", "2.0", "fixture_extra", "value = 99\n")
    source = tmp_path / "source"
    source.mkdir()
    for family in ("comfy", "foundation", "image", "media-io", "remote", "generation"):
        root = source / "packages" / f"dinkster-nodes-{family}"
        root.mkdir(parents=True)
        (root / "dinkster-pack.toml").write_text(
            f'[pack]\nname="dinkster-nodes-{family}"\npublisher="dinkster"\n'
            'requires = ["fixture-extra<3"]\n'
            '[pack.dependencies]\ndinkster-nodes-foundation=">=0.0.1"\n'
            '[pack.entry]\nnodes="nodes:NODES"\n'
        )
        (root / "nodes.py").write_text("raise AssertionError('node bodies must not load')\n")
        (root / "pyproject.toml").write_text(
            '[project]\nrequires-python=">=3.12"\n'
            'dependencies=["fixture-extra>=2", "dinkster-nodes-foundation>=0.0.1"]\n'
        )
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    tree = subprocess.check_output(["git", "-C", str(source), "write-tree"], text=True).strip()
    (source / ".git/HEAD").write_text(tree + "\n")
    (source / "packages/dinkster-nodes-comfy/pyproject.toml").write_text("dirty = true\n")
    (source / "packages/dinkster-nodes-comfy/dinkster-pack.toml").write_text("dirty = true\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("fixture-extra==1.2\n")
    output = tmp_path / "payloads"
    payloads = build(
        source,
        output,
        version="1.2.3",
        variant="linux-cpu",
        python=Path(sys.executable),
        constraints=constraints,
        find_links=[wheels],
    )
    assert len(payloads) == 11
    assert not list(output.glob("dinkster-nodes-generation-*"))
    inventory = Lockfile.from_record_json((output / "official-packs.lock").read_text())
    assert {entry.pack for entry in inventory.packs} == {
        "dinkster-nodes-comfy",
        "dinkster-nodes-foundation",
        "dinkster-nodes-image",
        "dinkster-nodes-media-io",
        "dinkster-nodes-remote",
    }
    for entry in inventory.packs:
        assert entry.version == "1.2.3"
        lock_path = output / f"{entry.pack}-1.2.3-linux-cpu.pack-lock.json"
        lock = PackLock.read(lock_path)
        assert lock.pack == entry.pack
        assert lock.variant == "linux-cpu"
        assert lock.requires_python == ">=3.12"
        assert lock.requirements == ("fixture-extra<3",)
        assert [(package.name, package.version) for package in lock.packages] == [
            ("fixture-extra", "1.2")
        ]
        assert lock.packages[0].wheels == (pinned,)
        extracted = unpack_artifact(
            output / f"{entry.pack}-1.2.3.zip",
            tmp_path / "unpacked" / entry.pack,
            entry.artifact_digest,
        )
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert set(manifest.requires) == set(lock.requirements)
        assert manifest.dependencies[0].pack == "dinkster-nodes-foundation"


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
    pack_root = tmp_path / "checkout"
    pack_root.mkdir()
    (pack_root / "dinkster-pack.toml").write_text(
        '[pack]\nname="example"\nnamespaces=["example"]\nrequires=["fixture-extra>=1"]\n'
        '[pack.entry]\nnodes="example:NODES"\n'
    )
    published = tmp_path / "published.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_workers.pack_lock",
            str(pack_root),
            "--variant",
            "linux-cpu",
            "--generation-python",
            str(parent),
            "--constraints",
            str(constraints),
            "--find-links",
            str(tmp_path),
            "--output",
            str(published),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert PackLock.read(published) == lock
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


def test_publisher_resolution_rejects_source_only_packages(tmp_path: Path) -> None:
    archive = tmp_path / "source_only_fixture-1.0.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(
            "source_only_fixture-1.0/pyproject.toml",
            '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
            '[project]\nname="source-only-fixture"\nversion="1.0"\n',
        )
        source.writestr(
            "source_only_fixture-1.0/PKG-INFO",
            "Metadata-Version: 2.1\nName: source-only-fixture\nVersion: 1.0\n",
        )
        source.writestr("source_only_fixture-1.0/source_only_fixture.py", "VALUE = 1\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    with pytest.raises(
        ProvisionError, match="pack example: source-only-fixture==1.0: source-only closure"
    ):
        resolve_pack_lock(
            pack="example",
            variant="linux-cpu",
            requirements=[f"source-only-fixture @ {archive.as_uri()}"],
            python=Path(sys.executable),
            constraints=constraints,
            find_links=[tmp_path],
        )

    empty = PackLock("example", "linux-cpu", ">=3.12", (), (), ())
    published = tmp_path / "pack-lock.json"
    empty.write(published)
    assert PackLock.read(published) == empty
    shared = replace(empty, packages=(LockedDistribution("source-only-fixture", "1.0", ()),))
    shared.write(published)
    assert PackLock.read(published) == shared
    facts = GenerationPackages({"source-only-fixture": "1.0"}, "3.12.0", frozenset(), {})
    plan = plan_pack(shared, facts, variant="linux-cpu", own_environment_size=9)
    assert plan.conflicts == ()
    assert plan.wheels == ()


def test_generation_adapter_preflights_rebuilds_and_reports_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 23\n")
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.12",
        ("fixture-extra==1.2",),
        (),
        (LockedDistribution("fixture-extra", "1.2", (extra,)),),
    )
    old = tmp_path / "generations/1"
    (old / "packs/example").mkdir(parents=True)
    lock.write(old / "packs/example/pack-lock.json")
    current = tmp_path / "current"
    current.write_text("1\n")
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("packaging==26.2\n")
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        9000,
    )
    with pytest.raises(ProvisionError, match="source generation is required"):
        preflight_generation(context)
    plans = preflight_generation(context, source_generation=old)
    assert plans[0].wheels == (extra,)
    assert not (target / "packs").exists()
    assert not context.wheel_store.exists()
    sealed = target / "generation.json"
    sealed.write_text("{}")
    with pytest.raises(ProvisionError, match="fresh, inactive target"):
        rebuild_generation(context, plans)
    sealed.unlink()
    current.write_text("2\n")
    with pytest.raises(ProvisionError, match="fresh, inactive target"):
        rebuild_generation(context, plans)
    current.write_text("1\n")
    unrelated = target / "packs/unrelated"
    unrelated.mkdir(parents=True)
    with pytest.raises(ProvisionError, match="already has pack venvs"):
        rebuild_generation(context, plans)
    unrelated.rmdir()
    assert not context.wheel_store.exists()
    assert rebuild_generation(context, plans) == ("rebuilt pack example from its dependency lock",)
    assert doctor_generation(context) == ()
    assert current.read_text() == "1\n"
    assert (old / "packs/example/pack-lock.json").is_file()
    args = [
        "--generation",
        str(target),
        "--generation-python",
        str(python),
        "--constraints",
        str(constraints),
        "--variant",
        "linux-cpu",
        "--worker-protocol",
        str(PROTOCOL_VERSION),
        "--wheel-store",
        str(context.wheel_store),
        "--uv-cache",
        str(context.uv_cache),
        "--own-environment-size",
        "9000",
    ]
    assert main(["doctor", *args]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changes"] == []
    assert result["packs"] == [
        {"pack": "example", "conflicts": [], "downloadBytes": 0, "ownEnvironmentBytes": 9000}
    ]
    pack_python = venv_python(target / "packs/example")
    unavailable = pack_python.with_name("unavailable")
    pack_python.rename(unavailable)
    assert main(["doctor", *args]) == 1
    assert json.loads(capsys.readouterr().out)["changes"] == [
        "example: interpreter unavailable; rebuild from pack-lock.json"
    ]
    unavailable.rename(pack_python)
    assert main(["terminal", *args, "--pack", "example"]) == 0
    terminal = json.loads(capsys.readouterr().out)["changes"][0]
    assert Path(terminal) == venv_python(target / "packs/example")
    assert main(["terminal", *args]) == 0
    scratch = Path(json.loads(capsys.readouterr().out)["changes"][0])
    assert scratch == venv_python(tmp_path / "scratch/2")
    assert scratch != python
    metadata = next((target / "packs/example").rglob("fixture_extra-1.2.dist-info")) / "METADATA"
    metadata.write_text("Name: fixture-extra\nVersion: 9.0\n")
    assert main(["doctor", *args]) == 1
    drift = json.loads(capsys.readouterr().out)["changes"]
    assert any("fixture-extra" in message and "found 9.0" in message for message in drift)
    metadata.write_text("Name: fixture-extra\nVersion: 1.2\n")
    moved = tmp_path / "relocated-generation"
    shutil.move(target, moved)
    config = moved / "packs/example/pyvenv.cfg"
    config.write_text(config.read_text().replace(str(target), str(moved)))
    result = subprocess.run(
        [
            str(venv_python(moved / "packs/example")),
            "-c",
            "import fixture_extra; print(fixture_extra.value)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "23"


def test_generation_preflight_never_silently_omits_packs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "packs/broken").mkdir(parents=True)
    target = tmp_path / "target"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
    )
    with pytest.raises(ProvisionError, match="broken.*lock missing"):
        preflight_generation(context, source_generation=source)
    wrong_variant = PackLock("broken", "win-cpu", ">=3.12", (), (), ())
    wrong_variant.write(source / "packs/broken/pack-lock.json")
    plans = preflight_generation(context, source_generation=source)
    assert plans[0].conflicts == ("variant win-cpu does not match linux-cpu",)
    with pytest.raises(ProvisionError, match="variant win-cpu"):
        rebuild_generation(context, plans)
    assert not (target / "packs").exists()
    replacement = tmp_path / "replacement.json"
    replace(wrong_variant, variant="linux-cpu").write(replacement)
    assert not preflight_generation(context, source_generation=source, locks=[replacement])[
        0
    ].conflicts


def test_in_process_packs_stage_in_fresh_generation_and_terminals_use_scratch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 73\n")
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    lock = PackLock(
        "in-process",
        "linux-cpu",
        ">=3.12",
        ("fixture-extra==1.2",),
        (),
        (LockedDistribution("fixture-extra", "1.2", (extra,)),),
    )
    paths = []
    for name in ("in-process", "shared", "isolated"):
        path = tmp_path / f"{name}.json"
        replace(lock, pack=name).write(path)
        paths.append(path)
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
        in_process_packs=("in-process", "shared"),
    )
    plans = preflight_generation(context, locks=paths)
    assert [plan.lock.pack for plan in plans] == ["in-process", "shared", "isolated"]
    assert [plan.wheels for plan in plans] == [(extra,), (), ()]
    assert not context.wheel_store.exists()
    assert not (target / "packs").exists()
    seal = target / "generation.json"
    seal.write_text("{}")
    with pytest.raises(ProvisionError, match="fresh, inactive target"):
        rebuild_generation(context, plans)
    assert not context.wheel_store.exists()
    seal.unlink()
    rebuild_generation(context, plans)
    for name in context.in_process_packs:
        assert list((target / "packs" / name).iterdir()) == [
            target / "packs" / name / "pack-lock.json"
        ]
    isolated = venv_python(target / "packs/isolated")
    result = subprocess.check_output(
        [str(isolated), "-c", "import fixture_extra; print(fixture_extra.value)"], text=True
    )
    assert result.strip() == "73"
    assert doctor_generation(context) == ()
    args = [
        "--generation",
        str(target),
        "--generation-python",
        str(python),
        "--constraints",
        str(constraints),
        "--variant",
        "linux-cpu",
        "--worker-protocol",
        str(PROTOCOL_VERSION),
        "--wheel-store",
        str(context.wheel_store),
        "--uv-cache",
        str(context.uv_cache),
        "--own-environment-size",
        "1000",
        "--in-process-pack",
        "in-process",
        "--in-process-pack",
        "shared",
    ]
    assert main(["terminal", *args, "--pack", "in-process"]) == 0
    scratch = Path(json.loads(capsys.readouterr().out)["changes"][0])
    assert scratch == venv_python(tmp_path / "scratch/2")
    assert scratch != python
    metadata = next((target / "python").rglob("fixture_extra-1.2.dist-info")) / "METADATA"
    metadata.write_text("Name: fixture-extra\nVersion: 9.0\n")
    assert main(["doctor", *args]) == 1
    drift = json.loads(capsys.readouterr().out)["changes"]
    assert any("in-process: fixture-extra: expected 1.2, found 9.0" in item for item in drift)


def test_in_process_dependencies_conflict_before_downloads(tmp_path: Path) -> None:
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    first = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 1\n")
    second = _wheel(tmp_path, "fixture-extra", "2.0", "fixture_extra", "value = 2\n")
    paths = []
    for name, version, wheel in (("shared", "1.2", first), ("isolated", "2.0", second)):
        path = tmp_path / f"{name}.json"
        PackLock(
            name,
            "linux-cpu",
            ">=3.12",
            (f"fixture-extra=={version}",),
            (),
            (LockedDistribution("fixture-extra", version, (wheel,)),),
        ).write(path)
        paths.append(path)
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
        in_process_packs=("shared",),
    )
    plans = preflight_generation(context, locks=paths)
    assert plans[1].lock.pack == "isolated"
    assert plans[1].conflicts == ("fixture-extra: generation has 1.2, lock needs 2.0",)
    with pytest.raises(ProvisionError, match="fixture-extra.*1.2.*2.0"):
        rebuild_generation(context, plans)
    assert not context.wheel_store.exists()
    assert not (target / "packs").exists()


@pytest.mark.parametrize("windows_error", [False, True], ids=["native", "windows-launch-error"])
def test_doctor_rejects_unavailable_generation_even_without_packs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    windows_error: bool,
) -> None:
    if windows_error:

        def missing_interpreter(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("[WinError 2] The system cannot find the file specified")

        monkeypatch.setattr(subprocess, "run", missing_interpreter)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    assert (
        main(
            [
                "doctor",
                "--generation",
                str(tmp_path),
                "--generation-python",
                str(tmp_path / "missing-python"),
                "--constraints",
                str(constraints),
                "--variant",
                "linux-cpu",
                "--worker-protocol",
                str(PROTOCOL_VERSION),
                "--wheel-store",
                str(tmp_path / "wheels"),
                "--uv-cache",
                str(tmp_path / "cache"),
                "--own-environment-size",
                "1000",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["packs"] == []
    assert "missing-python" in result["changes"][0]
