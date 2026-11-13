from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from dinkster_workers.pack_lock import PackLock


@pytest.mark.parametrize("pack", ["dinkster-test-helpers", "dinkster-test-consumer"])
def test_official_helper_and_consumer_locks_layer_on_later_compatible_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pack: str
) -> None:
    from dataclasses import replace

    from dinkster_workers.pack_environment import provision_pack
    from dinkster_workers.pack_lock import GenerationPackages, plan_pack
    from test_pack_environment import _generation, _wheel

    from scripts.build_official_packs import build

    files = {
        "packages/helpers/pyproject.toml": (
            '[project]\nname = "dinkster-test-helpers"\nversion = "0.0.2"\n'
            'requires-python = ">=3.12"\ndependencies = ["dinkster-api", "torch"]\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_test_helpers"]\n'
            '[tool.dinkster]\nofficial-library = true\nengine = ">=0.0.3.dev0,<0.1"\n'
        ),
        "packages/helpers/src/dinkster_test_helpers/__init__.py": (
            "from importlib.metadata import version\n"
            "def engine_version(): return version('dinkster-api')\n"
        ),
        "packages/dinkster-nodes-test/pyproject.toml": (
            '[project]\nname = "dinkster-test-consumer"\nversion = "0.0.4"\n'
        ),
        "packages/dinkster-nodes-test/dinkster-pack.toml": (
            '[pack]\nname = "dinkster-test-consumer"\nversion = "0.0.4"\n'
            'publisher = "dinkster"\nengine = ">=0.0.3.dev0,<0.1"\n'
            'requires = ["dinkster-test-helpers>=0.0.2,<1"]\n'
            '[pack.entry]\nnodes = "dinkster_test_consumer:NODES"\n'
        ),
        "packages/dinkster-nodes-test/src/dinkster_test_consumer/__init__.py": "NODES = []\n",
    }
    tracked = io.BytesIO()
    with tarfile.open(fileobj=tracked, mode="w") as archive:
        for name, text in files.items():
            data = text.encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    original = subprocess.check_output

    def git(command, **kwargs):
        if command[0] != "git":
            return original(command, **kwargs)
        if "rev-parse" in command:
            return "a" * 40
        if "archive" in command:
            return tracked.getvalue()
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "check_output", git)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    engine_a = "0.0.3.dev235+g3a30be8a6f04"
    engine_b = "0.0.3.dev236+g0123456789ab"
    _wheel(wheels, "dinkster-api", engine_a, "dinkster_api", "VALUE = 1\n")
    torch = _wheel(wheels, "torch", "2.14.1", "torch", "VALUE = 41\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(f"dinkster-api=={engine_a}\ntorch==2.14.1\n")
    output = tmp_path / "publication"
    output.mkdir()
    build(
        tmp_path,
        output,
        packs=("dinkster-test-helpers", "dinkster-test-consumer"),
        variant="linux-cpu",
        python=Path(sys.executable),
        constraints=constraints,
        find_links=(wheels,),
    )
    newer = tmp_path / "newer"
    newer.mkdir()
    api = _wheel(newer, "dinkster-api", engine_b, "dinkster_api", "VALUE = 2\n")
    parent = _generation(tmp_path / "generation/python", [api, torch])
    facts = GenerationPackages.inspect(parent)
    artifacts = {path.name: path.read_bytes() for path in output.iterdir()}
    for name, version in (("dinkster-test-helpers", "0.0.2"), ("dinkster-test-consumer", "0.0.4")):
        if name != pack:
            continue
        lock = PackLock.read(output / f"{name}-{version}-linux-cpu.pack-lock.json")
        plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=9000)
        assert not plan.conflicts, plan.conflicts
        assert "dinkster-api" not in {package.name for package in lock.packages}
        for versions, refused in (
            ({**facts.versions, "torch": "2.15.0"}, "torch"),
            ({**facts.versions, "dinkster-api": "0.1.0"}, "dinkster-api"),
            (
                {key: value for key, value in facts.versions.items() if key != "dinkster-api"},
                "dinkster-api",
            ),
        ):
            bad = plan_pack(
                lock,
                replace(facts, versions=versions),
                variant="linux-cpu",
                own_environment_size=9000,
            )
            assert any(refused in conflict for conflict in bad.conflicts), bad.conflicts
        # The published helper URL names immutable registry bytes. Seed those exact
        # bytes locally so the provisioning proof needs no registry or resolver.
        store = tmp_path / "store"
        store.mkdir(exist_ok=True)
        for wheel in plan.wheels:
            (store / wheel.filename).write_bytes(artifacts[wheel.filename])
        python = provision_pack(
            plan,
            generation_python=parent,
            packs=tmp_path / "generation/packs",
            wheel_store=store,
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
            offline=True,
        )
        if name == "dinkster-test-consumer":
            result = subprocess.check_output(
                [
                    str(python),
                    "-I",
                    "-c",
                    "from dinkster_test_helpers import engine_version; print(engine_version())",
                ],
                text=True,
            )
            assert result.strip() == engine_b
    assert {path.name: path.read_bytes() for path in output.iterdir()} == artifacts
