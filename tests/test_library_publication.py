from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest
from dinkster_workers.boundary import PROTOCOL_VERSION
from dinkster_workers.pack_lock import PackLock

from scripts import build_release_pack_locks as native
from scripts import prepare_official_publication as preparation


@pytest.mark.parametrize("corrupt_source", [False, True])
@pytest.mark.parametrize("native_zip_system", [0, 3])
def test_library_preparation_and_native_locks_preserve_exact_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_source: bool,
    native_zip_system: int,
) -> None:
    files = {
        "packages/helpers/pyproject.toml": (
            '[project]\nname = "dinkster-test-helpers"\nversion = "0.0.1"\n'
            'requires-python = ">=3.12"\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_test_helpers"]\n'
            "[tool.dinkster]\nofficial-library = true\n"
        ),
        "packages/helpers/src/dinkster_test_helpers/__init__.py": "VALUE = 37\n",
        "packages/dinkster-nodes-test/pyproject.toml": (
            '[project]\nname = "dinkster-test-consumer"\nversion = "0.0.1"\n'
        ),
        "packages/dinkster-nodes-test/dinkster-pack.toml": (
            '[pack]\nname = "dinkster-test-consumer"\nversion = "0.0.1"\n'
            'publisher = "dinkster"\nengine = ">=0.0.2,<0.1"\n'
            'requires = ["dinkster-test-helpers>=0.0.1"]\n'
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
    system, runner = {
        "Linux": ("linux", "ubuntu-latest"),
        "Windows": ("windows", "windows-latest"),
        "Darwin": ("macos", "macos-latest"),
    }[platform.system()]
    base = {
        "variant": "linux-cpu",
        "id": "sha256:" + "b" * 64,
        "platform": {"os": system},
        "workerProtocol": PROTOCOL_VERSION,
        "archive": {"parts": []},
        "interpreter": {"executable": sys.executable},
        "wheels": [],
        "nvidia": None,
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/base_bundles.json").write_text(json.dumps({"linux-cpu": "b" * 64}))
    (tmp_path / "uv.lock").write_text("package = []\n")
    monkeypatch.setattr(preparation, "acquire", lambda *args: [base_path])
    monkeypatch.setattr(preparation, "development_version", lambda *args: "0.0.3.dev1")
    monkeypatch.setattr(preparation, "select_dependency_wheels", lambda *args: {})
    work = tmp_path / "work"
    assert preparation.prepare_inputs(
        tmp_path, work, "a" * 40, ["dinkster-test-consumer"], candidate=True
    ) == [{"variant": "linux-cpu", "runner": runner}]
    sources = work / "sources"
    source_bytes = {path.name: path.read_bytes() for path in sources.iterdir()}
    assert set(source_bytes) == {
        "official-packs.lock",
        "dinkster_test_helpers-0.0.1-py3-none-any.whl",
        "dinkster_test_consumer-0.0.1-py3-none-any.whl",
    }
    run = subprocess.run

    def native_build(command, **kwargs):
        result = run(command, **kwargs)
        if command[:2] == ["uv", "build"]:
            directory = Path(command[command.index("--out-dir") + 1])
            for path in directory.glob("*.whl"):
                output = io.BytesIO()
                with ZipFile(path) as source, ZipFile(output, "w") as target:
                    for entry in source.infolist():
                        entry.create_system = native_zip_system
                        target.writestr(entry, source.read(entry))
                path.write_bytes(output.getvalue())
        return result

    monkeypatch.setattr(subprocess, "run", native_build)
    if corrupt_source:
        (sources / "dinkster_test_helpers-0.0.1-py3-none-any.whl").write_bytes(b"wrong artifact")
    monkeypatch.setattr(native, "qualify", lambda *args: base)
    monkeypatch.setattr(
        native, "extract_base", lambda parts, tree, **kwargs: (tree / "wheels").mkdir(parents=True)
    )
    monkeypatch.setattr(native, "engine_constraints", lambda *args: "")
    monkeypatch.setattr(native, "effective_constraints", lambda *args, **kwargs: "")
    output = tmp_path / "locks"
    if corrupt_source:
        with pytest.raises(ValueError, match="changed source artifact"):
            native.build_locks(tmp_path, work / "engine", base_path, output, sources=sources)
        assert not output.exists()
        return
    paths = native.build_locks(tmp_path, work / "engine", base_path, output, sources=sources)
    assert {path.name for path in paths} == {
        "dinkster-test-helpers-0.0.1-linux-cpu.pack-lock.json",
        "dinkster-test-consumer-0.0.1-linux-cpu.pack-lock.json",
    }
    consumer = PackLock.read(output / "dinkster-test-consumer-0.0.1-linux-cpu.pack-lock.json")
    wheel = consumer.packages[0].wheels[0]
    data = source_bytes["dinkster_test_helpers-0.0.1-py3-none-any.whl"]
    assert wheel.sha256 == hashlib.sha256(data).hexdigest()
    assert wheel.size == len(data)
    assert (
        wheel.url == f"https://registry.dinkster.org/static/wheels/{wheel.sha256}/{wheel.filename}"
    )
    assert {path.name: path.read_bytes() for path in sources.iterdir()} == source_bytes
