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


@pytest.mark.parametrize("published", [False, True])
def test_default_admission_probes_and_tests_selected_library_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    published: bool,
) -> None:
    import shutil

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.pack_wheel import build_pack_archive

    from scripts import check_default_pack_admission as admission

    library = tmp_path / "library"
    module = library / "src/dinkster_test_helpers"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("VALUE = 37\n")
    (library / "pyproject.toml").write_text(
        '[project]\nname = "dinkster-test-helpers"\nversion = "0.0.1"\n'
        'requires-python = ">=3.12"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_test_helpers"]\n'
        "[tool.dinkster]\nofficial-library = true\n"
    )
    wheels = tmp_path / "wheels"
    subprocess.run(["uv", "build", "--wheel", str(library), "--out-dir", str(wheels)], check=True)
    pack = tmp_path / "pack"
    (pack / "dinkster_text").mkdir(parents=True)
    (pack / "dinkster_text/__init__.py").write_text(
        "from dinkster_test_helpers import VALUE\nassert VALUE == 37\n"
        "from dinkster_api.v1 import Node, NodeSchema\n"
        "class Example(Node):\n"
        "    @classmethod\n    def define_schema(cls):\n"
        "        return NodeSchema(node_type='dinkster.text.example', "
        "display_name='Example', category='text', description='Test helper import.')\n"
        "    @classmethod\n    def execute(cls):\n        return {}\n"
        "NODES = [Example]\n"
    )
    (pack / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-text"\nversion = "0.0.1"\npublisher = "dinkster"\n'
        'engine = ">=0.0.2,<0.1"\nrequires = ["dinkster-test-helpers>=0.0.1"]\n'
        'namespaces = ["dinkster.text"]\n'
        '[pack.entry]\nnodes = "dinkster_text:NODES"\n'
        '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
    )
    build_pack_archive(pack, wheels / "dinkster_text-0.0.1-py3-none-any.whl")
    entries = [
        LockedPack(
            name,
            "0.0.1",
            artifact_digest(path.read_bytes()),
            "dinkster",
            ("dinkster-text",) if name == "dinkster-text" else (),
            source="git:Kosinkadink/Dinkster@" + "a" * 40,
        )
        for name, path in (
            ("dinkster-text", wheels / "dinkster_text-0.0.1-py3-none-any.whl"),
            ("dinkster-test-helpers", wheels / "dinkster_test_helpers-0.0.1-py3-none-any.whl"),
        )
    ]
    catalog = Lockfile.of(entries)
    engine = tmp_path / "engine"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "src/dinkster",
        engine / "src/dinkster",
        ignore=shutil.ignore_patterns("__pycache__", "official_packs"),
    )
    (engine / "pyproject.toml").write_text("")
    tests = engine / "tests"
    tests.mkdir()
    (tests / "test_comfy_text_nodes.py").write_text(
        "from dinkster_text import VALUE\nfrom dinkster_test_helpers import VALUE as HELPER\n"
        "def test_selected_library():\n    assert VALUE == HELPER == 37\n"
    )
    checkout_pack = engine / "packages/dinkster-nodes-text"
    checkout_pack.mkdir(parents=True)
    (checkout_pack / "pyproject.toml").write_text('[project]\nname = "dinkster-text"\n')
    checkout_library = engine / "packages/helpers"
    shutil.copytree(library, checkout_library)
    (checkout_library / "src/dinkster_test_helpers/__init__.py").write_text("VALUE = 99\n")
    subprocess.run(["git", "init", "-q", str(engine)], check=True)
    subprocess.run(["git", "-C", str(engine), "add", "."], check=True)

    def candidate(root, output, *, packs):
        shutil.copytree(wheels, output)
        (output / "official-packs.lock").write_text(catalog.record_json())

    monkeypatch.setattr(
        admission, "__file__", str(engine / "scripts/check_default_pack_admission.py")
    )
    monkeypatch.setattr(admission, "default_pack_minimums", lambda: {"dinkster-text": "0.0.1"})
    monkeypatch.setattr(admission, "select_packs", lambda *args, **kwargs: ("dinkster-text",))
    monkeypatch.setattr(admission, "build", candidate)
    monkeypatch.setattr(
        admission, "prepare_registry_sources", lambda *args, **kwargs: (Lockfile.of([]), {})
    )
    probe = admission.run_executable_probe
    monkeypatch.setattr(
        admission,
        "run_executable_probe",
        lambda root, **kwargs: probe(root, require_isolation=False, **kwargs),
    )
    publication = None
    if published:
        publication = tmp_path / "official-packs.lock"
        publication.write_text(catalog.record_json())
        original = subprocess.check_output

        def git(command, **kwargs):
            if command == ["git", "rev-parse", "HEAD"]:
                return "a" * 40
            return original(command, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", git)

        def resolve(registry, name, version, *, engine_version):
            assert engine_version == (None if name == "dinkster-test-helpers" else "0.0.3")
            return next(entry for entry in entries if entry.pack == name)

        monkeypatch.setattr(admission, "resolve_release", resolve)
        monkeypatch.setattr(
            admission,
            "http_registry_fetcher",
            lambda endpoint: (
                lambda entry: (
                    wheels / f"{entry.pack.replace('-', '_')}-{entry.version}-py3-none-any.whl"
                ).read_bytes()
            ),
        )
    result = admission.check_defaults(
        admission.NamedRegistry("", "https://registry.invalid"),
        engine_version="0.0.3",
        candidate_base=None if published else "b" * 40,
        published_sources=publication,
    )
    assert [row["pack"] for row in result["packs"]] == ["dinkster-test-helpers", "dinkster-text"]
    output = capfd.readouterr().out
    assert "selected dinkster-test-helpers@0.0.1" in output
    assert "Testing dinkster-text@0.0.1" in output
    assert "Testing dinkster-test-helpers" not in output
    with ZipFile(wheels / "dinkster_test_helpers-0.0.1-py3-none-any.whl", "a") as wheel:
        wheel.writestr("changed.txt", "library drift")
    with pytest.raises(ValueError, match="library dinkster-test-helpers size or digest mismatch"):
        admission.check_defaults(
            admission.NamedRegistry("", "https://registry.invalid"),
            engine_version="0.0.3",
            candidate_base=None if published else "b" * 40,
            published_sources=publication,
        )
