from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import pytest
from dinkster_workers.boundary import PROTOCOL_VERSION
from dinkster_workers.pack_lock import PackLock

from scripts import build_release_pack_locks as native
from scripts import prepare_official_publication as preparation


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
    # The service probe requires POSIX process groups and resource limits.
    with (
        pytest.raises(
            ValueError,
            match=r"default dinkster-text@0\.0\.1 admission refused: "
            r"probe\.dependencies-unavailable: Dependency provisioning failed\.",
        )
        if sys.platform == "win32"
        else nullcontext()
    ):
        result = admission.check_defaults(
            admission.NamedRegistry("", "https://registry.invalid"),
            engine_version="0.0.3",
            candidate_base=None if published else "b" * 40,
            published_sources=publication,
        )
        assert result == {
            "engineVersion": "0.0.3",
            "packs": [
                {
                    "pack": entry.pack,
                    "version": entry.version,
                    "artifactDigest": entry.artifact_digest,
                }
                for entry in (entries[1], entries[0])
            ],
        }
    output = capfd.readouterr().out
    assert "selected dinkster-test-helpers@0.0.1" in output
    assert ("Testing dinkster-text@0.0.1" in output) is (sys.platform != "win32")
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


@pytest.mark.parametrize("dependent", [False, True])
def test_probe_index_stages_selected_library_dependency_without_public_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependent: bool
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from test_pack_environment import _wheel

    from scripts.stage_publication_probe_index import stage_index

    monkeypatch.setenv("UV_NO_INDEX", "1")
    sources = tmp_path / "sources"
    sources.mkdir()
    helper = _wheel(sources, "test-helper", "0.0.1", "test_helper", "VALUE = 37\n")
    consumer = _wheel(
        sources,
        "test-consumer",
        "0.0.2",
        "test_consumer",
        "import test_helper\n",
        requires=("test-helper==0.0.1",) if dependent else (),
    )
    entries = tuple(
        LockedPack(
            name, version, artifact_digest((sources / wheel.filename).read_bytes()), "dinkster", ()
        )
        for name, version, wheel in (
            ("test-consumer", "0.0.2", consumer),
            ("test-helper", "0.0.1", helper),
        )
    )
    (sources / "official-packs.lock").write_text(Lockfile.of(entries).record_json())
    output = tmp_path / "index"
    result = stage_index(sources, output)
    if dependent:
        assert result["distributions"] == 1
        assert result["wheels"] == [
            {
                "package": "test-helper",
                "filename": helper.filename,
                "sha256": helper.sha256,
                "size": helper.size,
            }
        ]
        assert (output / helper.filename).read_bytes() == (sources / helper.filename).read_bytes()
        assert (
            f"../../{helper.filename}#sha256={helper.sha256}"
            in (output / "simple/test-helper/index.html").read_text()
        )
    else:
        assert result == {"distributions": 0, "wheels": []}
    with ZipFile(sources / consumer.filename, "a") as archive:
        archive.writestr("changed.txt", "source drift")
    with pytest.raises(ValueError, match="probe source digest mismatch: test-consumer"):
        stage_index(sources, tmp_path / "bad-index")
