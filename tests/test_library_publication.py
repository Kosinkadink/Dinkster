from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from contextlib import nullcontext
from dataclasses import replace
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
        "pyproject.toml": (
            '[project]\nname = "dinkster"\nversion = "0.0.3"\n'
            '[tool.uv.workspace]\nmembers = ["packages/*"]\n'
        ),
        "src/dinkster/__init__.py": '__version__ = "0.0.3"\n',
        "packages/dinkster-api/pyproject.toml": (
            '[project]\nname = "dinkster-api"\nversion = "0.0.3"\n'
            'requires-python = ">=3.12"\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_api"]\n'
        ),
        "packages/dinkster-api/src/dinkster_api/__init__.py": "",
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


@pytest.mark.parametrize(
    "mode", ["candidate", "published", "prepared", "registry", "missing", "installer"]
)
def test_default_admission_probes_and_tests_selected_library_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    mode: str,
    request: pytest.FixtureRequest,
    selected_pack_runtime,
) -> None:
    import shutil

    published = (
        False if mode == "candidate" else True if mode in {"published", "prepared"} else mode
    )

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
        + (
            'dependencies = ["dinkster-unavailable-test-library==0.0.1"]\n'
            if published == "installer"
            else ""
        )
        + '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
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
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for entry in entries:
        shutil.copyfile(
            wheels / f"{entry.pack.replace('-', '_')}-{entry.version}-py3-none-any.whl",
            artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip",
        )
    python, locks = selected_pack_runtime(catalog, artifacts)
    if published == "installer":
        locks["dinkster-text"] = replace(
            locks["dinkster-text"], requirements=("dinkster-unavailable-test-library==0.0.1",)
        )
    constraints = tmp_path / "constraints.txt"
    constraints.touch()
    context = {
        "generation_python": python,
        "base_id": "sha256:" + "f" * 64,
        "constraints": constraints,
    }
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

    def candidate(root, output, *, packs, **kwargs):
        shutil.copytree(wheels, output)
        (output / "official-packs.lock").write_text(catalog.record_json())
        for name, lock in locks.items():
            lock.write(output / f"{name}-0.0.1-linux-cpu.pack-lock.json")

    monkeypatch.setattr(
        admission, "__file__", str(engine / "scripts/check_default_pack_admission.py")
    )
    monkeypatch.setattr(admission, "default_pack_minimums", lambda: {"dinkster-text": "0.0.1"})
    monkeypatch.setattr(admission, "select_packs", lambda *args, **kwargs: ("dinkster-text",))
    monkeypatch.setattr(admission, "build", candidate)
    monkeypatch.setattr(
        admission,
        "read_registry_pack_locks",
        lambda config, catalog, **kwargs: tuple(locks[entry.pack] for entry in catalog.packs),
    )
    monkeypatch.setattr(
        admission, "prepare_registry_sources", lambda *args, **kwargs: (Lockfile.of([]), {})
    )
    registry_url = "https://registry.invalid"
    ordinary_requests: list[tuple[str, str | None]] = []
    if isinstance(published, str):
        from test_installer import _serve_artifacts, _static_release

        library_entry = entries[1]
        archive = wheels / "dinkster_test_helpers-0.0.1-py3-none-any.whl"
        data = archive.read_bytes()
        row = {**_static_release(library_entry, data), "library": True}
        files = {
            "/static/packs/index.json": json.dumps(
                {
                    "schemaVersion": 2,
                    "items": [
                        {
                            "name": library_entry.pack,
                            "latestVersion": "0.0.1",
                            "publisher": "dinkster",
                            "library": True,
                            "url": f"packs/{library_entry.pack}/index.json",
                        }
                    ],
                }
            ).encode(),
            f"/static/packs/{library_entry.pack}/index.json": json.dumps(
                {
                    "schemaVersion": 2,
                    "name": library_entry.pack,
                    "items": [] if published == "missing" else [row],
                }
            ).encode(),
            f"/static/wheels/{hashlib.sha256(data).hexdigest()}/{archive.name}": data,
        }
        server = _serve_artifacts(files, [])
        request.addfinalizer(server.shutdown)
        registry_url = f"http://127.0.0.1:{server.server_port}"
        ordinary = _serve_artifacts({}, ordinary_requests)
        request.addfinalizer(ordinary.shutdown)
        monkeypatch.setenv("UV_INDEX_URL", f"http://127.0.0.1:{ordinary.server_port}")

        def registry_defaults(registry, minimums, artifacts, *, engine_version):
            entry = entries[0]
            (artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip").write_bytes(
                (wheels / "dinkster_text-0.0.1-py3-none-any.whl").read_bytes()
            )
            return Lockfile.of([entry]), {}

        monkeypatch.setattr(admission, "prepare_registry_sources", registry_defaults)
    probe = admission.run_locked_probe

    def executable_probe(root, lock, **kwargs):
        return probe(root, lock, require_isolation=sys.platform == "win32", **kwargs)

    monkeypatch.setattr(
        admission,
        "run_locked_probe",
        executable_probe,
    )
    publication = None
    if mode in {"published", "prepared"}:
        original = subprocess.check_output

        def git(command, **kwargs):
            if command == ["git", "rev-parse", "HEAD"]:
                return "a" * 40
            return original(command, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", git)

    if mode == "published":
        publication = tmp_path / "official-packs.lock"
        publication.write_text(catalog.record_json())

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
    prepared = tmp_path / "prepared" if mode == "prepared" else None
    if prepared is not None:
        preparation = admission.check_defaults(
            admission.NamedRegistry("", "https://registry.invalid"),
            engine_version="0.0.3",
            candidate_base="b" * 40,
            prepare_output=prepared,
            **context,
        )
        assert preparation["groups"] == [["dinkster-text"]]
    if published == "missing":
        with pytest.raises(ValueError, match=r"dinkster-test-helpers.*>=0\.0\.1"):
            admission.check_defaults(
                admission.NamedRegistry("", registry_url), engine_version="0.0.3", **context
            )
        assert ordinary_requests == []
        return
    if published == "installer":
        if sys.platform == "win32":
            pytest.skip("probe installer requires POSIX process limits")
        with pytest.raises(ValueError, match="probe.dependencies-incompatible") as refusal:
            admission.check_defaults(
                admission.NamedRegistry("", registry_url), engine_version="0.0.3", **context
            )
        assert "dinkster-unavailable-test-library" in str(refusal.value)
        assert "not satisfied by the pack lock" in str(refusal.value)
        assert ordinary_requests == []
        return
    # Windows retains the service's isolation refusal instead of running a POSIX-only probe.
    with (
        pytest.raises(
            ValueError,
            match=r"default dinkster-text@0\.0\.1 admission refused: "
            r"probe\.isolation-unavailable: A full Linux probe jail is unavailable\.",
        )
        if sys.platform == "win32"
        else nullcontext()
    ):
        result = admission.check_defaults(
            admission.NamedRegistry("", registry_url),
            engine_version="0.0.3",
            candidate_base="b" * 40 if mode == "candidate" else None,
            published_sources=publication,
            prepared_sources=prepared,
            test_packs=("dinkster-text",) if prepared is not None else (),
            **context,
        )
        assert result == {
            "engineVersion": "0.0.3",
            "packs": [
                {
                    "pack": entry.pack,
                    "version": entry.version,
                    "artifactDigest": entry.artifact_digest,
                }
                for entry in ((entries[0],) if prepared is not None else (entries[1], entries[0]))
            ],
        }
    output = capfd.readouterr().out
    assert "selected dinkster-test-helpers@0.0.1" in output
    assert ("Testing dinkster-text@0.0.1" in output) is (sys.platform != "win32")
    assert "Testing dinkster-test-helpers" not in output
    assert ordinary_requests == []
    if isinstance(published, str):
        return
    with ZipFile(wheels / "dinkster_test_helpers-0.0.1-py3-none-any.whl", "a") as wheel:
        wheel.writestr("changed.txt", "library drift")
    if prepared is not None:
        digest = entries[1].artifact_digest.removeprefix("blake3:")
        shutil.copyfile(
            wheels / "dinkster_test_helpers-0.0.1-py3-none-any.whl",
            prepared / "artifacts" / f"{digest}.zip",
        )
    with pytest.raises(
        ValueError,
        match="dinkster-test-helpers: locked outer wheel differs from the archive"
        if prepared is not None
        else "library dinkster-test-helpers size or digest mismatch",
    ):
        admission.check_defaults(
            admission.NamedRegistry("", "https://registry.invalid"),
            engine_version="0.0.3",
            candidate_base="b" * 40 if mode == "candidate" else None,
            published_sources=publication,
            prepared_sources=prepared,
            test_packs=("dinkster-text",) if prepared is not None else (),
            **context,
        )


@pytest.mark.parametrize("mismatched_lock", [False, True])
def test_locked_default_admission_uses_verified_unpublished_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched_lock: bool
) -> None:
    from dataclasses import replace

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_store import pack_archive_path
    from dinkster_workers.pack_lock import LockedDistribution
    from dinkster_workers.pack_wheel import build_pack_archive
    from test_installer import _serve_artifacts
    from test_pack_environment import _generation, _wheel

    from scripts import check_default_pack_admission as admission

    requests = []
    server = _serve_artifacts({}, requests)
    try:
        sources = tmp_path / "sources"
        sources.mkdir()
        helper = _wheel(sources, "test-helper", "0.0.1", "test_helper", "VALUE = 37\n")
        helper = replace(helper, url=f"http://127.0.0.1:{server.server_port}/{helper.filename}")
        pack = tmp_path / "pack"
        pack.mkdir()
        (pack / "dinkster-pack.toml").write_text(
            '[pack]\nname = "test-consumer"\nversion = "0.0.2"\npublisher = "dinkster"\n'
            'engine = ">=0.0.2,<0.1"\nrequires = ["test-helper==0.0.1"]\n'
            '[pack.entry]\nnodes = "test_consumer:NODES"\n'
            '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        )
        module = pack / "test_consumer"
        module.mkdir()
        (module / "__init__.py").write_text(
            "from test_helper import VALUE\nassert VALUE == 37\nNODES = []\n"
        )
        consumer = sources / "test_consumer-0.0.2-py3-none-any.whl"
        build_pack_archive(pack, consumer)
        prepared = tmp_path / "prepared"
        artifacts = prepared / "artifacts"
        artifacts.mkdir(parents=True)
        lock_output = prepared / "pack-locks"
        lock_output.mkdir()
        entries = []
        for name, version, archive in (
            ("test-helper", "0.0.1", sources / helper.filename),
            ("test-consumer", "0.0.2", consumer),
        ):
            data = archive.read_bytes()
            entry = LockedPack(name, version, artifact_digest(data), "dinkster", ())
            entries.append(entry)
            pack_archive_path(artifacts, entry.artifact_digest).write_bytes(data)
            outer = admission.LockedWheel(
                archive.name, len(data), hashlib.sha256(data).hexdigest(), archive.as_uri()
            )
            dependency = replace(helper, sha256="f" * 64) if mismatched_lock else helper
            packages = (LockedDistribution(name, version, (outer,)),)
            if name == "test-consumer":
                packages += (LockedDistribution("test-helper", "0.0.1", (dependency,)),)
            PackLock(name, "linux-cpu", ">=3.12", (), (), packages).write(
                lock_output / f"{name}.json"
            )
        (prepared / "catalog.lock").write_text(Lockfile.of(entries).record_json())
        base_id = "sha256:" + "a" * 64
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(admission.__file__).resolve().parents[1],
            text=True,
        ).strip()
        (prepared / "engine.json").write_text(
            json.dumps(
                {
                    "commit": head,
                    "engineVersion": "0.0.3",
                    "baseId": base_id,
                    "variant": "linux-cpu",
                }
            )
        )
        constraints = tmp_path / "constraints.txt"
        constraints.touch()
        python = _generation(tmp_path / "generation/python", [])
        monkeypatch.setattr(admission, "default_pack_minimums", lambda: {"test-consumer": "0.0.2"})
        monkeypatch.setattr(admission, "run_selected_tests", lambda *args, **kwargs: None)
        executed = []

        def probe(root, settings, require_isolation, *, python, source_roots):
            value = subprocess.check_output(
                [str(python), "-I", "-c", "import test_consumer; print(test_consumer.VALUE)"],
                text=True,
            )
            executed.append(int(value))
            return admission.StageResult()

        # Keep real locked provisioning and imports independent of the jail transport.
        monkeypatch.setattr("dinkster_registry.archive_admission._run_executable_probe", probe)
        with (
            pytest.raises(ValueError, match="download/verification failed")
            if mismatched_lock
            else nullcontext()
        ):
            result = admission.check_defaults(
                admission.NamedRegistry("", "https://registry.invalid"),
                engine_version="0.0.3",
                prepared_sources=prepared,
                generation_python=python,
                base_id=base_id,
                constraints=constraints,
                test_packs=("test-consumer",),
            )
            assert result["packs"] == [
                {
                    "pack": "test-consumer",
                    "version": "0.0.2",
                    "artifactDigest": entries[1].artifact_digest,
                }
            ]
        assert bool(requests) is mismatched_lock
        assert executed == ([] if mismatched_lock else [37])
    finally:
        server.shutdown()


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("selected_engine", [False, True])
def test_probe_index_stages_selected_library_dependency_without_public_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependent: bool, selected_engine: bool
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from test_pack_environment import _wheel

    from scripts.stage_publication_probe_index import stage_index

    monkeypatch.setenv("UV_NO_INDEX", "1")
    sources = tmp_path / "selected %20 sources"
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
    engine = None
    if selected_engine:
        engine = tmp_path / "engine"
        engine.mkdir()
        api = _wheel(
            engine,
            "dinkster-api",
            "0.0.3",
            "dinkster_api",
            "raise RuntimeError('API metadata must not execute')\n",
            requires=("test-helper==0.0.1",),
        )
        with ZipFile(engine / api.filename, "a") as archive:
            archive.writestr("dinkster_workers/scaffold/dinkster-pack.toml", "[pack]\n")
    output = tmp_path / "index"
    result = stage_index(sources, output, engine=engine)
    if dependent or selected_engine:
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


def test_probe_index_refuses_local_dependency_outside_selected_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry import Lockfile

    from scripts.stage_publication_probe_index import stage_index

    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "official-packs.lock").write_text(Lockfile.of(()).record_json())
    outside = tmp_path / "outside %20 dependency.whl"
    outside.write_bytes(b"unselected wheel")
    output = tmp_path / "index"

    def resolve(*args, **kwargs):
        (output / "pylock.toml").write_text(
            f'[[packages]]\nname = "outside"\n[[packages.wheels]]\nurl = "{outside.as_uri()}"\n'
        )

    monkeypatch.setattr("scripts.stage_publication_probe_index.subprocess.run", resolve)
    with pytest.raises(ValueError, match="local probe wheel must be a selected source"):
        stage_index(sources, output)
    assert not (output / outside.name).exists()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "dependency", ["consumers", "transitive", "extras", "missing-extra", "fixed"]
)
def test_admission_library_closure_intersects_ranges_and_activates_extras(
    tmp_path: Path, dependency: str, reverse: bool
) -> None:
    from dataclasses import replace

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_store import pack_archive_path
    from dinkster_workers.pack_wheel import build_pack_archive
    from test_installer import _serve_artifacts, _static_release
    from test_pack_environment import _wheel

    from scripts.check_default_pack_admission import NamedRegistry, prepare_library_sources

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    rows = []
    files = {}
    library_entries = {}
    for name, versions in (("library-a", ("1.5", "2.0")), ("library-z", ("1.0",))):
        for version in versions:
            requires = ()
            if name == "library-a" and dependency in {"extras", "missing-extra"}:
                requires = ('library-z>=1; extra == "helpers"',)
            if name == "library-z" and dependency == "transitive":
                requires = ("library-a<2",)
            wheel = _wheel(wheels, name, version, name.replace("-", "_"), "", requires=requires)
            data = (wheels / wheel.filename).read_bytes()
            entry = LockedPack(name, version, artifact_digest(data), "dinkster", ())
            library_entries[name, version] = entry
            if name == "library-z" and dependency == "missing-extra":
                continue
            rows.append({**_static_release(entry, data), "library": True})
            files[f"/static/wheels/{wheel.sha256}/{wheel.filename}"] = data
    files["/static/packs/index.json"] = json.dumps(
        {
            "schemaVersion": 2,
            "items": [
                {
                    "name": name,
                    "latestVersion": version,
                    "publisher": "dinkster",
                    "library": True,
                    "url": f"packs/{name}/index.json",
                }
                for name, version in (("library-a", "2.0"), ("library-z", "1.0"))
            ],
        }
    ).encode()
    for name in ("library-a", "library-z"):
        files[f"/static/packs/{name}/index.json"] = json.dumps(
            {
                "schemaVersion": 2,
                "name": name,
                "items": [row for row in rows if row["name"] == name],
            }
        ).encode()
    requirements = ["library-a>=1", "library-a<2"]
    if dependency in {"extras", "missing-extra"}:
        requirements = ["library-a[helpers]>=1", "library-a>=1"]
    elif dependency == "transitive":
        requirements = ["library-a>=1", "library-z>=1"]
    elif dependency == "fixed":
        requirements = ["library-a<2", "library-a>=1"]
    if reverse:
        requirements.reverse()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    entries = []
    for number, requirement in enumerate(requirements):
        name = f"consumer-{number}"
        source = tmp_path / name
        source.mkdir()
        module = name.replace("-", "_")
        (source / module).mkdir()
        (source / module / "__init__.py").write_text("NODES = []\n")
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\nversion = "1.0"\npublisher = "dinkster"\n'
            f'engine = ">=0.0.2,<0.1"\nrequires = ["{requirement}"]\n'
            f'[pack.entry]\nnodes = "{module}:NODES"\n'
            '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        )
        archive = wheels / f"{name.replace('-', '_')}-1.0-py3-none-any.whl"
        build_pack_archive(source, archive)
        entry = LockedPack(name, "1.0", artifact_digest(archive.read_bytes()), "dinkster", ())
        pack_archive_path(artifacts, entry.artifact_digest).write_bytes(archive.read_bytes())
        entries.append(entry)
    if dependency == "fixed":
        entry = replace(library_entries["library-a", "2.0"], source="git:fixed-publication")
        pack_archive_path(artifacts, entry.artifact_digest).write_bytes(
            (wheels / "library_a-2.0-py3-none-any.whl").read_bytes()
        )
        entries.append(entry)
    server = _serve_artifacts(files, [])
    try:
        registry = NamedRegistry("", f"http://127.0.0.1:{server.server_port}")
        if dependency in {"missing-extra", "fixed"}:
            with pytest.raises(ValueError, match=r"library-(z.*>=1|a.*<2)"):
                prepare_library_sources(registry, Lockfile.of(entries), artifacts, tmp_path)
        else:
            selected = prepare_library_sources(registry, Lockfile.of(entries), artifacts, tmp_path)
            versions = {pack.pack: pack.version for pack in selected.packs}
            assert versions["library-a"] == ("2.0" if dependency == "extras" else "1.5")
            assert versions.get("library-z") == (
                "1.0" if dependency in {"extras", "transitive"} else None
            )
    finally:
        server.shutdown()
