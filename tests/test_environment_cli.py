"""Desktop and humans share the actual subprocess result contract."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import marshal
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
import zstandard
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_registry import Lockfile
from dinkster_store import object_path
from dinkster_supervisor.environment_artifacts import acquire_sources, discover
from dinkster_supervisor.environment_state import Journal, content_digest, writer_lock
from supervisor_subprocess import run as run_with_diagnostics
from test_environment_artifacts import _artifact, _json, _mirror, _release_sources, _wheel_payload
from test_environment_generations import _assert_shared, _workspace_tree
from test_environment_runtime import free_port, make_generation
from test_environment_state import _generation

from tests.fixtures.environments.build import channel_json as _channel_json
from tests.fixtures.environments.build import executable_release as _executable_release
from tests.fixtures.environments.build import read_channel

pytestmark = pytest.mark.usefixtures("signed_channels")


def _run(root: Path, *arguments: str, timeout: float = 120) -> tuple[int, dict]:
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(root),
        *arguments,
        "--json",
    ]
    if os.environ.get("DINKSTER_TEST_FORBID_RESOLUTION"):
        command[1:3] = [
            "-c",
            """
import runpy, sys
from pathlib import Path
from dinkster_supervisor import environment_local
def forbidden(*args, **kwargs):
    raise AssertionError("published paths must not resolve locally")
environment_local.prepare_local = forbidden
def audit(event, args):
    if event != "subprocess.Popen":
        return
    command = args[1]
    if isinstance(command, str):
        import ctypes
        count = ctypes.c_int()
        parse = ctypes.windll.shell32.CommandLineToArgvW
        parse.restype = ctypes.POINTER(ctypes.c_wchar_p)
        arguments = parse(ctypes.c_wchar_p(command), ctypes.byref(count))
        try:
            command = arguments[:count.value]
        finally:
            ctypes.windll.kernel32.LocalFree(ctypes.cast(arguments, ctypes.c_void_p))
    executable = args[0] or command[0]
    if Path(executable).stem.lower() != "uv":
        return
    if "compile" in command:
        forbidden()
    if "sync" in command or "install" in command:
        if "--offline" not in command or "--no-index" not in command:
            forbidden()
sys.addaudithook(audit)
runpy.run_module("dinkster_supervisor.environment", run_name="__main__")
""",
        ]
    result = run_with_diagnostics(command, timeout=timeout)
    assert result.stdout, result.stderr
    document = json.loads(result.stdout)
    assert document["format"] == "dinkster.environment-result/1"
    assert document["root"] == str(root)
    return result.returncode, document


@pytest.mark.parametrize("api", [1, 2, 4, 5])
@pytest.mark.parametrize("version", ["0.2.0", "0.2.0.dev423"])
def test_explicit_frontend_version_checks_verified_release_api_without_project_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api: int,
    version: str,
) -> None:
    from dinkster_supervisor import environment, environment_downloads

    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    _, engine, channel = _mirror(mirror)
    engine["schemaApi"] = api
    data = _json(engine) + b"\n"
    identity = hashlib.sha256(data).hexdigest()
    old = mirror / "engine" / channel["engine"]["release"][7:]
    release = mirror / "engine" / identity
    shutil.copytree(old, release)
    (release / "release-manifest.json").write_bytes(data)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"][0]["path"] = f"engine/{identity}/release-manifest.json"
    (mirror / "channels/nightly.json").write_bytes(_channel_json(channel))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "dinkster_frontend/manifest.json",
            json.dumps(
                {
                    "format": "dinkster.frontend/1",
                    "repository": "Kosinkadink/Dinkster-Frontend",
                    "version": version,
                    "engineApi": {"min": 2, "max": 4},
                }
            ),
        )
        archive.writestr("dinkster_frontend/static/index.html", "selected frontend")
    filename = f"dinkster_frontend-{version}-py3-none-any.whl"
    wheel = tmp_path / filename
    wheel.write_bytes(payload.getvalue())
    digest = hashlib.sha256(payload.getvalue()).hexdigest()
    selected = {
        "filename": filename,
        "url": "https://files.pythonhosted.org/packages/" + filename,
        "digests": {"sha256": digest},
        "size": wheel.stat().st_size,
    }
    read, fetch = environment_downloads.read_metadata, environment_downloads.fetch_artifact

    def metadata(location: str, path: str) -> bytes:
        if location == "https://pypi.org/pypi/dinkster-frontend":
            assert path == f"{version}/json"
            return json.dumps({"urls": [selected]}).encode()
        return read(location, path)

    def transfer(locations, name, destination, sha, size):
        assert locations == ("https://files.pythonhosted.org/packages",)
        return fetch((str(tmp_path),), name, destination, sha, size)

    monkeypatch.setattr(environment_downloads, "read_metadata", metadata)
    monkeypatch.setattr(environment_downloads, "fetch_artifact", transfer)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    arguments = [
        "--root",
        str(root),
        "setup",
        "--channel",
        "nightly",
        "--mirror",
        str(mirror),
        "--variant",
        "fixture-cpu",
        "--data",
        str(store),
        "--plan",
        "--json",
    ]

    def plan(*extra: str) -> tuple[int, dict]:
        with pytest.raises(SystemExit) as exited:
            environment.main([*arguments, *extra])
        assert isinstance(exited.value.code, int)
        return exited.value.code, json.loads(capsys.readouterr().out)

    code, result = plan("--frontend-version", version)
    if api in {1, 5}:
        assert code == 1
        assert "supports engine schema API 2 through 4" in result["error"]
    else:
        assert code == 0
        assert result["frontend"]["sha256"] == digest
        assert result["frontend"]["engineApi"] == {"min": 2, "max": 4}
        assert any(filename in row["path"] for row in result["runtimeInputs"])
        assert any(identity in row["path"] for row in result["runtimeInputs"])
        assert any(f"explicit published frontend {version}" in line for line in result["plan"])
        approval = result["frontendApproval"]
        assert approval.startswith("sha256:") and len(approval) == 71
        code, confirmed = plan("--frontend-version", version, "--frontend-approval", approval)
        assert code == 0 and confirmed["frontend"] == result["frontend"]
        code, refused = plan(
            "--frontend-version", version, "--frontend-approval", "sha256:" + "0" * 64
        )
        assert code == 1 and "changed after planning" in refused["error"]
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    code, result = plan("--frontend-version", version)
    assert code == 1
    assert "size" in result["error"] or "digest" in result["error"]
    code, result = plan()
    assert code == 0
    assert "frontend" not in result
    assert not any(filename in row["path"] for row in result["runtimeInputs"])
    assert not root.exists() and not store.exists()
    assert not (tmp_path / "user-data").exists()


def test_managed_git_pack_replaces_only_selected_sources_and_rolls_back_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_server
) -> None:
    import pygit2
    from dinkster_registry import LockedPack, build_artifact
    from dinkster_store import object_entries, pack_archive_path
    from dinkster_supervisor.environment_state import Generation, interpreter
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.pack_lock import PackLock
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES

    from tests.fixtures.environments import build as publisher

    original_wheel, encode = publisher.wheel, publisher.canonical_json
    monkeypatch.setattr(
        publisher, "wheel", lambda *args, **kwargs: original_wheel(*args, **kwargs, pack_tools=True)
    )
    monkeypatch.setattr(publisher, "standalone_tree", _workspace_tree)

    def encoded(document):
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        return encode(document)

    monkeypatch.setattr(publisher, "canonical_json", encoded)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = publisher.build(tmp_path / "fixture", "fixture-cpu", free_port())
    root = Path(fixture["root"])
    store = Path(fixture["store"])
    empty = tmp_path / "initial.lock"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    retained = tmp_path / "retained"
    retained.mkdir()
    (retained / "dinkster-pack.toml").write_text(
        HEALTHY_MANIFEST.replace("healthy", "retained").replace(
            'requires = ["numpy>=1.26"]', 'version = "1.0"\nengine = ">=0"\nrequires = []'
        )
    )
    (retained / "retained_nodes.py").write_text(HEALTHY_NODES.replace("healthy", "retained"))
    archive = artifacts / "retained.zip"
    retained_digest = build_artifact(retained, archive)
    archive.rename(artifacts / f"{retained_digest[7:]}.zip")
    retained_entry = LockedPack(
        "retained-nodes", "1.0", retained_digest, "dinkster", ("retained",), "registry:official"
    )
    empty.write_text(Lockfile.of([retained_entry]).record_json())
    dependency = tmp_path / "retained.pack-lock.json"
    PackLock("retained-nodes", "fixture-cpu", ">=3.12", (), (), ()).write(dependency)
    setup = fixture["commands"]["setup"]
    code, result = _run(
        root,
        *setup[setup.index("setup") : -1],
        "--catalog-lock",
        str(empty),
        "--artifacts",
        str(artifacts),
        "--pack-lock",
        str(dependency),
    )
    assert code == 0, result
    assert _run(root, "stop", "--yes")[0] == 0
    original = Generation.read(root / "builds/1/build.json")
    before = {
        path: content_digest(path)
        for parent in ("bases", "wheels")
        for path in (store / parent).rglob("*")
        if path.is_file() and path.name != ".objects.json"
    }
    repository = pygit2.Repository(git_server["root"] / "source.git")
    signature = pygit2.Signature("Fixture", "fixture@example.invalid", 1700000000, 0)
    commits = []
    for branch, value in (("managed-pack", 31), ("managed-pack-new", 79)):
        tree = pygit2.Index()
        for name, payload in {
            "dinkster-pack.toml": HEALTHY_MANIFEST.replace(
                'requires = ["numpy>=1.26"]', 'version = "1.0"\nengine = ">=0"\nrequires = []'
            ),
            "healthy_nodes.py": HEALTHY_NODES + f"\nVALUE = {value}\n",
        }.items():
            tree.add(
                pygit2.IndexEntry(
                    "nodes/" + name,
                    repository.create_blob(payload.encode()),
                    pygit2.enums.FileMode.BLOB,
                )
            )
        commits.append(
            repository.create_commit(
                "refs/heads/" + branch,
                signature,
                signature,
                branch,
                tree.write_tree(repository),
                commits[-1:],
            )
        )
    repository.free()
    endpoint = "https://github.com/Kosinkadink/source#subdirectory=nodes"
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_OFFLINE", "1")
    tools = tmp_path / "no-git-tools"
    tools.mkdir()
    uv = shutil.which("uv")
    assert uv is not None
    shutil.copyfile(uv, tools / Path(uv).name)
    (tools / Path(uv).name).chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    assert shutil.which("git") is None
    request_count = len(git_server["requests"])
    tunnel_count = len(git_server["tunnels"])
    try:
        code, refused = _run(
            root,
            "source",
            "add",
            endpoint,
            "--pack",
            "wrong-pack",
            "--ref",
            "managed-pack",
            "--git-mirror",
            git_server["proxy"],
            "--yes",
        )
        assert code == 1 and "different pack" in refused["error"], refused
        assert sorted(path.name for path in (root / "builds").iterdir()) == ["1"]
        assert _run(root, "status")[1]["current"] == 1
        for branch, expected, commit in (
            ("managed-pack", 31, commits[0]),
            ("managed-pack-new", 79, commits[1]),
        ):
            arguments = (
                ["add", endpoint, "--pack", "healthy-nodes", "--git-mirror", git_server["proxy"]]
                if expected == 31
                else ["update", "--pack", "healthy-nodes"]
            )
            selection = ("source", *arguments, "--ref", branch, "--package-index", index.as_uri())
            code, preview = _run(root, *selection, "--plan")
            assert code == 0, preview
            approval = preview["sourceApproval"]
            code, same = _run(root, *selection, "--source-approval", approval, "--plan")
            assert code == 0 and same["sourceApproval"] == approval, same
            code, refused = _run(
                root,
                *selection,
                "--package-index",
                index.as_uri() + "/changed",
                "--source-approval",
                approval,
                "--yes",
            )
            assert code == 1 and "source inputs changed" in refused["error"], refused
            code, refused = _run(
                root,
                *selection,
                "--ref",
                "managed-pack-new" if expected == 31 else "managed-pack",
                "--source-approval",
                approval,
                "--yes",
            )
            assert code == 1 and "source inputs changed" in refused["error"], refused
            assert _run(root, "status")[1]["current"] == preview["current"]
            code, result = _run(
                root, *selection, "--source-approval", approval, "--yes", timeout=180
            )
            assert code == 0, result
            current = root / "builds" / str(result["current"])
            record = Generation.read(current / "build.json")
            assert record.source == "source" and record.engine_release == original.engine_release
            assert record.base == original.base
            selected = Lockfile.from_record_json((current / "dinkster.lock").read_text())
            assert (
                next(entry for entry in selected.packs if entry.pack == "retained-nodes")
                == retained_entry
            )
            assert (current / "packs/retained-nodes/pack-lock.json").read_bytes() == (
                root / "builds/1/packs/retained-nodes/pack-lock.json"
            ).read_bytes()
            status = _run(root, "status")[1]
            assert status["sources"]["packs"]["healthy-nodes"]["commit"] == str(commit)
            assert status["published"] is False and status["reproducible"] is False
            assert record.drift(current) == ()
            assert (
                (current / "pack-sources/healthy-nodes/healthy_nodes.py")
                .read_text()
                .endswith(f"VALUE = {expected}\n")
            )
            assert subprocess.check_output(
                [
                    str(interpreter(current)),
                    "-I",
                    "-B",
                    "-c",
                    "import runpy, sys; print(runpy.run_path(sys.argv[1])['VALUE'])",
                    str(current / "pack-sources/healthy-nodes/healthy_nodes.py"),
                ],
                text=True,
            ).strip() == str(expected)
            assert {path: content_digest(path) for path in before} == before
            assert _run(root, "stop", "--yes")[0] == 0
        repository = pygit2.Repository(git_server["root"] / "source.git")
        repository.references["refs/heads/managed-pack"].set_target(commits[1])
        repository.free()
        first_lock = Lockfile.from_record_json((root / "builds/2/dinkster.lock").read_text())
        first_digest = next(
            entry.artifact_digest for entry in first_lock.packs if entry.pack == "healthy-nodes"
        )
        archives = [
            archive
            for _, engine in object_entries(store / "engines")
            if (archive := pack_archive_path(engine / "artifacts", first_digest)).is_file()
        ]
        assert archives
        for path in archives:
            path.unlink()
        code, restored = _run(root, "rollback", "--yes", timeout=180)
        assert code == 0, restored
        current = root / "builds" / str(restored["current"])
        assert (
            (current / "pack-sources/healthy-nodes/healthy_nodes.py")
            .read_text()
            .endswith("VALUE = 31\n")
        )
        assert _run(root, "status")[1]["sources"]["packs"]["healthy-nodes"]["commit"] == str(
            commits[0]
        )
        assert _run(root, "stop", "--yes")[0] == 0
        code, updated = _run(
            root,
            "update",
            "--catalog-lock",
            str(current / "dinkster.lock"),
            "--artifacts",
            str(object_path(store / "engines", original.engine_release) / "artifacts"),
            "--pack-lock",
            str(current / "packs/healthy-nodes/pack-lock.json"),
            "--pack-lock",
            str(current / "packs/retained-nodes/pack-lock.json"),
            "--yes",
            timeout=180,
        )
        assert code == 0, updated
        status = _run(root, "status")[1]
        assert status["sources"]["packs"]["healthy-nodes"]["commit"] == str(commits[0])
        assert status["source"] == "source" and status["published"] is False
        updated_record = Generation.read(root / "builds" / str(updated["current"]) / "build.json")
        assert updated_record.source == "source"
        code, history = _run(root, "history")
        assert code == 0, history
        selected_history = history["history"][-1]
        assert selected_history["sources"]["packs"]["healthy-nodes"]["commit"] == str(commits[0])
        assert selected_history["published"] is False
        assert selected_history["reproducible"] is False
        assert _run(root, "stop", "--yes")[0] == 0
        code, released = _run(root, "source", "release", "--yes", timeout=180)
        assert code == 0, released
        released_root = root / "builds" / str(released["current"])
        released_record = Generation.read(released_root / "build.json")
        assert released_record.source == "published"
        assert not (released_root / "pack-sources/healthy-nodes").exists()
        assert "sources" not in _run(root, "status")[1]
        assert len(git_server["requests"]) > request_count
        assert all(
            path.startswith("/Kosinkadink/source.git/")
            for path in git_server["requests"][request_count:]
        )
        assert len(git_server["tunnels"]) == tunnel_count
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


@pytest.fixture
def managed_git_source_project(tmp_path: Path, monkeypatch, git_server):
    import pygit2
    from dinkster_supervisor.environment_state import Generation
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES

    from tests.fixtures.environments import build as publisher

    original_wheel, encode = publisher.wheel, publisher.canonical_json
    monkeypatch.setattr(
        publisher, "wheel", lambda *args, **kwargs: original_wheel(*args, **kwargs, pack_tools=True)
    )
    monkeypatch.setattr(publisher, "standalone_tree", _workspace_tree)
    published_base = {}

    def encoded(document):
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        if document.get("format") == "dinkster.base/1" and "id" in document:
            published_base.update(document)
        if document.get("format") == "dinkster.engine/1":
            document["schemaApi"] = 1
            fields = ("name", "version", "filename", "size", "sha256", "index", "url")
            document["dependencies"] = {
                published_base["variant"]: {
                    "base": published_base["id"],
                    "wheels": [
                        {key: row[key] for key in fields} for row in published_base["wheels"]
                    ],
                }
            }
        return encode(document)

    monkeypatch.setattr(publisher, "canonical_json", encoded)
    for name in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path / "user"))
    fixture = publisher.build(tmp_path / "fixture", "linux-cpu", free_port())
    root, store = Path(fixture["root"]), Path(fixture["store"])
    catalog = tmp_path / "empty.lock"
    catalog.write_text(Lockfile.of([]).record_json())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    code, result = _run(
        root,
        "setup",
        "--mirror",
        fixture["mirror"],
        "--data",
        fixture["store"],
        "--variant",
        "linux-cpu",
        "--channel",
        "nightly",
        "--port",
        str(fixture["port"]),
        "--catalog-lock",
        str(catalog),
        "--artifacts",
        str(artifacts),
        "--yes",
    )
    assert code == 0, result
    assert _run(root, "stop", "--yes")[0] == 0
    original = Generation.read(root / "builds/1/build.json")
    before = {
        path: content_digest(path)
        for path in (object_path(store / "bases", original.base)).rglob("*")
        if path.is_file()
    }
    original_wheels = json.loads((root / "builds/1/installed-wheels.json").read_text())["wheels"]
    repository = pygit2.Repository(git_server["root"] / "source.git")
    signature = pygit2.Signature("Fixture", "fixture@example.invalid", 1700000000, 0)
    commits = []
    for branch, value in (("library-old", 31), ("library-new", 79), ("library-bad", 101)):
        tree = pygit2.Index()
        for folder, name, module, requirements in (
            ("inference", "dinkster-inference", "dinkster_inference", '["fixture-base==1.0"]'),
            ("helper", "fixture-helper", "fixture_helper", '["dinkster-inference==9.0"]'),
            ("bad", "fixture-helper", "fixture_helper", '["fixture-base==999"]'),
            ("base", "fixture-base", "fixture_base", "[]"),
        ):
            for filename, payload in {
                "pyproject.toml": (
                    f'[project]\nname="{name}"\nversion="9.0"\ndependencies={requirements}\n'
                    '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
                ),
                f"{module}/__init__.py": f"VALUE = {value}\n",
            }.items():
                tree.add(
                    pygit2.IndexEntry(
                        folder + "/" + filename,
                        repository.create_blob(payload.encode()),
                        pygit2.enums.FileMode.BLOB,
                    )
                )
        for filename, payload in {
            "nodes/dinkster-pack.toml": HEALTHY_MANIFEST.replace(
                'requires = ["numpy>=1.26"]', 'version = "1.0"\nengine = ">=0"\nrequires = []'
            ),
            "nodes/healthy_nodes.py": HEALTHY_NODES + f"\nVALUE = {value}\n",
            "frontend/package.json": '{"version":"0.2.0"}',
            "frontend/frontend-release.json": '{"engineApi":{"min":1,"max":2}}',
        }.items():
            tree.add(
                pygit2.IndexEntry(
                    filename, repository.create_blob(payload.encode()), pygit2.enums.FileMode.BLOB
                )
            )
        commits.append(
            repository.create_commit(
                None,
                signature,
                signature,
                branch,
                tree.write_tree(repository),
                [],
            )
        )
        repository.references.create("refs/heads/" + branch, commits[-1], force=True)
    repository.free()
    endpoint = "https://github.com/Kosinkadink/source#subdirectory="
    try:
        yield SimpleNamespace(
            fixture=fixture,
            root=root,
            store=store,
            original=original,
            before=before,
            original_wheels=original_wheels,
            signature=signature,
            commits=commits,
            endpoint=endpoint,
            original_wheel=original_wheel,
        )
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


def _select_git_libraries(project, git_server):
    root, endpoint = project.root, project.endpoint
    for folder, name in (("inference", "dinkster-inference"), ("helper", "fixture-helper")):
        code, result = _run(
            root,
            "source",
            "add",
            endpoint + folder,
            "--library",
            name,
            "--ref",
            "library-old",
            "--git-mirror",
            git_server["proxy"],
            "--yes",
            timeout=180,
        )
        assert code == 0, result
        assert _run(root, "stop", "--yes")[0] == 0
    first_root = root / "builds" / str(result["current"])
    return result["current"], json.loads((first_root / "engine-manifest.json").read_text())


@pytest.mark.parametrize(
    "replacement", ["git-ref", "published-wheel", "incompatible-wheel", "missing-wheel"]
)
def test_managed_git_libraries_retain_release_base_and_exact_rollback(
    managed_git_source_project, git_server, replacement, request, tmp_path, monkeypatch, capsys
):
    import pygit2
    from dinkster_supervisor.environment_state import Generation, interpreter, read_document

    if replacement != "git-ref":
        _assert_git_library_replacement(
            request.getfixturevalue("managed_git_selected_sources"),
            tmp_path,
            monkeypatch,
            git_server,
            capsys,
            replacement,
        )
        return
    project = managed_git_source_project
    root, endpoint, original = project.root, project.endpoint, project.original
    before, original_wheels, commits = project.before, project.original_wheels, project.commits
    try:
        code, preview = _run(
            root,
            "source",
            "add",
            endpoint + "inference",
            "--library",
            "dinkster-inference",
            "--ref",
            "library-old",
            "--git-mirror",
            git_server["proxy"],
            "--plan",
        )
        assert code == 0 and preview["status"] == "planned", preview
        assert sorted(path.name for path in (root / "builds").iterdir()) == ["1"]
        code, refused = _run(
            root,
            "source",
            "add",
            endpoint + "base",
            "--library",
            "--ref",
            "library-old",
            "--git-mirror",
            git_server["proxy"],
            "--yes",
        )
        assert code == 1 and "release-base" in refused["error"], refused
        first, first_manifest = _select_git_libraries(project, git_server)
        selection = ("source", "update", "--library", "dinkster-inference", "--ref", "library-new")
        code, preview = _run(root, *selection, "--plan")
        assert code == 0, preview
        approval = preview["sourceApproval"]
        settings_path = root / "environment.json"
        settings = settings_path.read_bytes()
        changed_settings = json.loads(settings)
        changed_settings["port"] += 1
        settings_path.write_text(json.dumps(changed_settings))
        code, refused = _run(root, *selection, "--source-approval", approval, "--yes")
        assert code == 1 and "source inputs changed" in refused["error"], refused
        assert _run(root, "status")[1]["current"] == first
        settings_path.write_bytes(settings)
        code, result = _run(root, *selection, "--source-approval", approval, "--yes", timeout=180)
        assert code == 0, result
        current = root / "builds" / str(result["current"])
        record = Generation.read(current / "build.json")
        assert record.base == original.base and record.source == "source"
        assert record.engine_version == original.engine_version and record.drift(current) == ()
        for module, expected in (("dinkster_inference", 79), ("fixture_helper", 31)):
            assert subprocess.check_output(
                [
                    str(interpreter(current)),
                    "-I",
                    "-B",
                    "-c",
                    f"import {module}; print({module}.VALUE)",
                ],
                text=True,
            ).strip() == str(expected)
        status = _run(root, "status")[1]
        assert status["published"] is False and status["reproducible"] is False
        assert (
            status["sources"]["libraries"]["fixture-helper"]
            == first_manifest["sources"]["libraries"]["fixture-helper"]
        )
        assert status["sources"]["libraries"]["dinkster-inference"]["commit"] == str(commits[1])
        for wheel in original_wheels:
            assert wheel in read_document(current / "installed-wheels.json")["wheels"]
        assert {path: content_digest(path) for path in before} == before
        assert _run(root, "stop", "--yes")[0] == 0
        code, refused = _run(
            root,
            "source",
            "add",
            endpoint + "bad",
            "--library",
            "fixture-helper",
            "--ref",
            "library-bad",
            "--git-mirror",
            git_server["proxy"],
            "--yes",
            timeout=180,
        )
        assert code == 1, refused
        assert _run(root, "status")[1]["current"] == result["current"]
        repository = pygit2.Repository(git_server["root"] / "source.git")
        repository.references["refs/heads/library-old"].set_target(commits[1])
        repository.free()
        code, restored = _run(root, "rollback", "--yes", timeout=180)
        assert code == 0, restored
        restored_root = root / "builds" / str(restored["current"])
        assert read_document(restored_root / "engine-manifest.json") == first_manifest
        assert (
            subprocess.check_output(
                [
                    str(interpreter(restored_root)),
                    "-I",
                    "-B",
                    "-c",
                    "import dinkster_inference; print(dinkster_inference.VALUE)",
                ],
                text=True,
            ).strip()
            == "31"
        )
        assert _run(root, "stop", "--yes")[0] == 0
        history = _run(root, "history")[1]["history"]
        assert history[-1]["sources"]["libraries"]["dinkster-inference"]["commit"] == str(
            commits[0]
        )
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


def test_managed_git_pack_and_frontend_source_approval(
    managed_git_source_project, tmp_path, monkeypatch, git_server, capsys
):
    from dinkster_supervisor import environment, environment_frontend_sources
    from test_environment_frontend_sources import _artifact as frontend_artifact

    project = managed_git_source_project
    root, endpoint, commits = project.root, project.endpoint, project.commits
    _, first_manifest = _select_git_libraries(project, git_server)
    try:
        code, packed = _run(
            root,
            "source",
            "add",
            endpoint + "nodes",
            "--pack",
            "healthy-nodes",
            "--ref",
            str(commits[0]),
            "--git-mirror",
            git_server["proxy"],
            "--yes",
            timeout=180,
        )
        assert code == 0, packed
        assert _run(root, "stop", "--yes")[0] == 0
        artifact = frontend_artifact(tmp_path, str(commits[0]))
        monkeypatch.setattr(
            environment_frontend_sources, "frontend_ci_artifact", lambda _: artifact
        )

        def fetch(roots, name, destination, sha, size):
            assert sha == artifact["digest"][7:]
            destination.write_bytes(artifact["path"].read_bytes())

        monkeypatch.setattr(environment_frontend_sources, "fetch_artifact", fetch)
        frontend_args = [
            "--root",
            str(root),
            "source",
            "add",
            endpoint + "frontend",
            "--frontend",
            "--ref",
            str(commits[0]),
            "--git-mirror",
            git_server["proxy"],
            "--yes",
            "--json",
        ]
        frontend_plan = ["--plan" if value == "--yes" else value for value in frontend_args]
        assert environment.run(environment.argument_parser().parse_args(frontend_plan)) == 0
        preview = json.loads(capsys.readouterr().out)
        approval = preview["sourceApproval"]
        approved = [*frontend_args, "--source-approval", approval]
        artifact["run"] += 1
        assert environment.run(environment.argument_parser().parse_args(approved)) == 1
        refused = json.loads(capsys.readouterr().out)
        assert "source inputs changed" in refused["error"], refused
        assert _run(root, "status")[1]["current"] == preview["current"]
        artifact["run"] -= 1
        with monkeypatch.context() as missing_bundle:
            missing_bundle.setattr(
                environment_frontend_sources, "frontend_ci_artifact", lambda _: None
            )
            assert environment.run(environment.argument_parser().parse_args(approved)) == 1
            refused = json.loads(capsys.readouterr().out)
            assert "source inputs changed" in refused["error"], refused
            assert _run(root, "status")[1]["current"] == preview["current"]
        assert environment.run(environment.argument_parser().parse_args(approved)) == 0
        json.loads(capsys.readouterr().out)
        all_sources = _run(root, "status")[1]["sources"]
        assert all_sources["packs"]["healthy-nodes"]["commit"] == str(commits[0])
        assert all_sources["frontend"]["commit"] == str(commits[0])
        assert all_sources["libraries"] == first_manifest["sources"]["libraries"]
        assert _run(root, "stop", "--yes")[0] == 0
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


@pytest.fixture
def managed_git_selected_sources(
    managed_git_source_project, tmp_path, monkeypatch, git_server, capsys
):
    from dinkster_supervisor import environment, environment_frontend_sources
    from test_environment_frontend_sources import _artifact as frontend_artifact

    project = managed_git_source_project
    root, endpoint, commits = project.root, project.endpoint, project.commits
    _, first_manifest = _select_git_libraries(project, git_server)
    artifact = frontend_artifact(tmp_path, str(commits[0]))
    monkeypatch.setattr(environment_frontend_sources, "frontend_ci_artifact", lambda _: artifact)

    def fetch(roots, name, destination, sha, size):
        assert sha == artifact["digest"][7:]
        destination.write_bytes(artifact["path"].read_bytes())

    monkeypatch.setattr(environment_frontend_sources, "fetch_artifact", fetch)
    frontend_args = [
        "--root",
        str(root),
        "source",
        "add",
        endpoint + "frontend",
        "--frontend",
        "--ref",
        str(commits[0]),
        "--git-mirror",
        git_server["proxy"],
        "--yes",
        "--json",
    ]
    assert environment.run(environment.argument_parser().parse_args(frontend_args)) == 0
    capsys.readouterr()
    assert _run(root, "stop", "--yes")[0] == 0
    code, packed = _run(
        root,
        "source",
        "add",
        endpoint + "nodes",
        "--pack",
        "healthy-nodes",
        "--ref",
        str(commits[0]),
        "--git-mirror",
        git_server["proxy"],
        "--yes",
        timeout=180,
    )
    assert code == 0, packed
    all_sources = _run(root, "status")[1]["sources"]
    assert all_sources["libraries"]["fixture-helper"]
    assert all_sources["libraries"]["dinkster-inference"]
    assert all_sources["packs"]["healthy-nodes"]
    assert all_sources["frontend"]
    assert _run(root, "stop", "--yes")[0] == 0
    return SimpleNamespace(project=project, first_manifest=first_manifest, all_sources=all_sources)


def test_managed_git_engine_selection_retains_sources(
    managed_git_selected_sources, tmp_path, monkeypatch, git_server, capsys
):
    import pygit2
    from dinkster_supervisor import environment
    from dinkster_supervisor.environment_state import Generation, interpreter
    from dinkster_workers.boundary import PROTOCOL_VERSION

    selected = managed_git_selected_sources
    project = selected.project
    root, store, original = project.root, project.store, project.original
    signature, fixture = project.signature, project.fixture
    first_manifest, all_sources, before = (
        selected.first_manifest,
        selected.all_sources,
        project.before,
    )
    try:
        engine_source = tmp_path / "engine-source"
        engine_source.mkdir()
        engine_wheel = next(Path(fixture["mirror"]).glob("engine/*/*.whl"))
        with zipfile.ZipFile(engine_wheel) as archive:
            for name in archive.namelist():
                if name.startswith("dinkster/"):
                    path = engine_source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
        (engine_source / "dinkster/__init__.py").write_text('__version__ = "1.1"\nVALUE = 411\n')
        (engine_source / "pyproject.toml").write_text(
            '[project]\nname="dinkster"\nversion="1.1"\nrequires-python=">=3.12"\n'
            'dependencies=["fixture-base==1.0"]\n[dependency-groups]\n'
            'engine-base=["fixture-base==1.0"]\n[build-system]\n'
            'requires=["hatchling"]\nbuild-backend="hatchling.build"\n'
        )
        boundary = engine_source / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
        boundary.parent.mkdir(parents=True)
        boundary.write_text(f"PROTOCOL_VERSION = {PROTOCOL_VERSION}\n")
        with monkeypatch.context() as offline:
            offline.setenv("UV_NO_INDEX", "1")
            offline.setenv(
                "UV_FIND_LINKS", str(object_path(store / "bases", original.base) / "wheels")
            )
            locked = subprocess.run(
                ["uv", "lock", "--offline", "--python", sys.executable],
                cwd=engine_source,
                capture_output=True,
                text=True,
            )
            assert locked.returncode == 0, locked.stderr
        lock = engine_source / "uv.lock"
        base_wheel = next((object_path(store / "bases", original.base) / "wheels").glob("*.whl"))
        lock.write_text(
            lock.read_text().replace(
                f'path = "{base_wheel.as_posix()}"',
                f'path = "{base_wheel.as_posix()}", hash = "{content_digest(base_wheel)}"',
            )
        )
        repository = pygit2.Repository(git_server["root"] / "source.git")
        tree = pygit2.Index()
        for path in engine_source.rglob("*"):
            if path.is_file():
                tree.add(
                    pygit2.IndexEntry(
                        path.relative_to(engine_source).as_posix(),
                        repository.create_blob(path.read_bytes()),
                        pygit2.enums.FileMode.BLOB,
                    )
                )
        engine_commit = repository.create_commit(
            "refs/heads/library-engine",
            signature,
            signature,
            "engine",
            tree.write_tree(repository),
            [],
        )
        repository.free()
        code, combined = _run(
            root,
            "source",
            "add",
            "https://github.com/Kosinkadink/source",
            "--ref",
            "library-engine",
            "--git-mirror",
            git_server["proxy"],
            "--yes",
            timeout=180,
        )
        assert code == 0, combined
        combined_status = _run(root, "status")[1]
        assert combined_status["sources"]["engine"]["commit"] == str(engine_commit)
        assert combined_status["sources"]["libraries"] == first_manifest["sources"]["libraries"]
        for kind in ("packs", "frontend"):
            assert combined_status["sources"][kind] == all_sources[kind]
        combined_root = root / "builds" / str(combined["current"])
        assert (
            (combined_root / "pack-sources/healthy-nodes/healthy_nodes.py")
            .read_text()
            .endswith("VALUE = 31\n")
        )
        assert (
            subprocess.check_output(
                [
                    str(interpreter(combined_root)),
                    "-I",
                    "-B",
                    "-c",
                    "from importlib.resources import files; "
                    "print(files('dinkster_frontend').joinpath('static/index.html').read_text())",
                ],
                text=True,
            ).strip()
            == "<main>selected frontend</main>"
        )
        assert (
            subprocess.check_output(
                [
                    str(interpreter(combined_root)),
                    "-I",
                    "-B",
                    "-c",
                    "import dinkster,dinkster_inference,fixture_helper; "
                    "print(dinkster.VALUE,dinkster_inference.VALUE,fixture_helper.VALUE)",
                ],
                text=True,
            ).strip()
            == "411 31 31"
        )
        assert _run(root, "stop", "--yes")[0] == 0
        frontend_plan = ["--root", str(root), "source", "update", "--frontend", "--plan", "--json"]
        assert environment.run(environment.argument_parser().parse_args(frontend_plan)) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "planned"
        requests = len(git_server["requests"])
        update = (
            "update",
            "--catalog-lock",
            str(combined_root / "dinkster.lock"),
            "--artifacts",
            str(
                object_path(
                    store / "engines",
                    Generation.read(combined_root / "build.json").engine_release,
                )
                / "artifacts"
            ),
            "--pack-lock",
            str(combined_root / "packs/healthy-nodes/pack-lock.json"),
        )
        code, preview = _run(root, *update, "--plan")
        assert code == 0, preview
        assert preview["retainedLibraries"] == {
            **first_manifest["sources"]["libraries"],
            "dinkster-frontend": all_sources["frontend"],
        }
        code, updated = _run(root, *update, "--yes", timeout=180)
        assert code == 0, updated
        assert len(git_server["requests"]) == requests
        updated_root = root / "builds" / str(updated["current"])
        assert Generation.read(updated_root / "build.json").source == "source"
        updated_sources = _run(root, "status")[1]["sources"]
        assert updated_sources["libraries"] == first_manifest["sources"]["libraries"]
        for kind in ("packs", "frontend"):
            assert updated_sources[kind] == all_sources[kind]
        assert {path: content_digest(path) for path in before} == before
        assert (
            subprocess.check_output(
                [
                    str(interpreter(updated_root)),
                    "-I",
                    "-B",
                    "-c",
                    "import dinkster_inference,fixture_helper; "
                    "print(dinkster_inference.VALUE,fixture_helper.VALUE)",
                ],
                text=True,
            ).strip()
            == "31 31"
        )
        assert _run(root, "stop", "--yes")[0] == 0
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


def _assert_git_library_replacement(selected, tmp_path, monkeypatch, git_server, capsys, contract):
    import pygit2
    from dinkster_supervisor import environment, environment_inference
    from dinkster_supervisor.environment_state import Generation, interpreter
    from test_installer import _serve_artifacts

    project = selected.project
    root, store, original = project.root, project.store, project.original
    commits, signature, original_wheel = project.commits, project.signature, project.original_wheel
    first_manifest, all_sources, before = (
        selected.first_manifest,
        selected.all_sources,
        project.before,
    )
    updated = _run(root, "status")[1]
    updated_sources = updated["sources"]
    updated_root = root / "builds" / str(updated["current"])
    update = (
        "update",
        "--catalog-lock",
        str(updated_root / "dinkster.lock"),
        "--artifacts",
        str(
            object_path(
                store / "engines",
                Generation.read(updated_root / "build.json").engine_release,
            )
            / "artifacts"
        ),
        "--pack-lock",
        str(updated_root / "packs/healthy-nodes/pack-lock.json"),
    )
    try:
        if contract == "missing-wheel":
            wheel = first_manifest["sources"]["libraries"]["fixture-helper"]["wheel"]
            cached = object_path(store / "wheels", "sha256:" + wheel["sha256"]) / wheel["filename"]
            payload = cached.read_bytes()
            cached.unlink()
            try:
                code, refused = _run(root, *update, "--yes", timeout=180)
                assert code == 1 and "retained Git library wheel" in refused["error"], refused
                assert _run(root, "status")[1]["current"] == updated["current"]
            finally:
                cached.write_bytes(payload)
            code, released = _run(root, "source", "release", "--yes", timeout=180)
            assert code == 0, released
            assert "sources" not in _run(root, "status")[1]
            return
        published = original_wheel(tmp_path, "dinkster_inference", "9.0", 167)
        incompatible = original_wheel(tmp_path, "dinkster_inference", "9.1", 229)
        page = "".join(
            f'<a href="/wheels/{path.name}#sha256={content_digest(path)[7:]}">{path.name}</a>'
            for path in (published, incompatible)
        ).encode()
        index_requests = []
        index_server = _serve_artifacts(
            {
                "/simple/dinkster-inference": page,
                "/simple/dinkster-inference/": page,
                **{"/wheels/" + path.name: path.read_bytes() for path in (published, incompatible)},
            },
            index_requests,
        )
        try:
            repository = pygit2.Repository(git_server["root"] / "source.git")
            # Independent cases reuse the module-scoped server, not its tag selection.
            for tag in ("v9.0", "v9.1"):
                reference = f"refs/tags/{tag}"
                if reference in repository.references:
                    repository.references.delete(reference)
            repository.create_tag(
                "v9.0", commits[0], pygit2.enums.ObjectType.COMMIT, signature, "9.0"
            )
            repository.create_tag(
                "v9.1", commits[1], pygit2.enums.ObjectType.COMMIT, signature, "9.1"
            )
            repository.free()
            monkeypatch.setattr(environment_inference, "INFERENCE_REPOSITORY", git_server["url"])
            replacement_args = [
                "--root",
                str(root),
                *update,
                "--inference",
                "9.0",
                "--package-index",
                f"http://127.0.0.1:{index_server.server_port}/simple",
                "--yes",
                "--json",
            ]
            if contract == "incompatible-wheel":
                refused_args = ["9.1" if value == "9.0" else value for value in replacement_args]
                assert environment.run(environment.argument_parser().parse_args(refused_args)) == 1
                refused = json.loads(capsys.readouterr().out)
                assert refused["status"] == "error"
                assert "dinkster-inference" in refused["error"] and "9.0" in refused["error"], (
                    refused
                )
                assert _run(root, "status")[1]["current"] == updated["current"]
                return
            code = environment.run(environment.argument_parser().parse_args(replacement_args))
            replacement = json.loads(capsys.readouterr().out)
            assert code == 0, replacement
            selected_sources = _run(root, "status")[1]["sources"]
            assert selected_sources["libraries"] == {
                "fixture-helper": first_manifest["sources"]["libraries"]["fixture-helper"]
            }
            for kind in ("packs", "frontend"):
                assert selected_sources[kind] == all_sources[kind]
            replacement_root = root / "builds" / str(replacement["current"])
            assert Generation.read(replacement_root / "build.json").base == original.base
            assert (
                subprocess.check_output(
                    [
                        str(interpreter(replacement_root)),
                        "-I",
                        "-B",
                        "-c",
                        "import dinkster_inference,fixture_helper; "
                        "print(dinkster_inference.VALUE,fixture_helper.VALUE)",
                    ],
                    text=True,
                ).strip()
                == "167 31"
            )
            assert index_requests
            assert {path: content_digest(path) for path in before} == before
            assert _run(root, "stop", "--yes")[0] == 0
            code, restored = _run(root, "rollback", "--yes", timeout=180)
            assert code == 0, restored
            assert _run(root, "status")[1]["sources"] == updated_sources
            restored_root = root / "builds" / str(restored["current"])
            assert (
                subprocess.check_output(
                    [
                        str(interpreter(restored_root)),
                        "-I",
                        "-B",
                        "-c",
                        "import dinkster_inference,fixture_helper; "
                        "print(dinkster_inference.VALUE,fixture_helper.VALUE)",
                    ],
                    text=True,
                ).strip()
                == "31 31"
            )
            assert _run(root, "stop", "--yes")[0] == 0
        finally:
            index_server.shutdown()
            index_server.server_close()
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


def test_managed_git_frontend_uses_verified_ci_assets_and_retains_exact_update(
    tmp_path, monkeypatch, git_server, capsys
):
    import pygit2
    from dinkster_supervisor import (
        environment,
        environment_artifacts,
        environment_downloads,
        environment_frontend_sources,
    )
    from dinkster_supervisor.environment_state import Generation, interpreter
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from test_environment_frontend_sources import _artifact

    from tests.fixtures.environments import build as publisher

    original_wheel, encode = publisher.wheel, publisher.canonical_json
    monkeypatch.setattr(
        publisher, "wheel", lambda *args, **kwargs: original_wheel(*args, **kwargs, pack_tools=True)
    )
    monkeypatch.setattr(publisher, "standalone_tree", _workspace_tree)

    def encoded(document):
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        if document.get("format") == "dinkster.engine/1":
            document["schemaApi"] = 1
        return encode(document)

    monkeypatch.setattr(publisher, "canonical_json", encoded)
    for name in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path / "user"))
    fixture = publisher.build(tmp_path / "fixture", "linux-cpu", free_port())
    root = Path(fixture["root"])
    catalog, artifacts = tmp_path / "empty.lock", tmp_path / "artifacts"
    catalog.write_text(Lockfile.of([]).record_json())
    artifacts.mkdir()
    code, result = _run(
        root,
        "setup",
        "--mirror",
        fixture["mirror"],
        "--data",
        fixture["store"],
        "--variant",
        "linux-cpu",
        "--channel",
        "nightly",
        "--port",
        str(fixture["port"]),
        "--catalog-lock",
        str(catalog),
        "--artifacts",
        str(artifacts),
        "--yes",
    )
    assert code == 0, result
    assert _run(root, "stop", "--yes")[0] == 0
    repository = pygit2.Repository(git_server["root"] / "source.git")
    signature = pygit2.Signature("Fixture", "fixture@example.invalid", 1700000000, 0)
    tree = pygit2.Index()
    for name, payload in {
        "package.json": b'{"version":"0.2.0"}',
        "frontend-release.json": b'{"engineApi":{"min":1,"max":2}}',
    }.items():
        tree.add(
            pygit2.IndexEntry(name, repository.create_blob(payload), pygit2.enums.FileMode.BLOB)
        )
    commit = repository.create_commit(
        "refs/heads/frontend-ci", signature, signature, "frontend", tree.write_tree(repository), []
    )
    tree.add(
        pygit2.IndexEntry(
            "selected-ref.txt", repository.create_blob(b"second ref"), pygit2.enums.FileMode.BLOB
        )
    )
    newer = repository.create_commit(
        "refs/heads/frontend-next",
        signature,
        signature,
        "next",
        tree.write_tree(repository),
        [commit],
    )
    library = pygit2.Index()
    for name, payload in {
        "pyproject.toml": (
            '[project]\nname="fixture-helper"\nversion="9.0"\n'
            'dependencies=["fixture-base==1.0"]\n'
            '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
        ),
        "fixture_helper/__init__.py": "VALUE = 83\n",
    }.items():
        library.add(
            pygit2.IndexEntry(
                name, repository.create_blob(payload.encode()), pygit2.enums.FileMode.BLOB
            )
        )
    helper = repository.create_commit(
        "refs/heads/frontend-helper",
        signature,
        signature,
        "helper",
        library.write_tree(repository),
        [],
    )
    repository.free()
    artifact = _artifact(tmp_path, str(commit))
    monkeypatch.setattr(
        environment_frontend_sources, "frontend_ci_artifact", lambda origin: artifact
    )

    def fetch(roots, name, destination, sha, size):
        assert sha == artifact["digest"][7:]
        destination.write_bytes(artifact["path"].read_bytes())

    monkeypatch.setattr(environment_frontend_sources, "fetch_artifact", fetch)
    args = [
        "--root",
        str(root),
        "source",
        "add",
        "https://github.com/Kosinkadink/source",
        "--frontend",
        "--ref",
        "frontend-ci",
        "--git-mirror",
        git_server["proxy"],
        "--json",
    ]
    try:
        code, combined = _run(
            root,
            "source",
            "add",
            "https://github.com/Kosinkadink/source",
            "--library",
            "fixture-helper",
            "--ref",
            "frontend-helper",
            "--git-mirror",
            git_server["proxy"],
            "--yes",
            timeout=180,
        )
        assert code == 0, combined
        assert _run(root, "stop", "--yes")[0] == 0
        helper_origin = _run(root, "status")[1]["sources"]["libraries"]["fixture-helper"]
        assert helper_origin["commit"] == str(helper)
        base = Generation.read(root / "builds/1/build.json").base
        base_files = {
            path: content_digest(path)
            for path in object_path(Path(fixture["store"]) / "bases", base).rglob("*")
            if path.is_file()
        }
        tools = tmp_path / "frontend-no-git-tools"
        tools.mkdir()
        uv = shutil.which("uv")
        assert uv is not None
        shutil.copyfile(uv, tools / Path(uv).name)
        (tools / Path(uv).name).chmod(0o755)
        monkeypatch.setenv("PATH", str(tools))
        assert shutil.which("git") is None
        requests = len(git_server["requests"])
        tunnels = len(git_server["tunnels"])
        generations = sorted(path.name for path in (root / "builds").iterdir())
        assert environment.run(environment.argument_parser().parse_args([*args, "--plan"])) == 0
        preview = json.loads(capsys.readouterr().out)
        assert any("CI bundle" in line for line in preview["plan"])
        assert sorted(path.name for path in (root / "builds").iterdir()) == generations
        assert environment.run(environment.argument_parser().parse_args([*args, "--yes"])) == 0
        applied = json.loads(capsys.readouterr().out)
        selected = _run(root, "status")[1]["sources"]["frontend"]
        assert selected["commit"] == str(commit) and selected["build"]["kind"] == "ci"
        current = root / "builds" / str(applied["current"])
        assert Generation.read(current / "build.json").source == "source"

        def html(generation):
            return subprocess.check_output(
                [
                    str(interpreter(generation)),
                    "-I",
                    "-B",
                    "-c",
                    """
import asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster.frontend import discover_frontend_bundle, install_frontend
async def serve():
    app = web.Application()
    install_frontend(app, bundle=discover_frontend_bundle())
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/")
        assert response.status == 200
        assert response.content_type == "text/html"
        print(await response.text())
asyncio.run(serve())
""",
                ],
                text=True,
            ).strip()

        assert html(current) == "<main>selected frontend</main>"
        assert _run(root, "stop", "--yes")[0] == 0
        code, updated = _run(
            root,
            "update",
            "--catalog-lock",
            str(catalog),
            "--artifacts",
            str(artifacts),
            "--yes",
            timeout=180,
        )
        assert code == 0, updated
        assert _run(root, "status")[1]["sources"]["frontend"] == selected
        assert html(root / "builds" / str(updated["current"])) == "<main>selected frontend</main>"
        assert _run(root, "stop", "--yes")[0] == 0
        code, restored = _run(root, "rollback", "--yes", timeout=180)
        assert code == 0, restored
        assert html(root / "builds" / str(restored["current"])) == "<main>selected frontend</main>"
        assert _run(root, "stop", "--yes")[0] == 0
        original_artifact = artifact
        next_assets = tmp_path / "next-assets"
        next_assets.mkdir()
        artifact = _artifact(next_assets, str(newer), extra="assets/next-ref.js")
        update_args = [
            "--root",
            str(root),
            "source",
            "update",
            "--frontend",
            "--ref",
            "frontend-next",
            "--yes",
            "--json",
        ]
        assert environment.run(environment.argument_parser().parse_args(update_args)) == 0
        changed = json.loads(capsys.readouterr().out)
        status = _run(root, "status")[1]
        assert status["sources"]["libraries"]["fixture-helper"] == helper_origin
        newer_origin = status["sources"]["frontend"]
        assert newer_origin["commit"] == str(newer)
        assert newer_origin["requestedRef"] == "frontend-next"
        assert newer_origin["wheel"]["sha256"] != selected["wheel"]["sha256"]
        changed_root = root / "builds" / str(changed["current"])
        assert (
            subprocess.check_output(
                [
                    str(interpreter(changed_root)),
                    "-I",
                    "-B",
                    "-c",
                    "from importlib.resources import files; import fixture_helper; "
                    "print(fixture_helper.VALUE, "
                    "files('dinkster_frontend').joinpath('static/assets/next-ref.js').read_text())",
                ],
                text=True,
            ).strip()
            == "83 marker=73"
        )
        assert _run(root, "stop", "--yes")[0] == 0
        before = _run(root, "status")[1]["current"]
        generations = sorted(path.name for path in (root / "builds").iterdir())
        assert environment.run(environment.argument_parser().parse_args([*args, "--yes"])) == 1
        refused = json.loads(capsys.readouterr().out)
        assert "identity differs" in refused["error"]
        assert _run(root, "status")[1]["current"] == before
        assert sorted(path.name for path in (root / "builds").iterdir()) == generations
        artifact = original_artifact
        repository = pygit2.Repository(git_server["root"] / "source.git")
        repository.references["refs/heads/frontend-ci"].set_target(newer)
        repository.free()
        code, restored = _run(root, "rollback", "--yes", timeout=180)
        assert code == 0, restored
        status = _run(root, "status")[1]
        assert status["sources"]["frontend"] == selected
        assert status["sources"]["libraries"]["fixture-helper"] == helper_origin
        restored_root = root / "builds" / str(restored["current"])
        assert (
            subprocess.check_output(
                [
                    str(interpreter(restored_root)),
                    "-I",
                    "-B",
                    "-c",
                    "from importlib.resources import files; "
                    "print(files('dinkster_frontend').joinpath('static/assets/next-ref.js').is_file())",
                ],
                text=True,
            ).strip()
            == "False"
        )
        assert _run(root, "stop", "--yes")[0] == 0
        history = _run(root, "history")[1]["history"]
        exported = history[-1]["snapshot"]
        manifest = json.loads(exported["metadata"]["engine-manifest.json"])
        assert exported["source"] == "source"
        assert manifest["sources"]["frontend"] == selected
        assert manifest["sources"]["libraries"]["fixture-helper"] == helper_origin
        wheel = selected["wheel"]
        store = Path(fixture["store"])
        cached = object_path(store / "wheels", "sha256:" + wheel["sha256"]) / wheel["filename"]
        payload = cached.read_bytes()
        cached.unlink()
        try:
            code, refused = _run(
                root,
                "update",
                "--catalog-lock",
                str(catalog),
                "--artifacts",
                str(artifacts),
                "--yes",
                timeout=180,
            )
            assert code == 1 and "retained Git library wheel" in refused["error"], refused
            assert _run(root, "status")[1]["current"] == restored["current"]
            assert _run(root, "status")[1]["sources"] == manifest["sources"]
        finally:
            cached.write_bytes(payload)
        published = original_wheel(tmp_path, "dinkster_frontend", "0.3.0", 0)
        with zipfile.ZipFile(published, "a") as archive:
            archive.writestr("dinkster_frontend/__init__.py", "")
            archive.writestr(
                "dinkster_frontend/manifest.json",
                json.dumps(
                    {
                        "format": "dinkster.frontend/1",
                        "repository": "Kosinkadink/Dinkster-Frontend",
                        "version": "0.3.0",
                        "engineApi": {"min": 1, "max": 2},
                    }
                ),
            )
            archive.writestr("dinkster_frontend/static/index.html", "published replacement")
        published_row = {
            "filename": published.name,
            "url": "https://files.pythonhosted.org/packages/" + published.name,
            "digests": {"sha256": content_digest(published)[7:]},
            "size": published.stat().st_size,
        }
        read = environment_downloads.read_metadata
        download = environment_downloads.fetch_artifact

        def metadata(location, path):
            if location == "https://pypi.org/pypi/dinkster-frontend":
                assert path == "0.3.0/json"
                return json.dumps({"urls": [published_row]}).encode()
            return read(location, path)

        def transfer(locations, name, destination, sha, size):
            if locations == ("https://files.pythonhosted.org/packages",):
                locations = (str(tmp_path),)
            return download(locations, name, destination, sha, size)

        monkeypatch.setattr(environment_downloads, "read_metadata", metadata)
        monkeypatch.setattr(environment_downloads, "fetch_artifact", transfer)
        monkeypatch.setattr(environment_artifacts, "fetch_artifact", transfer)
        published_args = [
            "--root",
            str(root),
            "update",
            "--frontend-version",
            "0.3.0",
            "--catalog-lock",
            str(catalog),
            "--artifacts",
            str(artifacts),
            "--yes",
            "--json",
        ]
        assert environment.run(environment.argument_parser().parse_args(published_args)) == 0
        replacement = json.loads(capsys.readouterr().out)
        status = _run(root, "status")[1]
        assert "frontend" not in status["sources"]
        assert status["sources"]["libraries"]["fixture-helper"] == helper_origin
        replacement_root = root / "builds" / str(replacement["current"])
        assert html(replacement_root) == "published replacement"
        assert (
            Generation.read(replacement_root / "build.json").base
            == Generation.read(restored_root / "build.json").base
        )
        assert _run(root, "stop", "--yes")[0] == 0
        code, recovered = _run(root, "rollback", "--yes", timeout=180)
        assert code == 0, recovered
        assert _run(root, "status")[1]["sources"]["frontend"] == selected
        assert html(root / "builds" / str(recovered["current"])) == "<main>selected frontend</main>"
        assert {path: content_digest(path) for path in base_files} == base_files
        assert len(git_server["requests"]) > requests
        assert all(
            path.startswith("/Kosinkadink/source.git/")
            for path in git_server["requests"][requests:]
        ), git_server["requests"][requests:]
        assert len(git_server["tunnels"]) == tunnels
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


def test_source_engine_ref_update_and_exact_commit_rollback(
    tmp_path: Path, monkeypatch, git_server
):
    from zipfile import ZipFile

    import pygit2
    from dinkster_supervisor.environment_state import Generation, interpreter, read_document
    from test_installer import _serve_artifacts

    from tests.fixtures.environments.build import build
    from tests.fixtures.environments.build import wheel as build_wheel

    for name in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path / "user"))
    fixture = build(tmp_path / "fixture", "linux-cpu", free_port())
    root = Path(fixture["root"])
    code, result = _run(
        root,
        "setup",
        "--mirror",
        fixture["mirror"],
        "--data",
        fixture["store"],
        "--variant",
        "linux-cpu",
        "--channel",
        "nightly",
        "--port",
        str(fixture["port"]),
        "--yes",
    )
    assert code == 0, result
    added = build_wheel(tmp_path, "fixture_new", "1.0", 853)
    requests = []
    index_server = _serve_artifacts(
        {
            "/simple/fixture-new/": (
                f'<a href="/wheels/{added.name}#sha256={content_digest(added)[7:]}">'
                f"{added.name}</a>"
            ).encode(),
            "/wheels/" + added.name: added.read_bytes(),
        },
        requests,
    )
    package_index = f"http://127.0.0.1:{index_server.server_port}/simple"
    try:
        assert _run(root, "stop", "--yes")[0] == 0
        source = tmp_path / "source"
        source.mkdir()
        engine_wheel = next(Path(fixture["mirror"]).glob("engine/*/*.whl"))
        with ZipFile(engine_wheel) as archive:
            for name in archive.namelist():
                if name.startswith("dinkster/"):
                    path = source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
        manager = source / "dinkster/manager.py"
        manager.write_text(
            manager.read_text().replace('version("fixture-engine")', 'version("dinkster")')
        )
        (source / "dinkster/__init__.py").write_text("VALUE = 911\n")
        (source / "pyproject.toml").write_text(
            '[project]\nname="dinkster"\nversion="1.1"\nrequires-python=">=3.12"\n'
            'dependencies=["fixture-base==1.0","fixture-new==1.0"]\n[dependency-groups]\n'
            'engine-base=["fixture-base==1.0"]\n[build-system]\n'
            'requires=["hatchling"]\nbuild-backend="hatchling.build"\n'
        )
        boundary = source / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
        boundary.parent.mkdir(parents=True)
        boundary.write_text("PROTOCOL_VERSION = 11\n")
        index = tmp_path / "index"
        index.mkdir()
        base = read_document(Path(fixture["baseManifest"]))
        assert isinstance(base["id"], str)
        base_wheel = next(
            (object_path(Path(fixture["store"]) / "bases", base["id"]) / "wheels").glob("*.whl")
        )
        wheel = index / base_wheel.name
        shutil.copyfile(base_wheel, wheel)
        shutil.copyfile(added, index / added.name)
        monkeypatch.setenv("UV_NO_INDEX", "1")
        monkeypatch.setenv("UV_FIND_LINKS", str(index))
        monkeypatch.setenv("UV_OFFLINE", "1")
        locked = subprocess.run(
            ["uv", "lock", "--offline", "--python", sys.executable],
            cwd=source,
            capture_output=True,
            text=True,
        )
        assert locked.returncode == 0, locked.stderr
        lock = source / "uv.lock"
        text = lock.read_text()
        for path in (wheel, index / added.name):
            text = text.replace(
                f'path = "{path.as_posix()}"',
                f'path = "{path.as_posix()}", hash = "{content_digest(path)}"',
            )
        lock.write_text(text)
        (index / added.name).unlink()
        wheel.unlink()
        monkeypatch.delenv("UV_NO_INDEX")
        monkeypatch.delenv("UV_OFFLINE")
        monkeypatch.delenv("UV_FIND_LINKS")
        before = {
            path: content_digest(path)
            for parent in ("bases", "wheels")
            for path in (Path(fixture["store"]) / parent).rglob("*")
            if path.is_file() and path.name != ".objects.json"
        }
        requests.clear()
        repository = pygit2.Repository(git_server["root"] / "source.git")
        git_index = pygit2.Index()
        for path in source.rglob("*"):
            if path.is_file():
                git_index.add(
                    pygit2.IndexEntry(
                        path.relative_to(source).as_posix(),
                        repository.create_blob(path.read_bytes()),
                        pygit2.enums.FileMode.BLOB,
                    )
                )
        signature = pygit2.Signature("Fixture", "fixture@example.invalid", 1700000000, 0)
        commit = repository.create_commit(
            "refs/heads/engine",
            signature,
            signature,
            "engine",
            git_index.write_tree(repository),
            [],
        )
        git_index.add(
            pygit2.IndexEntry(
                "dinkster/__init__.py",
                repository.create_blob(b"VALUE = 1229\n"),
                pygit2.enums.FileMode.BLOB,
            )
        )
        changed = repository.create_commit(
            "refs/heads/engine-changed",
            signature,
            signature,
            "changed",
            git_index.write_tree(repository),
            [commit],
        )
        repository.free()
        tools = tmp_path / "no-git-tools"
        tools.mkdir()
        uv = shutil.which("uv")
        assert uv is not None
        shutil.copyfile(uv, tools / Path(uv).name)
        (tools / Path(uv).name).chmod(0o755)
        monkeypatch.setenv("PATH", str(tools))
        assert shutil.which("git") is None
        selection = (
            "source",
            "add",
            "https://github.com/Kosinkadink/source",
            "--git-mirror",
            git_server["proxy"],
            "--ref",
            "engine",
            "--package-index",
            package_index,
        )
        code, preview = _run(root, *selection, "--plan")
        assert code == 0, preview
        approval = preview["sourceApproval"]
        repository = pygit2.Repository(git_server["root"] / "source.git")
        repository.references["refs/heads/engine"].set_target(changed)
        repository.free()
        code, refused = _run(root, *selection, "--source-approval", approval, "--yes")
        assert code == 1 and "source inputs changed" in refused["error"], refused
        assert _run(root, "status")[1]["current"] == preview["current"]
        assert not requests
        repository = pygit2.Repository(git_server["root"] / "source.git")
        repository.references["refs/heads/engine"].set_target(commit)
        repository.free()
        code, result = _run(root, *selection, "--source-approval", approval, "--yes")
        assert code == 0, json.dumps(result)
        assert requests and {path for path, _ in requests} == {
            "/simple/fixture-new/",
            "/wheels/" + added.name,
        }
        assert {path: content_digest(path) for path in before} == before
        requests.clear()
        status = _run(root, "status")[1]
        assert status["source"] == "source"
        assert status["published"] is False and status["reproducible"] is False
        assert status["sources"]["engine"]["commit"] == str(commit)
        assert status["sources"]["engine"]["requestedRef"] == "engine"
        assert _run(root, "stop", "--yes")[0] == 0
        code, result = _run(root, "source", "update", "--ref", "engine-changed", "--yes")
        assert code == 0, result
        status = _run(root, "status")[1]
        assert status["sources"]["engine"]["commit"] == str(changed)
        assert status["sources"]["engine"]["requestedRef"] == "engine-changed"
        assert _run(root, "stop", "--yes")[0] == 0
        original_release = Generation.read(root / "builds/2/build.json").engine_release
        repository = pygit2.Repository(git_server["root"] / "source.git")
        repository.references["refs/heads/engine"].set_target(changed)
        repository.free()
        shutil.rmtree(
            object_path(Path(fixture["store"]) / "engines", original_release) / "checkout"
        )
        code, result = _run(root, "rollback", "--yes")
        assert code == 0, result
        assert _run(root, "status")[1]["sources"]["engine"]["commit"] == str(commit)
        current = result["current"]
        assert (
            subprocess.check_output(
                [
                    str(interpreter(root / "builds" / str(current))),
                    "-I",
                    "-B",
                    "-c",
                    "import dinkster,fixture_new; print(dinkster.VALUE,fixture_new.VALUE)",
                ],
                text=True,
            ).strip()
            == "911 853"
        )
        assert requests == []
        assert {path: content_digest(path) for path in before} == before
        assert _run(root, "stop", "--yes")[0] == 0
        code, result = _run(root, "source", "release", "--yes")
        assert code == 0, result
        assert (
            _run(root, "status")[1]["engine"]["release"]
            == read_channel(Path(fixture["initialChannel"]), "nightly")["engine"]["release"]
        )
        assert _run(root, "stop", "--yes")[0] == 0
        code, result = _run(root, "rollback", "--yes")
        assert code == 0, result
        assert _run(root, "status")[1]["source"] == "source"
    finally:
        _run(root, "stop", "--yes")
        index_server.shutdown()
        index_server.server_close()


def test_catalog_uses_common_envelope_and_leaves_prospective_root_absent(tmp_path: Path):
    mirror = tmp_path / "mirror"
    base, _, channel = _mirror(mirror)
    root = tmp_path / "prospective project"
    code, result = _run(root, "catalog", "--channel", "nightly", "--mirror", str(mirror))
    assert code == 0
    assert result["operation"] == "catalog"
    assert result["plan"] == []
    assert result["status"] == "clean"
    assert result["channel"] == "nightly"
    assert result["channelManifest"] == dict(channel, channel="nightly")
    assert result["current"] is None
    assert result["variants"][0]["manifest"] == base
    assert not root.exists()


def test_setup_plan_labels_cached_channel_without_mutating_project_or_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor.environment_artifacts import acquire

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, store, root = tmp_path / "mirror", tmp_path / "store", tmp_path / "project"
    base, engine, _ = _mirror(mirror)
    acquire(discover((str(mirror),), "nightly"), "fixture-cpu", store)
    mirror.rename(tmp_path / "disconnected")
    before = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    code, result = _run(
        root,
        "setup",
        "--channel",
        "nightly",
        "--mirror",
        str(mirror),
        "--variant",
        "fixture-cpu",
        "--data",
        str(store),
        "--plan",
    )
    assert code == 0, result
    assert result["status"] == "planned"
    assert any("last acquired channel selection" in line for line in result["plan"])
    assert any(base["id"] in line and engine["version"] in line for line in result["plan"])
    assert result["runtimeDownloadBytes"] == 0
    assert all(row["cached"] for row in result["runtimeInputs"])
    assert {Path(row["path"]).name for row in result["runtimeInputs"]} == {
        "fixture_base-1.0-py3-none-any.whl",
        "fixture_engine-2.0-py3-none-any.whl",
        "constraints.txt",
    }
    assert not root.exists()
    assert {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize("selection", ["matching", "engine", "base", "unpaired", "local"])
def test_setup_requires_reviewed_release_before_creating_project_or_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
):
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    base, _, channel = _mirror(mirror)
    release = channel["engine"]["release"]
    arguments = [
        "setup",
        "--channel",
        "nightly",
        "--mirror",
        str(mirror),
        "--variant",
        "fixture-cpu",
        "--data",
        str(store),
        "--engine-release",
        "sha256:" + "a" * 64 if selection == "engine" else release,
    ]
    if selection != "unpaired":
        arguments.extend(("--base-id", "sha256:" + "b" * 64 if selection == "base" else base["id"]))
    if selection == "local":
        arguments.append("--from-scratch")
    arguments.append("--plan" if selection == "matching" else "--yes")
    code, result = _run(root, *arguments)
    if selection == "matching":
        assert code == 0, result
        assert result["status"] == "planned"
        assert any(base["id"] in line for line in result["plan"])
    else:
        assert code != 0, result
        expected = {
            "engine": "does not match the approved engine release/base",
            "base": "does not match the approved engine release/base",
            "unpaired": "requires both --engine-release and --base-id",
            "local": "cannot use --from-scratch",
        }
        assert expected[selection] in result["error"]
    assert not root.exists()
    assert not store.exists()
    assert not (tmp_path / "user-data").exists()


@pytest.mark.parametrize("changed", ["catalog", "dependency"])
def test_setup_refuses_inputs_changed_during_confirmation_before_creating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    changed: str,
):
    from dinkster_supervisor import environment

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    _mirror(mirror)
    catalog, dependency = tmp_path / "catalog.lock", tmp_path / "pack-lock.json"
    catalog.write_text(Lockfile.of(()).record_json())
    dependency.write_text(
        json.dumps(
            {
                "format": "dinkster.pack-lock/1",
                "pack": "demo",
                "variant": "fixture-cpu",
                "requiresPython": ">=3.12",
                "requirements": [],
                "indexes": [],
                "packages": [],
            }
        )
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    target = catalog if changed == "catalog" else dependency
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: True)
    monkeypatch.setattr(sys, "stdin", stream)

    def confirm(prompt: str) -> str:
        assert prompt == "Apply this plan? [y/N] "
        target.write_bytes(target.read_bytes() + b" ")
        return "yes"

    monkeypatch.setattr("builtins.input", confirm)
    with pytest.raises(SystemExit) as exited:
        environment.main(
            [
                "--root",
                str(root),
                "setup",
                "--channel",
                "nightly",
                "--mirror",
                str(mirror),
                "--variant",
                "fixture-cpu",
                "--data",
                str(store),
                "--catalog-lock",
                str(catalog),
                "--artifacts",
                str(artifacts),
                "--pack-lock",
                str(dependency),
            ]
        )
    captured = capsys.readouterr()
    assert exited.value.code == 1
    subject = "target catalog" if changed == "catalog" else "dependency inputs"
    assert f"{subject} changed after planning; request a new plan" in captured.err
    assert not root.exists()
    assert not store.exists()
    assert not (tmp_path / "user-data").exists()


@pytest.mark.parametrize(
    ("host", "architecture", "override", "expected"),
    [
        ("darwin", "arm64", None, "mac-arm64"),
        ("darwin", "arm64", "custom-base", "custom-base"),
        ("darwin", "x86_64", None, None),
        ("linux", "x86_64", None, None),
        ("win32", "AMD64", None, None),
    ],
)
def test_setup_defaults_only_apple_silicon_without_mutating_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    host: str,
    architecture: str,
    override: str | None,
    expected: str | None,
):
    from dinkster_supervisor import environment

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    _mirror(mirror, variant=override or "mac-arm64", target=("macos", "arm64"))
    arguments = [
        "dinkster-env",
        "--root",
        str(root),
        "setup",
        "--channel",
        "nightly",
        "--mirror",
        str(mirror),
        "--data",
        str(store),
        "--plan",
        "--json",
    ]
    if override:
        arguments.extend(("--variant", override))
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(environment, "sys", SimpleNamespace(platform=host))
    monkeypatch.setattr(environment, "machine", lambda: architecture, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        environment.main()
    result = json.loads(capsys.readouterr().out)
    if expected:
        assert exit_info.value.code == 0, result
        assert result["status"] == "planned"
        assert any(f"with {expected} base" in line for line in result["plan"])
    else:
        assert exit_info.value.code == 1
        assert result["error"] == "setup requires an explicit --variant"
    assert not root.exists()
    assert not store.exists()


def test_setup_names_published_variants_without_resolving_or_creating_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    _mirror(mirror, variant="linux-cpu")
    code, result = _run(
        root,
        "setup",
        "--variant",
        "absent-platform",
        "--channel",
        "nightly",
        "--mirror",
        str(mirror),
        "--data",
        str(store),
        "--yes",
    )
    assert code != 0
    assert "does not provide variant absent-platform" in result["error"]
    assert "supported published variants: linux-cpu" in result["error"]
    assert "--from-scratch" in result["error"]
    assert not root.exists()
    assert not store.exists()


@pytest.mark.parametrize("explicit_sources", [False, True])
def test_custom_cli_setup_retains_base_on_update_without_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    explicit_sources: bool,
):
    from dinkster_supervisor import environment
    from dinkster_supervisor.environment_artifacts import _BASE_IDENTITY

    from tests.fixtures.environments.build import artifact, build, wheel

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    base = json.loads(Path(fixture["baseManifest"]).read_bytes())
    custom = {**base, "variant": "studio-cpu", "source": "user-defined"}
    donor = Path(fixture["baseManifest"]).parent
    tree = tmp_path / "custom-payload"
    with (donor / base["archive"]["parts"][0]["filename"]).open("rb") as stream:
        with zstandard.ZstdDecompressor().stream_reader(stream) as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as archive:
                archive.extractall(tree, filter="data")
    torch = wheel(tree / "wheels", "torch", "2.14.1", 17)
    custom["wheels"] = [
        *base["wheels"],
        {
            **artifact(torch.name, torch.read_bytes(), key="filename"),
            "name": "torch",
            "version": "2.14.1",
            "lockedVersion": None,
            "index": "https://pypi.org/simple",
            "url": "https://example.invalid/torch.whl",
            "verification": "local",
        },
    ]
    custom["replaces"] = [*base["replaces"], "torch"]
    identity = {key: custom[key] for key in _BASE_IDENTITY}
    identity["archive"] = {"format": "tar+zstd"}
    custom["id"] = "sha256:" + hashlib.sha256(_json(identity)).hexdigest()
    custom["localBuild"] = {"extrasIndex": "https://pypi.org/simple"}
    retained = object_path(
        store / "downloads",
        custom["id"],
        label="studio-cpu/2026-11-03-py3.14.10-torch2.14.1" if not explicit_sources else None,
    )
    retained.mkdir(parents=True)
    part = retained / "base.tar.zst.001"
    with part.open("wb") as stream, zstandard.ZstdCompressor().stream_writer(stream) as encoded:
        with tarfile.open(fileobj=encoded, mode="w|") as archive:
            archive.add(tree / "python", "python")
            archive.add(tree / "wheels", "wheels")
    custom["archive"] = {
        "format": "tar+zstd",
        "parts": [artifact(part.name, part.read_bytes(), key="filename")],
    }
    definitions = tmp_path / "custom.toml"
    definitions.write_text(
        '[variants.studio-cpu]\ntorch="2.14.1"\n'
        'torch-index="https://download.pytorch.org/whl/cpu"\n'
    )

    def built(*_):
        from dinkster_supervisor.environment_artifacts import acquire_base

        destination = acquire_base(custom, (str(retained),), store)
        (destination / "manifest.json").write_bytes(_json(custom))
        return custom

    monkeypatch.setattr(environment, "build_custom_base", built)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-env",
            "--root",
            str(root),
            "setup",
            "--variant",
            "studio-cpu",
            "--variants",
            str(definitions),
            "--data",
            str(store),
            "--channel",
            "nightly",
            "--mirror",
            str(mirror),
            "--port",
            str(fixture["port"]),
            "--yes",
            "--json",
        ],
    )
    try:
        with pytest.raises(SystemExit) as setup:
            environment.main()
        result = json.loads(capsys.readouterr().out)
        assert setup.value.code == 0, result
        assert result["current"] == 1
        sources = json.loads((root / "builds/1/artifact-sources.json").read_bytes())
        assert sources["base"] == [str(retained)]
        assert _run(root, "stop", "--yes")[0] == 0
        definitions.unlink()
        (mirror / "channels/nightly.json").write_bytes(Path(fixture["updatedChannel"]).read_bytes())
        overrides = ()
        if explicit_sources:
            catalog, archives = tmp_path / "selected.lock", tmp_path / "selected-artifacts"
            catalog.write_text(Lockfile().record_json())
            archives.mkdir()
            overrides = ("--catalog-lock", str(catalog), "--artifacts", str(archives))
        code, updated = _run(root, "update", *overrides, "--yes")
        assert code == 0 and updated["current"] == 2, updated
        assert _run(root, "stop", "--yes")[0] == 0
        first, second = (root / "builds" / str(number) for number in (1, 2))
        assert (second / "base-manifest.json").read_bytes() == (
            first / "base-manifest.json"
        ).read_bytes()
        generation = json.loads((second / "build.json").read_bytes())
        assert generation["base"] == custom["id"]
        assert generation["variant"] == "studio-cpu"
        assert generation["engine"]["version"] == "2.0"
        assert not list(store.glob(".registry-*"))
        assert _run(root, "doctor")[1]["status"] == "clean"
        manifest = second / "base-manifest.json"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b" ")
        code, refused = _run(root, "update", "--yes")
        assert code == 1 and "differs from active generation" in refused["error"]
        assert (root / "current").read_text() == "2\n"
        assert not (root / "builds/3").exists()
        manifest.write_bytes(original)
        code, published = _run(root, "update", "--variant", "fixture-cpu", "--yes")
        assert code == 0 and published["current"] == 3, published
        assert _run(root, "stop", "--yes")[0] == 0
        code, rolled_back = _run(root, "rollback", "--yes")
        assert code == 0 and rolled_back["current"] == 4, rolled_back
        assert json.loads((root / "environment.json").read_bytes())["variant"] == "fixture-cpu"
        assert json.loads((root / "builds/4/build.json").read_bytes())["variant"] == ("studio-cpu")
        assert _run(root, "doctor")[1]["status"] == "clean"
    finally:
        _run(root, "stop", "--yes")


@pytest.mark.parametrize("operation", ["setup", "update"])
def test_incompatible_published_dependency_refuses_before_project_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, store, root = tmp_path / "mirror", tmp_path / "store", tmp_path / "project"
    base, engine, channel = _mirror(mirror)
    previous = mirror / "engine" / channel["engine"]["release"][7:]
    filename = "fixture_engine-2.0-py3-none-any.whl"
    payload = _wheel_payload("fixture_engine", "2.0", "fixture-base>=2")
    engine["artifacts"][1] = _artifact(filename, payload)
    fields = ("name", "version", "filename", "size", "sha256", "index", "url")
    engine["dependencies"] = {
        "fixture-cpu": {
            "base": base["id"],
            "wheels": [{key: base["wheels"][0][key] for key in fields}],
        }
    }
    data = _json(engine)
    identity = hashlib.sha256(data).hexdigest()
    release = previous.parent / identity
    release.mkdir()
    (release / "release-manifest.json").write_bytes(data)
    (release / "constraints.txt").write_bytes((previous / "constraints.txt").read_bytes())
    (release / filename).write_bytes(payload)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"] = [
        {"type": "mirror", "path": f"engine/{identity}/release-manifest.json"}
    ]
    (mirror / "channels/nightly.json").write_bytes(_channel_json(channel))
    if operation == "update":
        _generation(root, 1)
        (root / "current").write_bytes(b"1\n")
        (root / "environment.json").write_bytes(
            _json(
                {
                    "format": "dinkster.project/1",
                    "channel": "nightly",
                    "variant": "fixture-cpu",
                    "mirrors": [str(mirror)],
                    "port": 3639,
                }
            )
        )
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    code, result = _run(
        root,
        operation,
        "--mirror",
        str(mirror),
        "--channel",
        "nightly",
        "--variant",
        "fixture-cpu",
        "--data",
        str(store),
        "--yes",
    )
    assert code == 1 and result["status"] == "error"
    assert (
        "fixture-engine requires fixture-base>=2; base provides fixture-base 1.0" in result["error"]
    )
    assert {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".lock"
    } == before
    assert not (root / "store").exists()
    assert not list((root / "journal").glob("*.json"))
    assert not (store / "projects.json").exists()
    assert not (tmp_path / "user-data/dinkster/stores.json").exists()


@pytest.mark.parametrize("linked_parent", [False, True])
def test_status_returns_configured_channel_and_variant_without_starting_anything(
    tmp_path: Path, linked_parent: bool
):
    if linked_parent:
        if os.name == "nt":
            pytest.skip("symbolic links require privileges on Windows")
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        root = alias / "project"
    else:
        root = tmp_path / "project"
    code, result = _run(root, "status")
    assert code == 0
    assert result["engines"] == []
    assert result["current"] is result["channel"] is result["variant"] is None
    assert not root.exists()
    root.mkdir()
    settings = {
        "format": "dinkster.project/1",
        "channel": "nightly",
        "variant": "win-cpu",
        "port": 3639,
        "mirrors": [str(tmp_path / "mirror")],
    }
    (root / "environment.json").write_text(json.dumps(settings))
    before = (root / "environment.json").read_bytes()
    code, result = _run(root, "status")
    assert code == 0
    assert result["channel"] == "nightly"
    assert result["variant"] == "win-cpu"
    assert result["engines"] == []
    assert (root / "environment.json").read_bytes() == before
    assert not (root / "runtime").exists()


def test_status_and_doctor_label_the_active_user_defined_base(tmp_path: Path):
    generation = _generation(tmp_path, 1).parent
    manifest = {
        "format": "dinkster.base/1",
        "id": "sha256:" + "a" * 64,
        "variant": "my-cuda",
        "source": "user-defined",
        "wheels": [
            {
                "name": "torch",
                "version": "2.14.1+cu130",
                "index": "https://download.pytorch.org/whl/cu130",
            }
        ],
    }
    (generation / "base-manifest.json").write_text(json.dumps(manifest))
    metadata = generation / "build.json"
    document = json.loads(metadata.read_text())
    document["variant"] = "my-cuda"
    document["files"]["base-manifest.json"] = content_digest(generation / "base-manifest.json")
    metadata.write_text(json.dumps(document))
    (tmp_path / "current").write_text("1\n")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for operation in ("status", "doctor"):
        code, result = _run(tmp_path, operation)
        assert code == 0
        assert result["base"] == {
            "variant": "my-cuda",
            "origin": "user-defined",
            "torchVersion": "2.14.1+cu130",
            "torchIndex": "https://download.pytorch.org/whl/cu130",
            "label": None,
        }
        human = subprocess.run(
            [
                sys.executable,
                "-m",
                "dinkster_supervisor.environment",
                "--root",
                str(tmp_path),
                operation,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert "Variant: my-cuda (user-defined)" in human.stdout
        assert "Torch: 2.14.1+cu130" in human.stdout
        assert "Torch index: https://download.pytorch.org/whl/cu130" in human.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    "case",
    [
        "labelled",
        "legacy",
        "missing",
        "base-id",
        "engine-hash",
        "engine-version",
        "base-label",
        "engine-label",
    ],
)
def test_installed_labels_use_active_manifest_identities_without_channel_reads(
    tmp_path: Path, case: str
):
    root = tmp_path / "project"
    generation = _generation(root, 1).parent
    base, engine, _ = _mirror(tmp_path / "mirror")
    if case != "legacy":
        base["label"] = "2026-10-31-py3.14.10-torch1.0-" + base["id"][7:15]
        engine["label"] = "2026-11-01-0.4.0.dev1"
    if case == "base-label":
        base["label"] = ["not a label"]
    elif case == "engine-label":
        engine["label"] = 7
    store = tmp_path / "cache"
    base_path = store / "bases" / base["id"][7:] / "manifest.json"
    base_path.parent.mkdir(parents=True)
    base_path.write_text(json.dumps(base))
    (root / "store").write_text(str(store))
    engine_path = generation / "engine-manifest.json"
    engine_path.write_text(json.dumps(engine) + "\n")
    metadata = generation / "build.json"
    document = json.loads(metadata.read_bytes())
    document.update(base=base["id"], variant=base["variant"])
    document["engine"] = {"release": content_digest(engine_path), "version": engine["version"]}
    document["files"]["engine-manifest.json"] = content_digest(engine_path)
    if case == "base-id":
        base["python"] = "3.14.9"
        base_path.write_text(json.dumps(base))
    elif case == "engine-hash":
        engine_path.write_text(json.dumps({**engine, "label": "wrong-release"}))
    elif case == "engine-version":
        document["engine"]["version"] = "9.0"
    elif case == "missing":
        base_path.unlink()
        engine_path.unlink()
        del document["files"]["engine-manifest.json"]
    metadata.write_text(json.dumps(document))
    (root / "current").write_text("1\n")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for operation in ("status", "doctor"):
        code, result = _run(root, operation)
        if case in {"base-label", "engine-label"}:
            assert code == 1 and result["status"] == "error"
            assert "label must be a string or null" in result["error"]
        elif case in {"base-id", "engine-hash", "engine-version"}:
            assert code == 1 and result["status"] == "error"
            assert "match" in result["error"]
        elif case == "missing":
            assert code == 0
            assert "base" not in result and "engine" not in result
        else:
            assert code == 0
            assert result["base"]["variant"] == base["variant"]
            assert result["base"]["label"] == base.get("label")
            assert result["engine"] == {**document["engine"], "label": engine.get("label")}
            human = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dinkster_supervisor.environment",
                    "--root",
                    str(root),
                    operation,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            if case == "labelled":
                assert f"Base label: {base['label']}" in human.stdout
                assert f"Engine label: {engine['label']}" in human.stdout
            else:
                assert "Base label:" not in human.stdout and "Engine label:" not in human.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_malformed_existing_project_returns_error_not_empty_status(tmp_path: Path):
    (tmp_path / "environment.json").write_text('{"format":"dinkster.project/1","channel":null}')
    code, result = _run(tmp_path, "status")
    assert code == 1
    assert result["status"] == "error"
    assert "project channel" in result["error"]


@pytest.mark.parametrize("source", ["published", "local"])
def test_doctor_reports_actual_content_drift_with_nonzero_exit_and_no_repair(
    tmp_path: Path, source: str
):
    generation = _generation(tmp_path, 1).parent
    metadata = generation / "build.json"
    document = json.loads(metadata.read_text())
    document["source"] = source
    metadata.write_text(json.dumps(document))
    (tmp_path / "current").write_bytes(b"1\n")
    code, result = _run(tmp_path, "doctor")
    assert code == 0
    assert result["status"] == "clean"
    if source == "local":
        assert result["source"] == "local"
        assert "locally resolved, no base manifest" in result["plan"][0]
    module = generation / "python/module.py"
    module.write_bytes(b"ANSWER = 13\n")
    code, result = _run(tmp_path, "doctor")
    assert code == 1
    assert result["status"] == "drift"
    assert result["changes"] == ["modified: python/module.py"]
    assert module.read_bytes() == b"ANSWER = 13\n"
    assert (tmp_path / "current").read_bytes() == b"1\n"


def _runtime_project(root: Path) -> int:
    make_generation(root)
    port = free_port()
    (root / "current").write_text("1\n")
    (root / "environment.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "port": port,
                "mirrors": [str(root / "mirror")],
            }
        )
    )
    return port


def test_channel_update_refuses_local_generation_before_reading_channel(tmp_path: Path):
    _runtime_project(tmp_path)
    settings = tmp_path / "environment.json"
    project = json.loads(settings.read_text())
    project["mirrors"] = []
    settings.write_text(json.dumps(project))
    metadata = tmp_path / "builds/1/build.json"
    document = json.loads(metadata.read_text())
    document["source"] = "local"
    metadata.write_text(json.dumps(document))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    code, result = _run(tmp_path, "update", "--plan")
    assert code == 1
    assert "cannot channel-update a locally resolved generation" in result["error"]
    assert "fresh published generation" in result["error"]
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_from_scratch_plan_never_reads_release_hosts_or_creates_root(tmp_path: Path):
    root = tmp_path / "prospective"
    source = Path(__file__).resolve().parents[1]
    variant = {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu")
    code, result = _run(
        root,
        "setup",
        "--from-scratch",
        "--variant",
        variant,
        "--source",
        str(source),
        "--python",
        sys.executable,
        "--mirror",
        str(tmp_path / "nonexistent-release-host"),
        "--plan",
    )
    assert code == 0
    assert "resolve" in result["plan"][0]
    assert "uv.lock" in result["plan"][0]
    assert any("torch index" in line for line in result["plan"])
    assert not root.exists()


@pytest.mark.parametrize("hidden", [False, True])
def test_environment_plan_discloses_library_migrations_and_excluded_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden: bool
):
    root, library = tmp_path / "project", tmp_path / "library"
    monkeypatch.setenv("DINKSTER_LIBRARY_ROOT", str(library))
    library.mkdir()
    if hidden:
        (library / "project-folders.json").write_text('{"hiddenModels": {"art": ["model"]}}')
    retained = root / "history"
    retained.mkdir(parents=True)
    for number, version in enumerate(("0.0.3", "0.0.4", "0.0.14"), 1):
        (retained / f"{number:020d}.json").write_text(
            json.dumps(
                {
                    "format": "dinkster.install-history/1",
                    "id": number,
                    "snapshot": {"engine": {"version": version}, "metadata": {}},
                }
            )
        )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    variant = {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu")
    code, result = _run(
        root,
        "setup",
        "--from-scratch",
        "--variant",
        variant,
        "--source",
        str(Path(__file__).resolve().parents[1]),
        "--python",
        sys.executable,
        "--plan",
    )
    assert code == 0, result
    minimum = "0.0.15" if hidden else "0.0.4"
    assert f"library after apply requires Dinkster {minimum} or newer" in result["plan"]
    excluded = next(line for line in result["plan"] if line.startswith("library excludes"))
    assert "history entry 1 (Dinkster 0.0.3)" in excluded
    assert ("history entry 2 (Dinkster 0.0.4)" in excluded) is hidden
    assert ("history entry 3 (Dinkster 0.0.14)" in excluded) is hidden
    for arguments in (("history",), ("history", "show", "3")):
        code, listed = _run(root, *arguments)
        assert code == 0, listed
        if hidden:
            assert listed["plan"] == result["plan"][-2:]
        else:
            assert "library excludes no known retained release snapshots" in listed["plan"]
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("kind", ["foreign-variant", "pack-lock", "catalog"])
def test_from_scratch_refuses_incompatible_inputs_before_creation(tmp_path: Path, kind: str):
    variant = "linux-cpu" if os.name == "nt" else "win-cpu"
    extra = (
        ["--python", sys.executable]
        if kind == "foreign-variant"
        else ["--pack-lock", str(tmp_path / "pack.json")]
        if kind == "pack-lock"
        else ["--catalog-lock", str(tmp_path / "catalog.lock"), "--artifacts", str(tmp_path)]
    )
    root = tmp_path / "absent-project"
    code, result = _run(root, "setup", "--from-scratch", "--variant", variant, *extra, "--plan")
    assert code == 1
    assert ("does not match host" if kind == "foreign-variant" else "cannot be combined") in result[
        "error"
    ]
    assert not root.exists()


def test_from_scratch_rejects_a_checkout_lock_changed_after_planning(tmp_path: Path):
    from dinkster_supervisor.environment_local import (
        DEFAULT_VARIANTS,
        local_selection,
        prepare_local,
    )

    source = Path(__file__).resolve().parents[1]
    for relative in (
        "uv.lock",
        "pyproject.toml",
        "packages/dinkster-workers/src/dinkster_workers/boundary.py",
    ):
        target = tmp_path / "checkout" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    checkout = tmp_path / "checkout"
    variant = {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu")
    selection = local_selection(checkout, variant, DEFAULT_VARIANTS, Path(sys.executable))
    lock = checkout / "uv.lock"
    lock.write_bytes(lock.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checkout lock changed after planning"):
        prepare_local(checkout, selection, tmp_path / "absent-store")
    assert not (tmp_path / "absent-store").exists()


def test_missing_variant_row_requires_explicit_local_interpreter(tmp_path: Path):
    variants = tmp_path / "variants.toml"
    variants.write_text('[base]\npython = "3.14.10"\npackages = ["torch"]\n[variants]\n')
    variant = {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu")
    root = tmp_path / "prospective"
    arguments = (
        "setup",
        "--from-scratch",
        "--variant",
        variant,
        "--variants",
        str(variants),
        "--source",
        str(Path(__file__).resolve().parents[1]),
        "--plan",
    )
    code, result = _run(root, *arguments)
    assert code == 1
    assert f"missing variants.toml row: variants.{variant}" in result["error"]
    assert "--python" in result["error"]
    code, result = _run(root, *arguments, "--python", sys.executable)
    assert code == 0
    assert any(sys.executable in line for line in result["plan"])
    assert not root.exists()


def test_update_workflow_plan_uses_core_adapter_without_mutating_project_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_library import LibraryClient

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    _mirror(mirror, variant="linux-cpu")
    _runtime_project(root)
    shutil.rmtree(root / "builds/1/python")
    _workspace_tree(root / "builds/1/python")
    (root / "store").write_text(str(store) + "\n")
    settings = json.loads((root / "environment.json").read_bytes())
    asset = AssetRef(digest_bytes(b"required bytes"), "required.bin", 14).to_wire()
    settings.update(mirrors=[str(mirror)], additions={"assets": [asset]})
    (root / "environment.json").write_text(json.dumps(settings))
    (root / "builds/1/dinkster.lock").write_text(Lockfile().record_json())

    def snapshot() -> dict[Path, bytes | None]:
        # Helper attachment and SQLite shared-memory locks are not user records.
        return {
            path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
            for path in tmp_path.rglob("*")
            if path.name not in {"library.sqlite-wal", "library.sqlite-shm"}
            and not (
                path.parent == root / "dinkster-library"
                and (path.match("helper-*.lock") or path.name == "records.sqlite-shm")
            )
        }

    before = snapshot()
    for flags in ((), ("--plan",)):
        code, result = _run(root, "update", *flags)
        assert code == 0, result
        assert result["current"] == 1
        requirements = next(
            line for line in result["plan"] if line.startswith("workflow requirements: ")
        )
        assert json.loads(requirements.split(": ", 1)[1])["assets"] == [asset]
        assert any("feasibility: pending" in line for line in result["plan"])
        assert snapshot() == before

    workflow_asset = AssetRef(digest_bytes(b"workflow bytes"), "workflow.bin", 14).to_wire()
    document = {
        "format": "dinkster-workflow",
        "formatVersion": 1,
        "root": "main",
        "graphs": {"main": {"nodes": {}, "valueSources": {"asset": {"value": workflow_asset}}}},
    }
    data = json.dumps(document).encode()
    identity = digest_bytes(data)
    vault = AssetVault(root / "dinkster-library/vault")
    with vault.writer(identity) as writer:
        writer.write(data)
        writer.commit()
    library = LibraryStore(root / "dinkster-library/library.sqlite")
    try:
        record = library.create(
            "local", "Workflow", identity, "application/x-dinkster-workflow+json"
        )
    finally:
        library.close()
    with LibraryClient(root / "dinkster-library"):
        pass
    settings["workflows"] = [{"scope": "local", "id": record.id}]
    (root / "environment.json").write_text(json.dumps(settings))
    before = snapshot()
    code, combined = _run(root, "update", "--plan")
    assert code == 0, combined
    requirements = next(
        line for line in combined["plan"] if line.startswith("workflow requirements: ")
    )
    assert {row["digest"] for row in json.loads(requirements.split(": ", 1)[1])["assets"]} == {
        asset["digest"],
        workflow_asset["digest"],
    }
    assert snapshot() == before
    settings.pop("workflows")
    (root / "environment.json").write_text(json.dumps(settings))

    _release_sources(mirror)
    before = snapshot()
    code, missing = _run(root, "update", "--plan")
    assert code == 0 and any("feasibility: pending" in line for line in missing["plan"])
    assert snapshot() == before
    acquired = acquire_sources(discover((str(mirror),), "nightly"), store)
    assert acquired is not None
    lock, artifacts = acquired
    settings["additions"]["packs"] = [{"pack": "first-pack", "version": "9.9.9"}]
    (root / "environment.json").write_text(json.dumps(settings))
    before = snapshot()
    for extra in ((), ("--catalog-lock", str(lock), "--artifacts", str(artifacts))):
        code, unavailable = _run(root, "update", "--plan", *extra)
        assert code == 0 and any("feasibility: pending" in line for line in unavailable["plan"])
        assert '"version": "9.9.9"' in "\n".join(unavailable["plan"])
        assert snapshot() == before
    cached = next(artifacts.rglob("*.zip"))
    cached.write_bytes(b"x" * cached.stat().st_size)
    before = snapshot()
    code, corrupt = _run(root, "update", "--plan")
    assert code == 0 and any("feasibility: pending" in line for line in corrupt["plan"])
    assert snapshot() == before


@pytest.mark.parametrize("explicit", [False, True])
def test_workflow_update_prepares_target_registry_inputs_before_journaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
):
    from tests.fixtures.environments.build import build

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    command = fixture["commands"]["setup"]
    setup = run_with_diagnostics(command, timeout=60)
    assert setup.returncode == 0, setup.stdout + setup.stderr
    try:
        assert _run(root, "stop", "--yes")[0] == 0
        settings = json.loads((root / "environment.json").read_bytes())
        settings["additions"] = {"packs": [], "assets": []}
        (root / "environment.json").write_text(json.dumps(settings))
        (mirror / "channels/nightly.json").write_bytes(Path(fixture["updatedChannel"]).read_bytes())
        overrides = ()
        if explicit:
            catalog, artifacts = tmp_path / "selected.lock", tmp_path / "selected-artifacts"
            catalog.write_text(Lockfile().record_json())
            artifacts.mkdir()
            overrides = ("--catalog-lock", str(catalog), "--artifacts", str(artifacts))
        before_project = (root / "environment.json").read_bytes()
        before_journals = {path.name: path.read_bytes() for path in (root / "journal").iterdir()}
        for problem, expected in (
            ("refuse", "saved pack version is unavailable"),
            ("revision", "workflow inputs changed"),
            ("catalog", "target catalog changed"),
        ):
            monkeypatch.setenv("DINKSTER_FIXTURE_WORKFLOW_PROBLEM", problem)
            code, refused = _run(root, "update", *overrides, "--yes")
            assert code == 1 and expected in refused["error"], refused
            assert (root / "current").read_text() == "1\n"
            assert (root / "environment.json").read_bytes() == before_project
            assert {path.name: path.read_bytes() for path in (root / "journal").iterdir()} == (
                before_journals
            )
            assert not (root / "builds/2").exists()
            assert not list(store.glob(".registry-*"))
            assert not list(store.glob(".workflow-*"))
        monkeypatch.delenv("DINKSTER_FIXTURE_WORKFLOW_PROBLEM")
        code, applied = _run(root, "update", *overrides, "--yes")
        assert code == 0 and applied["current"] == 2, applied
        assert (root / "environment.json").read_bytes() == before_project
        assert json.loads((root / "builds/2/build.json").read_bytes())["engine"]["version"] == "2.0"
        assert _run(root, "doctor")[1]["status"] == "clean"
        assert not list(store.glob(".registry-*"))
        assert not list(store.glob(".workflow-*"))
    finally:
        _run(root, "stop", "--yes")


@pytest.mark.parametrize(
    ("flags", "terminal", "answer", "code", "status", "prompted"),
    [
        ((), False, "", 1, "error", False),
        ((), True, "n\n", 0, "planned", True),
        ((), True, "yes\n", 0, "applied", True),
        (("--yes",), False, "", 0, "applied", False),
        (("--plan",), True, "yes\n", 0, "planned", False),
        (("--json",), True, "yes\n", 0, "planned", False),
    ],
)
def test_runtime_consent_through_environment_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
    terminal: bool,
    answer: str,
    code: int,
    status: str,
    prompted: bool,
) -> None:
    from dinkster_supervisor import environment

    _runtime_project(tmp_path)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    stream = io.StringIO(answer)
    monkeypatch.setattr(stream, "isatty", lambda: terminal)
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(sys, "argv", ["caller", "unrelated-argument"])
    with pytest.raises(SystemExit) as exited:
        environment.main(["--root", str(tmp_path), "stop", *flags])
    captured = capsys.readouterr()
    assert exited.value.code == code
    assert sys.argv == ["caller", "unrelated-argument"]
    if "--json" in flags:
        assert json.loads(captured.out)["status"] == status
    else:
        assert f"stop: {status}" in captured.out
    assert ("Apply this plan? [y/N]" in captured.out) is prompted
    if code:
        assert "--yes" in captured.err and "--plan" in captured.err
    if status != "applied":
        after = {
            path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert not (tmp_path / ".lock").exists()


def test_runtime_confirmation_assignment_and_stop_first_through_cli(tmp_path: Path):
    port = _runtime_project(tmp_path)
    arguments = ("start", "--gpu", "0", "--gpu", "GPU-aB12")
    for flags in ((), ("--plan",)):
        code, result = _run(tmp_path, *arguments, *flags)
        assert code == 0
        assert result["status"] == "planned"
        assert "CUDA_VISIBLE_DEVICES=0,GPU-aB12" in result["plan"][1]
        assert not (tmp_path / "runtime").exists()
        assert not (tmp_path / ".lock").exists()
    code, result = _run(tmp_path, *arguments, "--yes")
    assert code == 0
    try:
        assert result["status"] == "applied"
        assert result["url"] == f"http://127.0.0.1:{port}"
        with urlopen(result["url"] + "/api/cuda", timeout=5) as response:  # noqa: S310
            assert json.loads(response.read()) == {"cuda": "0,GPU-aB12"}
        record_path = tmp_path / "runtime/1/main.json"
        original = record_path.read_bytes()
        code, repeated = _run(tmp_path, *arguments, "--yes")
        assert code == 0
        assert repeated["url"] == result["url"]
        assert record_path.read_bytes() == original
        code, changed = _run(tmp_path, "start", "--gpu", "1", "--yes")
        assert code == 1
        assert "stop it first" in changed["error"]
        assert record_path.read_bytes() == original
        code, planned = _run(tmp_path, "stop", "--plan")
        assert code == 0 and planned["status"] == "planned"
        assert record_path.read_bytes() == original
    finally:
        code, stopped = _run(tmp_path, "stop", "--yes")
        assert code == 0
        assert stopped["engines"][0]["state"] == "stopped"
    assert (tmp_path / "current").read_text() == "1\n"


@pytest.mark.parametrize("command", ["start", "stop"])
def test_runtime_waits_for_automatic_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
):
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    from threading import Event

    from dinkster_supervisor import environment, environment_collection

    _runtime_project(tmp_path)
    assert _run(tmp_path, "start", "--yes")[0] == 0
    entered, release = Event(), Event()
    policy = environment_collection.collection_policy

    def paused_policy(root):
        entered.set()
        assert release.wait(10), "cleanup was not released"
        return policy(root)

    monkeypatch.setattr(environment_collection, "collection_policy", paused_policy)
    args = environment.argument_parser().parse_args(
        ["--root", str(tmp_path), command, "--yes", "--json"]
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            cleanup = threads.submit(
                environment_collection.automatic_generation_collection, tmp_path
            )
            try:
                assert entered.wait(10), "cleanup did not acquire the project"
                started = threads.submit(environment.run, args)
                with pytest.raises(FutureTimeout):
                    started.result(timeout=0.5)
            finally:
                release.set()
            assert cleanup.result(timeout=10)["status"] == "deferred"
            assert started.result(timeout=10) == 0
        code, status = _run(tmp_path, "status")
        assert code == 0, status
        assert status["engines"][0]["state"] == ("ready" if command == "start" else "stopped")
    finally:
        release.set()
        assert _run(tmp_path, "stop", "--yes")[0] == 0


def test_cpu_assignment_masks_inherited_gpu_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    port = _runtime_project(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    code, started = _run(tmp_path, "start", "--yes")
    assert code == 0
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/cuda", timeout=5) as response:  # noqa: S310
            assert json.loads(response.read()) == {"cuda": ""}
        assert started["status"] == "applied"
    finally:
        assert _run(tmp_path, "stop", "--yes")[0] == 0


def test_runtime_apply_refuses_writer_contention_and_recovers_incomplete_journal(tmp_path: Path):
    _runtime_project(tmp_path)
    with writer_lock(tmp_path):
        code, refused = _run(tmp_path, "start", "--yes")
    assert code == 1
    assert "another environment writer" in refused["error"]
    assert _run(tmp_path, "status")[1]["engines"] == []
    Journal("update", 1, 2, "building").write(tmp_path / "journal/interrupted.json")
    code, recovered = _run(tmp_path, "start", "--yes")
    try:
        assert code == 0
        assert recovered["engines"][0]["state"] == "ready"
        assert Journal.read(tmp_path / "journal/interrupted.json").stage == "rolled-back"
        assert (tmp_path / "current").read_bytes() == b"1\n"
    finally:
        assert _run(tmp_path, "stop", "--yes")[0] == 0


@pytest.mark.parametrize("choice", ["example", "example=shared", "../escape=isolated"])
def test_invalid_inference_pack_switch_does_not_create_a_project(tmp_path: Path, choice: str):
    root = tmp_path / "new-project"
    code, refused = _run(
        root, "setup", "--variant", "linux-cpu", "--inference-pack", choice, "--yes"
    )
    assert code == 1
    assert "--inference-pack requires PACK=auto or PACK=isolated" in refused["error"]
    assert not root.exists()


def test_named_engine_requires_port_and_invalid_assignment_does_not_mutate(tmp_path: Path):
    _runtime_project(tmp_path)
    code, refused = _run(tmp_path, "start", "--engine", "extra", "--yes")
    assert code == 1
    assert "requires --port" in refused["error"]
    code, refused = _run(tmp_path, "start", "--gpu", "../not-a-device", "--yes")
    assert code == 1
    assert "GPU ids" in refused["error"]
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / ".lock").exists()


def test_confirmed_offline_setup_update_rollback_and_busy_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    uv = shutil.which("uv")
    assert uv is not None
    tools = tmp_path / "guarded-tools"
    tools.mkdir()
    guard = tools / "guard.py"
    guard.write_text(
        "import subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "if 'compile' in args or (('sync' in args or 'install' in args) and\n"
        "    ('--offline' not in args or '--no-index' not in args)):\n"
        "    sys.exit('published paths must not resolve dependencies')\n"
        f"sys.exit(subprocess.call([{uv!r}, *args]))\n"
    )
    shim = tools / ("uv.cmd" if os.name == "nt" else "uv")
    shim.write_text(
        f'@"{sys.executable}" "{guard}" %*\n'
        if os.name == "nt"
        else f'#!/bin/sh\n{shlex.quote(sys.executable)} {shlex.quote(str(guard))} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "UV_INDEX_URL"):
        monkeypatch.setenv(variable, "http://127.0.0.1:1/unreachable")
    fixture = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "fixture"),
                "--variant",
                "fixture-cpu",
            ],
            text=True,
            timeout=60,
        )
    )
    mirror = Path(fixture["mirror"])
    root, store = tmp_path / "project", tmp_path / "__pycache__/store"
    # Software-state comparisons must not read the live library's Windows byte lock.
    library = tmp_path / "library"
    monkeypatch.setenv("DINKSTER_LIBRARY_ROOT", str(library))
    base = json.loads(Path(fixture["baseManifest"]).read_bytes())
    base_id = base["id"][7:]
    port = free_port()
    package_index = "https://packages.example.org/simple"
    arguments = (
        "setup",
        "--mirror",
        str(mirror),
        "--package-index",
        package_index,
        "--channel",
        "nightly",
        "--variant",
        "fixture-cpu",
        "--data",
        str(store),
        "--port",
        str(port),
    )
    for flags in ((), ("--plan",)):
        code, planned = _run(root, *arguments, *flags)
        assert code == 0, planned
        assert planned["downloadBytes"] > 0 and planned["status"] == "planned"
        assert any(
            "engine binding main" in step
            and f"127.0.0.1:{port}" in step
            and step.endswith("CUDA_VISIBLE_DEVICES=")
            for step in planned["plan"]
        )
        assert not root.exists() and not store.exists() and not (tmp_path / "user-data").exists()
    code, assigned = _run(
        root, *arguments, "--engine", "preview", "--gpu", "GPU-aB12", "--gpu", "2", "--plan"
    )
    assert code == 0, assigned
    assert any(
        "engine binding preview" in step
        and f"127.0.0.1:{port}" in step
        and "CUDA_VISIBLE_DEVICES=GPU-aB12,2" in step
        for step in assigned["plan"]
    )
    assert not root.exists() and not store.exists() and not (tmp_path / "user-data").exists()
    root.mkdir()
    (root / "queue.json").write_text('{"running": [], "queued": []}')
    failed_store = tmp_path / "failed-store"
    selected_release = discover((str(mirror),), "nightly")
    engine_wheel = (
        mirror
        / "engine"
        / selected_release.channel.engine_release[7:]
        / "fixture_engine-1.0-py3-none-any.whl"
    )
    original_wheel = engine_wheel.read_bytes()
    engine_wheel.write_bytes(b"damaged release wheel")
    failed_arguments = list(arguments)
    failed_arguments[failed_arguments.index(str(store))] = str(failed_store)
    code, failed = _run(root, *failed_arguments, "--yes")
    assert code == 1 and failed["status"] == "error"
    assert engine_wheel.name in failed["error"], failed
    assert "corrupt or short data" in failed["error"], failed
    assert not (root / "environment.json").exists() and not (root / "store").exists()
    assert not (failed_store / "projects.json").exists()
    assert not (tmp_path / "user-data/dinkster/stores.json").exists()
    assert any(path.is_file() for path in (failed_store / "bases").rglob("*"))
    assert not list((root / "journal").glob("*.json"))
    engine_wheel.write_bytes(original_wheel)
    for problem, expected in (
        ("refuse", "fixture-pack@1.0 declares >=9"),
        ("version", "inconsistent source inputs"),
        ("path", "invalid lock paths"),
    ):
        monkeypatch.setenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM", problem)
        code, refused = _run(root, *arguments, "--yes")
        assert code == 1 and expected in refused["error"], refused
        assert not (root / "environment.json").exists() and not (root / "store").exists()
        assert not (store / "projects.json").exists()
        assert not list((root / "journal").glob("*.json"))
        assert not list(store.glob(".registry-*"))
    monkeypatch.delenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
    code, installed = _run(
        root,
        *arguments,
        "--engine-release",
        selected_release.channel.engine_release,
        "--base-id",
        base["id"],
        "--yes",
    )
    assert code == 0, installed
    try:
        assert installed["current"] == 1 and installed["status"] == "applied"
        assert installed["unavailablePacks"] == {"fixture-pack": "registry offline"}
        assert json.loads((root / "builds/1/dinkster.lock").read_text())["packs"] == []
        assert not list(store.glob(".registry-*"))
        assert installed["url"] == f"http://127.0.0.1:{port}"
        assert (library / ".engine.lock").is_file()
        code, competing = _run(
            root, "start", "--engine", "preview", "--port", str(free_port()), "--yes"
        )
        assert code == 1 and "already in use by engine main" in competing["error"], competing
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        assert _run(root, "doctor")[1]["status"] == "clean"
        initial_settings = (root / "environment.json").read_bytes()
        assert json.loads(initial_settings)["packageIndex"] == package_index
        assert json.loads((store / "projects.json").read_bytes())["projects"] == [str(root)]
        other = tmp_path / "second project"
        code, second_plan = _run(
            other,
            "setup",
            "--mirror",
            str(mirror),
            "--channel",
            "nightly",
            "--variant",
            "fixture-cpu",
            "--plan",
        )
        assert code == 0, second_plan
        assert second_plan["store"] == str(store) and not other.exists()
        code, conflict = _run(
            other,
            "setup",
            "--mirror",
            str(mirror),
            "--channel",
            "nightly",
            "--variant",
            "fixture-cpu",
            "--data",
            str(tmp_path / "another-store"),
            "--yes",
        )
        assert code == 1 and "volume already uses store" in conflict["error"]
        assert not other.exists() and not (tmp_path / "another-store").exists()
        subprocess.run(fixture["commands"]["advance"], check=True, capture_output=True, timeout=30)
        before_selection = {
            p.relative_to(root): content_digest(p)
            for p in root.rglob("*")
            if p.is_file()
            and not (p.is_relative_to(root / "runtime") and p.name.endswith((".log", ".lock")))
        }
        code, stale = _run(
            root,
            "update",
            "--channel",
            "stable",
            "--engine-release",
            selected_release.channel.engine_release,
            "--base-id",
            base["id"],
            "--yes",
        )
        assert code == 1 and "approved engine release/base" in stale["error"], stale
        code, selection_plan = _run(root, "update", "--channel", "stable", "--plan")
        assert code == 0 and any(
            "change project selection" in step for step in selection_plan["plan"]
        )
        assert {
            p.relative_to(root): content_digest(p)
            for p in root.rglob("*")
            if p.is_file()
            and not (p.is_relative_to(root / "runtime") and p.name.endswith((".log", ".lock")))
        } == before_selection
        (root / "queue.json").write_text('{"running": ["real-job"], "queued": []}')
        before = sorted(path.name for path in (root / "journal").iterdir())
        code, busy = _run(root, "update", "--yes")
        assert code == 1 and "Finish or cancel" in busy["error"]
        assert sorted(path.name for path in (root / "journal").iterdir()) == before
        assert (root / "current").read_text() == "1\n"
        (root / "queue.json").write_text('{"running": [], "queued": []}')
        before_registry = {path: path.read_bytes() for path in (root / "journal").glob("*.json")}
        monkeypatch.setenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM", "refuse")
        code, refused = _run(root, "update", "--yes")
        monkeypatch.delenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
        assert code == 1 and "fixture-pack@1.0 declares >=9" in refused["error"]
        assert (root / "current").read_text() == "1\n"
        assert (root / "environment.json").read_bytes() == initial_settings
        assert {
            path: path.read_bytes() for path in (root / "journal").glob("*.json")
        } == before_registry
        assert not (root / "builds/2").exists()
        assert not list(store.glob(".registry-*"))
        script = """
import os
import sys
from dinkster_supervisor import environment
def interrupt(**kwargs):
    os._exit(73)
environment.build_generation = interrupt
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'update', '--yes', '--json']
environment.main()
"""
        interrupted = run_with_diagnostics([sys.executable, "-c", script, str(root)], timeout=30)
        assert interrupted.returncode == 73
        journals = sorted((root / "journal").glob("*.json"))
        assert Journal.read(journals[-1]).stage == "building"
        assert (root / "current").read_text() == "1\n"
        started = time.monotonic()
        reviewed = discover((str(mirror),), "stable")
        from dinkster_supervisor import environment_collection as collection

        entered, release = Event(), Event()
        policy = collection.collection_policy

        def held_policy(project: Path):
            entered.set()
            assert release.wait(30), "test did not release maintenance"
            return policy(project)

        with monkeypatch.context() as maintenance_patch, ThreadPoolExecutor(max_workers=2) as pool:
            maintenance_patch.setattr(collection, "collection_policy", held_policy)
            maintenance = pool.submit(collection.automatic_generation_collection, root)
            try:
                assert entered.wait(10), "maintenance did not acquire the project"
                update = pool.submit(
                    _run,
                    root,
                    "update",
                    "--channel",
                    "stable",
                    "--engine-release",
                    reviewed.channel.engine_release,
                    "--base-id",
                    base["id"],
                    "--yes",
                )
                with pytest.raises(TimeoutError):
                    update.result(timeout=2)
            finally:
                release.set()
            maintenance.result(timeout=30)
            code, updated = update.result(timeout=120)
        elapsed = time.monotonic() - started
        assert code == 0, updated
        assert json.loads((root / "environment.json").read_bytes())["channel"] == "stable"
        assert json.loads((root / "environment.json").read_bytes())["packageIndex"] == package_index
        assert Journal.read(journals[-1]).stage == "rolled-back"
        assert updated["current"] == 2 and updated["url"] != installed["url"]
        first, second = (root / "builds" / str(n) for n in (1, 2))
        _assert_shared(next(first.rglob("fixture_base.py")), next(second.rglob("fixture_base.py")))
        python = second / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        assert (
            subprocess.check_output(
                [str(python), "-B", "-c", "import fixture_engine; print(fixture_engine.VALUE)"]
            ).strip()
            == b"59"
        )
        retained = python.stat()
        retained_digest = content_digest(python)
        retained_history = {path: path.read_bytes() for path in (root / "history").glob("*.json")}
        code, rolled_back = _run(root, "rollback", "--yes")
        assert code == 0, rolled_back
        assert json.loads((root / "environment.json").read_bytes())["channel"] == "stable"
        assert rolled_back["current"] == 3 and second.exists()
        if sys.platform == "linux" or os.name == "nt":
            assert not first.exists()
        else:
            assert first.exists()
        assert retained_history and all(
            path.read_bytes() == data for path, data in retained_history.items()
        )
        after = python.stat()
        assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
            retained.st_dev,
            retained.st_ino,
            retained.st_size,
            retained.st_mtime_ns,
        )
        assert content_digest(python) == retained_digest
        assert _run(root, "doctor")[1]["status"] == "clean"
        current = root / "builds/3"
        restored_python = current / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        assert (
            subprocess.check_output(
                [
                    str(restored_python),
                    "-B",
                    "-c",
                    "import fixture_engine; print(fixture_engine.VALUE)",
                ]
            ).strip()
            == b"41"
        )
        source_release, payloads = _release_sources(mirror)
        supplied = acquire_sources(discover((str(mirror),), "nightly"), store)
        assert supplied is not None
        before_journals = {path: path.read_bytes() for path in (root / "journal").glob("*.json")}
        state_paths = [root / name for name in ("current", "environment.json", "store")]
        state_paths.append(store / "projects.json")
        before_state = {path: path.read_bytes() for path in state_paths}
        code, source_refused = _run(
            root,
            "update",
            "--yes",
            "--catalog-lock",
            str(supplied[0]),
            "--artifacts",
            str(supplied[1]),
        )
        assert (
            code == 1 and "pack source staging: invalid admission result" in source_refused["error"]
        )
        # The empty-pack fixture must refuse nonempty explicit sources, not omit them.
        assert (root / "current").read_text() == "3\n" and not (root / "builds/4").exists()
        cached_release = object_path(store / "engines", "sha256:" + source_release.name)
        assert (cached_release / "official-packs.lock").read_bytes() == payloads[
            "official-packs.lock"
        ]
        assert sorted(
            path.read_bytes() for path in (cached_release / "artifacts").rglob("*.zip")
        ) == sorted(payload for name, payload in payloads.items() if name.endswith(".zip"))
        assert {path: path.read_bytes() for path in (root / "journal").glob("*.json")} == (
            before_journals
        )
        assert {path: path.read_bytes() for path in state_paths} == before_state
        assert not list(store.glob(".workflow-*"))
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        _executable_release(mirror, base, "3.0", 61)
        engine = read_channel(mirror / "channels/nightly.json")["engine"]["release"][7:]
        (mirror / "engine" / engine / "fixture_engine-3.0-py3-none-any.whl").write_bytes(b"corrupt")
        before_journals = {path: path.read_bytes() for path in (root / "journal").glob("*.json")}
        before_settings = (root / "environment.json").read_bytes()
        code, refused = _run(root, "update", "--channel", "nightly", "--yes")
        assert code == 1 and refused["status"] == "error"
        assert (root / "current").read_text() == "3\n" and not (root / "builds/4").exists()
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        assert {path: path.read_bytes() for path in (root / "journal").glob("*.json")} == (
            before_journals
        )
        assert (root / "environment.json").read_bytes() == before_settings
        mirror.rename(tmp_path / "unavailable-mirror")
        damaged = next(current.rglob("fixture_engine.py"))
        damaged.write_text("VALUE = 999\n")
        (current / "constraints.txt").unlink()
        retained = json.loads((current / "build.json").read_bytes())
        release_cache = object_path(store / "engines", retained["engine"]["release"])
        (release_cache / f"constraints-{retained['variant']}.txt").write_text(
            "fixture-engine==99.0\n"
        )
        code, drift = _run(root, "doctor")
        assert code == 1 and drift["status"] == "drift"
        assert "missing: constraints.txt" in drift["changes"]
        assert _run(root, "stop", "--yes")[0] == 0
        base_python = object_path(store / "bases", "sha256:" + base_id) / "python"
        bytecode = base_python / "__pycache__/runtime.cpython-312.pyc"
        bytecode.parent.mkdir(exist_ok=True)
        bytecode.write_bytes(b"disposable bytecode outside the sealed inventory")
        source_module = next(base_python.rglob("colorsys.py"))
        stale_cache = Path(importlib.util.cache_from_source(str(source_module)))
        stale_cache.parent.mkdir(exist_ok=True)
        module_stat = source_module.stat()
        stale_cache.write_bytes(
            importlib.util.MAGIC_NUMBER
            + struct.pack("<III", 0, int(module_stat.st_mtime), module_stat.st_size)
            + marshal.dumps(
                compile(
                    "raise RuntimeError('unverified bytecode executed')", str(source_module), "exec"
                )
            )
        )
        poisoned_import = subprocess.run(
            [
                str(base_python / ("python.exe" if os.name == "nt" else "bin/python3")),
                "-I",
                "-B",
                "-c",
                "import colorsys",
            ],
            capture_output=True,
            text=True,
        )
        assert poisoned_import.returncode != 0
        assert "unverified bytecode executed" in poisoned_import.stderr
        if os.name != "nt":
            outside_cache = tmp_path / "outside-cache.pyc"
            outside_cache.write_bytes(b"must not be traversed")
            (stale_cache.parent / "external.pyc").symlink_to(outside_cache)
            (base_python / "cache-alias").symlink_to(
                os.path.relpath(stale_cache.parent, base_python), target_is_directory=True
            )

        def snapshot() -> dict[Path, str | None]:
            return {
                path.relative_to(tmp_path): content_digest(path) if path.is_file() else None
                for directory in (root, store)
                for path in directory.rglob("*")
            }

        before = snapshot()
        for flags in ((), ("--plan",)):
            code, repair_plan = _run(root, "doctor", "--repair", *flags)
            assert code == 0 and repair_plan["status"] == "planned", repair_plan
            assert repair_plan["current"] == 3
            assert snapshot() == before
        if os.name != "nt":
            outside_cache.chmod(0)
        assert _run(root, "start", "--yes")[0] == 0
        cached_wheel = next((store / "wheels").glob("*/fixture_engine-1.0-*.whl"))
        verified = cached_wheel.read_bytes()
        cached_wheel.write_bytes(b"corrupt local wheel")
        code, repair_failed = _run(root, "doctor", "--repair", "--yes")
        assert code == 1 and "offline generation wheel missing or corrupt" in repair_failed["error"]
        assert (root / "current").read_text() == "3\n" and not (root / "builds/4").exists()
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        cached_wheel.write_bytes(verified)
        code, repaired = _run(root, "doctor", "--repair", "--yes")
        assert code == 0 and repaired["current"] == 4, repaired
        assert _run(root, "doctor")[1]["status"] == "clean"
        repaired_root = root / "builds/4"
        assert not (repaired_root / "python" / stale_cache.relative_to(base_python)).exists()
        assert not list((repaired_root / "python").rglob("external.pyc"))
        assert not (repaired_root / "python/cache-alias").exists()
        repaired_python = (
            repaired_root / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        )
        assert (
            subprocess.check_output(
                [
                    str(repaired_python),
                    "-I",
                    "-B",
                    "-c",
                    "import colorsys;print(colorsys.rgb_to_hsv(1,0,0))",
                ]
            ).strip()
            == b"(0.0, 1.0, 1)"
        )
        if os.name != "nt":
            outside_cache.chmod(0o600)
            (stale_cache.parent / "external.pyc").unlink()
            (base_python / "cache-alias").unlink()
        restored = next((root / "builds/4").rglob("fixture_engine.py"))
        assert restored.read_text() == "VALUE = 41\n"
        assert damaged.read_text() == "VALUE = 999\n"
        assert not (current / "constraints.txt").exists()
        assert Journal.read(sorted((root / "journal").glob("*.json"))[-1]).operation == "repair"
        base_module = next(base_python.rglob("antigravity.py"))
        original = base_module.read_bytes()
        base_module.write_bytes(b"corrupt retained base")
        journals = sorted((root / "journal").glob("*.json"))
        try:
            code, corrupt_base = _run(root, "doctor", "--repair", "--yes")
            assert code == 1 and "intact local base input" in corrupt_base["error"]
            assert sorted((root / "journal").glob("*.json")) == journals
            assert (root / "current").read_text() == "4\n"
        finally:
            base_module.write_bytes(original)
        if os.name != "nt":
            original_directory = base_module.parent
            parked = original_directory.with_name("parked-stdlib")
            original_directory.rename(parked)
            external = tmp_path / "same-shaped-stdlib"
            shutil.copytree(parked, external)
            (external / base_module.name).write_bytes(b"same names, unverified base code")
            original_directory.symlink_to(external, target_is_directory=True)
            try:
                code, linked_base = _run(root, "doctor", "--repair", "--yes")
                assert code == 1 and "linked local base inputs" in linked_base["error"]
                assert sorted((root / "journal").glob("*.json")) == journals
                assert (root / "current").read_text() == "4\n"
            finally:
                original_directory.unlink()
                parked.rename(original_directory)
        print(f"fixture update including sync/health took {elapsed:.3f}s")
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


@pytest.mark.parametrize("frontend_build", ["ci", "local"])
def test_development_setup_shares_the_generations_selected_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_server, frontend_build: str, capsys
) -> None:
    import pygit2
    from dinkster_supervisor import environment_frontend_sources
    from dinkster_supervisor.environment_development import prepare_development_packs
    from dinkster_supervisor.environment_state import EnvironmentError, interpreter
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from test_environment_frontend_sources import _artifact
    from test_environment_generations import _wheel
    from test_installer import write_publishable_pack

    from dinkster.cli import main

    source = Path(__file__).resolve().parents[1]
    root, store = tmp_path / "project", tmp_path / "store"
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    library, _ = _wheel(tmp_path, "selected_helper", "1.0", 73)
    pack = write_publishable_pack(tmp_path / "pack")
    (pack / "src/demo/__init__.py").write_text(
        "from dinkster_api.v1 import CORE_INT, Node, NodeSchema, OutputSpec, TypeExpr\n"
        "from selected_helper import VALUE\n"
        "class Helper(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='demo.helper', display_name='Helper', "
        "outputs=(OutputSpec('value', TypeExpr.concrete(CORE_INT)),))\n"
        "    @classmethod\n"
        "    def execute(cls):\n"
        "        return cls.outputs(value=VALUE)\n"
        "NODES = [Helper]\n"
    )
    manifest = pack / "dinkster-pack.toml"
    manifest.write_text(
        manifest.read_text().replace(
            "[pack.entry]", 'requires=["selected-helper==1.0"]\n[pack.entry]'
        )
    )
    repository = pygit2.Repository(git_server["root"] / "source.git")
    index = pygit2.Index()
    for name, data in {
        "package.json": '{"version":"0.2.0"}',
        "frontend-release.json": '{"engineApi":{"min":1,"max":9999}}',
    }.items():
        index.add(pygit2.IndexEntry(name, repository.create_blob(data), pygit2.enums.FileMode.BLOB))
    signature = pygit2.Signature("Fixture", "fixture@example.invalid")
    commit = repository.create_commit(
        f"refs/heads/development-build-{frontend_build}",
        signature,
        signature,
        "frontend",
        index.write_tree(repository),
        [],
    )
    repository.free()
    artifact = _artifact(tmp_path, str(commit))
    monkeypatch.setattr(
        environment_frontend_sources,
        "frontend_ci_artifact",
        lambda _: artifact if frontend_build == "ci" else None,
    )

    def fetch(roots, name, destination, sha, size):
        destination.write_bytes(artifact["path"].read_bytes())

    monkeypatch.setattr(environment_frontend_sources, "fetch_artifact", fetch)
    commands = []

    def build(command, *, cwd):
        commands.append(command)
        if command[-1] == "build":
            dist = cwd / "packages/app/dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("<main>local frontend</main>")

    monkeypatch.setattr(environment_frontend_sources, "run_local_command", build)
    which = shutil.which
    monkeypatch.setattr(
        environment_frontend_sources.shutil,
        "which",
        lambda name: None if name in {"node", "pnpm"} else which(name),
    )
    variant = {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu")
    frontend_source = f"{git_server['url']}#ref=development-build-{frontend_build}"
    arguments = [
        "develop",
        "--source",
        str(source),
        "--contributor",
        "--pack-source",
        str(pack),
        "--library-source",
        str(library),
        "--frontend-source",
        frontend_source,
        "--variant",
        {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu"),
        "--python",
        sys.executable,
        "--data",
        str(store),
        "--port",
        str(free_port()),
    ]

    def run(*extra):
        code = main([*arguments, "--root", str(root), *extra, "--json"])
        return code, json.loads(capsys.readouterr().out)

    code, preview = run("--plan")
    assert code == 0, preview
    if frontend_build == "local":
        assert any("requires Node and pnpm" in line for line in preview["plan"])
        code, refused = run("--approve", preview["sourceApproval"], "--yes")
        assert code == 1 and "local build requires node and pnpm" in refused["error"], refused
        assert not (root / "current").exists() and not commands
        monkeypatch.setattr(
            environment_frontend_sources.shutil,
            "which",
            lambda name: "/tools/" + name if name in {"node", "pnpm"} else which(name),
        )
    else:
        original = artifact["path"].read_bytes()
        artifact["path"].write_bytes(b"not a zip archive")
        code, refused = run("--approve", preview["sourceApproval"], "--yes")
        assert code == 1 and "invalid Git frontend build metadata" in refused["error"], refused
        assert not (root / "current").exists() and not commands
        artifact["path"].write_bytes(original)
    try:
        code, result = run("--approve", preview["sourceApproval"], "--yes")
        assert code == 0, result
        generation = root / "builds" / str(result["current"])
        html = subprocess.check_output(
            [
                str(interpreter(generation)),
                "-I",
                "-c",
                "import importlib.resources as r; "
                "print(r.files('dinkster_frontend').joinpath('static/index.html').read_text())",
            ],
            text=True,
        ).strip()
        assert html == (
            "<main>selected frontend</main>"
            if frontend_build == "ci"
            else "<main>local frontend</main>"
        )
    finally:
        if (root / "current").exists():
            assert _run(root, "stop", "--yes")[0] == 0
    generation = root / "builds" / (root / "current").read_text().strip()
    selection = {
        "variant": variant,
        "workerProtocol": PROTOCOL_VERSION,
        "development": json.loads((generation / "development-sources.json").read_text()),
    }
    release = json.loads((generation / "build.json").read_text())["engine"]["release"]
    for wheel_version in ("1.0", "2.0"):
        other = tmp_path / wheel_version
        other.mkdir()
        different, _ = _wheel(other, "selected_helper", wheel_version, 149)
        with pytest.raises(EnvironmentError, match="differs from the generation"):
            prepare_development_packs(
                generation,
                packs=(pack,),
                libraries=(different,),
                store=store,
                release=release,
                selection=selection,
            )
    assert [
        entry.pack
        for entry in Lockfile.from_record_json((generation / "dinkster.lock").read_text()).packs
    ] == ["demo"]
    assert (
        subprocess.check_output(
            [
                str(interpreter(generation)),
                "-I",
                "-c",
                "import selected_helper; print(selected_helper.VALUE)",
            ],
            text=True,
        ).strip()
        == "73"
    )


@pytest.mark.parametrize("public_command", [False, True])
@pytest.mark.parametrize("problem", [None, "missing", "engine", "pack", "library"])
def test_development_plan_fingerprints_sources_and_refuses_changed_approval(
    tmp_path: Path,
    problem: str | None,
    public_command: bool,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    source = Path(__file__).resolve().parents[1]
    engine, pack, library = (tmp_path / name for name in ("engine", "pack", "library"))
    for relative in (
        "uv.lock",
        "pyproject.toml",
        "packages/dinkster-workers/src/dinkster_workers/boundary.py",
    ):
        target = engine / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text('[pack]\nname="unpublished-pack"\n')
    library.mkdir()
    (library / "pyproject.toml").write_text('[project]\nname="unpublished-helper"\n')
    root, store = tmp_path / "prospective", tmp_path / "store"
    variant = {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu")
    arguments = (
        "develop",
        "--source",
        str(engine),
        "--pack-source",
        str(pack if problem != "missing" else tmp_path / "absent"),
        "--library-source",
        str(library),
        "--variant",
        variant,
        "--python",
        sys.executable,
        "--data",
        str(store),
    )

    def invoke(*extra):
        if not public_command:
            return _run(root, *arguments, *extra)
        from dinkster.cli import main

        code = main(["develop", "--root", str(root), *arguments[1:], *extra, "--json"])
        return code, json.loads(capsys.readouterr().out)

    code, plan = invoke("--plan")
    if problem == "missing":
        assert code == 1 and "source folder is missing" in plan["error"]
    else:
        assert code == 0, plan
        assert any("not reviewed by the registry" in line for line in plan["plan"])
        assert any("dependency ranges resolved at build time" in line for line in plan["plan"])
        assert [row["kind"] for row in plan["development"]["sources"]] == [
            "engine",
            "pack",
            "library",
        ]
        if problem is not None:
            changed = {"engine": engine, "pack": pack, "library": library}[problem]
            (changed / "untracked.py").write_text("VALUE = 23\n")
            code, refused = invoke("--approve", plan["sourceApproval"], "--plan")
            assert code == 1 and "changed from the approved plan" in refused["error"]
    assert not root.exists()
    assert not store.exists()


@pytest.mark.parametrize(
    ("url", "allow_remote", "accepted"),
    [
        ("http://localhost:5173/", False, True),
        ("http://127.0.0.2:5173", False, True),
        ("http://[::1]:5173", False, True),
        ("https://192.0.2.1:5173", False, False),
        ("https://192.0.2.1:5173", True, True),
        ("https://localhost.example:5173", False, False),
        ("http://user:password@localhost:5173", True, False),
        ("file:///tmp/frontend", True, False),
        ("http://localhost:5173/path", True, False),
        ("http://localhost:5173?token=value", True, False),
        ("http://localhost:5173#fragment", True, False),
        ("http://localhost:70000", True, False),
    ],
)
def test_development_frontend_plan_reviews_origin_and_remote_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str, allow_remote: bool, accepted: bool
) -> None:
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    root, store = tmp_path / "project", tmp_path / "store"
    arguments = [
        "develop",
        "--source",
        str(Path(__file__).resolve().parents[1]),
        "--contributor",
        "--variant",
        "win-cpu" if sys.platform == "win32" else "linux-cpu",
        "--data",
        str(store),
        "--frontend-dev",
        url,
        "--plan",
    ]
    if allow_remote:
        arguments.append("--allow-frontend-dev-remote")
    code, plan = _run(root, *arguments)
    assert code == (0 if accepted else 1), plan
    if accepted:
        assert plan["development"]["frontendDev"] == url.rstrip("/")
        assert plan["development"]["allowFrontendDevRemote"] is allow_remote
        assert any(url.rstrip("/") in line for line in plan["plan"])
        changed = arguments.copy()
        changed[changed.index(url)] = "http://127.0.0.1:5174"
        code, refusal = _run(root, *changed, "--approve", plan["sourceApproval"])
        assert code == 1 and "changed from the approved plan" in refusal["error"]
        if allow_remote:
            changed = [arg for arg in arguments if arg != "--allow-frontend-dev-remote"]
            changed[changed.index(url)] = "http://127.0.0.1:5173"
            code, local_plan = _run(root, *changed)
            assert code == 0, local_plan
            code, refusal = _run(
                root,
                *changed,
                "--allow-frontend-dev-remote",
                "--approve",
                local_plan["sourceApproval"],
            )
            assert code == 1 and "changed from the approved plan" in refusal["error"]
    else:
        assert "frontend dev server" in plan["error"]
    assert not root.exists()
    assert not store.exists()


@pytest.mark.parametrize("git_kind", ["engine", "pack", "library", "inference", "frontend"])
def test_development_git_plan_mixes_sources_and_keeps_approval_across_fetches(
    tmp_path: Path, git_server, git_kind: str, capsys, monkeypatch
):
    import pygit2

    from dinkster.cli import main

    source = Path(__file__).resolve().parents[1]
    engine, pack, library = (tmp_path / name for name in ("engine", "pack", "library"))
    for relative in (
        "uv.lock",
        "pyproject.toml",
        "packages/dinkster-workers/src/dinkster_workers/boundary.py",
    ):
        target = engine / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text('[pack]\nname="unpublished-pack"\n')
    library.mkdir()
    (library / "pyproject.toml").write_text(
        '[project]\nname="dinkster-inference"\nversion="0.1.3.dev7"\n'
        if git_kind == "inference"
        else '[project]\nname="unpublished-helper"\n'
    )
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text('{"version":"0.2.0"}')
    (frontend / "frontend-release.json").write_text('{"engineApi":{"min":1,"max":2}}')
    schema = engine / "packages/dinkster-schema/src/dinkster_schema/wire.py"
    schema.parent.mkdir(parents=True)
    schema.write_text("SCHEMA_WIRE_VERSION = 1\n")
    repository = pygit2.Repository(git_server["root"] / "source.git")
    index = pygit2.Index()
    for kind, folder in (
        ("engine", engine),
        ("pack", pack),
        ("library", library),
        ("frontend", frontend),
    ):
        for path in folder.rglob("*"):
            if path.is_file():
                index.add(
                    pygit2.IndexEntry(
                        f"{kind}/{path.relative_to(folder).as_posix()}",
                        repository.create_blob(path.read_bytes()),
                        pygit2.enums.FileMode.BLOB,
                    )
                )
    signature = pygit2.Signature("Fixture", "fixture@example.invalid")
    repository.create_commit(
        f"refs/heads/development-{git_kind}",
        signature,
        signature,
        "development sources",
        index.write_tree(repository),
        [],
    )
    repository.free()
    monkeypatch.setenv("PATH", "")
    arguments = ["develop", "--root", str(tmp_path / "project")]
    selected_kind = "library" if git_kind == "inference" else git_kind
    for kind, folder, option in (
        ("engine", engine, "--source"),
        ("pack", pack, "--pack-source"),
        ("library", library, "--inference" if git_kind == "inference" else "--library-source"),
    ):
        arguments.extend(
            [
                option,
                f"{git_server['url']}#ref=development-{git_kind}&subdirectory={kind}"
                if kind == selected_kind
                else str(folder),
            ]
        )
    if git_kind == "frontend":
        arguments.extend(
            [
                "--frontend-source",
                f"{git_server['url']}#ref=development-frontend&subdirectory=frontend",
            ]
        )
    arguments.extend(
        [
            "--variant",
            {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu"),
            "--python",
            sys.executable,
            "--plan",
            "--json",
        ]
    )
    assert main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert main([*arguments, "--approve", plan["sourceApproval"]]) == 0
    repeated = json.loads(capsys.readouterr().out)
    selected = next(
        row for row in repeated["development"]["sources"] if row["kind"] == selected_kind
    )
    assert selected["git"]["requestedRef"] == f"development-{git_kind}"
    assert selected["git"]["contentDigest"] == selected["contentDigest"]
    changed = [
        value.replace("ref=development", "ref=refs/heads/development") for value in arguments
    ]
    assert main([*changed, "--approve", plan["sourceApproval"]]) == 1
    assert "changed from the approved plan" in json.loads(capsys.readouterr().out)["error"]
    if git_kind == "frontend":
        assert selected["live"] is False
        assert selected["frontendArtifact"] is None
        artifact = {
            "id": 17,
            "run": 23,
            "name": "dinkster-frontend-" + "a" * 64,
            "digest": "sha256:" + "b" * 64,
            "size_in_bytes": 103,
        }
        monkeypatch.setattr(
            "dinkster_supervisor.environment_frontend_sources.frontend_ci_artifact",
            lambda origin: artifact,
        )
        assert main([*arguments, "--approve", plan["sourceApproval"]]) == 1
        assert "changed from the approved plan" in json.loads(capsys.readouterr().out)["error"]
        assert main(arguments) == 0
        ci_plan = json.loads(capsys.readouterr().out)
        artifact["id"] = 19
        assert main([*arguments, "--approve", ci_plan["sourceApproval"]]) == 1
        assert "changed from the approved plan" in json.loads(capsys.readouterr().out)["error"]
        schema.write_text("SCHEMA_WIRE_VERSION = 3\n")
        assert main(arguments) == 1
        assert "schema API" in json.loads(capsys.readouterr().out)["error"]
        schema.write_text("SCHEMA_WIRE_VERSION = 1\n")
    if git_kind == "engine":
        (frontend / "index.html").write_text("<title>Selected frontend</title>")
        previous_approval = plan["sourceApproval"]
        for extra, field, expected in (
            ([], None, None),
            (["--frontend-root", str(frontend)], "frontendRoot", str(frontend)),
            (["--frontend-dev", "http://localhost:5173"], "frontendDev", "http://localhost:5173"),
            (
                ["--frontend-dev", "https://192.0.2.1:5173", "--allow-frontend-dev-remote"],
                "frontendDev",
                "https://192.0.2.1:5173",
            ),
        ):
            engine_only = arguments.copy()
            for option in ("--pack-source", "--library-source"):
                option_index = engine_only.index(option)
                del engine_only[option_index : option_index + 2]
            selected_arguments = [*engine_only, *extra]
            assert main(selected_arguments) == 0
            selected_plan = json.loads(capsys.readouterr().out)
            assert [row["kind"] for row in selected_plan["development"]["sources"]] == ["engine"]
            assert all(not row["live"] for row in selected_plan["development"]["sources"])
            assert selected_plan["development"]["contributor"] is False
            if field:
                assert selected_plan["development"][field] == expected
            assert main([*selected_arguments, "--approve", selected_plan["sourceApproval"]]) == 0
            json.loads(capsys.readouterr().out)
            assert main([*selected_arguments, "--approve", previous_approval]) == 1
            assert "changed from the approved plan" in json.loads(capsys.readouterr().out)["error"]
            previous_approval = selected_plan["sourceApproval"]
            if "--allow-frontend-dev-remote" in extra:
                assert selected_plan["development"]["allowFrontendDevRemote"] is True
                assert (
                    main(
                        [
                            value
                            for value in selected_arguments
                            if value != "--allow-frontend-dev-remote"
                        ]
                    )
                    == 1
                )
                assert (
                    "requires --allow-frontend-dev-remote"
                    in json.loads(capsys.readouterr().out)["error"]
                )
    live_code = main([*arguments, "--live-engine"])
    live_plan = json.loads(capsys.readouterr().out)
    if git_kind == "engine":
        assert live_code == 1 and "requires a local folder" in live_plan["error"]
    else:
        assert live_code == 0
        assert live_plan["development"]["sources"][0]["live"] is True
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize("operation", ["develop", "setup", "update"])
def test_inference_requests_report_oldest_containing_release(
    tmp_path: Path, git_server, capsys, monkeypatch, operation: str
):
    """A missing prerelease cannot silently select an unrelated or newer stable release."""
    import pygit2
    from dinkster_supervisor import environment_inference

    from dinkster.cli import main

    source = Path(__file__).resolve().parents[1]
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text('[pack]\nname="example"\n')
    index = tmp_path / "index"
    index.mkdir()
    page = index / "dinkster-inference"
    versions = ["0.1.0", "0.2.0", "0.3.0", "0.3.0.dev4"]
    page.write_text(
        "".join(f'<a href="dinkster_inference-{v}-py3-none-any.whl">{v}</a>' for v in versions)
    )
    repository = pygit2.init_repository(
        git_server["root"] / f"inference-{operation}.git", bare=True
    )
    signature = pygit2.Signature("Fixture", "fixture@example.invalid")
    parent = []
    commits = {}
    for version in ("0.1.0", "0.2.0.dev3", "0.2.0", "0.3.0", "0.3.0.dev4"):
        tree = repository.TreeBuilder()
        tree.insert("version", repository.create_blob(version.encode()), pygit2.GIT_FILEMODE_BLOB)
        commit = repository.create_commit(
            "refs/heads/inference-requests", signature, signature, version, tree.write(), parent
        )
        repository.create_tag(
            "v" + version, commit, pygit2.enums.ObjectType.COMMIT, signature, version
        )
        commits[version] = str(commit)
        parent = [commit]
    repository.set_head("refs/heads/inference-requests")
    repository.free()
    monkeypatch.setattr(
        environment_inference,
        "INFERENCE_REPOSITORY",
        git_server["url"].replace("source.git", f"inference-{operation}.git"),
    )
    monkeypatch.setenv("PATH", "")
    arguments = [
        "develop",
        "--root",
        str(tmp_path / "project"),
        "--source",
        str(source),
        "--pack-source",
        str(pack),
        "--variant",
        {"win32": "win-cpu", "darwin": "mac-arm64"}.get(sys.platform, "linux-cpu"),
        "--python",
        sys.executable,
        "--package-index",
        str(index),
        "--plan",
        "--json",
    ]
    root = tmp_path / "project"
    setup_engine_packages: list[dict[str, str]] = []
    if operation != "develop":
        mirror = tmp_path / "mirror"
        _, engine, channel = _mirror(mirror)
        if operation == "setup":
            engine["defaultPacks"] = {"fixture-pack": "1.0"}
            data = _json(engine)
            identity = hashlib.sha256(data).hexdigest()
            previous = mirror / channel["engine"]["locations"][0]["path"]
            release = mirror / "engine" / identity
            shutil.copytree(previous.parent, release)
            (release / "release-manifest.json").write_bytes(data)
            channel["engine"].update(
                release="sha256:" + identity,
                locations=[{"type": "mirror", "path": f"engine/{identity}/release-manifest.json"}],
            )
            (mirror / "channels/nightly.json").write_bytes(_channel_json(channel))
            monkeypatch.setattr(
                "dinkster.registries.plan_registry_sources",
                lambda *args, **kwargs: (
                    setup_engine_packages.append(kwargs["engine_packages"]) or Lockfile.of(()),
                    (),
                ),
            )
        root = tmp_path / "project"
        arguments = [
            operation,
            "--root",
            str(root),
            "--mirror",
            str(mirror),
            "--channel",
            "nightly",
            "--variant",
            "fixture-cpu",
            "--package-index",
            str(index),
            "--plan",
            "--json",
        ]
        if operation == "update":
            _generation(root, 1)
            (root / "current").write_text("1\n")
            (root / "environment.json").write_text(
                json.dumps(
                    {
                        "format": "dinkster.project/1",
                        "channel": "nightly",
                        "variant": "fixture-cpu",
                        "mirrors": [str(mirror)],
                        "packageIndex": str(index),
                        "port": 58991,
                    }
                )
            )
    for request in ("0.2.0.dev3", commits["0.2.0.dev3"]):
        assert main([*arguments, "--inference", request]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["inference"]["version"] == "0.2.0"
        assert plan["inference"]["requestedCommit"] == commits["0.2.0.dev3"]
        assert plan["inference"]["sourceCommit"] == commits["0.2.0"]
        assert plan["inference"]["fallback"] is True
        assert any("oldest containing tagged release 0.2.0" in line for line in plan["plan"])
        if operation == "setup":
            assert setup_engine_packages[-1]["dinkster-inference"] == "0.2.0"
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    original_page = page.read_text()
    changed_page = original_page.replace(
        '<a href="dinkster_inference-0.2.0-py3-none-any.whl">0.2.0</a>', ""
    )
    apply_arguments = [argument for argument in arguments if argument != "--plan"]
    with monkeypatch.context() as approval_patch:
        if operation == "setup":
            resolve = environment_inference.resolve_inference

            def change_after_planning(*args, **kwargs):
                selected = resolve(*args, **kwargs)
                page.write_text(changed_page)
                return selected

            approval_patch.setattr(
                environment_inference, "resolve_inference", change_after_planning
            )
            approval_arguments = []
        else:
            page.write_text(changed_page)
            approval_arguments = (
                ["--approve", plan["sourceApproval"]]
                if operation == "develop"
                else ["--inference-approval", plan["inferenceApproval"]]
            )
        assert (
            main(
                [
                    *apply_arguments,
                    "--inference",
                    commits["0.2.0.dev3"],
                    "--yes",
                    *approval_arguments,
                ]
            )
            == 1
        )
        refused = capsys.readouterr()
        error = refused.err if operation == "setup" else json.loads(refused.out)["error"]
        assert "changed" in error and "request a new plan" in error
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    assert root.exists() == (operation == "update")
    page.write_text(original_page)
    assert main([*arguments, "--inference", "0.3.0.dev4"]) == 0
    assert json.loads(capsys.readouterr().out)["inference"]["version"] == "0.3.0.dev4"
    page.write_text(page.read_text().replace("0.3.0.dev4", "0.9.0.dev1"))
    assert main([*arguments, "--inference", commits["0.3.0.dev4"]]) == 1
    refused = capsys.readouterr()
    error = refused.err if operation == "setup" else json.loads(refused.out)["error"]
    assert "no published tagged release contains" in error
    if operation == "update":
        assert (root / "current").read_text() == "1\n"
    else:
        assert not (tmp_path / "project").exists()


@pytest.mark.parametrize("live", [False, True])
def test_development_doctor_separates_runtime_drift_from_missing_recovery(
    tmp_path: Path, live: bool
):
    live_pack = tmp_path / "live-pack"
    live_pack.mkdir()
    sources = [{"kind": "pack", "path": str(live_pack), "live": True}] if live else []
    generation = _generation(tmp_path, 1).parent
    store = tmp_path / "capture"
    base = {
        "source": "local",
        "variant": "linux-cpu",
        "workerProtocol": 1,
        "wheels": [],
        "interpreter": {
            "files": {"python/runtime": "sha256:" + hashlib.sha256(b"runtime").hexdigest()}
        },
    }
    base_id = "sha256:" + hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    base["id"] = base_id
    runtime = object_path(store / "bases", base_id) / "python/runtime"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    engine = {"source": "local", "version": "0.3.1", "workerProtocol": 1, "wheels": []}
    engine_id = "sha256:" + hashlib.sha256(json.dumps(engine, sort_keys=True).encode()).hexdigest()
    seal = generation / "build.json"
    record = json.loads(seal.read_text())
    record.update(source="local", base=base_id, engine={"version": "0.3.1", "release": engine_id})
    metadata = {"constraints.txt": (generation / "constraints.txt").read_text()}
    for name, document in (
        ("base-manifest.json", base),
        ("engine-manifest.json", engine),
        (
            "development-sources.json",
            {"format": "dinkster.development-sources/1", "sources": sources},
        ),
    ):
        path = generation / name
        path.write_text(json.dumps(document))
        record["files"][name] = content_digest(path)
        metadata[name] = path.read_text()
    seal.write_text(json.dumps(record))
    (tmp_path / "current").write_text("1\n")
    (tmp_path / "store").write_text(str(store))
    (tmp_path / "history").mkdir()
    (tmp_path / "history/00000000000000000001.json").write_text(
        json.dumps(
            {
                "format": "dinkster.install-history/1",
                "id": 1,
                "generation": 1,
                "snapshot": {
                    "format": "dinkster.install-snapshot/1",
                    "source": "local",
                    "metadata": metadata,
                    "base": {"id": base_id, "variant": "linux-cpu"},
                    "engine": {"release": engine_id, "version": "0.3.1"},
                    "workerProtocol": 1,
                    "inProcessPacks": [],
                },
            }
        )
    )
    for operation in ("status", "doctor", "history"):
        code, result = _run(tmp_path, operation)
        assert code == 0, result
        described = result["history"][0] if operation == "history" else result
        assert ("cannot be restored exactly" if live else "not reviewed by the registry") in (
            described["sourceDescription"]
        )
        if operation == "doctor":
            assert result["exactRecovery"] == {"available": not live, "missing": []}
    if live:
        author_file = live_pack / "node.py"
        author_file.write_text("VALUE = 37\n")
        (tmp_path / "environment.json").write_text(
            json.dumps(
                {
                    "format": "dinkster.project/1",
                    "channel": "stable",
                    "variant": "linux-cpu",
                    "mirrors": [],
                    "port": 58980,
                }
            )
        )
        code, refused = _run(tmp_path, "restore", "1", "--plan")
        assert code == 1 and "cannot be restored exactly" in refused["error"], refused
        assert author_file.read_text() == "VALUE = 37\n"
        assert (tmp_path / "current").read_text() == "1\n"
    runtime.unlink()
    code, missing = _run(tmp_path, "doctor")
    assert code == 0 and missing["status"] == "clean", missing
    assert missing["exactRecovery"]["available"] is False
    assert missing["exactRecovery"]["missing"][0]["kind"] == "local base"
    (generation / "python/module.py").write_text("ANSWER = 99\n")
    code, damaged = _run(tmp_path, "doctor")
    assert code == 1 and damaged["status"] == "drift", damaged
    assert ("cannot be restored exactly" if live else "not reviewed by the registry") in (
        damaged["sourceDescription"]
    )
