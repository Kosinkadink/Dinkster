"""Desktop and humans share the actual subprocess result contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import os
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import ProxyHandler, build_opener, urlopen

import pytest
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_registry import Lockfile
from dinkster_supervisor.environment_artifacts import acquire_sources, discover
from dinkster_supervisor.environment_state import Journal, content_digest, writer_lock
from supervisor_subprocess import run as run_with_diagnostics
from test_environment_artifacts import _artifact, _json, _mirror, _release_sources, _wheel_payload
from test_environment_generations import _assert_shared, _workspace_tree
from test_environment_runtime import free_port, make_generation
from test_environment_state import _generation

from tests.fixtures.environments.build import executable_release as _executable_release


def _run(root: Path, *arguments: str) -> tuple[int, dict]:
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
    result = (
        run_with_diagnostics(command)
        if "setup" in arguments
        or "move" in arguments
        or ("doctor" in arguments and "--repair" in arguments)
        else subprocess.run(command, capture_output=True, text=True, timeout=30)
    )
    assert result.stdout, result.stderr
    document = json.loads(result.stdout)
    assert document["format"] == "dinkster.environment-result/1"
    assert document["root"] == str(root)
    return result.returncode, document


def test_catalog_uses_common_envelope_and_leaves_prospective_root_absent(tmp_path: Path):
    mirror = tmp_path / "mirror"
    base, _, channel = _mirror(mirror)
    root = tmp_path / "prospective project"
    code, result = _run(root, "catalog", "--channel", "main", "--mirror", str(mirror))
    assert code == 0
    assert result["operation"] == "catalog"
    assert result["plan"] == []
    assert result["status"] == "clean"
    assert result["channel"] == "main"
    assert result["channelManifest"] == channel
    assert result["current"] is None
    assert result["variants"][0]["manifest"] == base
    assert not root.exists()


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
        "main",
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
        "main",
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
    (mirror / "channels/main.json").write_bytes(_json(channel))
    if operation == "update":
        _generation(root, 1)
        (root / "current").write_bytes(b"1\n")
        (root / "project.json").write_bytes(
            _json(
                {
                    "format": "dinkster.project/1",
                    "channel": "main",
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
        "main",
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


def test_status_returns_configured_channel_and_variant_without_starting_anything(tmp_path: Path):
    root = tmp_path / "project"
    code, result = _run(root, "status")
    assert code == 0
    assert result["engines"] == []
    assert result["current"] is result["channel"] is result["variant"] is None
    assert not root.exists()
    root.mkdir()
    settings = {
        "format": "dinkster.project/1",
        "channel": "main",
        "variant": "win-cpu",
        "port": 3639,
        "mirrors": [str(tmp_path / "mirror")],
    }
    (root / "project.json").write_text(json.dumps(settings))
    before = (root / "project.json").read_bytes()
    code, result = _run(root, "status")
    assert code == 0
    assert result["channel"] == "main"
    assert result["variant"] == "win-cpu"
    assert result["engines"] == []
    assert (root / "project.json").read_bytes() == before
    assert not (root / "runtime").exists()


def test_malformed_existing_project_returns_error_not_empty_status(tmp_path: Path):
    (tmp_path / "project.json").write_text('{"format":"dinkster.project/1","channel":null}')
    code, result = _run(tmp_path, "status")
    assert code == 1
    assert result["status"] == "error"
    assert "project channel" in result["error"]


@pytest.mark.parametrize("source", ["published", "local"])
def test_doctor_reports_actual_content_drift_with_nonzero_exit_and_no_repair(
    tmp_path: Path, source: str
):
    generation = _generation(tmp_path, 1).parent
    metadata = generation / "generation.json"
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
    (root / "project.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "main",
                "variant": "linux-cpu",
                "port": port,
                "mirrors": [str(root / "mirror")],
            }
        )
    )
    return port


def test_channel_update_refuses_local_generation_before_reading_channel(tmp_path: Path):
    _runtime_project(tmp_path)
    settings = tmp_path / "project.json"
    project = json.loads(settings.read_text())
    project["mirrors"] = []
    settings.write_text(json.dumps(project))
    metadata = tmp_path / "generations/1/generation.json"
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
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    _mirror(mirror, variant="linux-cpu")
    _runtime_project(root)
    shutil.rmtree(root / "generations/1/python")
    _workspace_tree(root / "generations/1/python")
    (root / "store").write_text(str(store) + "\n")
    settings = json.loads((root / "project.json").read_bytes())
    asset = AssetRef(digest_bytes(b"required bytes"), "required.bin", 14).to_wire()
    settings.update(mirrors=[str(mirror)], additions={"assets": [asset]})
    (root / "project.json").write_text(json.dumps(settings))
    (root / "generations/1/dinkster.lock").write_text(Lockfile().record_json())

    def snapshot() -> dict[Path, bytes | None]:
        return {
            path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
            for path in tmp_path.rglob("*")
            if path.name not in {"library.sqlite-wal", "library.sqlite-shm"}
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
    vault = AssetVault(root / "library/vault")
    with vault.writer(identity) as writer:
        writer.write(data)
        writer.commit()
    library = LibraryStore(root / "library/library.sqlite")
    try:
        record = library.create(
            "local", "Workflow", identity, "application/x-dinkster-workflow+json"
        )
    finally:
        library.close()
    settings["workflows"] = [{"scope": "local", "id": record.id}]
    (root / "project.json").write_text(json.dumps(settings))
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
    (root / "project.json").write_text(json.dumps(settings))

    _release_sources(mirror)
    before = snapshot()
    code, missing = _run(root, "update", "--plan")
    assert code == 0 and any("feasibility: pending" in line for line in missing["plan"])
    assert snapshot() == before
    acquired = acquire_sources(discover((str(mirror),), "main"), store)
    assert acquired is not None
    lock, artifacts = acquired
    settings["additions"]["packs"] = [{"pack": "first-pack", "version": "9.9.9"}]
    (root / "project.json").write_text(json.dumps(settings))
    before = snapshot()
    for extra in ((), ("--catalog-lock", str(lock), "--artifacts", str(artifacts))):
        code, unavailable = _run(root, "update", "--plan", *extra)
        assert code == 0 and any("feasibility: pending" in line for line in unavailable["plan"])
        assert '"version": "9.9.9"' in "\n".join(unavailable["plan"])
        assert snapshot() == before
    cached = next(artifacts.glob("*.zip"))
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
    setup = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert setup.returncode == 0, setup.stdout + setup.stderr
    try:
        assert _run(root, "stop", "--yes")[0] == 0
        settings = json.loads((root / "project.json").read_bytes())
        settings["additions"] = {"packs": [], "assets": []}
        (root / "project.json").write_text(json.dumps(settings))
        (mirror / "channels/main.json").write_bytes(Path(fixture["updatedChannel"]).read_bytes())
        overrides = ()
        if explicit:
            catalog, artifacts = tmp_path / "selected.lock", tmp_path / "selected-artifacts"
            catalog.write_text(Lockfile().record_json())
            artifacts.mkdir()
            overrides = ("--catalog-lock", str(catalog), "--artifacts", str(artifacts))
        before_project = (root / "project.json").read_bytes()
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
            assert (root / "project.json").read_bytes() == before_project
            assert {path.name: path.read_bytes() for path in (root / "journal").iterdir()} == (
                before_journals
            )
            assert not (root / "generations/2").exists()
            assert not list(store.glob(".registry-*"))
            assert not list(store.glob(".workflow-*"))
        monkeypatch.delenv("DINKSTER_FIXTURE_WORKFLOW_PROBLEM")
        code, applied = _run(root, "update", *overrides, "--yes")
        assert code == 0 and applied["current"] == 2, applied
        assert (root / "project.json").read_bytes() == before_project
        assert (
            json.loads((root / "generations/2/generation.json").read_bytes())["engine"]["version"]
            == "2.0"
        )
        assert _run(root, "doctor")[1]["status"] == "clean"
        assert not list(store.glob(".registry-*"))
        assert not list(store.glob(".workflow-*"))
    finally:
        _run(root, "stop", "--yes")


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
    assert not (tmp_path / "runtime").exists()
    Journal("update", 1, 2, "building").write(tmp_path / "journal/interrupted.json")
    code, recovered = _run(tmp_path, "start", "--yes")
    try:
        assert code == 0
        assert recovered["engines"][0]["state"] == "ready"
        assert Journal.read(tmp_path / "journal/interrupted.json").stage == "rolled-back"
        assert (tmp_path / "current").read_bytes() == b"1\n"
    finally:
        assert _run(tmp_path, "stop", "--yes")[0] == 0


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
        "main",
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
    selected_release = discover((str(mirror),), "main")
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
    assert not (root / "project.json").exists() and not (root / "store").exists()
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
        assert not (root / "project.json").exists() and not (root / "store").exists()
        assert not (store / "projects.json").exists()
        assert not list((root / "journal").glob("*.json"))
        assert not list(store.glob(".registry-*"))
    monkeypatch.delenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
    code, installed = _run(root, *arguments, "--yes")
    assert code == 0, installed
    try:
        assert installed["current"] == 1 and installed["status"] == "applied"
        assert installed["unavailablePacks"] == {"fixture-pack": "registry offline"}
        assert json.loads((root / "generations/1/dinkster.lock").read_text())["packs"] == []
        assert not list(store.glob(".registry-*"))
        assert installed["url"] == f"http://127.0.0.1:{port}"
        assert _run(root, "doctor")[1]["status"] == "clean"
        initial_settings = (root / "project.json").read_bytes()
        assert json.loads(initial_settings)["packageIndex"] == package_index
        assert json.loads((store / "projects.json").read_bytes())["projects"] == [str(root)]
        other = tmp_path / "second project"
        code, second_plan = _run(
            other,
            "setup",
            "--mirror",
            str(mirror),
            "--channel",
            "main",
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
            "main",
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
        assert (root / "project.json").read_bytes() == initial_settings
        assert {
            path: path.read_bytes() for path in (root / "journal").glob("*.json")
        } == before_registry
        assert not (root / "generations/2").exists()
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
        interrupted = subprocess.run([sys.executable, "-c", script, str(root)], timeout=30)
        assert interrupted.returncode == 73
        journals = sorted((root / "journal").glob("*.json"))
        assert Journal.read(journals[-1]).stage == "building"
        assert (root / "current").read_text() == "1\n"
        started = time.monotonic()
        code, updated = _run(root, "update", "--channel", "stable", "--yes")
        elapsed = time.monotonic() - started
        assert code == 0, updated
        assert json.loads((root / "project.json").read_bytes())["channel"] == "stable"
        assert json.loads((root / "project.json").read_bytes())["packageIndex"] == package_index
        assert Journal.read(journals[-1]).stage == "rolled-back"
        assert updated["current"] == 2 and updated["url"] != installed["url"]
        first, second = (root / "generations" / str(n) for n in (1, 2))
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
        code, rolled_back = _run(root, "rollback", "--yes")
        assert code == 0, rolled_back
        assert (root / "project.json").read_bytes() == initial_settings
        assert rolled_back["current"] == 1 and first.exists() and second.exists()
        after = python.stat()
        assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
            retained.st_dev,
            retained.st_ino,
            retained.st_size,
            retained.st_mtime_ns,
        )
        assert content_digest(python) == retained_digest
        assert _run(root, "doctor")[1]["status"] == "clean"
        source_release, payloads = _release_sources(mirror)
        supplied = acquire_sources(discover((str(mirror),), "main"), store)
        assert supplied is not None
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
        assert (root / "current").read_text() == "1\n" and not (root / "generations/3").exists()
        cached_release = store / "engines" / source_release.name
        assert (cached_release / "official-packs.lock").read_bytes() == payloads[
            "official-packs.lock"
        ]
        assert sorted(
            path.read_bytes() for path in (cached_release / "artifacts").glob("*.zip")
        ) == sorted(payload for name, payload in payloads.items() if name.endswith(".zip"))
        assert Journal.read(sorted((root / "journal").glob("*.json"))[-1]).stage == "rolled-back"
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        _executable_release(mirror, base, "3.0", 61)
        engine = json.loads((mirror / "channels/main.json").read_bytes())["engine"]["release"][7:]
        (mirror / "engine" / engine / "fixture_engine-3.0-py3-none-any.whl").write_bytes(b"corrupt")
        before_journals = {path: path.read_bytes() for path in (root / "journal").glob("*.json")}
        before_settings = (root / "project.json").read_bytes()
        code, refused = _run(root, "update", "--yes")
        assert code == 1 and refused["status"] == "error"
        assert (root / "current").read_text() == "1\n" and not (root / "generations/3").exists()
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        assert {path: path.read_bytes() for path in (root / "journal").glob("*.json")} == (
            before_journals
        )
        assert (root / "project.json").read_bytes() == before_settings
        mirror.rename(tmp_path / "unavailable-mirror")
        damaged = next(first.rglob("fixture_engine.py"))
        damaged.write_text("VALUE = 999\n")
        (first / "constraints.txt").unlink()
        code, drift = _run(root, "doctor")
        assert code == 1 and drift["status"] == "drift"
        assert "missing: constraints.txt" in drift["changes"]
        assert _run(root, "stop", "--yes")[0] == 0
        base_python = store / "bases" / base_id / "python"
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
            assert repair_plan["current"] == 1
            assert snapshot() == before
        if os.name != "nt":
            outside_cache.chmod(0)
        assert _run(root, "start", "--yes")[0] == 0
        cached_wheel = next((store / "wheels").glob("*/fixture_engine-1.0-*.whl"))
        verified = cached_wheel.read_bytes()
        cached_wheel.write_bytes(b"corrupt local wheel")
        code, repair_failed = _run(root, "doctor", "--repair", "--yes")
        assert code == 1 and "offline generation sync failed" in repair_failed["error"]
        assert (root / "current").read_text() == "1\n" and not (root / "generations/3").exists()
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        cached_wheel.write_bytes(verified)
        code, repaired = _run(root, "doctor", "--repair", "--yes")
        assert code == 0 and repaired["current"] == 3, repaired
        assert _run(root, "doctor")[1]["status"] == "clean"
        repaired_root = root / "generations/3"
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
        restored = next((root / "generations/3").rglob("fixture_engine.py"))
        assert restored.read_text() == "VALUE = 41\n"
        assert damaged.read_text() == "VALUE = 999\n"
        assert not (first / "constraints.txt").exists()
        assert Journal.read(sorted((root / "journal").glob("*.json"))[-1]).operation == "repair"
        base_module = next(base_python.rglob("antigravity.py"))
        original = base_module.read_bytes()
        base_module.write_bytes(b"corrupt retained base")
        journals = sorted((root / "journal").glob("*.json"))
        try:
            code, corrupt_base = _run(root, "doctor", "--repair", "--yes")
            assert code == 1 and "intact local base input" in corrupt_base["error"]
            assert sorted((root / "journal").glob("*.json")) == journals
            assert (root / "current").read_text() == "3\n"
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
                assert (root / "current").read_text() == "3\n"
            finally:
                original_directory.unlink()
                parked.rename(original_directory)
        print(f"fixture update including sync/health took {elapsed:.3f}s")
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


def test_public_fixture_commands_install_update_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("urllib.request._opener", None)
    builder = Path(__file__).parent / "fixtures/environments/build.py"
    build_command = [
        sys.executable,
        str(builder),
        "--output",
        str(tmp_path / "fixture"),
    ]
    fixture = json.loads(subprocess.check_output(build_command, text=True, timeout=120))
    assert json.loads((tmp_path / "fixture/fixture.json").read_bytes()) == fixture
    refused = subprocess.run(build_command, capture_output=True, text=True, timeout=120)
    assert refused.returncode != 0
    assert json.loads((tmp_path / "fixture/fixture.json").read_bytes()) == fixture
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "UV_INDEX_URL"):
        monkeypatch.setenv(variable, "http://127.0.0.1:1/unreachable")
    root = Path(fixture["root"])
    commands = fixture["commands"]

    def run(name: str) -> dict:
        completed = subprocess.run(commands[name], capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        return json.loads(completed.stdout)

    def check(number: int, expected: dict, operation: str) -> None:
        assert (root / "current").read_text() == f"{number}\n"
        generation = root / "generations" / str(number)
        manifest = json.loads((generation / "generation.json").read_bytes())
        assert manifest["generation"] == number
        assert manifest["engine"]["version"] == expected["version"]
        python = generation / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        actual = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-I",
                    "-B",
                    "-c",
                    "import fixture_engine, importlib.metadata, json; "
                    "print(json.dumps({'value': fixture_engine.VALUE, "
                    "'version': importlib.metadata.version('fixture-engine')}))",
                ],
                text=True,
            )
        )
        assert actual == expected
        journals = [json.loads(path.read_bytes()) for path in (root / "journal").glob("*.json")]
        assert all(row["stage"] == "complete" and row.get("error") is None for row in journals)
        assert any(row["operation"] == operation and row["target"] == number for row in journals)
        status = run("status")
        assert status["current"] == number
        assert status["engines"][0]["state"] == "ready"
        assert run("doctor")["status"] == "clean"
        # The offline proxy barrier applies to downloads, not this loopback observation.
        with build_opener(ProxyHandler({})).open(
            status["engines"][0]["url"] + "/api/generation", timeout=5
        ) as response:
            assert json.loads(response.read()) == {"generation": str(generation)}

    try:
        assert run("setup")["current"] == 1
        check(1, fixture["expected"]["initial"], "setup")
        run("advance")
        assert run("update")["current"] == 2
        check(2, fixture["expected"]["updated"], "update")
        assert run("rollback")["current"] == 1
        check(1, fixture["expected"]["initial"], "rollback")
        assert (root / "generations/2/generation.json").is_file()
        assert len(list((root / "journal").glob("*.json"))) == 3
    finally:
        if (root / "current").exists():
            run("stop")
            assert all(row["state"] == "stopped" for row in run("status")["engines"])
