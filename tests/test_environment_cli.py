"""Desktop and humans share the actual subprocess result contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import py7zr
import pytest
from dinkster_supervisor.environment_state import Journal, content_digest, writer_lock
from test_environment_artifacts import _artifact, _json, _mirror
from test_environment_generations import _assert_shared, _standalone_tree, _wheel
from test_environment_runtime import SERVE, free_port, make_generation
from test_environment_state import _generation


def _run(root: Path, *arguments: str) -> tuple[int, dict]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(root),
            *arguments,
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
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


def test_doctor_reports_actual_content_drift_with_nonzero_exit_and_no_repair(tmp_path: Path):
    generation = _generation(tmp_path, 1).parent
    (tmp_path / "current").write_text("1\n")
    code, result = _run(tmp_path, "doctor")
    assert code == 0
    assert result["status"] == "clean"
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


def _executable_release(mirror: Path, base: dict, version: str, value: int) -> None:
    scratch = mirror / "build"
    scratch.mkdir(exist_ok=True)
    wheel, _ = _wheel(scratch, "fixture_engine", version, value)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("dinkster/__init__.py", "")
        archive.writestr("dinkster/serve.py", SERVE)
    payload = wheel.read_bytes()
    constraints = (
        f"fixture-base==1.0 --hash=sha256:{base['wheels'][0]['sha256']}\n"
        f"fixture-engine=={version} --hash=sha256:{hashlib.sha256(payload).hexdigest()}\n"
    ).encode()
    engine = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "tag": "main-" + "d" * 40,
        "commit": "d" * 40,
        "version": version,
        "workerProtocol": 11,
        "artifacts": [_artifact("constraints.txt", constraints), _artifact(wheel.name, payload)],
    }
    data = _json(engine) + b"\n"
    release = hashlib.sha256(data).hexdigest()
    directory = mirror / "engine" / release
    directory.mkdir(parents=True)
    (directory / "release-manifest.json").write_bytes(data)
    (directory / "constraints.txt").write_bytes(constraints)
    (directory / wheel.name).write_bytes(payload)
    channel = json.loads((mirror / "channels/main.json").read_bytes())
    channel["engine"] = {
        "version": version,
        "release": "sha256:" + release,
        "locations": [
            {"type": "mirror", "path": f"engine/{release}/release-manifest.json"},
        ],
    }
    (mirror / "channels/main.json").write_bytes(_json(channel))


def test_confirmed_offline_setup_update_rollback_and_busy_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "UV_INDEX_URL"):
        monkeypatch.setenv(variable, "http://127.0.0.1:1/unreachable")
    mirror, root, store = tmp_path / "mirror", tmp_path / "project", tmp_path / "store"
    base, _, _ = _mirror(mirror)
    tree = tmp_path / "standalone"
    _standalone_tree(tree / "python")
    (tree / "wheels").mkdir()
    wheel, _ = _wheel(tree / "wheels", "fixture_base", "1.0", 37)
    base["wheels"][0].update(_artifact(wheel.name, wheel.read_bytes(), key="filename"))
    base["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    identity = {
        key: value for key, value in base.items() if key not in {"id", "parts", "signature"}
    }
    base_id = hashlib.sha256(_json(identity)).hexdigest()
    directory = mirror / "base/fixture-cpu" / base_id
    directory.mkdir()
    part = directory / "base.7z.001"
    with py7zr.SevenZipFile(part, "w", filters=[{"id": py7zr.FILTER_COPY}]) as archive:
        archive.writeall(tree / "python", "python")
        archive.writeall(tree / "wheels", "wheels")
    base.update(
        id="sha256:" + base_id, parts=[_artifact(part.name, part.read_bytes(), key="filename")]
    )
    (directory / "manifest.json").write_bytes(_json(base))
    channel_path = mirror / "channels/main.json"
    channel = json.loads(channel_path.read_bytes())
    channel["base"]["fixture-cpu"] = {
        "id": base["id"],
        "locations": [
            {"type": "mirror", "path": f"base/fixture-cpu/{base_id}/manifest.json"},
        ],
    }
    channel_path.write_bytes(_json(channel))
    _executable_release(mirror, base, "1.0", 41)
    port = free_port()
    arguments = (
        "setup",
        "--mirror",
        str(mirror),
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
    code, installed = _run(root, *arguments, "--yes")
    assert code == 0, installed
    try:
        assert installed["current"] == 1 and installed["status"] == "applied"
        assert installed["url"] == f"http://127.0.0.1:{port}"
        assert _run(root, "doctor")[1]["status"] == "clean"
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
        _executable_release(mirror, base, "2.0", 59)
        (root / "queue.json").write_text('{"running": ["real-job"], "queued": []}')
        before = sorted(path.name for path in (root / "journal").iterdir())
        code, busy = _run(root, "update", "--yes")
        assert code == 1 and "Finish or cancel" in busy["error"]
        assert sorted(path.name for path in (root / "journal").iterdir()) == before
        assert (root / "current").read_text() == "1\n"
        (root / "queue.json").write_text('{"running": [], "queued": []}')
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
        code, updated = _run(root, "update", "--yes")
        elapsed = time.monotonic() - started
        assert code == 0, updated
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
        _executable_release(mirror, base, "3.0", 61)
        engine = json.loads((mirror / "channels/main.json").read_bytes())["engine"]["release"][7:]
        (mirror / "engine" / engine / "fixture_engine-3.0-py3-none-any.whl").write_bytes(b"corrupt")
        code, refused = _run(root, "update", "--yes")
        assert code == 1 and refused["status"] == "error"
        assert (root / "current").read_text() == "1\n" and not (root / "generations/3").exists()
        assert _run(root, "status")[1]["engines"][0]["state"] == "ready"
        assert Journal.read(sorted((root / "journal").glob("*.json"))[-1]).stage == "rolled-back"
        print(f"fixture update including sync/health took {elapsed:.3f}s")
    finally:
        assert _run(root, "stop", "--yes")[0] == 0
