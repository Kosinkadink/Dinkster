"""Filesystem and activation guarantees with simulated native environment creation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from dinkster_registry import InstallError

from dinkster import projects
from dinkster.cli import main
from dinkster.engine_feed import Mirror
from dinkster.engine_install import EngineInstaller, environment_python
from dinkster.installer import Installer


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def feed(marker: str = "a") -> dict[str, bytes]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("python/bin/python", "tools/uv.exe" if os.name == "nt" else "tools/uv"):
            data = b"native executable fixture " + marker.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    base = buffer.getvalue()
    wheel = b"immutable code wheel fixture"
    base_path = f"base/linux-cu128/{marker * 64}.tar.gz"
    wheel_path = f"store/{digest(wheel)}"
    manifest = json.dumps(
        {
            "format": "dinkster.engine/1",
            "commit": marker * 40,
            "cell": "linux-cu128",
            "base": {
                "id": marker * 64,
                "archive": {"path": base_path, "sha256": digest(base), "size": len(base)},
                "python": "python/bin/python",
                "packages": {"torch": "2.13.0"},
            },
            "wheels": [
                {
                    "path": wheel_path,
                    "sha256": digest(wheel),
                    "size": len(wheel),
                    "filename": "sample_runtime-1.0-py3-none-any.whl",
                    "name": "sample-runtime",
                    "version": "1.0",
                    "environments": ["control", "execution"],
                }
            ],
        }
    ).encode()
    manifest_path = f"engine/{marker * 40}/linux-cu128.json"
    channel = json.dumps(
        {
            "format": "dinkster.engine-channel/1",
            "channel": "github-live",
            "commit": marker * 40,
            "minimumLauncherVersion": "0.0.1",
            "cells": {
                "linux-cu128": {
                    "path": manifest_path,
                    "sha256": digest(manifest),
                    "size": len(manifest),
                }
            },
        }
    ).encode()
    return {
        base_path: base,
        wheel_path: wheel,
        manifest_path: manifest,
        "channels/github-live.json": channel,
    }


@pytest.fixture
def mirror():
    objects = feed()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = objects.get(self.path.lstrip("/"))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Mirror(f"http://127.0.0.1:{server.server_port}", allow_local_http=True), objects
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def native_commands(monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []

    def run(_self: EngineInstaller, command: list[str]) -> None:
        commands.append(command)
        if "venv" in command:
            python = environment_python(Path(command[-1]))
            python.parent.mkdir(parents=True)
            python.write_bytes(b"thin interpreter fixture")

    monkeypatch.setattr(EngineInstaller, "_run", run)
    return commands


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_stage_install_rollback_and_gc_preserve_data(
    tmp_path: Path, mirror, native_commands
) -> None:
    data = tmp_path / "data"
    (data / "models").mkdir(parents=True)
    (data / "models/weights.bin").write_bytes(bytes(range(251)))
    (data / "settings.json").write_bytes(b'{"theme":"dark"}')
    before = tree(data)
    client, objects = mirror
    installer = Installer(tmp_path / "install")
    engine = EngineInstaller(installer)
    first = engine.install(client, channel="github-live", cell="linux-cu128")
    first_environment = installer.environment_of(first)
    objects.update(feed("b"))
    second = engine.install(client, channel="github-live", cell="linux-cu128", activate=False)
    assert installer.current_number() == first
    engine.activate(second)
    assert installer.current_number() == second
    restored = engine.rollback()
    assert installer.environment_of(restored) == first_environment
    garbage = installer.root / "engine-objects" / ("f" * 64)
    garbage.write_bytes(b"unreferenced")
    assert engine.gc_candidates() == (garbage,)
    assert engine.gc() == (garbage,)
    assert tree(data) == before
    installs = [command for command in native_commands if "install" in command]
    assert len(installs) == 4
    assert all("--offline" in command and "--no-index" in command for command in installs)
    assert all("--require-hashes" in command for command in installs)
    assert all(
        "--system-site-packages" in command for command in native_commands if "venv" in command
    )


def test_shared_objects_survive_other_project_gc(tmp_path: Path, mirror, native_commands) -> None:
    client, objects = mirror
    store = tmp_path / "shared"
    first = EngineInstaller(Installer(tmp_path / "first", shared_store=store))
    second = EngineInstaller(Installer(tmp_path / "second", shared_store=store))
    first.install(client, channel="github-live", cell="linux-cu128")
    wheel_path = next(key for key in objects if key.startswith("store/"))
    wheel = store / "engine-objects" / wheel_path.removeprefix("store/")
    identity = (wheel.stat().st_dev, wheel.stat().st_ino)
    second.install(client, channel="github-live", cell="linux-cu128")
    assert (wheel.stat().st_dev, wheel.stat().st_ino) == identity
    other_current = (second.root / "current").read_bytes()
    objects.update(feed("b"))
    first.install(client, channel="github-live", cell="linux-cu128")
    first.gc()
    assert wheel.is_file()
    assert (second.root / "current").read_bytes() == other_current
    assert second.interpreters(1)[0].is_file()


def test_failed_provision_and_missing_rollback_leave_current(
    tmp_path: Path, mirror, native_commands, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, objects = mirror
    engine = EngineInstaller(Installer(tmp_path / "install"))
    first = engine.install(client, channel="github-live", cell="linux-cu128")
    objects.update(feed("b"))
    second = engine.install(client, channel="github-live", cell="linux-cu128")
    control, _ = engine.interpreters(first)
    control.unlink()
    with pytest.raises(InstallError, match="interpreter is missing"):
        engine.rollback()
    assert engine.installer.current_number() == second

    def fail(_self, _command) -> None:
        raise InstallError("native installation failed")

    monkeypatch.setattr(EngineInstaller, "_run", fail)
    objects.update(feed("c"))
    with pytest.raises(InstallError, match="native installation failed"):
        engine.install(client, channel="github-live", cell="linux-cu128")
    assert engine.installer.current_number() == second


def test_gc_refuses_linked_owned_namespace(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "keep").write_bytes(b"data")
    engine = EngineInstaller(Installer(tmp_path / "install"))
    (engine.root / "engine-envs").symlink_to(data, target_is_directory=True)
    with pytest.raises(InstallError, match="symlink"):
        engine.gc()
    assert tree(data) == {"keep": b"data"}


def test_project_cli_keeps_data_separate_and_uses_project_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, mirror, native_commands
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "home"))
    assert main(["project", "create", "art", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert not Path(record["dataRoot"]).exists()
    assert main(["project", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [record]
    client, _ = mirror
    engine = EngineInstaller(Installer(Path(record["root"])))
    engine.install(client, channel="github-live", cell="linux-cu128")
    command = projects.supervisor_command(
        Path(record["root"]), Path(record["dataRoot"]), port=19373
    )
    assert command[command.index("--port") + 1] == "19373"
    assert command[0] == str(engine.interpreters(1)[0])
    assert command[command.index("--execution-python") + 1] == str(engine.interpreters(1)[1])
    assert "dinkster_supervisor.supervisor" in command
    assert main(["--project", "art", "generations", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["current"] is True


def test_project_rejects_data_inside_install_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path))
    with pytest.raises(InstallError, match="separate"):
        projects.create("art", tmp_path / "projects/art/data")
    assert not (tmp_path / "projects/art").exists()
