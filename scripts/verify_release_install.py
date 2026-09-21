"""Exercise the documented registry install against a fresh local backend install."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for(
    url: str,
    process: subprocess.Popen[str],
    ready: Callable[[dict[str, object]], bool] = lambda _: True,
) -> dict[str, object]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited with {process.returncode}; inspect its log")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                value = json.load(response)
                if ready(value):
                    return value
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"service did not become ready: {url}")


def stop(process: subprocess.Popen[str]) -> None:
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        _, alive = psutil.wait_procs(children, timeout=10)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    if alive:
        raise RuntimeError("owned release verification descendants survived teardown")


def verify(root: Path, state: Path, registry_command: Path, python: Path | None = None) -> None:
    root = root.resolve()
    registry_command = registry_command.resolve()
    state.mkdir(parents=True)
    if python is None:
        python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = python.resolve()
    registry_database = "sqlite:///" + (state / "registry.db").as_posix()
    registry_objects = str(state / "registry-objects")
    packs = str(state / "packs")
    registry_url = f"http://127.0.0.1:{available_port()}"
    backend_url = f"http://127.0.0.1:{available_port()}"

    def run(module: str, *args: str, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            [str(python), "-m", module, *args],
            cwd=state,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout

    def registry_admin(*args: str) -> str:
        return subprocess.run(
            [str(registry_command), "admin", "--database-url", registry_database, *args],
            cwd=state,
            text=True,
            capture_output=True,
            check=True,
        ).stdout

    registry_admin("add-user", "local", "--operator")
    registry_admin(
        "add-publisher",
        "local",
        "--owner",
        "local",
    )
    token = registry_admin(
        "mint-token",
        "local",
        "--user",
        "local",
        "--expires",
        (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    ).strip()
    with ExitStack() as stack:

        def service(name: str, *command: str) -> subprocess.Popen[str]:
            log = stack.enter_context((state / f"{name}.log").open("w"))
            process = subprocess.Popen(
                command,
                cwd=state,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stack.callback(stop, process)
            return process

        registry_process = service(
            "registry",
            str(registry_command),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            registry_url.rsplit(":", 1)[1],
            "--database-url",
            registry_database,
            "--object-store-root",
            registry_objects,
        )
        wait_for(registry_url + "/v1/health", registry_process)
        environment = {**os.environ, "DINKSTER_REGISTRY_TOKEN": token}
        publication = run(
            "dinkster.manager",
            "--registry",
            registry_url,
            "publish",
            str(root / "templates/pack"),
            "--version",
            "0.1.0",
            env=environment,
        )
        (state / "publish.log").write_text(publication)
        candidate_id = publication.strip().rsplit(" ", 1)[-1]
        subprocess.run(
            [
                str(registry_command),
                "scanner",
                "--once",
                "--database-url",
                registry_database,
                "--object-store-root",
                registry_objects,
            ],
            cwd=state,
            text=True,
            capture_output=True,
            check=True,
        )
        request = urllib.request.Request(
            registry_url + f"/v1/reviews/{candidate_id}",
            data=json.dumps({"decision": "accept", "reason": "Reviewed bundled template"}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["status"] == "accepted"
        workspace = []
        for package in (
            "workers",
            "protocol",
            "schema",
            "values",
            "api",
            "memory",
            "assets",
            "caches",
            "inference",
            "video",
        ):
            workspace.extend(
                ["--workspace-package", str(root / "packages" / f"dinkster-{package}")]
            )
        installation = run(
            "dinkster.manager",
            "--root",
            packs,
            "--accelerator",
            "cpu",
            "--registry",
            registry_url,
            *workspace,
            "install",
            "my-pack@0.1.0",
            "--yes",
        )
        (state / "install.log").write_text(installation)
        preparation = run(
            "dinkster.manager", "--root", packs, "--accelerator", "cpu", "prepare-catalogs"
        )
        (state / "catalog-preparation.log").write_text(preparation)
        backend = service(
            "backend",
            str(python),
            "-m",
            "dinkster.serve",
            "--host",
            "127.0.0.1",
            "--port",
            backend_url.rsplit(":", 1)[1],
            "--library-root",
            str(state / "library"),
            "--install-root",
            packs,
            "--no-default-packs",
        )
        wait_for(backend_url + "/api/composition", backend)
        nodes = wait_for(
            backend_url + "/api/nodes",
            backend,
            lambda value: "my-pack.shout" in json.dumps(value.get("nodes")),
        )
        (state / "nodes.json").write_text(json.dumps(nodes, indent=2) + "\n")
        request = urllib.request.Request(
            backend_url + "/api/jobs",
            data=json.dumps(
                {
                    "clientId": "release-check",
                    "jobId": "shout",
                    "graph": {
                        "nodes": {
                            "shout": {
                                "nodeType": "my-pack.shout",
                                "inputs": {"text": "hello", "times": 2},
                            }
                        }
                    },
                    "targets": ["shout"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        job = wait_for(
            backend_url + "/api/jobs/release-check/shout",
            backend,
            lambda value: value.get("state") in ("completed", "failed", "cancelled"),
        )
        assert job["state"] == "completed", job
        value = wait_for(
            backend_url
            + "/api/values?clientId=release-check&jobId=shout&nodeId=shout&outputId=shouted",
            backend,
        )
        assert isinstance(value["descriptor"], dict)
        assert value["descriptor"]["value"] == "HELLO!HELLO!", value
        (state / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        config = str(state / "installs.toml")
        run(
            "dinkster_supervisor.install_manager",
            "--config",
            config,
            "add",
            "local",
            "--root",
            packs,
            "--port",
            backend_url.rsplit(":", 1)[1],
            "--yes",
        )
        assert "local" in run("dinkster_supervisor.install_manager", "--config", config, "list")
    print(
        "Registry publication, pack installation, backend catalog and install registration "
        f"passed: {state}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--registry-command", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.state is not None:
        verify(args.root, args.state.resolve(), args.registry_command, args.python)
    else:
        with tempfile.TemporaryDirectory(prefix="dinkster-release-") as directory:
            verify(args.root, Path(directory) / "state", args.registry_command, args.python)
