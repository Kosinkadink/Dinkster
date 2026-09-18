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


def verify(root: Path, state: Path) -> None:
    root = root.resolve()
    state.mkdir(parents=True)
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    registry = str(state / "registry")
    packs = str(state / "packs")
    registry_url = f"http://127.0.0.1:{available_port()}"
    backend_url = f"http://127.0.0.1:{available_port()}"

    def run(module: str, *args: str, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            [str(python), "-m", module, *args],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout

    run("dinkster.registry_service", "--data", registry, "admin", "add-user", "local", "--operator")
    run(
        "dinkster.registry_service",
        "--data",
        registry,
        "admin",
        "add-publisher",
        "local",
        "--owner",
        "local",
    )
    token = run(
        "dinkster.registry_service",
        "--data",
        registry,
        "admin",
        "mint-token",
        "local",
        "--user",
        "local",
        "--expires",
        (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    ).strip()
    with ExitStack() as stack:

        def service(name: str, module: str, *args: str) -> subprocess.Popen[str]:
            log = stack.enter_context((state / f"{name}.log").open("w"))
            process = subprocess.Popen(
                [str(python), "-m", module, *args],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stack.callback(stop, process)
            return process

        registry_process = service(
            "registry",
            "dinkster.registry_service",
            "--data",
            registry,
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            registry_url.rsplit(":", 1)[1],
            "--probe-sandbox",
            "off",
        )
        wait_for(registry_url + "/index/packs", registry_process)
        environment = {**os.environ, "DINKSTER_REGISTRY_TOKEN": token}
        publication = run(
            "dinkster.manager",
            "--registry",
            registry_url,
            "publish",
            "templates/pack",
            "--version",
            "0.1.0",
            env=environment,
        )
        (state / "publish.log").write_text(publication)
        request = urllib.request.Request(
            registry_url + "/reviews/my-pack/versions/0.1.0/resolve",
            data=json.dumps(
                {"decision": "accepted", "reason": "Reviewed bundled template"}
            ).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["state"] == "accepted"
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
            "dinkster.install_manager",
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
        assert "local" in run("dinkster.install_manager", "--config", config, "list")
    print(
        "Registry publication, pack installation, backend catalog and install registration "
        f"passed: {state}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if args.state is not None:
        verify(args.root, args.state.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="dinkster-release-") as directory:
            verify(args.root, Path(directory) / "state")
