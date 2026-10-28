"""Exercise the documented registry install against a fresh local backend install."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import psutil
from dinkster_workers.provision import workspace_packages_for


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


def verify_launch(state: Path, python: Path, *, require_frontend: bool = False) -> None:
    state.mkdir(parents=True)
    url = f"http://127.0.0.1:{available_port()}"
    frontend_args: list[str] = []
    if require_frontend:
        bundle = subprocess.check_output(
            [
                str(python.absolute()),
                "-c",
                "from dinkster.frontend import discover_frontend_bundle; "
                "bundle = discover_frontend_bundle(); "
                "assert bundle is not None, 'installed frontend bundle not found'; print(bundle)",
            ],
            cwd=state,
            text=True,
        ).strip()
        frontend_args = ["--frontend-root", bundle]
    with (state / "backend.log").open("w") as log:
        backend = subprocess.Popen(
            [
                str(python.absolute()),
                "-m",
                "dinkster.serve",
                "--host",
                "127.0.0.1",
                "--port",
                url.rsplit(":", 1)[1],
                "--library-root",
                str(state / "library"),
                "--install-root",
                str(state / "packs"),
                "--no-default-packs",
                *frontend_args,
            ],
            cwd=state,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            composition = wait_for(url + "/api/composition", backend)
            nodes = wait_for(url + "/api/nodes", backend)
            assert nodes["schemaVersion"] == 1 and nodes["nodes"] == {}, nodes
            if require_frontend:
                with urllib.request.urlopen(url + "/", timeout=5) as response:
                    assert response.status == 200, response.status
                    assert response.headers.get_content_type() == "text/html", response.headers
                    document = response.read().decode("utf-8")
                parser = HTMLParser()
                roots: list[str] = []

                def find_root(tag: str, attrs: list[tuple[str, str | None]]) -> None:
                    if tag == "div" and ("id", "root") in attrs:
                        roots.append(tag)

                parser.handle_starttag = find_root
                parser.feed(document)
                assert roots, "GET / did not contain the frontend application root"
                (state / "index.html").write_text(document, encoding="utf-8")
                print("Installed frontend GET / passed: HTTP 200, text/html, div#root")
            (state / "composition.json").write_text(json.dumps(composition, indent=2) + "\n")
            (state / "nodes.json").write_text(json.dumps(nodes, indent=2) + "\n")
        finally:
            stop(backend)
    print(f"Independent backend launch passed: {state}")


def verify(root: Path, state: Path, registry_command: Path, python: Path | None = None) -> None:
    root = root.resolve()
    registry_command = registry_command.resolve()
    state.mkdir(parents=True)
    if python is None:
        python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = python.absolute()
    registry_database = "sqlite:///" + (state / "registry.db").as_posix()
    registry_objects = str(state / "registry-objects")
    packs = str(state / "packs")

    def run(module: str, *args: str, log_name: str, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            [str(python), "-m", module, *args],
            cwd=state,
            env=env,
            text=True,
            capture_output=True,
        )
        (state / log_name).write_text(result.stdout + result.stderr)
        result.check_returncode()
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
        registry_listener = stack.enter_context(socket.socket())
        backend_listener = stack.enter_context(socket.socket())
        registry_listener.bind(("127.0.0.1", 0))
        backend_listener.bind(("127.0.0.1", 0))
        registry_url = f"http://127.0.0.1:{registry_listener.getsockname()[1]}"
        backend_url = f"http://127.0.0.1:{backend_listener.getsockname()[1]}"
        registry_environment = {
            **os.environ,
            "DINKSTER_OBJECT_STORE_ROOT": registry_objects,
            "DINKSTER_PUBLIC_BASE_URL": registry_url,
        }

        def service(
            name: str, *command: str, env: dict[str, str] | None = None
        ) -> subprocess.Popen[str]:
            log = stack.enter_context((state / f"{name}.log").open("w"))
            process = subprocess.Popen(
                command,
                cwd=state,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stack.callback(stop, process)
            return process

        registry_listener.close()
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
            env=registry_environment,
        )
        wait_for(registry_url + "/v1/health", registry_process)
        environment = {**os.environ, "DINKSTER_REGISTRY_TOKEN": token}
        publication_root = state / "template"
        shutil.copytree(root / "templates/pack", publication_root)
        with (publication_root / "dinkster-pack.toml").open("a") as manifest:
            manifest.write(
                '\n[pack.release]\nversion = "0.1.0"\ndinkster = "==0.0.1"\n'
                'license = "GPL-3.0-or-later"\n'
            )
        publication = run(
            "dinkster.manager",
            "--registry",
            registry_url,
            "publish",
            str(publication_root),
            "--version",
            "0.1.0",
            log_name="publish.log",
            env=environment,
        )
        candidate_id = publication.strip().rsplit(" ", 1)[-1]
        with (state / "scanner.log").open("w") as log:
            subprocess.run(
                [
                    str(registry_command),
                    "scanner",
                    "--once",
                    "--database-url",
                    registry_database,
                ],
                cwd=state,
                env=registry_environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        request = urllib.request.Request(
            registry_url + f"/v1/publish/{candidate_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request) as response:
            candidate = json.load(response)
        if candidate["status"] == "review":
            request = urllib.request.Request(
                registry_url + f"/v1/reviews/{candidate_id}",
                data=json.dumps(
                    {"decision": "accept", "reason": "Reviewed bundled template"}
                ).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request) as response:
                candidate = json.load(response)
        (state / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
        assert candidate["status"] == "accepted", candidate
        workspace = []
        worker_root = root / "packages/dinkster-workers"
        for package in (worker_root, *workspace_packages_for(worker_root)):
            workspace.extend(["--workspace-package", str(package)])
        run(
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
            log_name="install.log",
        )
        run(
            "dinkster.manager",
            "--root",
            packs,
            "--accelerator",
            "cpu",
            "prepare-catalogs",
            log_name="catalog-preparation.log",
        )
        backend_listener.close()
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
            log_name="registration.log",
        )
        assert "local" in run(
            "dinkster_supervisor.install_manager",
            "--config",
            config,
            "list",
            log_name="install-list.log",
        )
    print(
        "Registry publication, pack installation, backend catalog and install registration "
        f"passed: {state}"
    )


def print_log_tails(state: Path) -> None:
    for path in sorted(state.glob("*.log")):
        print(f"--- {path.name} (last 100 lines) ---", file=sys.stderr)
        print("\n".join(path.read_text(errors="replace").splitlines()[-100:]), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--registry-command", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--require-frontend", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dinkster-release-") as directory:
        state = args.state.resolve() if args.state else Path(directory) / "state"
        try:
            if args.registry_command:
                verify(args.root, state, args.registry_command, args.python)
            else:
                python = args.python or args.root / ".venv" / (
                    "Scripts/python.exe" if os.name == "nt" else "bin/python"
                )
                verify_launch(state, python, require_frontend=args.require_frontend)
        except Exception:
            print_log_tails(state)
            raise
