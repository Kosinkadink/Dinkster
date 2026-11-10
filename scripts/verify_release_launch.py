"""Verify that installed release wheels serve the backend and bundled frontend."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import psutil


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for(url: str, process: subprocess.Popen[str]) -> dict[str, object]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited with {process.returncode}; inspect its log")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.load(response)
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


def print_log_tails(state: Path) -> None:
    for path in sorted(state.glob("*.log")):
        print(f"--- {path.name} (last 100 lines) ---", file=sys.stderr)
        print("\n".join(path.read_text(errors="replace").splitlines()[-100:]), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--require-frontend", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dinkster-release-") as directory:
        state = Path(directory) / "state"
        try:
            python = args.python or args.root / ".venv" / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            verify_launch(state, python, require_frontend=args.require_frontend)
        except Exception:
            print_log_tails(state)
            raise
