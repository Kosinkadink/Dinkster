"""Exercise folder edits and rollback with two real captured engine sources."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def verify(state: Path, older: Path, newer: Path, *, incompatible: bool = False) -> None:
    state.mkdir()
    root = state / "install"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "DINKSTER_ACCELERATOR": "cpu",
        "DINKSTER_HOME": str(state / "account"),
        "XDG_DATA_HOME": str(state / "user-data"),
        "LOCALAPPDATA": str(state / "user-data"),
    }

    def command(*arguments: str, check: bool = True) -> dict:
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
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        operation = arguments[0]
        (state / f"{operation}-{len(list(state.glob('*.json')))}.log").write_text(
            result.stderr, encoding="utf-8"
        )
        document = json.loads(result.stdout)
        (state / f"{operation}-{len(list(state.glob('*.json')))}.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        if check and result.returncode:
            raise RuntimeError(f"{operation}: {document.get('error')}; logs in {state}")
        return document

    def request(url: str, project: str, body: dict | None = None) -> dict:
        route = (
            f"/api/mounts/projects/{project}"
            if body is not None
            else f"/api/mounts?project={project}"
        )
        with urlopen(
            Request(
                url + route,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
                method="PUT" if body is not None else "GET",
            ),
            timeout=10,
        ) as response:
            return json.load(response)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        first = command(
            "develop",
            "--source",
            str(older),
            "--variant",
            "linux-cpu",
            "--data",
            str(state / "store"),
            "--port",
            str(port),
            "--yes",
        )
        current = command("develop", "--source", str(newer), "--variant", "linux-cpu", "--yes")
        url = current["engines"][0]["url"]
        folders = request(url, "first")
        ids = [row["id"] for row in folders["mounts"]]
        assert len(ids) >= 2, folders
        expected = {}
        for project, mounts in (("first", ids), ("second", list(reversed(ids)))):
            body = {"mounts": mounts, "outputMount": folders["outputMount"]}
            request(url, project, body)
            expected[project] = body
        refusal = None
        if incompatible:
            from dinkster.project_folders import HiddenModelPreference, ProjectFolders

            command("stop", "--yes")
            preferences = root / "dinkster-library" / "project-folders.json"
            identity = "blake3:" + "a" * 64
            ProjectFolders(preferences).set_model_hidden(
                "first",
                HiddenModelPreference("retained-model", "model/checkpoint", (identity,)),
                True,
            )
            restarted = command("start", "--yes")
            url = restarted["engines"][0]["url"]
            record = root / "runtime" / str(current["current"]) / "main.json"
            owner_pid = json.loads(record.read_text())["pid"]
            saved = preferences.read_bytes()
            refusal = command("rollback", "--yes", check=False)
            assert refusal["status"] == "error", refusal
            assert "requires Dinkster 0.0.15" in refusal["error"], refusal
            assert json.loads(record.read_text())["pid"] == owner_pid
            assert preferences.read_bytes() == saved
            assert command("status")["current"] == current["current"]
            with urlopen(url + "/api/health", timeout=10) as response:
                assert json.load(response)["ok"]
            with urlopen(
                Request(
                    url + "/api/models/hidden/first",
                    data=json.dumps({"identity": identity}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="DELETE",
                ),
                timeout=10,
            ) as response:
                assert json.load(response)["hiddenModels"] == []
        rolled_back = command("rollback", "--yes")
        url = rolled_back["engines"][0]["url"]
        for project, body in expected.items():
            observed = request(url, project)
            assert [row["id"] for row in observed["mounts"]] == body["mounts"], observed
            assert observed["outputMount"] == body["outputMount"], observed
        with urlopen(url + "/api/health", timeout=10) as response:
            health = json.load(response)
        assert health["ok"], health
        print(
            json.dumps(
                {
                    "older": first["current"],
                    "newer": current["current"],
                    "restored": rolled_back["current"],
                    "projects": expected,
                    "incompatibleRollbackRefusal": refusal,
                    "health": health,
                    "state": str(state),
                },
                indent=2,
            )
        )
    finally:
        command("stop", "--yes", check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--older-source", type=Path, required=True)
    parser.add_argument("--newer-source", type=Path, required=True)
    parser.add_argument("--hidden-model-refusal", action="store_true")
    args = parser.parse_args()
    verify(
        args.state.resolve(),
        args.older_source.resolve(),
        args.newer_source.resolve(),
        incompatible=args.hidden_model_refusal,
    )
