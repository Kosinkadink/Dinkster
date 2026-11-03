"""Build tools with the approved Environments interpreter on its native host."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

from scripts.build_environment_tools import acquire
from scripts.build_release import sha256


def build(environments: Path, supervisor_wheel: Path, output: Path, uv: str = "uv") -> None:
    config = tomllib.loads((environments / "variants.toml").read_text(encoding="ascii"))
    system = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform)
    if system is None:
        raise ValueError(f"unsupported native tools platform: {sys.platform}")
    interpreter = config["interpreters"][system]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-tools-", dir=output.parent) as directory:
        work = Path(directory)
        archive = work / "interpreter.tar.gz"
        if system == "windows":
            # The producer verifies the pinned override and probes mapped shared memory.
            subprocess.run(
                [
                    uv,
                    "run",
                    "--locked",
                    "--project",
                    str(environments),
                    "--group",
                    "sqlite-build",
                    "python",
                    str(environments / "windows_sqlite.py"),
                    "acquire",
                    "--config",
                    str(environments / "variants.toml"),
                    "--output",
                    str(work / "runtime"),
                ],
                check=True,
            )
            with tarfile.open(archive, "w:gz") as target:
                target.add(work / "runtime/python", "python")
        else:
            url = (
                "https://github.com/astral-sh/python-build-standalone/releases/download/"
                f"{config['base']['release']}/{interpreter['filename']}"
            )
            acquire(url, archive, interpreter["size"], interpreter["sha256"])
            with tarfile.open(archive) as source:
                source.extractall(work / "runtime", filter="data")
        python = work / "runtime" / interpreter["executable"]
        subprocess.run(
            [
                uv,
                "run",
                "--no-project",
                "--python",
                str(python),
                "--with",
                "packaging==26.2",
                "-m",
                "scripts.build_environment_tools",
                "--interpreter",
                str(archive),
                "--interpreter-size",
                str(archive.stat().st_size),
                "--interpreter-sha256",
                sha256(archive),
                "--supervisor-wheel",
                str(supervisor_wheel),
                "--output",
                str(output),
                "--uv",
                uv,
            ],
            env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments", required=True, type=Path)
    parser.add_argument("--supervisor-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    build(
        args.environments.resolve(), args.supervisor_wheel.resolve(), args.output.resolve(), args.uv
    )
