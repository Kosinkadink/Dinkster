"""Use development generations from a checkout with a torch-free supervisor bootstrap."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=ROOT.parent / f".{ROOT.name}-development")
    parser.add_argument("operation", choices=("setup", "run", "start", "stop", "status", "doctor"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    environment = dict(os.environ)
    environment["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".run/bootstrap")
    environment["UV_PYTHON_PREFERENCE"] = "only-managed"
    command = [
        "uv",
        "run",
        "--project",
        str(ROOT),
        "--locked",
        "--package",
        "dinkster-supervisor",
        "--no-dev",
        "--python",
        "3.12",
        "python",
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(args.root.expanduser().resolve()),
    ]
    if args.operation == "setup":
        variant = {"Linux": "linux-cpu", "Windows": "win-cpu", "Darwin": "mac-arm64"}[
            platform.system()
        ]
        command.extend(
            [
                "develop",
                "--source",
                str(ROOT),
                "--contributor",
                "--variant",
                variant,
            ]
        )
    else:
        command.append(args.operation)
    return subprocess.run([*command, *args.arguments], env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
