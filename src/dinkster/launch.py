"""Launch and manage the current generation without provisioning on start."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .setup import default_project


def main(argv: list[str] | None = None) -> int:
    from dinkster_supervisor.environment import main as environment_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    operation = "start"
    browser = True
    if arguments and arguments[0] in {"serve", "stop", "update", "rollback"}:
        operation = arguments.pop(0)
        browser = False
        if operation == "serve":
            operation = "start"
    parser = argparse.ArgumentParser(
        prog="dinkster" if browser else f"dinkster {operation}",
        description="Use the active managed generation; run dinkster setup first",
        add_help=browser,
        allow_abbrev=False,
    )
    parser.add_argument("--root", type=Path, default=default_project())
    if browser:
        parser.add_argument("--no-browser", action="store_true")
    args, remaining = parser.parse_known_args(arguments)
    command = ["--root", str(args.root.expanduser().resolve()), operation, *remaining]
    if operation in {"start", "stop"} and not any(
        flag in remaining for flag in ("--plan", "--yes")
    ):
        command.append("--yes")
    if browser and not args.no_browser:
        command.append("--open-browser")
    try:
        environment_main(command)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def serve_main() -> int:
    return main(["serve", *sys.argv[1:]])
