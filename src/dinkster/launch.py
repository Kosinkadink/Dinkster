"""Launch and manage the current snapshot without provisioning on start."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .setup import default_project


def open_home(argv: list[str]) -> int:
    """Start the always-on process, open the default installation in it, and show it."""
    home = importlib.import_module("dinkster_supervisor.home")
    environment_state = importlib.import_module("dinkster_supervisor.environment_state")

    parser = argparse.ArgumentParser(
        prog="dinkster",
        description="Open Dinkster in the browser; the installation's engine starts behind it",
        allow_abbrev=False,
    )
    parser.add_argument("--root", type=Path, default=default_project(), help="installation to open")
    parser.add_argument("--port", type=int, default=home.DEFAULT_PORT, help="Dinkster's address")
    parser.add_argument(
        "--frontend-root", type=Path, help="frontend folder, when the process is not running yet"
    )
    parser.add_argument("--no-browser", dest="open_browser", action="store_false")
    args = parser.parse_args(argv)
    try:
        url = home.ensure_running(args.port, frontend_root=args.frontend_root)
    except RuntimeError as error:
        print(f"dinkster: {error}", file=sys.stderr)
        return 1
    root = args.root.expanduser().resolve()
    if environment_state.read_current(root) is not None:
        refusal = home.open_installation(url, root)
        if refusal is not None:
            print(f"dinkster: cannot open {root}: {refusal}", file=sys.stderr)
    print(url)
    if args.open_browser:
        import webbrowser

        webbrowser.open(url)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {
        "serve",
        "stop",
        "update",
        "restore-previous",
        "rollback",
        "run",
    }:
        return open_home(arguments)
    environment = importlib.import_module("dinkster_supervisor.environment")
    operation = arguments.pop(0)
    program = "dinkster " + operation
    browser = False
    if operation == "serve":
        operation = "start"
    parser = environment.argument_parser(operation)
    parser.prog = program
    parser.description = "Use the active managed snapshot; run dinkster setup first"
    parser.allow_abbrev = False
    parser.add_argument(
        "--root", type=Path, default=default_project(), help="managed environment root"
    )
    if operation == "start":
        parser.add_argument("--no-browser", dest="open_browser", action="store_false")
        parser.set_defaults(open_browser=browser)
    parser.set_defaults(operation=operation)
    args = parser.parse_args(arguments)
    args.root = args.root.expanduser().resolve()
    if operation in {"start", "stop"} and not args.plan:
        args.yes = True
    return environment.run(args)


def serve_main() -> int:
    return main(["serve", *sys.argv[1:]])
