"""Launch and manage the current snapshot without provisioning on start."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import cast

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
    frontend = parser.add_mutually_exclusive_group()
    frontend.add_argument("--frontend-root", type=Path, help="serve this frontend folder")
    frontend.add_argument(
        "--frontend-dev", metavar="URL", help="serve the frontend from this development server"
    )
    parser.add_argument(
        "--allow-frontend-dev-remote",
        action="store_true",
        help="allow a frontend development server that is not on this machine",
    )
    parser.add_argument("--no-browser", dest="open_browser", action="store_false")
    args = parser.parse_args(argv)
    try:
        url = home.ensure_running(
            args.port,
            frontend_root=args.frontend_root,
            frontend_dev=args.frontend_dev,
            allow_frontend_dev_remote=args.allow_frontend_dev_remote,
        )
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
    if operation == "stop":
        parser.description = (
            "Stop Dinkster and every engine it started; with --engine-only, stop only "
            "the installation's engine"
        )
        parser.add_argument(
            "--engine-only",
            action="store_true",
            help="stop only the --root installation's --engine and leave Dinkster running",
        )
    parser.set_defaults(operation=operation)
    args = parser.parse_args(arguments)
    args.root = args.root.expanduser().resolve()
    if operation == "stop" and not args.plan:
        home = importlib.import_module("dinkster_supervisor.home")
        if not args.engine_only:
            answer = home.stop_running()
            if answer is not None:
                return _report_shutdown(answer, as_json=args.json)
        # The process stops its own open engine, so it never reads the stop as a failure.
        elif (status := home.stop_open_engine(args.root, args.engine)) is not None:
            if "error" in status:
                print(f"dinkster: {status['message']}", file=sys.stderr)
                return 1
    if operation in {"start", "stop"} and not args.plan:
        args.yes = True
    return environment.run(args)


def _report_shutdown(answer: dict[str, object], *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(answer))
    else:
        for row in cast("list[dict[str, str]]", answer.get("stopped", [])):
            print(f"stopped engine {row['engine']} of {row['root']}")
        for row in cast("list[dict[str, str]]", answer.get("failed", [])):
            print(f"dinkster: could not stop {row['root']}: {row['message']}", file=sys.stderr)
        if "message" in answer:
            print(f"dinkster: {answer['message']}", file=sys.stderr)
        if answer.get("status") == "stopped":
            print("Dinkster stopped")
    return 0 if answer.get("status") == "stopped" else 1


def serve_main() -> int:
    return main(["serve", *sys.argv[1:]])
