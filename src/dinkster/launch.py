"""Launch and manage the current generation without provisioning on start."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .setup import default_project


def main(argv: list[str] | None = None) -> int:
    environment = importlib.import_module("dinkster_supervisor.environment")

    arguments = list(sys.argv[1:] if argv is None else argv)
    operation = "start"
    program = "dinkster"
    browser = True
    if arguments and arguments[0] in {"serve", "stop", "update", "rollback"}:
        operation = arguments.pop(0)
        program += " " + operation
        browser = False
        if operation == "serve":
            operation = "start"
    parser = environment.argument_parser(operation)
    parser.prog = program
    parser.description = "Use the active managed generation; run dinkster setup first"
    parser.allow_abbrev = False
    parser.add_argument("--root", type=Path, default=default_project(), help="managed project root")
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
